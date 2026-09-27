"""Rejoin media an Android app cached through ExoPlayer.

ExoPlayer, Google's media library for Android (``com.google.android.exoplayer2``, now
``androidx.media3``), caches what an app streams in a ``SimpleCache`` folder. One
video is not one file there: it is split into pieces, each named for where it
starts in the video, and the name of the video is kept in a separate index. On the
Android images GLEAPP was tested with, Instagram, Snapchat, Reddit, X, Google Photos,
Google Maps, Spotify, Pinterest and many others kept caches this way.

The layout, from androidx/media 1.11.1
(https://github.com/androidx/media/tree/8c6678b657ede1e7883fc164ef73ed483c7796c3,
``libraries/datasource/.../cache``):

* a piece is ``<id>.<position>.<timestamp>.v3.exo`` in a subfolder ``0`` to ``9`` of
  the cache folder (``SimpleCacheSpan.getCacheFile``, ``SimpleCache``
  ``SUBDIRECTORY_COUNT``): ``id`` names the cached item, ``position`` is the byte
  offset the piece starts at, ``timestamp`` is its last-touch time in milliseconds.
  Older caches wrote ``<key>.<position>.<timestamp>.v2.exo`` (the key escaped with
  ``%xx``, ``Util.escapeFileName``) or ``.v1.exo`` (not escaped) in the cache folder
  itself, so the key is in the name and no index is needed.
* the index that maps an id to its key, normally the URL it was fetched from, is
  either ``cached_content_index.exi`` in the cache folder (``CachedContentIndex``
  legacy storage; ``.exi.bak``, when present, is the valid copy, ``AtomicFile``) or
  a table ``ExoPlayerCacheIndex<uid>`` in a database, by default
  ``exoplayer_internal.db`` (``StandaloneDatabaseProvider``), whose ``<uid>`` is the
  name of the ``<uid>.uid`` file in the cache folder.
* an item's metadata can record its full length (``exo_len``, an 8-byte big-endian
  number) and the address it was redirected to (``exo_redir``, UTF-8)
  (``ContentMetadata``, ``DefaultContentMetadata``).

The ingest keeps these files, whatever their extension would say, as container rows
(``kind = 'archive'``), so they stay out of the gallery like any other container,
are hashed, and are never mistaken for a video: the first piece of an MP4 opens
with a video header and used to be registered as a truncated video.
:func:`assemble` then joins each item's pieces, in order, from the start of the
item until the first gap, writes the result under ``<case>/extracted/`` and
registers it as an ordinary file linked to its first piece by ``container_id``.
What it did is recorded on that row in ``cache_info``: the app folder it came from,
the cache key, which index named it, how many pieces were joined, how many bytes,
the length the index recorded, and whether the join is complete, partial, or
stopped at a gap. Pieces after a gap are not joined, because without the bytes
before them they cannot be placed in a playable file; they stay in the case as
container rows.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import re
import shutil
import sqlite3
import struct
import tempfile
from pathlib import Path
from typing import Callable

from . import archive
from .ingest import _kind_from_magic
from .vendor import exoprobe
from .vendor.exoprobe import (DB_NAME, INDEX_NAME, MAX_ITEM_BYTES, TABLE_PREFIX,
                              file_handlers, is_cache_name, parse_index_file,
                              read_index_tables)

EXTRACT_DIR = "extracted"


def _base(name: str) -> str:
    return exoprobe.basename(name)


# --------------------------------------------------------------------------
def _logical(row) -> str:
    """The path the row had on the device or in the source, with ``/`` separators."""
    p = row["orig_path"] or row["rel_path"] or row["path"]
    return str(p).replace("\\", "/")


_cache_root = exoprobe.cache_root
_app_folder = exoprobe.app_folder


def _kind_ext(head: bytes) -> tuple[str, str]:
    """``(kind, ext)`` for a joined item, from its first bytes: the extension from
    exoprobe's sniff, the kind from GLEAPP's own (an MPEG transport stream, which
    GLEAPP's sniff has no rule for, is video)."""
    ext = exoprobe.sniff_ext(head)
    if ext == ".ts":
        return "video", ".ts"
    return _kind_from_magic(head[:16]), ext


class _Cache:
    """The rows of one cache folder in one source."""

    def __init__(self, source: str, root: str) -> None:
        self.source = source
        self.root = root
        self.items: dict = {}        # id (v3) or key (v1/v2) -> [(position, timestamp, row)]
        self.version: str | None = None
        self.index_rows: list = []
        self.uids: list[str] = []


def _collect(rows) -> dict:
    caches: dict = {}
    dbs = []
    for row in rows:
        logical = _logical(row)
        b = _base(logical)
        if b in (DB_NAME,):
            dbs.append(row)
            continue
        if b.startswith(DB_NAME):
            continue                                   # a sidecar, read with its database
        root = _cache_root(logical)
        c = caches.setdefault((row["source"], root), _Cache(row["source"], root))
        pn = exoprobe.parse_piece_name(b)
        if pn:
            c.version = "v3" if pn.version == "v3" else (c.version or pn.version)
            c.items.setdefault(pn.item, []).append((pn.position, pn.timestamp, row))
            continue
        if b in (INDEX_NAME, INDEX_NAME + ".bak"):
            c.index_rows.append(row)
        elif exoprobe.UID_FILE.match(b):
            c.uids.append(b[:-4].lower())
    return {"caches": [c for c in caches.values() if c.items], "dbs": dbs}


def _db_tables(case, recs: dict, db_rows, sidecars: dict) -> dict[str, dict]:
    """Every index table in every index database, keyed by uid. Each database is
    copied, with its journal, into a temporary folder first: opening evidence in
    place rewrites its ``-shm`` even read-only."""
    out: dict[str, dict] = {}
    for row in db_rows:
        with tempfile.TemporaryDirectory(dir=_tmp(case)) as td:
            try:
                with archive.local_copy(case.root, recs.get(row["source"]), row) as local:
                    shutil.copyfile(local, Path(td) / DB_NAME)
                for suffix in ("-wal", "-journal"):
                    side = sidecars.get((row["source"], _logical(row) + suffix))
                    if side is not None:
                        with archive.local_copy(case.root, recs.get(side["source"]), side) as local:
                            shutil.copyfile(local, Path(td) / (DB_NAME + suffix))
                out.update(read_index_tables(Path(td) / DB_NAME))
            except (archive.ArchiveUnavailable, OSError, sqlite3.DatabaseError):
                continue
    return out


def _tmp(case) -> str:
    d = Path(case.root) / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _index_for(case, recs: dict, c: _Cache, tables: dict) -> tuple[str, dict]:
    """``(how, entries)``: which index named this cache's items, and what it said."""
    if c.version in ("v1", "v2"):
        return "file names", {}
    rows = sorted(c.index_rows, key=lambda r: not _logical(r).endswith(".bak"))
    for row in rows:                       # the .bak first: AtomicFile restores it
        try:
            with archive.local_copy(case.root, recs.get(row["source"]), row) as local:
                got = parse_index_file(Path(local).read_bytes())
        except (archive.ArchiveUnavailable, OSError):
            continue
        if got["state"] == "read":
            return _base(_logical(row)), got["entries"]
        if got["state"] == "encrypted":
            encrypted = True
            break
    else:
        encrypted = False
    for uid in c.uids:
        if uid in tables:
            return f"{DB_NAME} table {TABLE_PREFIX}{uid}", tables[uid]
    return ("index encrypted" if encrypted else "no index found"), {}


@contextlib.contextmanager
def _open_row(case, recs: dict, row):
    with archive.local_copy(case.root, recs.get(row["source"]), row) as local:
        with open(local, "rb") as fh:
            yield fh


def _join(case, recs: dict, pieces: list, dest: Path) -> dict:
    """Write ``pieces`` into ``dest`` from position 0 until the first gap
    (``exoprobe.join``), each read through a local copy of its row.

    Returns the rows joined, the bytes written, the offset of the first gap (None
    when there is none) and the pieces left out after it.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    spans = [exoprobe.Piece(pos, ts, row["size"] or 0,
                            functools.partial(_open_row, case, recs, row), row)
             for pos, ts, row in pieces]
    with open(dest, "wb") as out:
        j = exoprobe.join(spans, out, max_bytes=MAX_ITEM_BYTES)
    return {**j, "used": [p.tag for p in j["used"]]}


_state = exoprobe.item_state


# ---- DASH and HLS ----------------------------------------------------------
# Which cached items make up a DASH or HLS stream is read from the stream's cached
# manifest or playlist, by exoprobe (``plan_streams``); see its notes for how a segment's
# cache key is built.


def _complete(rec) -> bool:
    """A joined item with no gap and no shortfall against the length the index recorded."""
    return exoprobe.is_complete(rec["j"]["bytes"], rec["entry"].get("length"), rec["j"]["gap"])


def _streams(joined: dict) -> tuple[list[dict], dict]:
    """``exoprobe.plan_streams`` over the joined items of one cache."""
    return exoprobe.plan_streams({ident: {"key": rec["key"], "path": rec["tmp"], "complete": _complete(rec)}
                               for ident, rec in joined.items()})


_is_audio = exoprobe.is_audio_stream


def _parts(st: dict) -> list:
    """A stream's items in written order; an HLS transport stream has no init segment."""
    return [i for i in [st["init"], *st["segs"]] if i is not None]


def _fmt(st: dict) -> str:
    return st.get("format", "DASH")


def _listing(st: dict) -> str:
    return "DASH manifest" if _fmt(st) == "DASH" else "HLS playlist"


def _stream_ext(st: dict) -> str:
    if _is_audio(st):
        return ".m4a"
    return ".mp4" if st["init"] is not None else ".ts"


def is_audio_item(row) -> bool:
    """True for a joined item recorded as audio only. Processing re-checks an
    'other' row by its first bytes and would call an audio stream a video."""
    raw = row["cache_info"] if "cache_info" in row.keys() else None
    if not raw:
        return False
    try:
        return json.loads(raw).get("track") == "audio"
    except (TypeError, ValueError):
        return False


def _stream_label(st: dict) -> str:
    rid = re.sub(r"[^A-Za-z0-9_-]", "", str(st["rep"].get("id") or ""))[:24] or "stream"
    return f"exoplayer_{_fmt(st).lower()}_{_parts(st)[0]}_{rid}{_stream_ext(st)}"


def _register_stream(case, c, joined: dict, st: dict, registered: list, tally: dict) -> None:
    """Write one stream's initialization segment and media segments, in order, into one
    file and register it. Fragmented MP4 is built to be read that way: DASH defines a
    representation as its initialization segment followed by its media segments, and an
    HLS media playlist names its initialization segment with #EXT-X-MAP. A transport
    stream has none and its segments are joined as they are."""
    r = st["rep"]
    ext = _stream_ext(st)
    kind = "other" if _is_audio(st) else "video"
    parts = _parts(st)
    first = joined[parts[0]]["first"]
    dest = _stream_path(case, c, st, ext)
    total = _write_stream(joined, st, dest)
    label = _stream_label(st)
    used = [row for ident in parts for row in joined[ident]["j"]["used"]]
    mtimes = [row["mtime"] for row in used if row["mtime"] is not None]
    other = next((o for o in registered if o is not st and o["manifest"] == st["manifest"]
                  and _is_audio(o) != _is_audio(st)), None)
    info = {
        "format": f"ExoPlayer cache ({c.version}), {_fmt(st)} stream",
        "dash": True,
        "stream_format": _fmt(st),
        "has_init": st["init"] is not None,
        "app_folder": _app_folder(c.root),
        "cache_folder": c.root,
        "key": st["manifest_key"],
        "key_from": f"{_listing(st)}, cache item {st['manifest']}",
        "representation": {k: r[k] for k in ("id", "bandwidth", "mime", "codecs", "width", "height")
                           if r.get(k)},
        "track": "audio" if _is_audio(st) else "video",
        "cache_ids": parts,
        "segments_joined": len(st["segs"]),
        "segments_listed": st["listed"],
        "bytes_joined": total,
        "state": "complete" if len(st["segs"]) == st["listed"] else "partial",
        "last_touched_ms": max(p[1] for ident in parts for p in joined[ident]["pieces"]),
        "member_first_ids": [joined[ident]["first"]["id"] for ident in parts],
    }
    if other is not None:
        info["other_track"] = _stream_label(other)
    case.db.upsert_file(
        str(dest),
        rel_path=f"{c.root}/{label}",
        orig_path=f"{c.root}/{label}",
        orig_name=label,
        source=c.source,
        kind=kind,
        ext=ext,
        size=total,
        mtime=max(mtimes) if mtimes else None,
        ctime=None,
        atime=None,
        origin=first["origin"],
        container_id=first["id"],
        cache_info=json.dumps(info),
    )
    tally["dash_streams" if _fmt(st) == "DASH" else "hls_streams"] += 1
    tally["added"] += 1
    return dest


def _stream_path(case, c, st: dict, ext: str) -> Path:
    # a DASH stream keeps the name it had before HLS was joined, so a second pass matches it
    tag = "dash" if _fmt(st) == "DASH" else "hls"
    h = hashlib.sha1(f"{c.source}\0{c.root}\0{tag}\0{_parts(st)[0]}".encode(
        "utf-8", "surrogatepass")).hexdigest()
    return Path(case.root) / EXTRACT_DIR / "exoplayer" / h[:2] / f"{h}{ext}"


def _write_stream(joined: dict, st: dict, dest: Path) -> int:
    """The initialization segment and media segments, in order, into ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    return exoprobe.write_stream([joined[i]["tmp"] for i in _parts(st)], dest)


def _track(rep: dict, label: str, segments: str | None = None) -> dict:
    """What the App cache line says about one audio track of a combined file."""
    t = {"label": label, "lang": rep.get("lang"), "bandwidth": rep.get("bandwidth")}
    if segments:
        t["segments"] = segments
    return {k: v for k, v in t.items() if v}


def _combine(case, c, video: Path, audio: list, *, seed: str, label: str, info: dict,
             first, mtime, tally: dict) -> None:
    """Put a video and its audio tracks in one file (``exoprobe.mux``) and register
    it; the first audio track is the one a player starts on and the others are
    alternatives. Nothing is written when they cannot be combined; the audit log
    counts it."""
    h = hashlib.sha1(f"{c.source}\0{c.root}\0av\0{seed}".encode(
        "utf-8", "surrogatepass")).hexdigest()
    dest = Path(case.root) / EXTRACT_DIR / "exoplayer" / h[:2] / f"{h}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        done = exoprobe.mux(video, audio, dest)
    except (exoprobe.MuxError, OSError, struct.error):
        tally["not_combined"] += 1
        with contextlib.suppress(OSError):
            dest.unlink()
        return
    info = {**info, "combined": True, "layout": done["layout"], "bytes_joined": done["bytes"]}
    case.db.upsert_file(
        str(dest), rel_path=f"{c.root}/{label}", orig_path=f"{c.root}/{label}",
        orig_name=label, source=c.source, kind="video", ext=".mp4", size=done["bytes"],
        mtime=mtime, ctime=None, atime=None, origin=first["origin"],
        container_id=first["id"], cache_info=json.dumps(info))
    tally["combined"] += 1
    tally["added"] += 1


def assemble(case, *, progress: Callable[[int], None] | None = None,
             include_other: bool = False, force: bool = False) -> int:
    """Rejoin every ExoPlayer cache item in the case. Returns the rows added.

    An item whose first piece already has a rejoined file is skipped unless
    ``force``. An item with no piece at position 0 has nothing that can open, so
    nothing is written for it. A joined item that is neither an image nor a video
    (an audio track, a manifest, a playlist) is registered only with
    ``include_other``, the way a nested archive's other members are.
    """
    rows = [r for r in case.db.iter_files("kind = 'archive'", ()) if is_cache_name(_logical(r))]
    if not rows:
        return 0
    recs = archive.source_records(case)
    sidecars = {(r["source"], _logical(r)): r for r in rows
                if _base(_logical(r)).startswith(DB_NAME + "-")}
    got = _collect(rows)
    tables = _db_tables(case, recs, got["dbs"], sidecars)
    have_children = {r["container_id"] for r in case.db.iter_files("container_id IS NOT NULL", ())}
    tally = {"added": 0, "items": 0, "no_start": 0, "skipped_other": 0, "failed": 0,
             "complete": 0, "dash_streams": 0, "hls_streams": 0, "in_streams": 0, "combined": 0,
             "not_combined": 0, "audio_left_out": 0}
    # the first pieces an earlier pass already put inside a DASH stream's file
    in_streams: set[int] = set()
    for r in case.db.iter_files("cache_info IS NOT NULL", ()):
        with contextlib.suppress(TypeError, ValueError):
            in_streams.update(json.loads(r["cache_info"]).get("member_first_ids") or ())
    for c in got["caches"]:
        how, entries = _index_for(case, recs, c, tables)
        joined: dict = {}
        for ident, pieces in c.items.items():
            tally["items"] += 1
            starts = [p for p in pieces if p[0] == 0]
            if not starts:
                tally["no_start"] += 1
                continue
            first = sorted(starts, key=lambda p: -p[1])[0][2]
            if (first["id"] in have_children or first["id"] in in_streams) and not force:
                continue
            entry = entries.get(ident, {}) if isinstance(ident, int) else {}
            key = entry.get("key") if isinstance(ident, int) else ident
            h = hashlib.sha1(f"{c.source}\0{c.root}\0{ident}".encode(
                "utf-8", "surrogatepass")).hexdigest()
            tmp = Path(case.root) / EXTRACT_DIR / "exoplayer" / h[:2] / f"{h}.part"
            try:
                j = _join(case, recs, pieces, tmp)
            except (archive.ArchiveUnavailable, OSError):
                tally["failed"] += 1
                with contextlib.suppress(OSError):
                    tmp.unlink()
                continue
            joined[ident] = {"tmp": tmp, "j": j, "entry": entry, "key": key,
                             "first": first, "pieces": pieces, "h": h}
        streams, pairs = _streams(joined)
        consumed = {i for st in streams for i in _parts(st)}
        # an audio stream is kept on the same terms as any other non-media item
        registered = [st for st in streams if include_other or not _is_audio(st)]
        tally["skipped_other"] += len(streams) - len(registered)
        files: dict = {}
        temps: list = []
        for st in streams:
            if st in registered:
                files[id(st)] = _register_stream(case, c, joined, st, registered, tally)
            else:
                tmp_audio = _stream_path(case, c, st, _stream_ext(st) + ".part")
                _write_stream(joined, st, tmp_audio)
                files[id(st)] = tmp_audio
                temps.append(tmp_audio)
        for st in streams:
            # an HLS video that is not video alone (a transport stream) carries its own sound
            if _is_audio(st) or (_fmt(st) == "HLS" and st["handlers"] != ["vide"]):
                continue
            # every cached audio stream of the same manifest, in the manifest's order (for
            # HLS, those of the AUDIO group the master playlist gives the video): a second
            # language (or bitrate) goes in as an alternative track, not a guess
            sound = sorted((o for o in streams if o["manifest"] == st["manifest"] and _is_audio(o)
                            and (_fmt(st) == "DASH" or not st["rep"].get("audio_group")
                                 or o["rep"].get("group") == st["rep"]["audio_group"])),
                           key=lambda o: o["rep"].get("order", 0))
            if not sound:
                continue
            parts = _parts(st) + [i for a in sound for i in _parts(a)]
            _combine(case, c, files[id(st)], [files[id(a)] for a in sound],
                     seed=f"{'dash' if _fmt(st) == 'DASH' else 'hls'}\0{_parts(st)[0]}",
                     label=f"exoplayer_av_{_parts(st)[0]}.mp4",
                     info={"format": f"ExoPlayer cache ({c.version}), {_fmt(st)} video and audio",
                           "app_folder": _app_folder(c.root), "cache_folder": c.root,
                           "key": st["manifest_key"],
                           "key_from": f"{_listing(st)}, cache item {st['manifest']}",
                           "video": _stream_label(st),
                           "video_segments": f"{len(st['segs'])} of {st['listed']}",
                           "audio_tracks": [_track(a["rep"], _stream_label(a) if a in registered else
                                                   f"audio stream {a['rep'].get('id')} (not kept on its own)",
                                                   f"{len(a['segs'])} of {a['listed']}")
                                            for a in sound],
                           "last_touched_ms": max(q[1] for i in parts for q in joined[i]["pieces"]),
                           "member_first_ids": [joined[i]["first"]["id"] for i in parts]},
                     first=joined[_parts(st)[0]]["first"], mtime=None, tally=tally)
        for t in temps:
            with contextlib.suppress(OSError):
                t.unlink()
        # decide every item's kind first, so a pairing names only files the case keeps
        kept: dict = {}
        for ident, rec in joined.items():
            if ident in consumed:
                tally["in_streams"] += 1
                with contextlib.suppress(OSError):
                    rec["tmp"].unlink()
                continue
            with open(rec["tmp"], "rb") as fh:
                kind, ext = _kind_ext(fh.read(3 * 188))
            # an MP4 whose only track is sound is audio, whatever its brand says
            rec["audio"] = ext == ".mp4" and file_handlers(rec["tmp"]) == ["soun"]
            if rec["audio"]:
                kind, ext = "other", ".m4a"
            if kind not in ("image", "video") and not include_other:
                tally["skipped_other"] += 1
                if not rec["audio"]:
                    rec["tmp"].unlink()
                continue                   # an audio file waits: a video may be combined with it
            label = f"exoplayer_{ident if isinstance(ident, int) else rec['h'][:12]}{ext}"
            kept[ident] = (kind, ext, label)
        for ident, (kind, ext, label) in kept.items():
            rec = joined[ident]
            tmp, j, entry, key, first, pieces, h = (
                rec["tmp"], rec["j"], rec["entry"], rec["key"], rec["first"], rec["pieces"], rec["h"])
            dest = tmp.with_name(h + ext)
            tmp.replace(dest)
            length = entry.get("length")
            state = _state(j["bytes"], length, j["gap"])
            tally["complete"] += state == "complete"
            info = {
                "format": f"ExoPlayer cache ({c.version})",
                "app_folder": _app_folder(c.root),
                "cache_folder": c.root,
                "cache_id": ident if isinstance(ident, int) else None,
                "key": key,
                "key_from": how,
                "pieces_joined": len(j["used"]),
                "pieces_total": len(pieces),
                "pieces_left_out": j["left_out"],
                "bytes_joined": j["bytes"],
                "length_recorded": length,
                "gap_at": j["gap"],
                "state": state,
                "last_touched_ms": max(p[1] for p in pieces
                                       if p[2]["id"] in {r["id"] for r in j["used"]}),
            }
            if rec.get("audio"):
                info["track"] = "audio"
            if entry.get("redirected_to"):
                info["redirected_to"] = entry["redirected_to"]
            if ident in pairs:
                pr = pairs[ident]
                info["manifest_key"] = pr["manifest_key"]
                info["representation"] = {k: v for k, v in pr["rep"].items()
                                          if k in ("id", "bandwidth", "mime", "codecs", "width", "height") and v}
                info["listed_with"] = [kept[o][2] for o in pr["with"] if o in kept]
            used = j["used"]
            mtimes = [r["mtime"] for r in used if r["mtime"] is not None]
            case.db.upsert_file(
                str(dest),
                rel_path=f'{c.root}/{label}',
                orig_path=f'{c.root}/{label}',
                orig_name=label,
                source=c.source,
                kind=kind,
                ext=ext,
                size=j["bytes"],
                mtime=max(mtimes) if mtimes else None,
                ctime=None,
                atime=None,
                origin=first["origin"],
                container_id=first["id"],
                cache_info=json.dumps(info),
            )
            rec["file"] = dest
            tally["added"] += 1
            if progress and tally["added"] % 50 == 0:
                progress(tally["added"])
        # a whole-file video with the cached whole-file audio its manifest lists
        for ident, (kind, _ext, label) in kept.items():
            if kind != "video" or ident not in pairs:
                continue
            sound = [o for o in pairs[ident]["with"] if joined[o].get("audio")]
            if not sound:
                continue
            # a file cut short at a gap carries no playable end; a combined copy of it
            # would only be a second broken file, so an incomplete audio is left out
            # and an incomplete video is not combined at all
            whole = [o for o in sound if _complete(joined[o])]
            if not _complete(joined[ident]) or not whole:
                tally["not_combined"] += 1
                continue
            tally["audio_left_out"] += len(sound) - len(whole)
            _combine(case, c, joined[ident]["file"],
                     [joined[o].get("file") or joined[o]["tmp"] for o in whole], seed=f"single\0{ident}",
                     label=f"exoplayer_av_{ident if isinstance(ident, int) else joined[ident]['h'][:12]}.mp4",
                     info={"format": f"ExoPlayer cache ({c.version}), DASH video and audio",
                           "app_folder": _app_folder(c.root), "cache_folder": c.root,
                           "key": pairs[ident]["manifest_key"],
                           "key_from": f"DASH manifest, cache item {pairs[ident]['manifest']}",
                           "video": label,
                           "audio_tracks": [_track(pairs[o]["rep"], kept[o][2] if o in kept else
                                                   f"cache item {o} (not kept on its own)")
                                            for o in whole],
                           "last_touched_ms": max(q[1] for i in (ident, *whole) for q in joined[i]["pieces"])},
                     first=joined[ident]["first"], mtime=None, tally=tally)
        for ident, rec in joined.items():
            if ident not in kept and ident not in consumed:
                with contextlib.suppress(OSError):
                    rec["tmp"].unlink()
        case.db.commit()
    case.db.commit()
    if tally["items"]:
        case.db.audit_log(case.examiner, "rejoin-exoplayer-cache",
                          ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in tally.items() if v))
    return tally["added"]


def describe(d) -> str:
    """``cache_info`` as one line a person reads. Empty for any other file.

    The HTML report, the CSV and JSON exports and the LAVA export all take this
    wording, so it cannot drift between them. The time is the latest one in the
    names of the pieces used, the device's clock shown in UTC
    (``System.currentTimeMillis()``): when ExoPlayer wrote the piece, or read it
    again where the app's evictor asks for touches and the cache keeps no file
    index. A cache with an ``ExoPlayerCacheFileMetadata`` table records later reads
    there and leaves the name alone (``SimpleCache.startFile`` and ``touchSpan``),
    so the name's time is not a statement about the last read.
    """
    raw = d.get("cache_info") if hasattr(d, "get") else None
    if not raw:
        return ""
    try:
        i = json.loads(raw)
    except (TypeError, ValueError):
        return ""
    parts = [i.get("format") or "ExoPlayer cache"]
    if i.get("app_folder"):
        parts.append(f"in the folder of {i['app_folder']}")
    if i.get("key"):
        parts.append(f"key {i['key']} (from {i.get('key_from')})")
    else:
        parts.append(f"no key ({i.get('key_from')})")
    if i.get("redirected_to"):
        parts.append(f"redirected to {i['redirected_to']}")
    if i.get("combined"):
        tracks = i.get("audio_tracks")
        if tracks is None:                 # written before audio tracks were listed
            parts.append(f"video {i.get('video')} and audio {i.get('audio')} put in one file, "
                         "no sample re-encoded")
            if i.get("video_segments"):
                parts.append(f"video segments {i['video_segments']}, audio segments {i['audio_segments']}")
        else:
            parts.append(f"video {i.get('video')} and {len(tracks)} audio track(s) put in one file, "
                         "no sample re-encoded")
            if i.get("video_segments"):
                parts.append(f"video segments {i['video_segments']}")
            for n, t in enumerate(tracks, 1):
                where = "playlist" if "HLS" in (i.get("format") or "") else "manifest"
                bits = [f"language {t['lang']} (as the {where} gives it)" if t.get("lang") else
                        f"no language in the {where}"]
                if str(t.get("bandwidth") or "").isdigit():
                    bits.append(f"{int(t['bandwidth']):,} bit/s")
                if t.get("segments"):
                    bits.append(f"segments {t['segments']}")
                role = "the track a player starts on" if n == 1 and len(tracks) > 1 else \
                    "an alternative" if n > 1 else ""
                parts.append(f"audio track {n}: {t.get('label')}, " + ", ".join(bits)
                             + (f", {role}" if role else ""))
        parts.append(f"{i.get('bytes_joined', 0):,} bytes")
        return _with_touch(parts, i)
    if i.get("dash"):
        r = i.get("representation") or {}
        what = ", ".join(str(v) for v in (
            r.get("mime"), r.get("codecs"),
            f"{r['width']}x{r['height']}" if r.get("width") and r.get("height") else None,
            f"{int(r['bandwidth']):,} bit/s" if str(r.get("bandwidth") or "").isdigit() else None)
            if v)
        parts.append(f"stream {r.get('id')}" + (f" ({what})" if what else ""))
        listing = "playlist" if i.get("stream_format") == "HLS" else "manifest"
        lead = "initialization segment and " if i.get("has_init", True) else ""
        parts.append(f"{lead}{i.get('segments_joined')} of {i.get('segments_listed')} listed media "
                     f"segments joined in the {listing}'s order, {i.get('bytes_joined', 0):,} bytes")
        parts.append(i.get("state") or "")
        if i.get("other_track"):
            parts.append(f"the other track of this stream is in {i['other_track']}")
        return _with_touch(parts, i)
    joined = f"{i.get('pieces_joined')} of {i.get('pieces_total')} pieces joined, " \
             f"{i.get('bytes_joined', 0):,} bytes"
    if i.get("length_recorded") is not None:
        joined += f" of the {i['length_recorded']:,} the index recorded"
    parts.append(joined)
    state = i.get("state") or ""
    if state == "stops at a gap" and i.get("gap_at") is not None:
        state += f" at byte {i['gap_at']:,}"
    if i.get("pieces_left_out"):
        state += f", {i['pieces_left_out']} later piece(s) not joined"
    parts.append(state)
    if i.get("manifest_key"):
        r = i.get("representation") or {}
        parts.append(f"listed as stream {r.get('id')} in the DASH manifest {i['manifest_key']}")
        if i.get("listed_with"):
            parts.append("the same manifest's other cached stream(s): " + ", ".join(i["listed_with"]))
    return _with_touch(parts, i)


def _with_touch(parts: list, i: dict) -> str:
    """Close a description with the last-touched time from the piece names."""
    import datetime

    ms = i.get("last_touched_ms")
    if isinstance(ms, int) and ms > 0:
        with contextlib.suppress(OverflowError, OSError, ValueError):
            t = datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc)
            parts.append(f"time in the piece names {t:%Y-%m-%d %H:%M:%S} UTC (device clock)")
    return "; ".join(p for p in parts if p)
