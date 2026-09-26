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
import hashlib
import io
import json
import re
import shutil
import sqlite3
import struct
import tempfile
from pathlib import Path, PurePosixPath
from typing import Callable

from . import archive
from .ingest import EXO_DB_NAME as DB_NAME
from .ingest import EXO_INDEX_NAME as INDEX_NAME
from .ingest import EXO_PIECE_V3 as PIECE_V3
from .ingest import EXO_PIECE_V12 as PIECE_V12
from .ingest import EXO_UID as _UID
from .ingest import _kind_from_magic
from .ingest import is_exoplayer_cache_name as is_cache_name

EXTRACT_DIR = "extracted"
TABLE_PREFIX = "ExoPlayerCacheIndex"

_ESCAPED = re.compile(r"%([A-Fa-f0-9]{2})")

# The same cap nested.py puts on one extracted member.
MAX_ITEM_BYTES = 2 * 1024 ** 3

_EXT_BY_MAGIC = (
    (lambda h: h[4:8] == b"ftyp", ".mp4"),
    (lambda h: h[:4] == b"\x1aE\xdf\xa3", ".webm"),
    (lambda h: h[:3] == b"\xff\xd8\xff", ".jpg"),
    (lambda h: h[:8] == b"\x89PNG\r\n\x1a\n", ".png"),
    (lambda h: h[:6] in (b"GIF87a", b"GIF89a"), ".gif"),
    (lambda h: h[:3] == b"FLV", ".flv"),
)


def _base(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def unescape_key(name: str) -> str | None:
    """``Util.unescapeFileName``: ``%xx`` back to its character, None if malformed."""
    want = len(name) - 2 * name.count("%")
    out = _ESCAPED.sub(lambda m: chr(int(m.group(1), 16)), name)
    return out if len(out) == want else None


def _java_utf(b: bytes) -> str:
    """A string ``DataOutputStream.writeUTF`` wrote (modified UTF-8)."""
    b = b.replace(b"\xc0\x80", b"\x00")
    try:
        return b.decode("utf-8", "surrogatepass").encode(
            "utf-16", "surrogatepass").decode("utf-16")
    except UnicodeError:
        return b.decode("utf-8", "replace")


def _metadata(data: io.BytesIO) -> dict:
    """``CachedContentIndex.readContentMetadata``: count, then name/length/value."""
    out = {}
    (n,) = struct.unpack(">i", data.read(4))
    for _ in range(n):
        (ln,) = struct.unpack(">H", data.read(2))
        name = _java_utf(data.read(ln))
        (size,) = struct.unpack(">i", data.read(4))
        if size < 0:
            raise ValueError("negative metadata size")
        value = data.read(size)
        if len(value) != size:
            raise ValueError("metadata runs past the end")
        out[name] = value
    return out


def _summary(meta: dict) -> dict:
    """The two keys ExoPlayer itself defines: the item's full length and the
    address it was redirected to."""
    out = {}
    v = meta.get("exo_len")
    if v is not None and len(v) == 8:
        out["length"] = struct.unpack(">q", v)[0]
    v = meta.get("exo_redir")
    if v is not None:
        out["redirected_to"] = v.decode("utf-8", "replace")
    return out


def parse_index_file(data: bytes) -> dict:
    """Read a ``cached_content_index.exi``.

    Returns ``{"state": ..., "entries": {id: {"key", "length"?, "redirected_to"?}}}``.
    ``state`` is ``read`` when every entry parsed and the file ended exactly after
    the trailing hash, ``encrypted`` when the file's flag says the entries are
    AES-encrypted (the key is not in the file), and ``unreadable`` otherwise.
    """
    if len(data) < 8:
        return {"state": "unreadable", "entries": {}}
    version, flags = struct.unpack(">ii", data[:8])
    if version < 0 or version > 2:
        return {"state": "unreadable", "entries": {}}
    if flags & 1:
        return {"state": "encrypted", "entries": {}}
    buf = io.BytesIO(data[8:])
    entries = {}
    try:
        (count,) = struct.unpack(">i", buf.read(4))
        for _ in range(count):
            (cid,) = struct.unpack(">i", buf.read(4))
            (ln,) = struct.unpack(">H", buf.read(2))
            key = _java_utf(buf.read(ln))
            if version < 2:
                (length,) = struct.unpack(">q", buf.read(8))
                info = {"length": length} if length >= 0 else {}
            else:
                info = _summary(_metadata(buf))
            entries[cid] = {"key": key, **info}
        buf.read(4)                                   # the trailing hash
        state = "read" if buf.read(1) == b"" else "unreadable"
    except (struct.error, ValueError):
        state = "unreadable"
    return {"state": state, "entries": entries}


def read_index_tables(db_path: Path) -> dict[str, dict]:
    """``{uid: {id: {"key", ...}}}`` for every ``ExoPlayerCacheIndex<uid>`` table in a
    copy of an index database. The caller owns ``db_path`` (never the evidence)."""
    out: dict[str, dict] = {}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE ?",
            (TABLE_PREFIX + "%",))]
        for t in names:
            rows = {}
            for cid, key, blob in con.execute(f'SELECT id, key, metadata FROM "{t}"'):
                info = {}
                if blob:
                    with contextlib.suppress(struct.error, ValueError):
                        info = _summary(_metadata(io.BytesIO(bytes(blob))))
                rows[int(cid)] = {"key": key, **info}
            out[t[len(TABLE_PREFIX):].lower()] = rows
    finally:
        con.close()
    return out


# --------------------------------------------------------------------------
def _logical(row) -> str:
    """The path the row had on the device or in the source, with ``/`` separators."""
    p = row["orig_path"] or row["rel_path"] or row["path"]
    return str(p).replace("\\", "/")


def _cache_root(logical: str) -> str:
    """A v3 piece sits in ``<cache>/<0-9>/``; the index and uid file sit in ``<cache>``."""
    parent = PurePosixPath(logical).parent
    if PIECE_V3.match(_base(logical)) and re.fullmatch(r"\d", parent.name):
        parent = parent.parent
    return str(parent)


def _app_folder(root: str) -> str | None:
    """The Android package whose folder holds the cache, read from the path."""
    m = re.search(r"(?:^|/)(?:data/data|data/user(?:_de)?/\d+|Android/data)/([^/]+)/", root + "/")
    return m.group(1) if m else None


def _is_mpeg_ts(head: bytes) -> bool:
    """An MPEG transport stream: 188-byte packets, each opening with 0x47. HLS
    serves video in these segments, and GLEAPP's shared sniff has no rule for them
    because a single 0x47 byte says nothing; three packets in a row do."""
    return len(head) >= 3 * 188 and all(head[i * 188] == 0x47 for i in range(3))


def _kind_ext(head: bytes) -> tuple[str, str]:
    """``(kind, ext)`` for a joined item, from its first bytes."""
    if _is_mpeg_ts(head):
        return "video", ".ts"
    for test, ext in _EXT_BY_MAGIC:
        if test(head):
            return _kind_from_magic(head[:16]), ext
    return _kind_from_magic(head[:16]), ""


def _copy_range(src: Path, out, skip: int, want: int | None) -> int:
    """Copy ``src`` from ``skip`` onward (``want`` bytes, or to the end) into ``out``."""
    n = 0
    with open(src, "rb") as fh:
        fh.seek(skip)
        while want is None or n < want:
            chunk = fh.read(1 << 20 if want is None else min(1 << 20, want - n))
            if not chunk:
                break
            out.write(chunk)
            n += len(chunk)
    return n


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
        m3 = PIECE_V3.match(b)
        if m3:
            c.version = "v3"
            c.items.setdefault(int(m3.group(1)), []).append(
                (int(m3.group(2)), int(m3.group(3)), row))
            continue
        m12 = PIECE_V12.match(b)
        if m12:
            c.version = c.version or f"v{m12.group(4)}"
            key = m12.group(1) if m12.group(4) == "1" else unescape_key(m12.group(1))
            c.items.setdefault(key if key is not None else m12.group(1), []).append(
                (int(m12.group(2)), int(m12.group(3)), row))
            continue
        if b in (INDEX_NAME, INDEX_NAME + ".bak"):
            c.index_rows.append(row)
        elif _UID.match(b):
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


def _join(case, recs: dict, pieces: list, dest: Path) -> dict:
    """Write ``pieces`` into ``dest`` from position 0 until the first gap.

    Returns the pieces joined, the bytes written, the offset of the first gap (None
    when there is none) and the pieces left out after it.
    """
    pieces = sorted(pieces, key=lambda p: (p[0], -p[1]))
    end, used, gap = 0, [], None
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as out:
        for pos, _ts, row in pieces:
            size = row["size"] or 0
            if pos + size <= end:
                continue                               # nothing this piece adds
            if pos > end:
                gap = end
                break
            if pos + size > MAX_ITEM_BYTES:
                gap = end
                break
            with archive.local_copy(case.root, recs.get(row["source"]), row) as local:
                end += _copy_range(Path(local), out, end - pos, None)
            used.append(row)
    ids = {r["id"] for r in used}
    left = [p for p in pieces if p[2]["id"] not in ids and p[0] + (p[2]["size"] or 0) > end]
    return {"used": used, "bytes": end, "gap": gap, "left_out": len(left)}


def _state(joined: int, length, gap) -> str:
    if length is not None and length >= 0:
        if joined >= length:
            return "complete"
        return "stops at a gap" if gap is not None else "partial"
    return "stops at a gap" if gap is not None else "length not recorded"


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
             "complete": 0}
    for c in got["caches"]:
        how, entries = _index_for(case, recs, c, tables)
        for ident, pieces in c.items.items():
            tally["items"] += 1
            starts = [p for p in pieces if p[0] == 0]
            if not starts:
                tally["no_start"] += 1
                continue
            first = sorted(starts, key=lambda p: -p[1])[0][2]
            if first["id"] in have_children and not force:
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
            with open(tmp, "rb") as fh:
                head = fh.read(3 * 188)
            kind, ext = _kind_ext(head)
            if kind not in ("image", "video") and not include_other:
                tally["skipped_other"] += 1
                tmp.unlink()
                continue
            dest = tmp.with_name(h + ext)
            tmp.replace(dest)
            length = entry.get("length")
            state = _state(j["bytes"], length, j["gap"])
            tally["complete"] += state == "complete"
            label = f"exoplayer_{ident if isinstance(ident, int) else h[:12]}{ext}"
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
            if entry.get("redirected_to"):
                info["redirected_to"] = entry["redirected_to"]
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
            tally["added"] += 1
            if progress and tally["added"] % 50 == 0:
                progress(tally["added"])
        case.db.commit()
    case.db.commit()
    if tally["items"]:
        case.db.audit_log(case.examiner, "rejoin-exoplayer-cache",
                          ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in tally.items() if v))
    return tally["added"]


def describe(d) -> str:
    """``cache_info`` as one line a person reads. Empty for any other file.

    The HTML report, the CSV and JSON exports and the LAVA export all take this
    wording, so it cannot drift between them. The last-touched time is the
    device's clock (``System.currentTimeMillis()`` when ExoPlayer wrote the piece
    or, where the app asks for it, read it again; ``SimpleCache.startFile`` and
    ``touchSpan``), shown in UTC.
    """
    import datetime

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
    ms = i.get("last_touched_ms")
    if isinstance(ms, int) and ms > 0:
        with contextlib.suppress(OverflowError, OSError, ValueError):
            t = datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc)
            parts.append(f"last written or read {t:%Y-%m-%d %H:%M:%S} UTC (device clock)")
    return "; ".join(p for p in parts if p)
