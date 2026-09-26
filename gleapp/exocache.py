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
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin
from typing import Callable

from . import archive, mp4mux
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


# ---- DASH ------------------------------------------------------------------
# A DASH player fetches a stream as an initialization segment and a list of media
# segments, and ExoPlayer caches each one as its own item, keyed (DashUtil
# .resolveCacheKey) by the representation's cache key when the app sets one and
# otherwise by the segment's address resolved against the representation's first
# BaseURL. The manifest (MPD) the app played from is cached beside them under its
# own address, so it says which items make up which stream and in what order. That
# is the only thing a stream is joined from: a key that merely looks similar, such as
# a segment address differing in a signature, is never used to guess membership.

MAX_MANIFEST_BYTES = 4 * 1024 * 1024


def _L(e) -> str:
    return e.tag.rsplit("}", 1)[-1]


def _kids(e, name: str) -> list:
    return [c for c in e if _L(c) == name]


def _base_url(e, parent: str) -> str:
    b = _kids(e, "BaseURL")
    return urljoin(parent, b[0].text.strip()) if b and b[0].text and b[0].text.strip() else parent


def representations(manifest: bytes, manifest_url: str) -> list[dict]:
    """The streams a DASH manifest lists: for each, its attributes and either the
    resolved addresses of its initialization and media segments, in order
    (``init``, ``media``), or the one address the whole stream is fetched from by
    byte range (``single``), which ExoPlayer caches as one item.

    ``lang`` is the AdaptationSet's, or its ContentComponent's, which are the two
    places ``DashManifestParser.parseAdaptationSet`` reads it (androidx/media
    1.11.1, lines 473 and 505); it is reported as written, never translated.

    A representation described by a SegmentTemplate is not returned; none was found
    in the Android test images. BaseURL is resolved at every level, the way ``DashManifestParser`` does,
    with RFC 3986 resolution (``UriUtil.resolve``; ``urljoin`` here).
    """
    root = ET.fromstring(manifest)
    out = []
    b0 = _base_url(root, manifest_url)
    for period in _kids(root, "Period"):
        b1 = _base_url(period, b0)
        for aset in _kids(period, "AdaptationSet"):
            b2 = _base_url(aset, b1)
            lang = aset.get("lang") or next((cc.get("lang") for cc in _kids(aset, "ContentComponent")
                                             if cc.get("lang")), None)
            for rep in _kids(aset, "Representation"):
                b3 = _base_url(rep, b2)
                about = {"id": rep.get("id"), "lang": lang, "order": len(out)}
                for name, attr in (("bandwidth", "bandwidth"), ("mime", "mimeType"),
                                   ("codecs", "codecs"), ("width", "width"), ("height", "height")):
                    about[name] = rep.get(attr) or aset.get(attr)   # a Representation inherits
                if _kids(rep, "SegmentTemplate") or _kids(aset, "SegmentTemplate"):
                    continue
                sl = _kids(rep, "SegmentList") or _kids(aset, "SegmentList")
                inits = [x.get("sourceURL") for s in sl[:1]
                         for x in _kids(s, "Initialization") if x.get("sourceURL")]
                media = [urljoin(b3, u.get("media")) for s in sl[:1]
                         for u in _kids(s, "SegmentURL") if u.get("media")]
                if inits and media:
                    out.append({**about, "init": urljoin(b3, inits[0]), "media": media})
                else:
                    # one address fetched by byte range (SegmentBase, or ranges in a
                    # SegmentList): ExoPlayer caches that as one item under the address
                    out.append({**about, "single": b3})
    return out


def _complete(rec) -> bool:
    """A joined item with no gap and no shortfall against the length the index recorded."""
    j, length = rec["j"], rec["entry"].get("length")
    return j["gap"] is None and (length is None or j["bytes"] >= length)


def _dash_streams(joined: dict) -> tuple[list[dict], dict]:
    """For every cached manifest, the streams whose initialization segment and first
    media segment are both cached and complete, with the media segments that follow
    in the manifest's order up to the first one missing or incomplete. One stream per
    initialization segment: where several manifests list it, the longest run wins.

    Also returns, for each cached item that is a whole stream by itself (one address
    fetched by byte range), the manifest that lists it and the other cached streams
    that manifest lists, so a silent video can say where its audio is."""
    by_key = {rec["key"]: ident for ident, rec in joined.items() if rec["key"]}
    best: dict = {}
    pairs: dict = {}
    for ident, rec in joined.items():
        key = rec["key"] or ""
        if not re.match(r"https?://", key) or rec["j"]["bytes"] > MAX_MANIFEST_BYTES:
            continue
        with open(rec["tmp"], "rb") as fh:
            data = fh.read(MAX_MANIFEST_BYTES)
        if b"<MPD" not in data[:1024]:
            continue
        try:
            reps = representations(data, key)
        except ET.ParseError:
            continue
        singles = [(by_key[r["single"]], r) for r in reps
                   if "single" in r and r["single"] in by_key]
        for ident_s, r in singles:
            pairs.setdefault(ident_s, {"manifest": ident, "manifest_key": key, "rep": r,
                                       "with": [o for o, _ in singles if o != ident_s]})
        for r in reps:
            if "single" in r:
                continue
            init = by_key.get(r["init"])
            if init is None or not _complete(joined[init]):
                continue
            segs = []
            for u in r["media"]:
                s = by_key.get(u)
                if s is None or not _complete(joined[s]):
                    break
                segs.append(s)
            if segs and (init not in best or len(segs) > len(best[init]["segs"])):
                with open(joined[init]["tmp"], "rb") as fh:
                    handlers = track_handlers(fh.read(MAX_MANIFEST_BYTES))
                best[init] = {"manifest": ident, "manifest_key": key, "rep": r, "init": init,
                              "segs": segs, "listed": len(r["media"]), "handlers": handlers}
    return list(best.values()), pairs


def _boxes(b: bytes, start: int, end: int):
    """``(type, payload start, box end)`` for each ISO-BMFF box in ``b[start:end]``."""
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", b[i:i + 8])
        hdr = 8
        if size == 1 and i + 16 <= end:
            size, hdr = struct.unpack(">Q", b[i + 8:i + 16])[0], 16
        elif size == 0:
            size = end - i
        if size < hdr:
            return
        yield typ, i + hdr, min(i + size, end)
        i += size


def track_handlers(init: bytes) -> list[str]:
    """The handler type of each track an initialization segment declares
    (``moov/trak/mdia/hdlr``, ISO/IEC 14496-12): ``vide`` for video, ``soun`` for
    audio. The segment's own statement of what it carries, where a manifest's
    contentType is optional and often absent."""
    out = []
    for typ, s0, e0 in _boxes(init, 0, len(init)):
        if typ != b"moov":
            continue
        for t1, s1, e1 in _boxes(init, s0, e0):
            if t1 != b"trak":
                continue
            for t2, s2, e2 in _boxes(init, s1, e1):
                if t2 != b"mdia":
                    continue
                for t3, s3, e3 in _boxes(init, s2, e2):
                    if t3 == b"hdlr" and s3 + 12 <= e3:
                        out.append(init[s3 + 8:s3 + 12].decode("latin-1"))
    return out


MAX_MOOV_BYTES = 16 * 1024 * 1024


def file_handlers(path) -> list[str]:
    """``track_handlers`` for a whole file: finds ``moov`` by stepping over the
    top-level boxes, since a file written progressively keeps it after ``mdat``."""
    try:
        with open(path, "rb") as fh:
            size = fh.seek(0, 2)
            pos = 0
            while pos + 8 <= size:
                fh.seek(pos)
                head = fh.read(16)
                box, typ = struct.unpack(">I4s", head[:8])
                hdr = 8
                if box == 1 and len(head) == 16:
                    box, hdr = struct.unpack(">Q", head[8:16])[0], 16
                elif box == 0:
                    box = size - pos
                if box < hdr:
                    return []
                if typ == b"moov":
                    if box > MAX_MOOV_BYTES:
                        return []
                    fh.seek(pos)
                    return track_handlers(fh.read(box))
                pos += box
    except OSError:
        pass
    return []


def _is_audio(st: dict) -> bool:
    return st["handlers"] == ["soun"]


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
    return f"exoplayer_dash_{st['init']}_{rid}{'.m4a' if _is_audio(st) else '.mp4'}"


def _register_stream(case, c, joined: dict, st: dict, registered: list, tally: dict) -> None:
    """Write one stream's initialization segment and media segments, in order, into one
    file and register it. Fragmented MP4 is built to be read that way: DASH defines a
    representation as its initialization segment followed by its media segments."""
    r = st["rep"]
    kind, ext = ("other", ".m4a") if _is_audio(st) else ("video", ".mp4")
    parts = [st["init"], *st["segs"]]
    dest = _stream_path(case, c, st, ext)
    total = _write_stream(joined, st, dest)
    label = _stream_label(st)
    used = [row for ident in parts for row in joined[ident]["j"]["used"]]
    mtimes = [row["mtime"] for row in used if row["mtime"] is not None]
    other = next((o for o in registered if o is not st and o["manifest"] == st["manifest"]
                  and _is_audio(o) != _is_audio(st)), None)
    info = {
        "format": f"ExoPlayer cache ({c.version}), DASH stream",
        "dash": True,
        "app_folder": _app_folder(c.root),
        "cache_folder": c.root,
        "key": st["manifest_key"],
        "key_from": f"DASH manifest, cache item {st['manifest']}",
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
        origin=joined[st["init"]]["first"]["origin"],
        container_id=joined[st["init"]]["first"]["id"],
        cache_info=json.dumps(info),
    )
    tally["dash_streams"] += 1
    tally["added"] += 1
    return dest


def _stream_path(case, c, st: dict, ext: str) -> Path:
    h = hashlib.sha1(f"{c.source}\0{c.root}\0dash\0{st['init']}".encode(
        "utf-8", "surrogatepass")).hexdigest()
    return Path(case.root) / EXTRACT_DIR / "exoplayer" / h[:2] / f"{h}{ext}"


def _write_stream(joined: dict, st: dict, dest: Path) -> int:
    """The initialization segment and media segments, in order, into ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(dest, "wb") as out:
        for ident in [st["init"], *st["segs"]]:
            with open(joined[ident]["tmp"], "rb") as fh:
                shutil.copyfileobj(fh, out, 1 << 20)
            total += joined[ident]["j"]["bytes"]
    return total


def _track(rep: dict, label: str, segments: str | None = None) -> dict:
    """What the App cache line says about one audio track of a combined file."""
    t = {"label": label, "lang": rep.get("lang"), "bandwidth": rep.get("bandwidth")}
    if segments:
        t["segments"] = segments
    return {k: v for k, v in t.items() if v}


def _combine(case, c, video: Path, audio: list, *, seed: str, label: str, info: dict,
             first, mtime, tally: dict) -> None:
    """Put a video and its audio tracks in one file (``gleapp/mp4mux.py``) and register
    it; the first audio track is the one a player starts on and the others are
    alternatives. Nothing is written when they cannot be combined; the audit log
    counts it."""
    h = hashlib.sha1(f"{c.source}\0{c.root}\0av\0{seed}".encode(
        "utf-8", "surrogatepass")).hexdigest()
    dest = Path(case.root) / EXTRACT_DIR / "exoplayer" / h[:2] / f"{h}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        done = mp4mux.mux(video, audio, dest)
    except (mp4mux.MuxError, OSError, struct.error):
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
             "complete": 0, "dash_streams": 0, "in_dash_streams": 0, "combined": 0,
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
        streams, pairs = _dash_streams(joined)
        consumed = {i for st in streams for i in [st["init"], *st["segs"]]}
        # an audio stream is kept on the same terms as any other non-media item
        registered = [st for st in streams if include_other or not _is_audio(st)]
        tally["skipped_other"] += len(streams) - len(registered)
        files: dict = {}
        temps: list = []
        for st in streams:
            if st in registered:
                files[id(st)] = _register_stream(case, c, joined, st, registered, tally)
            else:
                tmp_audio = _stream_path(case, c, st, ".m4a.part")
                _write_stream(joined, st, tmp_audio)
                files[id(st)] = tmp_audio
                temps.append(tmp_audio)
        for st in streams:
            if _is_audio(st):
                continue
            # every cached audio stream of the same manifest, in the manifest's order:
            # a second language (or bitrate) goes in as an alternative track, not a guess
            sound = sorted((o for o in streams if o["manifest"] == st["manifest"] and _is_audio(o)),
                           key=lambda o: o["rep"].get("order", 0))
            if not sound:
                continue
            parts = [st["init"], *st["segs"]] + [i for a in sound for i in [a["init"], *a["segs"]]]
            _combine(case, c, files[id(st)], [files[id(a)] for a in sound], seed=f"dash\0{st['init']}",
                     label=f"exoplayer_av_{st['init']}.mp4",
                     info={"format": f"ExoPlayer cache ({c.version}), DASH video and audio",
                           "app_folder": _app_folder(c.root), "cache_folder": c.root,
                           "key": st["manifest_key"],
                           "key_from": f"DASH manifest, cache item {st['manifest']}",
                           "video": _stream_label(st),
                           "video_segments": f"{len(st['segs'])} of {st['listed']}",
                           "audio_tracks": [_track(a["rep"], _stream_label(a) if a in registered else
                                                   f"audio stream {a['rep'].get('id')} (not kept on its own)",
                                                   f"{len(a['segs'])} of {a['listed']}")
                                            for a in sound],
                           "last_touched_ms": max(q[1] for i in parts for q in joined[i]["pieces"]),
                           "member_first_ids": [joined[i]["first"]["id"] for i in parts]},
                     first=joined[st["init"]]["first"], mtime=None, tally=tally)
        for t in temps:
            with contextlib.suppress(OSError):
                t.unlink()
        # decide every item's kind first, so a pairing names only files the case keeps
        kept: dict = {}
        for ident, rec in joined.items():
            if ident in consumed:
                tally["in_dash_streams"] += 1
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
    wording, so it cannot drift between them. The last-touched time is the
    device's clock (``System.currentTimeMillis()`` when ExoPlayer wrote the piece
    or, where the app asks for it, read it again; ``SimpleCache.startFile`` and
    ``touchSpan``), shown in UTC.
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
                bits = [f"language {t['lang']} (as the manifest gives it)" if t.get("lang") else
                        "no language in the manifest"]
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
        parts.append(f"initialization segment and {i.get('segments_joined')} of "
                     f"{i.get('segments_listed')} listed media segments joined in the manifest's "
                     f"order, {i.get('bytes_joined', 0):,} bytes")
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
            parts.append(f"last written or read {t:%Y-%m-%d %H:%M:%S} UTC (device clock)")
    return "; ".join(p for p in parts if p)
