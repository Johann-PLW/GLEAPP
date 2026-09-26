"""Joining the pieces of an app's ExoPlayer media cache (gleapp/exocache.py).

The fixture is written by ``_write_cache`` below, following the layout sourced in
gleapp/exocache.py's docstring; the reader was also run against the ExoPlayer caches
on the registered Android corpora, where every index it could read named every piece.
"""

import json
import sqlite3
import struct
import zipfile

import pytest

from gleapp import exocache, nested
from gleapp.vendor import exoprobe
from gleapp.case import Source, open_case
from gleapp.ingest import is_exoplayer_cache_name
from gleapp.pipeline import ingest_sources


@pytest.fixture(autouse=True)
def _isolate_appconfig(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("GLEAPP_CONFIG_DIR", str(tmp_path_factory.mktemp("gleapp-cfg")))


APP = "data/data/com.example.player/cache/exo"
MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + bytes(range(256)) * 40
TS = b"".join(b"\x47" + bytes([i % 256]) * 187 for i in range(12))
GAPPED = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x5a" * 3000
TS_MS = 1_700_000_000_000


def _utf(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack(">H", len(b)) + b


def _meta(length: int | None) -> bytes:
    if length is None:
        return struct.pack(">i", 0)
    return struct.pack(">i", 1) + _utf("exo_len") + struct.pack(">i", 8) + struct.pack(">q", length)


def _exi(entries: dict, *, encrypted: bool = False) -> bytes:
    """``CachedContentIndex`` legacy storage, version 2."""
    if encrypted:
        return struct.pack(">ii", 2, 1) + b"\x11" * 16 + b"\x99" * 48
    body = struct.pack(">i", len(entries))
    for cid, (key, length) in entries.items():
        body += struct.pack(">i", cid) + _utf(key) + _meta(length)
    return struct.pack(">ii", 2, 0) + body + struct.pack(">i", 0)


def _pieces(cid: int, data: bytes, cuts: list[int], sub: str = "3") -> dict:
    """``<cache>/<sub>/<id>.<position>.<timestamp>.v3.exo`` for each cut."""
    out = {}
    bounds = cuts + [len(data)]
    for i, start in enumerate(cuts):
        out[f"{sub}/{cid}.{start}.{TS_MS + i}.v3.exo"] = data[start:bounds[i + 1]]
    return out


def _db(uid: str, entries: dict) -> bytes:
    """An ``exoplayer_internal.db`` holding one ``ExoPlayerCacheIndex<uid>`` table."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.db"
        con = sqlite3.connect(p)
        con.execute("PRAGMA journal_mode=WAL")          # as an app's database usually is
        con.execute(f'CREATE TABLE "ExoPlayerCacheIndex{uid}" '
                    "(id INTEGER PRIMARY KEY NOT NULL, key TEXT NOT NULL, metadata BLOB NOT NULL)")
        for cid, (key, length) in entries.items():
            con.execute(f'INSERT INTO "ExoPlayerCacheIndex{uid}" VALUES (?,?,?)',
                        (cid, key, _meta(length)))
        con.commit()
        con.close()
        return p.read_bytes()


def _tree() -> dict[str, bytes]:
    """Every file of the synthetic extraction, by path."""
    files: dict[str, bytes] = {}
    # a cache indexed by cached_content_index.exi
    idx = {1: ("https://cdn.example.net/v/1.mp4", len(MP4)),
           2: ("https://cdn.example.net/v/2.mp4", len(GAPPED)),
           3: ("https://cdn.example.net/v/3.mp4", 5000),
           4: ("https://cdn.example.net/seg/4.ts", len(TS))}
    for rel, b in _pieces(1, MP4[:9000], [0, 4000, 8000], "3").items():
        files[f"{APP}/{rel}"] = b
    files[f"{APP}/7/1.9000.{TS_MS + 9}.v3.exo"] = MP4[9000:]      # a piece in another subfolder
    for rel, b in _pieces(2, GAPPED, [0, 1000, 2500]).items():
        if not rel.split("/")[1].startswith("2.1000."):              # drop the middle piece
            files[f"{APP}/{rel}"] = b
    files[f"{APP}/5/3.100.{TS_MS}.v3.exo"] = b"\x00" * 50            # no piece at position 0
    for rel, b in _pieces(4, TS, [0, 1000]).items():
        files[f"{APP}/{rel}"] = b
    files[f"{APP}/{exocache.INDEX_NAME}"] = _exi(idx)
    # a cache indexed by the database, tied to its table by the .uid file
    uid = "6a6d2629ba159365"
    other = "data/data/com.example.other/cache/media"
    for rel, b in _pieces(0, MP4, [0, 6000], "0").items():
        files[f"{other}/{rel}"] = b
    files[f"{other}/{uid}.uid"] = b""
    files[f"data/data/com.example.other/databases/{exocache.DB_NAME}"] = _db(
        uid, {0: ("https://media.example.org/clip.mp4", len(MP4))})
    # a v2 cache: the key is in the file name, %xx-escaped
    v2 = "data/data/com.example.old/cache/exo"
    key = "https://old.example.com/a:b"
    esc = key.replace("%", "%25").replace(":", "%3a").replace("/", "%2f")
    files[f"{v2}/{esc}.0.{TS_MS}.v2.exo"] = MP4[:5000]
    files[f"{v2}/{esc}.5000.{TS_MS + 1}.v2.exo"] = MP4[5000:]
    # a cache whose index is encrypted
    enc = "data/data/com.example.locked/cache/exo"
    for rel, b in _pieces(9, MP4, [0]).items():
        files[f"{enc}/{rel}"] = b
    files[f"{enc}/{exocache.INDEX_NAME}"] = _exi({}, encrypted=True)
    return files


def _joined(case):
    """Joined rows by ``<app folder>:<cache id, or key for a v1/v2 cache>``."""
    out = {}
    for r in case.db.iter_files("cache_info IS NOT NULL", ()):
        info = json.loads(r["cache_info"])
        ident = info["cache_id"] if info["cache_id"] is not None else info["key"]
        out[f'{info["app_folder"]}:{ident}'] = r
    return out


def _ingest(tmp_path, *, as_zip: bool, include_other: bool = False):
    files = _tree()
    if as_zip:
        src = tmp_path / "extraction.zip"
        with zipfile.ZipFile(src, "w") as z:
            for name, data in files.items():
                z.writestr(name, data)
        source = Source(name="ext", path=str(src), kind="archive", include_other=include_other)
    else:
        src = tmp_path / "ext"
        for name, data in files.items():
            p = src / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        source = Source(name="ext", path=str(src), include_other=include_other)
    case = open_case(tmp_path / "case", create=True, examiner="t")
    ingest_sources(case, [source])
    return case


@pytest.mark.parametrize("as_zip", [False, True], ids=["folder", "zip"])
def test_each_item_is_joined_in_order_and_described(tmp_path, as_zip):
    case = _ingest(tmp_path, as_zip=as_zip)
    try:
        got = _joined(case)
        # the MP4 was in four pieces across two subfolders
        one = got["com.example.player:1"]
        info = json.loads(one["cache_info"])
        from pathlib import Path
        assert Path(one["path"]).read_bytes() == MP4
        assert (one["kind"], one["ext"]) == ("video", ".mp4")
        assert info["key"] == "https://cdn.example.net/v/1.mp4"
        assert info["key_from"] == exocache.INDEX_NAME
        assert (info["pieces_joined"], info["pieces_total"], info["state"]) == (4, 4, "complete")
        assert info["last_touched_ms"] == TS_MS + 9
        # the middle piece is missing: joined up to the gap, the rest left out
        two = json.loads(got["com.example.player:2"]["cache_info"])
        assert Path(got["com.example.player:2"]["path"]).read_bytes() == GAPPED[:1000]
        assert (two["state"], two["gap_at"], two["pieces_left_out"]) == ("stops at a gap", 1000, 1)
        # an HLS segment is an MPEG transport stream
        ts = got["com.example.player:4"]
        assert (ts["kind"], ts["ext"]) == ("video", ".ts")
        assert Path(ts["path"]).read_bytes() == TS
        # indexed by the database, found through the .uid file
        db = json.loads(got["com.example.other:0"]["cache_info"])
        assert db["key"] == "https://media.example.org/clip.mp4"
        assert db["key_from"].startswith(exocache.DB_NAME)
        # a v2 cache names the key in the file, escaped
        old = json.loads(got["com.example.old:https://old.example.com/a:b"]["cache_info"])
        assert (old["key_from"], old["state"]) == ("file names", "length not recorded")
        # an encrypted index names nothing, and the item is still joined
        locked = json.loads(got["com.example.locked:9"]["cache_info"])
        assert (locked["key"], locked["key_from"]) == (None, "index encrypted")
        # no piece at position 0: nothing that could open, so nothing written
        assert "com.example.player:3" not in got
        assert len(got) == 6
        # each joined file is linked to its first piece
        for r in got.values():
            parent = case.db.get_file(r["container_id"])
            assert parent is not None and is_exoplayer_cache_name(parent["rel_path"])
    finally:
        case.close()


def test_the_pieces_are_containers_and_nothing_opens_them_as_archives(tmp_path, monkeypatch):
    case = _ingest(tmp_path, as_zip=True)
    opened = []
    real = nested._expand_one  # pylint: disable=protected-access
    monkeypatch.setattr(nested, "_expand_one",
                        lambda c, row, **kw: opened.append(row["id"]) or real(c, row, **kw))
    try:
        pieces = [r for r in case.db.iter_files("lower(ext) = '.exo'", ())]
        assert pieces and {r["kind"] for r in pieces} == {"archive"}
        # the first piece of an MP4 opens with a video header and used to be
        # registered as a truncated video
        assert not case.db.iter_files("kind = 'video' AND lower(ext) = '.exo'", ())
        assert nested.expand_containers(case, force=True) == 0
        assert not opened, "an ExoPlayer cache file was opened as an archive"
        assert all(not r["error"] for r in pieces)
    finally:
        case.close()


def test_a_second_pass_adds_nothing_and_a_forced_one_rewrites_the_same_rows(tmp_path):
    case = _ingest(tmp_path, as_zip=False)
    try:
        before = {r["id"]: r["path"] for r in case.db.iter_files("cache_info IS NOT NULL", ())}
        assert exocache.assemble(case) == 0
        assert exocache.assemble(case, force=True) == len(before)
        after = {r["id"]: r["path"] for r in case.db.iter_files("cache_info IS NOT NULL", ())}
        assert after == before
    finally:
        case.close()


def test_an_item_that_is_not_media_is_kept_only_when_asked(tmp_path):
    files = {f"{APP}/0/5.0.{TS_MS}.v3.exo": b"#EXTM3U\n#EXT-X-VERSION:3\n",
             f"{APP}/{exocache.INDEX_NAME}": _exi({5: ("https://x.example/p.m3u8", 25)})}
    for include in (False, True):
        src = tmp_path / f"ext{include}"
        for name, data in files.items():
            (src / name).parent.mkdir(parents=True, exist_ok=True)
            (src / name).write_bytes(data)
        case = open_case(tmp_path / f"case{include}", create=True, examiner="t")
        try:
            ingest_sources(case, [Source(name="ext", path=str(src), include_other=include)])
            rows = case.db.iter_files("cache_info IS NOT NULL", ())
            assert len(rows) == (1 if include else 0)
        finally:
            case.close()


def test_the_backup_index_wins(tmp_path):
    """``AtomicFile`` restores ``.bak`` over the base file when both exist."""
    files = {f"{APP}/0/8.0.{TS_MS}.v3.exo": MP4,
             f"{APP}/{exocache.INDEX_NAME}": _exi({8: ("https://stale.example/", 1)}),
             f"{APP}/{exocache.INDEX_NAME}.bak": _exi({8: ("https://good.example/", len(MP4))})}
    src = tmp_path / "ext"
    for name, data in files.items():
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_bytes(data)
    case = open_case(tmp_path / "case", create=True, examiner="t")
    try:
        ingest_sources(case, [Source(name="ext", path=str(src))])
        (row,) = case.db.iter_files("cache_info IS NOT NULL", ())
        info = json.loads(row["cache_info"])
        assert (info["key"], info["state"]) == ("https://good.example/", "complete")
    finally:
        case.close()


def test_parse_index_file_refuses_a_file_that_does_not_end_where_it_should():
    good = _exi({1: ("k", 10)})
    assert exocache.parse_index_file(good)["state"] == "read"
    assert exocache.parse_index_file(good + b"x")["state"] == "unreadable"
    assert exocache.parse_index_file(good[:-6])["state"] == "unreadable"
    assert exocache.parse_index_file(_exi({}, encrypted=True))["state"] == "encrypted"
    v1 = struct.pack(">ii", 1, 0) + struct.pack(">i", 1) + struct.pack(">i", 7) + _utf("k") \
        + struct.pack(">q", 123) + struct.pack(">i", 0)
    assert exocache.parse_index_file(v1)["entries"] == {7: {"key": "k", "length": 123}}


def test_unescape_key_follows_util_unescapefilename():
    assert exoprobe.unescape_key("a%3ab%2fc%25") == "a:b/c%"
    assert exoprobe.unescape_key("plain") == "plain"
    assert exoprobe.unescape_key("bad%zz") is None


def test_describe_says_what_was_joined(tmp_path):
    case = _ingest(tmp_path, as_zip=False)
    try:
        row = _joined(case)["com.example.player:2"]
        text = exocache.describe(dict(row))
        assert "com.example.player" in text and "https://cdn.example.net/v/2.mp4" in text
        assert "1 of 2 pieces joined" in text and "stops at a gap at byte 1,000" in text
        assert "last written or read 2023-11-14 22:13:20 UTC" in text
        assert exocache.describe({"cache_info": None}) == ""
    finally:
        case.close()


def test_the_index_database_is_read_from_a_copy(tmp_path):
    """Opening an evidence database in place rewrites its -shm even read-only."""
    case = _ingest(tmp_path, as_zip=False)
    try:
        db = tmp_path / "ext" / "data/data/com.example.other/databases" / exocache.DB_NAME
        assert not db.with_name(db.name + "-shm").exists()
        assert not db.with_name(db.name + "-wal").exists()
    finally:
        case.close()


def test_is_cache_name_is_decided_by_the_name():
    for yes in ("12.0.1700000000000.v3.exo", "a%2fb.0.1.v2.exo", "a.0.1.v1.exo",
                "cached_content_index.exi", "cached_content_index.exi.bak",
                "exoplayer_internal.db", "exoplayer_internal.db-wal", "3f2a.uid",
                "x/y/0/1.2.3.v3.exo"):
        assert is_exoplayer_cache_name(yes), yes
    for no in ("video.mp4", "1.2.v3.exo", "cached_content_index.exi2", "notes.uid",
               "exoplayer.db", "x.v4.exo"):
        assert not is_exoplayer_cache_name(no), no
