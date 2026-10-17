"""A C4All report index (C4P/C4M Index.xml) ingests like a Project VIC file: every
entry registered from the folder beside the index, with its hashes, original path,
times and category. Everything here is synthetic."""
from __future__ import annotations

# a pytest fixture and the test argument that receives it share a name, the same
# way tests/test_lava.py does
# pylint: disable=redefined-outer-name

import json
from pathlib import Path

import pytest

from gleapp import c4all
from gleapp.case import open_case, parse_source_spec
from gleapp.pipeline import ingest_sources

NOT_DELETED = "18446744073709551615"


def _entry(tag: str, name_tag: str, folder: str, name: str, *, eid: int, full: str,
           cat: int = 0, md5: str = "0" * 32) -> str:
    return f"""
    <{tag}>
        <path><![CDATA[{folder}]]></path>
        <{name_tag}>{name}</{name_tag}>
        <category>{cat}</category>
        <id>{eid}</id>
        <fileoffset>0</fileoffset>
        <fullpath><![CDATA[{full}]]></fullpath>
        <created>1500000000</created>
        <accessed>1500000100</accessed>
        <written>1500000200</written>
        <deleted>{NOT_DELETED}</deleted>
        <hash>{md5.upper()}</hash>
        <sha1Hash>{"A" * 40}</sha1Hash>
        <encaseHash></encaseHash>
        <myDescription>File, Archive</myDescription>
        <fileSize>4</fileSize>
        <monikers></monikers>
    </{tag}>"""


def _write_index(path: Path, entries: list[str]) -> Path:
    body = ('<?xml version="1.0" encoding="utf-16"?>\n<ReportIndex version="2.0">'
            + "".join(entries) + "\n</ReportIndex>\n")
    path.write_text(body, encoding="utf-16")
    return path


@pytest.fixture()
def case(tmp_path):
    c = open_case(tmp_path / "case", create=True, examiner="t")
    yield c
    c.close()


def test_picture_index_registers_files_beside_it(tmp_path, case):
    ev = tmp_path / "ev"
    (ev / "DVD01(X)" / "01").mkdir(parents=True)
    (ev / "DVD01(X)" / "01" / "00000001.JPG").write_bytes(b"\xff\xd8\xff\xe0")
    idx = _write_index(ev / "C4P Index.xml", [
        _entry("Image", "picture", "DVD01(X)\\01\\", "00000001.JPG", eid=1,
               full="X\\C\\DCIM\\IMG_1.JPG", cat=2, md5="ab" * 16),
        _entry("Image", "picture", "DVD01(X)\\01\\", "00000001.JPG", eid=2,
               full="X\\C\\Backup\\IMG_1.JPG"),
        _entry("Image", "picture", "DVD01(X)\\01\\", "00000002.JPG", eid=3,
               full="X\\C\\DCIM\\IMG_2.JPG"),
    ])
    assert c4all.is_c4all_file(idx)

    sources, _ = parse_source_spec(idx)
    assert [s.kind for s in sources] == ["c4all"]
    ingest_sources(case, sources, expand_archives=False)

    rows = {Path(r["path"]).name: r for r in case.db.conn.execute("SELECT * FROM files")}
    assert set(rows) == {"00000001.JPG", "00000002.JPG"}
    one = rows["00000001.JPG"]
    assert one["kind"] == "image"
    assert one["md5"] == "ab" * 16
    assert one["sha1"] == "a" * 40
    assert one["orig_path"] == "X\\C\\DCIM\\IMG_1.JPG"
    assert one["orig_name"] == "IMG_1.JPG"
    assert json.loads(one["alt_paths"]) == ["X\\C\\Backup\\IMG_1.JPG"]
    assert one["category"] == 2
    assert one["mtime"] == 1500000200
    assert not one["error"]
    assert rows["00000002.JPG"]["error"] == "file not found on disk"


def test_movie_index_is_video_and_zero_category_is_none(tmp_path, case):
    idx = _write_index(tmp_path / "C4M Index.xml", [
        _entry("Movie", "movie", "D\\01\\", "00000001.MOV", eid=1, full="X\\a.MOV"),
    ])
    sources, _ = parse_source_spec(idx)
    ingest_sources(case, sources, expand_archives=False)
    row = case.db.conn.execute("SELECT kind, category FROM files").fetchone()
    assert row["kind"] == "video"
    assert row["category"] in (None, 5)   # left uncategorised


def test_other_xml_is_not_c4all(tmp_path):
    p = tmp_path / "other.xml"
    p.write_text('<?xml version="1.0"?><plist/>', encoding="utf-8")
    assert not c4all.is_c4all_file(p)
