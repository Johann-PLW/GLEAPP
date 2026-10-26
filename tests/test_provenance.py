"""Content Credentials (C2PA): the opt-in read, what is kept, and where it shows.

Every file here is synthetic: real pictures made with Pillow with a C2PA manifest
store or an XMP packet written into them by tests/c2pabuild.py.
"""
# pylint: disable=redefined-outer-name
import csv
import io
import json
import re
import struct
import sys
import time
import zipfile
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gleapp import provenance, report
from gleapp.case import open_case, parse_source_spec
from gleapp.pipeline import ingest_sources, process

sys.path.insert(0, str(Path(__file__).parent))
import c2pabuild  # noqa: E402  pylint: disable=wrong-import-position


def _png(color=(30, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buf, "PNG")
    return buf.getvalue()


def _jpg(color=(200, 40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buf, "JPEG")
    return buf.getvalue()


def _png_with_store(store: bytes, color=(30, 120, 200)) -> bytes:
    """A real PNG with a caBX chunk before IEND."""
    data = _png(color)
    iend = data.rindex(b"IEND") - 4
    chunk = (struct.pack(">I", len(store)) + b"caBX" + store
             + struct.pack(">I", zlib.crc32(b"caBX" + store)))
    return data[:iend] + chunk + data[iend:]


def _jpg_with_xmp(packet: bytes, color=(200, 40, 40)) -> bytes:
    """A real JPEG with an APP1 XMP segment after SOI."""
    data = _jpg(color)
    seg = b"http://ns.adobe.com/xap/1.0/\x00" + packet
    return data[:2] + b"\xff\xe1" + struct.pack(">H", len(seg) + 2) + seg + data[2:]


AI_STORE = c2pabuild.store(
    c2pabuild.manifest("urn:c2pa:ingredient", source_types=("trainedAlgorithmicMedia",)),
    c2pabuild.manifest("urn:c2pa:active", source_types=(), actions=("c2pa.edited",)))


def _mp4(p: Path) -> None:
    # cv2 is a compiled module astroid cannot introspect, so its members read as absent
    # pylint: disable=no-member
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(10):
        vw.write(np.full((48, 64, 3), i * 20, np.uint8))
    vw.release()


def _evidence(tmp_path) -> Path:
    ev = tmp_path / "ev"
    ev.mkdir()
    (ev / "signed.png").write_bytes(_png_with_store(AI_STORE))
    (ev / "declared.jpg").write_bytes(_jpg_with_xmp(c2pabuild.xmp("digitalCapture")))
    (ev / "plain.jpg").write_bytes(_jpg((10, 200, 10)))
    # a PNG cut off before IEND: the walk cannot reach the end, so absence is not settled
    (ev / "cut.png").write_bytes(_png((90, 90, 90))[:-12])
    _mp4(ev / "clip.mp4")
    return ev


def _made_case(tmp_path):
    c = open_case(tmp_path / "case", create=True, examiner="t")
    sources, _ = parse_source_spec(str(_evidence(tmp_path)))
    ingest_sources(c, sources)
    process(c, workers=2, keyframes=2, screen=False, similar=False)
    return c


@pytest.fixture
def case(tmp_path):
    c = _made_case(tmp_path)
    yield c
    c.close()


def _by_name(case):
    return {Path(r["rel_path"]).name: r["id"] for r in case.db.iter_files()}


def test_nothing_is_read_until_asked(case):
    assert case.db.conn.execute("SELECT COUNT(*) FROM c2pa_meta").fetchone()[0] == 0
    ids = _by_name(case)
    assert provenance.for_file(case.db, ids["signed.png"]) is None


def test_a_pass_reads_each_file_once_and_keeps_what_it_found(case):
    counts = provenance.read_pass(case, workers=2)
    ids = _by_name(case)
    got = {name: provenance.for_file(case.db, i) for name, i in ids.items()}
    assert got["signed.png"]["credentials"] == "c2pa"
    # the AI statement is on the ingredient's manifest, and still counts
    assert got["signed.png"]["ai"] == "yes"
    assert got["signed.png"]["signer"] == "Example Org (Test Signer)"
    assert got["declared.jpg"]["credentials"] == "xmp"
    assert got["declared.jpg"]["ai"] == "no"
    assert got["plain.jpg"]["credentials"] == "none"
    assert got["plain.jpg"]["ai"] is None
    assert got["clip.mp4"]["credentials"] == "none"
    assert counts["read"] == len(ids) and counts["failed"] == 0
    assert counts["c2pa"] == 1 and counts["xmp"] == 1
    # a second pass reads nothing; a reread reads everything again
    assert provenance.read_pass(case)["read"] == 0
    assert provenance.read_pass(case, reread=True)["read"] == len(ids)
    actions = [r["action"] for r in case.db.conn.execute("SELECT action FROM audit")]
    assert actions.count("c2pa_read") == 3


def test_a_cut_file_is_not_settled_rather_than_none(tmp_path):
    p = tmp_path / "cut.png"
    p.write_bytes(_png()[:-12])
    assert provenance.summarize(provenance.read_path(p))["credentials"] == "unsettled"


def test_an_empty_file_is_none_found_and_an_unwalked_format_is_not_settled(tmp_path):
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    got = provenance.read_path(empty)
    assert got["file_size"] == 0
    assert provenance.summarize(got)["credentials"] == "none"
    assert ("Note", "the file is empty") in provenance.describe(
        {"credentials": "none", "ai": None, "record": got})[0][1]
    bmp = tmp_path / "a.bmp"
    Image.new("RGB", (4, 4)).save(bmp, "BMP")
    assert provenance.summarize(provenance.read_path(bmp))["credentials"] == "unsettled"


def test_exports_carry_the_reading_and_say_not_read(case, tmp_path):
    ids = _by_name(case)
    out = report.export_csv(case, tmp_path / "before.csv")
    rows = {Path(r["file_path"]).name: r for r in csv.DictReader(io.StringIO(out.read_text(encoding="utf-8")))}
    assert rows["signed.png"]["c2pa_credentials"] == "Not read"
    provenance.read_pass(case)
    out = report.export_csv(case, tmp_path / "after.csv")
    rows = {Path(r["file_path"]).name: r for r in csv.DictReader(io.StringIO(out.read_text(encoding="utf-8")))}
    assert rows["signed.png"]["c2pa_credentials"] == "C2PA manifest"
    assert rows["signed.png"]["c2pa_ai"] == "Yes"
    assert rows["signed.png"]["c2pa_signer_stated"] == "Example Org (Test Signer)"
    assert rows["signed.png"]["c2pa_source_types"] == "trainedAlgorithmicMedia"
    assert rows["declared.jpg"]["c2pa_source_types"] == "digitalCapture"
    assert rows["plain.jpg"]["c2pa_credentials"] == "None found"

    out = report.export_json(case, tmp_path / "r.json")
    files = {Path(f["file_path"]).name: f for f in json.loads(out.read_text(encoding="utf-8"))["files"]}
    assert files["signed.png"]["c2pa"]["credentials"] == "c2pa"
    assert files["signed.png"]["c2pa"]["record"]["active_manifest"] == "urn:c2pa:active"
    assert all("_c2pa_entry" not in f for f in files.values())
    assert set(files) == set(ids)


def test_the_report_card_block_is_opt_in(case, tmp_path):
    provenance.read_pass(case)
    off = report.export_html(case, tmp_path / "off", maps=False)
    on = report.export_html(case, tmp_path / "on", maps=False, c2pa=True)
    read = lambda p: (p if p.is_file() else p / "index.html").read_text(encoding="utf-8")  # noqa: E731
    assert "details class='exif c2pa'" not in read(off)
    html_on = read(on)
    assert html_on.count("details class='exif c2pa'") == 5
    assert "Content Credentials &middot; C2PA manifest" in html_on
    assert "Signed by (stated)" in html_on
    assert "Nothing is verified" in html_on


def _client(case_dir):
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    return create_app(str(case_dir)).test_client()


def _wait(cl, timeout=60):
    for _ in range(timeout * 4):
        job = cl.get("/api/job").get_json()
        if not job["running"]:
            return job
        time.sleep(0.25)
    raise AssertionError("job never finished")


def test_the_gallery_reads_later_filters_and_counts(tmp_path):
    case = _made_case(tmp_path)
    root = case.root
    ids = _by_name(case)
    case.close()
    cl = _client(root)
    ctx = cl.get("/api/context").get_json()
    assert ctx["c2pa"]["media"] == 5 and ctx["c2pa"]["unread"] == 5
    names = lambda q: sorted(Path(f["rel_path"]).name for f in   # noqa: E731
                             cl.get(f"/api/files?{q}").get_json()["files"])
    assert names("c2pa=unread") == ["clip.mp4", "cut.png", "declared.jpg", "plain.jpg",
                                     "signed.png"]
    assert cl.get(f"/api/file/{ids['signed.png']}/c2pa").get_json()["read"] is False

    assert cl.post("/api/c2pa", json={}).status_code == 200
    assert _wait(cl)["stage"] == "done"
    ctx = cl.get("/api/context").get_json()["c2pa"]
    assert (ctx["unread"], ctx["c2pa"], ctx["xmp"], ctx["none"], ctx["unsettled"],
            ctx["ai_yes"], ctx["ai_no"]) == (0, 1, 1, 2, 1, 1, 1)
    assert names("c2pa=c2pa") == ["signed.png"]
    assert names("c2pa=declared") == ["declared.jpg", "signed.png"]
    assert names("c2pa=none") == ["clip.mp4", "plain.jpg"]
    assert names("c2pa=unsettled") == ["cut.png"]
    assert names("c2pa_ai=yes") == ["signed.png"]
    assert names("c2pa_ai=no") == ["declared.jpg"]
    assert names("c2pa=unread") == []
    one = cl.get(f"/api/file/{ids['signed.png']}/c2pa").get_json()
    assert one["read"] is True and one["credentials"] == "c2pa"
    assert [g[0] for g in one["groups"]][:3] == [
        "Content Credentials", "Manifest 1 of 2 (ingredient)", "Manifest 2 of 2 (active)"]
    actions = [r["action"] for r in cl.get("/api/audit").get_json()]
    assert "c2pa_read" in actions


def test_the_ingest_option_reads_after_processing(tmp_path):
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    for on in (False, True):
        cl = create_app(None).test_client()
        d = tmp_path / f"c{int(on)}"
        assert cl.post("/api/case/create", json={"path": str(d), "name": "c",
                                                 "examiner": "t"}).status_code == 200
        (tmp_path / f"e{int(on)}").mkdir()
        ev = _evidence(tmp_path / f"e{int(on)}")
        r = cl.post("/api/case/ingest", json={"sources": [{"path": str(ev)}],
                                              "options": {"screen": False, "keyframes": 2,
                                                          "c2pa": on}})
        assert r.status_code == 200, r.get_json()
        job = _wait(cl)
        assert job["stage"] == "done", job
        c = open_case(d)
        n = c.db.conn.execute("SELECT COUNT(*) FROM c2pa_meta").fetchone()[0]
        c.close()
        assert n == (5 if on else 0)
        if on:
            assert job["stats"]["c2pa"]["c2pa"] == 1


def test_a_reference_mode_archive_is_read_from_the_archive(tmp_path):
    z = tmp_path / "ext.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("DCIM/signed.png", _png_with_store(AI_STORE))
        zf.writestr("DCIM/plain.jpg", _jpg())
    c = open_case(tmp_path / "case", create=True, examiner="t")
    sources, _ = parse_source_spec(str(z))
    assert not sources[0].stage
    ingest_sources(c, sources)
    process(c, workers=2, keyframes=2, screen=False, similar=False)
    counts = provenance.read_pass(c, workers=2)
    assert counts["read"] == 2 and counts["c2pa"] == 1
    tmp = c.root / "tmp"
    assert not tmp.exists() or not any(tmp.iterdir())
    c.close()


def test_the_screens_carry_the_option_and_the_filters():
    root = Path(__file__).resolve().parents[1] / "gleapp" / "web"
    html_ = (root / "templates" / "index.html").read_text(encoding="utf-8")
    js = (root / "static" / "app.js").read_text(encoding="utf-8")
    for id_ in ("optC2pa", "aeC2pa", "fc2pa", "fc2paAi", "btnC2pa", "rhC2pa"):
        assert f'id="{id_}"' in html_, id_
    assert re.search(r'c2pa:\s*\$\("#optC2pa"\)\.checked', js)
    assert re.search(r'c2pa:\s*\$\("#aeC2pa"\)\.checked', js)
    assert 'p.set("c2pa", $("#fc2pa").value)' in js
    assert 'j.stage === "c2pa"' in js


def test_the_gallery_tiles_carry_the_reading(tmp_path):
    case = _made_case(tmp_path)
    provenance.read_pass(case)
    root = case.root
    case.close()
    cl = _client(root)
    files = {Path(f["rel_path"]).name: f for f in
             cl.get("/api/files?limit=50&collapse=0").get_json()["files"]}
    assert (files["signed.png"]["c2pa_credentials"], files["signed.png"]["c2pa_ai"]) == ("c2pa", "yes")
    assert (files["declared.jpg"]["c2pa_credentials"], files["declared.jpg"]["c2pa_ai"]) == ("xmp", "no")
    assert files["plain.jpg"]["c2pa_ai"] is None
    js = (Path(__file__).resolve().parents[1] / "gleapp" / "web" / "static" / "app.js").read_text(
        encoding="utf-8")
    assert 'f.c2pa_ai === "yes"' in js and 'f.c2pa_credentials === "c2pa"' in js


def test_lava_carries_the_reading(tmp_path):
    import sqlite3                                      # pylint: disable=import-outside-toplevel
    from gleapp import lava                             # pylint: disable=import-outside-toplevel
    case = _made_case(tmp_path)
    out0 = tmp_path / "lava0"
    lava.export_lava(case, out0)
    manifest = json.loads((out0 / "_lava_data.lava").read_text(encoding="utf-8"))
    db0 = sqlite3.connect(out0 / manifest["lava_db_name"])
    try:                                                # nothing read: no empty table
        assert not db0.execute("SELECT 1 FROM sqlite_master WHERE name = "
                               "'content_credentials'").fetchone()
        assert db0.execute("SELECT content_credentials FROM media_files "
                           "WHERE file_name = 'signed.png'").fetchone()[0] == "Not read"
    finally:
        db0.close()
    provenance.read_pass(case)
    out = tmp_path / "lava1"
    lava.export_lava(case, out)
    manifest = json.loads((out / "_lava_data.lava").read_text(encoding="utf-8"))
    db = sqlite3.connect(out / manifest["lava_db_name"])
    try:
        rows = db.execute("SELECT file_name, manifest, ai_declared, digital_source_type, "
                          "metadata_source, signed_by_stated, signing_time_tsa, "
                          "content_credentials FROM content_credentials ORDER BY file_name, "
                          "manifest").fetchall()
        media = dict(db.execute("SELECT file_name, content_credentials FROM media_files"))
    finally:
        db.close()
    case.close()
    assert [(r[0], r[1], r[2], r[4]) for r in rows] == [
        ("declared.jpg", "IPTC tag", "No", "XMP/IPTC"),
        ("signed.png", "1 of 2", "Yes", "C2PA"),
        ("signed.png", "2 of 2 (active)", "Unknown", "C2PA")]
    assert rows[1][3] == "Created using Generative AI (trainedAlgorithmicMedia)"
    assert rows[1][5] == "Example Org (Test Signer)"
    assert rows[1][6] == 1748859072                    # 2025-06-02 10:11:12 UTC
    assert media["signed.png"] == "C2PA manifest" and media["plain.jpg"] == "None found"
