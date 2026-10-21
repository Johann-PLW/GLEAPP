"""ExifTool metadata for the details pane and the HTML report: how ExifTool's answer
is read, which fields are dropped, how fields fall into the export menu's categories,
the case cache, and the report's category filter. ExifTool itself is faked, so this
runs where it is not installed. Everything here is synthetic.
"""
from __future__ import annotations

# a pytest fixture and the test argument that receives it share a name, the same
# way tests/test_lava.py does
# pylint: disable=redefined-outer-name

import json

import pytest

from gleapp import exiftool, report
from gleapp.case import open_case

_ANSWER = json.dumps([{
    "SourceFile": "C:/somewhere/IMG_0001.JPG",
    "ExifTool:ExifToolVersion": 13.59,
    "File:System:FileName": "IMG_0001.JPG",
    "File:System:Directory": "C:/somewhere",
    "File:File:FileType": "JPEG",
    "EXIF:IFD0:Make": "Canon",
    "EXIF:ExifIFD:ExposureTime": "1/60",
    "EXIF:GPS:GPSLatitude": "12 deg 0' 0.00\" N",
    "MakerNotes:Canon:LensModel": "EF50mm",
    "XMP:XMP-dc:Subject": ["a", "b"],
    "Composite:Composite:GPSPosition": "12, 34",
    "QuickTime:Track1:Duration": "5 s",
    "Unusual:Thing:Tag": "x",
}])


def test_read_keeps_the_evidence_fields_and_drops_the_examiners_copy(monkeypatch):
    monkeypatch.setattr(exiftool, "_ask", lambda args: _ANSWER)
    fields = exiftool.read("ignored")
    keys = {(x["family"], x["group"], x["tag"]) for x in fields}
    assert ("EXIF", "IFD0", "Make") in keys
    assert ("XMP", "XMP-dc", "Subject") in keys
    # the copy's own name and folder, and ExifTool's version line, are not evidence
    assert not any(x["group"] == "System" or x["family"] == "ExifTool" for x in fields)
    assert not any("somewhere" in x["value"] for x in fields)
    subject = next(x for x in fields if x["tag"] == "Subject")
    assert json.loads(subject["value"]) == ["a", "b"]


def test_every_field_falls_in_one_of_the_menu_categories(monkeypatch):
    monkeypatch.setattr(exiftool, "_ask", lambda args: _ANSWER)
    got = {x["tag"]: exiftool.category(x) for x in exiftool.read("ignored")}
    assert got == {"FileType": "file", "Make": "exif", "ExposureTime": "exif",
                   "GPSLatitude": "gps", "LensModel": "makernotes", "Subject": "xmp",
                   "GPSPosition": "gps", "Duration": "video", "Tag": "other"}
    assert {k for k, _ in exiftool.CATEGORIES} >= set(got.values())


@pytest.fixture()
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("GLEAPP_CONFIG_DIR", str(tmp_path / "cfg"))
    pic = tmp_path / "ev" / "a.jpg"
    pic.parent.mkdir()
    pic.write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 64)
    c = open_case(tmp_path / "case", create=True, examiner="t")
    c.db.upsert_file(str(pic), rel_path="a.jpg", source="ev", kind="image", ext=".jpg", size=68)
    c.db.commit()
    yield c
    c.close()


def test_a_file_is_read_once_and_kept_in_the_case(case, monkeypatch):
    calls = []
    monkeypatch.setattr(exiftool, "find", lambda: "exiftool")
    monkeypatch.setattr(exiftool, "version", lambda: "13.59")
    monkeypatch.setattr(exiftool, "_ask", lambda args: _ANSWER)
    real_read = exiftool.read

    def fake_read(path):
        calls.append(path)
        return real_read(path)
    monkeypatch.setattr(exiftool, "read", fake_read)
    row = case.db.get_file(1)
    first = exiftool.for_row(case, row)
    second = exiftool.for_row(case, row)
    assert len(calls) == 1
    assert first["fields"] == second["fields"] and second["version"] == "13.59"


def test_without_exiftool_the_answer_says_so(case, monkeypatch):
    monkeypatch.setattr(exiftool, "find", lambda: None)
    assert exiftool.for_row(case, case.db.get_file(1)) == {"available": False, "fields": []}


def test_the_report_carries_only_the_categories_picked(case, monkeypatch):
    monkeypatch.setattr(exiftool, "find", lambda: "exiftool")
    monkeypatch.setattr(exiftool, "version", lambda: "13.59")
    monkeypatch.setattr(exiftool, "_ask", lambda args: _ANSWER)
    out = case.root / "r.html"
    everything = report.export_html(case, out, exiftool=True, maps=False).read_text("utf-8")
    assert everything.count("class='exifc'") == 7      # file, exif, gps, makernotes, xmp, video, other
    assert "Maker notes" in everything and "LensModel" in everything
    gps_only = report.export_html(case, out, exiftool=True, exiftool_categories=["gps"],
                                  maps=False).read_text("utf-8")
    assert gps_only.count("class='exifc'") == 1 and "GPSLatitude" in gps_only
    assert "LensModel" not in gps_only
    off = report.export_html(case, out, maps=False).read_text("utf-8")
    assert "class='exif'" not in off
