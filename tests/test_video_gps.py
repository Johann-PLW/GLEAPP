"""Location and capture date GLEAPP reads itself: a video's QuickTime metadata
(UserData ©xyz and Apple's keys/ilst), and the image GPS layouts beyond three-part
degrees/minutes/seconds, with XMP as a fallback. Everything here is synthetic.
"""
from __future__ import annotations

import struct

import piexif
import pytest
from PIL import Image

from gleapp import videometa
from gleapp.metadata import extract_image
from gleapp.pipeline import _add_video_metadata


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _mp4(moov_children: bytes) -> bytes:
    return (_box(b"ftyp", b"isom\0\0\0\0isommp41")
            + _box(b"moov", _box(b"mvhd", b"\0" * 100) + moov_children)
            + _box(b"mdat", b"\0" * 16))


def _xyz(text: str) -> bytes:
    raw = text.encode()
    return _box(b"udta", _box(b"\xa9xyz", struct.pack(">HH", len(raw), 0x15c7) + raw))


def _apple(items: dict[str, str]) -> bytes:
    keys = b"".join(_box(b"mdta", k.encode()) for k in items)
    ilst = b"".join(
        _box(struct.pack(">I", i), _box(b"data", struct.pack(">II", 1, 0) + v.encode()))
        for i, v in enumerate(items.values(), 1))
    return _box(b"meta", _box(b"hdlr", b"\0" * 24)
                + _box(b"keys", struct.pack(">II", 0, len(items)) + keys)
                + _box(b"ilst", ilst))


@pytest.mark.parametrize("text, want", [
    ("+42.6116-074.9313/", (42.6116, -74.9313)),
    ("+42.6116-074.9313+120.0/", (42.6116, -74.9313)),
    ("-33.8688+151.2093/", (-33.8688, 151.2093)),
    ("+4236.6963-07455.8817/", (42.611605, -74.931362)),      # degrees and minutes
    ("+423641.78-0745552.90/", (42.6116056, -74.9313611)),    # degrees, minutes, seconds
])
def test_iso6709_forms(text, want):
    got = videometa.parse_iso6709(text)
    assert got == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize("text", ["", "nonsense", "+0.0000+000.0000/", "+95.0+010.0/"])
def test_not_a_location(text):
    assert videometa.parse_iso6709(text) is None


def test_userdata_gps_coordinates(tmp_path):
    p = tmp_path / "android_clip"            # no extension, as in an extraction
    p.write_bytes(_mp4(_xyz("+42.6116-074.9313/")))
    got = videometa.read(p)
    assert (got["gps_lat"], got["gps_lon"]) == pytest.approx((42.6116, -74.9313))
    assert got["created_dt"] is None


def test_apple_location_and_creation_date(tmp_path):
    p = tmp_path / "IMG_0001.MOV"
    p.write_bytes(_mp4(_apple({
        "com.apple.quicktime.make": "Apple",
        "com.apple.quicktime.location.ISO6709": "+40.4463-079.9823+250.0/",
        "com.apple.quicktime.creationdate": "2019-05-04T10:11:12-0400",
    })))
    got = videometa.read(p)
    assert (got["gps_lat"], got["gps_lon"]) == pytest.approx((40.4463, -79.9823))
    assert got["created_dt"] == "2019-05-04T10:11:12"      # the device's local time


def test_a_video_without_metadata_or_not_a_video(tmp_path):
    plain = tmp_path / "plain.mp4"
    plain.write_bytes(_mp4(b""))
    junk = tmp_path / "junk"
    junk.write_bytes(b"\x00\x01garbage" * 10)
    empty = {"gps_lat": None, "gps_lon": None, "created_dt": None}
    assert videometa.read(plain) == empty
    assert videometa.read(junk) == empty
    assert videometa.read(tmp_path / "missing") == empty


def test_processing_keeps_an_imported_capture_date(tmp_path):
    p = tmp_path / "clip.mov"
    p.write_bytes(_mp4(_apple({
        "com.apple.quicktime.location.ISO6709": "+40.4463-079.9823/",
        "com.apple.quicktime.creationdate": "2019-05-04T10:11:12-0400"})))
    upd: dict = {}
    _add_video_metadata(upd, str(p), "2001-01-01T00:00:00")
    assert upd == {"gps_lat": 40.4463, "gps_lon": -79.9823}
    upd = {}
    _add_video_metadata(upd, str(p), None)
    assert upd["created_dt"] == "2019-05-04T10:11:12"


N = piexif.GPSIFD


def _photo(tmp_path, name, gps=None, xmp=None):
    p = tmp_path / name
    kw = {}
    if gps is not None:
        kw["exif"] = piexif.dump({"0th": {}, "Exif": {}, "GPS": gps})
    if xmp is not None:
        kw["xmp"] = xmp.encode()
    Image.new("RGB", (64, 48), (9, 9, 9)).save(p, "JPEG", **kw)
    return extract_image(p)


@pytest.mark.parametrize("lat, lon", [
    (((40, 1), (26, 1), (4680, 100)), ((79, 1), (58, 1), (5640, 100))),       # D M S
    (((40, 1), (26780, 1000)), ((79, 1), (58940, 1000))),                     # D M.m
    (((40, 1), (26780, 1000), (0, 0)), ((79, 1), (58940, 1000), (0, 0))),     # 0/0 seconds
    (((4044633, 100000),), ((7998233, 100000),)),                             # decimal
])
def test_every_exif_gps_layout(tmp_path, lat, lon):
    r = _photo(tmp_path, "p.jpg", {N.GPSLatitudeRef: b"N", N.GPSLatitude: lat,
                                    N.GPSLongitudeRef: b"W", N.GPSLongitude: lon})
    assert (r["gps_lat"], r["gps_lon"]) == pytest.approx((40.44633, -79.98233), abs=1e-4)


def test_no_fix_is_not_a_location(tmp_path):
    zero = ((0, 1), (0, 1), (0, 1))
    r = _photo(tmp_path, "p.jpg", {N.GPSLatitudeRef: b"N", N.GPSLatitude: zero,
                                    N.GPSLongitudeRef: b"E", N.GPSLongitude: zero})
    assert r["gps_lat"] is None and r["gps_lon"] is None


def test_xmp_gps_and_date_when_exif_has_none(tmp_path):
    xmp = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
           'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description '
           'xmlns:exif="http://ns.adobe.com/exif/1.0/" exif:GPSLatitude="40,26.78N" '
           'exif:GPSLongitude="79,58,56.4W" exif:DateTimeOriginal="2021-06-07T08:09:10-04:00"/>'
           '</rdf:RDF></x:xmpmeta>')
    r = _photo(tmp_path, "edited.jpg", xmp=xmp)
    assert (r["gps_lat"], r["gps_lon"]) == pytest.approx((40.4463333, -79.9823333))
    assert r["created_dt"] == "2021-06-07T08:09:10"
