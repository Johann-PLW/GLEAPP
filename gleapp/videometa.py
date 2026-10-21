"""Location and capture date from a video's own metadata (MP4 / MOV / 3GP).

Phones and cameras write a video's location as ISO 6709 text in the QuickTime
structure, in one of two places:

* ``moov/udta/©xyz``, the UserData GPSCoordinates ExifTool reports (Android and
  many cameras): a 16-bit length, a 16-bit language code, then the text.
* ``moov/meta`` with Apple's ``keys`` and ``ilst`` lists (iPhone):
  ``com.apple.quicktime.location.ISO6709``, beside
  ``com.apple.quicktime.creationdate``, the capture time with its UTC offset.

Both read only a handful of boxes, never the media data. The movie header's own
creation time is not used for "Captured": it is UTC on most devices, and Captured
is the camera's local time, as for photos.
"""
from __future__ import annotations

import math
import re
import struct
from pathlib import Path

_CONTAINERS = {b"moov", b"udta", b"trak", b"meta", b"ilst"}
_MAX_BOX_READ = 1 << 20          # a metadata box holding more than this is not one


def _boxes(fh, start: int, end: int):
    """(type, payload_start, payload_end) for each box between ``start`` and ``end``."""
    pos = start
    while pos + 8 <= end:
        fh.seek(pos)
        head = fh.read(8)
        if len(head) < 8:
            return
        size, kind = struct.unpack(">I4s", head)
        hdr = 8
        if size == 1:
            big = fh.read(8)
            if len(big) < 8:
                return
            size = struct.unpack(">Q", big)[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr or pos + size > end:
            return
        yield kind, pos + hdr, pos + size
        pos += size


def _walk(fh, start: int, end: int, path: tuple = ()):
    """Every box inside the metadata containers, with its path from the top."""
    for kind, a, b in _boxes(fh, start, end):
        here = path + (kind,)
        yield here, a, b
        if kind in _CONTAINERS:
            # an ISO "meta" box is a full box (4 bytes of version/flags first);
            # Apple's QuickTime "meta" is not. The first child tells them apart.
            inner = a
            if kind == b"meta":
                fh.seek(a + 4)
                if fh.read(4) not in (b"hdlr", b"keys", b"ilst"):
                    inner = a + 4
            yield from _walk(fh, inner, b, here)


def parse_iso6709(text: str) -> tuple[float, float] | None:
    """Latitude and longitude from an ISO 6709 string such as ``+42.6116-074.9314/``
    or ``+4236.69-07455.88+120/``, in decimal degrees; None if it is not one."""
    m = re.match(r"\s*([+-])(\d+(?:\.\d+)?)([+-])(\d+(?:\.\d+)?)", text or "")
    if not m:
        return None

    def deg(sign: str, num: str, deg_digits: int) -> float:
        whole, _, frac = num.partition(".")
        frac = "." + frac if frac else ""
        n = len(whole) - deg_digits          # 0: degrees, 2: +minutes, 4: +seconds
        if n == 0:
            v = float(whole + frac)
        elif n == 2:
            v = float(whole[:deg_digits]) + float(whole[deg_digits:] + frac) / 60
        elif n == 4:
            v = (float(whole[:deg_digits]) + float(whole[deg_digits:deg_digits + 2]) / 60
                 + float(whole[deg_digits + 2:] + frac) / 3600)
        else:
            raise ValueError(num)
        return -v if sign == "-" else v

    try:
        lat = deg(m.group(1), m.group(2), 2)
        lon = deg(m.group(3), m.group(4), 3)
    except ValueError:
        return None
    return valid_coords(lat, lon)


def valid_coords(lat, lon) -> tuple[float, float] | None:
    """``(lat, lon)`` rounded, or None unless both are real coordinates."""
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if math.isnan(lat) or math.isnan(lon) or abs(lat) > 90 or abs(lon) > 180:
        return None
    if lat == 0 and lon == 0:            # "no fix" written as a location
        return None
    return round(lat, 7), round(lon, 7)


def _apple_date(text: str) -> str | None:
    """``2019-05-04T10:11:12-0400`` -> ``2019-05-04T10:11:12``: the local time as the
    device recorded it, the same wall-clock form a photo's EXIF capture time has."""
    m = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})", text or "")
    return "{}-{}-{}T{}:{}:{}".format(*m.groups()) if m else None


def read(path: str | Path) -> dict:
    """``{"gps_lat", "gps_lon", "created_dt"}`` from a video's metadata, each None
    when the file does not carry it. Never raises for a file it cannot read."""
    out: dict = {"gps_lat": None, "gps_lon": None, "created_dt": None}
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            moov = next(((a, b) for k, a, b in _boxes(fh, 0, size) if k == b"moov"), None)
            if moov is None:
                return out
            keys: list[str] = []
            items: dict[int, str] = {}
            xyz = None
            for path_, a, b in _walk(fh, *moov):
                kind = path_[-1]
                if b - a > _MAX_BOX_READ:
                    continue
                if kind == b"\xa9xyz" and xyz is None:
                    fh.seek(a)
                    raw = fh.read(b - a)
                    if len(raw) >= 4:
                        n = struct.unpack(">H", raw[:2])[0]
                        xyz = raw[4:4 + n].decode("utf-8", "replace")
                elif kind == b"keys":
                    fh.seek(a + 8)             # version/flags, entry count
                    end = b
                    while fh.tell() + 8 <= end:
                        ksize, _ns = struct.unpack(">I4s", fh.read(8))
                        if ksize < 8:
                            break
                        keys.append(fh.read(ksize - 8).decode("utf-8", "replace"))
                elif len(path_) >= 2 and path_[-2] == b"ilst" and len(kind) == 4:
                    index = struct.unpack(">I", kind)[0]
                    for dk, da, db in _boxes(fh, a, b):
                        if dk == b"data" and db - da > 8:
                            fh.seek(da + 8)    # type, locale
                            items[index] = fh.read(db - da - 8).decode("utf-8", "replace")
                            break
            apple = {keys[i - 1]: v for i, v in items.items() if 0 < i <= len(keys)}
            loc = (parse_iso6709(apple.get("com.apple.quicktime.location.ISO6709", ""))
                   or parse_iso6709(xyz or ""))
            if loc:
                out["gps_lat"], out["gps_lon"] = loc
            out["created_dt"] = _apple_date(apple.get("com.apple.quicktime.creationdate", ""))
    except (OSError, struct.error, ValueError):
        pass
    return out
