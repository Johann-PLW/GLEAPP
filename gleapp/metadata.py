"""EXIF / metadata extraction for images (and best-effort for video)."""

from __future__ import annotations

import datetime as _dt
import math
import re
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from .videometa import valid_coords

_GPS_IFD = ExifTags.IFD.GPSInfo
_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
_GPSTAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}


def _part(x: Any) -> float:
    """One EXIF rational as a float. ``0/0``, which some writers use for an unused
    part, counts as 0 rather than becoming NaN."""
    if getattr(x, "denominator", None) == 0:
        return 0.0
    if isinstance(x, tuple) and len(x) == 2:
        return 0.0 if x[1] == 0 else x[0] / x[1]
    v = float(x)
    return 0.0 if math.isnan(v) else v


def _to_deg(value: Any, ref: Any) -> float | None:
    """Degrees from an EXIF GPS coordinate, given as degrees/minutes/seconds (the
    usual three values), degrees and decimal minutes (two), or decimal degrees
    (one). None when it is none of those."""
    seq = value if isinstance(value, (tuple, list)) else (value,)
    try:
        parts = [_part(x) for x in seq]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not 1 <= len(parts) <= 3:
        return None
    dec = sum(p / f for p, f in zip(parts, (1.0, 60.0, 3600.0)))
    if isinstance(ref, bytes):
        ref = ref.decode("ascii", "ignore")
    if str(ref or "").strip().upper()[:1] in ("S", "W"):
        dec = -dec
    return round(dec, 7)


_XMP_GPS = re.compile(rb'exif:GPS(Latitude|Longitude)(?:="([^"]+)"|>([^<]+)<)')
_XMP_DATE = re.compile(
    rb'(?:exif:DateTimeOriginal|photoshop:DateCreated|xmp:CreateDate)(?:="([^"]+)"|>([^<]+)<)')
_XMP_COORD = re.compile(
    r"\s*(\d+(?:\.\d+)?)(?:,(\d+(?:\.\d+)?))?(?:,(\d+(?:\.\d+)?))?\s*([NSEW])\s*", re.I)
_ISO_LOCAL = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})")


def _xmp_coord(text: str) -> float | None:
    """An XMP GPS value, ``DDD,MM.mmk`` or ``DDD,MM,SSk`` (k = N/S/E/W)."""
    m = _XMP_COORD.fullmatch(text)
    if not m:
        return None
    return _to_deg([float(g) for g in m.groups()[:3] if g is not None], m.group(4).upper())


def _from_xmp(path: str | Path) -> dict[str, Any]:
    """GPS and capture date from the file's XMP packet, for when EXIF has none.
    Only the start of the file is read, where an image keeps its XMP."""
    out: dict[str, Any] = {}
    try:
        with open(path, "rb") as fh:
            head = fh.read(1 << 20)
    except OSError:
        return out
    if b"<x:xmpmeta" not in head and b"<rdf:RDF" not in head:
        return out
    coords: dict[str, float | None] = {}
    for m in _XMP_GPS.finditer(head):
        val = (m.group(2) or m.group(3) or b"").decode("utf-8", "ignore")
        coords.setdefault(m.group(1).decode(), _xmp_coord(val))
    lat, lon = coords.get("Latitude"), coords.get("Longitude")
    if lat is not None and lon is not None:
        out["gps_lat"], out["gps_lon"] = lat, lon
    d = _XMP_DATE.search(head)
    if d:
        dt = _ISO_LOCAL.match((d.group(1) or d.group(2) or b"").decode("utf-8", "ignore"))
        if dt:
            out["created_dt"] = "{}-{}-{}T{}:{}:{}".format(*dt.groups())
    return out


def _parse_dt(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = str(raw).strip().replace("\x00", "")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S%z"):
        try:
            return _dt.datetime.strptime(raw, fmt).isoformat()
        except ValueError:
            continue
    return None


def extract_image(path: str | Path, img: Image.Image | None = None) -> dict[str, Any]:
    """Return width/height, capture time, GPS, camera string from an image."""
    out: dict[str, Any] = {
        "width": None, "height": None, "created_dt": None,
        "gps_lat": None, "gps_lon": None, "camera": None,
    }
    close = False
    if img is None:
        img = Image.open(path)
        close = True
    try:
        out["width"], out["height"] = img.size
        exif = img.getexif()
        if not exif:
            return _with_xmp(path, out)

        try:                       # DateTimeOriginal etc. live in the Exif sub-IFD
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        except Exception:
            exif_ifd = {}

        def tag(name: str) -> Any:
            code = _TAGS.get(name, -1)
            return exif.get(code) if code in exif else exif_ifd.get(code)

        out["created_dt"] = (
            _parse_dt(tag("DateTimeOriginal"))
            or _parse_dt(tag("DateTimeDigitized"))
            or _parse_dt(tag("DateTime"))
        )
        make, model = tag("Make"), tag("Model")
        if make or model:
            out["camera"] = " ".join(
                str(x).strip() for x in (make, model) if x
            ).strip() or None

        try:
            gps = exif.get_ifd(_GPS_IFD)
        except Exception:
            gps = {}
        if gps:
            lat = gps.get(_GPSTAGS.get("GPSLatitude", -1))
            lat_ref = gps.get(_GPSTAGS.get("GPSLatitudeRef", -1))
            lon = gps.get(_GPSTAGS.get("GPSLongitude", -1))
            lon_ref = gps.get(_GPSTAGS.get("GPSLongitudeRef", -1))
            if lat and lon:
                coords = valid_coords(_to_deg(lat, lat_ref), _to_deg(lon, lon_ref))
                if coords:
                    out["gps_lat"], out["gps_lon"] = coords
    except Exception:
        pass
    finally:
        if close:
            img.close()
    return _with_xmp(path, out)


def _with_xmp(path: str | Path, out: dict[str, Any]) -> dict[str, Any]:
    """Fill the GPS and capture date EXIF did not carry from the file's XMP
    (edited files, some apps and phones keep them only there)."""
    if out["gps_lat"] is not None and out["created_dt"] is not None:
        return out
    xmp = _from_xmp(path)
    if out["gps_lat"] is None and "gps_lat" in xmp:
        coords = valid_coords(xmp["gps_lat"], xmp["gps_lon"])
        if coords:
            out["gps_lat"], out["gps_lon"] = coords
    if out["created_dt"] is None and xmp.get("created_dt"):
        out["created_dt"] = xmp["created_dt"]
    return out


def best_created_dt(exif_dt: str | None, mtime: float = 0, ctime: float = 0) -> str | None:
    """The capture time, from EXIF/embedded metadata only.

    Filesystem timestamps (create/modify/access) are *not* a capture time and
    are kept in their own columns (``ctime`` / ``mtime`` / ``atime``); this
    returns a value only when the media itself carries one.  ``mtime``/``ctime``
    are accepted and ignored for backwards compatibility.
    """
    return exif_dt or None
