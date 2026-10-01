"""Media held inside a document: PDF, HTML, MHTML and a Safari web archive.

``gleapp/nested.py`` opens these the way it opens a zip, when the case asked for
documents to be expanded (``Source.documents``). Each reader yields
``(name, data)`` pairs; nothing is written here.

What comes out, and whether its bytes are the file that was put in:

* **PDF** - every image XObject in the file, whether a page draws it or not, and
  every embedded file (attachments, a movie or sound clip). A JPEG (DCTDecode) or
  JPEG 2000 (JPXDecode) stream *is* the image file as the producer wrote it, so it
  comes out verbatim and its hashes can match a known-hash set. Any other image is
  stored as pixels (Flate, run-length, fax, ...); it is decoded and written as a
  PNG, so its hashes are GLEAPP's, not an original file's. Those names carry
  ``_rebuilt`` so a reader can tell them apart. Embedded files are verbatim.
  An encrypted PDF that opens without a password (the usual kind: printing or
  copying restricted) is read. pypdf decrypts AES only through ``pycryptodome``
  (module ``Crypto``) or ``cryptography``, not the ``pycryptodomex`` the image
  readers use, which is why ``pycryptodome`` is a requirement too.
* **HTML** - media in ``data:`` URIs (``<img>``, ``<video>``, ``srcset``, CSS
  ``url()``), decoded from base64 or percent-encoding: verbatim. An ordinary
  ``src="photo.jpg"`` is a link, not content - the file it names sits beside the
  page and is ingested on its own.
* **MHTML** (``.mht`` / ``.mhtml``) - every image and video part of the saved
  page, plus any ``data:`` URI inside its HTML parts. Verbatim.
* **Safari ``.webarchive``** - a binary plist of the page's resources and
  subframes. Verbatim.

An image smaller than ``MIN_SIDE`` pixels on either side is dropped: documents
and web pages are full of icons, bullets and spacers.
"""

from __future__ import annotations

import base64
import binascii
import email
import email.policy
import io
import plistlib
import re
from pathlib import Path, PurePosixPath
from typing import Iterator
from urllib.parse import unquote_to_bytes, urlsplit

from PIL import Image

from .ingest import DOCUMENT_EXTS, _kind_from_magic, is_document  # noqa: F401  # pylint: disable=unused-import

# Smaller than this on either side is an icon or a spacer, not evidence.
MIN_SIDE = 32

FORMATS = ("pdf", "html", "mhtml", "webarchive")

_MIME_EXT = {
    "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/pjpeg": ".jpg",
    "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp",
    "image/bmp": ".bmp", "image/tiff": ".tif", "image/heic": ".heic",
    "image/heif": ".heif", "image/avif": ".avif", "image/x-icon": ".ico",
    "image/vnd.microsoft.icon": ".ico",
    "video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov",
    "video/ogg": ".ogv", "video/x-msvideo": ".avi", "video/3gpp": ".3gp",
}


def document_format(path: Path, head: bytes) -> str | None:
    """``pdf`` / ``html`` / ``mhtml`` / ``webarchive``, or None."""
    if b"%PDF-" in head[:1024]:
        return "pdf"
    ext = path.suffix.lower()
    if ext == ".webarchive" and head[:8] == b"bplist00":
        return "webarchive"
    if ext in (".mht", ".mhtml"):
        return "mhtml"
    if ext in (".html", ".htm", ".xhtml"):
        return "html"
    return None


def members(path: Path, fmt: str, tally: dict | None = None) -> Iterator[tuple[str, bytes]]:
    """``(name, data)`` for each piece of media in the document at ``path``.

    ``tally["too_small"]`` counts the images dropped under ``MIN_SIDE``.
    """
    tally = tally if tally is not None else {}
    if fmt == "pdf":
        it = _pdf_members(path, tally)
    elif fmt == "html":
        it = _data_uri_members(path.read_bytes())
    elif fmt == "mhtml":
        it = _mhtml_members(path.read_bytes())
    elif fmt == "webarchive":
        it = _webarchive_members(path.read_bytes())
    else:
        return
    seen: set[str] = set()
    for name, data in it:
        if not data or _too_small(data):
            if data:
                tally["too_small"] = tally.get("too_small", 0) + 1
            continue
        # one image can be used many times on a page (and a PDF name repeated)
        name = _unique(name, seen)
        yield name, data


# --------------------------------------------------------------------------
def _too_small(data: bytes) -> bool:
    if _kind_from_magic(data[:16]) != "image":
        return False
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        return False                  # let the processing pass decide what it is
    return w < MIN_SIDE or h < MIN_SIDE


def _unique(name: str, seen: set[str]) -> str:
    base, ext = name, ""
    p = PurePosixPath(name)
    if p.suffix:
        base, ext = name[: -len(p.suffix)], p.suffix
    out, i = name, 1
    while out.lower() in seen:
        i += 1
        out = f"{base}_{i}{ext}"
    seen.add(out.lower())
    return out


def _safe(name: str, limit: int = 80) -> str:
    name = re.sub(r"[^\w.\- ]+", "_", name).strip(" ._") or "item"
    return name[-limit:]


def _ext_for(data: bytes, mime: str | None = None) -> str:
    """An extension from the bytes, falling back on the declared MIME type."""
    h = data[:16]
    if h[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if h[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if h[:4] == b"GIF8":
        return ".gif"
    if h[:4] == b"RIFF" and h[8:12] == b"WEBP":
        return ".webp"
    if h[:6] == b"\x00\x00\x00\x0cjP":
        return ".jp2"
    if h[:4] == b"\xff\x4f\xff\x51":
        return ".j2k"                    # a bare codestream, not the .jp2 file format
    if h[4:8] == b"ftyp":
        return ".mp4" if _kind_from_magic(h) == "video" else ".heic"
    if h[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    return _MIME_EXT.get((mime or "").split(";")[0].strip().lower(), "")


def _url_name(url: str, i: int) -> str:
    """A member name from the URL a resource was fetched from."""
    stem = ""
    try:
        stem = PurePosixPath(urlsplit(url).path).name
    except ValueError:
        pass
    return f"{i:04d}_{_safe(stem)}" if stem else f"{i:04d}"


def _is_media_mime(mime: str) -> bool:
    mime = (mime or "").lower()
    return mime.startswith(("image/", "video/")) and "svg" not in mime


# ---------------------------------------------------------------- HTML ----
# data:<mime>[;params][;base64],<payload> - the payload runs to the closing quote,
# parenthesis or angle bracket. Base64 saved by some tools is line-wrapped.
_DATA_URI = re.compile(
    rb"data:((?:image|video)/[\w.+-]+)((?:;[\w.+-]+=?[\w.+-]*)*?)(;base64)?,"
    rb"([^\"'()<>]+)", re.I)


def _data_uri_members(html: bytes, prefix: str = "") -> Iterator[tuple[str, bytes]]:
    for i, m in enumerate(_DATA_URI.finditer(html), 1):
        mime = m.group(1).decode("ascii", "replace").lower()
        if not _is_media_mime(mime):
            continue
        payload = m.group(4)
        try:
            if m.group(3):
                data = base64.b64decode(re.sub(rb"\s+|%0[aAdD]", b"", payload) + b"==",
                                        validate=False)
            else:
                data = unquote_to_bytes(payload.strip())
        except (binascii.Error, ValueError):
            continue
        yield f"{prefix}data_uri_{i:04d}{_ext_for(data, mime)}", data


# --------------------------------------------------------------- MHTML ----
def _mhtml_members(raw: bytes) -> Iterator[tuple[str, bytes]]:
    msg = email.message_from_bytes(raw, policy=email.policy.compat32)
    for i, part in enumerate(msg.walk(), 1):
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        try:
            data = part.get_payload(decode=True) or b""
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            continue
        loc = part.get("Content-Location", "") or ""
        if _is_media_mime(ctype):
            name = _url_name(loc, i)
            if not PurePosixPath(name).suffix:
                name += _ext_for(data, ctype)
            yield name, data
        elif ctype in ("text/html", "text/css", "application/xhtml+xml"):
            yield from _data_uri_members(data, prefix=f"{i:04d}_")


# ---------------------------------------------------------- webarchive ----
def _webarchive_members(raw: bytes) -> Iterator[tuple[str, bytes]]:
    counter = [0]

    def resource(res: dict) -> Iterator[tuple[str, bytes]]:
        counter[0] += 1
        i = counter[0]
        data = res.get("WebResourceData") or b""
        mime = str(res.get("WebResourceMIMEType") or "")
        url = str(res.get("WebResourceURL") or "")
        if _is_media_mime(mime):
            name = _url_name(url, i)
            if not PurePosixPath(name).suffix:
                name += _ext_for(data, mime)
            yield name, data
        elif mime in ("text/html", "text/css", "application/xhtml+xml"):
            yield from _data_uri_members(data, prefix=f"{i:04d}_")

    def archive(arc: dict) -> Iterator[tuple[str, bytes]]:
        if isinstance(arc.get("WebMainResource"), dict):
            yield from resource(arc["WebMainResource"])
        for sub in arc.get("WebSubresources") or []:
            if isinstance(sub, dict):
                yield from resource(sub)
        for frame in arc.get("WebSubframeArchives") or []:
            if isinstance(frame, dict):
                yield from archive(frame)

    root = plistlib.loads(raw)
    if isinstance(root, dict):
        yield from archive(root)


# ----------------------------------------------------------------- PDF ----
_VERBATIM_FILTERS = {"/DCTDecode": ".jpg", "/JPXDecode": ".jp2"}


def _filters(obj) -> list[str]:
    f = obj.get("/Filter")
    if f is None:
        return []
    f = f.get_object()
    return [str(x) for x in f] if isinstance(f, list) else [str(f)]


def _pdf_members(path: Path, tally: dict) -> Iterator[tuple[str, bytes]]:
    from pypdf import PdfReader               # pylint: disable=import-outside-toplevel
    from pypdf.generic import IndirectObject, StreamObject  # noqa: I001  # pylint: disable=import-outside-toplevel

    reader = PdfReader(str(path), strict=False)
    if reader.is_encrypted:
        # most "encrypted" PDFs only restrict printing/copying and open with an
        # empty user password; one that needs a real password raises here
        reader.decrypt("")

    # the first page each image object is drawn on, for the name
    on_page: dict[int, int] = {}
    for pno, page in enumerate(reader.pages, 1):
        for idnum in _page_image_ids(page, set()):
            on_page.setdefault(idnum, pno)

    # names an embedded file was given, from the file specifications pointing at it
    names: dict[int, str] = {}
    ids = sorted({i for gen in reader.xref.values() for i in gen}
                 | set(reader.xref_objStm))
    gens = {i: g for g, d in reader.xref.items() for i in d}
    objs = []
    for idnum in ids:
        try:
            obj = reader.get_object(IndirectObject(idnum, gens.get(idnum, 0), reader))
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            continue
        objs.append((idnum, obj))
        if hasattr(obj, "get") and obj.get("/EF") is not None:
            try:
                ef = obj["/EF"].get_object()
                fname = str(obj.get("/UF") or obj.get("/F") or "")
                for key in ("/UF", "/F"):
                    ref = ef.get(key)
                    if isinstance(ref, IndirectObject) and fname:
                        names.setdefault(ref.idnum, fname)
            except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                pass

    # a soft mask is the alpha channel of another image, not a picture
    smasks = {getattr(o.get("/SMask"), "idnum", None) for _i, o in objs if hasattr(o, "get")}

    for idnum, obj in objs:
        if not isinstance(obj, StreamObject):
            continue
        subtype, otype = obj.get("/Subtype"), obj.get("/Type")
        if subtype == "/Image":
            if obj.get("/ImageMask") or idnum in smasks:
                continue
            try:
                w, h = int(obj.get("/Width", 0)), int(obj.get("/Height", 0))
            except (TypeError, ValueError):
                w = h = 0
            if w and h and (w < MIN_SIDE or h < MIN_SIDE):
                tally["too_small"] = tally.get("too_small", 0) + 1
                continue
            where = f"p{on_page[idnum]:04d}_" if idnum in on_page else "unused_"
            flt = _filters(obj)
            try:
                if len(flt) == 1 and flt[0] in _VERBATIM_FILTERS:
                    data = obj._data  # pylint: disable=protected-access
                    # a JPXDecode stream is a .jp2 file or a bare .j2k codestream
                    ext = _ext_for(data) if flt[0] == "/JPXDecode" else ""
                    yield f"{where}obj{idnum:05d}{ext or _VERBATIM_FILTERS[flt[0]]}", data
                else:
                    im = obj.decode_as_image()
                    buf = io.BytesIO()
                    im.save(buf, "PNG")
                    yield f"{where}obj{idnum:05d}_rebuilt.png", buf.getvalue()
            except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                continue
        elif otype == "/EmbeddedFile":
            try:
                data = obj.get_data()
            except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                continue
            fname = names.get(idnum, "")
            name = f"attachment_obj{idnum:05d}"
            name += f"_{_safe(fname)}" if fname else _ext_for(data)
            yield name, data


def _page_image_ids(node, seen: set[int]) -> Iterator[int]:
    """Object numbers of the images a page (or a form XObject in it) draws."""
    try:
        res = node.get("/Resources")
        res = res.get_object() if res is not None else None
        xobjs = res.get("/XObject") if res is not None else None
        xobjs = xobjs.get_object() if xobjs is not None else {}
    except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        return
    for ref in xobjs.values():
        idnum = getattr(ref, "idnum", None)
        if idnum is None or idnum in seen:
            continue
        seen.add(idnum)
        try:
            x = ref.get_object()
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            continue
        if x.get("/Subtype") == "/Image":
            yield idnum
        elif x.get("/Subtype") == "/Form":
            yield from _page_image_ids(x, seen)


# --------------------------------------------------------- describing ----
_PDF_NAME = re.compile(r"^(?:p(\d{4})|unused)_obj\d{5}(_rebuilt)?(?:_\d+)?\.\w+$", re.I)
_ATTACH_NAME = re.compile(r"^attachment_obj\d{5}(?:_(.+)|\.\w+)?$", re.I)
_DATA_URI_NAME = re.compile(r"^(?:\d{4}_)?data_uri_\d{4}(?:_\d+)?\.\w+$", re.I)
_RESOURCE_NAME = re.compile(r"^\d{4}(?:_(.+))?$")

VERBATIM = "byte for byte"
REBUILT = "rebuilt from the PDF's pixels as a PNG; its hashes are not an original file's"


def describe(member_name: str, container_ext: str) -> dict | None:
    """Where in its document an extracted item was, read back from the name this
    module gave it: ``{"where": ..., "copy": ...}``, or None when the container is
    not a document or the name is not one of ours.

    ``container_ext`` is the document's own extension ('' for a PDF an app saved
    without one, which is then recognized by the item names alone).
    """
    ext = (container_ext or "").lower()
    name = member_name.replace("\\", "/").rsplit("/", 1)[-1]
    if ext in (".pdf", ""):
        m = _PDF_NAME.match(name)
        if m:
            return {"where": f"page {int(m.group(1))}" if m.group(1)
                    else "an image no page draws",
                    "copy": REBUILT if m.group(2) else VERBATIM}
        m = _ATTACH_NAME.match(name)
        if m:
            return {"where": "attached file" + (f" ({m.group(1)})" if m.group(1) else ""),
                    "copy": VERBATIM}
        return None
    if ext not in DOCUMENT_EXTS:
        return None
    if _DATA_URI_NAME.match(name):
        return {"where": "embedded in the page (data: URI)", "copy": VERBATIM}
    if ext in (".mht", ".mhtml", ".webarchive"):
        m = _RESOURCE_NAME.match(PurePosixPath(name).stem)
        if m:
            return {"where": "a resource saved with the page"
                             + (f" ({name.split('_', 1)[1]})" if m.group(1) else ""),
                    "copy": VERBATIM}
    return None
