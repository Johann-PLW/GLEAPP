"""Media inside a PDF, an HTML page, an MHTML file and a Safari web archive.

Documents are kept as containers only when the ingest asks for it
(``expand_documents`` / ``"documents": true``); then ``gleapp/nested.py`` opens them
with ``gleapp/docmedia.py`` and registers what is inside, linked by ``container_id``.
"""

import base64
import hashlib
import io
import plistlib
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
from PIL import Image

from gleapp import docmedia, nested
from gleapp.case import Source, open_case
from gleapp.pipeline import ingest_sources, process

pypdf = pytest.importorskip("pypdf")


def _img(fmt, size=(200, 120), mode="RGB", colour=(200, 30, 30)) -> bytes:
    b = io.BytesIO()
    Image.new(mode, size, colour if mode == "RGB" else 128).save(b, fmt)
    return b.getvalue()


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _pdf(path: Path, *, attach: bytes | None = None) -> None:
    """Page 1 a JPEG (DCTDecode), page 2 a palette image (Flate), page 3 an icon."""
    pal = Image.new("P", (64, 48))
    pal.putpalette([i % 256 for i in range(768)])
    first = Image.open(io.BytesIO(_img("JPEG")))
    first.save(path, "PDF", save_all=True, append_images=[pal, Image.new("RGB", (10, 10))])
    if attach is not None:
        w = pypdf.PdfWriter(clone_from=str(path))
        w.add_attachment("clip.mp4", attach)
        w.write(path)


def _html(jpg: bytes, icon: bytes, png: bytes) -> str:
    b64 = lambda d: base64.b64encode(d).decode()  # noqa: E731
    return (f'<img src="data:image/jpeg;base64,{b64(jpg)}">'
            f'<img src="data:image/png;base64,{b64(icon)}">'
            f'<div style="background:url(data:image/png;base64,{b64(png)})"></div>'
            f'<img src="elsewhere.jpg">')


_MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\0" * 64


def _members(path: Path):
    fmt = docmedia.document_format(path, path.read_bytes()[:1024])
    tally: dict = {}
    return fmt, dict(docmedia.members(path, fmt, tally)), tally


# ---- the readers -------------------------------------------------------

def test_pdf_jpeg_is_verbatim_other_images_are_rebuilt(tmp_path):
    jpg = _img("JPEG")
    p = tmp_path / "doc.pdf"
    _pdf(p, attach=_MP4)
    fmt, got, tally = _members(p)
    assert fmt == "pdf"
    jpgs = [n for n in got if n.endswith(".jpg")]
    rebuilt = [n for n in got if n.endswith("_rebuilt.png")]
    assert len(jpgs) == 1 and jpgs[0].startswith("p0001_")
    # Pillow writes a JPEG page as the JPEG it was given
    assert _md5(got[jpgs[0]]) == _md5(jpg)
    assert len(rebuilt) == 1 and rebuilt[0].startswith("p0002_")
    assert Image.open(io.BytesIO(got[rebuilt[0]])).size == (64, 48)
    att = [n for n in got if n.startswith("attachment_")]
    assert att and att[0].endswith("clip.mp4") and got[att[0]] == _MP4
    assert tally.get("too_small") == 1, "the 10x10 page image is an icon"


def test_pdf_without_an_extension_is_recognized_by_magic(tmp_path):
    p = tmp_path / "blob"
    _pdf(p)
    assert docmedia.document_format(p, p.read_bytes()[:1024]) == "pdf"


def test_html_data_uris_are_verbatim_and_links_are_ignored(tmp_path):
    jpg, icon, png = _img("JPEG"), _img("PNG", (10, 10)), _img("PNG", (90, 90), "L")
    p = tmp_path / "page.html"
    p.write_text(_html(jpg, icon, png))
    fmt, got, tally = _members(p)
    assert fmt == "html"
    assert sorted(_md5(d) for d in got.values()) == sorted([_md5(jpg), _md5(png)])
    assert tally == {"too_small": 1}


def test_mhtml_parts_and_inline_data_uris(tmp_path):
    jpg, png = _img("JPEG"), _img("PNG", (90, 90), "L")
    html = _html(jpg, _img("PNG", (8, 8)), png)
    body = ("MIME-Version: 1.0\r\nContent-Type: multipart/related; boundary=\"B\"\r\n\r\n"
            "--B\r\nContent-Type: text/html\r\nContent-Location: https://ex.test/\r\n\r\n"
            f"{html}\r\n--B\r\nContent-Type: image/jpeg\r\n"
            "Content-Transfer-Encoding: base64\r\n"
            "Content-Location: https://ex.test/img/photo.jpg\r\n\r\n"
            f"{base64.encodebytes(jpg).decode()}\r\n--B--\r\n")
    p = tmp_path / "saved.mhtml"
    p.write_bytes(body.encode())
    fmt, got, _ = _members(p)
    assert fmt == "mhtml"
    assert any(n.endswith("_photo.jpg") for n in got), "named from Content-Location"
    assert sorted(_md5(d) for d in got.values()) == sorted([_md5(jpg), _md5(png), _md5(jpg)])


def test_webarchive_resources_and_subframes(tmp_path):
    jpg, png = _img("JPEG"), _img("PNG", (90, 90), "L")
    arc = {
        "WebMainResource": {"WebResourceData": b"<p>hi</p>",
                            "WebResourceMIMEType": "text/html",
                            "WebResourceURL": "https://ex.test/"},
        "WebSubresources": [{"WebResourceData": png, "WebResourceMIMEType": "image/png",
                             "WebResourceURL": "https://ex.test/a/b.png"}],
        "WebSubframeArchives": [{"WebMainResource": {
            "WebResourceData": jpg, "WebResourceMIMEType": "image/jpeg",
            "WebResourceURL": "https://ex.test/frame.jpg"}}],
    }
    p = tmp_path / "page.webarchive"
    p.write_bytes(plistlib.dumps(arc, fmt=plistlib.FMT_BINARY))  # pylint: disable=no-member  # set by an enum
    fmt, got, _ = _members(p)
    assert fmt == "webarchive"
    assert sorted(_md5(d) for d in got.values()) == sorted([_md5(jpg), _md5(png)])


def test_a_tar_whose_first_member_is_a_pdf_is_still_a_tar(tmp_path):
    pdf = tmp_path / "a.pdf"
    _pdf(pdf)
    t = tmp_path / "bundle.tar"
    with tarfile.open(t, "w") as tf:
        tf.add(pdf, arcname="a.pdf")
    assert nested._looks_like_container(t) == "tar"  # pylint: disable=protected-access


# ---- through the ingest ------------------------------------------------

def _build(ev: Path) -> None:
    _pdf(ev / "report.pdf", attach=_MP4)
    (ev / "page.html").write_text(_html(_img("JPEG"), _img("PNG", (10, 10)),
                                        _img("PNG", (90, 90), "L")))
    (ev / "notes.txt").write_text("not media")
    pdf2 = ev / "_tmp.pdf"
    _pdf(pdf2)
    with zipfile.ZipFile(ev / "mail.zip", "w") as zf:
        zf.write(pdf2, "attachments/invoice.pdf")
    pdf2.unlink()


def _ingest(tmp_path, **kw):
    ev = tmp_path / "ev"
    ev.mkdir()
    _build(ev)
    case = open_case(tmp_path / "case", create=True, examiner="t")
    ingest_sources(case, [Source(kind="folder", path=str(ev), name="ev")], **kw)
    return case, {r["rel_path"]: r for r in case.db.iter_files()}


def test_documents_are_not_kept_unless_asked(tmp_path):
    _case, rows = _ingest(tmp_path)
    assert "report.pdf" not in rows and "page.html" not in rows
    assert "mail.zip/attachments/invoice.pdf" not in rows


def test_documents_are_opened_when_asked(tmp_path):
    case, rows = _ingest(tmp_path, expand_documents=True)
    for name in ("report.pdf", "page.html", "mail.zip/attachments/invoice.pdf"):
        assert rows[name]["kind"] == "archive", name
    kids = lambda name: [r for r in rows.values()  # noqa: E731
                         if r["container_id"] == rows[name]["id"]]
    assert {r["kind"] for r in kids("report.pdf")} == {"image", "video"}
    assert len(kids("report.pdf")) == 3            # jpeg, rebuilt png, attachment
    assert len(kids("page.html")) == 2
    # the PDF inside the zip was opened too, since the case asked for documents
    assert len(kids("mail.zip/attachments/invoice.pdf")) == 2
    assert case.db.get_meta(nested.DOCUMENTS_META) == "1"

    process(case)
    after = {r["rel_path"]: r for r in case.db.iter_files()}
    pdf_kids = [r for r in after.values() if r["container_id"] == rows["report.pdf"]["id"]]
    imgs = [r for r in pdf_kids if r["kind"] == "image"]
    assert all(r["md5"] and r["thumb"] and not r["error"] for r in imgs)


def test_documents_alone_leave_other_archives_for_later(tmp_path):
    case, rows = _ingest(tmp_path, expand_archives=False, expand_documents=True)
    assert any(r["container_id"] == rows["report.pdf"]["id"] for r in rows.values())
    assert not any(r["container_id"] == rows["mail.zip"]["id"] for r in rows.values())
    # a later Expand archives opens the zip, and the PDF in it, from the case meta
    nested.expand_containers(case)
    rows = {r["rel_path"]: r for r in case.db.iter_files()}
    inv = rows["mail.zip/attachments/invoice.pdf"]
    assert inv["kind"] == "archive"
    assert any(r["container_id"] == inv["id"] for r in rows.values())


# ---- encrypted PDFs and JPEG 2000 --------------------------------------

@pytest.mark.parametrize("algorithm", ["RC4-128", "AES-128", "AES-256"])
def test_a_pdf_restricted_but_openable_is_read(tmp_path, algorithm):
    pytest.importorskip("Crypto", reason="pypdf's AES needs pycryptodome")
    plain = tmp_path / "plain.pdf"
    _pdf(plain)
    w = pypdf.PdfWriter(clone_from=str(plain))
    w.encrypt(user_password="", owner_password="owner", algorithm=algorithm)
    enc = tmp_path / "enc.pdf"
    w.write(enc)
    _, want, _ = _members(plain)
    _, got, _ = _members(enc)
    # decrypted: the same bytes as in the unencrypted copy
    assert sorted(_md5(d) for d in got.values()) == sorted(_md5(d) for d in want.values())


def test_jpeg_2000_is_an_image():
    from gleapp.ingest import _kind_from_magic, classify
    jp2 = _img("JPEG2000")
    assert classify(".jp2") == classify(".j2k") == "image"
    assert _kind_from_magic(jp2[:16]) == "image"
    b = io.BytesIO()
    Image.new("RGB", (64, 64)).save(b, "JPEG2000", no_jp2=True)
    assert _kind_from_magic(b.getvalue()[:16]) == "image", "bare codestream"


def test_a_jpeg_2000_image_in_a_pdf_is_extracted_and_processed(tmp_path):
    ev = tmp_path / "ev"
    ev.mkdir()
    Image.new("RGBA", (80, 60), (0, 255, 0, 128)).save(ev / "j.pdf", "PDF")  # Pillow writes RGBA as JPXDecode
    case = open_case(tmp_path / "case", create=True, examiner="t")
    ingest_sources(case, [Source(kind="folder", path=str(ev), name="ev")],
                   expand_documents=True)
    process(case)
    kids = [r for r in case.db.iter_files() if r["container_id"]]
    assert [r["ext"] for r in kids] == [".jp2"]
    assert kids[0]["kind"] == "image" and kids[0]["thumb"] and not kids[0]["error"]
    assert (kids[0]["width"], kids[0]["height"]) == (80, 60)


# ---- linking an extracted item back to its document (details pane) -------

def test_describe_reads_back_where_an_item_was():
    d = docmedia.describe
    assert d("p0002_obj00002_rebuilt.png", ".pdf") == {"where": "page 2", "copy": docmedia.REBUILT}
    assert d("p0001_obj00001.jpg", "") == {"where": "page 1", "copy": docmedia.VERBATIM}
    assert d("unused_obj00010.jpg", ".pdf")["where"] == "an image no page draws"
    assert d("attachment_obj00002_clip_0001.mp4", ".pdf")["where"] == "attached file (clip_0001.mp4)"
    assert d("data_uri_0003.webp", ".html")["where"].startswith("embedded in the page")
    assert d("0003_photo1.jpg", ".mhtml")["where"] == "a resource saved with the page (photo1.jpg)"
    # names that are not ours, or a container that is not a document
    assert d("IMG_0001.jpg", ".pdf") is None
    assert d("p0001_obj00001.jpg", ".zip") is None


def test_the_details_pane_names_the_document_and_filters_to_it(tmp_path):
    from gleapp.web.app import create_app
    case, rows = _ingest(tmp_path, expand_documents=True)
    process(case, workers=1, keyframes=0, screen=False)
    case.close()
    cl = create_app(None).test_client()
    cl.post("/api/case/open", json={"path": str(tmp_path / "case")})

    pdf = rows["report.pdf"]
    kid = next(r for r in rows.values()
               if r["container_id"] == pdf["id"] and r["rel_path"].endswith("_rebuilt.png"))
    d = cl.get(f"/api/file/{kid['id']}").get_json()
    assert d["container"]["id"] == pdf["id"] and d["container"]["is_document"]
    assert d["container"]["where"] == "page 2"
    assert d["container"]["copy"] == docmedia.REBUILT

    c = cl.get(f"/api/file/{pdf['id']}").get_json()
    assert c["members"]["media"] == 3 and c["members"]["is_document"]
    z = cl.get(f"/api/file/{rows['mail.zip']['id']}").get_json()
    assert z["members"] == {"media": 0, "containers": 1, "is_document": False, "is_cache": False}

    # everything from one document, whatever else is set
    got = cl.get(f"/api/files?container={pdf['id']}&q=nothing-matches").get_json()
    assert got["total"] == 3
    assert all(r["rel_path"].replace("\\", "/").startswith("report.pdf/") for r in got["files"])


# ---- a case ingested without documents, opened later -----------------------

@pytest.mark.parametrize("as_zip", [False, True], ids=["folder", "zip"])
def test_documents_can_be_added_to_a_case_after_ingest(tmp_path, as_zip):
    from gleapp.case import parse_source_spec
    from gleapp.pipeline import add_documents
    ev = tmp_path / "ev"
    ev.mkdir()
    _build(ev)
    if as_zip:
        z = tmp_path / "extraction.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for p in ev.rglob("*"):
                if p.is_file():
                    zf.write(p, "Dump/" + p.relative_to(ev).as_posix())
        sources = parse_source_spec(str(z))[0]
    else:
        sources = [Source(kind="folder", path=str(ev), name="ev")]
    case = open_case(tmp_path / "case", create=True, examiner="t")
    ingest_sources(case, sources, expand_archives=False)
    before = {r["id"]: r["rel_path"] for r in case.db.iter_files()}
    assert not any(r.endswith((".pdf", ".html")) for r in before.values())

    got = add_documents(case)
    assert got["documents"] == 2 and got["unavailable"] == []
    assert got["added"] == 5                   # 3 from the PDF, 2 from the page
    after = {r["id"]: r["rel_path"] for r in case.db.iter_files()}
    assert {k: after[k] for k in before} == before, "nothing already there moved"
    # the zip in the evidence waits for Expand archives; it then opens its PDF too
    assert add_documents(case) == {"documents": 0, "added": 0, "unavailable": []}
    case.close()


def test_a_source_that_has_gone_is_named_not_fatal(tmp_path):
    from gleapp.pipeline import add_documents
    case, _rows = _ingest(tmp_path)
    shutil.rmtree(tmp_path / "ev")
    assert add_documents(case) == {"documents": 0, "added": 0, "unavailable": ["ev"]}


def test_the_sidebar_tells_archives_documents_and_cache_pieces_apart(tmp_path):
    from gleapp.web.app import create_app
    case, _rows = _ingest(tmp_path, expand_documents=True)
    case.close()
    cl = create_app(None).test_client()
    cl.post("/api/case/open", json={"path": str(tmp_path / "case")})
    c = cl.get("/api/context").get_json()["containers"]
    # report.pdf, page.html, and the invoice.pdf inside mail.zip; mail.zip itself
    assert c["documents"] == 3 and c["archives"] == 1
    assert c["document_items"] == 7                   # 3 + 2 + 2, pictures and videos only
    assert c["archive_items"] == 0                    # mail.zip holds only a document
    total = lambda q: cl.get(f"/api/files?limit=1&{q}").get_json()["total"]  # noqa: E731
    assert total("kind=archive:document") == 3
    assert total("kind=archive:archive") == 1
    assert total("kind=archive:cache") == 0
    assert total("in_document=1") == 7
    assert total("in_archive=1") == 0
    assert total("kind=archive") == 4                 # every container, as before


def test_the_counts_are_ready_before_processing_ends(tmp_path):
    """Extract media from documents marks the job as soon as the documents are in the
    case, so the page can refresh its counts while processing and hash matching run."""
    import time
    from gleapp.web.app import create_app
    _case, _rows = _ingest(tmp_path)               # without documents
    _case.close()
    cl = create_app(None).test_client()
    cl.post("/api/case/open", json={"path": str(tmp_path / "case")})
    assert cl.get("/api/context").get_json()["containers"]["documents"] == 0
    assert cl.post("/api/extract-documents", json={}).status_code == 200
    seen_running = None
    for _ in range(200):
        j = cl.get("/api/job").get_json()
        if j.get("extracted") and j.get("running") and seen_running is None:
            seen_running = cl.get("/api/context").get_json()["containers"]
        if not j.get("running"):
            break
        time.sleep(0.1)
    assert j["stage"] == "done", j
    assert j["extracted"] == {"documents": 2, "added": 5}
    if seen_running is not None:                   # caught while processing ran on
        assert seen_running["documents"] == 2 and seen_running["document_items"] == 5
    c = cl.get("/api/context").get_json()["containers"]
    assert c["documents"] == 2 and c["document_items"] == 5
