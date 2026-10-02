"""The export dialog has to offer every format the export route can write.

The LAVA report was the case that made this worth a test: the route grew it
before the dialog did, so the one output that carries the media and the
location maps into the viewer the LEAPP family already uses could be produced
from the command line and not from the gallery an examiner works in.
"""

import re
import time
from pathlib import Path

from PIL import Image

# Written out as literals, not read from the code, so this fails when either
# side changes alone.
EXPECTED_FORMATS = {"html", "csv", "json", "kml", "md5", "vic", "lava"}

TEMPLATE = Path(__file__).resolve().parents[1] / "gleapp/web/templates/index.html"
APP_JS = Path(__file__).resolve().parents[1] / "gleapp/web/static/app.js"


def _dialog_formats() -> set[str]:
    """The format values the export dialog's checkboxes carry."""
    html = TEMPLATE.read_text(encoding="utf-8")
    dlg = html[html.index('<div id="reportDlg">'):]
    dlg = dlg[:dlg.index("</div></div>")]
    return set(re.findall(r'class="rfmt"\s+value="([a-z0-9]+)"', dlg))


def _dialog() -> str:
    """The export dialog's own markup."""
    html = TEMPLATE.read_text(encoding="utf-8")
    dlg = html[html.index('<div id="reportDlg">'):]
    return dlg[:dlg.index("</div></div>")]


def test_the_dialog_offers_every_format_the_route_writes():
    from gleapp.web.app import REPORT_FORMATS          # pylint: disable=import-outside-toplevel
    assert _dialog_formats() == EXPECTED_FORMATS
    assert set(REPORT_FORMATS) == EXPECTED_FORMATS


def test_only_html_is_checked_by_default():
    """CSV used to be pre-checked alongside HTML; an examiner who only wanted
    the report and clicked Export got a second file they did not ask for."""
    html = TEMPLATE.read_text(encoding="utf-8")
    dlg = html[html.index('<div id="reportDlg">'):]
    dlg = dlg[:dlg.index("</div></div>")]
    checked = set(re.findall(r'class="rfmt"\s+value="([a-z0-9]+)"\s+checked', dlg))
    assert checked == {"html"}


def _wait_for_export(cl):
    """An HTML export runs as a job so the bottom bar can follow it; wait for it."""
    for _ in range(240):
        job = cl.get("/api/job").get_json()
        if not job["running"]:
            assert job["stage"] == "done", job
            return job
        time.sleep(0.25)
    raise AssertionError(f"export still running: {job}")


def _case_with_one_image(tmp_path):
    from gleapp.case import Source, open_case          # pylint: disable=import-outside-toplevel
    from gleapp.pipeline import ingest_sources, process  # pylint: disable=import-outside-toplevel
    ev = tmp_path / "ev"
    ev.mkdir()
    Image.new("RGB", (40, 30), (10, 120, 200)).save(ev / "a.jpg")
    case = open_case(tmp_path / "case", create=True, examiner="t")
    ingest_sources(case, [Source(kind="folder", path=str(ev), name="ev")])
    process(case, workers=1, keyframes=0, screen=False)
    case.close()
    return tmp_path / "case"


def test_asking_for_lava_from_the_gallery_writes_a_lava_project(tmp_path):
    """The whole point of the finding: it has to be reachable from the UI."""
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    path = _case_with_one_image(tmp_path)
    cl = create_app(None).test_client()
    assert cl.post("/api/case/open", json={"path": str(path)}).status_code == 200

    r = cl.post("/api/report", json={"format": ["lava"], "scope": "all"})
    assert r.status_code == 200
    body = r.get_json()
    # It stages media and draws maps, so it runs as a job rather than blocking
    # the request until it is done.
    assert body["job"] is True

    for _ in range(120):
        job = cl.get("/api/job").get_json()
        if not job["running"]:
            break
        time.sleep(0.25)
    assert job["stage"] == "done", job
    written = job["stats"]["written"]
    assert len(written) == 1 and Path(written[0]).name == "_lava_data.lava"
    assert Path(written[0]).is_file()
    assert (Path(written[0]).parent / "_lava_artifacts.db").is_file()


def test_the_other_formats_still_come_back_in_the_response(tmp_path):
    """LAVA and HTML are slow enough to need a job; the rest stay inline."""
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    path = _case_with_one_image(tmp_path)
    cl = create_app(None).test_client()
    cl.post("/api/case/open", json={"path": str(path)})
    body = cl.post("/api/report", json={"format": ["csv"], "scope": "all"}).get_json()
    assert "job" not in body
    assert len(body["written"]) == 1 and body["written"][0].endswith(".csv")


def test_an_html_report_runs_as_a_job_the_bottom_bar_can_follow(tmp_path):
    """An HTML report embeds every file's full-size view, which is minutes on a real
    case; inline, the gallery showed nothing until it was done."""
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    path = _case_with_one_image(tmp_path)
    cl = create_app(None).test_client()
    cl.post("/api/case/open", json={"path": str(path)})
    body = cl.post("/api/report", json={"format": ["html"], "scope": "all"}).get_json()
    assert body["job"] is True
    job = _wait_for_export(cl)
    assert job["total"] == 1 and job["done"] == 1
    assert [Path(p).name for p in job["stats"]["written"]] == ["report.html"]


def test_the_dialog_offers_the_maps_toggle_the_route_reads():
    """The same drift as the LAVA format, in an option rather than a format.

    ``/api/report`` read ``maps`` from the request body from the day the report
    drew maps, and the dialog never sent it, so the maps could only be left out
    from the command line while the README and the manual said otherwise.
    """
    dlg = _dialog()
    assert 'id="rhMaps"' in dlg, "the export dialog has no maps control"
    js = APP_JS.read_text(encoding="utf-8")
    assert 'body.maps = $("#rhMaps").checked;' in js, "the dialog does not send maps"
    assert '$("#rhMaps").checked = p.maps !== false;' in js, "the choice is not restored"


def test_unticking_maps_reaches_the_writer_and_is_remembered(tmp_path, monkeypatch):
    """Sending ``maps: false`` has to arrive at the report writer, not just be
    accepted, and come back to the dialog the way the Media checkboxes do."""
    from gleapp import report                          # pylint: disable=import-outside-toplevel
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    path = _case_with_one_image(tmp_path)
    cl = create_app(None).test_client()
    cl.post("/api/case/open", json={"path": str(path)})

    seen = {}
    real = report.export_html

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(report, "export_html", spy)
    assert cl.post("/api/report",
                   json={"format": ["html"], "scope": "all",
                         "maps": False}).status_code == 200
    _wait_for_export(cl)
    assert seen.get("maps") is False, seen
    assert cl.get("/api/report/prefs").get_json()["maps"] is False

    seen.clear()
    cl.post("/api/report", json={"format": ["html"], "scope": "all", "maps": True})
    _wait_for_export(cl)
    assert seen.get("maps") is True, seen
    assert cl.get("/api/report/prefs").get_json()["maps"] is True


def test_maps_are_drawn_when_the_request_says_nothing(tmp_path, monkeypatch):
    """A request with no ``maps`` key still gets maps, so the CLI default and the
    dialog default cannot drift apart."""
    from gleapp import report                          # pylint: disable=import-outside-toplevel
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    path = _case_with_one_image(tmp_path)
    cl = create_app(None).test_client()
    cl.post("/api/case/open", json={"path": str(path)})

    seen = {}
    real = report.export_html
    monkeypatch.setattr(report, "export_html",
                        lambda *a, **kw: (seen.update(kw), real(*a, **kw))[1])
    cl.post("/api/report", json={"format": ["html"], "scope": "all"})
    _wait_for_export(cl)
    assert seen.get("maps") is True, seen


def test_the_dialog_offers_metadata_only_and_sends_it():
    dlg = _dialog()
    assert 'id="rNoMedia"' in dlg, "the export dialog has no metadata-only control"
    js = APP_JS.read_text(encoding="utf-8")
    assert 'body.no_media = $("#rNoMedia").checked;' in js, "the dialog does not send it"
    assert '$("#rNoMedia").checked = p.no_media === true;' in js, "the choice is not restored"


def test_a_metadata_only_html_report_holds_no_picture_of_any_file(tmp_path):
    """For discovery: every field stays, and no thumbnail, full-size copy or video
    frame is embedded or written beside the report."""
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    path = _case_with_one_image(tmp_path)
    cl = create_app(None).test_client()
    cl.post("/api/case/open", json={"path": str(path)})

    # the control: an ordinary report does carry the picture
    cl.post("/api/report", json={"format": ["html"], "scope": "all"})
    job = _wait_for_export(cl)
    plain = Path(job["stats"]["written"][0])
    page = plain.read_text(encoding="utf-8")
    assert "class='rimg" in page and "data:image/jpeg;base64," in page
    assert (plain.parent / f"{plain.stem}_media").is_dir()

    cl.post("/api/report", json={"format": ["html"], "scope": "all", "no_media": True})
    job = _wait_for_export(cl)
    out = Path(job["stats"]["written"][0])
    assert out.name == "report_metadata.html", "it must not overwrite the report with pictures"
    page = out.read_text(encoding="utf-8")
    assert "class='rimg" not in page and "class='thumbwrap'" not in page
    assert "data:image/jpeg;base64," not in page
    assert "id='btnBlur'" not in page
    assert not (out.parent / f"{out.stem}_media").exists()
    assert "Not included: metadata only" in page
    assert "<details class='meta' open><summary>a.jpg</summary>" in page
    assert cl.get("/api/report/prefs").get_json()["no_media"] is True


def test_a_metadata_only_kmz_carries_no_thumbnail(tmp_path):
    import zipfile                                     # pylint: disable=import-outside-toplevel
    from gleapp import report                          # pylint: disable=import-outside-toplevel
    from gleapp.case import open_case                  # pylint: disable=import-outside-toplevel
    case = open_case(_case_with_one_image(tmp_path))
    with case.db.lock:
        case.db.conn.execute("UPDATE files SET gps_lat=28.5, gps_lon=-81.4")
        case.db.conn.commit()
    with_pic = report.export_kml(case, tmp_path / "a.kmz")
    without = report.export_kml(case, tmp_path / "b.kmz", thumbs=False)
    case.close()
    with zipfile.ZipFile(with_pic) as z:
        assert any(n.endswith(".jpg") for n in z.namelist())
    with zipfile.ZipFile(without) as z:
        assert z.namelist() == ["doc.kml"]
        kml = z.read("doc.kml").decode("utf-8")
    assert "<img" not in kml and "<Placemark>" in kml


def test_lava_is_refused_when_the_export_is_metadata_only(tmp_path):
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    path = _case_with_one_image(tmp_path)
    cl = create_app(None).test_client()
    cl.post("/api/case/open", json={"path": str(path)})
    r = cl.post("/api/report", json={"format": ["html", "lava"], "scope": "all",
                                     "no_media": True})
    assert r.status_code == 400
    assert not cl.get("/api/job").get_json()["running"]


def test_a_report_without_media_removes_the_folder_an_earlier_export_left(tmp_path):
    """Written over a report that had a media folder, a metadata-only report must
    not end up sitting beside that folder of pictures."""
    from gleapp import report                          # pylint: disable=import-outside-toplevel
    from gleapp.case import open_case                  # pylint: disable=import-outside-toplevel
    case = open_case(_case_with_one_image(tmp_path))
    dest = tmp_path / "out" / "r.html"
    dest.parent.mkdir()
    report.export_html(case, dest, maps=False)
    assert any((tmp_path / "out" / "r_media").rglob("*.*"))
    report.export_html(case, dest, maps=False, thumbs=False)
    case.close()
    assert not (tmp_path / "out" / "r_media").exists()
