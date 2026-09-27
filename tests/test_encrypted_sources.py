"""An encrypted Apple disk image as a source, through the web API and the command line.

The password is asked for when the source is added (the ingest request answers with
the images it cannot open yet, and the next request carries their passwords in its
body), held in memory for the session, and asked for again in a new session through
the source's Unlock control. It is never written into the case.
"""

import io
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ewfwriter.py sits beside this file; pytest puts that directory on the path.
from ewfwriter import write_encrypted  # pylint: disable=import-error
from gleapp import archive, cli
from gleapp.case import open_case

PASSWORD = "gleapp-web-test-password"


@pytest.fixture(autouse=True)
def _isolate(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("GLEAPP_CONFIG_DIR", str(tmp_path_factory.mktemp("gleapp-cfg")))
    monkeypatch.setattr(archive, "_PASSWORDS", {})      # each test is a new session
    from gleapp import hashstore, stash  # pylint: disable=import-outside-toplevel
    hashstore.close()
    stash.close()
    yield
    archive.close_zips()
    hashstore.close()
    stash.close()


def _jpg(seed):
    rnd = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rnd.integers(0, 256, (72, 96, 3), dtype="uint8")).save(buf, "JPEG")
    return buf.getvalue()


def _image(tmp_path) -> Path:
    """An encrypted image of a disk with two photos on block boundaries, no filesystem."""
    out = bytearray(4096)
    for seed in (1, 2):
        out += _jpg(seed)
        out += bytes(-len(out) % 4096)
    ev = tmp_path / "ev"
    ev.mkdir()
    return Path(write_encrypted(ev / "locked.dmg", bytes(out), PASSWORD))


def _client():
    from gleapp.web.app import create_app  # pylint: disable=import-outside-toplevel
    return create_app(None).test_client()


def _wait(cl, timeout=60):
    for _ in range(timeout * 4):
        job = cl.get("/api/job").get_json()
        if not job["running"]:
            return job
        time.sleep(0.25)
    raise AssertionError("job never finished")


_OPTS = {"screen": False, "keyframes": 0, "carve": True}


def test_the_web_ingest_asks_for_the_password_and_then_reads_the_image(tmp_path):
    image = _image(tmp_path)
    cl = _client()
    assert cl.post("/api/case/create", json={"path": str(tmp_path / "c"), "name": "c",
                                             "examiner": "t"}).status_code == 200
    body = {"sources": [{"path": str(image)}], "options": _OPTS}
    r = cl.post("/api/case/ingest", json=body)
    assert r.status_code == 409
    need = r.get_json()["needs_password"]
    assert [(n["name"], n["wrong"]) for n in need] == [("locked.dmg", False)]
    assert cl.get("/api/job").get_json()["running"] is False, "nothing started"

    r = cl.post("/api/case/ingest", json={**body, "passwords": {need[0]["path"]: "not it"}})
    assert r.status_code == 409 and r.get_json()["needs_password"][0]["wrong"] is True

    r = cl.post("/api/case/ingest", json={**body, "passwords": {need[0]["path"]: PASSWORD}})
    assert r.status_code == 200, r.get_json()
    job = _wait(cl)
    assert job["stage"] == "done", job
    status = cl.get("/api/sources").get_json()
    assert [(s["name"], s["status"], s["format"]) for s in status] == [("locked.dmg", "ok", "ewf")]
    cl.post("/api/case/close")
    case_files = [p for p in (tmp_path / "c").rglob("*") if p.is_file()]
    assert case_files, "the case folder is empty, so the check below proves nothing"
    # The settings folder receives nothing in this flow; it is read anyway so that a
    # later change writing there is covered too.
    settings = [p for p in Path(os.environ["GLEAPP_CONFIG_DIR"]).rglob("*") if p.is_file()]
    for p in case_files + settings:
        assert PASSWORD.encode() not in p.read_bytes(), f"the password reached {p.name}"


def test_a_locked_source_is_unlocked_through_the_api(tmp_path, monkeypatch):
    image = _image(tmp_path)
    assert archive.unlock_image(image, PASSWORD)
    cl = _client()
    cl.post("/api/case/create", json={"path": str(tmp_path / "c"), "name": "c",
                                      "examiner": "t"})
    assert cl.post("/api/case/ingest", json={"sources": [{"path": str(image)}],
                                             "options": _OPTS}).status_code == 200
    assert _wait(cl)["stage"] == "done"
    archive.close_zips()
    monkeypatch.setattr(archive, "_PASSWORDS", {})      # the app was restarted
    assert [s["status"] for s in cl.get("/api/sources").get_json()] == ["locked"]
    r = cl.post("/api/source/unlock", json={"name": "locked.dmg", "password": "not it"})
    assert r.get_json() == {"ok": False, "wrong": True}
    r = cl.post("/api/source/unlock", json={"name": "locked.dmg", "password": PASSWORD})
    assert r.get_json() == {"ok": True, "wrong": False}
    assert [s["status"] for s in cl.get("/api/sources").get_json()] == ["ok"]
    assert cl.post("/api/source/unlock", json={"name": "nope", "password": "x"}).status_code == 404


def test_the_command_line_takes_the_password_from_a_variable_or_refuses(tmp_path,
                                                                        monkeypatch, capsys):
    image = _image(tmp_path)
    case = str(tmp_path / "c")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))            # not a terminal
    assert cli.main(["-c", case, "ingest", str(image), "--no-process"]) == 2
    assert "--password-file or --password-env" in capsys.readouterr().err
    monkeypatch.setenv("GLEAPP_TEST_PW", "not it")
    assert cli.main(["-c", case, "--password-env", "GLEAPP_TEST_PW", "ingest", str(image),
                     "--no-process"]) == 2
    monkeypatch.setenv("GLEAPP_TEST_PW", PASSWORD)
    pw_file = tmp_path / "pw.txt"
    pw_file.write_bytes(b"not it\n")
    assert cli.main(["-c", case, "--password-file", str(pw_file), "--password-env",
                     "GLEAPP_TEST_PW", "ingest", str(image), "--no-process"]) == 0
    c = open_case(case)
    try:
        assert [(s["name"], s["status"]) for s in archive.source_status(c)] == [
            ("locked.dmg", "ok")]
    finally:
        c.close()
    out = capsys.readouterr()
    assert PASSWORD not in out.out + out.err
