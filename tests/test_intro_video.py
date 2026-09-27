"""The intro video: it opens with the launcher until the examiner ticks "Don't show
this at startup", and Watch intro in either menu plays it again. The choice is
per-user config (appconfig ``show_intro``), never part of a case.
"""

import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "gleapp/web/templates/index.html"
APP_JS = ROOT / "gleapp/web/static/app.js"
VIDEO = ROOT / "gleapp/web/static/intro/gleapp-intro.mp4"
POSTER = ROOT / "gleapp/web/static/intro/gleapp-intro-poster.jpg"


@pytest.fixture(name="client")
def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GLEAPP_CONFIG_DIR", str(tmp_path / "cfg"))
    from gleapp.web.app import create_app  # pylint: disable=import-outside-toplevel
    return create_app(None).test_client()


def test_the_intro_shows_until_it_is_switched_off(client):
    assert client.get("/api/context").get_json()["show_intro"] is True

    r = client.post("/api/settings", json={"show_intro": False})
    assert r.status_code == 200 and r.get_json()["show_intro"] is False
    assert client.get("/api/context").get_json()["show_intro"] is False

    r = client.post("/api/settings", json={"show_intro": True})
    assert r.status_code == 200 and r.get_json()["show_intro"] is True
    assert client.get("/api/context").get_json()["show_intro"] is True


def test_the_choice_is_saved_in_the_user_config_not_a_case(client, tmp_path):
    from gleapp import appconfig  # pylint: disable=import-outside-toplevel

    client.post("/api/settings", json={"show_intro": False})
    assert appconfig.load()["show_intro"] is False
    assert (tmp_path / "cfg" / "config.json").is_file()
    # switching it back on removes the key, so the default stays the default
    client.post("/api/settings", json={"show_intro": True})
    assert "show_intro" not in appconfig.load()


@pytest.mark.parametrize("value", ["false", 0, None, "no"])
def test_only_a_boolean_is_accepted(client, value):
    r = client.post("/api/settings", json={"show_intro": value})
    assert r.status_code == 400
    assert client.get("/api/context").get_json()["show_intro"] is True


def test_the_page_has_the_player_the_box_and_both_menu_entries():
    html = TEMPLATE.read_text(encoding="utf-8")
    assert 'id="introDlg"' in html and 'id="introVideo"' in html
    assert 'id="introHide"' in html and "Don't show this at startup" in html
    assert 'id="introClose"' in html
    assert 'id="btnIntroLauncher"' in html     # launcher ☰ Menu
    assert 'id="btnIntro"' in html             # case ☰ Menu, under Help
    assert 'poster="/static/intro/gleapp-intro-poster.jpg"' in html


def test_the_script_points_at_the_shipped_file_and_opens_it_only_when_asked():
    js = APP_JS.read_text(encoding="utf-8")
    assert '"/static/intro/gleapp-intro.mp4"' in js
    assert "if (Intro.show) openIntro(true);" in js
    assert VIDEO.is_file() and POSTER.is_file()


def _top_level_boxes(data: bytes) -> list[str]:
    boxes, pos = [], 0
    while pos + 8 <= len(data):
        size, kind = struct.unpack(">I4s", data[pos:pos + 8])
        if size == 1:
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
        if size < 8:
            break
        boxes.append(kind.decode("latin-1"))
        pos += size
    return boxes


def test_the_video_starts_playing_before_it_has_all_loaded_and_stays_small():
    data = VIDEO.read_bytes()
    boxes = _top_level_boxes(data)
    assert boxes[0] == "ftyp"
    # moov ahead of mdat is what lets the player start on the first bytes it gets
    assert boxes.index("moov") < boxes.index("mdat")
    # the 1080p60 master is 21 MB; the shipped copy is re-encoded for the window
    assert len(data) < 10 * 1024 * 1024
