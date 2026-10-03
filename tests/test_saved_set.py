"""Saved set: the examiner's own categories and (if they choose) flags, carried
from case to case through ``config_dir()/categories.json`` (savedset.py)."""

from __future__ import annotations

import json
from pathlib import Path

from gleapp import appconfig, savedset
from gleapp.case import open_case

TEMPLATE = Path(__file__).resolve().parents[1] / "gleapp/web/templates/index.html"


def _case(tmp_path, name):
    return open_case(tmp_path / name, create=True, examiner="t")


def _custom(db):
    return [(r["code"], r["name"]) for r in db.list_categories()
            if not r["locked"] and r["active"]]


def test_the_file_is_in_the_config_dir():
    assert savedset._path().parent == appconfig.config_dir()  # pylint: disable=protected-access


def test_nothing_saved_reads_as_empty():
    savedset._path().unlink(missing_ok=True)  # pylint: disable=protected-access
    assert savedset.load() == {"categories": [], "flags": []}


def test_save_keeps_only_the_examiners_own_named_shown_categories(tmp_path):
    c = _case(tmp_path, "a")
    try:
        db = c.db
        w = db.add_category("Weapons")
        db.add_category("")                      # unnamed - nothing to match on
        hidden = db.add_category("Old")
        db.update_category(hidden, active=0)
        db.add_flag("Evidence")
        db.add_flag(" ")
        counts = savedset.save_from_case(db, categories=True, flags=False)
        saved = savedset.load()
        assert counts == {"categories": 1, "flags": len(saved["flags"])}
        assert [(x["code"], x["name"]) for x in saved["categories"]] == [(w, "Weapons")]
        assert all(x["code"] >= 6 for x in saved["categories"])
    finally:
        c.close()


def test_saving_one_part_leaves_the_other_as_it_was(tmp_path):
    c = _case(tmp_path, "a")
    try:
        c.db.add_category("Weapons")
        c.db.add_flag("Evidence")
        savedset.save_from_case(c.db, categories=True, flags=True)
        c.db.add_flag("Follow up")
        c.db.add_category("Drugs")
        savedset.save_from_case(c.db, categories=True, flags=False)
        saved = savedset.load()
        assert [f["name"] for f in saved["flags"]] == ["Evidence"]
        assert [x["name"] for x in saved["categories"]] == ["Weapons", "Drugs"]
    finally:
        c.close()


def test_a_new_case_gets_the_same_codes_colors_and_order(tmp_path):
    a = _case(tmp_path, "a")
    try:
        a.db.add_category("Weapons")
        d = a.db.add_category("Drugs")
        a.db.update_category(d, color="#123456", notable=0)
        a.db.add_category("Cash")
        a.db.reorder_categories([d, 6, 8])
        savedset.save_from_case(a.db, categories=True, flags=False)
        want = _custom(a.db)
    finally:
        a.close()
    b = _case(tmp_path, "b")
    try:
        res = savedset.apply_to_case(b.db, categories=True, flags=False)
        assert {r["status"] for r in res["categories"]} == {"added"}
        assert res["flags"] == []
        assert _custom(b.db) == want
        row = b.db.get_category(d)
        assert row["color"] == "#123456" and row["notable"] == 0
    finally:
        b.close()


def test_merging_skips_names_and_renumbers_a_taken_code(tmp_path):
    a = _case(tmp_path, "a")
    try:
        a.db.add_category("Weapons")    # 6
        a.db.add_category("Drugs")      # 7
        a.db.add_flag("Evidence")
        a.db.add_flag("Follow up")
        savedset.save_from_case(a.db, categories=True, flags=True)
    finally:
        a.close()
    b = _case(tmp_path, "b")
    try:
        b.db.add_category("weapons")    # 6, same name, other case
        b.db.add_category("Documents")  # 7 taken by something else
        b.db.add_flag("EVIDENCE")
        res = savedset.apply_to_case(b.db, categories=True, flags=True)
        by = {r["name"]: r for r in res["categories"]}
        assert by["Weapons"]["status"] == "exists"
        assert by["Drugs"]["status"] == "renumbered"
        assert by["Drugs"]["saved_code"] == 7 and by["Drugs"]["code"] == 8
        assert b.db.category_name(8) == "Drugs"
        assert b.db.category_name(7) == "Documents"
        assert [(f["name"], f["status"]) for f in res["flags"]] == [
            ("Evidence", "exists"), ("Follow up", "added")]
        # applying again changes nothing
        again = savedset.apply_to_case(b.db, categories=True, flags=True)
        assert {r["status"] for r in again["categories"]} == {"exists"}
        assert {f["status"] for f in again["flags"]} == {"exists"}
    finally:
        b.close()


def test_a_hidden_category_of_the_same_name_is_shown_again(tmp_path):
    a = _case(tmp_path, "a")
    try:
        a.db.add_category("Weapons")
        savedset.save_from_case(a.db, categories=True, flags=False)
    finally:
        a.close()
    b = _case(tmp_path, "b")
    try:
        code = b.db.add_category("Weapons")
        b.db.update_category(code, active=0)
        res = savedset.apply_to_case(b.db, categories=True, flags=False)
        assert res["categories"][0]["status"] == "shown"
        assert b.db.get_category(code)["active"] == 1
    finally:
        b.close()


def test_a_saved_code_in_the_preset_range_is_never_used(tmp_path):
    """An old case may hold an examiner category on a preset slot (0-5)."""
    savedset._write({"categories": [  # pylint: disable=protected-access
        {"code": 3, "name": "Weapons", "color": "#000000", "notable": True}]})
    b = _case(tmp_path, "b")
    try:
        res = savedset.apply_to_case(b.db, categories=True, flags=False)
        assert res["categories"][0]["code"] >= 6
        assert b.db.get_category(3)["locked"] == 1
    finally:
        b.close()


def test_flags_are_left_out_unless_asked_for(tmp_path):
    a = _case(tmp_path, "a")
    try:
        a.db.add_flag("Evidence")
        savedset.save_from_case(a.db, categories=False, flags=True)
    finally:
        a.close()
    b = _case(tmp_path, "b")
    try:
        savedset.apply_to_case(b.db, categories=True, flags=False)
        assert b.db.list_flags() == []
    finally:
        b.close()


def test_a_broken_file_reads_as_empty():
    savedset._path().write_text("{not json", encoding="utf-8")  # pylint: disable=protected-access
    assert savedset.load()["categories"] == []
    savedset._path().write_text(json.dumps([1, 2]), encoding="utf-8")  # pylint: disable=protected-access
    assert savedset.load()["flags"] == []


# ---- web ------------------------------------------------------------------

def _client():
    from gleapp.web.app import create_app  # pylint: disable=import-outside-toplevel
    return create_app(None).test_client()


def test_web_save_then_new_case_starts_with_the_set(tmp_path):
    savedset._path().unlink(missing_ok=True)  # pylint: disable=protected-access
    cl = _client()
    assert cl.post("/api/case/create", json={"path": str(tmp_path / "a")}).status_code == 200
    cl.post("/api/categories", json={"name": "Weapons"})
    cl.post("/api/flags", json={"name": "Evidence"})
    r = cl.post("/api/savedset/save", json={"categories": True, "flags": True}).get_json()
    assert r["categories"] == 1 and r["flags"] == 1
    assert cl.post("/api/savedset/save", json={}).status_code == 400

    cl.post("/api/case/close")
    ctx = cl.get("/api/context").get_json()
    assert [c["name"] for c in ctx["saved_set"]["categories"]] == ["Weapons"]

    assert cl.post("/api/case/create", json={
        "path": str(tmp_path / "b"), "use_categories": True, "use_flags": False,
    }).status_code == 200
    names = [c["name"] for c in cl.get("/api/categories").get_json()]
    assert "Weapons" in names
    assert cl.get("/api/flags").get_json() == []

    res = cl.post("/api/savedset/apply", json={"flags": True}).get_json()
    assert res["flags"] == [{"name": "Evidence", "status": "added"}]
    cl.post("/api/case/close")


def test_a_new_case_without_the_boxes_gets_nothing(tmp_path):
    cl = _client()
    savedset._write({"categories": [  # pylint: disable=protected-access
        {"code": 6, "name": "Weapons", "color": "#000000", "notable": True}], "flags": []})
    assert cl.post("/api/case/create", json={"path": str(tmp_path / "c")}).status_code == 200
    names = [c["name"] for c in cl.get("/api/categories").get_json()]
    assert "Weapons" not in names
    cl.post("/api/case/close")


def test_the_ui_has_the_buttons_dialog_and_new_case_boxes():
    html = TEMPLATE.read_text(encoding="utf-8")
    assert html.count('data-setdlg="save"') == 2      # category and flag editors
    assert html.count('data-setdlg="apply"') == 2
    assert 'id="setDlg"' in html
    assert 'id="optSavedCats"' in html and 'id="optSavedFlags"' in html


def test_the_new_case_boxes_never_start_ticked():
    js = (TEMPLATE.parents[1] / "static/app.js").read_text(encoding="utf-8")
    assert '$("#optSavedCats").checked = false;' in js
    assert '$("#optSavedFlags").checked = false;' in js
    html = TEMPLATE.read_text(encoding="utf-8")
    assert 'id="optSavedCats" checked' not in html
    assert 'id="optSavedFlags" checked' not in html


def test_the_ingest_options_sit_in_a_collapsed_section():
    html = TEMPLATE.read_text(encoding="utf-8")
    start = html.index('<details class="psec" id="procOpts">')
    end = html.index("</details>", start)
    sec = html[start:end]
    assert "<summary>Processing options</summary>" in sec
    for opt in ("optScreen", "optStash", "optSavedCats", "optSavedFlags", "optStage",
                "optExpand", "optDocs", "optCarve", "optKf"):
        assert f'id="{opt}"' in sec


# ---- launcher: editing the saved categories with no case open -------------

def test_set_categories_cleans_and_numbers():
    got = savedset.set_categories([
        {"code": 7, "name": " Drugs ", "color": "#123456"},
        {"code": 7, "name": "Cash"},            # code taken - renumbered
        {"name": "drugs"},                      # same name - dropped
        {"name": "   "},                        # blank - dropped
        {"code": 2, "name": "Weapons"},         # preset code - renumbered
        {"name": "New category 1"},             # no code yet
        "junk",
    ])
    assert [(c["code"], c["name"]) for c in got] == [
        (7, "Drugs"), (8, "Cash"), (9, "Weapons"), (10, "New category 1")]
    assert got[0]["color"] == "#123456"
    assert all(c["color"].startswith("#") for c in got)
    assert savedset.load()["categories"] == got


def test_set_categories_leaves_the_saved_flags_alone():
    savedset._write({"categories": [], "flags": [  # pylint: disable=protected-access
        {"name": "Evidence", "color": "#000000"}]})
    savedset.set_categories([{"name": "Weapons"}])
    assert [f["name"] for f in savedset.load()["flags"]] == ["Evidence"]


def test_web_edits_the_saved_categories_with_no_case_open():
    cl = _client()
    r = cl.post("/api/savedset/categories", json={"categories": [{"name": "Weapons"}]})
    assert r.status_code == 200
    assert [(c["code"], c["name"]) for c in r.get_json()["categories"]] == [(6, "Weapons")]
    assert cl.post("/api/savedset/categories", json={"categories": "x"}).status_code == 400


def test_the_launcher_editor_gets_the_project_vic_presets():
    from gleapp.db import VIC_PRESETS  # pylint: disable=import-outside-toplevel
    got = _client().get("/api/savedset").get_json()["presets"]
    assert [(p["code"], p["name"]) for p in got] == [(c, n) for c, n, *_ in VIC_PRESETS]


def test_the_launcher_menu_has_categories():
    html = TEMPLATE.read_text(encoding="utf-8")
    menu = html[html.index('id="refMenuLauncher"'):]
    menu = menu[:menu.index("</div>")]
    assert 'id="btnCatsLauncher"' in menu
    assert 'id="myCatDlg"' in html
