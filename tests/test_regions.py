"""Regional category schemes: a case is seeded with the region chosen app-wide
when it is created and keeps it; the hash stash keeps each entry under its own
region and a case matches only its own region's entries."""
from __future__ import annotations

import csv
import sqlite3

import pytest

from gleapp import appconfig, hashdb, projectvic, regions, stash
from gleapp.case import open_case


@pytest.fixture(autouse=True)
def _fresh_stash(tmp_path, monkeypatch):
    monkeypatch.setenv("GLEAPP_STASH_PATH", str(tmp_path / "stash.hstash"))
    stash.close()
    yield
    stash.close()


def _set_region(rid):
    cfg = appconfig.load()
    if rid is None:
        cfg.pop("category_region", None)
        appconfig.save(cfg)
    else:
        appconfig.set_region(rid)


@pytest.fixture(autouse=True)
def _no_region():
    _set_region(None)
    yield
    _set_region(None)


def test_every_scheme_has_one_uncategorized_and_unique_codes():
    for rid, s in regions.REGIONS.items():
        codes = [p.code for p in s.presets]
        assert len(codes) == len(set(codes)), rid
        assert s.uncategorized in codes, rid
        assert s.nonpertinent is None or s.nonpertinent in codes, rid
        assert s.first_custom > max(codes), rid


def test_us_scheme_is_unchanged():
    us = regions.get("US")
    assert [(p.code, p.notable) for p in us.presets] == [
        (0, False), (1, True), (2, True), (3, True), (4, False), (5, False)]
    assert us.uncategorized == 5 and us.nonpertinent == 0
    assert us.first_custom == 6 and us.notable_codes == (1, 2, 3)
    assert regions.get(None) is us and regions.get("nowhere") is us


def test_a_case_made_with_no_region_chosen_is_us(tmp_path):
    c = open_case(tmp_path / "c", create=True, examiner="t")
    try:
        assert c.db.get_meta("category_region") == "US"
        assert c.db.category_name(5) == "Uncategorized"
    finally:
        c.close()


def test_a_uk_case_is_seeded_with_the_uk_presets_and_keeps_them(tmp_path):
    _set_region("UK")
    c = open_case(tmp_path / "uk", create=True, examiner="t")
    root = c.root
    try:
        rows = {r["code"]: r for r in c.db.list_categories()}
        assert rows[1]["name"] == "SC Category A" and rows[1]["locked"]
        assert rows[0]["name"] == "Uncategorised"
        assert set(rows) == set(range(10))
        fid = c.db.upsert_file("/x/a.jpg", kind="image")
        assert c.db.get_file(fid)["category"] == 0
        assert c.db.is_uncategorized(0) and not c.db.is_uncategorized(5)
        with pytest.raises(ValueError):
            c.db.update_category(1, name="renamed")
        assert c.db.add_category("Mine") == 10
    finally:
        c.close()

    _set_region("AU")               # changing the app's region later
    c = open_case(root)
    try:
        assert c.db.scheme.id == "UK"
        assert c.db.get_category(1)["name"] == "SC Category A"
    finally:
        c.close()


def test_an_older_case_without_a_region_opens_as_us(tmp_path):
    _set_region("UK")
    c = open_case(tmp_path / "old", create=True, examiner="t")
    c.db.conn.execute("DELETE FROM meta WHERE key='category_region'")
    c.db.commit()
    root = c.root
    c.close()
    c = open_case(root)
    try:
        # the meta was gone, so it reads as a pre-region case: US
        assert c.db.scheme.id == "US"
    finally:
        c.close()


def _hit_case(tmp_path, region, name):
    _set_region(region)
    c = open_case(tmp_path / name, create=True, examiner="t")
    fid = c.db.upsert_file(f"/x/{name}.jpg", kind="image",
                           md5="0123456789abcdef0123456789abcdef")
    return c, c.db.get_file(fid)


def test_stash_entries_match_only_their_own_region(tmp_path):
    md5 = "0123456789abcdef0123456789abcdef"
    stash.add([(md5, 5, "uk case")], "UK")     # UK 5 Extreme Pornography
    assert (stash.lookup(md5, "UK") or {}).get("category") == 5
    assert stash.lookup(md5, "US") is None
    # 5 is not a stash category in the US scheme
    assert stash.add([(md5, 5, "us case")], "US")["submitted"] == 0

    uk, row = _hit_case(tmp_path, "UK", "uk")
    try:
        hits = hashdb.match_all(uk.db, row)
        assert [h["category"] for h in hits] == [5]
        assert hashdb.asserted_category(hits, uk.db) == 5
    finally:
        uk.close()
    us, row = _hit_case(tmp_path, "US", "us")
    try:
        assert hashdb.match_all(us.db, row) == []
    finally:
        us.close()


def test_a_format_1_stash_becomes_us_entries(tmp_path, monkeypatch):
    old = tmp_path / "old.hstash"
    c = sqlite3.connect(old)
    c.executescript("""
        CREATE TABLE stash_meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE stash (md5 TEXT PRIMARY KEY, category INTEGER NOT NULL,
                            added_at REAL, source TEXT);
        INSERT INTO stash VALUES ('0123456789abcdef0123456789abcdef', 2, 1, 'x');
        INSERT INTO stash_meta VALUES ('migrated_from_global', '0');
    """)
    c.commit()
    c.close()
    monkeypatch.setenv("GLEAPP_STASH_PATH", str(old))
    stash.close()
    assert (stash.lookup("0123456789abcdef0123456789abcdef", "US") or {}).get("category") == 2
    s = stash.summary("US")
    assert s["by_region"] == {"US": 1} and s["by_category"] == {2: 1}


def test_csv_export_and_merge_keep_the_region(tmp_path):
    a, b = "a" * 32, "b" * 32
    stash.add([(a, 1, "s")], "UK")
    stash.add([(b, 2, "s")], "US")
    out = stash.export(tmp_path / "s.csv")
    rows = list(csv.reader(open(out, encoding="utf-8")))
    assert rows[0] == ["md5", "category", "source", "region"]
    stash.clear()
    stash.merge(out)
    assert (stash.lookup(a, "UK") or {}).get("category") == 1
    assert (stash.lookup(b, "US") or {}).get("category") == 2
    assert stash.lookup(a, "US") is None


def test_known_good_lands_only_where_the_region_names_non_pertinent(tmp_path):
    from gleapp.pipeline import rematch_hashes
    md5 = "c" * 32
    for rid, want in (("US", 0), ("UK", 0), ("EUROPOL", 3)):
        _set_region(rid)
        c = open_case(tmp_path / rid, create=True, examiner="t")
        try:
            hs = c.db.create_hashset("good", "test", "known-good")
            c.db.add_hashset_entries(hs, [("md5", md5, None)])
            fid = c.db.upsert_file(f"/x/{rid}.jpg", kind="image", md5=md5)
            rematch_hashes(c)
            got = c.db.get_file(fid)["category"]
            # the UK scheme has no non-pertinent category: stays Uncategorised (0)
            assert got == want, rid
            assert c.db.is_uncategorized(got) == (rid == "UK")
        finally:
            c.close()


def test_vic_region_mismatch_is_reported():
    class _Case:
        class db:
            scheme = regions.get("UK")
    ctx = "http://github.com/VICSDATAMODEL/ProjectVic/DataModels/2.0.xml/US/$metadata#Cases"
    assert projectvic.vic_region(ctx) == "US"
    assert "US data model" in projectvic.region_mismatch(_Case, ctx)
    assert projectvic.region_mismatch(_Case, ctx.replace("/US/", "/UK/")) is None
    assert projectvic.region_mismatch(_Case, None) is None


def test_settings_api_sets_the_region():
    from gleapp.web.app import create_app
    app = create_app(None)
    cl = app.test_client()
    ctx = cl.get("/api/context").get_json()
    assert ctx["region"] is None
    assert ctx["region_choices"][0]["id"] == "US"
    assert cl.post("/api/settings", json={"region": "nowhere"}).status_code == 400
    r = cl.post("/api/settings", json={"region": "se"}).get_json()
    assert r["region"] == "SE"
    assert appconfig.get_region() == "SE"
