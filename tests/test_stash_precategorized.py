"""Merge external pre-categorized hash set: JSON records with an MD5 and a numeric
Category from another tool go into the stash,
categories 1-3 only, and a file with no such records is refused rather than reported
as a merge that added nothing.

Everything here is synthetic.
"""

from __future__ import annotations

import json

import pytest

from gleapp import hashstore, stash


@pytest.fixture(autouse=True)
def _isolate(tmp_path_factory, monkeypatch):
    cfg = tmp_path_factory.mktemp("cfg")
    monkeypatch.setenv("GLEAPP_CONFIG_DIR", str(cfg))
    monkeypatch.setenv("GLEAPP_STASH_PATH", str(cfg / "stash.hstash"))
    hashstore.close()
    stash.close()
    yield
    hashstore.close()
    stash.close()


def _md5(n: int) -> str:
    return f"{n:032x}"


def _write(path, records, *, wrapped=True):
    doc = {"@odata.context": "x", "value": records} if wrapped else records
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_categories_one_to_three_are_merged_and_the_rest_counted(tmp_path):
    src = _write(tmp_path / "hashes.json", [
        {"MediaID": 1, "MD5": _md5(1).upper(), "SHA1": "a" * 40, "Category": 1},
        {"MediaID": 2, "MD5": _md5(2), "Category": 2},
        {"MediaID": 3, "MD5": _md5(3), "Category": 3},
        {"MediaID": 4, "MD5": _md5(4), "Category": 4},
        {"MediaID": 5, "SHA1": "b" * 40, "Category": 1},
    ])
    res = stash.merge_precategorized(src)
    assert res["added"] == 3 and res["total"] == 3
    assert res["records"] == 5
    assert res["skipped_category"] == 1 and res["skipped_no_md5"] == 1
    assert (stash.lookup(_md5(1)) or {}).get("category") == 1
    assert stash.lookup(_md5(4)) is None
    rows = {md5: src_ for md5, _cat, src_, _ in stash.iter_all()}
    assert rows[_md5(2)] == "pre-categorized: hashes.json"


def test_a_bare_list_is_read_too(tmp_path):
    src = _write(tmp_path / "list.json", [{"MD5": _md5(9), "Category": "2"}], wrapped=False)
    assert stash.merge_precategorized(src)["added"] == 1


def test_a_file_without_md5_and_category_is_refused(tmp_path):
    src = _write(tmp_path / "other.json", [{"name": "x"}, {"name": "y"}])
    with pytest.raises(ValueError, match="no records with an MD5 and a Category"):
        stash.merge_precategorized(src)
    plain = tmp_path / "list.csv"
    plain.write_text("MD5,Category\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON file"):
        stash.merge_precategorized(plain)
