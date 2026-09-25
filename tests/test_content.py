"""Find similar, similar content (content.py): model import, index and ranking.

The real model is an 88 MB file the examiner imports, so these tests stand in for it:
the import check is exercised with a file that is not the model, and the index is
filled with hand-made vectors so the ranking, the cutoff and the combined Find similar
reply can be checked exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from gleapp import content
from gleapp.case import open_case


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(name="case_with_vectors")
def _case_with_vectors(tmp_path):
    c = open_case(tmp_path / "case", create=True, examiner="t")
    rng = np.random.default_rng(0)
    base = _unit(rng.normal(size=content.DIM))
    ids = {}
    for name, noise in (("query", 0.0), ("near", 0.3), ("far", 0.9), ("other", None)):
        ids[name] = c.db.upsert_file(f"/x/{name}.jpg", kind="image", thumb=f"{name}.jpg", md5=name)
        v = _unit(rng.normal(size=content.DIM)) if noise is None else \
            _unit(base + noise * _unit(rng.normal(size=content.DIM)))
        with c.db.lock:
            content._ensure(c.db.conn)  # pylint: disable=protected-access
            c.db.conn.execute("INSERT INTO content_vecs VALUES (?, ?)",
                              (ids[name], v.astype(np.float16).tobytes()))
    c.db.conn.commit()
    yield c, ids
    c.close()


def test_ranked_best_first_down_to_the_cutoff(case_with_vectors):
    c, ids = case_with_vectors
    hits = content.find_content(c, ids["query"], min_similarity=0.5)
    assert [h["id"] for h in hits] == [ids["near"], ids["far"]]
    assert hits[0]["similarity"] > hits[1]["similarity"] and all(h["match"] == "content" for h in hits)
    assert [h["id"] for h in content.find_content(c, ids["query"], min_similarity=0.9)] == [ids["near"]]


def test_files_listed_elsewhere_are_left_out(case_with_vectors):
    c, ids = case_with_vectors
    hits = content.find_content(c, ids["query"], min_similarity=0.5, exclude={ids["near"]})
    assert [h["id"] for h in hits] == [ids["far"]]


def test_a_file_that_is_not_the_model_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(content, "model_path", lambda: tmp_path / "models" / "m.onnx")
    fake = tmp_path / "model.onnx"
    fake.write_bytes(b"not the model")
    with pytest.raises(ValueError, match="not the expected model file"):
        content.import_model(fake)
    assert not (tmp_path / "models" / "m.onnx").exists()


def test_build_needs_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(content, "model_path", lambda: tmp_path / "missing.onnx")
    c = open_case(tmp_path / "case", create=True, examiner="t")
    try:
        with pytest.raises(ValueError, match="import the model"):
            content.build_index(c)
    finally:
        c.close()


def test_find_similar_lists_copies_then_similar_content(case_with_vectors, monkeypatch):
    from gleapp.web.app import create_app
    c, ids = case_with_vectors
    monkeypatch.setattr(content, "model_ready", lambda: True)
    dup = c.db.upsert_file("/x/query_copy.jpg", kind="image", thumb="query.jpg", md5="query")
    c.db.conn.execute("UPDATE files SET stack_id = ? WHERE id IN (?, ?)", (ids["query"], ids["query"], dup))
    c.db.conn.commit()
    root = c.root
    client = create_app(str(root)).test_client()
    try:
        d = client.get(f"/api/similar/{ids['query']}?min=50").get_json()
        order = [(f["id"], f["match"]) for f in d["files"]]
        assert order[0] == (ids["query"], "query")
        assert (dup, "copy") in order
        content_ids = [i for i, m in order if m == "content"]
        assert content_ids == [ids["near"], ids["far"]] and d["content_on"]
        tighter = client.get(f"/api/similar/{ids['query']}?min=90").get_json()
        assert [f["id"] for f in tighter["files"] if f["match"] == "content"] == [ids["near"]]
    finally:
        client.post("/api/case/close")
