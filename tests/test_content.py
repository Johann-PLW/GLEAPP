"""Find similar, similar content (content.py): the bundled model, index and ranking.

The index is filled with hand-made vectors so the ranking, the cutoff and the combined
Find similar reply can be checked exactly without running the model. One test does run
it, to check that the installed OpenCV reads the bundled model and computes it right.
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
        ids[name] = c.db.upsert_file(f"/x/{name}.jpg", kind="image", thumb=f"{name}.jpg", md5=name,
                                     width=400, height=300)
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


def test_the_bundled_model_is_the_recorded_file():
    """The model ships in gleapp/models; a truncated or swapped file must not pass."""
    import hashlib
    path = content.model_path()
    assert path.is_file(), "gleapp/models/dinov2_small.onnx is missing"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == content.MODEL_SHA256


# The stored vector (content._vector) for _fixed_input(), as ONNX Runtime 1.30.0 computes
# it, at the eight values most sensitive to the model's cubic Resize. The same model with
# that Resize made linear moves each of them by 2.9e-3 or more; OpenCV 5.0.0 matched all
# 768 values to within 2.5e-7 (measured 2026-09-30, macOS arm64).
_ORT_VALUES = {123: -0.038436, 155: -0.0320167, 213: 0.0668716, 269: -0.0570372,
               405: -0.0018007, 558: 0.037907, 592: -0.0569444, 648: -0.0375552}


def _fixed_input() -> np.ndarray:
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float64)
    chans = [np.sin(xx / 9.0 + c) * np.cos(yy / 13.0 - c) + (xx - yy) / 224.0 for c in range(3)]
    return np.stack(chans)[None].astype(np.float32)


def test_the_installed_opencv_reads_the_bundled_model():
    """Every OpenCV 4.x imports fine and cannot read this model: 4.8 stops on its Expand
    node and 4.9 through 4.14 on its cubic Resize. On such an install every content pass
    failed, so requirements.txt asks for 5.0. The values are checked against ONNX Runtime
    so an OpenCV that reads the model but computes it differently fails too."""
    import cv2
    try:
        net = cv2.dnn.readNetFromONNX(str(content.model_path()))  # pylint: disable=no-member
    except cv2.error as exc:  # pylint: disable=catching-non-exception,no-member
        version = cv2.__version__  # pylint: disable=no-member
        pytest.fail(f"OpenCV {version} cannot read the content model, which needs "
                    f"OpenCV 5.0 or newer: {' '.join(str(exc).split())[-120:]}")
    net.setInput(_fixed_input())
    v = content._vector(net.forward()[0].astype(np.float32))  # pylint: disable=protected-access
    assert v.shape == (content.DIM,)
    assert {i: float(v[i]) for i in _ORT_VALUES} == pytest.approx(_ORT_VALUES, abs=1e-4)


def test_build_needs_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(content, "model_path", lambda: tmp_path / "missing.onnx")
    c = open_case(tmp_path / "case", create=True, examiner="t")
    try:
        with pytest.raises(ValueError, match="missing from this build"):
            content.build_index(c)
    finally:
        c.close()


def test_find_similar_lists_copies_then_similar_content(case_with_vectors, monkeypatch):
    from gleapp.web.app import create_app
    c, ids = case_with_vectors
    monkeypatch.setattr(content, "model_ready", lambda: True)
    dup = c.db.upsert_file("/x/query_copy.jpg", kind="image", thumb="query.jpg", md5="query",
                           width=400, height=300)
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


def test_which_pictures_are_described_and_in_what_order(tmp_path):
    """One per exact-duplicate group, none under MIN_SIDE, system artwork last."""
    c = open_case(tmp_path / "case", create=True, examiner="t")
    try:
        up = lambda p, w, **kw: c.db.upsert_file(p, kind="image", thumb="t.jpg", width=w, height=w, **kw)
        photo = up("/data/DCIM/a.jpg", 400)
        dup = up("/data/Backup/a.jpg", 400)
        icon = up("/data/icons/i.png", 32)
        system = up("/Basic data partition/Windows/Web/wall.jpg", 1920)
        big = up("/data/DCIM/b.jpg", 3000)
        c.db.conn.execute("UPDATE files SET stack_id = ? WHERE id IN (?, ?)", (photo, photo, dup))
        c.db.conn.commit()
        with c.db.lock:
            content._ensure(c.db.conn)  # pylint: disable=protected-access
        todo = [r[0] for r in c.db.conn.execute(
            f"SELECT id FROM files WHERE id IN ({content._todo_sql()}) "  # pylint: disable=protected-access
            f"ORDER BY {content.SYSTEM_PATH_SQL}, MAX(COALESCE(width, 0), COALESCE(height, 0)) DESC, id")]  # pylint: disable=protected-access
        assert todo == [big, photo, system]
        assert icon not in todo and dup not in todo
        assert content.status(c)["indexable"] == 3
    finally:
        c.close()


def test_an_undescribed_duplicate_searches_with_its_group(case_with_vectors):
    """A file that shares its group's description searches with it, and its own group
    (here the described original) is listed first."""
    c, ids = case_with_vectors
    dup = c.db.upsert_file("/x/query_again.jpg", kind="image", thumb="query.jpg", md5="query")
    c.db.conn.execute("UPDATE files SET stack_id = ? WHERE id IN (?, ?)", (ids["query"], ids["query"], dup))
    c.db.conn.commit()
    hits = content.find_content(c, dup, min_similarity=0.5)
    assert [h["id"] for h in hits] == [ids["query"], ids["near"], ids["far"]]


def test_a_hit_brings_its_exact_duplicates_with_it(case_with_vectors):
    """Exact duplicates share one description; when it is a hit, every copy is listed."""
    c, ids = case_with_vectors
    again = c.db.upsert_file("/x/near_again.jpg", kind="image", thumb="near.jpg", md5="near",
                             width=400, height=300)
    c.db.conn.execute("UPDATE files SET stack_id = ? WHERE id IN (?, ?)", (ids["near"], ids["near"], again))
    c.db.conn.commit()
    hits = content.find_content(c, ids["query"], min_similarity=0.5)
    order = [h["id"] for h in hits]
    assert order[:2] == [ids["near"], again] and ids["far"] in order
    assert hits[0]["similarity"] == hits[1]["similarity"]
    with c.db.lock:
        todo = {r[0] for r in c.db.conn.execute(content._todo_sql())}  # pylint: disable=protected-access
    assert ids["near"] in todo and again not in todo
