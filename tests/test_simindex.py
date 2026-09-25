"""Find similar, copies (simindex.py): index, shortlist and confirmation.

Synthetic images only: a drawn scene is edited the ways a copy of a picture gets edited
(resized, cropped, mirrored, rotated, bordered, captioned, pasted into a screenshot) and
must be found among unrelated drawn scenes, which must not be.
"""

from __future__ import annotations

import random

import pytest
from PIL import Image, ImageDraw, ImageOps

from gleapp import simindex
from gleapp.case import open_case


def _scene(seed: int, size=(900, 650)) -> Image.Image:
    rng = random.Random(seed)
    im = Image.new("RGB", size, tuple(rng.randrange(40, 220) for _ in range(3)))
    d = ImageDraw.Draw(im)
    for _ in range(140):
        x, y = rng.randrange(size[0]), rng.randrange(size[1])
        col = tuple(rng.randrange(256) for _ in range(3))
        if rng.random() < 0.5:
            d.ellipse((x, y, x + rng.randrange(8, 70), y + rng.randrange(8, 70)), fill=col)
        else:
            d.rectangle((x, y, x + rng.randrange(8, 70), y + rng.randrange(8, 70)), outline=col, width=3)
    return im


def _edits(im: Image.Image) -> dict[str, Image.Image]:
    w, h = im.size
    border = Image.new("RGB", (w + 120, h + 120), "black")
    border.paste(im, (60, 60))
    caption = Image.new("RGB", (w, int(h * 1.2)), "white")
    caption.paste(im, (0, 0))
    ImageDraw.Draw(caption).text((20, int(h * 1.07)), "look at this", fill="black", font_size=40)
    shot = Image.new("RGB", (1170, 2532), (20, 20, 20))
    shot.paste(im.resize((1170, int(h * 1170 / w))), (0, 900))
    return {"half": im.resize((w // 2, h // 2)),
            "crop": im.crop((int(w * .1), int(h * .1), int(w * .9), int(h * .9))),
            "mirror": ImageOps.mirror(im),
            "rot90": im.transpose(Image.Transpose.ROTATE_90),
            "border": border, "caption": caption, "screenshot": shot}


def _add(case, name: str, im: Image.Image) -> int:
    t = im.copy()
    t.thumbnail((320, 320))
    t.save(case.thumb_dir / f"{name}.jpg", quality=90)
    return case.db.upsert_file(f"/evidence/{name}.jpg", kind="image", thumb=f"{name}.jpg",
                               md5=name, width=im.width, height=im.height)


@pytest.fixture(name="indexed")
def _indexed(tmp_path):
    c = open_case(tmp_path / "case", create=True, examiner="t")
    c.thumb_dir.mkdir(parents=True, exist_ok=True)
    src = _scene(1)
    ids = {"source": _add(c, "source", src)}
    for k, im in _edits(src).items():
        ids[k] = _add(c, k, im)
    for n in range(12):
        ids[f"other{n}"] = _add(c, f"other{n}", _scene(100 + n))
    grad = Image.linear_gradient("L").resize((600, 400)).convert("RGB")
    ids["gradient"] = _add(c, "gradient", grad)
    ids["gradient_half"] = _add(c, "gradient_half", grad.resize((300, 200)))
    c.db.conn.commit()
    simindex.build_index(c, workers=2)
    yield c, ids
    c.close()


def test_every_kind_of_copy_is_found_and_nothing_unrelated(indexed):
    c, ids = indexed
    hits = simindex.find_copies(c, ids["source"])
    assert hits[0]["id"] == ids["source"] and hits[0]["match"] == "query"
    found = {h["id"] for h in hits[1:]}
    copies = {ids[k] for k in ("half", "crop", "mirror", "rot90", "border", "caption", "screenshot")}
    assert copies <= found, sorted(k for k in ids if ids[k] in copies - found)
    assert not found & {ids[f"other{n}"] for n in range(12)}
    for h in hits[1:]:
        assert h["match"] == "copy" and h["points"] >= simindex.MIN_POINTS


def test_a_copy_finds_the_original_too(indexed):
    c, ids = indexed
    found = {h["id"] for h in simindex.find_copies(c, ids["screenshot"])[1:]}
    assert ids["source"] in found


def test_a_featureless_picture_is_matched_by_its_fingerprint(indexed):
    c, ids = indexed
    hits = simindex.find_copies(c, ids["gradient"])
    match = [h for h in hits[1:] if h["id"] == ids["gradient_half"]]
    assert match and match[0]["points"] is None


def test_status_and_building_only_new_images(indexed):
    c, ids = indexed
    st = simindex.status(c)
    assert st["vocab"] and st["indexed"] == st["indexable"] == len(ids)
    new = _add(c, "late", _scene(55))
    c.db.conn.commit()
    assert simindex.build_index(c, workers=2) == 1
    assert simindex.status(c)["indexed"] == len(ids) + 1
    assert new in {r[0] for r in c.db.conn.execute("SELECT file_id FROM sim_items")}


def test_an_index_of_another_version_is_discarded(indexed):
    c, _ = indexed
    c.db.conn.execute("UPDATE meta SET value='0' WHERE key='sim_index_version'")
    c.db.conn.commit()
    st = simindex.status(c)
    assert st["indexed"] == 0 and not st["vocab"]


def test_the_route_uses_the_index_once_built(tmp_path):
    from gleapp.web.app import create_app
    c = open_case(tmp_path / "case", create=True, examiner="t")
    c.thumb_dir.mkdir(parents=True, exist_ok=True)
    src = _scene(3)
    sid = _add(c, "source", src)
    mid = _add(c, "mirror", ImageOps.mirror(src))
    for n in range(6):
        _add(c, f"other{n}", _scene(200 + n))
    c.db.conn.commit()
    root = c.root
    c.close()
    client = create_app(str(root)).test_client()
    try:
        d = client.get(f"/api/similar/{sid}").get_json()
        assert d["engine"] == "hash"
        assert client.post("/api/simindex/build").get_json()["ok"]
        import time
        for _ in range(600):
            if not client.get("/api/job").get_json()["running"]:
                break
            time.sleep(0.1)
        assert client.get("/api/simindex/status").get_json()["indexed"] == 8
        d = client.get(f"/api/similar/{sid}").get_json()
        assert d["engine"] == "index" and mid in {f["id"] for f in d["files"]}
    finally:
        client.post("/api/case/close")
