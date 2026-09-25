"""Find similar, similar content: other photos of the same person, place, object or
scene, and other drawings of the same character, ranked by what is in the picture.

Copies of a picture are simindex.py's job. This describes each image by its content with
DINOv2-small (Meta AI, Apache-2.0), run through OpenCV's DNN module, and ranks the case
by cosine similarity to the searched image.

The model is not shipped. The examiner imports the file once; it is accepted only if its
SHA-256 is the one below, and copied under the app's data folder. Each image's vector
(768 float16s: the model's summary token and the average of its patch tokens, each made
unit length) is made from the thumbnail the case already holds, so building the index
never reads the evidence again.

Measured on a real laptop case (belkawindows), 93 images in 9 hand-labeled groups among
5,000 other images of that case: ranking every image by this similarity, the other
members of the group came first for the chair (all 10 photos, from every angle), the
yellow sticker character (other poses; the only others in its top 13 were two poses the
labels had missed), the THANK YOU team photos (4 of 4) and the room renders (16 of 16).
Above a cutoff of 0.65, 97% of group members were found. Scores differ by subject (the
chair photos scored 0.70-0.85 with each other, the sticker poses 0.88-0.91), so results
are ranked and cut off by a strictness the examiner sets. Building the index ran at about
25 images a second on an 8-core laptop (57,379 images: about 40 minutes).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import appconfig

# onnx-community/dinov2-small, onnx/model.onnx at commit 8b1f705, a conversion of
# facebook/dinov2-small; 88,532,934 bytes.
MODEL_SHA256 = "f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a"
MODEL_URL = "https://huggingface.co/onnx-community/dinov2-small/resolve/8b1f705/onnx/model.onnx"
MODEL_NAME = "dinov2_small.onnx"
INDEX_VERSION = "1"
DIM = 768
DEFAULT_MIN = 0.70
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_CACHE: dict = {}


# ---- the model --------------------------------------------------------------------------
def model_path() -> Path:
    return appconfig.data_dir() / "models" / MODEL_NAME


def model_ready() -> bool:
    return model_path().is_file()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_model(src) -> Path:
    """Copy the model file into place, only if it is exactly the expected one."""
    src = Path(str(src).strip().strip('"'))
    if not src.is_file():
        raise ValueError(f"not a file: {src}")
    got = _sha256(src)
    if got != MODEL_SHA256:
        raise ValueError(f"this is not the expected model file (SHA-256 {got}, "
                         f"expected {MODEL_SHA256})")
    dest = model_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)
    return dest


# ---- storage ----------------------------------------------------------------------------
def _ensure(conn) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS content_vecs (file_id INTEGER PRIMARY KEY "
                 "REFERENCES files(id) ON DELETE CASCADE, vec BLOB NOT NULL)")
    row = conn.execute("SELECT value FROM meta WHERE key='content_index_version'").fetchone()
    if row is None or row[0] != INDEX_VERSION:
        conn.execute("DELETE FROM content_vecs")
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('content_index_version', ?)",
                     (INDEX_VERSION,))
    conn.commit()


_INDEXABLE = "kind IN ('image', 'video') AND thumb IS NOT NULL"


def status(case) -> dict:
    with case.db.lock:
        _ensure(case.db.conn)
        q = lambda sql: case.db.conn.execute(sql).fetchone()[0]
        return {"model": model_ready(), "indexed": q("SELECT COUNT(*) FROM content_vecs"),
                "indexable": q(f"SELECT COUNT(*) FROM files WHERE {_INDEXABLE}")}


# ---- vectors ----------------------------------------------------------------------------
def _prep(path: Path) -> np.ndarray:
    """The whole image, fitted inside 224 x 224 and padded, as the model's input."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((224, 224), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (224, 224), tuple(int(x * 255) for x in _MEAN))
        canvas.paste(im, ((224 - im.width) // 2, (224 - im.height) // 2))
        a = np.asarray(canvas, np.float32) / 255.0
    return ((a - _MEAN) / _STD).transpose(2, 0, 1)[None]


def _vector(tokens: np.ndarray) -> np.ndarray:
    cls, mean = tokens[0], tokens[1:].mean(0)
    v = np.concatenate([cls / (np.linalg.norm(cls) + 1e-12), mean / (np.linalg.norm(mean) + 1e-12)])
    return v / (np.linalg.norm(v) + 1e-12)


def build_index(case, *, workers: int = 6, progress=None) -> int:
    """Describe every image and video thumbnail not yet indexed. Resumable: a stopped run
    keeps what it wrote and the next picks up the rest."""
    import cv2
    if not model_ready():
        raise ValueError("import the model file first")
    with case.db.lock:
        _ensure(case.db.conn)
        rows = case.db.conn.execute(
            f"SELECT id, thumb FROM files WHERE {_INDEXABLE} "
            "AND id NOT IN (SELECT file_id FROM content_vecs) ORDER BY id").fetchall()
    total = len(rows)
    local = threading.local()

    def one(r):
        if not hasattr(local, "net"):
            local.net = cv2.dnn.readNetFromONNX(str(model_path()))  # pylint: disable=no-member
        try:
            local.net.setInput(_prep(case.thumb_dir / r[1]))
            v = _vector(local.net.forward()[0].astype(np.float32))
        # pylint: disable-next=broad-exception-caught
        except Exception:  # noqa: BLE001 - one unreadable thumbnail must not stop the run
            return r[0], None
        return r[0], v.astype(np.float16).tobytes()

    done, batch = 0, []
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for fid, blob in ex.map(one, rows):
            done += 1
            if blob is not None:
                batch.append((fid, blob))
            if len(batch) >= 200 or done == total:
                with case.db.lock:
                    case.db.conn.executemany("INSERT OR REPLACE INTO content_vecs VALUES (?, ?)", batch)
                    case.db.conn.commit()
                batch = []
            if progress and (done % 25 == 0 or done == total):
                progress(done, total)
    _CACHE.clear()
    return done


# ---- search -----------------------------------------------------------------------------
def _matrix(case):
    conn = case.db.conn
    key = (id(conn), conn.execute("SELECT COUNT(*), MAX(file_id) FROM content_vecs").fetchone())
    if key not in _CACHE:
        rows = conn.execute("SELECT file_id, vec FROM content_vecs ORDER BY file_id").fetchall()
        ids = np.array([r[0] for r in rows], np.int64)
        mat = (np.frombuffer(b"".join(r[1] for r in rows), np.float16).reshape(-1, DIM)
               .astype(np.float32) if rows else np.zeros((0, DIM), np.float32))
        _CACHE.clear()
        _CACHE[key] = (ids, mat)
    return _CACHE[key]


def find_content(case, file_id: int, *, min_similarity: float = DEFAULT_MIN,
                 limit: int = 300, exclude=()) -> list[dict]:
    """Other files ranked by content similarity to ``file_id``, best first, down to
    ``min_similarity``; ``exclude`` holds ids already listed elsewhere."""
    with case.db.lock:
        _ensure(case.db.conn)
        ids, mat = _matrix(case)
    at = np.searchsorted(ids, file_id)
    if at >= len(ids) or ids[at] != file_id:
        return []
    sims = mat @ mat[at]
    skip = set(exclude) | {file_id}
    out = []
    for k in np.argsort(-sims):
        s = float(sims[k])
        if s < min_similarity or len(out) >= limit:
            break
        fid = int(ids[k])
        if fid in skip:
            continue
        row = case.db.get_file(fid)
        if row is not None:
            out.append(dict(row, match="content", similarity=round(100 * min(1.0, s), 1)))
    return out
