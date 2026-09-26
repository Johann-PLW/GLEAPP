"""Find similar, similar content: other photos of the same person, place, object or
scene, and other drawings of the same character, ranked by what is in the picture.

Copies of a picture are simindex.py's job. This describes each image by its content with
DINOv2-small (Meta AI, Apache-2.0), run through OpenCV's DNN module, and ranks the case
by cosine similarity to the searched image.

The model ships with GLEAPP in gleapp/models: Meta's own facebook/dinov2-small weights
(Hugging Face revision ed25f3a31f01632728cabb09d1542f84ab7b0056), converted to ONNX
with PyTorch 2.14 (opset 17); its vectors matched those of the onnx-community conversion
the measurements below were made with to a cosine of 1.00000 on 300 real thumbnails.
Each image's vector (768 float16s: the model's summary token and the average of its
patch tokens, each made unit length) is made from the thumbnail the case already holds,
so building the index never reads the evidence again.

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

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# gleapp/models/dinov2_small.onnx, 88,411,480 bytes; tests/test_content.py checks it.
MODEL_SHA256 = "b88d3157590250f1ed413bd41289ce2b4f6c65f3699684eaa913efc2330e2822"
MODEL_NAME = "dinov2_small.onnx"
INDEX_VERSION = "1"
DIM = 768
DEFAULT_MIN = 0.70
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_CACHE: dict = {}


# ---- the model --------------------------------------------------------------------------
def model_path() -> Path:
    return Path(__file__).with_name("models") / MODEL_NAME


def model_ready() -> bool:
    return model_path().is_file()


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


CHUNK = 64               # files a build describes between checks for a request to stop
MIN_SIDE = 128          # smaller pictures (icons, cursors, buttons) are not described
_INDEXABLE = ("kind IN ('image', 'video') AND thumb IS NOT NULL "
              f"AND MAX(COALESCE(width, 0), COALESCE(height, 0)) >= {MIN_SIDE}")
# Folders that hold the operating system's and applications' own artwork. Their images
# are described last, so a search over the examiner's likely material works early.
SYSTEM_PATH_SQL = ("(LOWER(COALESCE(orig_path, path)) LIKE '%/windows/%' "
                "OR LOWER(COALESCE(orig_path, path)) LIKE '%/program files%' "
                "OR LOWER(COALESCE(orig_path, path)) LIKE '%/programdata/%' "
                "OR LOWER(COALESCE(orig_path, path)) LIKE '%/system/library/%' "
                "OR LOWER(COALESCE(orig_path, path)) LIKE '%/applications/%.app/%' "
                "OR LOWER(COALESCE(orig_path, path)) LIKE '%\\windows\\%' "
                "OR LOWER(COALESCE(orig_path, path)) LIKE '%\\program files%')")


def _todo_sql() -> str:
    """One row per picture still to describe: exact duplicates share one description
    (the group's lowest id), icons under MIN_SIDE are skipped, and the examiner's likely
    material comes before the system's and applications' own artwork, larger first.

    Measured on four real cases: skipping duplicates and icons left 4,975 of 57,406 images
    on a Windows laptop (about 40 minutes of model time down to about 3.5), 7,270 of 24,320
    on another laptop, 15,346 of 27,188 and 19,888 of 33,109 on two phones."""
    return (f"SELECT MIN(id) AS id FROM files WHERE {_INDEXABLE} "
            "GROUP BY COALESCE(stack_id, id)")


def status(case) -> dict:
    with case.db.lock:
        _ensure(case.db.conn)
        q = lambda sql: case.db.conn.execute(sql).fetchone()[0]
        return {"model": model_ready(), "indexed": q("SELECT COUNT(*) FROM content_vecs"),
                "indexable": q(f"SELECT COUNT(*) FROM ({_todo_sql()})")}


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


def build_index(case, *, workers: int = 6, progress=None, stop=None) -> int:
    """Describe every image and video thumbnail not yet indexed. Resumable: ``stop`` (a
    callable) is checked between chunks of CHUNK files; what is written stays and the
    next run picks up the rest."""
    import cv2
    if not model_ready():
        raise ValueError("the content model is missing from this build")
    with case.db.lock:
        _ensure(case.db.conn)
        rows = case.db.conn.execute(
            f"SELECT id, thumb FROM files WHERE id IN ({_todo_sql()}) "
            "AND id NOT IN (SELECT file_id FROM content_vecs) "
            f"ORDER BY {SYSTEM_PATH_SQL}, MAX(COALESCE(width, 0), COALESCE(height, 0)) DESC, id"
            ).fetchall()
    total = len(rows)
    local = threading.local()
    stopped = stop or (lambda: False)

    def one(r):
        if stopped():
            return r[0], None                # not written: the next build does it
        if not hasattr(local, "net"):
            local.net = cv2.dnn.readNetFromONNX(str(model_path()))  # pylint: disable=no-member
        try:
            local.net.setInput(_prep(case.thumb_dir / r[1]))
            v = _vector(local.net.forward()[0].astype(np.float32))
        # pylint: disable-next=broad-exception-caught
        except Exception:  # noqa: BLE001 - one unreadable thumbnail must not stop the run
            return r[0], None
        return r[0], v.astype(np.float16).tobytes()

    done = 0
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for k in range(0, total, CHUNK):
            if stopped():
                break
            batch = []
            for fid, blob in ex.map(one, rows[k:k + CHUNK]):
                done += 1
                if blob is not None:
                    batch.append((fid, blob))
            with case.db.lock:
                case.db.conn.executemany("INSERT OR REPLACE INTO content_vecs VALUES (?, ?)", batch)
                case.db.conn.commit()
            if progress:
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
        # an exact duplicate that was not described itself uses its group's description
        row = case.db.get_file(file_id)
        others = [] if row is None or row["stack_id"] is None else [
            r[0] for r in case.db.conn.execute(
                "SELECT id FROM files WHERE stack_id = ? ORDER BY id", (row["stack_id"],))]
        found = [np.searchsorted(ids, o) for o in others]
        found = [k for k, o in zip(found, others) if k < len(ids) and ids[k] == o]
        if not found:
            return []
        at = found[0]
        exclude = set(exclude) | set(others)     # the group itself is listed as copies
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
