"""Find similar, copies: the same picture resized, recompressed, cropped, mirrored, rotated,
recolored, captioned, bordered, watermarked or pasted into a screenshot.

Two stages, both working from the thumbnails processing already made:

1. Shortlist. Two independent views of every image, each keeping its best 150:
   * fingerprints: a 256-bit pHash of the picture, of the picture with solid borders
     trimmed, and of five 75% windows of it; the searched image is also tried mirrored
     and at 90, 180 and 270 degrees;
   * points: the visual words of up to 300 SIFT points on the thumbnail, which find a
     picture inside another one (a screenshot, a captioned repost), where a whole-image
     fingerprint cannot.
2. Confirmation. SIFT points of the two thumbnails are matched one-to-one and must fit
   one shape-preserving transform (move, scale, rotate; the mirrored search covers
   flips): at least MIN_POINTS of them, covering at least MIN_COVER of either image.
   An image with too little detail for points (a gradient, a flat wallpaper) is
   confirmed instead by a strict fingerprint match.

Measured through find_copies on 300 photos from a real 26,586-image phone case, each
edited 15 ways and searched among the whole case: of the 262 photos with detail, 95.5%
of the edited copies found (82% of those pasted into a phone screenshot, 92-99% for
every other edit); of the 38 smooth wallpapers, gradients and flat fills, 36%, since a
near-uniform picture gives neither points nor a stable fingerprint. In a hand-checked
sample the other files returned were related pictures already in the case (versions
of the same page, the same map with different overlays, light and dark versions of
one icon). Index: 10.6 MB and 6 minutes for 31,086 images; a search took 1.2 s
typically and at most about 8 s.

The index is small (about 400 bytes an image) and lives in the case database in two
tables of its own; an index written by another version is discarded and rebuilt.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .case import Case

INDEX_VERSION = "1"
FP_SIDE = 160           # fingerprints are taken from the thumbnail at this long side
WORD_POINTS = 300       # SIFT points per thumbnail for the points shortlist
CHECK_POINTS = 500      # SIFT points per thumbnail for the confirmation (800 found the same
                        # copies on the 300-photo test and a phone case's videos, 20-25% slower)
SHORTLIST = 150         # from each of the two shortlists
MIN_POINTS = 10         # confirmation: aligned one-to-one points...
MIN_COVER = 0.05        # ...covering at least this share of either image
MIN_BOTH = 0.02         # ...and at least this share of the other one: a logo or watermark
# matched whole inside a video frame is not a copy of the video (measured on a real phone
# case: the TikTok logo and a 'TikTok Shop' banner were 4 'copies' of two TikTok videos at 0,
# none at 0.02, with every copy of the 300-photo test still found; 0.05 lost 10% of the
# photos pasted into a screenshot)
FEW_POINTS = 25         # fewer points than this: confirm by fingerprint instead
FEW_FP = 24             # ...at most this many of 256 bits apart
RATIO = 0.8
TOP, LEAF = 64, 64      # vocabulary: 64 x 64 = 4,096 visual words
_POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def _bits(x: np.ndarray) -> np.ndarray:
    """Set bits of each 256-bit fingerprint in ``x`` (uint64, last axis of 4), summed.
    NumPy 2 counts a 64-bit word at a time; older NumPy falls back to a byte table.
    Measured: the whole-case fingerprint comparison of one picture fell from about
    0.19 s to a few hundredths on a 14,000-picture case."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).sum(-1, dtype=np.int32)
    b = np.ascontiguousarray(x).view(np.uint8)
    return _POP[b].reshape(*x.shape[:-1], -1).sum(-1, dtype=np.int32)

_CACHE: dict = {}
_SIFT = threading.local()


# ---- storage ---------------------------------------------------------------------
def _ensure(conn) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS sim_items (file_id INTEGER PRIMARY KEY "
                 "REFERENCES files(id) ON DELETE CASCADE, fp BLOB NOT NULL, "
                 "words BLOB, npts INTEGER NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS sim_vocab (id INTEGER PRIMARY KEY, "
                 "top BLOB NOT NULL, leaf BLOB NOT NULL)")
    row = conn.execute("SELECT value FROM meta WHERE key='sim_index_version'").fetchone()
    if row is None or row[0] != INDEX_VERSION:
        conn.execute("DELETE FROM sim_items")
        conn.execute("DELETE FROM sim_vocab")
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('sim_index_version', ?)",
                     (INDEX_VERSION,))
    conn.commit()


def _indexable_sql() -> str:
    # a video is indexed by its thumbnail (a frame from it); a video *search* also
    # checks each of its key frames, so a still taken from it is found too
    return "kind IN ('image', 'video') AND thumb IS NOT NULL"


def status(case: Case) -> dict:
    with case.db.lock:
        _ensure(case.db.conn)
        q = lambda sql: case.db.conn.execute(sql).fetchone()[0]
        return {"indexed": q("SELECT COUNT(*) FROM sim_items"),
                "indexable": q(f"SELECT COUNT(DISTINCT COALESCE(stack_id, id)) FROM files "
                               f"WHERE {_indexable_sql()}"),
                "vocab": bool(q("SELECT COUNT(*) FROM sim_vocab"))}


def ready(case: Case) -> bool:
    st = status(case)
    return st["vocab"] and st["indexed"] > 0 and st["indexed"] >= st["indexable"]


# ---- per-image features --------------------------------------------------------------
def _sift(n: int):
    import cv2
    key = f"s{n}"
    if not hasattr(_SIFT, key):
        setattr(_SIFT, key, cv2.SIFT_create(nfeatures=n))  # pylint: disable=no-member
    return getattr(_SIFT, key)


def _gray(path: Path) -> np.ndarray | None:
    from PIL import Image
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("L"), np.uint8)
    # pylint: disable-next=broad-exception-caught
    except Exception:  # noqa: BLE001 - a missing or unreadable thumbnail is skipped
        return None


def _phash256(im) -> np.ndarray:
    import imagehash
    return np.packbits(imagehash.phash(im, hash_size=16).hash.flatten())


def _trim(im, tol: float = 6.0):
    """The picture with solid-color borders (letterboxing, a frame) cut off."""
    a = np.asarray(im, np.int16)
    t, b, l, r = 0, a.shape[0], 0, a.shape[1]
    while b - t > 16 and a[t, l:r].std() < tol:
        t += 1
    while b - t > 16 and a[b - 1, l:r].std() < tol:
        b -= 1
    while r - l > 16 and a[t:b, l].std() < tol:
        l += 1
    while r - l > 16 and a[t:b, r - 1].std() < tol:
        r -= 1
    return im.crop((l, t, r, b))


def _windows(im):
    w, h = im.size
    ww, hh = int(w * 0.75), int(h * 0.75)
    for x0, y0 in ((0, 0), (w - ww, 0), (0, h - hh), (w - ww, h - hh),
                   ((w - ww) // 2, (h - hh) // 2)):
        yield im.crop((x0, y0, x0 + ww, y0 + hh))


def _fingerprints(gray: np.ndarray) -> np.ndarray:
    """7 x 32 bytes: the picture, the picture trimmed, five windows of the trimmed one."""
    from PIL import Image
    im = Image.fromarray(gray)
    im.thumbnail((FP_SIDE, FP_SIDE), Image.Resampling.LANCZOS)
    tr = _trim(im)
    return np.stack([_phash256(im), _phash256(tr)] + [_phash256(w) for w in _windows(tr)])


def _probes(gray: np.ndarray) -> np.ndarray:
    """The searched image's whole-picture fingerprints: as it is, trimmed, mirrored,
    and turned 90, 180 and 270 degrees (each also trimmed)."""
    from PIL import Image, ImageOps
    im = Image.fromarray(gray)
    im.thumbnail((FP_SIDE, FP_SIDE), Image.Resampling.LANCZOS)
    out = []
    for base in (im, _trim(im)):
        out += [_phash256(base), _phash256(ImageOps.mirror(base))]
        out += [_phash256(base.rotate(a, expand=True)) for a in (90, 180, 270)]
    return np.stack(out)


def _rootsift(desc: np.ndarray) -> np.ndarray:
    d = desc.astype(np.float32)
    d /= d.sum(axis=1, keepdims=True) + 1e-7
    return np.sqrt(d)


def _nearest(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return ((x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None]).argmin(1)


def _words(desc: np.ndarray, vocab) -> np.ndarray:
    top, leaf = vocab
    d = _rootsift(desc)
    b = _nearest(d, top)
    w = np.empty(len(d), np.int64)
    for t in np.unique(b):
        sel = b == t
        w[sel] = t * LEAF + _nearest(d[sel], leaf[t])
    return np.unique(w).astype(np.uint16)


# ---- build -----------------------------------------------------------------------------
def _kmeans(data: np.ndarray, k: int, seed: int) -> np.ndarray:
    import cv2
    k = max(1, min(k, len(data)))
    cv2.setRNGSeed(seed)  # pylint: disable=no-member
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1e-3)  # pylint: disable=no-member
    _, _, c = cv2.kmeans(data, k, None, crit, 1, cv2.KMEANS_PP_CENTERS)  # pylint: disable=no-member
    return c.astype(np.float32)


def _train(sample: np.ndarray):
    top = np.resize(_kmeans(sample, TOP, 0), (TOP, sample.shape[1]))
    lab = _nearest(sample, top)
    leaf = np.zeros((TOP, LEAF, sample.shape[1]), np.float32)
    for t in range(TOP):
        m = sample[lab == t]
        leaf[t] = np.resize(_kmeans(m, LEAF, t + 1) if len(m) else top[t:t + 1], (LEAF, sample.shape[1]))
    return top, leaf


def _vocab(conn):
    row = conn.execute("SELECT top, leaf FROM sim_vocab WHERE id = 1").fetchone()
    if row is None:
        return None
    top = np.frombuffer(row[0], np.float32).reshape(TOP, 128)
    return top, np.frombuffer(row[1], np.float32).reshape(TOP, LEAF, 128)


CHUNK = 256              # files a build handles between checks for a request to stop


def build_index(case: Case, *, workers: int = 6, progress=None, stage_cb=None,
                stop=None) -> int:
    """Index every picture and video with a thumbnail not yet indexed; returns how many.
    The first build learns the case's visual vocabulary from a sample of its thumbnails;
    later builds reuse it. Likely examiner material goes first, the system's and
    applications' own artwork last. ``stop`` (a callable) is checked between chunks of
    CHUNK files: what is written stays, and the next build picks up the rest."""
    from .content import SYSTEM_PATH_SQL
    say = stage_cb or (lambda m: None)
    stopped = stop or (lambda: False)
    conn = case.db.conn
    with case.db.lock:
        _ensure(conn)
        # exact duplicates share one entry (the group's lowest id): the same bytes make
        # the same thumbnail, and Find similar lists a file's exact duplicates anyway
        rows = conn.execute(
            f"SELECT id, thumb FROM files WHERE id IN (SELECT MIN(id) FROM files WHERE "
            f"{_indexable_sql()} GROUP BY COALESCE(stack_id, id)) "
            "AND id NOT IN (SELECT file_id FROM sim_items) "
            f"ORDER BY {SYSTEM_PATH_SQL}, MAX(COALESCE(width, 0), COALESCE(height, 0)) DESC, id"
        ).fetchall()
        rows = [(r[0], r[1]) for r in rows]
        vocab = _vocab(conn)
    if not rows:
        return 0
    thumbs = case.thumb_dir

    def features(item, interruptible=True):
        fid, thumb = item
        if interruptible and stopped():
            return fid, None                 # not written: the next build does it
        g = _gray(thumbs / thumb)
        if g is None:
            return fid, None
        kps, desc = _sift(WORD_POINTS).detectAndCompute(g, None)
        # the point count only decides "fewer than FEW_POINTS" (confirm by fingerprint
        # instead); this pass's count answers that exactly, so no second pass is needed
        return fid, (_fingerprints(g), desc, len(kps))

    if vocab is None:
        say("Learning the case's visual vocabulary…")
        rng = np.random.default_rng(0)
        by_id = sorted(rows)                # the sample must not depend on the build order
        pick = [by_id[i] for i in rng.choice(len(by_id), min(len(by_id), 6000), replace=False)]
        parts = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            # the vocabulary sample is never cut short: a partial one would change it
            for _, f in ex.map(lambda it: features(it, interruptible=False), pick):
                if f is not None and f[1] is not None:
                    d = f[1]
                    parts.append(d[rng.choice(len(d), min(8, len(d)), replace=False)])
        if parts:
            vocab = _train(_rootsift(np.concatenate(parts)))
            with case.db.lock:
                conn.execute("INSERT OR REPLACE INTO sim_vocab VALUES (1, ?, ?)",
                             (vocab[0].tobytes(), vocab[1].tobytes()))
                conn.commit()

    say(f"Indexing {len(rows):,} files for Find similar (copies)…")
    total, n = len(rows), 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for k in range(0, total, CHUNK):
            if stopped():
                break
            batch = []
            for fid, f in ex.map(features, rows[k:k + CHUNK]):
                n += 1
                if f is not None:
                    fp, desc, npts = f
                    words = _words(desc, vocab).tobytes() if desc is not None and vocab else None
                    batch.append((fid, fp.tobytes(), words, npts))
            with case.db.lock:
                conn.executemany("INSERT OR REPLACE INTO sim_items VALUES (?,?,?,?)", batch)
                conn.commit()
            if progress:
                progress(n, total)
    _CACHE.clear()
    return n


# ---- search ----------------------------------------------------------------------------
def _loaded(case: Case):
    conn = case.db.conn
    key = (id(conn), conn.execute("SELECT COUNT(*), MAX(file_id) FROM sim_items").fetchone())
    if key not in _CACHE:
        rows = conn.execute("SELECT file_id, fp, words, npts FROM sim_items ORDER BY file_id").fetchall()
        ids = np.array([r[0] for r in rows], np.int64)
        fp = np.frombuffer(b"".join(r[1] for r in rows), np.uint8).reshape(len(rows), 7, 32) \
            if rows else np.zeros((0, 7, 32), np.uint8)
        npts = np.array([r[3] for r in rows], np.int64)
        wl = [np.frombuffer(r[2], np.uint16) if r[2] else np.zeros(0, np.uint16) for r in rows]
        owner = np.concatenate([np.full(len(w), i, np.int64) for i, w in enumerate(wl)]) if wl else np.zeros(0, np.int64)
        allw = np.concatenate(wl) if wl else np.zeros(0, np.uint16)
        df = np.bincount(allw, minlength=TOP * LEAF)
        norm = np.sqrt(np.maximum(1, [len(w) for w in wl])) if wl else np.zeros(0)
        _CACHE.clear()
        _CACHE[key] = (ids, fp, npts, wl, owner, allw, df, norm)
    return _CACHE[key]


_CHECK: dict = {}
_CHECK_LOCK = threading.Lock()


def _check_features(thumb_dir: Path, thumb: str, mirror: bool = False):
    key = (str(thumb_dir), thumb, mirror)
    with _CHECK_LOCK:
        if key in _CHECK:
            return _CHECK[key]
    g = _gray(thumb_dir / thumb)
    out = None
    if g is not None:
        if mirror:
            g = np.ascontiguousarray(g[:, ::-1])
        kps, desc = _sift(CHECK_POINTS).detectAndCompute(g, None)
        pts = np.array([k.pt for k in kps], np.float32) if kps else np.zeros((0, 2), np.float32)
        out = (pts, _rootsift(desc) if desc is not None else None, g.shape)
    with _CHECK_LOCK:
        if len(_CHECK) > 20000:
            _CHECK.clear()
        _CHECK[key] = out
    return out


def _confirm(a, b) -> tuple[int, float, float]:
    """(aligned one-to-one points, the larger and the smaller share of the two images
    they cover)."""
    import cv2
    if a is None or b is None or a[1] is None or b[1] is None or len(a[0]) < 8 or len(b[0]) < 8:
        return 0, 0.0, 0.0
    good: dict[int, tuple[float, int]] = {}
    for m in cv2.BFMatcher(cv2.NORM_L2).knnMatch(a[1], b[1], k=2):  # pylint: disable=no-member
        if len(m) == 2 and m[0].distance < RATIO * m[1].distance:
            prev = good.get(m[0].trainIdx)
            if prev is None or m[0].distance < prev[0]:
                good[m[0].trainIdx] = (m[0].distance, m[0].queryIdx)
    if len(good) < 6:
        return 0, 0.0, 0.0
    ci = np.array(list(good))
    qi = np.array([v[1] for v in good.values()])
    src, dst = a[0][qi], b[0][ci]
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,  # pylint: disable=no-member
                                          ransacReprojThreshold=3.0, maxIters=2000)
    if M is None:
        return 0, 0.0, 0.0
    inl = mask.ravel().astype(bool)
    if not 0.1 < math.hypot(M[0, 0], M[1, 0]) < 10:
        return 0, 0.0, 0.0
    n = min(len(np.unique(np.round(src[inl]), axis=0)), len(np.unique(np.round(dst[inl]), axis=0)))

    def cover(p, shape):
        if len(p) < 3:
            return 0.0
        return float(cv2.contourArea(cv2.convexHull(p.astype(np.float32)))) / (shape[0] * shape[1])  # pylint: disable=no-member
    ca, cb = cover(src[inl], a[2]), cover(dst[inl], b[2])
    return n, max(ca, cb), min(ca, cb)


def _passes(r: tuple) -> bool:
    """A confirmation result (points, larger cover, smaller cover) that makes a copy."""
    return r[0] >= MIN_POINTS and r[1] >= MIN_COVER and r[2] >= MIN_BOTH


SAME_FRAME = 20          # key frames this close (bits of 256) to one already searched are skipped


def find_copies(case: Case, file_id: int, *, limit: int = 300, first_frame_only: bool = False) -> list[dict]:
    """Copies of ``file_id``, strongest first; the file itself first. Each result
    carries ``points`` (aligned points; None when confirmed by fingerprint) and
    ``similarity``, a display figure: how much of the picture the match covers. A video
    is searched by its thumbnail and by each of its key frames that differs from those
    already searched; each candidate is checked against the frames that shortlisted it."""
    target = case.db.get_file(file_id)
    if target is None or not target["thumb"]:
        return [dict(target, similarity=100.0, match="query")] if target else []
    with case.db.lock:
        _ensure(case.db.conn)
        vocab = _vocab(case.db.conn)
        loaded = _loaded(case)
        thumbs = {r[0]: r[1] for r in case.db.conn.execute(
            f"SELECT id, thumb FROM files WHERE {_indexable_sql()}")}
    out = [dict(target, similarity=100.0, match="query", points=None)]
    ids = loaded[0]
    if len(ids) == 0:
        return out
    frames = [target["thumb"]]
    if target["kind"] == "video" and not first_frame_only:
        frames += [k["thumb"] for k in case.db.keyframes_for(file_id) if k["thumb"]]
    # stage 1 for every distinct frame; which frames shortlisted which candidate
    by_cand: dict[int, list] = {}
    kept: list = []
    for frame in dict.fromkeys(frames):
        g = _gray(case.thumb_dir / frame)
        if g is None:
            continue
        probes = _probes(g)
        if any(int(_POP[probes[0] ^ k[1][0]].sum()) <= SAME_FRAME for k in kept):
            continue                         # (nearly) the same picture as a frame already searched
        kept.append((frame, probes))
        for i in _shortlist(g, probes, file_id, vocab, loaded):
            by_cand.setdefault(i, []).append((frame, probes))
    # stage 2: each candidate against the frames that shortlisted it
    _, fp, npts = loaded[0], loaded[1], loaded[2]

    def check(item):
        i, fr = item
        fid = int(ids[i])
        th = thumbs.get(fid)
        if not th or fid == file_id:
            return None
        b = _check_features(case.thumb_dir, th)
        best = None
        for frame, probes in fr:
            q = _check_features(case.thumb_dir, frame)
            r = _confirm(q, b)
            if not _passes(r):               # a mirrored copy: try the frame mirrored
                r2 = _confirm(_check_features(case.thumb_dir, frame, mirror=True), b)
                r = r2 if _passes(r2) or r2[0] > r[0] else r
            if _passes(r):
                if best is None or (r[0], r[1]) > (best[0] or 0, best[1]):  # pylint: disable=unsubscriptable-object
                    best = (r[0], r[1])
                continue
            q_few = q is None or len(q[0]) < FEW_POINTS
            if best is None and (q_few or npts[i] < FEW_POINTS):
                d = int(_POP[fp[i, 0][None, :] ^ probes].sum(1).min())
                if d <= FEW_FP:
                    best = (None, 1.0 - d / 256)
        return None if best is None else (fid, best[0], best[1])

    with ThreadPoolExecutor(max_workers=6) as ex:
        found = [r for r in ex.map(check, by_cand.items()) if r is not None]
    found.sort(key=lambda r: (-(r[1] or 0), -r[2]))
    for fid, pts, cov in found[:max(limit - 1, 0)]:
        row = case.db.get_file(fid)
        if row is not None:
            out.append(dict(row, match="copy", points=pts,
                            similarity=round(100 * min(1.0, cov), 1)))
    return out


def _shortlist(g, probes, file_id: int, vocab, loaded) -> set:
    """Stage 1 for one picture: the index rows of its fingerprint and point shortlists."""
    ids, fp, _, _, owner, allw, df, norm = loaded
    f64 = np.ascontiguousarray(fp).view(np.uint64)          # rows x 7 x 4
    p64 = np.ascontiguousarray(probes).view(np.uint64)      # probes x 4
    # the others' picture, trimmed picture and windows against this picture's probes ...
    best = _bits(f64[:, :, None, :] ^ p64[None, None, :, :]).min(axis=(1, 2))
    own = np.ascontiguousarray(_fingerprints(g)).view(np.uint64)
    # ... and this picture's windows against the others' whole pictures
    best = np.minimum(best, _bits(f64[:, :2, None, :] ^ own[None, None, 2:, :]).min(axis=(1, 2)))
    self_i = np.searchsorted(ids, file_id)
    is_self = self_i < len(ids) and ids[self_i] == file_id
    if is_self:
        best[self_i] = 10 ** 6
    short = set(np.argsort(best, kind="stable")[:SHORTLIST].tolist())
    if vocab is not None:
        _, desc = _sift(WORD_POINTS).detectAndCompute(g, None)
        if desc is not None and len(allw):
            qw = _words(desc, vocab)
            idf = np.log(len(ids) / np.maximum(df, 1))
            hit = np.isin(allw, qw)
            sc = np.bincount(owner[hit], weights=idf[allw[hit]], minlength=len(ids)) / norm
            if is_self:
                sc[self_i] = -1
            top = np.argsort(-sc)[:SHORTLIST]
            short |= set(top[sc[top] > 0].tolist())
    if is_self:
        short.discard(int(self_i))
    return short
