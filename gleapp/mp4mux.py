"""Put a video-only MP4 and an audio-only MP4 into one file, without re-encoding.

DASH serves a video and its sound as separate streams, so the ExoPlayer cache holds
a silent video and a picture-less audio track. This joins them the way a muxer
would, by rewriting the ISO base media file boxes (ISO/IEC 14496-12) that describe
the tracks and copying every sample's bytes unchanged:

* **Fragmented** (the ``moov`` holds an ``mvex``, as a DASH initialization segment
  does): one ``moov`` with both tracks and both ``trex`` boxes, the audio track
  renumbered when it shares the video's id, followed by every ``moof`` + ``mdat``
  of both inputs, interleaved by decode time (``tfdt`` over the track's ``mdhd``
  timescale). A ``trun`` offset counts from its ``moof`` unless ``tfhd`` carries an
  absolute ``base_data_offset``, which is rewritten; ``mfhd`` sequence numbers are
  renumbered in the new order. ``styp``, ``sidx`` and the other per-segment boxes
  are left out: a ``sidx`` indexes one stream's fragments at their old offsets.
* **Progressive** (a ``moov`` with sample tables): one ``moov`` with both tracks,
  then one ``mdat`` holding both input files whole. Each track's chunk offsets
  (``stco``, rewritten as ``co64``) are shifted by where its file now starts, so
  they still point at the same bytes. The input ``moov`` boxes ride along inside the
  ``mdat`` unread, which costs their size and nothing else.

Durations that are counted in the movie timescale (``tkhd``, ``elst``) are moved to
the video's. Encrypted tracks (a ``sinf`` box) and inputs with more than one track
are refused: a key cannot be carried over, and which track to take would be a guess.
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path


class MuxError(ValueError):
    """The two inputs cannot be combined; the message says why."""


def _boxes(b: bytes, start: int = 0, end: int | None = None):
    """``(type, start, header length, end)`` for each box in ``b[start:end]``."""
    end = len(b) if end is None else end
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", b[i:i + 8])
        hdr = 8
        if size == 1:
            if i + 16 > end:
                raise MuxError("box header runs past the end")
            size, hdr = struct.unpack(">Q", b[i + 8:i + 16])[0], 16
        elif size == 0:
            size = end - i
        if size < hdr or i + size > end:
            raise MuxError(f"box {typ!r} runs past the end")
        yield typ, i, hdr, i + size
        i += size


def _box(typ: bytes, payload: bytes) -> bytes:
    if len(payload) + 8 <= 0xFFFFFFFF:
        return struct.pack(">I4s", len(payload) + 8, typ) + payload
    return struct.pack(">I4sQ", 1, typ, len(payload) + 16) + payload


def _child(b: bytes, parent, typ: bytes):
    _, s, h, e = parent
    for box in _boxes(b, s + h, e):
        if box[0] == typ:
            return box
    return None


def _path(b: bytes, parent, *types: bytes):
    box = parent
    for t in types:
        box = _child(b, box, t)
        if box is None:
            return None
    return box


def _payload(b: bytes, box) -> bytes:
    return b[box[1] + box[2]:box[3]]


def _top(b: bytes, typ: bytes):
    return next((x for x in _boxes(b) if x[0] == typ), None)


class _Movie:
    """The ``moov`` of one input, with the facts the mux needs."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.moov = _top(data, b"moov")
        if self.moov is None:
            raise MuxError("no moov box")
        traks = [x for x in _boxes(data, self.moov[1] + self.moov[2], self.moov[3]) if x[0] == b"trak"]
        if len(traks) != 1:
            raise MuxError(f"{len(traks)} tracks; one was expected")
        self.trak = traks[0]
        if _find_deep(data, self.trak, b"sinf"):
            raise MuxError("the track is encrypted")
        mvhd = _child(data, self.moov, b"mvhd")
        p = _payload(data, mvhd)
        self.timescale = struct.unpack(">I", p[20:24] if p[0] == 1 else p[12:16])[0]
        self.fragmented = _child(data, self.moov, b"mvex") is not None
        tkhd = _child(data, self.trak, b"tkhd")
        tp = _payload(data, tkhd)
        self.track_id = struct.unpack(">I", tp[20:24] if tp[0] == 1 else tp[12:16])[0]
        mdhd = _path(data, self.trak, b"mdia", b"mdhd")
        mp = _payload(data, mdhd)
        self.media_timescale = struct.unpack(">I", mp[20:24] if mp[0] == 1 else mp[12:16])[0]
        hdlr = _path(data, self.trak, b"mdia", b"hdlr")
        self.handler = _payload(data, hdlr)[8:12] if hdlr else b""


def _find_deep(b: bytes, parent, typ: bytes) -> bool:
    containers = {b"trak", b"mdia", b"minf", b"stbl", b"stsd", b"encv", b"enca", b"sinf"}
    _, s, h, e = parent
    start = s + h
    if parent[0] == b"stsd":
        start += 8
    elif parent[0] in (b"encv", b"enca"):
        return True
    try:
        for box in _boxes(b, start, e):
            if box[0] == typ:
                return True
            if box[0] in containers and _find_deep(b, box, typ):
                return True
    except MuxError:
        return False
    return False


# ---- rewriting the boxes that name a track or count in the movie timescale --

def _rescale(v: int, src: int, dst: int) -> int:
    return v if src == dst or src == 0 else v * dst // src


def _tkhd(p: bytes, track_id: int, src_ts: int, dst_ts: int) -> bytes:
    p = bytearray(p)
    if p[0] == 1:
        p[20:24] = struct.pack(">I", track_id)
        (d,) = struct.unpack(">Q", p[28:36])
        p[28:36] = struct.pack(">Q", _rescale(d, src_ts, dst_ts))
    else:
        p[12:16] = struct.pack(">I", track_id)
        (d,) = struct.unpack(">I", p[20:24])
        if d != 0xFFFFFFFF:
            p[20:24] = struct.pack(">I", min(_rescale(d, src_ts, dst_ts), 0xFFFFFFFE))
    return bytes(p)


def _elst(p: bytes, src_ts: int, dst_ts: int) -> bytes:
    p = bytearray(p)
    (n,) = struct.unpack(">I", p[4:8])
    wide = p[0] == 1
    step = 20 if wide else 12
    for k in range(n):
        o = 8 + k * step
        if wide:
            (d,) = struct.unpack(">Q", p[o:o + 8])
            p[o:o + 8] = struct.pack(">Q", _rescale(d, src_ts, dst_ts))
        else:
            (d,) = struct.unpack(">I", p[o:o + 4])
            p[o:o + 4] = struct.pack(">I", min(_rescale(d, src_ts, dst_ts), 0xFFFFFFFF))
    return bytes(p)


def _rebuild(b: bytes, box, fix) -> bytes:
    """``box`` re-serialised, with ``fix(type, payload)`` given the chance to replace
    any descendant: None keeps it, a payload replaces the payload, a ``(type,
    payload)`` pair replaces both."""
    typ, s, h, e = box
    containers = {b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"mvex", b"moov", b"moof", b"traf"}
    got = fix(typ, b[s + h:e])
    if isinstance(got, tuple):
        return _box(*got)
    if got is not None:
        return _box(typ, got)
    if typ not in containers:
        return b[s:e]
    return _box(typ, b"".join(_rebuild(b, c, fix) for c in _boxes(b, s + h, e)))


def _co64(typ: bytes, p: bytes, shift: int) -> tuple[bytes, bytes]:
    """An ``stco`` or ``co64`` payload as a ``co64`` with every offset moved by ``shift``."""
    (n,) = struct.unpack(">I", p[4:8])
    fmt, width = (">Q", 8) if typ == b"co64" else (">I", 4)
    offs = (struct.unpack(fmt, p[8 + k * width:8 + (k + 1) * width])[0] for k in range(n))
    return b"co64", bytes(4) + struct.pack(">I", n) + b"".join(struct.pack(">Q", o + shift) for o in offs)


def _trak(m: _Movie, track_id: int, dst_ts: int, offset_shift: int | None = None) -> bytes:
    """``m``'s track, renumbered to ``track_id``, its movie-timescale durations moved
    to ``dst_ts``, and with ``offset_shift`` its chunk offsets moved as a ``co64``."""
    def fix(typ: bytes, p: bytes):
        if typ == b"tkhd":
            return _tkhd(p, track_id, m.timescale, dst_ts)
        if typ == b"elst":
            return _elst(p, m.timescale, dst_ts)
        if offset_shift is not None and typ in (b"stco", b"co64"):
            return _co64(typ, p, offset_shift)
        return None

    return _rebuild(m.data, m.trak, fix)


def _mvhd(m: _Movie, next_id: int, duration: int | None = None) -> bytes:
    p = bytearray(_payload(m.data, _child(m.data, m.moov, b"mvhd")))
    p[-4:] = struct.pack(">I", next_id)
    if duration is not None:
        if p[0] == 1:
            p[24:32] = struct.pack(">Q", duration)
        else:
            p[16:20] = struct.pack(">I", min(duration, 0xFFFFFFFF))
    return _box(b"mvhd", bytes(p))


def _duration(m: _Movie, dst_ts: int) -> int:
    p = _payload(m.data, _child(m.data, m.moov, b"mvhd"))
    d = struct.unpack(">Q", p[24:32])[0] if p[0] == 1 else struct.unpack(">I", p[16:20])[0]
    return _rescale(d, m.timescale, dst_ts)


# ---- the two layouts --------------------------------------------------------

def _fragments(m: _Movie, track_id: int) -> list[tuple[float, bytes, int]]:
    """``(decode time in seconds, moof+mdat bytes, where the moof started)`` per fragment,
    the ``tfhd`` already naming ``track_id``."""
    out = []
    boxes = list(_boxes(m.data))
    for i, box in enumerate(boxes):
        if box[0] != b"moof":
            continue
        mdat = boxes[i + 1] if i + 1 < len(boxes) and boxes[i + 1][0] == b"mdat" else None
        if mdat is None:
            raise MuxError("a moof with no mdat after it")
        traf = _child(m.data, box, b"traf")
        tfdt = _child(m.data, traf, b"tfdt") if traf else None
        t = 0.0
        if tfdt:
            p = _payload(m.data, tfdt)
            v = struct.unpack(">Q", p[4:12])[0] if p[0] == 1 else struct.unpack(">I", p[4:8])[0]
            t = v / m.media_timescale
        def fix(typ: bytes, p: bytes):
            if typ == b"tfhd":
                return p[:4] + struct.pack(">I", track_id) + p[8:]
            return None

        out.append((t, _rebuild(m.data, box, fix) + m.data[mdat[1]:mdat[3]], box[1]))
    return out


def _renumber(chunk: bytes, seq: int, new_start: int, old_start: int) -> bytes:
    moof = next(_boxes(chunk))

    def fix(typ: bytes, p: bytes):
        if typ == b"mfhd":
            return p[:4] + struct.pack(">I", seq)
        if typ == b"tfhd":
            (fl,) = struct.unpack(">I", b"\x00" + p[1:4])
            if fl & 0x1:
                (base,) = struct.unpack(">Q", p[8:16])
                return p[:8] + struct.pack(">Q", base - old_start + new_start) + p[16:]
        return None

    out = _rebuild(chunk, moof, fix)
    return out + chunk[moof[3]:]


def mux(video: str | Path, audio: str | Path, dest: str | Path) -> dict:
    """Write ``video``'s track and ``audio``'s track into ``dest``. Returns what was
    done: the layout, the two track ids in the output, the fragments or the bytes
    carried. Raises :class:`MuxError` when the pair cannot be combined."""
    v = _Movie(Path(video).read_bytes())
    a = _Movie(Path(audio).read_bytes())
    if v.handler != b"vide" or a.handler != b"soun":
        raise MuxError("the first input must be video and the second audio")
    if v.fragmented != a.fragmented:
        raise MuxError("one input is fragmented and the other is not")
    vid, aid = v.track_id, v.track_id + 1 if a.track_id == v.track_id else a.track_id
    next_id = max(vid, aid) + 1
    ftyp = _top(v.data, b"ftyp")
    head = v.data[ftyp[1]:ftyp[3]] if ftyp else b""
    dest = Path(dest)
    if v.fragmented:
        trex = []
        for m, tid in ((v, vid), (a, aid)):
            box = _path(m.data, m.moov, b"mvex", b"trex")
            if box is None:
                raise MuxError("an mvex with no trex")
            p = bytearray(_payload(m.data, box))
            p[4:8] = struct.pack(">I", tid)
            trex.append(_box(b"trex", bytes(p)))
        mvex = _box(b"mvex", b"".join(trex))
        extra = b"".join(v.data[x[1]:x[3]] for x in _boxes(v.data, v.moov[1] + v.moov[2], v.moov[3])
                         if x[0] not in (b"mvhd", b"trak", b"mvex"))
        moov = _box(b"moov", _mvhd(v, next_id) + _trak(v, vid, v.timescale)
                    + _trak(a, aid, v.timescale) + mvex + extra)
        frags = sorted([(t, 0, c, s) for t, c, s in _fragments(v, vid)]
                       + [(t, 1, c, s) for t, c, s in _fragments(a, aid)],
                       key=lambda f: (f[0], f[1]))
        if not frags:
            raise MuxError("no fragments")
        pos = len(head) + len(moov)
        with open(dest, "wb") as out:
            out.write(head)
            out.write(moov)
            for seq, (_t, _which, chunk, old) in enumerate(frags, 1):
                chunk = _renumber(chunk, seq, pos, old)
                out.write(chunk)
                pos += len(chunk)
        return {"layout": "fragmented", "video_track": vid, "audio_track": aid,
                "fragments": len(frags), "bytes": pos}
    # progressive: sizes first, since the offsets depend on where the mdat starts
    extra = b"".join(v.data[x[1]:x[3]] for x in _boxes(v.data, v.moov[1] + v.moov[2], v.moov[3])
                     if x[0] not in (b"mvhd", b"trak"))
    duration = max(_duration(v, v.timescale), _duration(a, v.timescale))

    def build(vshift: int, ashift: int) -> bytes:
        return _box(b"moov", _mvhd(v, next_id, duration) + _trak(v, vid, v.timescale, vshift)
                    + _trak(a, aid, v.timescale, ashift) + extra)

    size = len(build(0, 0))
    vstart = len(head) + size + 16                      # a 64-bit mdat header
    astart = vstart + len(v.data)
    moov = build(vstart, astart)
    if len(moov) != size:
        raise MuxError("the header changed size")
    with open(dest, "wb") as out:
        out.write(head)
        out.write(moov)
        out.write(struct.pack(">I4sQ", 1, b"mdat", 16 + len(v.data) + len(a.data)))
        with open(video, "rb") as fh:
            shutil.copyfileobj(fh, out, 1 << 20)
        with open(audio, "rb") as fh:
            shutil.copyfileobj(fh, out, 1 << 20)
    return {"layout": "progressive", "video_track": vid, "audio_track": aid,
            "bytes": astart + len(a.data)}
