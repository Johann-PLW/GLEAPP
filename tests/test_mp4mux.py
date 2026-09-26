"""Putting a video-only and an audio-only MP4 in one file (gleapp/mp4mux.py).

Fixtures, all ffmpeg's test sources:

* ``fixtures/dash/``: a fragmented video and audio stream (see test_exocache_dash.py).
* ``fixtures/mux/video.mp4``: 2 s of testsrc, H.264, ``moov`` after ``mdat``;
  ``fixtures/mux/audio.m4a``: 2 s of a 660 Hz tone, AAC, ``-movflags +faststart``
  (``moov`` first). Commands:

      ffmpeg -f lavfi -i testsrc=size=160x120:rate=10 -t 2 -c:v libx264 -pix_fmt yuv420p -an video.mp4
      ffmpeg -f lavfi -i sine=frequency=660:sample_rate=22050 -t 2 -c:a aac -b:a 32k -vn \\
          -movflags +faststart audio.m4a

The checks read the output's own boxes rather than trust the writer: both tracks are
declared, every input fragment's media bytes appear unchanged, and every rewritten
chunk offset points at the bytes it pointed at before. Where ffmpeg is installed, the
per-frame MD5 of each stream is also compared with the input's.
"""

import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from gleapp import mp4mux
from gleapp.exocache import track_handlers

FIX = Path(__file__).parent / "fixtures"
DASH_V = ["init-0.mp4", "seg-0-1.m4s", "seg-0-2.m4s", "seg-0-3.m4s"]
DASH_A = ["init-1.mp4", "seg-1-1.m4s", "seg-1-2.m4s", "seg-1-3.m4s", "seg-1-4.m4s"]


def _cat(tmp_path, names, out):
    p = tmp_path / out
    p.write_bytes(b"".join((FIX / "dash" / n).read_bytes() for n in names))
    return p


def _boxes(b, start=0, end=None):
    end = len(b) if end is None else end
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", b[i:i + 8])
        hdr = 8
        if size == 1:
            size, hdr = struct.unpack(">Q", b[i + 8:i + 16])[0], 16
        yield typ, i, hdr, i + size
        i += size


def _find_all(b, typ, start=0, end=None, into=(b"moov", b"trak", b"mdia", b"minf", b"stbl")):
    for t, s, h, e in _boxes(b, start, end):
        if t == typ:
            yield s, h, e
        if t in into:
            yield from _find_all(b, typ, s + h, e, into)


def _chunk_offsets(b):
    """Per track, in order: every chunk offset its stco/co64 records."""
    out = []
    for s, h, e in _find_all(b, b"stbl"):
        for t, s2, h2, _e2 in _boxes(b, s + h, e):
            if t in (b"stco", b"co64"):
                n = struct.unpack(">I", b[s2 + h2 + 4:s2 + h2 + 8])[0]
                w = 8 if t == b"co64" else 4
                fmt = ">Q" if w == 8 else ">I"
                base = s2 + h2 + 8
                out.append([struct.unpack(fmt, b[base + k * w:base + (k + 1) * w])[0] for k in range(n)])
    return out


def _framemd5(path, sel):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:{sel}:0",
                        "-f", "framemd5", "-"], capture_output=True, text=True, check=False)
    return [ln.rsplit(",", 1)[-1].strip() for ln in r.stdout.splitlines() if ln and not ln.startswith("#")]


def test_fragmented_streams_become_one_file_with_both_tracks(tmp_path):
    v, a = _cat(tmp_path, DASH_V, "v.mp4"), _cat(tmp_path, DASH_A, "a.mp4")
    out = tmp_path / "av.mp4"
    got = mp4mux.mux(v, a, out)
    data = out.read_bytes()
    assert got["layout"] == "fragmented" and got["fragments"] == 7
    assert track_handlers(data) == ["vide", "soun"]
    # every input fragment's media bytes, unchanged
    for src in (v, a):
        b = src.read_bytes()
        for t, s, _h, e in _boxes(b):
            if t == b"mdat":
                assert b[s:e] in data
    # both tracks named in the fragments, and the sequence renumbered from 1
    tfhd_ids, seqs = [], []
    for t, s, h, e in _boxes(data):
        if t == b"moof":
            for t2, s2, h2, e2 in _boxes(data, s + h, e):
                if t2 == b"mfhd":
                    seqs.append(struct.unpack(">I", data[s2 + h2 + 4:s2 + h2 + 8])[0])
                if t2 == b"traf":
                    for t3, s3, h3, _e3 in _boxes(data, s2 + h2, e2):
                        if t3 == b"tfhd":
                            tfhd_ids.append(struct.unpack(">I", data[s3 + h3 + 4:s3 + h3 + 8])[0])
    assert seqs == list(range(1, 8)) and set(tfhd_ids) == {1, 2}


def test_progressive_files_keep_every_chunk_offset_pointing_at_the_same_bytes(tmp_path):
    v, a = FIX / "mux" / "video.mp4", FIX / "mux" / "audio.m4a"
    out = tmp_path / "av.mp4"
    assert mp4mux.mux(v, a, out)["layout"] == "progressive"
    data, vb, ab = out.read_bytes(), v.read_bytes(), a.read_bytes()
    assert track_handlers(data) == ["vide", "soun"]
    new = _chunk_offsets(data)
    for old_bytes, old, now in ((vb, _chunk_offsets(vb)[0], new[0]), (ab, _chunk_offsets(ab)[0], new[1])):
        assert len(old) == len(now) > 0
        for o, n in zip(old, now):
            assert data[n:n + 64] == old_bytes[o:o + 64]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg as an independent reader")
@pytest.mark.parametrize("layout", ["fragmented", "progressive"])
def test_ffmpeg_reads_the_same_frames_from_the_combined_file(tmp_path, layout):
    if layout == "fragmented":
        v, a = _cat(tmp_path, DASH_V, "v.mp4"), _cat(tmp_path, DASH_A, "a.mp4")
    else:
        v, a = FIX / "mux" / "video.mp4", FIX / "mux" / "audio.m4a"
    out = tmp_path / "av.mp4"
    mp4mux.mux(v, a, out)
    assert _framemd5(out, "v") == _framemd5(v, "v") and len(_framemd5(v, "v")) > 0
    assert _framemd5(out, "a") == _framemd5(a, "a") and len(_framemd5(a, "a")) > 0


def test_what_cannot_be_combined_is_refused(tmp_path):
    v, a = _cat(tmp_path, DASH_V, "v.mp4"), _cat(tmp_path, DASH_A, "a.mp4")
    with pytest.raises(mp4mux.MuxError, match="video and the second audio"):
        mp4mux.mux(a, v, tmp_path / "x.mp4")
    with pytest.raises(mp4mux.MuxError, match="fragmented and the other is not"):
        mp4mux.mux(v, FIX / "mux" / "audio.m4a", tmp_path / "x.mp4")
    with pytest.raises(mp4mux.MuxError, match="no moov"):
        mp4mux.mux(FIX / "dash" / "seg-0-1.m4s", a, tmp_path / "x.mp4")
    two = tmp_path / "two.mp4"
    mp4mux.mux(FIX / "mux" / "video.mp4", FIX / "mux" / "audio.m4a", two)
    with pytest.raises(mp4mux.MuxError, match="2 tracks"):
        mp4mux.mux(two, FIX / "mux" / "audio.m4a", tmp_path / "x.mp4")


def test_movie_timescale_durations_are_moved_to_the_videos():
    """ffmpeg writes both fixtures at one movie timescale, so this is exercised directly."""
    elst = bytes(4) + struct.pack(">I", 1) + struct.pack(">IiI", 2000, 0, 0x10000)
    got = mp4mux._elst(elst, 1000, 600)          # pylint: disable=protected-access
    assert struct.unpack(">I", got[8:12])[0] == 1200
    tkhd = bytes(4) + bytes(8) + struct.pack(">I", 5) + bytes(4) + struct.pack(">I", 3000) + bytes(60)
    got = mp4mux._tkhd(tkhd, 2, 1000, 600)       # pylint: disable=protected-access
    assert struct.unpack(">II", got[12:16] + got[20:24]) == (2, 1800)


def test_an_absolute_base_data_offset_follows_its_fragment():
    """A tfhd with base-data-offset-present counts from the file, not the moof."""
    tfhd = struct.pack(">I4s", 24, b"tfhd") + struct.pack(">II", 0x000001, 1) + struct.pack(">Q", 5000)
    moof = struct.pack(">I4s", 8 + 8 + len(tfhd), b"moof") + struct.pack(">I4s", 8 + len(tfhd), b"traf") + tfhd
    out = mp4mux._renumber(moof, 7, new_start=300, old_start=4000)   # pylint: disable=protected-access
    base = out.index(b"tfhd") + 4 + 8
    assert struct.unpack(">Q", out[base:base + 8])[0] == 5000 - 4000 + 300
