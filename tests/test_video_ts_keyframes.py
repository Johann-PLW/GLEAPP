"""Key frames from an MPEG transport stream, which OpenCV cannot seek by frame number.

``tests/fixtures/h264-segment.ts`` is ffmpeg's own test pattern, 160x120 at 10 fps for
2 s, H.264 in MPEG-TS, the way an HLS stream serves video:

    ffmpeg -f lavfi -i testsrc=size=160x120:rate=10 -t 2 -c:v libx264 \\
        -pix_fmt yuv420p -bsf:v h264_mp4toannexb -f mpegts h264-segment.ts

Reading it in order gives every frame; seeking to a frame number gave none, so the
key-frame sampler found nothing and processing called the file undecodable.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from gleapp import media
from gleapp.case import Source, open_case
from gleapp.pipeline import ingest_sources, process

TS = Path(__file__).parent / "fixtures" / "h264-segment.ts"


@pytest.fixture(autouse=True)
def _isolate_appconfig(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("GLEAPP_CONFIG_DIR", str(tmp_path_factory.mktemp("gleapp-cfg")))


def test_a_transport_stream_gives_the_frames_it_was_asked_for(tmp_path):
    frames = media.extract_keyframes(TS, tmp_path, count=4)
    assert len(frames) == 4
    stamps = [ts for ts, _name, _pil in frames]
    assert stamps == sorted(stamps) and stamps[0] == 0 and stamps[-1] <= 2.0
    assert all((tmp_path / name).is_file() for _ts, name, _pil in frames)
    # evenly spaced frames of a moving pattern, not one frame four times
    assert len({pil.tobytes() for _ts, _name, pil in frames}) == 4


def test_a_seekable_video_still_takes_the_seek_path(tmp_path, monkeypatch):
    p = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"),  # pylint: disable=no-member
                        10.0, (64, 48))
    for i in range(20):
        w.write(np.full((48, 64, 3), i * 12, np.uint8))
    w.release()
    opened = []
    real = cv2.VideoCapture  # pylint: disable=no-member
    monkeypatch.setattr(media.cv2, "VideoCapture", lambda *a: opened.append(a) or real(*a))
    assert len(media.extract_keyframes(p, tmp_path / "kf", count=4)) == 4
    assert len(opened) == 1          # no second, read-in-order pass


def test_processing_a_transport_stream_gives_a_thumbnail_and_no_error(tmp_path):
    src = tmp_path / "ev"
    src.mkdir()
    (src / "segment.ts").write_bytes(TS.read_bytes())
    case = open_case(tmp_path / "case", create=True, examiner="t")
    try:
        ingest_sources(case, [Source(name="ev", path=str(src))])
        process(case, workers=1, keyframes=3, screen=False)
        (row,) = case.db.iter_files("", ())
        assert row["kind"] == "video"
        assert row["thumb"] and not row["error"]
    finally:
        case.close()
