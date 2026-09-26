"""Joining a DASH stream's segments from an ExoPlayer cache (gleapp/exocache.py).

``tests/fixtures/dash/`` is ffmpeg's test pattern and a 440 Hz tone, 3 s, written as a
DASH SegmentList, the form Google Maps' cached manifests take:

    ffmpeg -f lavfi -i testsrc=size=160x120:rate=10 -f lavfi -i sine=frequency=440:sample_rate=22050 \\
        -t 3 -map 0:v -map 1:a -c:v libx264 -g 10 -keyint_min 10 -sc_threshold 0 -pix_fmt yuv420p \\
        -c:a aac -b:a 32k -f dash -seg_duration 1 -use_template 0 -use_timeline 0 \\
        -init_seg_name 'init-$RepresentationID$.mp4' -media_seg_name 'seg-$RepresentationID$-$Number$.m4s' \\
        stream.mpd

Representation 0 is the video (init-0.mp4, seg-0-1 to 3), 1 the audio (init-1.mp4,
seg-1-1 to 4). The cache built below holds each file as one cached item, keyed by the
address ExoPlayer would request it at, the way DashUtil.resolveCacheKey builds it.
"""

import json
import struct
from pathlib import Path

import pytest

from gleapp import exocache
from gleapp.case import Source, open_case
from gleapp.pipeline import ingest_sources, process

FIX = Path(__file__).parent / "fixtures" / "dash"
BASE = "https://cdn.example.net/dash/"
APP = "data/data/com.example.maps/cache/exo"
TS_MS = 1_700_000_000_000
VIDEO = ["init-0.mp4", "seg-0-1.m4s", "seg-0-2.m4s", "seg-0-3.m4s"]
AUDIO = ["init-1.mp4", "seg-1-1.m4s", "seg-1-2.m4s", "seg-1-3.m4s", "seg-1-4.m4s"]


@pytest.fixture(autouse=True)
def _isolate_appconfig(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("GLEAPP_CONFIG_DIR", str(tmp_path_factory.mktemp("gleapp-cfg")))


def _utf(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack(">H", len(b)) + b


def _exi(entries: dict) -> bytes:
    body = struct.pack(">i", len(entries))
    for cid, (key, length) in entries.items():
        body += struct.pack(">i", cid) + _utf(key) + struct.pack(">i", 1) + _utf("exo_len") \
            + struct.pack(">i", 8) + struct.pack(">q", length)
    return struct.pack(">ii", 2, 0) + body + struct.pack(">i", 0)


def _cache(files: dict[str, bytes], *, reverse_ids: bool = False) -> dict[str, bytes]:
    """One cached item per file, keyed by its address, ids in the given or reverse order."""
    names = list(files)
    ids = range(len(names), 0, -1) if reverse_ids else range(1, len(names) + 1)
    out, idx = {}, {}
    for cid, name in zip(ids, names):
        out[f"{APP}/{cid % 10}/{cid}.0.{TS_MS + cid}.v3.exo"] = files[name]
        idx[cid] = (BASE + name, len(files[name]))
    out[f"{APP}/{exocache.INDEX_NAME}"] = _exi(idx)
    return out


def _fixture(drop: tuple = (), extra: dict | None = None) -> dict[str, bytes]:
    files = {n: (FIX / n).read_bytes() for n in ["stream.mpd", *VIDEO, *AUDIO] if n not in drop}
    files.update(extra or {})
    return files


def _ingest(tmp_path, files, *, include_other=False, reverse_ids=False):
    src = tmp_path / "ext"
    for name, data in _cache(files, reverse_ids=reverse_ids).items():
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_bytes(data)
    case = open_case(tmp_path / "case", create=True, examiner="t")
    ingest_sources(case, [Source(name="ext", path=str(src), include_other=include_other)])
    return case


def _streams(case):
    return {json.loads(r["cache_info"])["track"]: r
            for r in case.db.iter_files("cache_info IS NOT NULL", ())
            if json.loads(r["cache_info"]).get("dash")}


def test_a_stream_is_its_init_and_media_segments_in_the_manifests_order(tmp_path):
    case = _ingest(tmp_path, _fixture(), reverse_ids=True)
    try:
        streams = _streams(case)
        assert set(streams) == {"video"}                  # audio needs include_other
        row = streams["video"]
        info = json.loads(row["cache_info"])
        assert Path(row["path"]).read_bytes() == b"".join((FIX / n).read_bytes() for n in VIDEO)
        assert (row["kind"], row["ext"]) == ("video", ".mp4")
        assert (info["segments_joined"], info["segments_listed"], info["state"]) == (3, 3, "complete")
        assert info["key"] == BASE + "stream.mpd"
        assert info["representation"]["codecs"] == "avc1.64000a"
        assert "other_track" not in info                  # the audio is not in the case
        # the segments the stream holds are not registered again on their own; the
        # only other row is the video combined with its audio
        rows = case.db.iter_files("cache_info IS NOT NULL", ())
        assert len(rows) == 2
        (av,) = [r for r in rows if json.loads(r["cache_info"]).get("combined")]
        got = json.loads(av["cache_info"])
        assert (av["kind"], got["layout"]) == ("video", "fragmented")
        assert got["video"] == row["orig_name"] and "not kept on its own" in got["audio"]
        assert (got["video_segments"], got["audio_segments"]) == ("3 of 3", "4 of 4")
        assert exocache.track_handlers(Path(av["path"]).read_bytes()) == ["vide", "soun"]
    finally:
        case.close()


def test_with_other_files_the_audio_is_kept_as_audio_and_each_names_the_other(tmp_path):
    case = _ingest(tmp_path, _fixture(), include_other=True)
    try:
        streams = _streams(case)
        audio, video = streams["audio"], streams["video"]
        assert (audio["kind"], audio["ext"]) == ("other", ".m4a")
        assert Path(audio["path"]).read_bytes() == b"".join((FIX / n).read_bytes() for n in AUDIO)
        assert json.loads(audio["cache_info"])["segments_joined"] == 4
        assert json.loads(video["cache_info"])["other_track"] == audio["orig_name"]
        assert json.loads(audio["cache_info"])["other_track"] == video["orig_name"]
        # processing re-checks an 'other' file by its bytes; it must stay audio
        process(case, workers=1, keyframes=2, screen=False)
        assert case.db.get_file(audio["id"])["kind"] == "other"
    finally:
        case.close()


def test_a_missing_segment_ends_the_stream_there(tmp_path):
    case = _ingest(tmp_path, _fixture(drop=("seg-0-2.m4s",)))
    try:
        info = json.loads(_streams(case)["video"]["cache_info"])
        assert (info["segments_joined"], info["segments_listed"], info["state"]) == (1, 3, "partial")
    finally:
        case.close()


def test_a_look_alike_segment_the_manifest_does_not_list_is_not_joined(tmp_path):
    extra = {"seg-0-4.m4s": (FIX / "seg-0-3.m4s").read_bytes()}
    case = _ingest(tmp_path, _fixture(extra=extra))
    try:
        info = json.loads(_streams(case)["video"]["cache_info"])
        assert info["segments_joined"] == 3 and len(info["cache_ids"]) == 4
    finally:
        case.close()


def test_without_the_manifest_nothing_is_joined_as_a_stream(tmp_path):
    case = _ingest(tmp_path, _fixture(drop=("stream.mpd",)))
    try:
        assert not _streams(case)
    finally:
        case.close()


def test_the_joined_stream_plays(tmp_path):
    case = _ingest(tmp_path, _fixture())
    try:
        process(case, workers=1, keyframes=3, screen=False)
        row = case.db.get_file(_streams(case)["video"]["id"])
        assert row["thumb"] and not row["error"]
        assert row["duration"] == pytest.approx(3.0, abs=0.2)
        assert (row["width"], row["height"]) == (160, 120)
    finally:
        case.close()


def _combined(case):
    return [r for r in case.db.iter_files("cache_info IS NOT NULL", ())
            if json.loads(r["cache_info"]).get("combined")]


def test_the_combined_file_plays_with_both_tracks(tmp_path):
    case = _ingest(tmp_path, _fixture(), include_other=True)
    try:
        (av,) = _combined(case)
        assert json.loads(av["cache_info"])["audio"] == _streams(case)["audio"]["orig_name"]
        process(case, workers=1, keyframes=2, screen=False)
        row = case.db.get_file(av["id"])
        assert row["thumb"] and not row["error"]
        assert row["duration"] == pytest.approx(3.0, abs=0.2)
    finally:
        case.close()


def test_two_audio_streams_are_not_guessed_between(tmp_path):
    """A second, different audio stream (a second language, say): which one belongs
    with the video would be a guess, so neither is combined with it."""
    mpd = (FIX / "stream.mpd").read_text()
    start = mpd.index('<AdaptationSet id="1"')
    block = mpd[start:mpd.index("</AdaptationSet>", start) + len("</AdaptationSet>")]
    other = (block.replace('AdaptationSet id="1"', 'AdaptationSet id="2"')
             .replace('Representation id="1"', 'Representation id="9"')
             .replace("init-1.mp4", "init-9.mp4").replace("seg-1-", "seg-9-"))
    extra = {"stream.mpd": mpd.replace("</Period>", other + "</Period>").encode()}
    for n in AUDIO:
        extra[n.replace("init-1", "init-9").replace("seg-1-", "seg-9-")] = (FIX / n).read_bytes()
    case = _ingest(tmp_path, _fixture(extra=extra))
    try:
        assert not _combined(case)
        assert set(_streams(case)) == {"video"}
    finally:
        case.close()


def test_a_second_pass_adds_nothing(tmp_path):
    case = _ingest(tmp_path, _fixture(), include_other=True)
    try:
        before = sorted(r["id"] for r in case.db.iter_files("cache_info IS NOT NULL", ()))
        assert exocache.assemble(case, include_other=True) == 0
        assert exocache.assemble(case, include_other=True, force=True) == len(before)
        assert sorted(r["id"] for r in case.db.iter_files("cache_info IS NOT NULL", ())) == before
    finally:
        case.close()


def test_whole_file_streams_are_paired_by_their_manifest(tmp_path):
    """A SegmentBase stream is one address fetched by byte range, so one cached item."""
    video = b"".join((FIX / n).read_bytes() for n in VIDEO)
    audio = b"".join((FIX / n).read_bytes() for n in AUDIO)
    mpd = (b'<?xml version="1.0"?><MPD xmlns="urn:mpeg:dash:schema:mpd:2011"><Period>'
           b'<AdaptationSet contentType="video"><Representation id="v" mimeType="video/mp4">'
           b'<BaseURL>DASH_120.mp4</BaseURL><SegmentBase indexRange="0-1"/></Representation>'
           b'</AdaptationSet><AdaptationSet contentType="audio"><Representation id="a" '
           b'mimeType="audio/mp4"><BaseURL>DASH_audio.mp4</BaseURL><SegmentBase indexRange="0-1"/>'
           b'</Representation></AdaptationSet></Period></MPD>')
    files = {"stream.mpd": mpd, "DASH_120.mp4": video, "DASH_audio.mp4": audio}
    for include in (False, True):
        case = _ingest(tmp_path / str(include), files, include_other=include)
        try:
            rows = {json.loads(r["cache_info"])["representation"]["id"]: r
                    for r in case.db.iter_files("cache_info IS NOT NULL", ())
                    if json.loads(r["cache_info"]).get("manifest_key")}
            assert rows["v"]["kind"] == "video"
            listed = json.loads(rows["v"]["cache_info"]).get("listed_with", [])
            if include:
                assert (rows["a"]["kind"], json.loads(rows["a"]["cache_info"])["track"]) == ("other", "audio")
                assert listed == [rows["a"]["orig_name"]]
            else:
                assert set(rows) == {"v"} and listed == []
            # the manifest pairs them, so the video is also combined with its audio
            (av,) = _combined(case)
            assert json.loads(av["cache_info"])["video"] == rows["v"]["orig_name"]
            assert exocache.track_handlers(Path(av["path"]).read_bytes()) == ["vide", "soun"]
        finally:
            case.close()


def test_track_handlers_reads_the_init_segment():
    assert exocache.track_handlers((FIX / "init-0.mp4").read_bytes()) == ["vide"]
    assert exocache.track_handlers((FIX / "init-1.mp4").read_bytes()) == ["soun"]
    assert exocache.track_handlers(b"not a box") == []


def test_describe_a_stream(tmp_path):
    case = _ingest(tmp_path, _fixture(drop=("seg-0-3.m4s",)))
    try:
        text = exocache.describe(dict(_streams(case)["video"]))
        assert "DASH stream" in text and "com.example.maps" in text
        assert "2 of 3 listed media segments joined" in text and "partial" in text
        assert "160x120" in text and BASE + "stream.mpd" in text
    finally:
        case.close()


def test_describe_a_combined_file(tmp_path):
    case = _ingest(tmp_path, _fixture())
    try:
        (av,) = _combined(case)
        text = exocache.describe(dict(av))
        assert "DASH video and audio" in text and "no sample re-encoded" in text
        assert "video segments 3 of 3, audio segments 4 of 4" in text
    finally:
        case.close()


def test_a_whole_file_video_cut_short_is_not_combined(tmp_path):
    """Reddit's whole-file streams can be cached only in part; combining one that
    stops at a gap would add a second broken file, so it is left alone."""
    video = b"".join((FIX / n).read_bytes() for n in VIDEO)
    audio = b"".join((FIX / n).read_bytes() for n in AUDIO)
    mpd = (b'<?xml version="1.0"?><MPD xmlns="urn:mpeg:dash:schema:mpd:2011"><Period>'
           b'<AdaptationSet><Representation id="v" mimeType="video/mp4"><BaseURL>v.mp4</BaseURL>'
           b'<SegmentBase indexRange="0-1"/></Representation></AdaptationSet><AdaptationSet>'
           b'<Representation id="a" mimeType="audio/mp4"><BaseURL>a.mp4</BaseURL>'
           b'<SegmentBase indexRange="0-1"/></Representation></AdaptationSet></Period></MPD>')
    src = tmp_path / "ext"
    files = {"stream.mpd": mpd, "v.mp4": video, "a.mp4": audio}
    cache = _cache(files)
    # the index records the video's full length, but only its init segment and first
    # media segment are cached: a clean fragment boundary, which would mux, and short
    cut = len((FIX / VIDEO[0]).read_bytes()) + len((FIX / VIDEO[1]).read_bytes())
    vpiece = next(k for k, v in cache.items() if v == video)
    cache[vpiece] = video[:cut]
    for name, data in cache.items():
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_bytes(data)
    case = open_case(tmp_path / "case", create=True, examiner="t")
    try:
        ingest_sources(case, [Source(name="ext", path=str(src))])
        assert not _combined(case)
    finally:
        case.close()
