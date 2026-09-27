"""Joining an HLS stream's segments from an ExoPlayer cache (gleapp/exocache.py, through the
vendored exoprobe's ``plan_streams``).

The fragmented-MP4 streams reuse ``fixtures/dash/`` (see test_exocache_dash.py), listed by
HLS playlists instead of a DASH manifest, which RFC 8216 allows with #EXT-X-MAP.
``fixtures/hls/ts-0.ts`` and ``ts-1.ts`` are 2 s of ffmpeg's test pattern and a 330 Hz
tone as two MPEG transport stream segments:

    ffmpeg -f lavfi -i testsrc=size=160x120:rate=10 -f lavfi -i sine=frequency=330:sample_rate=22050 \\
        -t 2 -c:v libx264 -g 10 -keyint_min 10 -sc_threshold 0 -pix_fmt yuv420p -c:a aac -b:a 32k \\
        -f hls -hls_time 1 -hls_list_size 0 -hls_segment_filename 'ts-%d.ts' ts.m3u8
"""

import json
from pathlib import Path

import pytest

from gleapp import exocache
from gleapp.pipeline import process
from gleapp.vendor import exoprobe
from test_exocache_dash import AUDIO, FIX, VIDEO, _ingest, _isolate_appconfig  # noqa: F401  pylint: disable=unused-import

HLS = Path(__file__).parent / "fixtures" / "hls"
HBASE = "https://video.example.net/amplify_video/42/pl/"


def _hls(*, audio_group: str = "aud") -> dict:
    def media(folder, init, segs):
        return ("#EXTM3U\n#EXT-X-VERSION:7\n" + f'#EXT-X-MAP:URI="{folder}/{init}"\n'
                + "".join(f"#EXTINF:1.0,\n{folder}/{u}\n" for u in segs) + "#EXT-X-ENDLIST\n").encode()
    master = ("#EXTM3U\n"
              f'#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="{audio_group}",NAME="English",LANGUAGE="en",URI="a/en.m3u8"\n'
              f'#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="{audio_group}",NAME="Spanish",LANGUAGE="es",URI="a/es.m3u8"\n'
              '#EXT-X-STREAM-INF:BANDWIDTH=40000,RESOLUTION=160x120,AUDIO="aud"\nv/video.m3u8\n').encode()
    items = {"master.m3u8": master, "v/video.m3u8": media("v", "init.mp4", ["s1.m4s", "s2.m4s", "s3.m4s"])}
    for n, f in zip(["init.mp4", "s1.m4s", "s2.m4s", "s3.m4s"], VIDEO):
        items["v/v/" + n] = (FIX / f).read_bytes()
    for lang in ("en", "es"):
        items[f"a/{lang}.m3u8"] = media(lang, "init.mp4", ["s1.m4s", "s2.m4s", "s3.m4s", "s4.m4s"])
        for n, f in zip(["init.mp4", "s1.m4s", "s2.m4s", "s3.m4s", "s4.m4s"], AUDIO):
            items[f"a/{lang}/" + n] = (FIX / f).read_bytes()
    return items


def _ingest_hls(tmp_path, items, **kw):
    """test_exocache_dash's cache builder keys each item by its BASE + name; point BASE at
    the HLS address for the duration of the ingest."""
    import test_exocache_dash as d   # pylint: disable=import-outside-toplevel
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(d, "BASE", HBASE)
        return _ingest(tmp_path, items, **kw)


def _rows(case):
    return [(dict(r), json.loads(r["cache_info"])) for r in case.db.iter_files("cache_info IS NOT NULL", ())]


def test_an_hls_video_joins_and_combines_with_its_audio_group(tmp_path):
    case = _ingest_hls(tmp_path, _hls())
    try:
        rows = _rows(case)
        (video, vi), = [(r, i) for r, i in rows if i.get("stream_format") == "HLS" and i["track"] == "video"]
        assert (video["kind"], video["ext"]) == ("video", ".mp4")
        assert Path(video["path"]).read_bytes() == b"".join((FIX / n).read_bytes() for n in VIDEO)
        assert video["orig_name"].startswith("exoplayer_hls_") and vi["key"] == HBASE + "master.m3u8"
        assert vi["key_from"].startswith("HLS playlist, cache item")
        (av, ai), = [(r, i) for r, i in rows if i.get("combined")]
        assert [t["lang"] for t in ai["audio_tracks"]] == ["en", "es"]
        assert exoprobe.file_handlers(av["path"]) == ["vide", "soun", "soun"]
        text = exocache.describe(av)
        assert "HLS video and audio" in text and "language en (as the playlist gives it)" in text
        assert "listed media segments joined in the playlist's order" in exocache.describe(video)
        # the segments a stream holds are not registered again one by one
        assert not [i for _r, i in rows if not i.get("dash") and not i.get("combined")
                    and (i.get("key") or "").endswith(".m4s")]
    finally:
        case.close()


def test_hls_audio_outside_the_videos_group_is_not_combined(tmp_path):
    case = _ingest_hls(tmp_path, _hls(audio_group="other"))
    try:
        assert not [i for _r, i in _rows(case) if i.get("combined")]
    finally:
        case.close()


def test_a_transport_stream_playlist_joins_and_plays(tmp_path):
    items = {"ts.m3u8": b"#EXTM3U\n#EXTINF:1.0,\nts-0.ts\n#EXTINF:1.0,\nts-1.ts\n#EXT-X-ENDLIST\n",
             "ts-1.ts": (HLS / "ts-1.ts").read_bytes(), "ts-0.ts": (HLS / "ts-0.ts").read_bytes()}
    case = _ingest_hls(tmp_path, items)
    try:
        (row, info), = [(r, i) for r, i in _rows(case) if i.get("stream_format") == "HLS"]
        assert (row["kind"], row["ext"]) == ("video", ".ts") and info["has_init"] is False
        assert Path(row["path"]).read_bytes() == (HLS / "ts-0.ts").read_bytes() + (HLS / "ts-1.ts").read_bytes()
        assert "2 of 2 listed media segments joined in the playlist's order" in exocache.describe(row)
        assert "initialization segment" not in exocache.describe(row)
        process(case, workers=1, keyframes=2, screen=False)
        got = case.db.get_file(row["id"])
        assert got["thumb"] and not got["error"]
        assert got["duration"] == pytest.approx(2.0, abs=0.3)
    finally:
        case.close()


def test_a_subtitles_playlist_is_left_alone(tmp_path):
    items = {"s0/sub.m3u8": b"#EXTM3U\n#EXTINF:7.0,\nsub.vtt\n#EXT-X-ENDLIST\n",
             "s0/sub.vtt": b"WEBVTT\n\n00:00:00.033 --> 00:00:07.040\nHello\n"}
    case = _ingest_hls(tmp_path, items, include_other=True)
    try:
        assert not [i for _r, i in _rows(case) if i.get("stream_format")]
    finally:
        case.close()


def test_a_second_pass_adds_no_hls_rows(tmp_path):
    case = _ingest_hls(tmp_path, _hls(), include_other=True)
    try:
        before = sorted(r["id"] for r, _i in _rows(case))
        assert exocache.assemble(case, include_other=True) == 0
        assert sorted(r["id"] for r, _i in _rows(case)) == before
    finally:
        case.close()


def test_an_hls_variant_with_its_own_sound_is_not_combined_with_the_audio_group(tmp_path):
    """A transport-stream variant carries video and sound together, so the master's audio
    renditions are alternatives to it, not a missing track."""
    items = {k: v for k, v in _hls().items() if not k.startswith("v/")}
    items["v/video.m3u8"] = b"#EXTM3U\n#EXTINF:1.0,\nts-0.ts\n#EXTINF:1.0,\nts-1.ts\n#EXT-X-ENDLIST\n"
    items["v/ts-0.ts"] = (HLS / "ts-0.ts").read_bytes()
    items["v/ts-1.ts"] = (HLS / "ts-1.ts").read_bytes()
    case = _ingest_hls(tmp_path, items)
    try:
        rows = _rows(case)
        assert [r["ext"] for r, i in rows if i.get("stream_format") == "HLS" and i["track"] == "video"] == [".ts"]
        assert not [i for _r, i in rows if i.get("combined")]
        # and no combination was even attempted
        (log,) = [a for a in case.db.iter_audit() if a["action"] == "rejoin-exoplayer-cache"]
        assert "not combined" not in log["detail"]
    finally:
        case.close()


def test_a_dash_stream_keeps_the_file_name_it_had_before_hls(tmp_path):
    """A case ingested before HLS was joined named a DASH stream's file from
    sha1(source, cache folder, 'dash', init id); a second pass must find the same file."""
    import hashlib   # pylint: disable=import-outside-toplevel
    from test_exocache_dash import _fixture   # pylint: disable=import-outside-toplevel
    case = _ingest(tmp_path, _fixture())
    try:
        (row, info), = [(r, i) for r, i in _rows(case) if i.get("dash") and i["track"] == "video"]
        want = hashlib.sha1(f"{row['source']}\0{info['cache_folder']}\0dash\0{info['cache_ids'][0]}".encode()).hexdigest()
        assert Path(row["path"]).name == want + ".mp4"
    finally:
        case.close()
