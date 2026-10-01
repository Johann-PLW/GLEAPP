"""What a walk copies into a case is what the volume holds, and no more.

A file's recorded size and the bytes a volume stores for it are two numbers. They
part company for a sparse file, whose unwritten ranges read as zeros and take no
room; for a cloud provider's online-only placeholder, which has a size and none of
its content; and for a file the volume holds under two names, which is one file.
Read naively, each of those is written out in full: a placeholder as zeros of the
recorded size, hashed as if they were the file.

These run against ``ntfs-windows.img.gz``, a volume Windows 11 wrote, beside
``ntfs-windows.known.tsv``, which is what Windows itself reported for each of its
files: length, attributes, size on disk, SHA-256, and whether it could read the
file at all with no cloud provider running. So the expected values here are
Windows's, not this code's. Both come from the vendored reader's own repository
(abrignoni/qnxprobe, ``tests/fixtures``), where the scripts that made them are.

What the fixture does not hold, and so nothing here settles: a placeholder with
part of its content on the volume (Windows would not make one without a provider
running), and an APFS file marked dataless.
"""

from __future__ import annotations

import errno
import gzip
import hashlib
import io
import os
import time
from pathlib import Path

import pytest

from gleapp import archive, nested
from gleapp.case import open_case, parse_source_spec
from gleapp.pipeline import ingest_sources, process

FIXTURES = Path(__file__).parent / "fixtures"
VOLUME = "lba128"                       # the fixture's one volume, named by where it starts
PLACEHOLDERS = {"CloudRoot/online_only_doc.pdf", "CloudRoot/online_only_photo.jpg",
                "CloudRoot/online_only_video.mp4"}
SPARSE = ("Pictures/sparse_photo.jpg", "sparse/all_hole.bin", "sparse/both_ends.bin",
          "sparse/leading_hole.bin", "sparse/trailing_hole.bin")


def _known() -> dict[str, dict]:
    """The manifest: what Windows reported for each file of the fixture."""
    out = {}
    with open(FIXTURES / "ntfs-windows.known.tsv", encoding="utf-8", newline="") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            path, length, attrs, on_disk, sha, windows = line.rstrip("\n").split("\t")
            out[path] = {"length": int(length), "attributes": int(attrs, 16),
                         "on_disk": int(on_disk), "sha256": None if sha == "-" else sha,
                         "windows": windows}
    return out


KNOWN = _known()
# Windows compressed five files with LZX, which the vendored reader names and does
# not decode. Their first bytes cannot be read either, so the walk leaves them out
# and counts them.
LZX = {p for p in KNOWN if p.startswith("wof/lzx/")}
READABLE = {p for p, k in KNOWN.items() if k["sha256"] and p not in LZX}


@pytest.fixture(scope="module", name="image")
def _image(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("windows-volume") / "ntfs-windows.img"
    out.write_bytes(gzip.decompress((FIXTURES / "ntfs-windows.img.gz").read_bytes()))
    return out


def _ingest(tmp_path, image, name="case", **kw):
    case = open_case(tmp_path / name, create=True, examiner="t")
    sources, _ = parse_source_spec(image)
    for key, val in kw.items():
        setattr(sources[0], key, val)
    ingest_sources(case, sources)
    return case, sources[0].name


def _rows(case) -> dict:
    """Rows by the path the manifest uses: the volume's name taken off the front."""
    return {r["rel_path"].split("/", 1)[1]: r for r in case.db.iter_files()}


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _files_under(folder: Path) -> list[Path]:
    return [p for p in Path(folder).rglob("*") if p.is_file()]


def test_the_manifest_is_the_one_these_tests_were_written_against():
    """A fixture that changed shape would retire assertions below without failing them."""
    assert len(KNOWN) == 35
    assert {p for p, k in KNOWN.items() if k["windows"] == "refused"} == PLACEHOLDERS
    assert sum(1 for k in KNOWN.values() if k["sha256"]) == 32
    assert len(LZX) == 5 and len(READABLE) == 32 - 5
    assert all(KNOWN[p]["on_disk"] < KNOWN[p]["length"] for p in SPARSE)


@pytest.mark.parametrize("stage", [False, True], ids=["on-demand", "copied-in"])
def test_a_cloud_placeholder_is_a_row_with_no_copy_and_no_hash(tmp_path, image, stage):
    """The three files Windows refused to read are the three kept as rows only."""
    case, name = _ingest(tmp_path, image, include_other=True, stage=stage)
    rows = _rows(case)
    marked = {p for p, r in rows.items() if archive.is_placeholder_error(r["error"])}
    assert marked == PLACEHOLDERS
    assert archive.source_record(case, name)["placeholders"] == 3
    process(case, screen=False, similar=False)
    process(case, screen=False, similar=False, force=True)     # a forced run is no different
    rows = _rows(case)
    for path in PLACEHOLDERS:
        row = rows[path]
        assert row["size"] == KNOWN[path]["length"]
        assert archive.is_placeholder_error(row["error"]), row["error"]
        assert (row["md5"], row["sha1"], row["sha256"]) == (None, None, None)
        assert not row["thumb"]
        assert not Path(row["path"]).exists()
    # by its name, since there are no first bytes to go by
    assert rows["CloudRoot/online_only_photo.jpg"]["kind"] == "image"
    assert rows["CloudRoot/online_only_video.mp4"]["kind"] == "video"
    # and nothing was left behind for one in the folders on-demand copies go to
    for folder in (archive.TMP_DIR, archive.CACHE_DIR):
        assert not _files_under(case.root / folder)


@pytest.mark.parametrize("stage", [False, True], ids=["on-demand", "copied-in"])
def test_every_file_windows_hashed_and_the_reader_reads_hashes_the_same(
        tmp_path, image, stage):
    case, name = _ingest(tmp_path, image, include_other=True, stage=stage)
    process(case, screen=False, similar=False)
    rows = _rows(case)
    assert READABLE <= set(rows)
    wrong = {p: rows[p]["sha256"] for p in READABLE if rows[p]["sha256"] != KNOWN[p]["sha256"]}
    assert not wrong
    # the LZX ones are not registered, and the source's record says five were not read
    assert not LZX & set(rows)
    assert case.db.get_meta(f"archive:{name}:failed") == "5"


def test_a_file_the_volume_holds_under_two_names_is_copied_once(tmp_path, image):
    case, name = _ingest(tmp_path, image, stage=True)
    rows = _rows(case)
    one, two = rows["Pictures/plain.jpg"], rows["Pictures/hardlink.jpg"]
    assert one["path"] != two["path"]
    assert _sha256(one["path"]) == _sha256(two["path"]) == KNOWN["Pictures/plain.jpg"]["sha256"]
    assert os.path.samefile(one["path"], two["path"])
    assert archive.source_record(case, name)["linked"] == 1


def test_a_filesystem_with_no_hard_links_gets_a_second_copy(tmp_path, image, monkeypatch):
    monkeypatch.setattr(archive, "_link", lambda first, dest: False)
    case, name = _ingest(tmp_path, image, stage=True)
    rows = _rows(case)
    one, two = rows["Pictures/plain.jpg"], rows["Pictures/hardlink.jpg"]
    assert not os.path.samefile(one["path"], two["path"])
    assert _sha256(two["path"]) == KNOWN["Pictures/hardlink.jpg"]["sha256"]
    assert archive.source_record(case, name)["linked"] == 0


def _zeros_and(*parts) -> bytes:
    """Bytes built from ``(length, fill)`` runs; a fill of 0 is a hole to be."""
    return b"".join(bytes([fill]) * length for length, fill in parts)


PIECE = archive._HOLE                                  # pylint: disable=protected-access
SHAPES = {
    "empty": b"",
    "one byte": b"\x07",
    "all zeros": bytes(3 * PIECE + 11),
    "all zeros, one piece exactly": bytes(PIECE),
    "zeros then data": _zeros_and((2 * PIECE, 0), (100, 9)),
    "data then zeros": _zeros_and((100, 9), (2 * PIECE + 5, 0)),
    "data, hole, data": _zeros_and((PIECE, 1), (4 * PIECE, 0), (PIECE + 3, 2)),
    "a zero run shorter than a piece": _zeros_and((10, 1), (PIECE // 2, 0), (10, 2)),
    "a hole that straddles a read": _zeros_and((archive._CHUNK - PIECE, 3),   # pylint: disable=protected-access
                                               (3 * PIECE, 0), (17, 4)),
    "no zeros at all": bytes([5]) * (2 * PIECE + 1),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_writing_holes_as_holes_changes_no_byte(tmp_path, shape):
    data = SHAPES[shape]
    dest = tmp_path / "out.bin"
    archive._write_stream(io.BytesIO(data), dest, holes=True)   # pylint: disable=protected-access
    assert dest.read_bytes() == data
    # and a head handed over beside the stream is written once, in front of it
    head, rest = data[:16], data[16:]
    archive._write_stream(io.BytesIO(rest), dest, head, holes=True)   # pylint: disable=protected-access
    assert dest.read_bytes() == data
    assert not [p for p in tmp_path.iterdir() if p != dest]     # no part file left


def test_a_mostly_empty_file_takes_less_room_where_the_filesystem_keeps_holes(tmp_path):
    """Asked of the filesystem the test runs on. One that keeps no holes (FAT, exFAT)
    skips, rather than pass on a check that did not run. The CI runners' filesystems
    (ext4, APFS, NTFS) all keep them, so there a skip would mean the hole writing
    or the probe is broken on that platform, and it fails instead."""
    if not archive._keeps_holes(tmp_path):                      # pylint: disable=protected-access
        if os.environ.get("CI"):
            pytest.fail("no hole was kept on a CI runner's filesystem")
        pytest.skip("this filesystem writes holes out in full")
    span = 2 * archive._HOLE_PROBE                              # pylint: disable=protected-access
    data = _zeros_and((span, 0), (PIECE, 6), (span, 0))
    dest = tmp_path / "sparse.bin"
    archive._write_stream(io.BytesIO(data), dest, holes=True)   # pylint: disable=protected-access
    assert dest.stat().st_size == len(data)
    taken = archive._allocated(dest)                            # pylint: disable=protected-access
    assert taken is not None and taken < len(data) // 4
    plain = tmp_path / "plain.bin"
    archive._write_stream(io.BytesIO(data), plain)              # pylint: disable=protected-access
    assert archive._allocated(plain) >= len(data)               # pylint: disable=protected-access
    assert plain.read_bytes() == dest.read_bytes()


def test_what_a_copy_is_counted_at():
    count = archive._copy_bytes                                 # pylint: disable=protected-access
    slack = archive._HOLE_SLACK                                 # pylint: disable=protected-access
    sparse = {"sparse": True, "compression": "", "stored": 4096}
    assert count(50_000_000, None, True) == 50_000_000          # the reader could not say
    assert count(50_000_000, sparse, False) == 50_000_000       # holes are not kept here
    assert count(50_000_000, sparse, True) == 4096 + slack
    assert count(5000, sparse, True) == 5000                    # never more than the file
    # a compressed file is written out uncompressed, whatever little it stores
    packed = {"sparse": True, "compression": "wof-xpress4k", "stored": 4096}
    assert count(50_000_000, packed, True) == 50_000_000
    assert count(50_000_000, {"sparse": True, "compression": "", "stored": None},
                 True) == 50_000_000
    assert count(50_000_000, {"sparse": False, "compression": "", "stored": 4096},
                 True) == 50_000_000


def test_every_copy_of_a_walked_file_leaves_holes(tmp_path, image, monkeypatch):
    real = archive._write_stream                               # pylint: disable=protected-access
    asked = []

    def watching(fin, dest, head=b"", *, holes=False):
        asked.append(holes)
        return real(fin, dest, head, holes=holes)

    monkeypatch.setattr(archive, "_write_stream", watching)
    case, name = _ingest(tmp_path, image)
    assert not asked                                            # nothing copied at ingest
    row = _rows(case)["Pictures/sparse_photo.jpg"]
    with archive.local_copy(case.root, archive.source_record(case, name), row) as local:
        assert _sha256(local) == KNOWN["Pictures/sparse_photo.jpg"]["sha256"]
    assert asked == [True]
    archive.stage_source(case, name)
    assert len(asked) > 1 and all(asked)
    del asked[:]
    _ingest(tmp_path, image, name="copied-in", stage=True)      # and at ingest
    assert asked and all(asked)


def test_a_copy_that_cannot_fit_is_refused_before_anything_is_written(
        tmp_path, image, monkeypatch):
    monkeypatch.setattr(archive, "_free_bytes", lambda folder: 4096)
    case = open_case(tmp_path / "case", create=True, examiner="t")
    sources, _ = parse_source_spec(image)
    sources[0].stage = True
    with pytest.raises(ValueError, match="needs about .* bytes .* has 4,096 free"):
        ingest_sources(case, sources)
    assert not case.db.iter_files()
    assert not _files_under(case.staged_dir)
    assert archive.source_record(case, sources[0].name) is None


def _room_asked_for(tmp_path, image, monkeypatch, *, holes: bool, name: str) -> tuple[int, dict]:
    asked = []
    monkeypatch.setattr(archive, "_keeps_holes", lambda folder: holes)
    monkeypatch.setattr(archive, "_require_room",
                        lambda folder, need, what: asked.append(need))
    case, _ = _ingest(tmp_path, image, name=name, include_other=True, stage=True)
    assert len(asked) == 1
    return asked[0], _rows(case)


def test_the_room_asked_for_counts_a_file_once_and_a_placeholder_not_at_all(
        tmp_path, image, monkeypatch):
    need, rows = _room_asked_for(tmp_path, image, monkeypatch, holes=False, name="plain")
    sizes = sum(r["size"] for p, r in rows.items() if p not in PLACEHOLDERS)
    # every copy at its recorded size, less the second name of the one linked file
    assert need == sizes - KNOWN["Pictures/hardlink.jpg"]["length"]
    assert need < sizes < sizes + sum(KNOWN[p]["length"] for p in PLACEHOLDERS)


def test_where_holes_are_kept_a_sparse_file_is_counted_at_what_the_volume_stores(
        tmp_path, image, monkeypatch):
    full, _ = _room_asked_for(tmp_path, image, monkeypatch, holes=False, name="full")
    kept, _ = _room_asked_for(tmp_path, image, monkeypatch, holes=True, name="kept")
    slack = archive._HOLE_SLACK                                 # pylint: disable=protected-access
    # from Windows's own size on disk for the five files it reported as sparse
    saved = sum(KNOWN[p]["length"] - min(KNOWN[p]["length"], KNOWN[p]["on_disk"] + slack)
                for p in SPARSE)
    assert saved > 20_000_000
    assert full - kept == saved


def test_a_copy_pass_stops_when_the_volume_fills(tmp_path, image, monkeypatch):
    """Every file after the first that does not fit would fail the same way, and be
    counted as one the reader could not read."""
    real = archive._write_stream                               # pylint: disable=protected-access
    calls = []

    def filling(fin, dest, head=b"", *, holes=False):
        calls.append(dest)
        if len(calls) == 3:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(fin, dest, head, holes=holes)

    monkeypatch.setattr(archive, "_write_stream", filling)
    case = open_case(tmp_path / "case", create=True, examiner="t")
    sources, _ = parse_source_spec(image)
    sources[0].stage = True
    sources[0].include_other = True
    with pytest.raises(ValueError, match="filled while copying"):
        ingest_sources(case, sources)
    assert len(calls) == 3                                      # and it went no further
    assert len(case.db.iter_files()) == 2                       # the two that were written


def test_copying_a_source_in_later_writes_what_the_ingest_would_have(tmp_path, image):
    case, name = _ingest(tmp_path, image, include_other=True)
    assert not _files_under(case.staged_dir)
    rows = _rows(case)
    written = archive.stage_source(case, name)
    assert written == len(rows) - len(PLACEHOLDERS)
    for path in READABLE:
        assert _sha256(rows[path]["path"]) == KNOWN[path]["sha256"], path
    for path in PLACEHOLDERS:
        assert not Path(rows[path]["path"]).exists()
    assert os.path.samefile(rows["Pictures/plain.jpg"]["path"],
                            rows["Pictures/hardlink.jpg"]["path"])
    rec = archive.source_record(case, name)
    assert (rec["mode"], rec["linked"], rec["placeholders"]) == (archive.MODE_STAGED, 1, 3)
    assert archive.stage_source(case, name) == 0                # nothing left to copy


def test_copying_in_later_asks_for_the_room_the_ingest_would_have(
        tmp_path, image, monkeypatch):
    monkeypatch.setattr(archive, "_keeps_holes", lambda folder: False)
    case, name = _ingest(tmp_path, image, include_other=True)
    asked = []
    monkeypatch.setattr(archive, "_require_room",
                        lambda folder, need, what: asked.append(need))
    archive.stage_source(case, name)
    sizes = sum(r["size"] for p, r in _rows(case).items() if p not in PLACEHOLDERS)
    assert asked == [sizes - KNOWN["Pictures/hardlink.jpg"]["length"]]


def test_copying_in_later_is_refused_when_it_cannot_fit(tmp_path, image, monkeypatch):
    case, name = _ingest(tmp_path, image)
    monkeypatch.setattr(archive, "_free_bytes", lambda folder: 4096)
    with pytest.raises(ValueError, match="needs about"):
        archive.stage_source(case, name)
    assert not _files_under(case.staged_dir)
    assert archive.source_record(case, name)["mode"] == archive.MODE_REFERENCE


def test_a_placeholder_an_older_case_read_as_zeros_is_found_when_copying_in(
        tmp_path, image):
    """A case ingested before the reader could tell: the row carries no marker and
    the hash of the zeros it was read as."""
    case, name = _ingest(tmp_path, image)
    row = _rows(case)["CloudRoot/online_only_photo.jpg"]
    zeros = hashlib.md5(bytes(row["size"])).hexdigest()
    case.db.update_file(row["id"], error=None, md5=zeros, sha1="0" * 40, sha256="0" * 64)
    case.db.set_meta(f"archive:{name}:placeholders", "1")
    case.db.commit()
    archive.stage_source(case, name)
    row = _rows(case)["CloudRoot/online_only_photo.jpg"]
    assert archive.is_placeholder_error(row["error"])
    assert (row["md5"], row["sha1"], row["sha256"]) == (None, None, None)
    assert not Path(row["path"]).exists()
    assert archive.source_record(case, name)["placeholders"] == 2


def test_ingesting_again_marks_a_placeholder_an_older_case_had_hashed(tmp_path, image):
    case, _ = _ingest(tmp_path, image)
    row = _rows(case)["CloudRoot/online_only_video.mp4"]
    case.db.update_file(row["id"], error=None, md5="0" * 32, sha1="0" * 40, sha256="0" * 64)
    case.db.commit()
    sources, _ = parse_source_spec(image)
    ingest_sources(case, sources)
    again = _rows(case)["CloudRoot/online_only_video.mp4"]
    assert again["id"] == row["id"]
    assert archive.is_placeholder_error(again["error"])
    assert (again["md5"], again["sha1"], again["sha256"]) == (None, None, None)


def test_one_file_the_reader_declines_does_not_stop_the_rest_being_copied_in(
        tmp_path, image, monkeypatch):
    case, name = _ingest(tmp_path, image, include_other=True)
    rows = _rows(case)
    declined = rows["plain_5000.txt"]["orig_path"]
    real = archive._extract_walked_member                      # pylint: disable=protected-access

    def declining(rec, row, dest):
        if row["orig_path"] == declined:
            raise archive.ArchiveUnavailable("could not read it from the source image")
        return real(rec, row, dest)

    monkeypatch.setattr(archive, "_extract_walked_member", declining)
    written = archive.stage_source(case, name)
    assert written == len(rows) - len(PLACEHOLDERS) - 1
    assert not Path(rows["plain_5000.txt"]["path"]).exists()
    assert Path(rows["resident_300.txt"]["path"]).exists()
    said = [r["detail"] for r in case.db.conn.execute(
        "SELECT detail FROM audit WHERE action = 'stage-source'")]
    assert len(said) == 1 and "1 could not be read from the image" in said[0]


def test_an_image_that_has_gone_still_stops_the_copy(tmp_path, image, monkeypatch):
    case, name = _ingest(tmp_path, image)

    def gone(rec, row, dest):
        raise archive.ArchiveUnavailable("the source image is no longer there")

    monkeypatch.setattr(archive, "_extract_walked_member", gone)
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists",
                        lambda self: False if self == Path(image) else real_exists(self))
    with pytest.raises(archive.ArchiveUnavailable):
        archive.stage_source(case, name)


def test_the_volume_filling_while_copying_in_is_not_an_unreadable_file(
        tmp_path, image, monkeypatch):
    case, name = _ingest(tmp_path, image)

    def filling(fin, dest, head=b"", *, holes=False):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(archive, "_write_stream", filling)
    with pytest.raises(OSError) as caught:
        archive.stage_source(case, name)
    assert caught.value.errno == errno.ENOSPC
    assert archive.source_record(case, name)["mode"] == archive.MODE_REFERENCE


def test_a_placeholder_document_is_not_opened_as_a_container(tmp_path, image):
    """Kept as a container by its name, and its row goes on saying what it is: the
    pass that opens containers must not write its own message over that."""
    case, _ = _ingest(tmp_path, image, documents=True)
    row = _rows(case)["CloudRoot/online_only_doc.pdf"]
    assert row["kind"] == "archive"
    assert archive.is_placeholder_error(row["error"])
    nested.expand_containers(case, documents=True, force=True)
    row = _rows(case)["CloudRoot/online_only_doc.pdf"]
    assert archive.is_placeholder_error(row["error"])
    assert not nested.is_expansion_error(row["error"])


def _wait_for_the_job(client, tries=240):
    for _ in range(tries):
        job = client.get("/api/job").get_json()
        if not job["running"] and job["stage"] in ("done", "error"):
            return job
        time.sleep(0.25)
    raise AssertionError("the job did not finish")


def test_a_placeholder_is_listed_with_the_errors_and_is_not_a_file_to_retry(tmp_path, image):
    """Its row carries a note in the error column, so the error filter shows it. A
    retry can do nothing with it, so the count on the retry button leaves it out and
    the retry leaves its row alone."""
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel

    client = create_app(None).test_client()
    client.post("/api/case/create", json={"path": str(tmp_path / "case"), "name": "C"})
    client.post("/api/case/ingest", json={
        "sources": [{"name": "disk", "path": str(image)}],
        "options": {"keyframes": 0, "screen": False}})
    assert _wait_for_the_job(client)["stage"] == "done"

    def marked():
        listed = client.get("/api/files?error=1&limit=500").get_json()
        listed = listed["files"] if isinstance(listed, dict) else listed
        return sorted(f["orig_name"] for f in listed
                      if archive.is_placeholder_error(f["error"]))

    ctx = client.get("/api/context").get_json()
    assert marked() == ["online_only_photo.jpg", "online_only_video.mp4"]
    assert ctx["errors"] - ctx["retryable"] == 2
    assert ctx["archive_sources"][0]["placeholders"] == 2
    started = client.post("/api/reprocess-errors").get_json()
    assert started["count"] == ctx["retryable"]
    _wait_for_the_job(client)
    after = client.get("/api/context").get_json()
    assert marked() == ["online_only_photo.jpg", "online_only_video.mp4"]
    assert after["errors"] - after["retryable"] == 2
