"""XFS, Btrfs and ReFS, walked with the readers qnxprobe 1.68 added.

The fixtures are qnxprobe's own, at v1.70 (commit 98c4fc8861efd819913679a6e6462479b6bea9e4):
``refs-fixture.bin.gz`` and ``refs-dirty.bin.gz`` keep only the bytes of a ReFS 3.14
volume Windows 11 wrote that the reader reads, each at its own offset
(qnxprobe ``_RefsSparseImage``), and ``refs-fixture.known.tsv.gz`` is what Windows
reported about every file while the volume was mounted, its SHA-256 among it.
``btrfs-raid1-dev2.img.gz`` is one device of a two-device Btrfs RAID1.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from gleapp import archive
from gleapp.case import open_case, parse_source_spec
from gleapp.pipeline import ingest_sources

FIXTURES = Path(__file__).parent / "fixtures"


class _Sparse:
    """A qnxprobe sparse ReFS fixture with the surface GLEAPP's image readers offer."""

    def __init__(self, path: Path) -> None:
        self._img = archive.qnxprobe._RefsSparseImage(  # pylint: disable=protected-access
            gzip.decompress(path.read_bytes()))
        self.media_size = self._img.size
        self.paths = [str(path)]
        self.stored_hashes = {"head-tail-sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.not_shipped = set(self._img.head.get("not_shipped", []))

    def seek(self, off, whence=0):
        return self._img.seek(off, whence)

    def tell(self):
        return self._img.tell()

    def read(self, n=-1):
        return self._img.read(n)

    def close(self):
        pass


def _ingest_sparse(tmp_path, monkeypatch, fixture: str):
    """Ingest a sparse fixture as a raw image source: the file named stands for the
    volume, and every read of it is served from the fixture."""
    stand_in = tmp_path / "ev" / f"{fixture}.img"
    stand_in.parent.mkdir(parents=True, exist_ok=True)
    stand_in.write_bytes(b"stand-in")
    real_base, real_format = archive._open_base, archive.archive_format  # pylint: disable=protected-access
    sparse = FIXTURES / f"{fixture}.bin.gz"

    def base(path):
        if Path(path).name == stand_in.name:
            return _Sparse(sparse)
        return real_base(path)

    def fmt(path):
        if Path(path).name == stand_in.name:
            return archive.FORMAT_RAW
        return real_format(path)

    monkeypatch.setattr(archive, "_open_base", base)
    monkeypatch.setattr(archive, "archive_format", fmt)
    case = open_case(tmp_path / "case", create=True, examiner="t")
    sources, _ = parse_source_spec(stand_in)
    sources[0].include_other = True
    ingest_sources(case, sources)
    return case, sources[0].name


def test_a_refs_volume_is_walked_and_read_to_the_hashes_windows_recorded(tmp_path, monkeypatch):
    """A ReFS file's node is the directory naming it and its name as stored, in
    bytes. It used to stop the ingest at the JSON it is stored as, so nothing was
    registered; it is now stored as itself and read back through it."""
    case, name = _ingest_sparse(tmp_path, monkeypatch, "refs-fixture")
    rows = {r["orig_path"].split("/", 1)[1]: r for r in case.db.iter_files("source = ?", (name,))}
    lines = gzip.decompress((FIXTURES / "refs-fixture.known.tsv.gz").read_bytes()).decode(
        "utf-8").splitlines()
    head = lines[0].split("\t")
    known = {k["path"]: k for k in (dict(zip(head, line.split("\t"))) for line in lines[1:])
             if k["kind"] == "file"}
    assert len(known) == 2544
    assert set(known) <= set(rows)
    # what Windows did not list is its own System Volume Information
    assert all(p.startswith("System Volume Information/") for p in set(rows) - set(known))
    not_shipped = _Sparse(FIXTURES / "refs-fixture.bin.gz").not_shipped
    rec = archive.source_record(case, name)
    checked = 0
    for path, k in known.items():
        if path in not_shipped:
            continue                # its random content is not in the fixture
        with archive.local_copy(case.root, rec, rows[path]) as local:
            assert hashlib.sha256(Path(local).read_bytes()).hexdigest() == k["sha256"], path
        checked += 1
    assert checked == 2540
    node = json.loads(rows[next(iter(known))]["member_node"])
    assert node["type"] == "RefsEntry"
    case.close()


def test_a_refs_volume_with_nothing_listed_says_why(tmp_path, monkeypatch):
    case, name = _ingest_sparse(tmp_path, monkeypatch, "refs-dirty")
    rec = archive.source_record(case, name)
    assert "nothing listed: 16 metadata log records follow the newest checkpoint" in \
        rec["volumes_not_read"]
    case.close()


def test_one_device_of_a_btrfs_raid1_says_why_nothing_is_listed(tmp_path):
    image = tmp_path / "ev" / "btrfs-raid1-dev2.img"
    image.parent.mkdir()
    image.write_bytes(gzip.decompress((FIXTURES / "btrfs-raid1-dev2.img.gz").read_bytes()))
    case = open_case(tmp_path / "case", create=True, examiner="t")
    sources, _ = parse_source_spec(image)
    sources[0].include_other = True
    ingest_sources(case, sources)
    rec = archive.source_record(case, image.name)
    assert case.db.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    assert "nothing listed: one device (devid 2) of 2" in rec["volumes_not_read"]
    case.close()


def test_a_node_is_stored_as_it_always_was_and_read_back_as_itself():
    # the forms every walker GLEAPP walked before hands out keep their JSON exactly
    assert archive.node_json(1234) == "1234"
    assert archive.node_json((5, 4096, False)) == "[5, 4096, false]"
    assert archive.node_from_json("[5, 4096, false]") == (5, 4096, False)
    # bytes, and the walkers' own node types, come back as themselves
    entry = archive.qnxprobe.RefsEntry(1792, b"n\x00a\x00m\x00e\x00")
    back = archive.node_from_json(archive.node_json(entry))
    assert back == entry and type(back) is archive.qnxprobe.RefsEntry
    stream = archive.qnxprobe.NtfsStreamRef(42, "Zone.Identifier")
    back = archive.node_from_json(archive.node_json(stream))
    assert back == stream and type(back) is archive.qnxprobe.NtfsStreamRef
