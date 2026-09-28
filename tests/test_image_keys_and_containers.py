"""Disk images the vendored readers added, and what opens the locked ones.

qnxprobe 1.49 and ewfprobe 0.12.0 read AFF4 and virtual machine disks (VHD, VHDX, VMDK,
QCOW), an AFF encrypted with a passphrase, an image sealed to a certificate (opened with
that certificate's private key), and BitLocker volumes inside an image (opened with a
password, a recovery password or a startup key). These tests hold GLEAPP's own code to
each: how a source is routed, where what opens it is kept, and the command line and web
API that take it.

Fixtures: pyaff4-zlib.aff4, qcow2-v3.qcow2.gz, vmdk-base.vmdk.gz, dirty-log.vhdx.gz,
aff-enc-pass.aff (passphrase ewfprobe-aff-password) and aff-enc-inplace-cert.aff with
its test key aff-enc-test-key.pem come from ewfprobe's tests/fixtures; ftk-ad-cert-ad1.ad1
(an AD1 FTK Imager 4.7.3.61 encrypted to the certificate of ad-cert-test-key-2048.pem)
and bitlocker-xts128.img.gz with its startup key bitlocker-xts128.BEK come from the LEAPP
cores' raw image fixtures. The keys were made for these fixtures and open nothing else.
"""

import gzip
import io
import json
import shutil
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gleapp import archive, cli
from gleapp.case import open_case, parse_source_spec
from gleapp.pipeline import ingest_sources

sys.path.insert(0, str(Path(__file__).parent))
from fatwriter import build_fat32                      # pylint: disable=import-error,wrong-import-position

FIXTURES = Path(__file__).parent / "fixtures"
AFF_PASSWORD = "ewfprobe-aff-password"
_OPTS = {"screen": False, "keyframes": 0, "carve": True}


@pytest.fixture(autouse=True)
def _isolate(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("GLEAPP_CONFIG_DIR", str(tmp_path_factory.mktemp("gleapp-cfg")))
    # each test is a new session: nothing that opened an image carries over
    monkeypatch.setattr(archive, "_PASSWORDS", {})
    monkeypatch.setattr(archive, "_PRIVATE_KEYS", {})
    monkeypatch.setattr(archive, "_BITLOCKER", {})
    from gleapp import hashstore, stash  # pylint: disable=import-outside-toplevel
    hashstore.close()
    stash.close()
    yield
    archive.close_zips()
    hashstore.close()
    stash.close()


def _copy(tmp_path, name) -> Path:
    """A fixture copied beside the test, unzipped when it is gzipped."""
    src = FIXTURES / name
    folder = tmp_path / "ev"
    folder.mkdir(exist_ok=True)
    if name.endswith(".gz"):
        dest = folder / name[:-3]
        dest.write_bytes(gzip.decompress(src.read_bytes()))
    else:
        dest = folder / name
        shutil.copyfile(src, dest)
    return dest


def _jpg(seed) -> bytes:
    rnd = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rnd.integers(0, 256, (48, 64, 3), dtype="uint8")).save(buf, "JPEG")
    return buf.getvalue()


def _fixed_vhd(path: Path, disk: bytes) -> Path:
    """``disk`` as a fixed VHD: the disk's bytes and then the 512-byte footer the
    Virtual Hard Disk Image Format Specification (October 2006) describes."""
    size = len(disk)
    sectors = min(size // 512, 65535 * 16 * 255)
    heads, per_track = 16, 255                  # the specification's CHS algorithm
    if sectors < 65535 * 16 * 63:
        per_track = 17
        cyl_heads = sectors // per_track
        heads = max(4, (cyl_heads + 1023) // 1024)
        if cyl_heads >= heads * 1024 or heads > 16:
            per_track, heads = 31, 16
            cyl_heads = sectors // per_track
        if cyl_heads >= heads * 1024:
            per_track, heads = 63, 16
            cyl_heads = sectors // per_track
    else:
        cyl_heads = sectors // per_track
    cylinders = cyl_heads // heads
    footer = bytearray(512)
    footer[0:8] = b"conectix"
    struct.pack_into(">IIQI", footer, 8, 2, 0x00010000, 0xFFFFFFFFFFFFFFFF, 0)
    footer[28:32] = b"glp "
    struct.pack_into(">I", footer, 32, 0x00010000)
    footer[36:40] = b"Wi2k"
    struct.pack_into(">QQHBBI", footer, 40, size, size, cylinders, heads, per_track, 2)
    footer[68:84] = bytes(range(16))
    struct.pack_into(">I", footer, 64, ~sum(footer) & 0xFFFFFFFF)
    path.write_bytes(disk + bytes(footer))
    return path


def _ingest(case, path, **options) -> dict:
    sources, _ = parse_source_spec(path)
    for s in sources:
        for k, v in options.items():
            setattr(s, k, v)
    ingest_sources(case, sources)
    return {s["name"]: s for s in archive.source_status(case)}[Path(path).name]


# ---- containers ------------------------------------------------------------
def test_an_aff4_is_read_as_an_image_and_not_opened_as_a_zip(tmp_path):
    """An AFF4 is a ZIP container, so a reader that does not name it takes it for an
    extraction archive: before AFF4 was listed it came back as ``zip``."""
    aff4 = _copy(tmp_path, "pyaff4-zlib.aff4")
    assert zipfile.is_zipfile(aff4), "the control: an AFF4 is a zip by its bytes"
    assert archive.archive_format(aff4) == archive.FORMAT_EWF
    img = archive._open_image_file(aff4)                     # pylint: disable=protected-access
    try:
        assert img.media_size == 63894
    finally:
        img.close()
    plain = tmp_path / "ev" / "plain.zip"
    with zipfile.ZipFile(plain, "w") as zf:
        zf.writestr("DCIM/a.jpg", _jpg(1))
    assert archive.archive_format(plain) == archive.FORMAT_ZIP


@pytest.mark.parametrize("name", ["qcow2-v3.qcow2.gz", "vmdk-base.vmdk.gz",
                                  "dirty-log.vhdx.gz"])
def test_a_virtual_machine_disk_is_read_as_an_image(tmp_path, name):
    """None of these was recognised before VHDX, VMDK and QCOW were listed."""
    disk = _copy(tmp_path, name)
    assert archive.archive_format(disk) == archive.FORMAT_EWF


def test_a_fixed_vhd_is_walked_through_its_reader_and_not_as_raw_bytes(tmp_path):
    """A fixed VHD is the disk followed by a footer, so a raw read happens to find the
    same filesystem. The reader is what drops the footer: the recorded size is the
    disk's, not the file's."""
    volume = build_fat32([("PHOTO", "JPG", _jpg(2), (2024, 5, 6, 7, 8, 10))])
    vhd = _fixed_vhd(tmp_path / "disk.vhd", volume)
    assert archive.archive_format(vhd) == archive.FORMAT_EWF
    case = open_case(tmp_path / "case", create=True, examiner="t")
    try:
        row = _ingest(case, vhd)
        assert (row["format"], row["media_size"]) == ("ewf", len(volume))
        assert row["walked"] == 1
        names = [r["rel_path"] for r in case.db.conn.execute("SELECT rel_path FROM files")]
        assert names == ["lba0/PHOTO.JPG"]
    finally:
        case.close()


# ---- an AFF encrypted with a passphrase -----------------------------------
def test_an_aff_encrypted_with_a_passphrase_opens_with_the_password_given(tmp_path):
    """The password was taken and then not used: the AFF was opened without it, since
    only an Apple disk image and an AD-encrypted set were read with one."""
    aff = _copy(tmp_path, "aff-enc-pass.aff")
    assert archive.needs_password(aff) and not archive.needs_private_key(aff)
    with pytest.raises(archive.ImagePasswordNeeded, match="an encrypted AFF"):
        archive._open_image_file(aff)                        # pylint: disable=protected-access
    assert archive.unlock_image(aff, "not it") is False
    assert archive.unlock_image(aff, AFF_PASSWORD) is True
    img = archive._open_image_file(aff)                      # pylint: disable=protected-access
    img.close()
    case = open_case(tmp_path / "case", create=True, examiner="t")
    try:
        row = _ingest(case, aff, carve=True)
        assert (row["status"], row["format"]) == ("ok", "ewf")
        assert row["carved"] == 6, "the six pictures on the disk"
    finally:
        case.close()


# ---- an image sealed to a certificate --------------------------------------
def test_an_image_sealed_to_a_certificate_opens_with_its_private_key(tmp_path, monkeypatch):
    aff = _copy(tmp_path, "aff-enc-inplace-cert.aff")
    key = (FIXTURES / "aff-enc-test-key.pem").read_bytes()
    stranger = (FIXTURES / "ad-cert-test-key-2048.pem").read_bytes()
    assert archive.needs_private_key(aff) and not archive.needs_password(aff)
    with pytest.raises(archive.ImagePasswordNeeded,
                       match="the private key of a certificate it is sealed to"):
        archive._open_image_file(aff)                        # pylint: disable=protected-access
    assert archive.unlock_image(aff, private_key=stranger) is False
    assert not archive.is_unlocked(aff)
    assert archive.unlock_image(aff, private_key=key) is True
    assert archive.is_unlocked(aff)
    case = open_case(tmp_path / "case", create=True, examiner="t")
    try:
        row = _ingest(case, aff, carve=True)
        assert (row["status"], row["carved"]) == ("ok", 6)
        monkeypatch.setattr(archive, "_PRIVATE_KEYS", {})    # the app was restarted
        row = {s["name"]: s for s in archive.source_status(case)}[aff.name]
        assert (row["status"], row["locked_by"]) == ("locked", "private key")
    finally:
        case.close()


def test_an_ad1_is_refused_as_logical_evidence_even_once_decrypted(tmp_path):
    ad1 = _copy(tmp_path, "ftk-ad-cert-ad1.ad1")
    key = (FIXTURES / "ad-cert-test-key-2048.pem").read_bytes()
    assert archive.needs_private_key(ad1)
    assert archive.unlock_image(ad1, private_key=key) is True
    with pytest.raises(archive.ArchiveUnavailable,
                       match=r"FTK Imager logical evidence \(AD1\), encrypted with AD"):
        archive._open_image_file(ad1)                        # pylint: disable=protected-access
    plain = tmp_path / "ev" / "plain.ad1"
    plain.write_bytes(b"ADSEGMENTEDFILE\x00" + bytes(496))
    assert "FTK Imager logical evidence (AD1)" in (archive.container_refusal(plain) or "")
    with pytest.raises(ValueError, match="logical evidence"):
        parse_source_spec(plain)


def test_the_command_line_takes_a_private_key_or_refuses(tmp_path, monkeypatch, capsys):
    aff = _copy(tmp_path, "aff-enc-inplace-cert.aff")
    case = str(tmp_path / "c")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))            # not a terminal
    assert cli.main(["-c", case, "ingest", str(aff), "--no-process"]) == 2
    assert "--private-key" in capsys.readouterr().err
    wrong = FIXTURES / "ad-cert-test-key-2048.pem"
    assert cli.main(["-c", case, "--private-key", str(wrong), "ingest", str(aff),
                     "--no-process"]) == 2
    assert cli.main(["-c", case, "--private-key", str(wrong), "--private-key",
                     str(FIXTURES / "aff-enc-test-key.pem"), "ingest", str(aff),
                     "--no-process"]) == 0
    c = open_case(case)
    try:
        assert [(s["name"], s["status"]) for s in archive.source_status(c)] == [
            (aff.name, "ok")]
    finally:
        c.close()


# ---- BitLocker -------------------------------------------------------------
def _job(tmp_path, image) -> Path:
    """A job file that registers every walked file, so a count shows the walk."""
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"sources": [{"path": str(image), "include_other": True}]}))
    return job


def test_a_bitlocker_volume_is_named_as_locked_and_opens_with_its_startup_key(
        tmp_path, monkeypatch, capsys):
    image = _copy(tmp_path, "bitlocker-xts128.img.gz")
    bek = FIXTURES / "bitlocker-xts128.BEK"
    monkeypatch.setattr("sys.stdin", io.StringIO(""))            # not a terminal

    locked = str(tmp_path / "locked")
    assert cli.main(["-c", locked, "ingest", str(_job(tmp_path, image)),
                     "--no-process"]) == 0
    assert "stays locked and its files are not read" in capsys.readouterr().err
    c = open_case(locked)
    try:
        row = {s["name"]: s for s in archive.source_status(c)}[image.name]
        refused = json.loads(row["volumes_not_read"])
        assert len(refused) == 1 and "BitLocker-encrypted and not read" in refused[0]
        assert "NoneType" not in refused[0]
        assert row["walked"] == 0 and row["status"] == "ok" and not row["bitlocker"]
    finally:
        c.close()

    opened = str(tmp_path / "opened")
    assert cli.main(["-c", opened, "--bitlocker-key", str(bek), "ingest",
                     str(_job(tmp_path, image)), "--no-process"]) == 0
    assert "stays locked" not in capsys.readouterr().err
    c = open_case(opened)
    try:
        row = {s["name"]: s for s in archive.source_status(c)}[image.name]
        assert not row["volumes_not_read"]
        assert [v["kind"] for v in json.loads(row["volumes"])] == ["squashfs"]
        assert row["walked"] > 600, "the SquashFS inside, 612 files"
        rec = archive.source_record(c, image.name)
        walked = next(f for f in c.db.iter_files("source = ?", (image.name,))
                      if f["size"])
        archive.close_zips()
        monkeypatch.setattr(archive, "_BITLOCKER", {})       # the app was restarted
        row = {s["name"]: s for s in archive.source_status(c)}[image.name]
        assert (row["status"], row["locked_by"]) == ("locked", "BitLocker")
        with pytest.raises(archive.ArchiveUnavailable):     # ciphertext, until unlocked
            archive.cached_copy(c.root, rec, walked)
        assert archive.unlock_bitlocker(image, secret="not it") is False
        assert archive.unlock_bitlocker(image, startup_key=bek.read_bytes()) is True
        row = {s["name"]: s for s in archive.source_status(c)}[image.name]
        assert row["status"] == "ok"
        # the handle opened while it was locked is not the one read now
        assert archive.cached_copy(c.root, rec, walked).stat().st_size == walked["size"]
    finally:
        c.close()


# ---- the web API -----------------------------------------------------------
def _client():
    from gleapp.web.app import create_app  # pylint: disable=import-outside-toplevel
    return create_app(None).test_client()


def _wait(cl, timeout=60):
    for _ in range(timeout * 4):
        job = cl.get("/api/job").get_json()
        if not job["running"]:
            return job
        time.sleep(0.25)
    raise AssertionError("job never finished")


def test_the_web_ingest_asks_for_a_private_key_and_reads_it_from_its_path(tmp_path):
    aff = _copy(tmp_path, "aff-enc-inplace-cert.aff")
    key = FIXTURES / "aff-enc-test-key.pem"
    cl = _client()
    assert cl.post("/api/case/create", json={"path": str(tmp_path / "c"), "name": "c",
                                             "examiner": "t"}).status_code == 200
    body = {"sources": [{"path": str(aff)}], "options": _OPTS}
    r = cl.post("/api/case/ingest", json=body)
    assert r.status_code == 409
    need = r.get_json()["needs_password"]
    assert [(n["needs"], n["wrong"]) for n in need] == [("private key", False)]
    r = cl.post("/api/case/ingest", json={
        **body, "private_keys": {need[0]["path"]: str(FIXTURES / "ad-cert-test-key-2048.pem")}})
    assert r.status_code == 409 and r.get_json()["needs_password"][0]["wrong"] is True
    r = cl.post("/api/case/ingest", json={**body, "private_keys": {need[0]["path"]: str(key)}})
    assert r.status_code == 200, r.get_json()
    assert _wait(cl)["stage"] == "done"
    status = cl.get("/api/sources").get_json()
    assert [(s["name"], s["status"], s["format"]) for s in status] == [(aff.name, "ok", "ewf")]
    cl.post("/api/case/close")
    secret = key.read_bytes().split(b"\n")[1]              # a line of the key itself
    for p in (tmp_path / "c").rglob("*"):
        if p.is_file():
            assert secret not in p.read_bytes(), f"the key reached {p.name}"


def test_the_web_ingest_asks_for_a_bitlocker_key_or_leaves_the_volume_locked(tmp_path):
    image = _copy(tmp_path, "bitlocker-xts128.img.gz")
    cl = _client()
    cl.post("/api/case/create", json={"path": str(tmp_path / "c"), "name": "c",
                                      "examiner": "t"})
    body = {"sources": [{"path": str(image)}], "options": {**_OPTS, "carve": False}}
    r = cl.post("/api/case/ingest", json=body)
    assert r.status_code == 409
    need = r.get_json()["needs_password"]
    assert [(n["needs"], n["volume"], n["wrong"]) for n in need] == [
        ("bitlocker", "whole image", False)]
    assert "startup key" in need[0]["note"]
    r = cl.post("/api/case/ingest", json={**body, "bitlocker": {
        need[0]["path"]: {"secret": "not it"}}})
    assert r.status_code == 409 and r.get_json()["needs_password"][0]["wrong"] is True

    # leaving it locked reads the rest of the image and says the volume was not read
    r = cl.post("/api/case/ingest", json={**body, "bitlocker_skip": [need[0]["path"]]})
    assert r.status_code == 200, r.get_json()
    assert _wait(cl)["stage"] == "done"
    row = cl.get("/api/sources").get_json()[0]
    assert "BitLocker-encrypted and not read" in row["volumes_not_read"]
    cl.post("/api/case/close")

    cl.post("/api/case/create", json={"path": str(tmp_path / "c2"), "name": "c2",
                                      "examiner": "t"})
    r = cl.post("/api/case/ingest", json={**body, "bitlocker": {
        need[0]["path"]: {"key_file": str(FIXTURES / "bitlocker-xts128.BEK")}}})
    assert r.status_code == 200, r.get_json()
    assert _wait(cl)["stage"] == "done"
    row = cl.get("/api/sources").get_json()[0]
    assert not row["volumes_not_read"] and row["bitlocker"] == "[0]"


def test_a_source_holding_bitlocker_volumes_is_unlocked_through_the_api(tmp_path,
                                                                        monkeypatch):
    image = _copy(tmp_path, "bitlocker-xts128.img.gz")
    bek = FIXTURES / "bitlocker-xts128.BEK"
    assert archive.unlock_bitlocker(image, startup_key=bek.read_bytes())
    cl = _client()
    cl.post("/api/case/create", json={"path": str(tmp_path / "c"), "name": "c",
                                      "examiner": "t"})
    assert cl.post("/api/case/ingest", json={"sources": [{"path": str(image)}],
                                             "options": {**_OPTS, "carve": False}}
                   ).status_code == 200
    assert _wait(cl)["stage"] == "done"
    archive.close_zips()
    monkeypatch.setattr(archive, "_BITLOCKER", {})           # the app was restarted
    [row] = cl.get("/api/sources").get_json()
    assert (row["status"], row["locked_by"]) == ("locked", "BitLocker")
    r = cl.post("/api/source/unlock", json={"name": image.name,
                                            "bitlocker": {"secret": "not it"}})
    assert r.get_json() == {"ok": False, "wrong": True}
    r = cl.post("/api/source/unlock", json={"name": image.name,
                                            "bitlocker": {"key_file": str(bek)}})
    assert r.get_json() == {"ok": True, "wrong": False}
    assert [s["status"] for s in cl.get("/api/sources").get_json()] == ["ok"]
