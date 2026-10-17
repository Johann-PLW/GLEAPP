"""Several member images read as one source: the disks of a RAID set.

The fixtures are qnxprobe's own (tests/fixtures/raid at qnxprobe v1.70, commit
98c4fc8861efd819913679a6e6462479b6bea9e4), written by its make_raid_fixtures.sh with
mdadm 4.5 and lvm2 2.03.31 on Linux 7.0. ``raid-trees.sha256`` holds the SHA-256 of
every file the kernel presented on each array when it was made, so a file walked
from an array GLEAPP read is checked against the bytes Linux itself read, not
against another run of the same reader. The trees hold no pictures, so the sources
here register every file (``include_other``).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gleapp import archive
from gleapp.case import image_set_source, open_case, parse_source_spec
from gleapp.cli import main as cli_main
from gleapp.pipeline import ingest_sources

sys.path.insert(0, str(Path(__file__).parent))
from fatwriter import build_fat32                      # pylint: disable=import-error,wrong-import-position

RAID = Path(__file__).parent / "fixtures" / "raid"
ROOT = Path(__file__).resolve().parents[1]
KNOWN = dict(reversed(line.split(None, 1)) for line in
             (RAID / "raid-trees.sha256").read_text(encoding="utf-8").splitlines())
KNOWN = {k.strip(): v for k, v in KNOWN.items()}


def _members(folder: Path, stem: str, roles) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i in roles:
        p = folder / f"{stem}-{i}.img"
        p.write_bytes(gzip.decompress((RAID / f"{stem}-{i}.img.gz").read_bytes()))
        out.append(p)
    return out


def _ingest(case_dir: Path, src):
    case = open_case(case_dir, create=True, examiner="t")
    src.include_other = True
    ingest_sources(case, [src])
    return case


def _walked(case, name: str) -> dict[str, str]:
    """{stored name: sha256 of the bytes read back} for every row of ``name``."""
    rec = archive.source_record(case, name)
    out = {}
    for r in case.db.iter_files("source = ?", (name,)):
        with archive.local_copy(case.root, rec, r) as local:
            out[r["orig_path"]] = hashlib.sha256(Path(local).read_bytes()).hexdigest()
    return out


def _tree(walked: dict[str, str], prefix: str, tree: str) -> dict[str, str]:
    """The rows under ``prefix`` keyed as raid-trees.sha256 keys them."""
    return {f"{tree}/{k[len(prefix):]}": v for k, v in walked.items() if k.startswith(prefix)}


def _expected(tree: str) -> dict[str, str]:
    return {k: v for k, v in KNOWN.items() if k.startswith(tree + "/")}


# ---- a set given together ----------------------------------------------------
def test_an_explicit_set_reads_its_array_to_the_bytes_the_kernel_presented(tmp_path):
    files = _members(tmp_path / "ev", "md-r5-ls", range(4))
    case = _ingest(tmp_path / "case", image_set_source(files))
    name = files[0].name
    walked = _walked(case, name)
    # md_<name>_<first eight hex digits of the uuid>, as qnxprobe names the array
    prefix = "md_qp_qp21_3090d8cb/lba0/"
    assert set(walked) == {f"{prefix}{k[len('md-r5-ls/'):]}" for k in _expected("md-r5-ls")}
    assert _tree(walked, prefix, "md-r5-ls") == _expected("md-r5-ls")

    rec = archive.source_record(case, name)
    # the first member stays where a single image's path is, for older readers
    assert rec["path"] == str(files[0].resolve())
    assert [m["path"] for m in rec["members"]] == [str(f.resolve()) for f in files]
    assert not any(m["found"] for m in rec["members"])
    assert all(m["media_hash"].startswith("head-tail-sha256:") for m in rec["members"])
    assert rec["media_hash"].startswith("member-set-sha256:")
    # a member read into an array is not reported as a volume left unread
    assert rec["volumes_not_read"] == ""
    assert any("raid5 left-symmetric" in line for line in rec["raid"])
    status = next(s for s in archive.source_status(case) if s["name"] == name)
    assert status["status"] == "ok"
    assert [m["status"] for m in status["members"]] == ["ok"] * 4
    case.close()


def test_a_raid5_set_missing_one_member_is_read_degraded_from_parity(tmp_path):
    # role 2 is not given; RAID5 can lose one member, so its chunks are rebuilt
    files = _members(tmp_path / "ev", "md-r5ls-090", (0, 1, 3))
    case = _ingest(tmp_path / "case", image_set_source(files))
    walked = _walked(case, files[0].name)
    prefix = "md_8f61b69f/lba0/"                       # metadata 0.90 records no name
    assert _tree(walked, prefix, "md-r5ls-090") == _expected("md-r5ls-090")
    rec = archive.source_record(case, files[0].name)
    assert any("rebuilt from parity" in line for line in rec["raid"])
    case.close()


def test_a_lone_raid5_member_is_recorded_as_not_read_with_its_reason(tmp_path):
    """One member given alone is an ordinary image source: its array cannot be
    read, and the case says so rather than registering nothing in silence."""
    files = _members(tmp_path / "ev", "md-r5-ls", (0,))
    case = open_case(tmp_path / "case", create=True, examiner="t")
    sources, _ = parse_source_spec(files[0])
    assert sources[0].members == []
    ingest_sources(case, sources)
    rec = archive.source_record(case, files[0].name)
    assert "members" not in rec
    assert "md RAID member" in rec["volumes_not_read"]
    case.close()


def test_volume_groups_spread_over_the_members_are_read_as_their_volumes(tmp_path):
    files = _members(tmp_path / "ev", "lvm-raid-multi", range(5))
    case = _ingest(tmp_path / "case", image_set_source(files))
    walked = _walked(case, files[0].name)
    for lv in ("r5", "r6", "r10"):
        got = _tree(walked, f"lvm_qpraid2_{lv}/lba0/", f"lvm-raid-multi-{lv}")
        assert got == _expected(f"lvm-raid-multi-{lv}"), lv
    rec = archive.source_record(case, files[0].name)
    # each physical volume is read as part of the volume group, not left unread
    assert rec["volumes_not_read"] == ""
    case.close()


def test_a_mirror_whose_data_starts_at_the_first_byte_is_read_from_its_first_member(tmp_path):
    """Metadata 0.90 sits at the end of a RAID1 member, so the member itself is the
    filesystem. As qnxprobe's own command line reads it (``same_as``), the array is
    read from its first member, as that member's own volume; the other member holds
    the same data and is listed as a copy of it, not walked again."""
    files = _members(tmp_path / "ev", "md-r1-090", (0, 1))
    case = _ingest(tmp_path / "case", image_set_source(files))
    walked = _walked(case, files[0].name)
    assert {k.split("/", 1)[0] for k in walked} == {files[0].name}
    assert _tree(walked, f"{files[0].name}/lba0/", "md-r1-090") == _expected("md-r1-090")
    rec = archive.source_record(case, files[0].name)
    assert any("not read again" in line for line in rec["raid"])
    assert rec["set_notes"] == [f"{files[1].name}/lba0 is a copy of {files[0].name}, a member "
                                "of the same RAID1 array, whose files are listed; it is not "
                                "walked again"]
    assert rec["volumes_not_read"] == ""
    case.close()


def test_a_path_that_is_not_a_disk_image_is_refused_as_a_member(tmp_path):
    files = _members(tmp_path / "ev", "md-r1", (0, 1))
    note = tmp_path / "ev" / "notes.txt"
    note.write_text("not a disk", encoding="utf-8")
    with pytest.raises(ValueError, match="not a disk image"):
        image_set_source([files[0], note])


def test_a_job_file_lists_the_members_of_one_source(tmp_path):
    files = _members(tmp_path / "ev", "md-r1", (0, 1))
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"sources": [{"name": "Array", "members": [
        str(f) for f in files], "include_other": True}]}), encoding="utf-8")
    sources, _ = parse_source_spec(job)
    assert len(sources) == 1 and sources[0].name == "Array"
    assert sources[0].members == [str(f.resolve()) for f in files]
    case = open_case(tmp_path / "case", create=True, examiner="t")
    ingest_sources(case, sources)
    assert _tree(_walked(case, "Array"), "md_qp_qp10_01173252/lba0/", "md-r1") == \
        _expected("md-r1")
    case.close()


# ---- finding the other members ------------------------------------------------
def _with_decoys(folder: Path) -> list[Path]:
    files = _members(folder, "md-r5-ls", range(4))
    _members(folder / "other", "md-r1", (0,))
    shutil.move(folder / "other" / "md-r1-0.img", folder / "decoy-other-array.img")
    (folder / "other").rmdir()
    (folder / "notes.txt").write_text("not a disk", encoding="utf-8")
    (folder / "fat.img").write_bytes(build_fat32([("A", "TXT", b"x", (2020, 1, 1, 0, 0, 0))]))
    bad = bytearray(files[3].read_bytes())
    bad[4096 + 0x30] ^= 0xFF                           # inside the 1.2 superblock: its checksum fails
    (folder / "damaged-copy.img").write_bytes(bytes(bad))
    return files


def test_discovery_adds_the_members_and_names_every_decoy(tmp_path):
    files = _with_decoys(tmp_path / "ev")
    found = archive.find_members(files[2])
    assert found["searched"]
    assert sorted(Path(a["path"]).name for a in found["added"]) == [
        "md-r5-ls-0.img", "md-r5-ls-1.img", "md-r5-ls-3.img"]
    assert found["paths"][0] == str(files[2])
    codes = {Path(u["path"]).name: u["code"] for u in found["unused"]}
    assert codes["decoy-other-array.img"] == "different-array"
    assert codes["notes.txt"] == "too-small"
    assert codes["fat.img"] == "no-raid-metadata"
    assert codes["damaged-copy.img"] == "bad-metadata"
    assert found["missing"] == [] and found["warnings"] == []


def test_discovery_looks_for_nothing_beside_an_ordinary_image(tmp_path):
    folder = tmp_path / "ev"
    folder.mkdir()
    plain = folder / "plain.img"
    plain.write_bytes(build_fat32([("A", "TXT", b"x", (2020, 1, 1, 0, 0, 0))]))
    _members(folder, "md-r5-ls", (0,))
    found = archive.find_members(plain)
    assert not found["searched"] and found["added"] == [] and found["unused"] == []


def test_the_command_line_reads_the_members_found_and_prints_them(tmp_path, capsys):
    files = _with_decoys(tmp_path / "ev")
    case_dir = tmp_path / "case"
    assert cli_main(["-c", str(case_dir), "ingest", str(files[2]), "--no-process"]) == 0
    out = capsys.readouterr().out
    assert "not used  decoy-other-array.img" in out
    assert out.count("(found beside the image given)") == 3
    case = open_case(case_dir)
    rec = archive.source_record(case, files[2].name)
    assert [Path(m["path"]).name for m in rec["members"]] == [
        "md-r5-ls-2.img", "md-r5-ls-0.img", "md-r5-ls-1.img", "md-r5-ls-3.img"]
    assert [m["found"] for m in rec["members"]] == [False, True, True, True]
    case.close()


def test_the_command_line_reads_a_degraded_set_from_the_members_found(tmp_path, capsys):
    files = _members(tmp_path / "ev", "md-r5ls-090", (0, 1, 3))
    case_dir = tmp_path / "case"
    assert cli_main(["-c", str(case_dir), "ingest", str(files[1]), "--no-process"]) == 0
    assert "missing" in capsys.readouterr().out
    case = open_case(case_dir)
    rec = archive.source_record(case, files[1].name)
    assert len(rec["members"]) == 3
    # the array was read, rebuilt from parity, not left among the volumes not read
    assert [v["label"] for v in json.loads(rec["volumes"])] == ["md_8f61b69f/lba0"]
    assert rec["volumes_not_read"] == ""
    assert rec["discovery"]["choice"] == "used"
    case.close()


def test_the_command_line_prints_the_missing_role_and_its_rebuild(tmp_path, capsys):
    # role 2 is not given; the console says so, as qnxprobe and DLEAPP do
    files = _members(tmp_path / "ev", "md-r5ls-090", (0, 1, 3))
    case_dir = tmp_path / "case"
    assert cli_main(["-c", str(case_dir), "ingest", *map(str, files), "--no-process"]) == 0
    out = capsys.readouterr().out
    assert "    role 2: missing" in out
    assert "1 member(s) missing: their chunks are rebuilt from parity" in out
    # after the Sources list, before the count, and the same lines the case keeps
    assert out.index("Sources (1):") < out.index("role 2: missing") < out.index("Registered")
    case = open_case(case_dir)
    rec = archive.source_record(case, files[0].name)
    assert all(line in out for line in rec["raid"] if line)
    case.close()


def test_the_command_line_prints_a_member_left_out_and_why(tmp_path, capsys):
    files = _members(tmp_path / "ev", "md-r5-ls", range(4))
    copy = tmp_path / "ev" / "copy-of-3.img"
    shutil.copy(files[3], copy)                        # a second holder of role 3
    case_dir = tmp_path / "case"
    assert cli_main(["-c", str(case_dir), "ingest", *map(str, files), str(copy),
                     "--no-process"]) == 0
    out = capsys.readouterr().out
    assert "  left out: copy-of-3.img: role 3 again, held by md-r5-ls-3.img" in out


def test_the_raid_report_names_no_option_gleapp_does_not_have(tmp_path, capsys):
    """Two members of a four-member RAID5: the reader's heading counts no array as
    assembled (qnxprobe 1.72), and its hint naming its own --find-members option,
    which GLEAPP does not have, is not printed or kept."""
    files = _members(tmp_path / "ev", "md-r5-ls", (0, 1))
    case_dir = tmp_path / "case"
    assert cli_main(["-c", str(case_dir), "ingest", *map(str, files), "--no-process"]) == 0
    out = capsys.readouterr().out
    assert "RAID: 1 array(s) or spanned volume(s) found in the images given, " \
        "0 of them assembled" in out
    assert "not read from the images given: role(s) 2, 3 of 4 are in none of them" in out
    assert "add --find-members" not in out
    case = open_case(case_dir)
    rec = archive.source_record(case, files[0].name)
    assert not any("--find-members" in line for line in rec["raid"])
    case.close()


def test_the_command_line_prints_no_raid_report_for_an_ordinary_image(tmp_path, capsys):
    image = tmp_path / "ev" / "fat.img"
    image.parent.mkdir(parents=True)
    image.write_bytes(build_fat32([("A", "TXT", b"x", (2020, 1, 1, 0, 0, 0))]))
    assert cli_main(["-c", str(tmp_path / "case"), "ingest", str(image), "--no-process"]) == 0
    assert "RAID:" not in capsys.readouterr().out


def test_no_find_members_reads_the_image_given_alone(tmp_path, capsys):
    files = _with_decoys(tmp_path / "ev")
    case_dir = tmp_path / "case"
    assert cli_main(["-c", str(case_dir), "ingest", str(files[2]), "--no-process",
                     "--no-find-members"]) == 0
    assert "Looking for the other member images" not in capsys.readouterr().out
    case = open_case(case_dir)
    rec = archive.source_record(case, files[2].name)
    assert "members" not in rec
    case.close()


def test_the_command_line_takes_several_images_as_one_source(tmp_path):
    files = _members(tmp_path / "ev", "md-r1", (0, 1))
    case_dir = tmp_path / "case"
    assert cli_main(["-c", str(case_dir), "ingest", *map(str, files), "--no-process"]) == 0
    case = open_case(case_dir)
    recs = archive.source_records(case)
    assert list(recs) == [files[0].name]
    assert [m["found"] for m in recs[files[0].name]["members"]] == [False, False]
    case.close()


# ---- the web: picking, finding and confirming -----------------------------------
def test_the_evidence_picker_allows_several_files(monkeypatch):
    from gleapp.web import app as appmod               # pylint: disable=import-outside-toplevel
    seen = {}

    class FakeWin:
        def create_file_dialog(self, _kind, allow_multiple=False, **_given):
            seen["multiple"] = allow_multiple
            return ["C:/ev/disk0.img", "C:/ev/disk1.img"]

    class FakeWebview:
        OPEN_DIALOG = "open"
        FOLDER_DIALOG = "folder"
        windows = [FakeWin()]

    monkeypatch.setitem(sys.modules, "webview", FakeWebview)
    if sys.platform == "win32":
        # several files on Windows go through the STA dialog, not pywebview's
        from gleapp import _win_open_dialog  # pylint: disable=import-outside-toplevel

        def fake_open_files(file_types, multiple=True, **_given):
            assert any("*.xml" in ft for ft in file_types)
            return FakeWin().create_file_dialog("open", allow_multiple=multiple)
        monkeypatch.setattr(_win_open_dialog, "open_files", fake_open_files)
    cl = appmod.create_app(None, native=True).test_client()
    body = cl.post("/api/pick", json={"kind": "ingestfile", "multiple": True}).get_json()
    assert seen["multiple"] is True
    assert body == {"path": "C:/ev/disk0.img", "paths": ["C:/ev/disk0.img", "C:/ev/disk1.img"]}
    # a picker that does not ask for several keeps the call it always made
    body = cl.post("/api/pick", json={"kind": "ingestfile"}).get_json()
    assert seen["multiple"] is False and body == {"path": "C:/ev/disk0.img"}


def test_the_web_finds_members_and_ingests_the_set_as_one_source(tmp_path):
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    files = _with_decoys(tmp_path / "ev")
    cl = create_app(None).test_client()
    found = cl.post("/api/source/members", json={"path": str(files[0])}).get_json()
    assert len(found["added"]) == 3 and found["unused"]
    assert cl.post("/api/case/create", json={"path": str(tmp_path / "case"), "name": "c",
                                             "examiner": "t"}).status_code == 200
    r = cl.post("/api/case/ingest", json={"sources": [{
        "name": files[0].name, "path": found["paths"][0], "paths": found["paths"],
        "found": [a["path"] for a in found["added"]]}], "options": {"screen": False}})
    assert r.status_code == 200, r.get_json()
    for _ in range(600):                               # capped: at most 60 s
        if not cl.get("/api/job").get_json()["running"]:
            break
        import time                                    # pylint: disable=import-outside-toplevel
        time.sleep(0.1)
    ctx = cl.get("/api/context").get_json()
    src = next(s for s in ctx["archive_sources"] if s["name"] == files[0].name)
    assert [m["found"] for m in src["members"]] == [False, True, True, True]
    assert src["volumes"] and "md_qp_qp21_3090d8cb/lba0" in src["volumes"]


def _node() -> str | None:
    node = shutil.which("node")
    if node is None and os.environ.get("CI"):
        pytest.fail("node is needed to run the confirmation's choice in CI")
    return node


def test_the_confirmation_turns_each_answer_into_the_sources_it_names():
    """``membersSources`` in app.js is what the dialog's answer becomes: "use" one
    source of the image picked and the members found, "only" the image picked
    alone, "cancel" nothing."""
    node = _node()
    if node is None:
        pytest.skip("node is not installed")
    js = (ROOT / "gleapp/web/static/app.js").read_text(encoding="utf-8")
    start = js.index("function membersSources(")
    body = js[start:js.index("\n}\n", start) + 3]
    # the regexes it uses, read from app.js so the copy cannot drift from them
    regexes = "".join(re.search(rf"^const {name} = .*$", js, re.M).group(0) + "\n"
                      for name in ("SPEC_RE", "ARCHIVE_RE"))
    script = (regexes + body
              + """
const found = {searched: true, paths: ["/e/m2.img", "/e/m0.img", "/e/m1.img"],
               added: [{path: "/e/m0.img"}, {path: "/e/m1.img"}],
               unused: [{path: "/e/x.img", why: "a member of a different array"}],
               missing: []};
const none = {searched: true, paths: ["/e/m2.img"], added: [], unused: [],
              missing: ["role 3 is in no file"]};
console.log(JSON.stringify({
  use: membersSources(["/e/m2.img"], found, "use"),
  only: membersSources(["/e/m2.img"], found, "only"),
  cancel: membersSources(["/e/m2.img"], found, "cancel"),
  picked: membersSources(["/e/a.img", "/e/b.img"], null, "only"),
  nothing: membersSources(["/e/m2.img"], none, "only"),
  ordinary: membersSources(["/e/plain.img"], null, "only"),
}));
""")
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True,
                         stdin=subprocess.DEVNULL, timeout=60).stdout
    got = json.loads(out)
    searched = {"added": ["/e/m0.img", "/e/m1.img"],
                "unused": [{"path": "/e/x.img", "why": "a member of a different array"}],
                "missing": []}
    assert got["use"] == [{"kind": "set", "path": "/e/m2.img", "paths": found_paths(),
                           "found": ["/e/m0.img", "/e/m1.img"],
                           "discovery": {"choice": "used", **searched}}]
    # the examiner's choice travels with the source, so the case records it
    assert got["only"] == [{"kind": "folder", "path": "/e/m2.img",
                            "discovery": {"choice": "only", **searched}}]
    assert got["cancel"] == []
    assert got["picked"] == [{"kind": "set", "path": "/e/a.img",
                              "paths": ["/e/a.img", "/e/b.img"], "found": []}]
    assert got["nothing"] == [{"kind": "folder", "path": "/e/m2.img", "discovery": {
        "choice": "nothing-added", "added": [], "unused": [],
        "missing": ["role 3 is in no file"]}}]
    assert got["ordinary"] == [{"kind": "folder", "path": "/e/plain.img"}]


def found_paths() -> list[str]:
    return ["/e/m2.img", "/e/m0.img", "/e/m1.img"]


def test_the_confirmation_is_wired_to_the_evidence_pickers():
    js = (ROOT / "gleapp/web/static/app.js").read_text(encoding="utf-8")
    assert 'data-r="use"' in js and 'data-r="only"' in js and 'data-r="cancel"' in js
    assert "Only the file I picked" in js and "Use them" in js
    assert '$("#addFile").onclick = async () => addSource(await pickMany("ingestfile"));' in js
    assert '$("#aeAddFile").onclick = async () => aeAddSource(await pickMany("ingestfile"));' in js
    # shown only when the search would add files
    assert ('!(found.added || []).length)\n    return membersSources(clean, found && '
            '!found.error ? found : null, "only");') in js


# ---- relink and status --------------------------------------------------------
def test_a_set_relinks_from_one_member_and_reads_again(tmp_path):
    files = _members(tmp_path / "ev", "md-r5-ls", range(4))
    case = _ingest(tmp_path / "case", image_set_source(files))
    name = files[0].name
    before = _walked(case, name)
    moved = tmp_path / "moved"
    # Reading the rows leaves the members open in the image cache, as a single
    # image's reads do; Windows cannot move a folder with a file open in it, so
    # they are let go first, as the single-image relink tests do.
    archive.close_zips()
    shutil.move(tmp_path / "ev", moved)
    status = next(s for s in archive.source_status(case) if s["name"] == name)
    assert status["status"] == "missing"
    assert {m["status"] for m in status["members"]} == {"missing"}
    with pytest.raises(archive.ArchiveUnavailable):
        _walked(case, name)
    got = archive.relink_source(case, name, moved / "md-r5-ls-2.img")
    assert got["status"] == "ok"
    assert [m["path"] for m in got["members"]] == [str((moved / f.name).resolve())
                                                   for f in files]
    assert archive.source_record(case, name)["path"] == str((moved / files[0].name).resolve())
    assert _walked(case, name) == before
    case.close()


def test_a_set_relinks_from_paths_given_in_any_order_and_keeps_its_own(tmp_path):
    files = _members(tmp_path / "ev", "md-r5-ls", range(4))
    case = _ingest(tmp_path / "case", image_set_source(files))
    name = files[0].name
    moved = tmp_path / "moved"
    moved.mkdir()
    # renamed, so only what each file is can match it to its member
    renamed = []
    archive.close_zips()                               # nothing of the set is held open
    for i, f in enumerate(reversed(files)):
        dest = moved / f"disk{i}.img"
        shutil.move(f, dest)
        renamed.append(dest)
    got = archive.relink_source(case, name, [str(p) for p in renamed])
    assert [Path(m["path"]).name for m in got["members"]] == [
        "disk3.img", "disk2.img", "disk1.img", "disk0.img"]
    assert got["status"] == "ok"
    case.close()


def test_a_set_refuses_a_relink_that_lacks_a_member(tmp_path):
    files = _members(tmp_path / "ev", "md-r5-ls", range(4))
    case = _ingest(tmp_path / "case", image_set_source(files))
    name = files[0].name
    other = _members(tmp_path / "other", "md-r1", (0,))[0]
    archive.close_zips()
    shutil.copy(other, tmp_path / "ev" / "md-r5-ls-3.img")    # same name, another disk
    with pytest.raises(ValueError, match="md-r5-ls-3.img: no image given or beside it"):
        archive.relink_source(case, name, files[0])
    rec = archive.source_record(case, name)
    assert [m["path"] for m in rec["members"]] == [str(f.resolve()) for f in files]
    case.close()


def test_a_set_can_be_copied_into_the_case(tmp_path):
    files = _members(tmp_path / "ev", "md-r5-ls", range(4))
    case = _ingest(tmp_path / "case", image_set_source(files))
    name = files[0].name
    before = _walked(case, name)
    assert archive.stage_source(case, name) == len(before)
    staged = {r["orig_path"]: hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest()
              for r in case.db.iter_files("source = ?", (name,))}
    assert staged == before
    # and dropped again, which the set allows only while it still holds every file
    assert archive.unstage_source(case, name) == len(before)
    assert _walked(case, name) == before
    # ext4 reports no free space, so a scoped carve reads every part, as it reads
    # a single ext4 image whole; it must not cross from one part into the next
    assert archive.carve_source(case, name, unallocated_only=True) == 0
    assert case.db.get_meta(f"archive:{name}:carve_scope") is None
    case.close()


def _jpeg(colour) -> bytes:
    from PIL import Image                              # pylint: disable=import-outside-toplevel
    import io                                          # pylint: disable=import-outside-toplevel
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), colour).save(buf, "JPEG")
    return buf.getvalue()


def test_a_set_is_carved_part_by_part_in_its_own_address_space(tmp_path):
    """Two images with a deleted-looking picture each in free space: a scoped carve
    reads each part's free space at its place in the set, and each hit reads back
    as the picture."""
    red, green = _jpeg((200, 30, 30)), _jpeg((30, 200, 30))
    folder = tmp_path / "ev"
    folder.mkdir()
    paths = []
    for i, pic in enumerate((red, green)):
        vol = bytearray(build_fat32([("KEEP", "TXT", b"kept", (2020, 1, 1, 0, 0, 0))]))
        at = len(vol) - 64 * 1024                      # in clusters the FAT marks free
        vol[at:at + len(pic)] = pic
        p = folder / f"disk{i}.img"
        p.write_bytes(bytes(vol))
        paths.append(p)
    case = _ingest(tmp_path / "case", image_set_source(paths))
    name = paths[0].name
    # the deleted-record pass the gallery runs before a carve reads a set too
    assert archive.recover_deleted(case, name)[0] == 0
    added = archive.carve_source(case, name, unallocated_only=True)
    assert "space no volume claims" in case.db.get_meta(f"archive:{name}:carve_scope")
    rec = archive.source_record(case, name)
    carved = [r for r in case.db.iter_files("source = ? AND origin = 'carve'", (name,))]
    assert added == len(carved) == 2
    got = {}
    for r in carved:
        with archive.local_copy(case.root, rec, r) as local:
            got[int(r["member_offset"])] = Path(local).read_bytes()
    first, second = sorted(got)
    assert got[first] == red and got[second] == green
    assert second >= archive._SET_ALIGN               # pylint: disable=protected-access
    case.close()


# ---- what did not change -------------------------------------------------------
def test_a_single_image_case_reads_as_it_did(tmp_path):
    """A case of one image, as every case made before sets existed is: no member
    keys in its record or status, its path is what is opened, and its rows read."""
    folder = tmp_path / "ev"
    folder.mkdir()
    image = folder / "usb.img"
    image.write_bytes(build_fat32([("A", "TXT", b"one file", (2020, 1, 1, 0, 0, 0))]))
    case = open_case(tmp_path / "case", create=True, examiner="t")
    sources, _ = parse_source_spec(image)
    sources[0].include_other = True
    ingest_sources(case, sources)
    assert not [r["key"] for r in case.db.conn.execute(
        "SELECT key FROM meta WHERE key LIKE '%:members' OR key LIKE '%:raid'")]
    rec = archive.source_record(case, "usb.img")
    assert "members" not in rec and "raid" not in rec
    assert archive.image_ref(rec) == str(image.resolve())
    status = archive.source_status(case)[0]
    assert "members" not in status and status["status"] == "ok"
    assert _walked(case, "usb.img") == {"lba0/A.TXT": hashlib.sha256(b"one file").hexdigest()}
    case.close()


# ---- one disk named twice --------------------------------------------------------
def test_paths_that_open_the_same_disk_are_one_member_and_said_so(tmp_path):
    """The same path twice, another spelling of it, a hard link, a symbolic link, the
    second segment of a split set and the second segment of an E01 set each open a
    disk already given: read once, and the source says which."""
    from ewfwriter import write_ewf                    # pylint: disable=import-outside-toplevel,import-error
    ev = tmp_path / "ev"
    files = _members(ev, "md-r5-ls", range(4))
    raw0 = files[0].read_bytes()
    files[0].unlink()
    seg1, seg2 = ev / "m0.001", ev / "m0.002"           # member 0 as a split raw set
    seg1.write_bytes(raw0[:len(raw0) // 2])
    seg2.write_bytes(raw0[len(raw0) // 2:])
    e01 = [Path(p) for p in write_ewf(ev, "m1e", files[1].read_bytes(),
                                       chunks_per_segment=400)]
    assert len(e01) >= 2
    hard = ev / "hard-m2.img"
    os.link(files[2], hard)
    soft = ev / "soft-m3.img"
    soft.symlink_to(files[3])
    spelled = ev / "." / files[3].name
    given = [seg1, seg2, e01[0], e01[1], files[2], hard, files[3], soft, spelled, files[3]]
    src = image_set_source(given)
    assert [Path(m).name for m in src.members] == ["m0.001", "m1e.E01", "md-r5-ls-2.img",
                                                   "md-r5-ls-3.img"]
    assert len(src.notes) == 6, src.notes
    # a later file of a set is a segment of it, not the same disk (qnxprobe 1.72)
    assert src.notes[0] == "m0.002 is read as a segment of the split set opened from m0.001, " \
        "not on its own"
    assert src.notes[1] == "m1e.E02 is read as a segment of the EWF set opened from m1e.E01, " \
        "not on its own"
    assert "hard-m2.img is the same disk as md-r5-ls-2.img" in src.notes[2]
    case = _ingest(tmp_path / "case", src)
    rec = archive.source_record(case, "m0.001")
    assert rec["set_notes"] == src.notes
    assert _tree(_walked(case, "m0.001"), "md_qp_qp21_3090d8cb/lba0/", "md-r5-ls") == \
        _expected("md-r5-ls")
    case.close()


def test_the_command_line_and_the_web_say_a_disk_given_twice_is_read_once(tmp_path, capsys):
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    files = _members(tmp_path / "ev", "md-r1", (0, 1))
    case_dir = tmp_path / "case"
    assert cli_main(["-c", str(case_dir), "ingest", str(files[0]), str(files[1]),
                     str(files[0]), "--no-process"]) == 0
    assert "Note: md-r1-0.img was given twice and is read once." in capsys.readouterr().out
    case = open_case(case_dir)
    assert len(archive.source_record(case, files[0].name)["members"]) == 2
    case.close()
    cl = create_app(None).test_client()
    got = cl.post("/api/source/members", json={"paths": [str(files[0]), str(files[1]),
                                                         str(files[1])]}).get_json()
    assert [Path(p).name for p in got["paths"]] == ["md-r1-0.img", "md-r1-1.img"]
    assert got["same"] == ["md-r1-1.img was given twice and is read once"]


# ---- a disk that lacks nothing is not searched beside ------------------------------
def _opened_beside(monkeypatch, folder: Path, start: Path, fn):
    """Run ``fn`` and return the names of the files in ``folder``, other than
    ``start``, that anything opened while it ran."""
    import builtins                                    # pylint: disable=import-outside-toplevel
    seen = []
    real_open, real_os_open = builtins.open, os.open

    def spy_open(p, *a, **k):
        if isinstance(p, (str, bytes, os.PathLike)):
            seen.append(os.fspath(p))
        return real_open(p, *a, **k)

    def spy_os_open(p, *a, **k):
        seen.append(os.fspath(p))
        return real_os_open(p, *a, **k)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(os, "open", spy_os_open)
    try:
        result = fn()
    finally:
        monkeypatch.setattr(builtins, "open", real_open)
        monkeypatch.setattr(os, "open", real_os_open)
    names = {Path(os.fsdecode(p)).name for p in seen
             if Path(os.fsdecode(p)).parent == folder and Path(os.fsdecode(p)).name != start.name}
    return result, names


def test_a_one_disk_lvm_image_opens_nothing_beside_it(tmp_path, monkeypatch, capsys):
    """lvm-fixture holds a whole volume group (two physical volumes, both in it).
    The search for other members used to open every disk image beside it; it now
    opens none, says nothing and records nothing."""
    folder = tmp_path / "ev"
    folder.mkdir()
    image = folder / "lvm.img"
    image.write_bytes(gzip.decompress((Path(__file__).parent / "fixtures" /
                                       "lvm-fixture.img.gz").read_bytes()))
    _members(folder, "md-r5-ls", (0, 1, 2))
    # the vendored search on its own opens the three images beside it
    _r, before = _opened_beside(monkeypatch, folder, image,
                                lambda: archive.qnxprobe.raid_siblings(str(image)))
    assert before == {"md-r5-ls-0.img", "md-r5-ls-1.img", "md-r5-ls-2.img"}
    found, after = _opened_beside(monkeypatch, folder, image,
                                  lambda: archive.find_members(image))
    assert after == set()
    assert not found["searched"] and not found["incomplete"] and found["lines"] == []
    case_dir = tmp_path / "case"
    assert cli_main(["-c", str(case_dir), "ingest", str(image), "--no-process"]) == 0
    out = capsys.readouterr().out
    assert "member" not in out.lower() and "raid" not in out.lower()
    case = open_case(case_dir)
    rec = archive.source_record(case, "lvm.img")
    assert "discovery" not in rec and "members" not in rec
    case.close()


# ---- the examiner's choice is in the case ------------------------------------------
def test_a_search_that_found_nothing_is_recorded(tmp_path):
    files = _members(tmp_path / "ev", "md-r5-ls", (0,))
    case_dir = tmp_path / "case"
    assert cli_main(["-c", str(case_dir), "ingest", str(files[0]), "--no-process"]) == 0
    case = open_case(case_dir)
    rec = archive.source_record(case, files[0].name)
    assert rec["discovery"]["choice"] == "nothing-added"
    assert rec["discovery"]["missing"]
    assert rec["discovery_lines"][0].startswith("The other member images were looked for")
    status = archive.source_status(case)[0]
    assert status["discovery_lines"] == rec["discovery_lines"]
    case.close()


def test_no_find_members_is_recorded_when_the_image_lacks_members(tmp_path):
    files = _members(tmp_path / "ev", "md-r5-ls", range(4))
    case_dir = tmp_path / "case"
    assert cli_main(["-c", str(case_dir), "ingest", str(files[0]), "--no-process",
                     "--no-find-members"]) == 0
    case = open_case(case_dir)
    rec = archive.source_record(case, files[0].name)
    assert rec["discovery"]["choice"] == "not-searched"
    assert "members" not in rec
    case.close()


def test_only_the_file_picked_is_recorded_from_the_web_and_in_the_report(tmp_path):
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    from gleapp import lava                            # pylint: disable=import-outside-toplevel
    files = _members(tmp_path / "ev", "md-r5-ls", range(4))
    cl = create_app(None).test_client()
    found = cl.post("/api/source/members", json={"path": str(files[0])}).get_json()
    assert len(found["added"]) == 3
    assert cl.post("/api/case/create", json={"path": str(tmp_path / "case"), "name": "c",
                                             "examiner": "t"}).status_code == 200
    r = cl.post("/api/case/ingest", json={"sources": [{
        "name": files[0].name, "path": str(files[0]),
        "discovery": {"choice": "only", "added": [a["path"] for a in found["added"]],
                      "unused": found["unused"], "missing": found["missing"]}}],
        "options": {"screen": False}})
    assert r.status_code == 200, r.get_json()
    for _ in range(600):                               # capped: at most 60 s
        if not cl.get("/api/job").get_json()["running"]:
            break
        import time                                    # pylint: disable=import-outside-toplevel
        time.sleep(0.1)
    src = next(s for s in cl.get("/api/context").get_json()["archive_sources"]
               if s["name"] == files[0].name)
    assert "members" not in src
    assert src["discovery"]["choice"] == "only"
    assert src["discovery"]["found"] == ["md-r5-ls-1.img", "md-r5-ls-2.img", "md-r5-ls-3.img"]
    assert any("chose to read only the image given" in line for line in src["discovery_lines"])
    # no folder of the examiner's reaches the record
    assert str(tmp_path) not in json.dumps(src["discovery"])
    case = open_case(tmp_path / "case")
    dest = tmp_path / "lava"
    dest.mkdir()
    lava._write_device_info(case, dest, tz_name=None)   # pylint: disable=protected-access
    page = (dest / "DeviceInfo.html").read_text(encoding="utf-8")
    assert "Member image search" in page and "chose to read only the image given" in page
    assert str(tmp_path) not in page
    case.close()


# ---- what opens a volume inside a member or an array -------------------------------
LUKS_PASSPHRASE = "correct horse battery staple"


def _luks_known() -> dict[str, str]:
    lines = (Path(__file__).parent / "fixtures" / "luks-fixture.sha256").read_text(
        encoding="utf-8").splitlines()
    return {p.strip(): h for h, p in (line.split(None, 1) for line in lines)}


def _raid5_holding(folder: Path, payload: bytes) -> list[Path]:
    """md-r5-ls's four members with ``payload`` written as the array's bytes: each
    32 KiB chunk on the disk the left-symmetric layout gives it, and each stripe's
    parity the XOR of its data (md's own rule for RAID5, layout 2). Checked by
    reading the array back through the vendored reader before it is used."""
    files = _members(folder, "md-r5-ls", range(4))
    disks = [bytearray(f.read_bytes()) for f in files]
    n, chunk, data_at, per_disk = 4, 32 * 1024, 1 << 20, 3 << 20
    payload = payload.ljust((n - 1) * per_disk, b"\0")
    for stripe in range(per_disk // chunk):
        parity_disk = (n - 1) - stripe % n
        parity = bytearray(chunk)
        for i in range(n - 1):
            piece = payload[(stripe * (n - 1) + i) * chunk:(stripe * (n - 1) + i + 1) * chunk]
            disk = (parity_disk + 1 + i) % n
            disks[disk][data_at + stripe * chunk:data_at + (stripe + 1) * chunk] = piece
            parity = bytearray(a ^ b for a, b in zip(parity, piece))
        disks[parity_disk][data_at + stripe * chunk:data_at + (stripe + 1) * chunk] = parity
    for f, d in zip(files, disks):
        f.write_bytes(bytes(d))
    with archive.qnxprobe.raid_open([str(f) for f in files]) as rs:
        arr = rs.arrays[0]
        arr.image.seek(0)
        assert arr.image.read(len(payload)) == payload   # the layout written is md's
    return files


def test_a_passphrase_opens_a_luks_container_inside_an_array(tmp_path):
    luks = gzip.decompress((Path(__file__).parent / "fixtures" / "luks1-cbc.img.gz").read_bytes())
    files = _raid5_holding(tmp_path / "ev", luks)
    ref = archive.MemberSet(str(f.resolve()) for f in files)
    assert [v["open"] for v in archive.luks_volumes(ref)] == [False]
    assert archive.luks_volumes(ref)[0]["label"].startswith("md_qp_qp21_3090d8cb/")
    case_dir = tmp_path / "case"
    pw = tmp_path / "pass.txt"
    pw.write_text(LUKS_PASSPHRASE + "\n", encoding="utf-8")
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"sources": [{"name": "set", "members": [str(f) for f in files],
                                            "include_other": True}]}), encoding="utf-8")
    assert cli_main(["-c", str(case_dir), "--password-file", str(pw), "ingest", str(job),
                     "--no-process"]) == 0
    case = open_case(case_dir)
    walked = _walked(case, "set")
    got = {k.split("/", 2)[2]: v for k, v in walked.items()}
    assert got == _luks_known()
    assert json.loads(archive.source_record(case, "set")["luks"])
    case.close()


def test_a_wrong_passphrase_is_recorded_as_the_console_says_it(tmp_path, capsys):
    # the case keeps the reason the console gives, not the one for no passphrase
    luks = gzip.decompress((Path(__file__).parent / "fixtures" / "luks1-cbc.img.gz").read_bytes())
    files = _raid5_holding(tmp_path / "ev", luks)
    wrong = tmp_path / "wrong.txt"
    wrong.write_text("not it\n", encoding="utf-8")
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"sources": [{"name": "set", "members": [str(f) for f in files],
                                            "include_other": True}]}), encoding="utf-8")
    case_dir = tmp_path / "case"
    assert cli_main(["-c", str(case_dir), "--password-file", str(wrong), "ingest", str(job),
                     "--no-process"]) == 0
    assert "no passphrase given opens a keyslot" in capsys.readouterr().err
    case = open_case(case_dir)
    got = json.loads(archive.source_record(case, "set")["volumes_not_read"])
    case.close()
    assert len(got) == 1 and got[0].endswith(
        "(luks): LUKS1, locked; LUKS-encrypted and not read: no passphrase given opens a keyslot")


def test_a_passphrase_that_cannot_be_tried_is_recorded_with_the_readers_reason(
        tmp_path, monkeypatch, capsys):
    from gleapp.vendor import qnxprobe              # pylint: disable=import-outside-toplevel
    monkeypatch.setattr(qnxprobe, "_BDE_AES", None)  # as without pycryptodome
    luks = gzip.decompress((Path(__file__).parent / "fixtures" / "luks1-cbc.img.gz").read_bytes())
    files = _raid5_holding(tmp_path / "ev", luks)
    pw = tmp_path / "pass.txt"
    pw.write_text(LUKS_PASSPHRASE + "\n", encoding="utf-8")
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"sources": [{"name": "set", "members": [str(f) for f in files],
                                            "include_other": True}]}), encoding="utf-8")
    assert cli_main(["-c", str(tmp_path / "case"), "--password-file", str(pw), "ingest",
                     str(job), "--no-process"]) == 0
    # since qnxprobe 1.72 the reader knows this before any passphrase is tried
    reason = ("LUKS-encrypted and not read: reading it needs the optional pycryptodome "
              "package, which this Python does not have")
    err = capsys.readouterr().err
    assert reason in err
    assert "so the passphrase given was not tried on it" in err
    case = open_case(tmp_path / "case")
    got = archive.source_record(case, "set")["volumes_not_read"]
    case.close()
    assert reason in got


def test_no_passphrase_is_recorded_as_none_given(tmp_path):
    luks = gzip.decompress((Path(__file__).parent / "fixtures" / "luks1-cbc.img.gz").read_bytes())
    files = _raid5_holding(tmp_path / "ev", luks)
    case = _ingest(tmp_path / "case", image_set_source(files))
    got = json.loads(archive.source_record(case, files[0].name)["volumes_not_read"])
    case.close()
    assert len(got) == 1 and got[0].endswith("LUKS-encrypted and not read: give its passphrase")


def test_a_passphrase_opens_a_luks_container_in_a_member_of_a_set(tmp_path):
    folder = tmp_path / "ev"
    folder.mkdir()
    luks = folder / "luks.img"
    luks.write_bytes(gzip.decompress((Path(__file__).parent / "fixtures" /
                                      "luks1-cbc.img.gz").read_bytes()))
    other = folder / "usb.img"
    other.write_bytes(build_fat32([("A", "TXT", b"x", (2020, 1, 1, 0, 0, 0))]))
    # the container is in the second member, so it is found only by looking in
    # each member, not at the start of the set
    ref = archive.MemberSet([str(other), str(luks)])
    assert archive.unlock_luks(ref, "not it") is False
    assert archive.unlock_luks(ref, LUKS_PASSPHRASE) is True
    case = _ingest(tmp_path / "case", image_set_source([other, luks]))
    walked = _walked(case, "usb.img")
    got = {k.split("/", 2)[2]: v for k, v in walked.items() if k.startswith("luks.img/")}
    assert got == _luks_known()
    case.close()


# ---- what is not a disk image -------------------------------------------------------
def test_logical_evidence_in_a_set_is_refused_before_anything_is_read(tmp_path, capsys):
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    files = _members(tmp_path / "ev", "md-r1", (0, 1))
    ad1 = tmp_path / "ev" / "logical.ad1"
    ad1.write_bytes(b"ADSEGMENTEDFILE\x00" + bytes(4096))
    l01 = tmp_path / "ev" / "logical.L01"
    l01.write_bytes(b"LVF\t\r\n\xff\x00" + bytes(4096))
    for logical in (ad1, l01):
        with pytest.raises(ValueError, match="logical evidence"):
            image_set_source([files[0], logical, files[1]])
    case_dir = tmp_path / "case"
    assert cli_main(["-c", str(case_dir), "ingest", str(files[0]), str(ad1),
                     "--no-process"]) != 0
    assert "logical.ad1 is FTK Imager logical evidence (AD1)" in capsys.readouterr().err
    case = open_case(case_dir)
    assert archive.source_records(case) == {}
    assert case.db.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    case.close()
    cl = create_app(None).test_client()
    r = cl.post("/api/source/members", json={"paths": [str(files[0]), str(ad1)]})
    assert r.status_code == 400 and "logical evidence" in r.get_json()["message"]


# ---- two members of one name ------------------------------------------------------
def test_two_members_of_one_name_keep_their_files_apart(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "disk.img").write_bytes(build_fat32([("ONE", "TXT", b"first", (2020, 1, 1, 0, 0, 0))]))
    (b / "disk.img").write_bytes(build_fat32([("TWO", "TXT", b"second", (2020, 1, 1, 0, 0, 0))]))
    case = _ingest(tmp_path / "case", image_set_source([a / "disk.img", b / "disk.img"]))
    walked = _walked(case, "disk.img")
    assert walked == {"disk.img/lba0/ONE.TXT": hashlib.sha256(b"first").hexdigest(),
                      "disk.img (2)/lba0/TWO.TXT": hashlib.sha256(b"second").hexdigest()}
    case.close()



def test_the_web_asks_for_a_luks_passphrase_inside_an_array_and_reads_it(tmp_path):
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    luks = gzip.decompress((Path(__file__).parent / "fixtures" / "luks1-cbc.img.gz").read_bytes())
    files = _raid5_holding(tmp_path / "ev", luks)
    paths = [str(f.resolve()) for f in files]
    cl = create_app(None).test_client()
    assert cl.post("/api/case/create", json={"path": str(tmp_path / "case"), "name": "c",
                                             "examiner": "t"}).status_code == 200
    body = {"sources": [{"name": "set", "path": paths[0], "paths": paths}],
            "options": {"screen": False}}
    r = cl.post("/api/case/ingest", json=body)
    assert r.status_code == 409
    need = r.get_json()["needs_password"]
    assert [n["needs"] for n in need] == ["luks"]
    assert need[0]["volume"].startswith("md_qp_qp21_3090d8cb/")
    r = cl.post("/api/case/ingest", json={**body, "luks": {need[0]["path"]: {
        "secret": LUKS_PASSPHRASE}}})
    assert r.status_code == 200, r.get_json()
    for _ in range(600):                               # capped: at most 60 s
        if not cl.get("/api/job").get_json()["running"]:
            break
        import time                                    # pylint: disable=import-outside-toplevel
        time.sleep(0.1)
    src = next(s for s in cl.get("/api/context").get_json()["archive_sources"]
               if s["name"] == "set")
    assert json.loads(src["luks"]) and "luks_lba" in src["volumes"]


# ---- a set is read through the layout it was registered with ------------------------
def _hashes(case, name):
    return {r["orig_path"]: (r["sha256"], r["error"])
            for r in case.db.iter_files("source = ?", (name,))}


def _process(case):
    """A forced processing pass, as a new session runs it: nothing cached."""
    from gleapp.pipeline import process                # pylint: disable=import-outside-toplevel
    archive.close_zips()                               # closes every cached handle too
    process(case, force=True, workers=2, screen=False, similar=False)


def test_a_part_that_does_not_appear_this_session_moves_no_other_part(tmp_path, monkeypatch):
    """The reviewer's case without the cipher: member 0 is an LVM2 disk whose split and
    striped logical volumes are read through maps, each with a place of its own after
    member 0. In a session where they do not appear (a LUKS container around them
    left locked looks so), the parts after them used to move up, and a forced
    processing pass hashed member 2's bytes for member 1's rows. Now every part keeps
    its recorded place: the rows of the volumes that are not here fail with why and
    keep their hashes, and every other row reads as before."""
    folder = tmp_path / "ev"
    folder.mkdir()
    lvm = folder / "m0-lvm.img"
    lvm.write_bytes(gzip.decompress((Path(__file__).parent / "fixtures" /
                                     "lvm-fixture.img.gz").read_bytes()))
    one = folder / "m1.img"
    one.write_bytes(build_fat32([("ONE", "TXT", b"member one", (2020, 1, 1, 0, 0, 0))]))
    two = folder / "m2.img"
    two.write_bytes(build_fat32([("TWO", "TXT", b"member two", (2020, 1, 1, 0, 0, 0))]))
    case = _ingest(tmp_path / "case", image_set_source([lvm, one, two]))
    name = lvm.name
    _process(case)
    first = _hashes(case, name)
    assert first["m1.img/lba0/ONE.TXT"] == (hashlib.sha256(b"member one").hexdigest(), None)
    mapped = [k for k in first if "fixvg_data" in k or "fixvg_striped" in k]
    assert mapped and all(first[k][0] for k in mapped)
    layout = archive.source_record(case, name)["layout"]
    assert any("|lv:" in p["key"] for p in layout)

    # a session in which the mapped logical volumes do not appear
    monkeypatch.setattr(archive.qnxprobe, "lvm_image", lambda fh, size=None: fh)
    _process(case)
    later = _hashes(case, name)
    assert {k: v[0] for k, v in later.items()} == {k: v[0] for k, v in first.items()}
    for k in mapped:
        assert "not available" in later[k][1], later[k]
    assert later["m1.img/lba0/ONE.TXT"][1] is None
    assert later["m2.img/lba0/TWO.TXT"][1] is None
    assert archive.source_status(case)[0]["status"] == "ok"

    # and with them back, every row reads again
    monkeypatch.undo()
    _process(case)
    assert _hashes(case, name) == first
    case.close()


def test_an_array_assembled_for_the_first_time_takes_a_new_place(tmp_path, monkeypatch):
    """A later reader that assembles one more array, listed before the ones it
    assembled when the case was made, must not move them: the new one is placed
    after every recorded part, and every recorded row reads as before."""
    files = _members(tmp_path / "ev", "md-r5-ls", range(4))
    case = _ingest(tmp_path / "case", image_set_source(files))
    name = files[0].name
    _process(case)
    first = _hashes(case, name)
    recorded = archive.source_record(case, name)["layout"]
    end = max(p["start"] + p["size"] for p in recorded)
    extra_bytes = build_fat32([("NEW", "TXT", b"new array", (2020, 1, 1, 0, 0, 0))])
    real = archive.qnxprobe.raid_assemble

    def one_more(images):
        rs = real(images)
        img = archive.qnxprobe.RaidImage(
            lambda off, n: extra_bytes[off:off + n], len(extra_bytes), "md array extra")
        extra = type("Extra", (), {"label": "md array extra", "safe_name": "md_extra_00000000",
                                   "uuid": "00000000:00000000:00000000:00000001",
                                   "image": img, "refused": None, "same_as": None,
                                   "members": [], "lines": lambda self: [],
                                   "describe": lambda self: {}})()
        rs.arrays.insert(0, extra)
        return rs

    monkeypatch.setattr(archive.qnxprobe, "raid_assemble", one_more)
    _process(case)
    assert _hashes(case, name) == first
    with archive._open_image_file(archive.image_ref(archive.source_record(case, name))) as img:  # pylint: disable=protected-access
        placed = {p["key"]: p["start"] for p in img.layout_record()}
    assert placed["md:00000000:00000000:00000000:00000001"] >= end
    for p in recorded:
        assert placed[p["key"]] == p["start"]
    case.close()


# ---- a byte copy of a disk is read, and said ----------------------------------------
def test_a_byte_copy_of_a_disk_is_read_and_said_to_be_one(tmp_path, capsys):
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    files = _members(tmp_path / "ev", "md-r5-ls", range(4))
    copy = tmp_path / "ev" / "copy-of-3.img"
    shutil.copy(files[3], copy)                        # another inode, the same bytes
    src = image_set_source([*files, copy])
    assert len(src.members) == 5                       # not collapsed: it is another file
    assert src.copies == [f"copy-of-3.img may be a copy of md-r5-ls-3.img: the same size "
                          f"({files[3].stat().st_size:,} bytes) and the same first and last "
                          "1 MiB; both are read"]
    assert src.notes == []
    case_dir = tmp_path / "case"
    assert cli_main(["-c", str(case_dir), "ingest", *map(str, files), str(copy),
                     "--no-process"]) == 0
    assert "Warning: copy-of-3.img may be a copy of md-r5-ls-3.img" in capsys.readouterr().out
    case = open_case(case_dir)
    rec = archive.source_record(case, files[0].name)
    assert len(rec["members"]) == 5 and rec["set_notes"] == src.copies
    case.close()
    got = create_app(None).test_client().post(
        "/api/source/members", json={"paths": [str(files[3]), str(copy)]}).get_json()
    assert got["same"] == [] and len(got["copies"]) == 1 and len(got["paths"]) == 2


# ---- the dialog's keys ----------------------------------------------------------------
def test_enter_activates_the_focused_button_of_the_member_dialog():
    node = _node()
    if node is None:
        pytest.skip("node is not installed")
    js = (ROOT / "gleapp/web/static/app.js").read_text(encoding="utf-8")
    start = js.index("function membersKey(")
    body = js[start:js.index("\n}\n", start) + 3]
    out = subprocess.run([node, "-e", body + """
console.log(JSON.stringify([membersKey("Enter", "use"), membersKey("Enter", "only"),
  membersKey("Enter", "cancel"), membersKey("Enter", null), membersKey("Escape", "use"),
  membersKey("a", "use")]));"""], capture_output=True, text=True, check=True,
                         stdin=subprocess.DEVNULL, timeout=60).stdout
    assert json.loads(out) == ["use", "only", "cancel", None, "cancel", None]
    # the dialog opens with "Use them" focused, which is what makes it the default
    assert "wrap.querySelector('[data-r=\"use\"]').focus();" in js


# ---- what the record says when a set is not all well -------------------------------------
def test_a_relink_naming_an_image_that_is_no_member_is_refused(tmp_path):
    files = _members(tmp_path / "ev", "md-r5-ls", range(4))
    case = _ingest(tmp_path / "case", image_set_source(files))
    stranger = tmp_path / "other" / "stranger.img"
    stranger.parent.mkdir()
    stranger.write_bytes(build_fat32([("X", "TXT", b"x", (2020, 1, 1, 0, 0, 0))]))
    with pytest.raises(ValueError, match="stranger.img: not one of"):
        archive.relink_source(case, files[0].name, [*map(str, files), str(stranger)])
    case.close()


def test_an_array_that_cannot_be_read_is_recorded_as_not_read(tmp_path):
    files = _members(tmp_path / "ev", "md-r5-ls", (0, 1))          # 2 of 4: too few
    case = _ingest(tmp_path / "case", image_set_source(files))
    rec = archive.source_record(case, files[0].name)
    assert "md_qp_qp21_3090d8cb (md array qp:qp21): not read:" in rec["volumes_not_read"]
    case.close()


def test_a_member_whose_file_changed_makes_the_set_changed(tmp_path):
    files = _members(tmp_path / "ev", "md-r5-ls", range(4))
    case = _ingest(tmp_path / "case", image_set_source(files))
    os.utime(files[2], (1_000_000_000, 1_000_000_000))
    status = archive.source_status(case)[0]
    assert status["status"] == "changed"
    assert [m["status"] for m in status["members"]] == ["ok", "ok", "changed", "ok"]
    case.close()


# ---- second review ------------------------------------------------------------------
def _lvm_and_fat_set(tmp_path):
    folder = tmp_path / "ev"
    folder.mkdir()
    lvm = folder / "m0-lvm.img"
    lvm.write_bytes(gzip.decompress((Path(__file__).parent / "fixtures" /
                                     "lvm-fixture.img.gz").read_bytes()))
    one = folder / "m1.img"
    one.write_bytes(build_fat32([("ONE", "TXT", b"member one", (2020, 1, 1, 0, 0, 0))]))
    return lvm, one


def test_a_set_recorded_without_a_layout_is_refused_not_placed_again(tmp_path, monkeypatch):
    """A set an early build of this feature ingested has its members and no layout.
    Placing its parts again by what this session sees can put a row on another
    part's bytes, so its rows are refused, with what to do, and keep their hashes."""
    lvm, one = _lvm_and_fat_set(tmp_path)
    case = _ingest(tmp_path / "case", image_set_source([lvm, one]))
    name = lvm.name
    _process(case)
    first = _hashes(case, name)
    case.db.conn.execute("DELETE FROM meta WHERE key = ?", (f"archive:{name}:layout",))
    case.db.commit()
    # and a session that sees fewer parts than the ingest did
    monkeypatch.setattr(archive.qnxprobe, "lvm_image", lambda fh, size=None: fh)
    _process(case)
    later = _hashes(case, name)
    assert {k: v[0] for k, v in later.items()} == {k: v[0] for k, v in first.items()}
    assert all("ingest the member images again" in (v[1] or "") for v in later.values())
    with pytest.raises(ValueError, match="ingest the member images again"):
        ingest_sources(case, [image_set_source([lvm, one])])
    case.close()


def test_a_part_whose_size_changed_is_not_read_at_its_old_place(tmp_path):
    folder = tmp_path / "ev"
    folder.mkdir()
    one, two = folder / "m1.img", folder / "m2.img"
    one.write_bytes(build_fat32([("ONE", "TXT", b"member one", (2020, 1, 1, 0, 0, 0))]))
    two.write_bytes(build_fat32([("TWO", "TXT", b"member two", (2020, 1, 1, 0, 0, 0))]))
    case = _ingest(tmp_path / "case", image_set_source([one, two]))
    name = one.name
    before = _walked(case, name)
    archive.close_zips()                               # let go of what the reads opened
    with open(two, "ab") as fh:                        # the second member grows
        fh.write(bytes(1 << 20))
    rec = archive.source_record(case, name)
    rows = {r["orig_path"]: r for r in case.db.iter_files("source = ?", (name,))}
    with archive.local_copy(case.root, rec, rows["m1.img/lba0/ONE.TXT"]) as local:
        assert hashlib.sha256(Path(local).read_bytes()).hexdigest() == \
            before["m1.img/lba0/ONE.TXT"]
    with pytest.raises(archive.ArchiveUnavailable, match="bytes this session and was"):
        with archive.local_copy(case.root, rec, rows["m2.img/lba0/TWO.TXT"]):
            pass
    case.close()


def test_a_volume_group_part_is_keyed_by_its_group_id(tmp_path):
    files = _members(tmp_path / "ev", "lvm-raid-multi", range(5))
    case = _ingest(tmp_path / "case", image_set_source(files))
    keys = [p["key"] for p in archive.source_record(case, files[0].name)["layout"]]
    lvm_keys = [k for k in keys if k.startswith("lvm:")]
    assert len(lvm_keys) == 3
    ids = {k[len("lvm:"):].split("/", 1)[0] for k in lvm_keys}
    assert len(ids) == 1 and ids != {"qpraid2"}
    assert all(len(i) == 38 and i.count("-") == 6 for i in ids)   # an lvm2 id, not a name
    case.close()


def _other_group(blob: bytes) -> bytes:
    """The same disk under another volume group id and other physical volume ids,
    every checksum made good: what qnxprobe 1.71's own self-test builds
    (_lvm_same_name_checks), with its helpers."""
    import io                                          # pylint: disable=import-outside-toplevel
    import struct                                      # pylint: disable=import-outside-toplevel
    q = archive.qnxprobe
    pv0 = q._lvm_pv(io.BytesIO(blob), 0, None)        # pylint: disable=protected-access
    vg = pv0.vg[1]
    ids = [vg["id"]] + [r["id"] for r in vg["physical_volumes"].values()]

    def flip(x):
        return ("B" if x[0] != "B" else "C") + x[1:]

    def text(t):
        for i in ids:
            t = t.replace(i, flip(i))
        return t

    out = bytearray(q._lvm_text_change(blob, text))   # pylint: disable=protected-access
    pv = q._lvm_pv(io.BytesIO(bytes(out)), 0, None)  # pylint: disable=protected-access
    at = pv.label_sector * q.SECTOR
    off, = struct.unpack_from("<I", out, at + 20)
    out[at + off:at + off + 32] = flip(out[at + off:at + off + 32].decode()).encode()
    struct.pack_into("<I", out, at + 16, q.lvm_crc(bytes(out[at + 20:at + q.SECTOR])))
    return bytes(out)


def test_two_volume_groups_of_one_name_are_each_read_and_kept_apart(tmp_path):
    """Two different volume groups named qpraid2 (the second the first's disks with
    other ids), given as one set of ten images. qnxprobe 1.71 reads each by its id;
    GLEAPP keys each logical volume by its group's whole id, so no two share a
    recorded place, and every file of all six volumes reads to the kernel's bytes."""
    first = _members(tmp_path / "ev", "lvm-raid-multi", range(5))
    second = []
    for f in first:
        p = f.with_name("other-" + f.name)
        p.write_bytes(_other_group(f.read_bytes()))
        second.append(p)
    case = _ingest(tmp_path / "case", image_set_source(first + second))
    name = first[0].name
    keys = [p["key"] for p in archive.source_record(case, name)["layout"] if p["key"].startswith("lvm:")]
    assert len(keys) == 6 and len(set(keys)) == 6
    groups = {k[len("lvm:"):].split("/", 1)[0] for k in keys}
    assert len(groups) == 2
    assert all(len(g) == 38 and g.count("-") == 6 for g in groups)   # whole lvm2 ids
    _process(case)
    walked = _walked(case, name)
    tops = {k.split("/", 1)[0] for k in walked}
    # qnxprobe labels each "qpraid2@<first six of its id>", which the safe name
    # keeps as a part of its own
    assert len(tops) == 6 and len({t.split("_")[2] for t in tops}) == 2
    for top in tops:
        lv = top.rsplit("_", 1)[-1]
        assert _tree(walked, f"{top}/lba0/", f"lvm-raid-multi-{lv}") == \
            _expected(f"lvm-raid-multi-{lv}"), top
    case.close()


def test_tab_keeps_the_focus_inside_the_member_dialog():
    node = _node()
    if node is None:
        pytest.skip("node is not installed")
    js = (ROOT / "gleapp/web/static/app.js").read_text(encoding="utf-8")
    start = js.index("function membersTab(")
    body = js[start:js.index("\n}\n", start) + 3]
    out = subprocess.run([node, "-e", body + """
console.log(JSON.stringify([membersTab(0, 4, false), membersTab(3, 4, false),
  membersTab(0, 4, true), membersTab(2, 4, true), membersTab(-1, 4, false),
  membersTab(-1, 4, true), membersTab(-1, 0, false)]));"""], capture_output=True,
                         text=True, check=True, stdin=subprocess.DEVNULL, timeout=60).stdout
    assert json.loads(out) == [1, 0, 3, 1, 0, 3, -1]
    assert 'if (e.key === "Tab") {' in js and "if (!inside) e.preventDefault();" in js


# ---- DDF and IMSM members (qnxprobe 1.71) ---------------------------------------------
def _imsm_disk(folder: Path, name: str, level: int) -> Path:
    """An MBR disk with one FAT32 partition holding a picture and a text file, ending
    in the IMSM metadata qnxprobe's self-test builds (_imsm_test_tail): one volume of
    ``level`` over two disks."""
    import struct                                      # pylint: disable=import-outside-toplevel
    fat = build_fat32([("A", "TXT", b"on the partition", (2020, 1, 1, 0, 0, 0))])
    disk = bytearray(1 << 20) + bytearray(fat) + bytearray(1 << 20)
    struct.pack_into("<BBBBBBBBII", disk, 446, 0, 0, 0, 0, 0x0C, 0, 0, 0, 2048, len(fat) // 512)
    disk[510:512] = b"\x55\xaa"
    disk[-1024:] = archive.qnxprobe._imsm_test_tail(level)   # pylint: disable=protected-access
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_bytes(bytes(disk))
    return p


def _ingest_one(tmp_path, image: Path, case_name: str):
    case = open_case(tmp_path / case_name, create=True, examiner="t")
    sources, _ = parse_source_spec(image)
    sources[0].include_other = True
    ingest_sources(case, sources)
    return case, archive.source_record(case, image.name)


def test_a_partition_of_an_imsm_raid0_member_is_not_walked(tmp_path):
    """The partition table of a RAID0 volume sits on its first member, so on that disk
    alone the partition reads as a filesystem whose data is half another disk's. As
    qnxprobe 1.71 rules, it is a raid member, named with the level, and not walked;
    the same disk recording RAID1 is walked, and says it is a mirror member."""
    raid0 = _imsm_disk(tmp_path / "ev", "raid0-disk.img", 0)
    case, rec = _ingest_one(tmp_path, raid0, "c0")
    assert case.db.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    assert "RAID0" in rec["volumes_not_read"] and "it starts with fat32" in rec["volumes_not_read"]
    assert json.loads(rec["volumes"])[0]["kind"] == "raid member"
    case.close()
    raid1 = _imsm_disk(tmp_path / "ev", "raid1-disk.img", 1)
    case, rec = _ingest_one(tmp_path, raid1, "c1")
    assert [r["orig_path"] for r in case.db.iter_files()] == ["lba2048/A.TXT"]
    assert "RAID1 (a mirror)" in rec["volume_notes"][0]
    case.close()


def test_a_ddf_raid5_member_read_alone_is_not_walked(tmp_path):
    """md-ddf-2 begins with an ext4 superblock, a RAID5 stripe of a DDF array; qnxprobe
    1.67 walked it into a 3,096 PiB file. It is a raid member and not walked."""
    image = tmp_path / "ev" / "md-ddf-2.img"
    image.parent.mkdir()
    image.write_bytes(gzip.decompress((RAID / "md-ddf-2.img.gz").read_bytes()))
    case, rec = _ingest_one(tmp_path, image, "c")
    assert case.db.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    assert "DDF" in rec["volumes_not_read"] and "RAID5" in rec["volumes_not_read"]
    case.close()


# ---- a job file's options survive the search for member images ----------------------
def test_every_job_file_option_survives_the_members_found(tmp_path, monkeypatch):
    """The source rebuilt around the members found keeps what the job file set on the
    image it named, not only its name."""
    from gleapp import cli                             # pylint: disable=import-outside-toplevel
    from gleapp.case import SPEC_OPTIONS               # pylint: disable=import-outside-toplevel
    files = _members(tmp_path / "ev", "md-r5-ls", range(4))
    given = {"include_other": True, "follow_symlinks": True, "max_mb": 7, "stage": True,
             "carve": True, "unallocated_only": True, "documents": True}
    assert set(given) == set(SPEC_OPTIONS)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"sources": [{"name": "x", "path": str(files[0]), **given}]}),
                   encoding="utf-8")
    seen = []
    monkeypatch.setattr(cli, "ingest_sources", lambda _case, sources: seen.extend(sources) or 0)
    assert cli_main(["-c", str(tmp_path / "case"), "ingest", str(job), "--no-process"]) == 0
    assert len(seen) == 1
    src = seen[0]
    assert len(src.members) == 4 and src.name == "x"
    assert {opt: getattr(src, opt) for opt in SPEC_OPTIONS} == given


def test_include_other_from_a_job_file_registers_the_members_found(tmp_path):
    files = _members(tmp_path / "ev", "md-r5-ls", range(4))
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"sources": [{"name": "x", "path": str(files[0]),
                                            "include_other": True}]}), encoding="utf-8")
    assert cli_main(["-c", str(tmp_path / "case"), "ingest", str(job), "--no-process"]) == 0
    case = open_case(tmp_path / "case")
    walked = _walked(case, "x")
    assert _tree(walked, "md_qp_qp21_3090d8cb/lba0/", "md-r5-ls") == _expected("md-r5-ls")
    case.close()


# ---- one image whose own partitions hold an array ------------------------------------
def _gpt_disk(path: Path, parts: list[bytes]) -> Path:
    """A GPT disk holding each of ``parts`` as one partition, the first at LBA 2048,
    each followed by a 1 MiB gap, with both headers and their CRCs as the UEFI
    specification lays them out."""
    import struct                                      # pylint: disable=import-outside-toplevel
    import uuid                                        # pylint: disable=import-outside-toplevel
    import zlib                                        # pylint: disable=import-outside-toplevel
    ss, linux_raid = 512, uuid.UUID("A19D880F-05FC-4D3B-A006-743F0F84911E")
    spans, lba = [], 2048
    for blob in parts:
        assert len(blob) % ss == 0
        spans.append((lba, lba + len(blob) // ss - 1))
        lba += len(blob) // ss + 2048
    total = lba + 2048
    img = bytearray(total * ss)
    mbr = bytearray(ss)
    mbr[446:462] = struct.pack("<BBBBBBBBII", 0, 0, 2, 0, 0xEE, 0xFF, 0xFF, 0xFF, 1,
                               min(total - 1, 0xFFFFFFFF))
    mbr[510:512] = b"\x55\xaa"
    img[0:ss] = mbr
    ents = bytearray(128 * 128)
    for i, ((first, last), blob) in enumerate(zip(spans, parts)):
        ents[i * 128:(i + 1) * 128] = (linux_raid.bytes_le + uuid.UUID(int=i + 1).bytes_le
                                       + struct.pack("<QQQ", first, last, 0)
                                       + f"p{i}".encode("utf-16-le").ljust(72, b"\0"))
        img[first * ss:(last + 1) * ss] = blob
    disk = uuid.UUID(int=99).bytes_le

    def header(cur, alt, ent_lba):
        h = bytearray(struct.pack("<8sIIIIQQQQ16sQIII", b"EFI PART", 0x10000, 92, 0, 0, cur,
                                  alt, 34, total - 34, disk, ent_lba, 128, 128,
                                  zlib.crc32(ents)))
        h[16:20] = struct.pack("<I", zlib.crc32(h))
        return bytes(h).ljust(ss, b"\0")
    img[ss:2 * ss] = header(1, total - 1, 2)
    img[2 * ss:2 * ss + len(ents)] = ents
    img[(total - 33) * ss:(total - 33) * ss + len(ents)] = ents
    img[(total - 1) * ss:total * ss] = header(total - 1, 1, total - 33)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(img))
    return path


def _member_blobs(stem: str, roles) -> list[bytes]:
    return [gzip.decompress((RAID / f"{stem}-{i}.img.gz").read_bytes()) for i in roles]


_FAT = build_fat32([("A", "TXT", b"beside the array", (2020, 1, 1, 0, 0, 0))])


@pytest.mark.parametrize("stem, prefix", [("md-r5-ls", "md_qp_qp21_3090d8cb/lba0/"),
                                          ("md-r5ls-090", "md_8f61b69f/lba0/")])
def test_an_array_in_one_images_own_partitions_is_read_to_the_known_data(
        tmp_path, stem, prefix):
    """One disk whose partitions hold all four members of a RAID5 array, metadata 1.2
    and 0.90, with a FAT32 partition beside them: the array is assembled from the
    partitions, as qnxprobe does, and every file it holds is the kernel's bytes; the
    FAT32 partition is walked as the member's own volume; the member partitions are not
    walked or reported unread. A later session reads every row from the place recorded."""
    image = _gpt_disk(tmp_path / "ev" / "one.img", _member_blobs(stem, range(4)) + [_FAT])
    assert archive.holds_own_array(image)
    case = open_case(tmp_path / "case", create=True, examiner="t")
    sources, _ = parse_source_spec(image)
    assert sources[0].members == []
    sources[0].include_other = True
    ingest_sources(case, sources)
    walked = _walked(case, "one.img")
    assert _tree(walked, prefix, stem) == _expected(stem)
    fat = {k: v for k, v in walked.items() if not k.startswith(prefix)}
    assert fat == {"one.img/p4/A.TXT": hashlib.sha256(b"beside the array").hexdigest()}
    rec = archive.source_record(case, "one.img")
    assert [Path(m["path"]).name for m in rec["members"]] == ["one.img"]
    assert [p["key"].split(":")[0] for p in rec["layout"]] == ["member", "md"]
    assert rec["volumes_not_read"] == ""
    assert isinstance(archive.image_ref(rec), archive.MemberSet)
    status = archive.source_status(case)[0]
    assert status["status"] == "ok" and [m["status"] for m in status["members"]] == ["ok"]
    layout = case.db.get_meta("archive:one.img:layout")
    case.close()
    archive.close_zips()
    case = open_case(tmp_path / "case")               # a new session
    assert _walked(case, "one.img") == walked
    assert case.db.get_meta("archive:one.img:layout") == layout
    case.close()


def test_an_image_holding_part_of_an_array_is_not_read_as_one(tmp_path):
    """Two of four RAID5 members in one image's partitions cannot be read: it stays an
    ordinary image source and its record says why the partitions were not walked."""
    image = _gpt_disk(tmp_path / "ev" / "half.img", _member_blobs("md-r5-ls", (0, 1)))
    assert not archive.holds_own_array(image)
    case, rec = _ingest_one(tmp_path, image, "c")
    assert "members" not in rec
    refused = json.loads(rec["volumes_not_read"])
    assert len(refused) == 2 and all("md RAID member" in r and "two or more of its members" in r
                                     for r in refused), refused
    case.close()


def test_a_lone_whole_disk_mirror_member_is_not_read_as_a_set_of_itself(tmp_path):
    """qnxprobe assembles a RAID1 array degraded from one member; a disk that is one
    member and nothing else is still read as an ordinary image, as before."""
    image = _members(tmp_path / "ev", "md-r1", (0,))[0]
    assert not archive.holds_own_array(image)
    case, rec = _ingest_one(tmp_path, image, "c")
    assert "members" not in rec and "md RAID member" in rec["volumes_not_read"]
    case.close()


def test_a_source_registered_as_one_image_keeps_being_read_as_one(tmp_path, monkeypatch):
    """A case whose earlier ingest registered rows from such an image as an ordinary
    image (what a GLEAPP before this one did) is not switched to a set by a later pass,
    so its rows stay where they were registered."""
    image = _gpt_disk(tmp_path / "ev" / "one.img", _member_blobs("md-r5-ls", range(4)) + [_FAT])
    monkeypatch.setattr(archive, "holds_own_array", lambda _p: False)
    case, rec = _ingest_one(tmp_path, image, "c")
    assert "members" not in rec
    before = _walked(case, "one.img")
    assert list(before) == ["p4/A.TXT"]
    monkeypatch.undo()
    sources, _ = parse_source_spec(image)
    sources[0].include_other = True
    ingest_sources(case, sources)
    rec = archive.source_record(case, "one.img")
    assert "members" not in rec and "layout" not in rec
    assert _walked(case, "one.img") == before
    case.close()


def test_a_passphrase_opens_luks_inside_an_array_in_one_images_partitions(tmp_path, capsys):
    luks = gzip.decompress((Path(__file__).parent / "fixtures" / "luks1-cbc.img.gz").read_bytes())
    files = _raid5_holding(tmp_path / "ev", luks)
    image = _gpt_disk(tmp_path / "ev" / "one.img", [f.read_bytes() for f in files])
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"sources": [{"name": "one", "path": str(image),
                                            "include_other": True}]}), encoding="utf-8")
    wrong = tmp_path / "wrong.txt"
    wrong.write_text("not it\n", encoding="utf-8")
    assert cli_main(["-c", str(tmp_path / "locked"), "--password-file", str(wrong), "ingest",
                     str(job), "--no-process"]) == 0
    err = capsys.readouterr().err
    assert "md_qp_qp21_3090d8cb/whole image of one.img stays locked" in err
    assert "no passphrase given opens a keyslot" in err
    locked = open_case(tmp_path / "locked")
    assert "no passphrase given opens a keyslot" in archive.source_record(
        locked, "one")["volumes_not_read"]
    locked.close()
    pw = tmp_path / "pass.txt"
    pw.write_text(LUKS_PASSPHRASE + "\n", encoding="utf-8")
    assert cli_main(["-c", str(tmp_path / "case"), "--password-file", str(pw), "ingest",
                     str(job), "--no-process"]) == 0
    assert "stays locked" not in capsys.readouterr().err
    case = open_case(tmp_path / "case")
    got = {k.split("/", 2)[2]: v for k, v in _walked(case, "one").items()}
    assert got == _luks_known()
    case.close()


def test_the_web_asks_for_luks_inside_an_array_in_one_images_partitions(tmp_path):
    from gleapp.web.app import create_app              # pylint: disable=import-outside-toplevel
    luks = gzip.decompress((Path(__file__).parent / "fixtures" / "luks1-cbc.img.gz").read_bytes())
    files = _raid5_holding(tmp_path / "ev", luks)
    image = _gpt_disk(tmp_path / "ev" / "one.img", [f.read_bytes() for f in files])
    cl = create_app(None).test_client()
    assert cl.post("/api/case/create", json={"path": str(tmp_path / "case"), "name": "c",
                                             "examiner": "t"}).status_code == 200
    body = {"sources": [{"name": "one", "path": str(image)}], "options": {"screen": False}}
    r = cl.post("/api/case/ingest", json=body)
    assert r.status_code == 409
    [need] = r.get_json()["needs_password"]
    assert need["needs"] == "luks" and need["volume"].startswith("md_qp_qp21_3090d8cb/")
    r = cl.post("/api/case/ingest", json={**body, "luks": {need["path"]: {"secret": "not it"}}})
    [need] = r.get_json()["needs_password"]
    assert need["wrong"] is True and "no passphrase given opens a keyslot" in need["note"]
    r = cl.post("/api/case/ingest", json={**body, "luks": {need["path"]: {
        "secret": LUKS_PASSPHRASE}}})
    assert r.status_code == 200, r.get_json()
    for _ in range(600):                               # capped: at most 60 s
        if not cl.get("/api/job").get_json()["running"]:
            break
        import time                                    # pylint: disable=import-outside-toplevel
        time.sleep(0.1)
    src = next(s for s in cl.get("/api/context").get_json()["archive_sources"]
               if s["name"] == "one")
    assert json.loads(src["luks"]) and "luks_lba" in src["volumes"]
