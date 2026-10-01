"""The context's counts are worked out once and kept until a row is written.

The gallery asks for ``/api/context`` on load, after imports and at intervals
during a job. Every count in it is a pass over the ``files`` table, and the
folder check read every row of each folder source, so on a large case each
request cost hundreds of milliseconds and kept the shared connection busy.
``CaseDB.derived`` keeps what comes from the rows until the connection's running
total of changed rows moves. These tests hold both halves: a repeat reads no
rows, and no write is ever followed by an old answer.
"""

from __future__ import annotations

import io
import shutil
import zipfile

import pytest
from PIL import Image

from gleapp import archive, relink
from gleapp.case import Source, open_case, parse_source_spec
from gleapp.db import CaseDB
from gleapp.pipeline import ingest_sources


@pytest.fixture(autouse=True)
def _isolate_appconfig(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("GLEAPP_CONFIG_DIR", str(tmp_path_factory.mktemp("gleapp-cfg")))


def _evidence(tmp_path, n: int = 4):
    ev = tmp_path / "ev"
    (ev / "sub").mkdir(parents=True)
    for i in range(n):
        where = ev / ("sub" if i % 2 else "") / f"p{i}.png"
        Image.new("RGB", (24, 24), (i * 40, 90, 200 - i * 30)).save(where)
    return ev


def _open(tmp_path):
    """A case with one folder source of four pictures, opened through the app."""
    from gleapp.web.app import create_app           # pylint: disable=import-outside-toplevel
    ev = _evidence(tmp_path)
    case = open_case(tmp_path / "case", create=True, examiner="t")
    ingest_sources(case, [Source(name="ev", path=str(ev))])
    case.close()
    app = create_app(None)
    client = app.test_client()
    assert client.post("/api/case/open", json={"path": str(tmp_path / "case")}).status_code == 200
    return app.config["STATE"]["case"], client, ev


def _statements(case, call) -> list[str]:
    """Every statement the case's connection runs while ``call()`` does."""
    seen: list[str] = []
    case.db.conn.set_trace_callback(seen.append)
    try:
        call()
    finally:
        case.db.conn.set_trace_callback(None)
    return [" ".join(s.split()) for s in seen]


def _reads_files(statements: list[str]) -> list[str]:
    return [s for s in statements if " files" in s or "file_flags" in s]


def test_derived_is_worked_out_again_after_any_write(tmp_path):
    db = CaseDB(tmp_path / "case.gleapp")
    try:
        calls = []

        def compute():
            calls.append(1)
            return len(calls)

        assert db.derived("n", compute) == 1
        assert db.derived("n", compute) == 1            # nothing written: kept
        assert db.derived("other", compute) == 2        # another name is its own answer
        assert db.derived("n", compute) == 1

        fid = db.upsert_file("/ev/a.jpg", source="ev", kind="image")     # an insert
        assert db.derived("n", compute) == 3
        db.update_file(fid, notes="x")                                   # an update, not committed
        assert db.derived("n", compute) == 4
        db.conn.execute("INSERT INTO meta(key, value) VALUES('k', 'v')")  # SQL not through a helper
        assert db.derived("n", compute) == 5
        db.conn.execute("DELETE FROM meta WHERE key = 'k'")
        assert db.derived("n", compute) == 6
        db.commit()                                     # a commit changes no row
        assert db.derived("n", compute) == 6
        db.conn.execute("UPDATE files SET notes = 'y' WHERE id = ?", (fid,))
        db.conn.rollback()                              # a write taken back still counts
        assert db.derived("n", compute) == 7

        # a row written while an answer is being worked out (another thread's write):
        # the answer may not include it, so it is not kept past that write
        def slow():
            seen = db.conn.execute("SELECT notes FROM files WHERE id = ?", (fid,)).fetchone()[0]
            db.update_file(fid, notes="written meanwhile")
            return seen

        assert db.derived("slow", slow) != "written meanwhile"
        assert db.derived("slow", lambda: "worked out again") == "worked out again"
    finally:
        db.close()
    # a reopened case starts with nothing kept
    again = CaseDB(tmp_path / "case.gleapp")
    try:
        assert again.derived("n", lambda: "fresh") == "fresh"
    finally:
        again.close()


def test_a_repeated_context_reads_no_rows_and_gives_the_same_answer(tmp_path):
    case, client, _ev = _open(tmp_path)
    first = {}
    cold = _statements(case, lambda: first.update(client.get("/api/context").get_json()))
    assert _reads_files(cold), "the first context counts the rows"
    again = {}
    warm = _statements(case, lambda: again.update(client.get("/api/context").get_json()))
    assert _reads_files(warm) == []
    assert again == first
    assert first["stats"]["total"] == 4 and first["sources"] == ["ev"]


def test_no_write_is_followed_by_an_old_context(tmp_path):
    case, client, _ev = _open(tmp_path)
    ids = [r["id"] for r in case.db.conn.execute("SELECT id FROM files ORDER BY id")]

    def ctx():
        return client.get("/api/context").get_json()

    start = ctx()
    assert start["stats"]["by_category"] == {"5": 4}
    assert start["stats"]["reviewed"] == 0 and start["stats"]["flagged"] == 0

    assert client.post("/api/categorize", json={"ids": ids[:2], "category": 1}).status_code == 200
    assert ctx()["stats"]["by_category"] == {"1": 2, "5": 2}

    assert client.post("/api/review", json={"ids": ids[:3], "reviewed": True}).status_code == 200
    assert ctx()["stats"]["reviewed"] == 3

    flag = client.post("/api/flags", json={"name": "follow up"}).get_json()["code"]
    assert [f["name"] for f in ctx()["flags"]][-1] == "follow up"
    assert client.post("/api/flag", json={"ids": ids[:1], "add": [flag]}).status_code == 200
    assert ctx()["stats"]["flagged"] == 1

    # a row a job adds, before the job commits it
    case.db.upsert_file("/elsewhere/new.jpg", rel_path="new.jpg", source="second",
                        kind="image", ext=".jpg", size=3)
    after = ctx()
    assert after["stats"]["total"] == 5 and after["sources"] == ["ev", "second"]

    # SQL that goes straight to the connection
    case.db.conn.execute("UPDATE files SET error = 'unreadable' WHERE id = ?", (ids[0],))
    assert ctx()["errors"] == 1
    case.db.conn.execute("DELETE FROM files WHERE source = 'second'")
    gone = ctx()
    assert gone["stats"]["total"] == 4 and gone["sources"] == ["ev"]


def test_folder_status_reads_the_rows_once_and_looks_on_disk_every_time(tmp_path):
    case, _client, ev = _open(tmp_path)
    want = [{"name": "ev", "root": str(ev), "files": 4, "status": "ok"}]
    cold = _statements(case, lambda: relink.folder_status(case))
    assert _reads_files(cold)
    assert relink.folder_status(case) == want
    assert _reads_files(_statements(case, lambda: relink.folder_status(case))) == []

    # the folder moves and nothing is written to the case: the next answer says so
    moved = tmp_path / "moved"
    shutil.move(str(ev), str(moved))
    assert relink.folder_status(case)[0]["status"] == "missing"

    # relinking writes the new paths, and the answer follows them
    relink.relink_folder(case, "ev", moved)
    assert relink.folder_status(case) == [
        {"name": "ev", "root": str(moved), "files": 4, "status": "ok"}]

    # one more file of the source, registered by a job: counted at once
    Image.new("RGB", (8, 8)).save(moved / "late.png")
    case.db.upsert_file(str(moved / "late.png"), rel_path="late.png", source="ev",
                        kind="image", ext=".png", size=1)
    assert relink.folder_status(case)[0]["files"] == 5


def _png_bytes(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, "PNG")
    return buf.getvalue()


def test_archive_source_counts_are_read_once_and_the_archive_is_looked_for_every_time(tmp_path):
    from gleapp.web.app import create_app           # pylint: disable=import-outside-toplevel
    zpath = tmp_path / "extraction.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for i in range(3):
            zf.writestr(f"Dump/data/media/0/DCIM/p{i}.png", _png_bytes((i * 70, 20, 90)))
    case = open_case(tmp_path / "case", create=True, examiner="t")
    sources, _meta = parse_source_spec(str(zpath))
    ingest_sources(case, sources)
    name = sources[0].name
    case.close()
    app = create_app(None)
    client = app.test_client()
    assert client.post("/api/case/open", json={"path": str(tmp_path / "case")}).status_code == 200
    case = app.config["STATE"]["case"]

    def entry():
        got = client.get("/api/context").get_json()["archive_sources"]
        assert [s["name"] for s in got] == [name]
        return got[0]

    assert (entry()["files"], entry()["status"]) == (3, "ok")
    assert _reads_files(_statements(case, entry)) == []
    assert _reads_files(_statements(case, lambda: archive.source_status(case))) == []

    # one more row of the source, as a carve adds them: counted at once
    case.db.upsert_file(str(tmp_path / "case" / "staged" / "late.png"), rel_path="carved/late.png",
                        source=name, kind="image", ext=".png", size=1, origin="carve")
    after = entry()
    assert (after["files"], after["carved"]) == (4, 1)

    # the archive moves and nothing is written to the case: the next answer says so
    shutil.move(str(zpath), str(tmp_path / "elsewhere.zip"))
    assert entry()["status"] == "missing"
