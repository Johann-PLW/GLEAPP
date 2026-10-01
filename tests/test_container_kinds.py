"""Archives, documents and ExoPlayer cache pieces are told apart without running
Python inside SQLite.

A case's one connection is shared by every thread. While it carried a Python SQL
function for "is this an ExoPlayer cache piece", a query that called it per row
held SQLite's connection mutex and waited for the interpreter lock, and a thread
binding a parameter held the interpreter lock and waited for that mutex. Opening
the gallery of a case with many container rows froze the server. The answer is
now stored on the row when it is written (``files.exo_cache``).
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import containerrows                                # pylint: disable=import-error
from gleapp.case import open_case
from gleapp.db import CaseDB
from gleapp.vendor import exoprobe

ROOT = Path(__file__).resolve().parents[1]

# Enough container rows for one count query to run long enough to overlap a
# write: with the function on the connection, 6,000 rows and 40 rounds froze on
# 10 runs of 10 (macOS arm64, Python 3.14). Without it the same run takes about
# 3 s, so the timeout is only ever reached by a deadlock.
DEADLOCK_ROWS = 6000
DEADLOCK_ROUNDS = 40
DEADLOCK_TIMEOUT = 180


def _case(tmp_path, n: int):
    case = open_case(tmp_path / "case", create=True, examiner="t")
    ids = containerrows.build(case.db, n)
    return case, ids


def test_container_counts_are_read_while_another_thread_writes(tmp_path):
    case, _ids = _case(tmp_path, DEADLOCK_ROWS)
    case.close()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        done = subprocess.run(
            [sys.executable, str(ROOT / "tests" / "containerrows.py"),
             str(tmp_path / "case"), str(DEADLOCK_ROUNDS)],
            cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=DEADLOCK_TIMEOUT, check=False)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"the readers and the writer were still waiting on each other after "
                    f"{DEADLOCK_TIMEOUT} s; output so far: {exc.stdout!r}")
    assert done.returncode == 0 and "FINISHED" in done.stdout, (
        f"exit {done.returncode}\n{done.stdout}\n{done.stderr}")


def test_no_python_function_is_registered_on_the_case_connection(tmp_path):
    db = CaseDB(tmp_path / "case.gleapp")
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such function"):
            db.conn.execute("SELECT is_exo_cache_name('1.0.2.v3.exo')")
        # and it knows no function a plain connection does not
        plain = sqlite3.connect(":memory:")
        try:
            listing = "SELECT name, narg FROM pragma_function_list"
            try:
                own = {tuple(r) for r in plain.execute(listing)}
            except sqlite3.OperationalError:        # an SQLite without the pragma
                own = None
            if own is not None:
                assert {tuple(r) for r in db.conn.execute(listing)} == own
        finally:
            plain.close()
    finally:
        db.close()
    # and nothing in the package registers one on any connection
    for path in sorted((ROOT / "gleapp").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for call in ("create_function", "create_collation", "create_aggregate",
                     "create_window_function"):
            assert call + "(" not in text, f"{path.relative_to(ROOT)} calls {call}"


def test_the_filters_and_the_sidebar_tell_the_three_kinds_apart(tmp_path):
    from gleapp.web.app import create_app           # pylint: disable=import-outside-toplevel
    case, ids = _case(tmp_path, 400)
    case.close()
    assert all(ids[k] for k in ids), "the fixture holds every kind"
    client = create_app(None).test_client()
    client.post("/api/case/open", json={"path": str(tmp_path / "case")})
    assert containerrows.read_all(client, ids) == containerrows.expected(ids)


def test_the_fixtures_own_labels_agree_with_the_cache_rule():
    # the labels in containerrows.TEMPLATES are written by hand; this says the rule
    # the app applies (exoprobe's) gives the same answer for each name
    for rel, orig, _ext, what in containerrows.TEMPLATES:
        name = (orig or rel).format(i=12)
        assert exoprobe.is_cache_name(name) == (what == containerrows.CACHE), name


def _flags(db) -> dict:
    return {r["id"]: r["exo_cache"] for r in db.conn.execute(
        "SELECT id, exo_cache FROM files")}


def test_the_flag_is_stored_when_a_row_is_written_and_follows_its_name(tmp_path):
    case, ids = _case(tmp_path, 160)
    db = case.db
    try:
        flags = _flags(db)
        assert None not in flags.values()
        cache = {fid for fid, flag in flags.items() if flag}
        other = db.conn.execute(
            "SELECT id FROM files WHERE kind = 'other'").fetchone()["id"]
        # the one non-container row named like a piece carries the flag too: it is
        # about the name, and the filters add kind = 'archive' themselves
        assert cache == ids[containerrows.CACHE] | {other}

        # a name that changes takes the flag with it, through either writer
        fid = min(ids[containerrows.ARCHIVE])
        db.update_file(fid, orig_path="dev/9.0.1.v3.exo")
        assert db.get_file(fid)["exo_cache"] == 1
        db.update_file(fid, orig_path="dev/plain.bin")
        assert db.get_file(fid)["exo_cache"] == 0
        path = db.get_file(fid)["path"]
        db.upsert_file(path, orig_path="", rel_path="a/cached_content_index.exi")
        assert db.get_file(fid)["exo_cache"] == 1
        db.upsert_file(path, rel_path="a/plain.bin")
        assert db.get_file(fid)["exo_cache"] == 0
        # and a write that leaves the name alone leaves the flag alone
        db.update_file(min(ids[containerrows.CACHE]), notes="seen")
        assert db.get_file(min(ids[containerrows.CACHE]))["exo_cache"] == 1
    finally:
        case.close()


def test_a_case_made_before_the_column_is_filled_once_on_open(tmp_path, monkeypatch):
    from gleapp.web.app import create_app           # pylint: disable=import-outside-toplevel
    case, ids = _case(tmp_path, 400)
    db = case.db
    want = _flags(db)
    # lay the case out the way v2026.5.3 left it: no column, schema 18
    db.conn.execute("DROP INDEX idx_files_exo_unset")
    db.conn.execute("ALTER TABLE files DROP COLUMN exo_cache")
    db.set_meta("schema_version", "18")
    db.commit()
    case.close()

    asked = []
    rule = exoprobe.is_cache_name
    monkeypatch.setattr(exoprobe, "is_cache_name",
                        lambda name: asked.append(name) or rule(name))
    path = tmp_path / "case" / "case.gleapp"
    reopened = CaseDB(path)
    try:
        assert _flags(reopened) == want
        # the v18 swap of category codes 0 and 5 does not run on a schema 18 case
        assert reopened.conn.execute(
            "SELECT COUNT(*) FROM files WHERE category != 5").fetchone()[0] == 0
    finally:
        reopened.close()
    assert len(asked) == len(want), "every row was decided, once"

    del asked[:]
    again = CaseDB(path)                             # nothing left to fill
    try:
        assert not asked, "a second open decides nothing again"
        assert again.fill_exo_cache() == 0
        assert _flags(again) == want
    finally:
        again.close()

    client = create_app(None).test_client()
    client.post("/api/case/open", json={"path": str(tmp_path / "case")})
    assert containerrows.read_all(client, ids) == containerrows.expected(ids)


def test_rows_an_older_build_added_to_a_newer_case_are_filled_on_open(tmp_path):
    # v2026.5.3 opens a case that has the column and registers rows without it
    case, ids = _case(tmp_path, 160)
    db = case.db
    want = _flags(db)
    with db.lock:
        db.conn.execute(
            "INSERT INTO files (path, rel_path, kind, ext, category) "
            "VALUES ('/case/staged/late', 'app/cache/3.0.4.v3.exo', 'archive', '.exo', 5)")
        db.conn.execute(
            "INSERT INTO files (path, rel_path, kind, ext, category) "
            "VALUES ('/case/staged/late2', 'dl/late.zip', 'archive', '.zip', 5)")
        db.conn.commit()
    late = {r["path"]: r["id"] for r in db.conn.execute(
        "SELECT id, path FROM files WHERE path LIKE '/case/staged/late%'")}
    assert db.get_file(late["/case/staged/late"])["exo_cache"] is None
    case.close()

    reopened = CaseDB(tmp_path / "case" / "case.gleapp")
    try:
        got = _flags(reopened)
        assert got.pop(late["/case/staged/late"]) == 1
        assert got.pop(late["/case/staged/late2"]) == 0
        assert got == want
        assert len(ids[containerrows.CACHE]) + 2 == sum(1 for v in _flags(reopened).values() if v)
    finally:
        reopened.close()
