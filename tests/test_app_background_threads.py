"""The web app's background threads stop when asked, and the snapshot loop outlives a
case closed under it.

``create_app`` starts a snapshot loop and holds a Find similar indexer. Both used to run
until the process ended: nothing could stop the loop, and the indexer slept through a
stop request for up to ``BUSY_POLL`` seconds. A test suite that builds hundreds of apps
carried every one of them into the tests after it (see ``tests/conftest.py``).
"""

from __future__ import annotations

import threading
import time

from gleapp.case import open_case
from gleapp.web import app as appmod
from gleapp.web import indexer as indexermod


def _app_with_case(tmp_path):
    root = tmp_path / "case"
    open_case(root, create=True, examiner="t").close()
    return appmod.create_app(str(root))


def test_shutdown_stops_the_snapshot_loop_and_the_indexer_at_once(tmp_path, monkeypatch):
    """With a job running the indexer only waits. It used to wait by sleeping BUSY_POLL
    seconds, so a stop request sat unanswered for the whole sleep; set to a minute here,
    that would hold shutdown for a minute."""
    monkeypatch.setattr(indexermod, "BUSY_POLL", 60.0)
    before = set(threading.enumerate())
    app = _app_with_case(tmp_path)
    state = app.config["STATE"]
    state["job"]["running"] = True          # the indexer pauses while a job runs
    state["indexer"].start()
    for _ in range(100):
        if state["indexer"].status["paused"]:
            break
        time.sleep(0.01)
    assert state["indexer"].status["paused"]
    started = {t.name for t in set(threading.enumerate()) - before}
    assert started == {"auto-snapshot", "find-similar-indexer"}, started

    t0 = time.monotonic()
    state["shutdown"]()
    took = time.monotonic() - t0

    alive = {t.name for t in set(threading.enumerate()) - before if t.is_alive()}
    assert not alive, alive
    assert took < 5, f"shutdown took {took:.1f} s"
    state["job"]["running"] = False
    state["case"].close()


def test_the_snapshot_loop_outlives_a_case_closed_under_it(tmp_path, monkeypatch):
    """Something can close the case while the app still holds it (a caller of
    ``close_current``, a test that handed the app its own case). A dirty closed case
    raised ``ProgrammingError`` out of the loop, and nothing restarted it: no more timed
    snapshots for the life of the app."""
    monkeypatch.setattr(appmod, "AUTO_BACKUP_POLL", 0.02)
    raised = []
    monkeypatch.setattr(threading, "excepthook", raised.append)
    before = set(threading.enumerate())
    app = _app_with_case(tmp_path)
    state = app.config["STATE"]
    loop = [t for t in set(threading.enumerate()) - before if t.name == "auto-snapshot"]
    assert len(loop) == 1
    case = state["case"]
    case.db.dirty = True
    case.close()                            # still state["case"]
    time.sleep(0.5)                         # about 25 rounds of the loop
    assert not raised, [a.exc_value for a in raised]
    assert any(t.is_alive() for t in loop)
    state["shutdown"]()
    assert not any(t.is_alive() for t in loop)
