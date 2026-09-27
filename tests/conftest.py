"""Session-wide isolation of the per-user config directory.

``gleapp.appconfig`` reads ``$GLEAPP_CONFIG_DIR`` and falls back to the
platform's per-user application directory when it is unset.  Opening a case
through the web app calls ``appconfig.push_recent`` (``gleapp/web/app.py``), so
a test that builds the Flask app against a tmp case writes into the examiner's
own recent-cases list, and the twelve-entry cap in ``push_recent`` then evicts
the cases they actually work on.  Measured on macOS: after one suite run all
twelve entries in the real config were pytest tmp paths.

Thirteen test modules already pointed the variable at a tmp directory in their
own autouse fixture and nineteen did not, so isolation was per module rather
than global.  This fixture sets the variable for the whole session before any
test runs, on every platform, so the suite cannot reach a real config whichever
modules are collected.  The per-module fixtures keep working unchanged: they
narrow it further per test, and their teardown now restores this directory
instead of unsetting the variable.

The hash store and the stash live under the same directory and each cache a
module-level connection, so both are closed around the session.
"""

from __future__ import annotations

import threading

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_user_config(tmp_path_factory):
    """Point the config, app-data and stash locations at a tmp directory."""
    cfg = tmp_path_factory.mktemp("user-config")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GLEAPP_CONFIG_DIR", str(cfg))
        # stash_path() reads this before it consults the config dir, so an
        # examiner who keeps their stash on a shared drive would otherwise have
        # the suite write into it.
        mp.delenv("GLEAPP_STASH_PATH", raising=False)
        from gleapp import hashstore, stash
        hashstore.close()  # drop anything opened while collecting modules
        stash.close()
        try:
            yield cfg
        finally:
            hashstore.close()
            stash.close()


@pytest.fixture(autouse=True)
def _stop_app_threads(monkeypatch):
    """Stop the background threads of every app a test builds, when the test ends.

    ``create_app`` starts a snapshot loop, and an ingest through the app starts the
    Find similar indexer. Neither ever stopped in a test, so a full run carried them
    into every later test: 82 snapshot loops and six indexers still running by the
    ``test_pipeline`` ingest tests (measured 2026-09-27). With numpy on OpenBLAS on
    macOS, a ``fork`` for a worker while a leftover indexer was inside a
    multithreaded OpenBLAS call deadlocked in OpenBLAS's ``pthread_atfork`` handler,
    with the GIL held, and the run never finished. Tests look ``create_app`` up when
    they run, so wrapping the module attribute reaches them; ``gleapp.desktop`` binds
    its own name at import and is wrapped as well. A test that still leaves one of
    these threads running fails at teardown.
    """
    from gleapp import desktop
    from gleapp.web import app as appmod
    made = []
    real = appmod.create_app

    def create_app(*args, **kwargs):
        app = real(*args, **kwargs)
        made.append(app)
        return app

    monkeypatch.setattr(appmod, "create_app", create_app)
    monkeypatch.setattr(desktop, "create_app", create_app)
    before = set(threading.enumerate())
    yield
    for app in made:
        app.config["STATE"]["shutdown"]()
    left = sorted(t.name for t in set(threading.enumerate()) - before
                  if t.name in ("auto-snapshot", "find-similar-indexer"))
    assert not left, f"background threads outlived their test: {left}"
