"""Find similar's indexes, built in the background while the examiner works.

Processing used to build the copy and content indexes as its last stages, which added
8 to 20 minutes to an ingest (measured on four real cases). Here they are built after
processing instead, one thread per open case, at lower parallelism so the gallery stays
responsive: likely examiner material first, the system's and applications' own artwork
last. Find similar searches whatever is indexed so far. The indexer pauses while any
job runs (an ingest, screening, a re-scan) and stops before the case closes; both
builders write in chunks and pick up where they left off, so nothing is lost.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import traceback

IDLE_POLL = 10.0      # seconds between checks for new files when everything is indexed
BUSY_POLL = 2.0       # seconds between checks while a job runs


class BackgroundIndexer:
    def __init__(self, state: dict) -> None:
        self.state = state
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._searches = 0                 # Find similar searches running right now
        self._search_lock = threading.Lock()
        self.status = {"running": False, "stage": "", "done": 0, "total": 0,
                       "paused": False, "error": None}

    def start(self) -> None:
        """Start indexing the open case, unless it is already being indexed."""
        with self._lock:
            case = self.state.get("case")
            if case is None or (self._thread is not None and self._thread.is_alive()):
                return
            self._stop.clear()
            self.status.update(running=True, stage="", done=0, total=0, paused=False, error=None)
            self._thread = threading.Thread(target=self._run, args=(case,), daemon=True,
                                             name="find-similar-indexer")
            self._thread.start()

    def stop(self, wait: bool = True) -> None:
        """Ask the indexer to stop; with ``wait``, return once it has (it finishes the
        chunk in hand, a few seconds at most)."""
        self._stop.set()
        t = self._thread
        if wait and t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=300)

    def searching(self):
        """Context manager around a Find similar search: the indexer stands aside (it
        skips the files in hand and stops at once) until no search is running, since
        both compete for the same cores. Measured: a warm search took 0.36 s alone and
        0.73 s beside a content build."""
        indexer = self

        class _Search:
            def __enter__(self):
                with indexer._search_lock:
                    indexer._searches += 1

            def __exit__(self, *exc):
                with indexer._search_lock:
                    indexer._searches -= 1
        return _Search()

    # ---------------------------------------------------------------------------------
    def _should_yield(self, case) -> bool:
        return (self._stop.is_set() or self.state.get("case") is not case
                or self.state["job"]["running"] or self._searches > 0)

    def _run(self, case) -> None:
        from .. import content, simindex
        workers = max(2, (os.cpu_count() or 4) // 2)
        progress = lambda d, t: self.status.update(done=d, total=t)
        # how many files each pass covered when it last ran to its end: a pass runs
        # again whenever that number changes, however the files arrived (a source
        # ingested during the other pass used to stay unindexed until the case reopened)
        finished = {"copies": None, "content": None}
        try:
            while not self._stop.is_set() and self.state.get("case") is case:
                if self.state["job"]["running"]:
                    self.status.update(paused=True)
                    self._stop.wait(BUSY_POLL)   # returns at once when stop() is called
                    continue
                if self._searches:
                    self._stop.wait(0.2)     # a search is running: wait for it to finish
                    continue
                self.status.update(paused=False)
                st = simindex.status(case)
                if st["indexed"] < st["indexable"] and finished["copies"] != st["indexable"]:
                    self.status.update(stage="copies", done=0, total=st["indexable"] - st["indexed"])
                    simindex.build_index(case, workers=workers, progress=progress,
                                         stop=lambda: self._should_yield(case))
                    if not self._should_yield(case):
                        finished["copies"] = st["indexable"]
                    continue
                ct = content.status(case)
                if (ct["model"] and ct["indexed"] < ct["indexable"]
                        and finished["content"] != ct["indexable"]):
                    self.status.update(stage="content", done=0, total=ct["indexable"] - ct["indexed"])
                    content.build_index(case, workers=workers, progress=progress,
                                        stop=lambda: self._should_yield(case))
                    if not self._should_yield(case):
                        finished["content"] = ct["indexable"]
                    continue
                # up to date (an unreadable thumbnail is not retried until new files
                # arrive): idle, then look again for files a later job adds
                self.status.update(stage="idle", done=0, total=0)
                for _ in range(int(IDLE_POLL / BUSY_POLL)):
                    if self._stop.is_set() or self.state.get("case") is not case:
                        return
                    self._stop.wait(BUSY_POLL)
        except sqlite3.ProgrammingError:
            pass                  # the case was closed under it (not through the app): stop
        # pylint: disable-next=broad-exception-caught
        except Exception as exc:  # noqa: BLE001 - reported in the Find similar section
            traceback.print_exc()
            self.status.update(error=f"{type(exc).__name__}: {exc}"[:300])
        finally:
            self.status.update(running=False, paused=False, stage="")
