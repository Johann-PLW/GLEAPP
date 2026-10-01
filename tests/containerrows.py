"""A case of container rows, and a run that reads their counts while another
thread writes to the same case.

The gallery tells three kinds of ``kind = 'archive'`` row apart: an archive, a
document, and a piece of an app's ExoPlayer cache. ``build`` registers rows of
all three, with names the cache rule accepts and names that only look like one,
and ``expected`` says which is which. The expectations are written out beside
each name, so they do not move with the code that decides.

Run as a script, this file opens the case through the web app and reads the
sidebar's container counts and the Type filters from one thread while another
writes to the case's connection. It is run in a child process under a timeout by
``test_container_kinds.py``, because the defect it guards against is a deadlock:
with a Python SQL function on the connection every thread shares, one thread
waited for the interpreter lock inside SQLite while another held that lock and
waited for SQLite.
"""

from __future__ import annotations

import sys
import threading

ARCHIVE, DOCUMENT, CACHE = "archive", "document", "cache"

# (rel_path, orig_path, ext, what the row is). ``{i}`` is the row's number.
# The name that decides is orig_path when it is not empty, else rel_path.
TEMPLATES = (
    ("dl/a{i}.zip", None, ".zip", ARCHIVE),
    ("docs/r{i}.pdf", None, ".pdf", DOCUMENT),
    ("app/cache/exo/{i}.0.1700000000000.v3.exo", None, ".exo", CACHE),
    ("app/cache/key{i}.0.17.v2.exo", None, ".exo", CACHE),
    ("app{i}/cache/cached_content_index.exi", None, ".exi", CACHE),
    ("app{i}/cache/exoplayer_internal.db-wal", None, ".db-wal", CACHE),
    ("app{i}/cache/0a1b2c3d.uid", None, ".uid", CACHE),
    # near misses: two numbers where a piece has three, an upper-case .uid name,
    # the index file's name with something after it
    ("app/cache/{i}.0.v3.exo", None, ".exo", ARCHIVE),
    ("app{i}/cache/0A1B2C3D.uid", None, ".uid", ARCHIVE),
    ("app{i}/cache/cached_content_index.exi.old", None, ".old", ARCHIVE),
    # orig_path decides when it holds a name, whatever rel_path says
    ("x/{i}.0.5.v3.exo", "dev/plain{i}.bin", ".bin", ARCHIVE),
    ("y/plain{i}.bin", "dev/{i}.0.5.v3.exo", ".exo", CACHE),
    ("z/{i}.0.6.v3.exo", "", ".exo", CACHE),
    # a path written with the other separator
    ("w/plain{i}.bin", "C:\\Users\\u\\cache\\{i}.0.7.v3.exo", ".exo", CACHE),
    ("dl/b{i}.7z", None, ".7z", ARCHIVE),
    ("docs/page{i}.html", None, ".html", DOCUMENT),
)

# every this-many containers gets one picture extracted from it
CHILD_EVERY = 7
# and this many documents saved with no extension, known by what came out of them
BARE_DOCUMENTS = 3


def build(db, n: int) -> dict:
    """Register ``n`` container rows and what was extracted from some of them.
    Returns the ids of each kind, and of the pictures pulled from each kind."""
    ids = {ARCHIVE: set(), DOCUMENT: set(), CACHE: set(),
           "in_archive": set(), "in_document": set()}
    for i in range(n):
        rel, orig, ext, what = TEMPLATES[i % len(TEMPLATES)]
        fields = {"rel_path": rel.format(i=i), "source": "ev", "kind": "archive",
                  "ext": ext, "size": 10 + i}
        if orig is not None:
            fields["orig_path"] = orig.format(i=i)
        fid = db.upsert_file(f"/case/staged/c{i:06d}", **fields)
        ids[what].add(fid)
        if i % CHILD_EVERY == 0:
            kid = db.upsert_file(f"/case/staged/c{i:06d}.d/pic.jpg",
                                 rel_path=f"{rel.format(i=i)}/pic.jpg", source="ev",
                                 kind="image", ext=".jpg", size=5, container_id=fid,
                                 orig_name="pic.jpg")
            if what == ARCHIVE:
                ids["in_archive"].add(kid)
            elif what == DOCUMENT:
                ids["in_document"].add(kid)
    for i in range(BARE_DOCUMENTS):
        fid = db.upsert_file(f"/case/staged/bare{i}", rel_path=f"app/files/bare{i}",
                             source="ev", kind="archive", ext="", size=99)
        ids[DOCUMENT].add(fid)
        ids["in_document"].add(db.upsert_file(
            f"/case/staged/bare{i}.d/p0001_obj00001.jpg",
            rel_path=f"app/files/bare{i}/p0001_obj00001.jpg", source="ev", kind="image",
            ext=".jpg", size=5, container_id=fid, orig_name="p0001_obj00001.jpg"))
    # rows that are not containers, one of them named like a cache piece: the
    # filters are about containers and must leave these out
    db.upsert_file("/case/staged/photo.jpg", rel_path="dcim/photo.jpg", source="ev",
                   kind="image", ext=".jpg", size=7)
    db.upsert_file("/case/staged/other.exo", rel_path="app/cache/1.0.2.v3.exo",
                   source="ev", kind="other", ext=".exo", size=7)
    db.commit()
    return ids


def read_all(client, ids: dict) -> dict:
    """What the app reports for the same questions ``build`` answered: the ids each
    Type filter and each "pulled from" filter returns, and the sidebar's counts."""
    limit = sum(len(v) for v in ids.values()) + 10
    got = {}
    for name, query in ((ARCHIVE, "kind=archive:archive"),
                        (DOCUMENT, "kind=archive:document"),
                        (CACHE, "kind=archive:cache"),
                        ("in_archive", "in_archive=1"),
                        ("in_document", "in_document=1")):
        page = client.get(f"/api/files?limit={limit}&{query}").get_json()
        got[name] = {f["id"] for f in page["files"]}
        got[name + "_total"] = page["total"]
    got["containers"] = client.get("/api/context").get_json().get("containers")
    return got


def expected(ids: dict) -> dict:
    want = {}
    for name in (ARCHIVE, DOCUMENT, CACHE, "in_archive", "in_document"):
        want[name] = set(ids[name])
        want[name + "_total"] = len(ids[name])
    want["containers"] = {"archives": len(ids[ARCHIVE]),
                          "archive_items": len(ids["in_archive"]),
                          "documents": len(ids[DOCUMENT]),
                          "documents_opened": len(ids["in_document"]),
                          "document_items": len(ids["in_document"])}
    return want


def main(argv: list[str]) -> int:
    """``containerrows.py <case folder> <rounds>``: read the container counts
    ``rounds`` times from one thread while another writes."""
    from gleapp.web.app import create_app          # pylint: disable=import-outside-toplevel
    case_dir, rounds = argv[0], int(argv[1])
    app = create_app(None)
    opened = app.test_client().post("/api/case/open", json={"path": case_dir})
    assert opened.status_code == 200, opened.get_data(as_text=True)
    case = app.config["STATE"]["case"]
    some = [r["id"] for r in case.db.conn.execute("SELECT id FROM files LIMIT 50")]
    stop = threading.Event()
    problems: list[str] = []
    writes = [0]

    def write() -> None:
        # the kind of write and read a running job makes, each with bound
        # parameters. The read's text is this thread's own: two threads running
        # the same statement text on one connection is a different hazard, and
        # this run is about the one a Python SQL function caused.
        k = 0
        while not stop.is_set():
            fid = some[k % len(some)]
            case.db.update_file(fid, notes=f"note {k}")
            case.db.conn.execute(
                "SELECT notes FROM files WHERE id = ? AND size >= ?", (fid, 0)).fetchone()
            k += 1
        writes[0] = k

    def read() -> None:
        client = app.test_client()
        for _ in range(rounds):
            ctx = client.get("/api/context").get_json()
            if not ctx.get("containers"):
                problems.append("the context came back without its container counts")
                return
            page = client.get("/api/files?limit=1&kind=archive:cache").get_json()
            if not page["total"]:
                problems.append(f"cache pieces: {page['total']}")
                return
            client.get("/api/files?limit=1&kind=archive:archive").get_json()
            client.get("/api/files?limit=1&in_archive=1").get_json()

    writer = threading.Thread(target=write, name="writer")
    reader = threading.Thread(target=read, name="reader")
    writer.start()
    reader.start()
    reader.join()
    stop.set()
    writer.join()
    app.config["STATE"]["shutdown"]()
    if problems:
        print("PROBLEM " + "; ".join(problems))
        return 1
    print(f"FINISHED rounds={rounds} writes={writes[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
