"""C4All report index import.

C4All writes one index per media type next to the files it exported, ``C4P Index.xml``
for pictures and ``C4M Index.xml`` for movies. Both are UTF-16 XML of the same shape::

    <ReportIndex version="2.0">
        <Image>                                   (<Movie> in a C4M index)
            <path><![CDATA[DVD01(Name)\\01\\]]></path>   folder, relative to the index
            <picture>00000001.JPG</picture>       (<movie> in a C4M index)
            <category>0</category>
            <id>1</id>
            <fullpath><![CDATA[...\\DCIM\\DSCN0001.JPG]]></fullpath>   original location
            <created>1569647772</created>         Unix seconds, as are accessed/written
            <deleted>18446744073709551615</deleted>   all ones: not deleted
            <hash>...</hash>                      MD5
            <sha1Hash>...</sha1Hash>
            <fileSize>3434508</fileSize>
            ...
        </Image>
    </ReportIndex>

`import_c4all` registers every entry as a file, resolving ``path`` + file name
against the index's folder, and carries the hashes, original path, times and any
category into the case the way `projectvic.import_vic` does for a VIC file. The
pipeline then hashes and thumbnails as usual.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterator

from .db import is_uncategorized

# The element that holds one file, and the child naming the exported copy.
_ENTRY_NAME = {"Image": "picture", "Movie": "movie"}
_KIND = {"Image": "image", "Movie": "video"}
# ``deleted`` is a 64-bit FILETIME-style field whose all-ones value means "never".
_NOT_DELETED = 0xFFFFFFFFFFFFFFFF


@dataclass
class C4Record:
    entry_id: int | None
    tag: str
    rel_path: str
    abs_path: str
    exists: bool
    md5: str | None
    sha1: str | None
    category: int | None
    orig_path: str | None
    size: int | None
    created: float | None
    written: float | None
    accessed: float | None
    deleted: float | None


def is_c4all_file(path: str | Path) -> bool:
    """Cheap check on the first few KB: an ``.xml`` whose root is ``ReportIndex``."""
    p = Path(path)
    if p.suffix.lower() != ".xml":
        return False
    try:
        with open(p, "rb") as fh:
            raw = fh.read(4096)
    except OSError:
        return False
    for enc in ("utf-16", "utf-8-sig"):
        try:
            head = raw.decode(enc, errors="ignore")
        except (UnicodeError, LookupError):
            continue
        if "<ReportIndex" in head:
            return True
    return False


def _int(text: str | None) -> int | None:
    try:
        return int((text or "").strip())
    except ValueError:
        return None


def _ts(text: str | None) -> float | None:
    v = _int(text)
    if v is None or v <= 0 or v >= _NOT_DELETED:
        return None
    return float(v)


def _hex(text: str | None) -> str | None:
    return (text or "").strip().lower() or None


def iter_records(path: str | Path) -> Iterator[C4Record]:
    """Stream the entries of a C4All index. ``iterparse`` reads the XML declaration,
    so the UTF-16 encoding needs no special handling, and a large index is never
    held in memory whole."""
    path = Path(path).resolve()
    base = path.parent
    for _, el in ET.iterparse(str(path), events=("end",)):
        name_tag = _ENTRY_NAME.get(el.tag)
        if name_tag is None:
            continue
        get = el.findtext
        folder = (get("path") or "").strip()
        fname = (get(name_tag) or "").strip()
        if not fname:
            el.clear()
            continue
        # The index is written on Windows; split its backslashes on every platform.
        rel = PureWindowsPath(folder, fname).as_posix()
        abs_path = (base / rel).resolve()
        cat = _int(get("category"))
        yield C4Record(
            entry_id=_int(get("id")),
            tag=el.tag,
            rel_path=rel,
            abs_path=str(abs_path),
            exists=abs_path.is_file(),
            md5=_hex(get("hash")),
            sha1=_hex(get("sha1Hash")),
            # C4All writes 0 for an entry nobody has categorized.
            category=cat if cat else None,
            orig_path=(get("fullpath") or "").strip() or None,
            size=_int(get("fileSize")),
            created=_ts(get("created")),
            written=_ts(get("written")),
            accessed=_ts(get("accessed")),
            deleted=_ts(get("deleted")),
        )
        el.clear()


def import_c4all(case, index_path: str | Path) -> tuple[int, int]:
    """Register every entry of a C4All index into ``case``.

    Returns (registered, missing), counted in entries. Entries naming the same
    exported file collapse onto one row, the other original paths kept as
    ``alt_paths``, as `projectvic.import_vic` does.
    """
    from .projectvic import _ensure_category

    index_path = Path(index_path).resolve()
    case.db.set_meta("c4all_source_xml", str(index_path))

    groups: dict[str, list[C4Record]] = {}
    for r in iter_records(index_path):
        groups.setdefault(r.abs_path, []).append(r)

    seen_cats: set[int] = set()
    registered = missing = 0
    for abs_path, recs in groups.items():
        r = recs[0]
        orig = r.orig_path
        fields = dict(
            rel_path=r.rel_path,
            source=index_path.name,
            kind=_KIND[r.tag],
            ext=Path(abs_path).suffix.lower(),
            size=r.size,
            md5=r.md5,
            sha1=r.sha1,
            ctime=r.created,
            mtime=r.written,
            atime=r.accessed,
            orig_name=PureWindowsPath(orig).name if orig else None,
            orig_path=orig,
        )
        others: list[str] = []
        for o in recs[1:]:
            if o.orig_path and o.orig_path != orig and o.orig_path not in others:
                others.append(o.orig_path)
        if others:
            fields["alt_paths"] = json.dumps(others)
        cat = next((x.category for x in recs
                    if x.category is not None and not is_uncategorized(x.category)),
                   None)
        if cat is not None:
            fields["category"] = cat
            seen_cats.add(cat)
        if r.exists:
            registered += len(recs)
        else:
            fields["error"] = "file not found on disk"
            missing += len(recs)
        case.db.upsert_file(abs_path, **fields)

    for code in sorted(seen_cats):
        if case.db.get_category(code) is None:
            _ensure_category(case.db, code)
    case.db.commit()
    case.db.audit_log(case.examiner, "import_c4all",
                      f"{index_path.name}: {registered} files, {missing} missing")
    return registered, missing
