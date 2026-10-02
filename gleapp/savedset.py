"""The examiner's saved set of categories and flags, carried from case to case.

Stored per user in ``config_dir()/categories.json``, outside any case. Saving
copies the open case's own categories (never the locked Project VIC presets,
which every case gets anyway) and/or its flags into the file; applying copies
them into a case. It is a copy, not a link: editing a case never changes the
saved set, and saving a new set never reaches back into an older case.

Categories keep their code where the target case has it free, so a category is
the same number in every case (and in every Project VIC export). Both lists
merge by name, ignoring case: a name the case already has is left alone.
"""

from __future__ import annotations

import json
from typing import Any

from . import appconfig
from .db import CATEGORY_PALETTE, VIC_PRESETS


def _path():
    return appconfig.config_dir() / "categories.json"


def load() -> dict:
    """``{categories: [...], flags: [...], use_categories, use_flags}``."""
    try:
        raw = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    cats = [c for c in raw.get("categories") or []
            if isinstance(c, dict) and str(c.get("name") or "").strip()]
    flags = [f for f in raw.get("flags") or []
             if isinstance(f, dict) and str(f.get("name") or "").strip()]
    return {"categories": cats, "flags": flags,
            "use_categories": raw.get("use_categories") is not False,
            "use_flags": raw.get("use_flags") is True}


def _write(data: dict) -> None:
    _path().write_text(json.dumps(data, indent=2), encoding="utf-8")


def set_defaults(*, use_categories: bool, use_flags: bool) -> None:
    """Remember how the New case checkboxes were left."""
    data = load()
    data["use_categories"] = bool(use_categories)
    data["use_flags"] = bool(use_flags)
    _write(data)


def case_categories(db) -> list[dict]:
    """The case's own categories that can be saved: named, shown, unlocked."""
    return [{"code": int(r["code"]), "name": r["name"].strip(),
             "color": r["color"], "notable": bool(r["notable"])}
            for r in db.list_categories(include_inactive=False)
            if not r["locked"] and (r["name"] or "").strip()]


def case_flags(db) -> list[dict]:
    return [{"name": r["name"].strip(), "color": r["color"]}
            for r in db.list_flags() if (r["name"] or "").strip()]


def save_from_case(db, *, categories: bool, flags: bool) -> dict:
    """Replace the saved categories and/or flags with the case's. A part not
    asked for keeps what was saved before."""
    data = load()
    if categories:
        data["categories"] = case_categories(db)
    if flags:
        data["flags"] = case_flags(db)
    _write(data)
    return {"categories": len(data["categories"]), "flags": len(data["flags"])}


def apply_to_case(db, *, categories: bool, flags: bool) -> dict[str, list[dict]]:
    """Add the saved categories and/or flags to a case.

    Returns what happened to each, for the examiner and the audit log:
    categories ``{name, code, saved_code, status}`` with status ``added``
    (same code), ``renumbered`` (that code was in use), ``shown`` (the case
    had it hidden) or ``exists``; flags ``{name, status}`` (``added`` or
    ``exists``).
    """
    data = load()
    out: dict[str, list[dict]] = {"categories": [], "flags": []}
    if categories:
        for c in data["categories"]:
            out["categories"].append(_apply_category(db, c))
    if flags:
        have = {(r["name"] or "").strip().lower() for r in db.list_flags()}
        for f in data["flags"]:
            name = str(f["name"]).strip()
            if name.lower() in have:
                out["flags"].append({"name": name, "status": "exists"})
                continue
            code = db.add_flag(name)
            if _is_color(f.get("color")):
                db.update_flag(code, color=f["color"])
            have.add(name.lower())
            out["flags"].append({"name": name, "status": "added"})
    return out


def _is_color(v: Any) -> bool:
    return (isinstance(v, str) and len(v) == 7 and v.startswith("#")
            and all(ch in "0123456789abcdefABCDEF" for ch in v[1:]))


def _apply_category(db, c: dict) -> dict:
    name = str(c["name"]).strip()
    try:
        want = int(c.get("code"))
    except (TypeError, ValueError):
        want = None
    rows = db.list_categories()
    for r in rows:
        if (r["name"] or "").strip().lower() == name.lower():
            if not r["active"] and not r["locked"]:
                db.update_category(r["code"], active=1)
                return {"name": name, "code": r["code"], "saved_code": want,
                        "status": "shown"}
            return {"name": name, "code": r["code"], "saved_code": want,
                    "status": "exists"}
    used = {r["code"] for r in rows}
    if want is not None and want >= len(VIC_PRESETS) and want not in used:
        code, status = want, "added"
    else:
        code = max(max(used, default=0) + 1, len(VIC_PRESETS))
        status = "added" if want is None else "renumbered"
    color = c.get("color") if _is_color(c.get("color")) else \
        CATEGORY_PALETTE[(code - 1) % len(CATEGORY_PALETTE)]
    db.insert_category(code, name, color=color, notable=c.get("notable", True))
    return {"name": name, "code": code, "saved_code": want, "status": status}
