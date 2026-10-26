"""Content Credentials (C2PA) and the IPTC digital source type, read when asked for.

What a picture or video declares about its own provenance: a C2PA manifest store
(who signed it, with what tool, the actions and digital source types it states,
its ingredients) and an IPTC DigitalSourceType in XMP. The reading is the vendored
c2paprobe's (``abrignoni/c2paprobe``, ``gleapp/vendor/c2paprobe.py``), shared with
iLEAPP and ALEAPP; fix it there and re-vendor.

Nothing is verified. No signature is checked and no certificate chain is built, so
a signer is what the file states. An AI reading is a declaration, not a finding,
and a file with no credentials says nothing about whether it is authentic: many
platforms and tools remove them.

The read is opt-in: the launcher's "Read Content Credentials" option runs it after
processing, and ``read_pass`` runs it on a case later. Each file read gets one row in
``c2pa_meta``; a file with none has not been read, which the gallery shows as such
and never as "none found".
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from . import archive
from .vendor import c2paprobe

READER = f"c2paprobe {c2paprobe.__version__}"

# The kinds a pass reads. c2paprobe decides the container from the bytes; a file
# GLEAPP called 'other' is not media and is not read.
KINDS = ("image", "video")

CREDENTIALS = ("c2pa", "xmp", "none", "unsettled")
CREDENTIAL_LABELS = {
    "c2pa": "C2PA manifest",
    "xmp": "XMP source type only",
    "none": "None found",
    "unsettled": "Not settled",
    None: "Not read",
}
CREDENTIAL_HELP = {
    "c2pa": "The file holds a C2PA manifest store.",
    "xmp": "No manifest store, but the file's XMP states an IPTC digital source type.",
    "none": "The whole container was read and holds neither.",
    "unsettled": ("The read stopped early, or the file's container is not one the reader "
                  "walks, so absence is not established."),
    None: "Content Credentials have not been read for this file.",
}
AI_VALUES = ("yes", "no", "unknown")
AI_LABELS = {"yes": "Yes", "no": "No", "unknown": "Unknown", None: ""}

CAVEAT = (
    "Content Credentials record what the file declares. Nothing is verified: no "
    "signature is checked, so a signer is as stated. An AI reading is a declaration, "
    "not a finding. A file without credentials says nothing about whether it is "
    "authentic; many platforms and tools remove them.")

AI_RULE = (
    "Yes for trainedAlgorithmicMedia, compositeWithTrainedAlgorithmicMedia and "
    "compositeSynthetic, the IPTC codes whose definition names generative AI; No for "
    "the capture, scan, human edit, screen capture and software image codes; Unknown "
    "for every other code. Read over every manifest in the store and every XMP "
    "statement, so an AI ingredient makes the file Yes.")


def _ai_of(values) -> str | None:
    """'yes' when any stated reading is True, else 'no' when any is False, else
    'unknown' when something was stated, None when nothing was."""
    values = list(values)
    if any(v is True for v in values):
        return "yes"
    if any(v is False for v in values):
        return "no"
    return "unknown" if values else None


def summarize(result: dict) -> dict:
    """The columns ``c2pa_meta`` keeps beside the whole result, for filtering."""
    stated = []
    for m in result.get("manifests") or []:
        stated.extend(st["ai"] for st in m.get("source_types") or [])
    stated.extend(x["source_type"]["ai"] for x in result.get("xmp") or [] if x.get("source_type"))
    if result.get("stores"):
        credentials = "c2pa"
    elif any(x.get("source_type") for x in result.get("xmp") or []):
        credentials = "xmp"
    elif result.get("complete") or result.get("file_size") == 0:
        # an empty file holds nothing to find, whatever its name says it is
        credentials = "none"
    else:
        credentials = "unsettled"
    signer = ""
    manifests = result.get("manifests") or []
    if manifests:
        signer = ((manifests[-1].get("signature") or {}).get("signed_by")) or ""
    return {"credentials": credentials, "ai": _ai_of(stated), "signer": signer or None}


def read_path(path) -> dict:
    """c2paprobe's result for the file at ``path``, with the file's size beside it
    (``file_size``). Raises OSError when it cannot be opened; anything wrong inside
    the file is in the result's ``errors``."""
    result = c2paprobe.probe(path)
    result["file_size"] = os.path.getsize(path)
    return result


def save(db, file_id: int, result: dict) -> None:
    """Store one file's result. The caller commits."""
    s = summarize(result)
    with db.lock:
        db.conn.execute(
            "INSERT OR REPLACE INTO c2pa_meta(file_id, reader, read_at, credentials, ai, "
            "signer, record) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (file_id, READER, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             s["credentials"], s["ai"], s["signer"], json.dumps(result)))


def for_file(db, file_id: int) -> dict | None:
    """What is kept for a file, or None when it has not been read."""
    got = db.conn.execute(
        "SELECT reader, read_at, credentials, ai, signer, record FROM c2pa_meta "
        "WHERE file_id = ?", (file_id,)).fetchone()
    if not got:
        return None
    return {"reader": got["reader"], "read_at": got["read_at"],
            "credentials": got["credentials"], "ai": got["ai"], "signer": got["signer"],
            "record": json.loads(got["record"])}


def all_summaries(db) -> dict[int, dict]:
    """file id -> its row, for the exports, read in one statement."""
    out = {}
    for r in db.conn.execute(
            "SELECT file_id, reader, credentials, ai, signer, record FROM c2pa_meta"):
        out[r["file_id"]] = {"reader": r["reader"], "credentials": r["credentials"],
                             "ai": r["ai"], "signer": r["signer"],
                             "record": json.loads(r["record"])}
    return out


def _codes(record: dict) -> list[str]:
    """Every stated digital source type, manifests first then XMP, each once."""
    out = []
    for m in record.get("manifests") or []:
        out.extend(st["code"] for st in m.get("source_types") or [])
    out.extend(x["source_type"]["code"] for x in record.get("xmp") or [] if x.get("source_type"))
    return list(dict.fromkeys(out))


def source_type_text(st: dict) -> str:
    """IPTC's label and the code, or the URI as stated when IPTC has no label for it."""
    return f"{st['label']} ({st['code']})" if st.get("label") else st.get("uri", "")


def describe(entry: dict | None) -> list[tuple[str, list[tuple[str, str]]]]:
    """A file's reading as titled groups of (field, value), for the report card.
    ``entry`` is what ``all_summaries`` or ``for_file`` keeps, or None."""
    if entry is None:
        return [("Content Credentials", [("Status", CREDENTIAL_LABELS[None])])]
    rec = entry["record"]
    head = [("Status", CREDENTIAL_LABELS[entry["credentials"]]),
            ("AI declared", AI_LABELS[entry["ai"]] or "Nothing stated")]
    if rec.get("stores"):
        head.append(("Found in", rec["stores"][0]["where"]))
    if rec.get("remote_manifest"):
        head.append(("Remote manifest (not fetched)", rec["remote_manifest"]))
    if rec.get("file_size") == 0:
        head.append(("Note", "the file is empty"))
    if entry["credentials"] == "unsettled" and rec.get("errors"):
        head.append(("Why not settled", "; ".join(rec["errors"][:3])))
    elif entry["credentials"] == "unsettled":
        head.append(("Why not settled", "the reader does not walk this file's container "
                                        "(it reads JPEG, PNG, HEIF/MP4/MOV, WebP, GIF, "
                                        "TIFF and JPEG XL)"))
    out = [("Content Credentials", head)]
    manifests = rec.get("manifests") or []
    for i, m in enumerate(manifests, 1):
        sig = m.get("signature") or {}
        role = "active" if i == len(manifests) else "ingredient"
        rows = [("Manifest", m.get("label", "")),
                ("Claim generator", m.get("claim_generator", "")),
                ("Signed by (stated)", sig.get("signed_by", "")),
                ("Certificate issuer (stated)", sig.get("issuer", "")),
                ("Time-stamp (stated, UTC)", sig.get("signing_time", "")),
                ("Actions", ", ".join(m.get("actions") or [])),
                ("Software agents", ", ".join(m.get("software_agents") or [])),
                ("Digital source type", ", ".join(source_type_text(st) for st in m.get("source_types") or [])),
                ("Title", m.get("title", "")), ("Format", m.get("format", "")),
                ("Ingredients", ", ".join(
                    f"{g['title'] or '(unnamed)'}" + (f" ({g['relationship']})" if g.get("relationship") else "")
                    for g in m.get("ingredients") or []))]
        out.append((f"Manifest {i} of {len(manifests)} ({role})", [r for r in rows if r[1]]))
    for x in rec.get("xmp") or []:
        if x.get("source_type"):
            rows = [("Found in", x["where"]), ("Digital source type", source_type_text(x["source_type"])),
                    ("Creator tool", x.get("creator_tool", "")),
                    ("Creators", ", ".join(x.get("creators") or []))]
            out.append(("XMP statement (unsigned)", [r for r in rows if r[1]]))
    out.append(("Read", [("Reader", entry.get("reader") or READER)]))
    return out


def flat(entry: dict | None) -> dict:
    """The CSV columns for one file: ``entry`` is its ``all_summaries`` value, or
    None when it has not been read."""
    if entry is None:
        return {"c2pa_credentials": CREDENTIAL_LABELS[None], "c2pa_ai": "",
                "c2pa_signer_stated": "", "c2pa_generator": "", "c2pa_source_types": "",
                "c2pa_actions": ""}
    rec = entry["record"]
    active = (rec.get("manifests") or [{}])[-1]
    return {
        "c2pa_credentials": CREDENTIAL_LABELS[entry["credentials"]],
        "c2pa_ai": AI_LABELS[entry["ai"]],
        "c2pa_signer_stated": entry["signer"] or "",
        "c2pa_generator": active.get("claim_generator", "") or next(
            (x["creator_tool"] for x in rec.get("xmp") or [] if x.get("creator_tool")), ""),
        "c2pa_source_types": ", ".join(_codes(rec)),
        "c2pa_actions": ", ".join(active.get("actions") or []),
    }


def _read_row(case_root, rec, row) -> tuple[int, dict | None, str | None]:
    try:
        with archive.local_copy(case_root, rec, row) as local:
            return row["id"], read_path(local), None
    except archive.ArchiveUnavailable as exc:
        return row["id"], None, str(exc)
    except OSError as exc:
        return row["id"], None, f"{type(exc).__name__}: {exc.strerror or exc}"


def read_pass(case, *, reread: bool = False, workers: int = 4, progress=None) -> dict:
    """Read every picture and video of the case not yet read (every one, with
    ``reread``). Returns counts: read, could not be read, and the credentials found.

    A file whose bytes cannot be produced (an archive no longer where the case
    recorded it, a file gone from disk) gets no row, so it stays "Not read" and a
    later pass tries it again."""
    where = f"kind IN ({','.join('?' * len(KINDS))})"
    if not reread:
        where += " AND id NOT IN (SELECT file_id FROM c2pa_meta)"
    rows = case.db.iter_files(where, KINDS)
    recs = archive.source_records(case)
    total = len(rows)
    counts = {"read": 0, "failed": 0, **{k: 0 for k in CREDENTIALS}}
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for file_id, result, _err in ex.map(
                lambda r: _read_row(case.root, recs.get(r["source"]), r), rows):
            done += 1
            if result is None:
                counts["failed"] += 1
            else:
                save(case.db, file_id, result)
                counts["read"] += 1
                counts[summarize(result)["credentials"]] += 1
            if done % 100 == 0 or done == total:
                case.db.commit()
                if progress:
                    progress(done, total)
    case.db.commit()
    case.db.audit_log(case.examiner, "c2pa_read", json.dumps(
        {**counts, "reader": READER, "reread": reread}))
    return counts
