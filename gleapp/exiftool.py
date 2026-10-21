"""Full metadata from ExifTool, for the details pane.

GLEAPP's own reading (``metadata.extract_image``) keeps a handful of fields. ExifTool
reads the rest: maker notes, XMP, IPTC, lens and exposure, QuickTime. It is run
only to read, in its ``-stay_open`` batch mode so one process answers every request,
and only when a file is opened in the details pane; the answer is cached in the case.

Which ExifTool: ``$GLEAPP_EXIFTOOL``, then the copy a frozen Windows build carries
(``exiftool/exiftool.exe`` beside the bundle), then ``exiftool`` on PATH. None found
means the section says so; nothing else depends on it.

ExifTool is Phil Harvey's, distributed under the same terms as Perl; the Windows
package bundles Strawberry Perl. Their license texts ship with it.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

# Groups that describe the examiner's copy rather than the evidence: its name,
# folder, permissions and times on this machine, and ExifTool's own version line.
_DROP_GROUPS = {"System", "ExifTool"}
_READY = "{ready}"

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_job = None   # Windows job object the batch process is tied to (see _tie_to_us)


def find() -> str | None:
    """The ExifTool executable to use, or None."""
    env = os.environ.get("GLEAPP_EXIFTOOL")
    if env and Path(env).is_file():
        return env
    base = getattr(sys, "_MEIPASS", None)
    if base:
        bundled = Path(base) / "exiftool" / ("exiftool.exe" if os.name == "nt" else "exiftool")
        if bundled.is_file():
            return str(bundled)
    return shutil.which("exiftool")


def _tie_to_us(proc: subprocess.Popen) -> None:
    """On Windows, put the batch process (and the Perl it starts) in a job object
    that ends them when GLEAPP ends, however it ends. Without it, the ExifTool
    process outlived a GLEAPP window that was closed, holding files in the build
    folder open. Elsewhere ``stop`` at exit is what ends it."""
    global _job  # pylint: disable=global-statement
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int,
                                                ctypes.c_void_p, wintypes.DWORD)
        k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        if _job is None:
            class _Basic(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                            ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD),
                            ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t),
                            ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t),
                            ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class _Extended(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", _Basic),
                            ("IoInfo", ctypes.c_uint64 * 6),
                            ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t),
                            ("PeakJobMemoryUsed", ctypes.c_size_t)]
            job = k32.CreateJobObjectW(None, None)
            if not job:
                return
            info = _Extended()
            info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
            k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
            _job = job
        k32.AssignProcessToJobObject(_job, int(proc._handle))  # pylint: disable=protected-access
    except (OSError, AttributeError):
        pass


def _start() -> subprocess.Popen:
    exe = find()
    if not exe:
        raise FileNotFoundError("ExifTool is not available")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        [exe, "-stay_open", "True", "-@", "-"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        creationflags=flags)
    _tie_to_us(proc)
    return proc


def _ask(args: list[str]) -> str:
    global _proc  # pylint: disable=global-statement
    if _proc is None or _proc.poll() is not None:
        _proc = _start()
    assert _proc.stdin is not None and _proc.stdout is not None
    _proc.stdin.write(("\n".join(args) + "\n-execute\n").encode("utf-8"))
    _proc.stdin.flush()
    out = bytearray()
    while True:
        line = _proc.stdout.readline()
        if not line:
            _proc = None
            raise RuntimeError("ExifTool stopped")
        if line.strip() == _READY.encode():
            break
        out += line
    return out.decode("utf-8", errors="replace")


def version() -> str | None:
    try:
        with _lock:
            return _ask(["-ver"]).strip() or None
    except (OSError, RuntimeError):
        return None


def read(path: str | Path) -> list[dict]:
    """Every tag ExifTool reports for ``path``, as ``{"family", "group", "tag",
    "value"}``: ``family`` is ExifTool's top-level group (EXIF, MakerNotes, XMP, ...)
    and ``group`` the specific one (IFD0, GPS, XMP-dc, ...),
    in the order it gives them. Binary values stay as ExifTool's size placeholder."""
    args = ["-json", "-G0:1", "-a", "-charset", "filename=utf8",
            "-api", "LargeFileSupport=1", str(path)]
    with _lock:
        raw = _ask(args)
    try:
        doc = json.loads(raw or "[]")
    except ValueError as exc:
        raise RuntimeError("ExifTool gave no readable answer") from exc
    rec = doc[0] if doc else {}
    out = []
    for key, value in rec.items():
        if key == "SourceFile" or ":" not in key:
            continue
        parts = key.split(":")
        if len(parts) >= 3:
            family, group, tag = parts[0], parts[1], ":".join(parts[2:])
        else:
            family = group = parts[0]
            tag = parts[1]
        if family in _DROP_GROUPS or group in _DROP_GROUPS:
            continue
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        out.append({"family": family, "group": group, "tag": tag, "value": str(value)})
    return out


def for_row(case, row) -> dict:
    """ExifTool's reading of a case row, from the case if it was read before, else
    read now and kept. ``{"available", "version", "fields"}``, with ``error`` when
    the file could not be read; ``available`` is false when no ExifTool is found.
    A row whose bytes live in an archive or disk image is read from a temporary copy."""
    from . import archive
    got = case.db.conn.execute(
        "SELECT version, fields FROM exiftool_meta WHERE file_id = ?",
        (row["id"],)).fetchone()
    if got:
        fields = json.loads(got["fields"])
        if all("family" in x for x in fields):
            return {"available": True, "version": got["version"], "fields": fields}
        # read before the top-level group was recorded: read again below
    if not find():
        return {"available": False, "fields": []}
    rec = archive.source_record(case, row["source"]) if row["source"] else None
    try:
        with archive.local_copy(case.root, rec, row) as p:
            if not Path(p).is_file():
                return {"available": True, "fields": [], "error": "the file is not on disk"}
            fields = read(p)
    except archive.ArchiveUnavailable as exc:
        return {"available": True, "fields": [], "error": str(exc)}
    except (OSError, RuntimeError) as exc:
        return {"available": True, "fields": [], "error": str(exc)}
    ver = version()
    with case.db.lock:
        case.db.conn.execute(
            "INSERT OR REPLACE INTO exiftool_meta(file_id, version, fields) VALUES(?, ?, ?)",
            (row["id"], ver, json.dumps(fields)))
        case.db.commit()
    return {"available": True, "version": ver, "fields": fields}


# The categories a report can pick from, in the order the export dialog lists them,
# each a readable name over ExifTool's top-level groups. "other" takes the rest.
CATEGORIES = (
    ("exif", "Camera & capture (EXIF)"),
    ("gps", "GPS"),
    ("makernotes", "Maker notes"),
    ("xmp", "XMP"),
    ("iptc", "IPTC"),
    ("video", "Video"),
    ("composite", "Calculated (Composite)"),
    ("file", "File & format"),
    ("other", "Other"),
)
_VIDEO = {"QuickTime", "RIFF", "Matroska", "M2TS", "H264", "MPEG", "ASF", "FLV", "MXF"}
_FILE = {"File", "JFIF", "PNG", "ICC_Profile", "Photoshop", "JPEG", "GIF", "BMP",
         "Adobe", "APP14", "FlashPix", "MPF", "PrintIM"}


def category(field: dict) -> str:
    """Which of ``CATEGORIES`` a field from ``read`` belongs to."""
    fam, grp = field.get("family", ""), field.get("group", "")
    if grp == "GPS" or (fam == "Composite" and field.get("tag", "").startswith("GPS")):
        return "gps"
    if fam == "EXIF":
        return "exif"
    return {"MakerNotes": "makernotes", "XMP": "xmp", "IPTC": "iptc",
            "Composite": "composite"}.get(fam) or (
        "video" if fam in _VIDEO else "file" if fam in _FILE else "other")


def stop() -> None:
    """End the batch process, if one is running."""
    global _proc  # pylint: disable=global-statement
    with _lock:
        if _proc is not None and _proc.poll() is None:
            try:
                assert _proc.stdin is not None
                _proc.stdin.write(b"-stay_open\nFalse\n")
                _proc.stdin.flush()
                _proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                _proc.kill()
        _proc = None


atexit.register(stop)
