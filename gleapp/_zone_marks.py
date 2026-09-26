"""Clear Windows' downloaded-from-the-internet mark from the frozen build's own DLLs.

Extracting a downloaded zip with Explorer copies the zip's mark, an NTFS alternate data
stream named ``Zone.Identifier``, onto every file inside it. .NET Framework will not load
an assembly that carries a zone 3 (internet) mark, so the portable build could not open
its window: pythonnet stopped at ``Failed to resolve Python.Runtime.Loader.Initialize``
before any GLEAPP code ran. Removing the stream is what the Unblock box in a file's
Properties does.

Only ``.dll`` files are touched, only under the bundle's own folder, and only when running
as the frozen build on Windows. A file that has no mark, or whose mark cannot be removed
(a read-only share, a folder the account cannot write), is left as it is: the window then
starts or fails exactly as it did before.
"""

from __future__ import annotations

import os
import sys

STREAM = ":Zone.Identifier"


def clear(root: str | os.PathLike, *, windows: bool | None = None) -> int:
    """Remove the Zone.Identifier stream from every ``.dll`` under ``root``.

    Returns how many were removed. Never raises. ``windows`` defaults to whether this is
    Windows; elsewhere a colon names part of a file, not a stream, so nothing is done.
    """
    if windows is None:
        windows = os.name == "nt"
    if not windows:
        return 0
    removed = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if not name.lower().endswith(".dll"):
                continue
            try:
                os.remove(os.path.join(dirpath, name) + STREAM)
            except OSError:
                continue
            removed += 1
    return removed


def clear_frozen_bundle() -> int:
    """Clear the marks under the frozen build's ``_internal`` folder; 0 when not frozen."""
    root = getattr(sys, "_MEIPASS", None)
    if not getattr(sys, "frozen", False) or not root:
        return 0
    return clear(root)
