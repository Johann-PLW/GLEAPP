"""The Windows open-file dialog, shown from a thread of its own in a single-threaded
apartment.

pywebview shows its dialog on whichever thread asks, here a Flask request thread, which
is in the multithreaded apartment (pythonnet's default). The shell's file dialog expects
a single-threaded one. Picking one file got away with it; with ``Multiselect`` on, the
dialog stopped responding as soon as a file was selected (measured 2026-10-17 on
Windows 11 with pywebview 6.2.1: the same file and keypress returned at once with one
file allowed and never returned with several). This module runs the dialog on a .NET
thread set to STA, owned by a hidden top-most form so it opens in front.
"""
from __future__ import annotations


def open_files(file_types: tuple[str, ...], *, multiple: bool = True,
               title: str | None = None, directory: str | None = None) -> tuple[str, ...] | None:
    """Show the dialog and wait for it. ``file_types`` are pywebview-style
    ``"Label (*.a;*.b)"`` strings. Returns the paths chosen, or None if cancelled."""
    import re

    # pythonnet, loaded by pywebview already; System is .NET, which pylint cannot see
    # pylint: disable=import-error,no-member
    import clr
    clr.AddReference("System.Windows.Forms")
    from System.Threading import ApartmentState, Thread, ThreadStart
    from System.Windows import Forms

    filters = []
    for ft in file_types:
        m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", ft)
        label, pattern = (m.group(1), m.group(2)) if m else (ft, "*.*")
        filters.append(f"{label} ({pattern})|{pattern}")
    result: dict = {}

    def show():
        try:
            owner = Forms.Form()
            owner.TopMost = True
            owner.ShowInTaskbar = False
            owner.FormBorderStyle = Forms.FormBorderStyle.None_ \
                if hasattr(Forms.FormBorderStyle, "None_") else getattr(Forms.FormBorderStyle, "None")
            owner.Opacity = 0
            owner.Show()
            dialog = Forms.OpenFileDialog()
            dialog.Multiselect = multiple
            dialog.RestoreDirectory = True
            if filters:
                dialog.Filter = "|".join(filters)
            if directory:
                dialog.InitialDirectory = directory
            if title:
                dialog.Title = title
            if dialog.ShowDialog(owner) == Forms.DialogResult.OK:
                result["paths"] = tuple(dialog.FileNames)
            owner.Close()
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            result["error"] = exc

    t = Thread(ThreadStart(show))
    t.SetApartmentState(ApartmentState.STA)
    t.Start()
    t.Join()
    if "error" in result:
        raise result["error"]
    return result.get("paths")
