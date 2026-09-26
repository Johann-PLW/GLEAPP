"""The frozen build clears the downloaded-from-the-internet mark from its own DLLs.

On Windows ``name:Zone.Identifier`` is an alternate data stream of ``name``; elsewhere it
is an ordinary file whose name contains a colon. Both are created, found and removed
through the same path string, so these tests exercise the same calls on either.
"""

import ast
import os
import sys
from pathlib import Path

from gleapp import _zone_marks

ROOT = Path(__file__).resolve().parents[1]
MARK = "[ZoneTransfer]\r\nZoneId=3\r\n"


def _mark(path: Path) -> Path:
    stream = Path(str(path) + _zone_marks.STREAM)
    stream.write_text(MARK, encoding="utf-8")
    return stream


def _tree(tmp_path: Path):
    rt = tmp_path / "_internal" / "pythonnet" / "runtime"
    rt.mkdir(parents=True)
    dll = rt / "Python.Runtime.dll"
    upper = tmp_path / "_internal" / "WEBVIEW.DLL"
    other = tmp_path / "_internal" / "base_library.zip"
    unmarked = rt / "netstandard.dll"
    for f in (dll, upper, other, unmarked):
        f.write_bytes(b"x")
    return dll, upper, other, unmarked


def test_clear_removes_the_mark_from_every_dll_and_nothing_else(tmp_path):
    dll, upper, other, unmarked = _tree(tmp_path)
    marks = [_mark(dll), _mark(upper)]
    other_mark = _mark(other)

    assert _zone_marks.clear(tmp_path, windows=True) == 2

    assert not any(os.path.exists(m) for m in marks)
    assert os.path.exists(other_mark)          # not a DLL: left alone
    assert dll.read_bytes() == b"x"            # the file itself is untouched
    assert unmarked.exists()


def test_clear_does_nothing_off_windows(tmp_path):
    dll, *_ = _tree(tmp_path)
    mark = _mark(dll)
    assert _zone_marks.clear(tmp_path, windows=False) == 0
    assert os.path.exists(mark)


def test_clear_frozen_bundle_is_a_no_op_from_source(monkeypatch, tmp_path):
    dll, *_ = _tree(tmp_path)
    mark = _mark(dll)
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(_zone_marks.os, "name", "nt")  # only the frozen check may stop it
    assert _zone_marks.clear_frozen_bundle() == 0
    assert os.path.exists(mark)


def test_clear_frozen_bundle_clears_the_bundle_folder(monkeypatch, tmp_path):
    dll, *_ = _tree(tmp_path)
    mark = _mark(dll)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(_zone_marks.os, "name", "nt")
    assert _zone_marks.clear_frozen_bundle() == 1
    assert not os.path.exists(mark)


def _line_of(tree, predicate):
    return min(n.lineno for n in ast.walk(tree) if predicate(n))


def test_entrypoint_clears_marks_before_anything_can_load_pythonnet():
    """The clear has to run before --selfcheck and before the desktop shell is imported."""
    src = (ROOT / "packaging" / "entrypoint.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    clear_call = _line_of(tree, lambda n: isinstance(n, ast.Call)
                          and getattr(n.func, "id", None) == "clear_frozen_bundle")
    selfcheck = _line_of(tree, lambda n: isinstance(n, ast.Constant)
                         and n.value == "--selfcheck")
    desktop = _line_of(tree, lambda n: isinstance(n, ast.ImportFrom)
                       and n.module == "gleapp.desktop")
    assert clear_call < selfcheck < desktop
