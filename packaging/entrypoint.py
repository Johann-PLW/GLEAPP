"""Frozen-app entry point. Kept outside the package so PyInstaller runs it as a
plain script while ``gleapp`` stays importable as a package."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # isolated workers re-invoke GLEAPP.exe (crash-safe decoding)
    if len(sys.argv) > 1 and sys.argv[1] == "--vidworker":
        from gleapp._vidworker import main as vw_main
        sys.exit(vw_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "--texworker":
        from gleapp._texworker import main as tw_main
        sys.exit(tw_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "--edbworker":
        from gleapp._edbworker import main as edb_main
        sys.exit(edb_main(sys.argv[2:]))
    # --version answers without a window, so a frozen build can be smoke-tested
    # headless and it matches `python gleapp.py --version`. desktop.main() does not
    # handle it.
    if "--version" in sys.argv[1:]:
        from gleapp import __version__
        print(f"GLEAPP {__version__}")
        sys.exit(0)
    # Before anything loads pythonnet: a portable build extracted from a downloaded zip
    # carries the internet mark on every DLL, and .NET Framework refuses to load them.
    # Workers never load .NET, so they skip the folder walk. See gleapp/_zone_marks.py.
    from gleapp._zone_marks import clear_frozen_bundle
    clear_frozen_bundle()
    # --selfcheck imports what the desktop shell imports, the native stack included,
    # and exits without opening a window. --version answers above this line and
    # --texworker only reaches Pillow, so neither of them loads cv2: the macOS build of
    # v2026.5.0 passed both and still could not start, because a harfbuzz collision in
    # the bundle made importing cv2 fail. A frozen build that cannot import these
    # cannot run, so this is what a smoke test has to call.
    # On Windows it also imports pywebview's WinForms backend, which is where pythonnet
    # loads .NET and the window's assemblies: gleapp.desktop imports webview only inside
    # main(), so without it the check passed on a v2026.5.1 portable build that could not
    # open its window. This build has no console, so an uncaught exception would open a
    # dialog and wait for a click; the check reports the failure and exits 1 instead.
    if "--selfcheck" in sys.argv[1:]:
        import importlib
        import traceback
        names = ["numpy", "PIL.Image", "cv2", "gleapp.web.app", "gleapp.desktop"]
        if sys.platform == "win32":
            names.append("webview.platforms.winforms")
        for name in names:
            try:
                importlib.import_module(name)
            except Exception:  # pylint: disable=broad-exception-caught
                traceback.print_exc()
                print(f"selfcheck failed importing {name}", file=sys.stderr)
                sys.exit(1)
            print(f"ok {name}")
        print("selfcheck passed")
        sys.exit(0)
    from gleapp.desktop import main
    sys.exit(main())
