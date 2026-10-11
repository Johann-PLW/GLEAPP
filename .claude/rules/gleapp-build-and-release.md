# Build and release

One driver, `packaging/build.py`, on every platform, in two phases so a signed build is
possible. The PowerShell script it replaced ran only on Windows and rewrote the tracked
spec on every build.

    python packaging/build.py exe               phase 1: dist/GLEAPP/ (one-folder); on macOS also dist/GLEAPP.app
    python packaging/build.py exe --onefile     phase 1: one executable; no installer can be built from this
    python packaging/build.py installer         phase 2: Windows dist/GLEAPP-Setup-<v>.exe, macOS dist/GLEAPP-<v>.dmg
    python packaging/build.py installer --sign-tool NAME
                                                Windows: Inno Setup signs the installer and the uninstaller
    python packaging/build.py all               both phases, unsigned; refuses --sign-tool and --onefile
    python packaging/build.py verify PATH ...   a signature is present and valid; --subject checks the signer

## Signing goes between the phases

Sign `dist/GLEAPP/GLEAPP.exe`, or codesign the `.app`, after phase 1 and before phase 2, or
the installer ships an unsigned executable inside a signed wrapper. `all` refuses
`--sign-tool` for exactly that reason. `verify` is the last step before anything is
uploaded: a signature that is missing or invalid looks like a good one until a user's
machine rejects it.

On Windows, `release.yml` signs with SignPath through its GitHub action, not with
`--sign-tool`: SignPath signs only what a workflow stored as an artifact of its own run,
which is how it checks the binary was built from this repository on GitHub's runners, so
nothing on the build machine can sign. Windows releases ship only the single-file
`dist/GLEAPP.exe`, so it is the one file sent, in one request, as `GLEAPP.exe`. The
third-party libraries packed inside it are not signed on their own. The artifact
configuration SignPath applies, `gleapp-portable`, is kept in `packaging/signpath/` and
must be edited there and in SignPath together. Signing appends to the executable and the
single file finds its archive by reading from its end, so it is smoke-tested again once
signed (`--version`, `--texworker`, `--selfcheck`). `build.py installer --sign-tool` still
works on Windows for a local build; releases do not use it.

## What the driver guarantees

The version is read from `gleapp/__init__.py` as text and passed to Inno Setup as
`/DAppVer`; `installer.iss` refuses to compile without it, so the installer and the app
cannot disagree. `ONEFILE` reaches the spec through the `GLEAPP_ONEFILE` environment
variable; the spec is never edited by a build. PyInstaller is pinned in the `[build]`
extra, which phase 1 installs. Every artifact is asserted to exist after the step that
makes it; an exit code is not evidence. `build/` and `dist/` are git-ignored.

## What is and is not wired up

Windows: releases ship only a `--onefile` build, zipped alone as the portable download so
it keeps the name `GLEAPP.exe`. There is no installer since 2026-10-11: it was a second
file to sign, as well as the folder build inside it. The cost is that the single file
unpacks itself to `%TEMP%` on every start, all of it (the content model alone is 121 MB),
so it is slower to start and blocked where AppLocker or WDAC forbid running programs from
`%TEMP%`; the footer sends those users to the source. `test_builds.yml` prints how long
three `--version` starts took. The video, texture and Windows.edb workers re-run
`sys.executable` and reuse the parent's unpacked folder rather than unpacking again: that
is PyInstaller's default since 6.9 (`PYINSTALLER_RESET_ENVIRONMENT` is what turns it off;
read in the 6.22.2 bootloader, not measured). The bootloader runs GLEAPP in a child process
and puts it in no job object, so ending the process that was started leaves GLEAPP
running; `test_builds.yml` ends the `web` smoke test with `taskkill /T /IM GLEAPP.exe` for
that reason. `test_builds.yml` builds and smoke-tests that single file, marked as
downloaded, and opens its window; it no longer builds the folder or the installer on
Windows. The Inno Setup script and `build.py installer` remain for local builds.

macOS: `.app` and `.dmg`; the release workflow signs the `.app` with a Developer ID and
the hardened runtime (`packaging/entitlements.plist`), then signs, notarises and staples
the `.dmg`, when the `MACOS_*` secrets are set. A tag refuses to publish without them; a
dispatched rehearsal builds unsigned. The signed bundle is smoke-tested again, because the
hardened runtime is what breaks a frozen app and the first smoke test ran the unsigned
one-folder build. Linux: the one-folder build runs; no AppImage yet, since an AppImage
needs a `.desktop` file of its own. Windows on ARM needs a CI leg on an ARM runner (the
release's asset name already carries the architecture); check that runner's availability
and billing for this repository's visibility before adding it. Windows and macOS ship
`packaging/gleapp.ico` and `packaging/gleapp.icns`.

## Windows signing with SignPath

Windows is signed by SignPath when the `SIGNPATH_API_TOKEN` repository secret is set. The
build job carries `actions: read` so SignPath can download the uploaded artifact, and the
SignPath GitHub App (github.com/apps/signpath) must be installed on the repository:
SignPath's documentation calls it optional, but without it every request fails with
"Failed to retrieve GitHub App token" (seen on iLEAPP, 2026-10-10). Only the owner of this
personal-account repository can install it. Repository variables:
`SIGNPATH_ORGANIZATION_ID`, `SIGNPATH_PROJECT_SLUG` (the LEAPPs share one SignPath project,
each with its own artifact configuration), `SIGNPATH_SIGNING_POLICY` (`test-signing` or
`release-signing`), `SIGNPATH_CERT_SUBJECT` (optional, passed to `verify --subject`) and,
under `test-signing`, `SIGNPATH_TEST_CERT_B64`, the root of the test certificate's chain as
a base64 `.cer`, which only the runner is told to trust so `verify` still checks a chain. A
test-signed binary is trusted by no Windows, so a tag signs only under `release-signing`;
under any other policy, or without the token, a tag builds unsigned and says so in a
warning, as releases did before signing was wired. A rehearsal signs under whichever policy
is set. When `release-signing` is in place, change the footer's "not signed yet" paragraph,
the README's Download section and its code signing policy, and make a tag refuse an
unsigned Windows build the way macOS does, since the footer will then promise a signature.

A repository secret is usable by a workflow on any branch of this repository, though never
by a pull request from a fork. Before `release-signing`, SignPath should require a manual
approval of each request, which shows the branch and commit it came from. The owner can go
further: rulesets restricting who creates `v*` tags, and a `release` environment admitting
only those refs, holding the token, with `environment: release` on the build job. Only the
owner can create environments and rulesets on this personal-account repository.
