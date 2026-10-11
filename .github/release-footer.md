
---

## Which file to download

| Platform | File |
|---|---|
| Windows 10 or 11, 64-bit | `-windows-x64-portable.zip` |
| macOS, Apple silicon | `-macos-arm64.dmg` |
| macOS, Intel | `-macos-x64.dmg` |
| Linux, 64-bit | `-linux-x64.tar.gz` |

**On Windows the zip holds the whole program as one file, `GLEAPP.exe`**, which can be put
wherever you like; there is no installer any more. If an earlier release's installer put
GLEAPP in Program Files, uninstall it from Settings, Apps; your cases and settings are kept
elsewhere and stay. `GLEAPP.exe` unpacks itself to a temporary folder each time it starts,
so it is slower to open than an installed copy was, and it does not run where a policy
(AppLocker, WDAC) forbids running programs from the temporary folder. There, run GLEAPP
from source; the README has the steps.

## First launch

The macOS disk images are signed with a Developer ID and notarised by Apple, so they
open without a warning.

The Windows binaries are not signed yet, so SmartScreen says "Windows protected your
PC" the first time. Choose More info, then Run anyway. If you would rather not clear a
warning, run from source instead; the README has the steps.

## Verify what you downloaded

`SHA256SUMS.txt` covers every binary in this release. On macOS or Linux, from the folder
you downloaded into:

```bash
grep <the file you downloaded> SHA256SUMS.txt | shasum -a 256 -c -
```

Use `sha256sum` in place of `shasum -a 256` on Linux. On Windows, in PowerShell:

```powershell
(Get-FileHash -Algorithm SHA256 .\<the file you downloaded>).Hash
```

and compare it with the line in `SHA256SUMS.txt`, which is lower case.

## Linux

The build is made on Ubuntu 24.04, so it needs glibc 2.39 or newer and will not start on
an older distribution. The native desktop window needs a GTK or Qt webview toolkit that
pip does not install, and CI does not exercise it, so `gleapp web` in a browser is the
tested path there.

## Code signing policy

The [code signing policy](https://github.com/abrignoni/GLEAPP#code-signing-policy) covers
Windows signing, team roles and privacy.
