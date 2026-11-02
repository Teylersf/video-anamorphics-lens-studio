#!/bin/bash
# Local Apple Silicon setup. Never installs Homebrew, uses sudo, or starts Studio.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "$ROOT"
export PATH="/opt/homebrew/opt/ffmpeg-full/bin:/opt/homebrew/opt/node@22/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

fail() { printf '\nSetup stopped: %s\nSee docs/MACOS.md.\n' "$*" >&2; exit 1; }
trap 'printf "\nSetup did not complete. Resolve the error above, then run SETUP MAC.command again.\n" >&2' ERR

[[ "$(uname -s)" == "Darwin" ]] || fail "This setup is for macOS."
[[ "$(uname -m)" == "arm64" ]] || fail "Use a native Apple Silicon terminal, not Rosetta or an Intel Mac."
[[ "$(sysctl -in sysctl.proc_translated 2>/dev/null || printf '0')" != "1" ]] || fail "Rosetta is active. Reopen Terminal without Open using Rosetta."
xcode-select -p >/dev/null 2>&1 || fail "Install Apple's command-line tools first: xcode-select --install"

PYTHON="${PYTHON3:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -x /opt/homebrew/opt/python@3.13/bin/python3.13 ]]; then
        PYTHON=/opt/homebrew/opt/python@3.13/bin/python3.13
    elif command -v python3.13 >/dev/null 2>&1; then
        PYTHON="$(command -v python3.13)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    else
        fail "Install native Python 3.13. With Homebrew: brew install python@3.13"
    fi
fi
"$PYTHON" - <<'PY'
import platform
import sys
if sys.version_info[:2] != (3, 13):
    raise SystemExit('Python 3.13 is required for this setup; set PYTHON3 to its executable path.')
if platform.machine() != 'arm64':
    raise SystemExit('Python must run as native arm64, not x86_64 under Rosetta.')
if tuple(map(int, platform.mac_ver()[0].split('.')[:2])) < (14, 0):
    raise SystemExit('macOS 14 or newer is required by the pinned dependencies.')
print('Native Python:', sys.version.split()[0], platform.machine())
PY

command -v node >/dev/null 2>&1 || fail "Install Node.js 22. With Homebrew: brew install node@22"
command -v npm >/dev/null 2>&1 || fail "npm is missing. Reinstall Node.js 22."
node -e 'const [major,minor]=process.versions.node.split(".").map(Number); if(major!==22 || minor<12 || process.arch!=="arm64") {console.error("Use native arm64 Node.js 22.12 or newer in the 22.x series.");process.exit(1)}'
command -v ffmpeg >/dev/null 2>&1 || fail "Install the HDR-capable build: brew install ffmpeg-full"
command -v ffprobe >/dev/null 2>&1 || fail "ffprobe is missing. Install the full build: brew install ffmpeg-full"
ffmpeg -version >/dev/null
ffprobe -version >/dev/null
ffmpeg_filters="$(ffmpeg -hide_banner -filters 2>/dev/null)" || fail "Could not inspect the selected FFmpeg filters."
for required_filter in zscale tonemap; do
    printf '%s\n' "$ffmpeg_filters" | awk -v wanted="$required_filter" '$2 == wanted { found=1 } END { exit !found }' || fail "Selected FFmpeg ($(command -v ffmpeg)) lacks the $required_filter HDR preview filter. Run: brew install ffmpeg-full. These launchers prefer its keg-only bin directory without changing global links."
done

if [[ -e "$ROOT/.venv-studio" ]]; then
    [[ -x "$ROOT/.venv-studio/bin/python" ]] || fail "Existing .venv-studio is not a usable Mac environment. Move it aside manually; setup will not delete it."
else
    "$PYTHON" -m venv "$ROOT/.venv-studio"
fi
VENV_PYTHON="$ROOT/.venv-studio/bin/python"
"$VENV_PYTHON" - <<'PY'
import platform
import sys
from pathlib import Path
if sys.version_info[:2] != (3, 13) or platform.machine() != 'arm64':
    raise SystemExit('Existing .venv-studio must use native ARM64 Python 3.13. Move it aside manually to create a new one.')
if Path(sys.prefix).resolve() != (Path.cwd() / '.venv-studio').resolve():
    raise SystemExit('The selected interpreter is not using this project virtual environment.')
PY

printf '\nInstalling the pinned Mac environment…\n'
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install --only-binary=:all: -r "$ROOT/requirements-macos.txt"
"$VENV_PYTHON" -m pip check
"$VENV_PYTHON" - <<'PY'
import av
import numpy
import sys
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0, str(Path.cwd() / 'tools/anamorphic-studio'))
from runtime_backend import select_device, backend_info, empty_cache
from studio_engine import StudioRenderer, PROFILES, to_display_rgb
print('PyTorch:', torch.__version__, '| PyAV:', av.__version__)
device = select_device()
print('Renderer:', backend_info(device))
with torch.inference_mode():
    frame = torch.full((3, 73, 129), .12, device=device)
    frame[:, 32:41, 60:69] = 1
    original = frame.clone()
    for name in ('modern', 'vintage'):
        output = StudioRenderer(PROFILES[name]).render(frame)
        display = to_display_rgb(output, 'bt709')
        if output.shape != frame.shape or not bool(torch.isfinite(display).all()) or not torch.equal(frame, original):
            raise SystemExit(f'{name} renderer verification failed on {device}.')
    del display, output, original, frame
empty_cache(device)
print('Small Modern/Vintage effect renders passed. This is not a full-resolution performance benchmark.')
PY

printf '\nBuilding the local interface…\n'
(cd "$ROOT/app" && npm ci && npm run build)
printf '\nSetup complete. Setup did not start Studio.\nRun START ANAMORPHIC STUDIO.command when you want to open it.\n'
