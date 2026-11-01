#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "$ROOT"
export PATH="/opt/homebrew/opt/node@22/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
[[ "$(uname -s)" == "Darwin" ]] || { printf 'This launcher is for macOS.\n' >&2; exit 1; }
[[ -x "$ROOT/.venv-studio/bin/python" ]] || { printf 'Run SETUP MAC.command first. See docs/MACOS.md.\n' >&2; exit 1; }
exec "$ROOT/.venv-studio/bin/python" "$ROOT/tools/anamorphic-studio/launch.py" "$@"
