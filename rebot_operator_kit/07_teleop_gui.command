#!/bin/zsh

set -euo pipefail

KIT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$KIT_ROOT/env.sh"
GUI_URL="http://127.0.0.1:8765/"

if curl -fsS "$GUI_URL/api/status" 2>/dev/null | grep -q '"state"'; then
  print -- "ReBot Teleop GUI is already running: $GUI_URL"
  open "$GUI_URL"
  exit 0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  print -u2 -- "ReBot Python environment is missing: $PYTHON_BIN"
  print -u2 -- "Run ../rebot_setup/setup.sh first or set RUNTIME_ROOT explicitly."
  read -r "?Press Return to close..."
  exit 1
fi

export PYTHONUNBUFFERED=1

print -- "Starting ReBot Teleop GUI at $GUI_URL"
print -- "Keep this terminal open. Press Ctrl+C here to stop the GUI."
exec "$PYTHON_BIN" "$KIT_ROOT/teleop_gui/server.py" \
  --host 127.0.0.1 \
  --port 8765 \
  --open-browser
