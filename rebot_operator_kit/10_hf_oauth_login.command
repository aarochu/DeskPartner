#!/bin/zsh

set -euo pipefail
KIT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$KIT_ROOT/env.sh"

UV_BIN="${REBOT_UV_BIN:-$(command -v uv || true)}"
[[ -n "$UV_BIN" && -x "$UV_BIN" ]] || {
  print -u2 -- "uv is required. Install it from https://docs.astral.sh/uv/ first."
  exit 1
}

HF_VENV="$KIT_STATE_ROOT/hf-cli"
HF_BIN="$HF_VENV/bin/hf"

if [[ ! -x "$HF_BIN" ]]; then
  "$UV_BIN" venv "$HF_VENV"
fi
"$UV_BIN" pip install --python "$HF_VENV/bin/python" --upgrade hf

print -- "Starting Hugging Face browser OAuth. No token needs to be pasted."
HF_HOME="$HF_HOME" "$HF_BIN" auth login --force
HF_HOME="$HF_HOME" "$HF_BIN" auth whoami
