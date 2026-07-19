#!/bin/zsh

set -euo pipefail
KIT_ROOT="$(cd "$(dirname "$0")" && pwd)"

usage() {
  print -- "Usage:"
  print -- "  $0 hub <owner-or-org/repo> <destination> [revision]"
  print -- "  $0 verify <dataset-directory>"
}

[[ $# -ge 2 ]] || { usage; exit 2; }
MODE="$1"

if [[ "$MODE" == "verify" ]]; then
  DATASET_ROOT="$2"
elif [[ "$MODE" == "hub" ]]; then
  [[ $# -ge 3 ]] || { usage; exit 2; }
  REPO_ID="$2"
  DATASET_ROOT="$3"
  REVISION="${4:-main}"
  if [[ -f "$KIT_ROOT/env.sh" ]]; then
    export KIT_ROOT
    source "$KIT_ROOT/env.sh"
  fi
  HF_BIN="${VENV:+$VENV/bin/hf}"
  if [[ ! -x "$HF_BIN" ]]; then
    HF_BIN="$(command -v hf || true)"
  fi
  [[ -n "$HF_BIN" && -x "$HF_BIN" ]] || {
    print -u2 -- "Install huggingface_hub or configure the ReBot runtime first."
    exit 1
  }
  "$HF_BIN" auth whoami >/dev/null || {
    print -u2 -- "Private dataset login required. Run: $HF_BIN auth login"
    exit 1
  }
  mkdir -p "$DATASET_ROOT"
  "$HF_BIN" download "$REPO_ID" --repo-type dataset \
    --revision "$REVISION" --local-dir "$DATASET_ROOT"
else
  usage
  exit 2
fi

if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN:-}" ]]; then
  PYTHON_FOR_VERIFY="$PYTHON_BIN"
else
  PYTHON_FOR_VERIFY="$(command -v python3)"
fi
"$PYTHON_FOR_VERIFY" "$KIT_ROOT/dataset_share.py" verify --dataset-root "$DATASET_ROOT"

if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN:-}" ]]; then
  DATASET_NAME="$(basename "$DATASET_ROOT")"
  "$PYTHON_BIN" "$KIT_ROOT/validate_dataset.py" \
    --repo-id "local/$DATASET_NAME" \
    --root "$DATASET_ROOT" \
    --minimum-episodes 1
  print -- "Checksum and full LeRobot validation passed."
else
  print -- "Checksums passed. Configure the pinned ReBot runtime for full LeRobot validation."
fi
