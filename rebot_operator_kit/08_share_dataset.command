#!/bin/zsh

set -euo pipefail
KIT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$KIT_ROOT/env.sh"

usage() {
  print -- "Usage:"
  print -- "  $0 hub <owner-or-org/repo> [dataset-name]"
  print -- "  $0 rsync <user@host:/absolute/destination> [dataset-name]"
  print -- ""
  print -- "Default dataset: rebot-can-sort-stage1-v1-smoke"
}

[[ $# -ge 2 ]] || { usage; exit 2; }
MODE="$1"
DESTINATION="$2"
DATASET_NAME="${3:-rebot-can-sort-stage1-v1-smoke}"
DATASET_ROOT="$KIT_DATA_ROOT/$DATASET_NAME"
MINIMUM_EPISODES="${REBOT_SHARE_MINIMUM_EPISODES:-1}"
HF_BIN="$VENV/bin/hf"

[[ -d "$DATASET_ROOT" ]] || { print -u2 -- "Dataset not found: $DATASET_ROOT"; exit 1; }

case "$MODE" in
  hub)
    [[ "$DESTINATION" == */* ]] || { print -u2 -- "Hub repo must be owner-or-org/repo"; exit 2; }
    [[ "${DESTINATION##*/}" == "$DATASET_NAME" ]] || {
      print -u2 -- "Hub repo slug must match the dataset contract: $DATASET_NAME"
      exit 2
    }
    VISIBILITY="private"
    ;;
  rsync)
    [[ "$DESTINATION" == *:* ]] || { print -u2 -- "rsync destination must be user@host:/path"; exit 2; }
    VISIBILITY="team"
    ;;
  *)
    usage
    exit 2
    ;;
esac

"$PYTHON_BIN" "$KIT_ROOT/dataset_share.py" guard \
  --base-url "${REBOT_GUI_URL:-http://127.0.0.1:8765}" \
  --dataset "$DATASET_NAME"

SNAPSHOT_STAMP="$(date -u '+%Y%m%dT%H%M%SZ')-$PPID-$RANDOM"
SNAPSHOT_PARENT="$KIT_STATE_ROOT/share-snapshots/$DATASET_NAME/$SNAPSHOT_STAMP"
SNAPSHOT_ROOT="$SNAPSHOT_PARENT/$DATASET_NAME"
mkdir -p "$SNAPSHOT_PARENT"
if ! /bin/cp -cR "$DATASET_ROOT" "$SNAPSHOT_ROOT" 2>/dev/null; then
  mkdir -p "$SNAPSHOT_ROOT"
  rsync -a "$DATASET_ROOT/" "$SNAPSHOT_ROOT/"
fi

# Catch a collector that began after the first check. Never publish that copy.
"$PYTHON_BIN" "$KIT_ROOT/dataset_share.py" guard \
  --base-url "${REBOT_GUI_URL:-http://127.0.0.1:8765}" \
  --dataset "$DATASET_NAME"

"$PYTHON_BIN" "$KIT_ROOT/dataset_share.py" prepare \
  --dataset-root "$SNAPSHOT_ROOT" \
  --attempt-root "$KIT_ROOT/training-runs/attempts" \
  --destination "$DESTINATION" \
  --visibility "$VISIBILITY"

"$PYTHON_BIN" "$KIT_ROOT/validate_dataset.py" \
  --repo-id "local/$DATASET_NAME" \
  --root "$SNAPSHOT_ROOT" \
  --minimum-episodes "$MINIMUM_EPISODES"

"$PYTHON_BIN" "$KIT_ROOT/dataset_share.py" verify --dataset-root "$SNAPSHOT_ROOT"

# The active dataset has not been read since the snapshot. This last check
# proves the checkpoint was prepared only after an idle, finalized PASS.
"$PYTHON_BIN" "$KIT_ROOT/dataset_share.py" guard \
  --base-url "${REBOT_GUI_URL:-http://127.0.0.1:8765}" \
  --dataset "$DATASET_NAME"

if [[ "$MODE" == "rsync" ]]; then
  rsync -av --partial --delay-updates --info=progress2 \
    "$SNAPSHOT_ROOT/" "$DESTINATION/"
  print -- "Transfer complete. Ask the receiver to run:"
  print -- "  ./09_receive_dataset.command verify <received-dataset-directory>"
  exit 0
fi

[[ -x "$HF_BIN" ]] || { print -u2 -- "Hugging Face CLI not found: $HF_BIN"; exit 1; }
"$HF_BIN" auth whoami >/dev/null || {
  print -u2 -- "Hugging Face login required. Run: $HF_BIN auth login"
  exit 1
}
"$HF_BIN" repo create "$DESTINATION" --repo-type dataset --private --exist-ok
HF_XET_HIGH_PERFORMANCE=1 "$HF_BIN" upload-large-folder \
  "$DESTINATION" "$SNAPSHOT_ROOT" --repo-type dataset --private

VERIFY_ROOT="$KIT_STATE_ROOT/share-verification/$DATASET_NAME"
mkdir -p "$VERIFY_ROOT"
"$HF_BIN" download "$DESTINATION" SHARE_MANIFEST.json \
  --repo-type dataset --local-dir "$VERIFY_ROOT"
cmp "$SNAPSHOT_ROOT/SHARE_MANIFEST.json" "$VERIFY_ROOT/SHARE_MANIFEST.json"

REVISION="$($PYTHON_BIN -c 'from huggingface_hub import HfApi; import sys; print(HfApi().repo_info(sys.argv[1], repo_type="dataset").sha)' "$DESTINATION")"
print -- "Uploaded and remotely verified: https://huggingface.co/datasets/$DESTINATION"
print -- "Immutable revision: $REVISION"
