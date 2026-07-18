#!/bin/zsh

set -euo pipefail
KIT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$KIT_ROOT/env.sh"

DATASET_ID="local/deskpartner_crumpled_paper_smoke_v1"
DATASET_ROOT="$KIT_DATA_ROOT/crumpled_paper_smoke_v1"

"$PYTHON_BIN" "$KIT_ROOT/validate_dataset.py" \
  --repo-id "$DATASET_ID" \
  --root "$DATASET_ROOT" \
  --minimum-episodes 5

touch "$KIT_STATE_ROOT/dataset_smoke_validated"
print -- "Dataset validation passed."
read -r "?Press Return to close this window..."
