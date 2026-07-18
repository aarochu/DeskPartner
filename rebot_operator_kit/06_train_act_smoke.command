#!/bin/zsh

set -euo pipefail
KIT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$KIT_ROOT/env.sh"

DATASET_ID="local/deskpartner_crumpled_paper_smoke_v1"
DATASET_ROOT="$KIT_DATA_ROOT/crumpled_paper_smoke_v1"
MODEL_ROOT="$KIT_MODEL_ROOT/act_crumpled_paper_smoke_v1"
STEPS="${ACT_STEPS:-2000}"

"$PYTHON_BIN" "$KIT_ROOT/validate_dataset.py" \
  --repo-id "$DATASET_ID" \
  --root "$DATASET_ROOT" \
  --minimum-episodes 5

POLICY_DEVICE="${POLICY_DEVICE:-$($PYTHON_BIN -c 'import torch; mps=getattr(torch.backends,"mps",None); print("cuda" if torch.cuda.is_available() else "mps" if mps and mps.is_available() else "cpu")')}"

if [[ -d "$MODEL_ROOT" ]] && [[ -n "$(find "$MODEL_ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  print -u2 -- "Model output is not empty: $MODEL_ROOT"
  print -u2 -- "Choose a new model version or archive the existing output."
  exit 2
fi

print -- "ACT smoke training"
print -- "device: $POLICY_DEVICE"
print -- "steps:  $STEPS"
print -- "output: $MODEL_ROOT"
if [[ "$POLICY_DEVICE" == "cpu" ]]; then
  print -- "WARNING: MPS is unavailable on this Mac. CPU training will be slow."
fi
print -- "This five-episode run checks the pipeline only; it is not a useful autonomous policy."
read -r "answer?Type TRAIN to begin: "
[[ "$answer" == "TRAIN" ]] || { print -u2 -- "Training confirmation not received."; exit 1; }

"$TRAIN_BIN" \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=act \
  --policy.device="$POLICY_DEVICE" \
  --policy.push_to_hub=false \
  --output_dir="$MODEL_ROOT" \
  --job_name=act_crumpled_paper_smoke_v1 \
  --batch_size=8 \
  --num_workers=0 \
  --steps="$STEPS" \
  --save_freq=1000 \
  --wandb.enable=false

print -- "Training smoke test finished."
read -r "?Press Return to close this window..."
