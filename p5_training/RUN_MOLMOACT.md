# Running a MolmoAct2 fine-tune (momo pipeline)

Real MolmoAct2 fine-tune on Modal, via LeRobot's `lerobot-train`. No placeholder,
no organizer sign-off needed — MolmoAct2 is a native LeRobot policy.
Proven working on the reBot can-sort smoke batch (5 eps → loss 0.115→0.071, L4, ~5 min).

All commands run from `DeskPartner-momo/` (branch `train-momo`) using the venv:
`./.venv/bin/...`

## One-time setup (already done)
- `pip install modal && modal token new`  (workspace: csplatti)
- Modal secret `huggingface` (only needed for Hub pulls; volume path below skips it)
- Base model `allenai/MolmoAct2` is cached in the `deskpartner-molmoact2-ckpts` volume.

## Compute note
- Runs on **L4 (24 GB)** — works with NO Modal payment method, fits action-expert-only
  (~15 GB used). Set `DESKPARTNER_MODAL_GPU=A100-80GB` (needs a card on file) for speed/LoRA.

## Launch a run — 3 steps

### 1. Put the dataset on the Modal volume (robust; no HF token needed)
```bash
./.venv/bin/modal volume put deskpartner-data \
    /path/to/local/lerobot-dataset  /<HF_REPO_ID>
```
The training reads it from `/data/<HF_REPO_ID>`. The local copy IS the pinned
revision — record that revision in the run manifest.
(Alternative: skip the upload and pull from the Hub by passing `--dataset-root ""`,
but that needs the `huggingface` secret to have read access to the private repo.)

### 2. Point the config at it — edit `configs/molmoact2_single_arm.yaml`
```yaml
exp_name: <unique_name_per_run>          # e.g. rebot_v2_run1  (own checkpoint dir)
dataset_repo_id: <HF_REPO_ID>
dataset_revision: <full_commit_sha>      # pinned; recorded for reproducibility
train:
  steps: 8000                            # ~150 demos; watch the loss and adjust
  batch_size: 16                         # L4 fits action-expert-only at 16
```
Keep: 7-dim action/state, `image_keys: [observation.images.front, observation.images.side]`,
`finetune.mode: action_expert_only` (or `lora`), `normalize_gripper: true`.

### 3. Train (streams the loss curve, saves checkpoints to the volume)
```bash
./.venv/bin/python -m p5_training.modal_finetune
```
Checkpoints land in the `deskpartner-molmoact2-ckpts` volume at `/<exp_name>/checkpoint-*`.
Preview the exact command first with `--dry-print`.

## Get the trained weights OFF Modal
```bash
./.venv/bin/modal volume ls  deskpartner-molmoact2-ckpts /<exp_name>
./.venv/bin/modal volume get deskpartner-molmoact2-ckpts /<exp_name>  ./outputs/momo/<exp_name>
```
Each `checkpoint-N/pretrained_model/` is a LeRobot MolmoAct2 checkpoint (loads with
`--policy.path=...` for more training, eval, or `lerobot-rollout` on the arm).

## Hard constraints (do not violate)
- Keep the **7-value** action/state (do NOT convert to 6-value SO-101).
- Dataset is **read-only**; never edit episodes (incl. the deliberate QUALITY_EXCEPTION).
- Never full fine-tune (`build_train_argv` hard-blocks it).
- Never hard-code an HF token; use the Modal secret.
