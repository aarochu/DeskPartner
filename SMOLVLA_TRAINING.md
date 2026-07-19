# SmolVLA — local training (Track B)

How to fine-tune a **SmolVLA** policy locally on this machine's GPU, using the
vendored LeRobot. This is the SmolVLA half of Track B; MolmoAct is trained
separately on Modal.

Verified working on an Apple Silicon Mac (MPS) on 2026-07-18.

**Fine-tune, don't train from scratch.** We warm-start from the pretrained
`lerobot/smolvla_base` checkpoint — a SmolVLA that already knows how to move an
arm — and adapt it to our data. This matches the official hackathon recipe
("warm-start from the SO-101 base"). Training from scratch (`--policy.type=smolvla`)
will not learn a usable policy on ~150 episodes, so we always pass
`--policy.path=lerobot/smolvla_base` instead.

---

## One-time setup

The system default Python (3.13) is too new for the ML stack — use **Python 3.11**.

```bash
# from the repo root
python3.11 -m venv .venv-lerobot
./.venv-lerobot/bin/python -m pip install --upgrade pip
./.venv-lerobot/bin/python -m pip install -e "./rebot_setup/vendor/rebot_lerobot/lerobot[smolvla]"
```

Confirm it worked:

```bash
./.venv-lerobot/bin/lerobot-train --help          # should print usage
./.venv-lerobot/bin/python -c "import torch; print('mps:', torch.backends.mps.is_available())"
```

## Important: the `pyav` flag is mandatory on this Mac

The default video reader (`torchcodec`) is broken here — it can't link the system
FFmpeg libraries and crashes the moment training reads a recording. **Every**
training command must therefore include:

```
--dataset.video_backend=pyav
```

`pyav` is already installed and bundles its own FFmpeg, so no system install is
needed. Leave this flag out and training fails at the first batch.

## Train

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 ./.venv-lerobot/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<DATASET_REPO_ID> \
  --dataset.video_backend=pyav \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --steps=5000 \
  --save_freq=1000 \
  --output_dir=outputs/smolvla/<run_name> \
  --wandb.enable=false
```

- Replace `<DATASET_REPO_ID>` with the recorded dataset (e.g. the LeRobot repo id
  the data team uses; a locally recorded dataset lives at
  `~/.cache/huggingface/lerobot/<repo_id>`).
- Checkpoints are written under `outputs/smolvla/<run_name>/checkpoints/`. The
  `pretrained_model/` folder inside each checkpoint is what the inference/harness
  side loads.
- `outputs/` is git-ignored — sync checkpoints off-machine (Drive / HF) rather
  than committing them.

## Smoke test (no real data needed)

To prove the machine can train before real recordings exist, run a few steps on a
public SmolVLA example dataset:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 ./.venv-lerobot/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --dataset.episodes='[0, 1]' \
  --dataset.video_backend=pyav \
  --policy.device=mps \
  --batch_size=2 --steps=20 --save_freq=20 --eval_freq=0 \
  --output_dir=outputs/smolvla/smoke_test \
  --wandb.enable=false
```

Success = loss prints and decreases, and a checkpoint appears under
`outputs/smolvla/smoke_test/checkpoints/`.
