# SmolVLA — local training (Track B)

How to fine-tune a **SmolVLA** policy locally on this machine's GPU, using the
vendored LeRobot. This is the SmolVLA half of Track B; MolmoAct is trained
separately on Modal.

Verified working on an Apple Silicon Mac (MPS) on 2026-07-18 using the
repository's standard LeRobot environment,
`rebot_setup/vendor/rebot_lerobot/.venv`.

**Fine-tune, don't train from scratch.** We warm-start from the pretrained
`lerobot/smolvla_base` checkpoint — a SmolVLA that already knows how to move an
arm — and adapt it to our data. This matches the official hackathon recipe
("warm-start from the SO-101 base"). Training from scratch (`--policy.type=smolvla`)
will not learn a usable policy on ~150 episodes, so we always pass
`--policy.path=lerobot/smolvla_base` instead.

---

## One-time setup

The system default Python (3.13) is too new for the ML stack. Build the
repository's pinned **Python 3.11** environment, then add the SmolVLA policy
dependencies to that same environment:

```bash
# from the repo root
./rebot_setup/setup.sh
uv pip install \
  --python rebot_setup/vendor/rebot_lerobot/.venv/bin/python \
  'transformers>=4.57.1,<5.0.0' \
  'num2words>=0.5.14,<0.6.0' \
  'accelerate>=1.7.0,<2.0.0' \
  'safetensors>=0.4.3,<1.0.0'
```

The `uv pip` form is intentional: the environment built by `setup.sh` does not
install the `pip` module. Also do not install the full vendored
`lerobot[smolvla]` extra into this shared environment: that package metadata
pins `rerun-sdk<0.27` and would downgrade the Rerun 0.34 runtime required by the
Query API integration. `setup.sh` has already installed vendored LeRobot; the
four packages above are the only SmolVLA additions it needs.

Confirm it worked:

```bash
rebot_setup/vendor/rebot_lerobot/.venv/bin/lerobot-train --help
rebot_setup/vendor/rebot_lerobot/.venv/bin/python -c \
  "import torch; print('mps:', torch.backends.mps.is_available())"
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

## Important: camera names must match the base model

`smolvla_base` was pretrained expecting **three** cameras named
`observation.images.camera1`, `camera2`, `camera3`. Our dataset's cameras are
named differently (`front`, `side`) and there are only two of them, so training
stops with a "Feature mismatch" error unless you:

1. **Rename** your camera keys to `camera1`/`camera2` with `--rename_map`, and
2. **Pad** the missing third camera with `--policy.empty_cameras=1`.

For our reBot dataset (`front` = overhead, `side` = 45°) that is:

```
--rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}'
--policy.empty_cameras=1
```

(Confirm the exact camera key names against the real dataset's `meta/info.json`
before the first real run — adjust the map if they differ.)

## Train

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 rebot_setup/vendor/rebot_lerobot/.venv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<DATASET_REPO_ID> \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
  --policy.empty_cameras=1 \
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
HF_HUB_ENABLE_HF_TRANSFER=1 rebot_setup/vendor/rebot_lerobot/.venv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --dataset.episodes='[0, 1]' \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.side": "observation.images.camera1", "observation.images.up": "observation.images.camera2"}' \
  --policy.empty_cameras=1 \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --batch_size=2 --steps=20 --log_freq=1 --save_freq=20 --eval_freq=0 \
  --num_workers=0 \
  --output_dir=outputs/smolvla/smoke_test \
  --wandb.enable=false
```

(This public dataset's cameras are `side`/`up`, hence the different rename map
than our real reBot data.) Success = all 20 steps report finite loss and
`checkpoints/000020/pretrained_model/model.safetensors` exists. Do not require a
monotonic loss curve from only 20 shuffled mini-batches. This exact command path
was verified end-to-end on MPS on 2026-07-18: it loaded 2 public episodes / 569
frames, trained 100M of 450M parameters, printed finite losses for every step,
saved the step-20 checkpoint, and exited 0. It does not validate our reBot data
or autonomous arm motion.
