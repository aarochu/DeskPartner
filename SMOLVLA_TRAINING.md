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
HF_HUB_ENABLE_HF_TRANSFER=1 ./.venv-lerobot/bin/lerobot-train \
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
HF_HUB_ENABLE_HF_TRANSFER=1 ./.venv-lerobot/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --dataset.episodes='[0, 1]' \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.side": "observation.images.camera1", "observation.images.up": "observation.images.camera2"}' \
  --policy.empty_cameras=1 \
  --policy.device=mps \
  --batch_size=2 --steps=20 --save_freq=20 --eval_freq=0 \
  --output_dir=outputs/smolvla/smoke_test \
  --wandb.enable=false
```

(This public dataset's cameras are `side`/`up`, hence the different rename map
than our real reBot data.) Success = loss prints and decreases, and a checkpoint
appears under `outputs/smolvla/smoke_test/checkpoints/`. Verified end-to-end on
MPS 2026-07-18.

## Training on multiple datasets (merge first)

`lerobot-train` only accepts ONE dataset (the multi-dataset path is disabled in
this vendored version). To train on several recorded datasets together, first
**merge** them into one with LeRobot's aggregate tool, then train on the result.
The datasets must share the same schema (action/state dims, camera keys, fps,
codebase version) — the aggregate tool validates this and refuses otherwise.

1. **Log in to Hugging Face** (private Cornerf datasets need a **classic Read**
   token — a fine-grained token shows zero datasets):

   ```bash
   hf auth login   # paste a Read token from https://huggingface.co/settings/tokens
   ```

2. **Download each dataset** at a pinned revision (get the sha from the dataset's
   HF page or `HfApi().dataset_info(repo_id).sha`):

   ```bash
   hf download <repo_id> --repo-type dataset --revision <sha> \
     --local-dir ~/rebot-training/data/<name>
   ```

3. **Merge** them into one combined dataset (originals stay read-only):

   ```python
   from pathlib import Path
   from lerobot.datasets.aggregate import aggregate_datasets
   aggregate_datasets(
       repo_ids=["<repo_a>", "<repo_b>"],
       aggr_repo_id="<combined_name>",
       roots=[Path("~/rebot-training/data/<a>"), Path("~/rebot-training/data/<b>")],
       aggr_root=Path("~/rebot-training/data/<combined>"),
   )
   ```

4. **Train** on the merged dataset (same warm-start command, point `--dataset.root`
   at the combined dataset; no `--dataset.revision` since it is local):

   ```bash
   HF_HUB_ENABLE_HF_TRANSFER=1 ./.venv-lerobot/bin/lerobot-train \
     --policy.path=lerobot/smolvla_base \
     --dataset.repo_id=<combined_name> \
     --dataset.root=~/rebot-training/data/<combined> \
     --dataset.video_backend=pyav \
     --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
     --policy.empty_cameras=1 \
     --policy.device=mps --policy.push_to_hub=false \
     --batch_size=4 --steps=20000 --save_freq=1000 --log_freq=200 --eval_freq=0 \
     --output_dir=outputs/smolvla/<run_name> \
     --wandb.enable=false
   ```

Done 2026-07-18: merged `rebot-can-sort-stage1-v1-smoke` (52 eps) +
`rebot-two-can-recycle-v2-smoke` (25 eps) → 77 eps / 51,207 frames, trained on MPS.

## Note on checkpoints and datasets

`outputs/` (trained checkpoints) and the datasets under `~/rebot-training/data/`
are NOT in git — checkpoints are hundreds of MB each and datasets are multi-GB.
Back checkpoints up to Hugging Face or Drive; the datasets already live on the HF
Hub under the `Cornerf` org.
