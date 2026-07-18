# P5 — MolmoAct 2 fine-tune + bake-off (Track B, bonus)

**Owns (per GROUND_TRUTH):** MolmoAct 2 **single-arm** fine-tune (LoRA or action-expert-only — **not** full FT), Modal train + serve, Sunday bake-off.

No zero-shot SO-100/DROID help for reBot — venue demos carry embodiment learning.

**Never blocks Track A.** Cut B with zero guilt if Friday teleop is rough or A slips (GT §8).

---

## Pipeline

```bash
# after P4 verified dataset handoff
python -m p5_training.build_dataset_mixture --dataset-root PATH_TO_LEROBOT_DS
python -m p5_training.modal_finetune --mixture p5_training/configs/mixture_deskpartner.yaml

# Saturday night eval (see GO_NO_GO.md — one data retry, then cut)
python -m p5_training.eval_checkpoint --checkpoint PATH --trials 10

# Sunday morning bake-off (GT §5 / §7)
python -m p5_training.bakeoff --trials 10 --checkpoint PATH
# read the unambiguous line:
# VERDICT: WINNER=scripted|molmoact2 (...)
```

## Decision gates

See [`GO_NO_GO.md`](./GO_NO_GO.md) and [`BAKEOFF.md`](./BAKEOFF.md).

- Sat night clearly bad → **one** retry with fixed/cleaner data only — no hyperparam spelunking.  
- Retry also bad → **cut Track B**.  
- Tie at bake-off → **scripted** (reliability).  
- Policy losing is fine — still a talking point.

## Venue confirm (GT §9) before training

- [ ] newt vs Ai2 `launch_scripts/train_lerobot.py`  
- [ ] base checkpoint string (default `allenai/MolmoAct2`)  
- [ ] set `train_cmd` in `configs/molmoact2_single_arm.yaml`  

Until `train_cmd` is set, `modal_finetune` still provisions GPU + wall-clock timeout + checkpoint volume, but uses a **placeholder trainer** so wiring can be tested.

## Pitch if B wins (GT §1)

> Fine-tuned a foundation VLA on data collected here, on hardware nobody else in the room ran.

## Layout

See [`STRUCTURE.md`](./STRUCTURE.md). Config: `configs/molmoact2_single_arm.yaml`.
