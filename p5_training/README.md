# P5 — Training & Bake-off (Track B, bonus)

**Owns:** **MolmoAct 2** single-arm fine-tune, dataset mixture from P4, checkpoint eval, Modal serving, Sunday bake-off.

## Constraints

- Fine-tune a foundation VLA — **not** training ACT from scratch, **not** full fine-tuning (that's a cluster job).
- Hackathon scope: **LoRA** (`enable_lora_vlm`) and/or **action-expert-only** — action expert stays trainable.
- **Single-arm** setup descriptor + our exact camera keys (`front`, `side`) + actions matching our control mode.
- Confirm with organizers tonight: fine-tune via their **newt** platform **or** Ai2's released MolmoAct 2 scripts on our **Modal** GPUs. Either way Modal can serve inference; model outputs **action chunks** so per-call latency is survivable.
- Training does not need the arm. Eval slots: Saturday night + Sunday morning.

Refs:
- [MolmoAct 2 (Ai2)](https://allenai.org/blog/molmoact2)
- [allenai/molmoact2](https://github.com/allenai/molmoact2)
- [LeRobot MolmoAct2 policy docs](https://huggingface.co/docs/lerobot/main/en/molmoact2)

## Modal

Get team access Friday night. Weave into the pitch if Track B wins:

> Fine-tuned a foundation VLA on data collected here, on hardware nobody else in the room ran.

## Bake-off (Sunday AM)

| Arm | Trials | Success = |
|-----|--------|-----------|
| Skill A scripted | 10 | paper in trash, no assist |
| Skill B MolmoAct 2 | 10 | same objects / staging |

Winner runs the public demo. Loser sits out. Tie → prefer scripted.

## Cut criteria

- Friday teleop rough or A slipping → cut B.
- First LoRA run looks bad Saturday night → **one** retry with fixed data, then cut B. No 4am hyperparameter spelunking.
