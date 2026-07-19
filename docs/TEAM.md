# Team assignment — Saturday gameplan (can sorting)

**Goal:** messy table → organized. **Stage 1 = pick ONE can, place it in the taped zone.** Multi-can variation comes only after 150 clean episodes.

> The P1–P5 labels below are REUSED for new roles and do **not** match the folder
> numbering. Folders were intentionally **not** renamed (renaming breaks everyone's work
> and causes merge conflicts). This file is the source of truth for who does what.
> The README / GROUND_TRUTH still describe the older crumpled-paper plan — superseded here.

## Roles at a glance

| Person | Role | Works in | Hands off to |
|--------|------|----------|--------------|
| P1 | Teleop driver | `p4_data_collection/` | P3 (recorded episodes) |
| P2 | Resetter + data QC | `p4_data_collection/` | P3 (verified clean dataset) |
| P3 (Corbin) | Training pipeline | `p5_training/` | P4 (trained checkpoint) |
| P4 | Harness / run-loop | `p3_vlm_orchestrator/` | P5 (working autonomous pick) |
| P5 | Integration + bounty | repo root, `rebot_setup/` | demo / judges |

---

## P1 — Teleop driver

**Job:** physically drive the arm to record good pick episodes.

- **Warm up** ~15 min on the leader arm until picks feel smooth.
- **Record episodes:** `python -m p4_data_collection.record_episode --num 1` (one at a time), or `python -m p4_data_collection.batch_record --num 50` (press Enter between resets). Teleop practice: `./p4_data_collection/scripts/teleop.sh`.
- **Each episode:** pick the ONE can, place it in the zone, end the episode. Call "bad" out loud the instant a take is jerky so P2 deletes it.
- **Swap** with P2 every ~30–40 min to stay sharp.
- **Milestones:** 10 episodes ASAP → hand to P3 · 50 by mid-afternoon · 150 by evening.

**Hardware:** follower arm on `/dev/ttyACM0`, leader on `/dev/ttyUSB0`. Two cameras (`front` = overhead, `side` = 45°) must stay locked and in frame.

**Don't:** change `config/recording.yaml` mid-dataset · move a camera · pick more than one can per episode.

---

## P2 — Resetter + data QC

**Job:** keep the dataset clean. Bad data poisons a small dataset, so be ruthless.

- **Reset the can** to a NEW random spot every episode — cover corners, edges, center; use the whole reachable workspace.
- **Delete bad episodes on the spot.** Jerky-but-successful still gets deleted.
- **Keep a tally:** episode count + deleted count.
- **Verify format** after episodes: `python -m p4_data_collection.verify_episode_format` — must print **PASS**. This catches phantom second-arm channels and missing cameras.
- **Spot-check every ~25 episodes** in the LeRobot display (Rerun): both camera views visible, joints logged, nothing drifted.
- **Confirm cameras locked** once, then guard them: `python -m p4_data_collection.check_camera_lock --save-ref` (first time), `check_camera_lock` (later, must PASS). Nobody bumps cameras; tape stays down.

**Don't:** let a single bad episode into the dataset · let anyone move the rig.

---

## P3 — Training pipeline (Corbin)

**Job:** turn recorded episodes into trained models, retrain as data grows, hand the best checkpoint to P4.

**Where the data lives:** `~/.cache/huggingface/lerobot/<repo_id>` (repo_id is set in `config/recording.yaml`). Grab episodes 1–10 the moment they exist.

**Always verify before training:** `python -m p4_data_collection.verify_episode_format --dataset-root <path>` must PASS.

**Two model routes — start with the easy one:**

- **SmolVLA (do first — ready now, no cloud):** warm-start
  `lerobot/smolvla_base` in the repository's vendored LeRobot Python 3.11
  environment; do not train from scratch with `--policy.type=smolvla` on this
  small dataset. Use the tested install, camera rename/padding, PyAV, and MPS
  commands in [`SMOLVLA_TRAINING.md`](../SMOLVLA_TRAINING.md). Expect
  auth/format errors on the first real-data run — fixing them early is the
  point of starting at episode 10.
- **MolmoAct (bigger, more setup):** `python -m p5_training.build_dataset_mixture --dataset-root <path>` → `python -m p5_training.modal_finetune --mixture ...`. Runs on **Modal** (needs `pip install modal` + `modal token new` + credits). ⚠️ Runs a **placeholder trainer** until the real `train_cmd` + base checkpoint are set in `p5_training/configs/molmoact2_single_arm.yaml` — those come from the organizers.

**Retrain** checkpoints as data hits 50 / 100 / 150. Log which model looks better.

**Sync checkpoints off the venue machine** every couple hours (coordinate with P5).

**Evening:** hand P4 your best checkpoint for the first live eval on the real arm (P4 owns the arm run-loop; you provide the model + how to load it).

---

## P4 — Harness / run-loop

**Job:** make a trained model actually drive the arm, safely.

- **Run-loop:** load the policy → feed it camera frames + joint state → get actions → send to the arm. Repeat.
- **Borrow arm control** from `p1_arm_motion/arm_client.py` (`ArmClient` / `DryRunArmClient`) — import it, don't rewrite it. The existing `p3_vlm_orchestrator/orchestrator.py` and `run_clean.py` are VLM-based references for the loop shape.
- **Inference wrapper:** frames + joint state in, actions out, with **safety clamps** on workspace bounds (`config/workspace.yaml`, `config/arm.yaml`).
- **Episode logic:** when to start a pick, when to declare done, when to retry.
- **Keyboard kill switch** — non-negotiable for live runs.
- **Stub against a dummy policy NOW** so you're ready the second P3 hands you a checkpoint.
- **Stretch:** the "table is messy → keep picking until clear" outer loop.

**Needs from P3:** a checkpoint + the exact way to load and call it. Agree this format early.

**Don't:** run the arm without the kill switch + workspace clamps.

---

## P5 — Integration + bounty

**Job:** package the work for the bounty, keep everything backed up, tell the story.

- **Read the bounty post** (WhatsApp) and run `pixi run learn`; read their data-collection docs.
- **Go/no-go by early afternoon:** package our Rerun GUI as the **non-SO-101 integration bounty** (reBot URDF + collection + training, documented and replicable).
- **If yes:** repo cleanup, README, add the URDF, write reproducible steps.
- **Off-site sync** of all episodes + checkpoints (Drive / Hugging Face) every couple hours — this is the backup if the venue machine dies.
- **Film** collection b-roll + the first successful autonomous pick for the demo video.
- **Draft the 60-second pitch:** ambition, accuracy, consistency, harness + skill fit.

**Bounties on the table:** (1) Rerun integration for non-SO-101 robots ← chase this, we're 80% there · (2) viewer integration · (3) query API over episodes.

---

## Shared files — coordinate before editing (conflict hotspots)

| File | Who owns edits | Everyone else |
|------|----------------|---------------|
| `config/recording.yaml` | Data team (one person) | reads only |
| `config/arm.yaml`, `config/workspace.yaml` | P4 for clamps, P1 for ports | read only |
| `shared/` (handoff formats) | Group agreement only | — |

**Merge rule:** conflicts happen only when two people edit the *same lines of the same file*. Stay in your folder and this stays a non-issue. Small, frequent PRs + `git pull` before each session.

## Open items before/at collection start

- [ ] **Task swap paper → cans:** `config/recording.yaml` still says `single_task: "Pick crumpled paper..."` and `repo_id: .../crumpled_paper_molmoact2`. Data team updates both before episode 1.
- [ ] **MolmoAct:** confirm base checkpoint + training script with organizers; set `train_cmd` (until then it's a placeholder).
- [ ] **Modal:** confirm install + token + credits work before relying on it.

## Shared checkpoints (times)

| Time | Milestone |
|------|-----------|
| ~1pm | 10 episodes recorded, test train started (P3) |
| ~3pm | Conversion pipeline green, both model runs launched, bounty go/no-go |
| ~6pm | 100+ episodes, first checkpoint evaluated on arm |
| ~9pm | 150 episodes, best model picked, everything synced |
| 10:15pm | Hands off, pack, tomorrow plan agreed |
