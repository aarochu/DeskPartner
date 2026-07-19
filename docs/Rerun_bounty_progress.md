# Rerun bounty progress (reBot / non-SO-101)

**Prize:** $1,000 — Best end-to-end port of a non-SO-101 robot  
**Goal:** Reproduce the full SO-101 Rerun data/learning loop on **reBot**, not just URDF or telemetry.

**Reference repo:** https://github.com/mission-robotics-ai/so100-hackathon  
**Team gameplan:** `saturday_gameplan.md` (Person 5 = integration + bounty)  
**Code:** [`p5_rerun_port/`](../p5_rerun_port/) · docs: [`p5_rerun_port/`](./p5_rerun_port/) · branch `feat/rebot-rerun-port`

---

## Progress (update as we go)

| Item | Status | Notes |
|------|--------|--------|
| Orient / map SO→reBot | done | Step 1 table filled; package scaffolded |
| Slice A — `log_rebot` | done (dry-run) | `--fake` verified; live arm still TODO on venue |
| Slice B — calibrate docs | done | Documented in `p5_rerun_port/README.md` |
| Slice C — `record_episode` → `.rrd` | done (dry-run) | Fake episode → catalog + `.traj.npz` + frames |
| Slice D — `query_dataset` | done | Tag filter + entity inspect via traj sidecar |
| Slice D2 — **Rerun Query API** | done (dry-run) | `query_api_cli`: Server + `reader()` + goal-vs-position; see `QUERY_API.md` |
| Slice E — `export_lerobot` | done (dry-run) | `robot_type=seeed_b601_dm_follower`, 7 joints, front/side |
| Slice F — `replay_episode` | done (dry-run) | Fake replay OK; live follower TODO on venue |
| Live arm e2e (no `--fake`) | TODO | Venue: log → record → export → replay |
| Demo video + 60s pitch | TODO | |
| Off-site sync (RRDs / checkpoints) | TODO | |

**Last updated:** 2026-07-18 — Query API CLI added (`rr.server.Server` + dataframe reader); dry-run green; live arm still TODO.

**Pipeline flowchart:** see [`p5_rerun_port/README.md`](./p5_rerun_port/README.md#how-it-connects-flowchart) (GUI track vs Rerun bounty loop vs training).  
**Query API docs:** [`p5_rerun_port/QUERY_API.md`](./p5_rerun_port/QUERY_API.md)

### Other Rerun prizes (status)

| Prize | Status |
|-------|--------|
| $1k non-SO-101 port | Code + dry-run; live e2e TODO |
| $2k Query API | Implemented: `query_api_cli` + docs (dry-run on fake `.rrd`) |
| $2k interesting Viewer | Not implemented (no custom blueprints/views yet) |

---

## What “done” looks like

Judges expect a **working, reproducible** loop:

| Stage | SO-101 reference | Your reBot proof |
|--------|------------------|------------------|
| Setup / cal | `calibrate-so100`, `log-so100` | Zero + cal + cameras + URDF in Rerun |
| Collect | `record-episode` → `.rrd` | Teleop episodes as Rerun recordings |
| Curate | `query-dataset` / tags | List, filter, tag episodes in local catalog |
| Export | `export-lerobot` / `finetune` | Selected eps → **LeRobot v3** |
| Close loop | `replay-episode` + policy | Replay trajectory **or** deploy trained policy |

**Not enough:** URDF-only in the viewer, or LeRobot record without Rerun + catalog.

---

## Useful links

### Reference pipeline
- Repo: https://github.com/mission-robotics-ai/so100-hackathon
- Fine-tune guide: https://missionrobotics.ai/hackers/fine-tune
- Call-your-model: https://missionrobotics.ai/hackers/call-your-model

### Rerun
- Getting started: https://rerun.io/docs/getting-started
- Examples: https://rerun.io/examples
- Viewer: https://rerun.io/docs/getting-started/configure-the-viewer
- Blueprints: https://rerun.io/docs/concepts/visualization/blueprints
- Custom views: https://rerun.io/docs/howto/visualization/extend-ui
- Query API: https://rerun.io/docs/concepts/query-and-transform/dataframe-queries
- URDF: https://rerun.io/docs/howto/logging-and-ingestion/urdf
- Export RRD → LeRobot: https://www.rerun.io/examples/robotics/rerun_export
- `rerun-lerobot`: https://github.com/rerun-io/rerun-lerobot

---

## Workspace setup

Use **two Cursor windows** (sibling folders, not nested):

```text
Desktop/
  DeskPartner/          ← your reBot / team work (this repo)
  so100-hackathon/      ← reference only
```

### Clone reference

```powershell
cd C:\Users\aaron\Desktop
git clone https://github.com/mission-robotics-ai/so100-hackathon.git
cd so100-hackathon
```

### Install Pixi (Windows) if missing

```powershell
irm https://pixi.sh/install.ps1 | iex
```

Close the terminal, open a **new** PowerShell, then:

```powershell
pixi --version
cd C:\Users\aaron\Desktop\so100-hackathon
pixi install
```

If `pixi` still not found:

```powershell
$env:Path += ";$env:USERPROFILE\.pixi\bin"
pixi --version
```

### Start the guided course

```powershell
pixi run learn
# open http://localhost:3000
```

Follow set up → collect → refine → prepare → deploy once. CLI-only path is in their README for debugging.

---

## Step 0 — Orient (30–45 min, no arm)

- [x] Install Pixi, clone `so100-hackathon`, run `pixi install`.
- [x] Run `pixi run learn` / skim README CLI path: server → cameras → log → calibrate → teleop → record → query → export → replay.
- [x] Write a **diff list**: every `so100` / `so101` / Feetech / USB-modem assumption → Damiao / reBot / Seeed LeRobot types / your COM or `/dev` path.
- [ ] Early afternoon: **go/no-go** on packaging the reBot Rerun GUI as this bounty (gameplan target ~3pm).

---

## Step 1 — Research like a port engineer

For each `pixi run …` command, fill:

| Command | Entrypoint file | SO-specific assumption | reBot replacement | Status |
|---------|-----------------|------------------------|-------------------|--------|
| `so100-server` | `tools/apps/so100_server.py` | On `/arms/connect` opens dual-arm Feetech teleop via `ArmSession`; setup chains `calibrate_so100` leader→follower and `log_so100 --teleop`; cal files under `calibrations/*.json` with leader/follower kind | **Local catalog** via `p5_rerun_port/catalog.py` (`recordings/catalog.json`) — no gRPC server required for bounty loop | done (thin) |
| `check-cameras` | `tools/apps/check_cameras.py` | No arm bus; macOS USB/AVFoundation/OpenCV probe + skip Continuity/built-in webcam (same filter as arm logging). Entity paths `camera/cam{N}` | Cams logged inside `log_rebot` / `record_episode` as `camera/cam0`/`cam1`; hygiene tools still `preview_camera` / `check_camera_lock` | good enough |
| `log-so100` | `tools/apps/log_so100.py` → `apis/log_arms.py` | Feetech STS3215 ×6 @ 1e6 baud; ports `/dev/cu.usbmodem*`; entities `leader`/`follower` + `{name}/position`; URDF `data/so100/so100.urdf` or `data/so101_leader/…`; joints `shoulder_pan`…`gripper` | **`python -m p5_rerun_port.log_rebot`** (`--fake` / live LeRobot). Entities `follower/position` (+ `/goal` with `--teleop`), cams, URDF | done |
| `calibrate-so100` | `tools/apps/calibrate_so100.py` → `apis/calibrate.py` | Positional `leader`\|`follower`; writes portugal JSON `calibrations/<usb_id>.json` (degrees / LINEAR gripper) **and** LeRobot dual-write under `so101_follower` / `so101_leader` | Documented in `p5_rerun_port/README.md` (SDK zero + `lerobot-calibrate` leader) | good enough |
| `teleop-so100` | `tools/apps/log_so100.py --teleop --fps 60` | Needs one calibrated leader + follower; maps leader° → follower raw via **follower** cal; logs `{follower}/goal`; Feetech P/torque tuned like lerobot SO follower | `log_rebot --teleop` + `record_episode` (logs `follower/goal`) | done |
| `record-episode` | `tools/apps/record_episode.py` | Opens SO arms with `teleop=True` (exclusive serial); writes calibrated **degrees** into `.rrd` with same entity layout as log/teleop | **`python -m p5_rerun_port.record_episode`** → `.rrd` + `.traj.npz` + frames + catalog tag | done |
| `query-dataset` | `tools/apps/query_dataset.py` | Catalog-only (robot-agnostic). Docs/examples assume SO entity paths like `follower/position` | **`python -m p5_rerun_port.query_dataset`** | done |
| `export-lerobot` | `tools/apps/export_lerobot.py` (+ `_export_lerobot_writer.py`) | Action=`*/goal`, state=`*/position`; degrees → lerobot ±100 via follower calib; writer stamps `robot_type="so100_follower"`; motors `DEFAULT_MOTOR_NAMES` | **`python -m p5_rerun_port.export_lerobot`** — `robot_type=seeed_b601_dm_follower`, 7 joints, `front`/`side` | done |
| `replay-episode` | `tools/apps/replay_episode.py` | Reads first `*/goal` as calibrated degrees; opens calibrated **follower** only on usbmodem; Feetech torque + `configure_follower_control` | **`python -m p5_rerun_port.replay_episode`** (`--fake` or live) | done |
| `finetune` | `tools/apps/finetune.py` | Reuses `export_lerobot` (same SO units/schema/`robot_type`) then `newt finetune`; no live bus | **DeskPartner:** `p5_training/modal_finetune.py` (unchanged) | exists (diff stack) |

### Shared reBot constants (fill these into every port)

| Item | Value |
|------|--------|
| Follower | `seeed_b601_dm_follower` · port `/dev/ttyACM0` · `can_adapter: damiao` · id `follower1` |
| Leader | `rebot_arm_102_leader` · port `/dev/ttyUSB0` · id `rebot_arm_102_leader` |
| Config | `config/arm.yaml`, `config/recording.yaml`, `config/cameras.yaml` |
| Joints (7) | `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_yaw`, `wrist_roll`, `gripper` |
| Cameras | LeRobot: `front`, `side` · Rerun (keep SO-compat): `camera/cam0`, `camera/cam1` |
| URDF | `~/reBotArm_control_py/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf` |
| Windows ports | Use `lerobot-find-port` → `COM*` instead of `/dev/ttyACM0` / `ttyUSB0` |

Also note:

- Rerun **entity paths** — prefer keeping `follower/position`, `follower/goal`, `camera/cam{N}` so `query`/`export`/`replay` stay close to SO
- Joint **units** — confirm Seeed/LeRobot convention before copying SO degrees→±100
- Dual-write cal into HF cache under `seeed_b601_dm_follower` / `rebot_arm_102_leader` if you mirror their portugal JSON story
- **Critical DOF diff:** SO = 6 joints; reBot = **7** (`wrist_yaw` extra) — export/replay motor lists must match

---

## Step 2 — Map what you already have

- [x] reBot SDK: zero, teleop, URDF (`reBotArm_control_py` / venue)
- [x] LeRobot: `seeed_b601_dm_follower` + `rebot_arm_102_leader`
- [x] Wired into product shape via `p5_rerun_port` (Rerun path; P4 LeRobot record left alone)

```text
live teleop → .rrd + catalog tags → export LeRobot v3 → replay / policy deploy
```

---

## Step 3 — Build in thin vertical slices

Ship **one good episode** through the full loop before scaling.

### Slice A — Smoke (arm + Rerun + URDF) — done (dry-run)
- [x] `python -m p5_rerun_port.log_rebot --fake --teleop`
- [ ] Live: viewer shows arm moving + both real cams

### Slice B — Calibrate docs — done
- [x] Documented in `p5_rerun_port/README.md` (SDK zero + leader `lerobot-calibrate`)

### Slice C — Record → `.rrd` — done (dry-run)
- [x] Fake episode → `recordings/<dataset>/episode_XX.rrd` + catalog tag
- [ ] Live teleop can → zone episode tagged `Good episode`

### Slice D — Query / curate — done
- [x] `query_dataset` list / `--tag` / `--entity follower/position`

### Slice E — Export LeRobot v3 — done (dry-run)
- [x] Fallback export with reBot schema (`seeed_b601_dm_follower`, 7 joints, front/side)
- [ ] Live export used by P3 train path (optional; P4 LeRobot record still available)

### Slice F — Close the loop — done (dry-run)
- [x] `replay_episode --fake` plays `follower/goal` trajectory
- [ ] Live replay on follower; stretch = deploy P3 checkpoint

---

## Step 4 — Team sync (don’t steal the arm)

| Time | Person 5 | Need from others |
|------|----------|------------------|
| Morning | `pixi run learn`, entity map, README skeleton | — |
| ~1pm | Confirm first 10 eps in Rerun + tagged | P1/P2 episodes |
| ~3pm | **Bounty go/no-go**; export green on those 10 | P3 starts test train |
| Afternoon | Harden export/replay; sync RRDs + datasets off-site | Growing dataset |
| Evening | Film b-roll + first auto pick; 60s pitch draft | P3 checkpoint + P4 harness |

**You own:** integration, docs, bounty packaging, backup sync, video/pitch.  
**P3 owns:** train. **P4 owns:** policy run-loop.

---

## Step 5 — SO-101 CLI cheat sheet (reference)

Long-lived server (leave running):

```bash
pixi run so100-server
```

Setup:

```bash
pixi run check-cameras
pixi run log-so100
pixi run calibrate-so100 leader
pixi run calibrate-so100 follower
pixi run teleop-so100
```

Collect:

```bash
pixi run record-episode -- --dataset my_task --task "Pick up the can" --tag "Good episode"
```

Refine:

```bash
pixi run query-dataset
pixi run query-dataset -- --dataset my_task
pixi run query-dataset -- --dataset my_task --tag "Good episode"
```

Prepare:

```bash
pixi run export-lerobot -- --dataset my_task --repo-id <team>/my_task
pixi run finetune -- --dataset my_task   # or --dry-run after export only
```

Deploy:

```bash
pixi run replay-episode -- --dataset my_task --episode episode_01 --speed 0.5
```

Your job is to recreate this shape for **reBot** (`log-rebot`, `record-episode`, etc.).

---

## Step 6 — Bounty deliverable checklist

- [x] README: reproduce commands in `p5_rerun_port/README.md` (venue + `--fake`)
- [x] Diff from SO-101 documented (hardware, entities, cal, units) — this file + package README
- [x] Scripts/tasks: log, record, query, export-lerobot, replay
- [x] Sample: dry-run `.rrd` + exported LeRobot folder locally (gitignored under `recordings/` / `datasets/`)
- [ ] Video: record → Rerun inspect → export → replay/policy
- [x] Explicit: **single-arm**, no phantom bimanual channels (config + export schema)
- [ ] Episodes + checkpoints synced off-site every couple hours
- [ ] 60s pitch drafted: ambition, accuracy, consistency, harness + skill fit
- [ ] Live venue validation (drop `--fake`)

---

## Data collection rules (shared with P1/P2)

- One can picked per episode
- New spot every episode; cover whole reachable workspace
- Jerky takes deleted — never train on bad episodes
- Cameras + lighting fixed; window out of frame
- Spot-check in Rerun every ~25 episodes

---

## Failure modes to avoid

- Shipping URDF-only “integration”
- Recording only in raw LeRobot with no curated `.rrd` + catalog
- Export units mismatch (degrees vs normalized)
- Bad episodes in the export set
- Docs that only work on your personal machine paths

---

## One-line priority

**Get one can episode through Rerun → tag → LeRobot v3 → replay on the follower, then document it.**

Episode count (→150) is P1/P2. Your win condition is the **pipeline product**.

---

## Shared team checkpoints (from Saturday gameplan)

- [ ] ~1pm: 10 episodes recorded, test train started
- [ ] ~3pm: conversion green, model runs launched, **bounty go/no-go**
- [ ] ~6pm: 100+ episodes, first checkpoint on arm
- [ ] ~9pm: 150 episodes, best model, everything synced
- [ ] 10:15pm: hands off, pack, Sunday plan
