# DeskPartner — Desk Cleaner MVP

Overhead camera + VLM planner + reBot B601-DM arm that tidies a desk zone into trash / pen cup / tray, closed-loop until clean.

**Honorable Mention** at the [Embodied Metal Hackathon](https://luma.com/embodied-metal?tk=J10GoD), hosted by Mission Robotics, New Theory, Savant, and North Star.

**Ground truth / SOW:** [`docs/GROUND_TRUTH.md`](./docs/GROUND_TRUTH.md)  
**GitHub:** https://github.com/aarochu/DeskPartner  
**Hardware deck:** [ReBot Arm Workshop](https://docs.google.com/presentation/d/1LXgehBvwPy5EhWO7aaYQffvmlN4eS1zFmJWYoDcTm-c/edit)  
**Config:** single-arm follower+leader · Track B = **MolmoAct 2** fine-tune (Logitech overhead + Innomaker wrist cams locked)

---

## Repo layout

```
DeskPartner/
├── README.md                # this file — operate the system
├── docs/                    # SOW, team, track guides, bounty progress
├── config/                  # arm, cameras, workspace, destinations
├── shared/                  # handoff types & fake fixtures for parallel work
├── scripts/                 # host setup helpers
├── p1_arm_motion/           # Track A — IK pick/drop, party tricks
├── p2_vision_calibration/   # Track A — ArUco, CV, pixel→arm
├── p3_vlm_orchestrator/     # Track A — VLM + closed-loop state machine
├── p4_data_collection/      # Track B — teleop demos (LeRobot, single-arm only)
├── p5_training/             # Track B — MolmoAct 2 LoRA + bake-off
├── p5_rerun_port/           # Rerun bounty — log/record/query/export/replay on reBot
└── rebot_operator_kit/      # macOS GUI: teleop, two-camera collection, review, validation
```

**Docs index:** [`docs/README.md`](./docs/README.md)  
**Rerun non-SO-101 port:** [`docs/p5_rerun_port/`](./docs/p5_rerun_port/) · progress: [`docs/Rerun_bounty_progress.md`](./docs/Rerun_bounty_progress.md)

---

## Host requirements

| Item | Value |
|------|--------|
| OS | Ubuntu 22.04 |
| Python | 3.10+ |
| Follower | reBot B601-DM on `/dev/ttyACM0` (`can_adapter=damiao`) |
| Leader | reBot 102 on `/dev/ttyUSB0` |
| Cameras | Logitech BRIO overhead at index 0 + Innomaker wrist/claw at index 1. Built-in Mac webcam is not recorded. |
| Package managers | `uv` (SDK) + `pip` (LeRobot path) |

---

## One-time setup

### 1. Permissions

```bash
sudo chmod 666 /dev/ttyACM* /dev/ttyUSB*
# or: ./scripts/setup_permissions.sh
```

### 2. reBot SDK (Track A motion / IK / gravity-comp)

```bash
# from deck
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/vectorBH6/reBotArm_control_py ~/reBotArm_control_py
cd ~/reBotArm_control_py && uv sync
# SDK ports: reBotArm_control_py/config/rebotarm_dm.yaml (channel)
# DeskPartner ports: config/arm.yaml + config/recording.yaml (keep in sync)
```

Critical first script — **always before anything else after physical reconfig:**

```bash
cd ~/reBotArm_control_py
uv run python example/2_zero_and_read.py
```

Useful SDK examples (from the deck):

| Script | Use |
|--------|-----|
| `1_damiao_text.py` | Single-motor debug |
| `2_zero_and_read.py` | Zero all joints + live angles |
| `5_fk_test.py` / `6_ik_test.py` | FK/IK offline |
| `7_arm_ik_control.py` | Real-time IK jog |
| `8_arm_traj_control.py` | SE(3) trajectories |
| `9_gravity_compensation.py` | Weightless / teach-by-grabbing act 2 |

### 3. LeRobot path (Track B teleop + record)

```bash
mkdir -p ~/rebot_lerobot && cd ~/rebot_lerobot
git clone https://github.com/Seeed-Projects/lerobot.git
git clone https://github.com/Seeed-Projects/lerobot-teleoperator-rebot-arm-102.git
git clone https://github.com/Seeed-Projects/lerobot-robot-seeed-b601.git
pip install -e ./lerobot
pip install -e ./lerobot-teleoperator-rebot-arm-102
pip install -e ./lerobot-robot-seeed-b601
pip install motorbridge
```

Find ports / cameras:

```bash
lerobot-find-port
lerobot-find-cameras opencv
```

Update [`config/arm.yaml`](./config/arm.yaml) and [`config/cameras.yaml`](./config/cameras.yaml) with the values you get.

### 4. Calibrate arms (LeRobot)

Follower auto-calibrates on run; place B601 at zero with gripper closed first.

```bash
# Leader
lerobot-calibrate \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=rebot_arm_102_leader
```

### 5. DeskPartner Python env (harness)

```bash
cd /path/to/DeskPartner
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config/.env.example config/.env   # add VLM API keys
```

---

## Friday night bring-up (order matters)

1. **Zero** follower (`2_zero_and_read.py`).
2. **Home pose** — fully out of camera frame (P1).
3. **Reach-check** all four zone corners before taping.
4. Tape zone + ArUco 0–3 + mount camera 60–75 cm; **lock exposure/WB**.
5. P2: homography + plane→arm. **Exit: click pixel → tip ≤ 5 mm.**
6. P1: one hardcoded crumpled-paper pick-and-drop.
7. P4: teleop sanity + one throwaway episode.

Do not start VLM wiring until step 5 passes.

---

## Operating the cleaner (Track A demo)

```bash
source .venv/bin/activate

# Optional: CV-only if wifi/API dies
export DESKPARTNER_PERCEPTION=cv   # or: vlm (default)

python -m p3_vlm_orchestrator.run_clean \
  --config config/workspace.yaml \
  --arm-config config/arm.yaml
```

Loop (owned by P3):

`home → photo → plan → pick → drop → repeat` with max 2 retries per object, then skip.

Logs land in `runs/<timestamp>/` (photos + JSON decisions). Show this screen to judges.

### Perception switch

| Mode | Env / flag | When |
|------|------------|------|
| `vlm` | default | Normal demo |
| `cv` | `DESKPARTNER_PERCEPTION=cv` | API down / slow wifi |

Rehearse the switch once so it is boring.

### Recalibration (camera bumped)

See [`docs/p2_vision_calibration/RECALIBRATION.md`](./docs/p2_vision_calibration/RECALIBRATION.md) — target **≤ 10 minutes**.

---

## Track B — data & training (bonus)

For the tested macOS workflow, start with
[`docs/rebot_operator_kit/README.md`](./docs/rebot_operator_kit/README.md), run
`rebot_setup/setup.sh` on a new machine, then double-click
`rebot_operator_kit/07_teleop_gui.command`.

**MolmoAct 2 · single-arm:** fine-tune a foundation VLA (LoRA or action-expert-only — not full FT). Station is normally bimanual; we record **one arm only**.

**Cameras:** Logitech overhead (`front`, index 0) + Innomaker wrist/claw (`side`, index 1). Lock both before episode 1 — MolmoAct 2 trains on whatever views are in the dataset. Keep the wrist lens and gripper contact point unobstructed.

Teleop preview (dual cam):

```bash
# see p4_data_collection/scripts/teleop.sh
```

If joint directions feel inverted, apply the deck's `joint_directions` tuning (see `docs/p4_data_collection/README.md`).

**Friday night trap:** after the throwaway episode, run `python -m p4_data_collection.verify_episode_format` (must PASS — no phantom second-arm channels). Do not wait until Saturday.

```bash
python -m p4_data_collection.check_camera_lock --save-ref   # once cams locked
python -m p4_data_collection.record_episode --num 1
python -m p4_data_collection.verify_episode_format
python -m p4_data_collection.batch_record --num 50          # Saturday midday
```

Fine-tune on Modal (P5) — confirm newt vs Ai2 MolmoAct 2 scripts with organizers:

```bash
# see docs/p5_training/, configs/molmoact2_single_arm.yaml
python -m p5_training.bakeoff --trials 10 --checkpoint PATH   # Sunday: VERDICT line
```

Bake-off Sunday AM: 10 scripted vs 10 MolmoAct 2, same objects. Winner ships.

---

## Party tricks (second act)

```bash
cd ~/reBotArm_control_py
uv run python example/9_gravity_compensation.py   # weightless — hand arm to judge
# teach-by-grabbing replay: p1_arm_motion/teach_replay.py (stub)
```

Hand on e-stop during early runs. Speed limits in `config/arm.yaml`.

---

## Fallback tiers (demo day)

1. Full VLM autonomy  
2. CV-only autonomy  
3. Pre-staged canned run (`p3_vlm_orchestrator/canned_run.py`)  
4. Backup video  

Every tier is a working demo. One designated demo driver.

---

## Arm contention schedule

| Window | Owner |
|--------|--------|
| Friday night | Track A |
| Saturday morning | Track A |
| Saturday midday | Track B data block (P3 works on photos only) |
| Saturday night / Sunday AM | Track B eval slots |

Track B never blocks A. If Friday teleop is rough and you are four people: **cut B**, put the fourth on polish / wipe / backup video.

---

## Person folders

| Code | Docs | Role |
|------|------|------|
| [`p1_arm_motion/`](./p1_arm_motion/) | [`docs/p1_arm_motion/`](./docs/p1_arm_motion/) | Arm, motion, destinations, gravity-comp |
| [`p2_vision_calibration/`](./p2_vision_calibration/) | [`docs/p2_vision_calibration/`](./docs/p2_vision_calibration/) | Camera, ArUco, CV, calibration |
| [`p3_vlm_orchestrator/`](./p3_vlm_orchestrator/) | [`docs/p3_vlm_orchestrator/`](./docs/p3_vlm_orchestrator/) | VLM + closed-loop harness |
| [`p4_data_collection/`](./p4_data_collection/) | [`docs/p4_data_collection/`](./docs/p4_data_collection/) | Teleop + LeRobot episodes |
| [`p5_training/`](./p5_training/) | [`docs/p5_training/`](./docs/p5_training/) | MolmoAct 2 LoRA + bake-off |
| [`p5_rerun_port/`](./p5_rerun_port/) | [`docs/p5_rerun_port/`](./docs/p5_rerun_port/) | Rerun bounty port |
| [`rebot_operator_kit/`](./rebot_operator_kit/) | [`docs/rebot_operator_kit/`](./docs/rebot_operator_kit/) | macOS operator GUI |

Read each area's docs under `docs/` before coding.

---

## Safety habits

- Hand on e-stop during early bring-up.  
- Respect `max_speed` / `max_accel` in config.  
- Never photo with arm in frame (occludes + confuses VLM).  
- Foam tape on gripper fingers; top-down grasps only.  
- C-clamp the base before demos.  
