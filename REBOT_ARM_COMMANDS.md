# ReBot arm replication commands

This runbook reproduces the tested **single-arm** DeskPartner hardware path on a
second computer:

- reBot B601-DM follower (`seeed_b601_dm_follower`)
- reBot Arm 102 leader (`rebot_arm_102_leader`)
- Damiao serial adapter for the follower
- optional fixed `front` + `side` OpenCV cameras

It deliberately keeps the robot runtime outside the DeskPartner Python
environment. Do not install the SDK, LeRobot, and the DeskPartner harness into
one shared global Python.

The commands below were captured from the working bring-up on 2026-07-17/18.
They use LeRobot `0.4.4` and pin every upstream repository to the exact commit
used during that bring-up.

The physical follower/leader bring-up was verified on macOS. The Ubuntu 22.04
commands match DeskPartner's target host and upstream interfaces but were not
re-run against the physical arm during this capture.

## 0. Safety boundary

Do not run a command that writes motor targets until all of these are true:

- The B601 follower base is clamped down and uses its **24 V** supply.
- The reBot 102 leader uses its **12 V** supply. Never connect the follower's
  24 V supply to the leader.
- The leader UC-01 red LED is on and its 3-pin servo bus is secure.
- Both arms are in the documented folded zero pose with grippers closed before
  calibration.
- The full follower workspace is clear by at least 1 m.
- One person can immediately press the hardware e-stop or remove motor power.
- Only one process owns each serial port.
- Support the follower before disabling motor power; an unpowered pose can fall
  under gravity.

The current leader driver is not a certified fail-closed safety controller. A
read error can reuse the last available action, so software disconnect handling
does not replace `Ctrl+C`, the e-stop, or removal of motor power.

Start with the five-second low-speed test. The faster profiles are not a
substitute for a successful low-speed test on the new machine.

## 1. Tested source and package versions

| Component | Tested version / commit |
| --- | --- |
| Python | `3.10.20` |
| uv | `0.11.28` |
| Seeed LeRobot fork | `0f392484458cb5ebca0310c0c4c47390a31c80ed` (`lerobot==0.4.4`) |
| B601 follower plugin | `198bbf2b37a0c97be506ba778829ac915c6a2f6e` |
| reBot 102 leader plugin | `4fc47f38f8ef8ee88cc286241e3e3bc1c88a996d` |
| reBot SDK | `0324b734c86866ea23d6ea4c097833739329d6d3` |
| motorbridge | `0.4.9` |
| motorbridge-smart-servo | `0.0.4` |
| pyserial | `3.5` |
| torch | `2.7.1` |
| torchvision | `0.22.1` |
| python-can | `4.6.1` |
| opencv-python-headless | `4.12.0.88` |
| rerun-sdk | `0.26.2` |
| av | `15.1.0` |
| datasets | `4.1.1` |
| draccus | `0.10.0` |
| numpy | `2.2.6` |

## 2. Host prerequisites

### Ubuntu 22.04

```bash
sudo apt update
sudo apt install -y git git-lfs curl ffmpeg lsof build-essential python3-dev libgl1 libglib2.0-0
git lfs install
sudo usermod -aG dialout "$USER"
```

Log out and back in after adding the user to `dialout`. Avoid permanent
`chmod 666` rules for serial devices.

### macOS on Apple Silicon

```bash
xcode-select -p || xcode-select --install
brew install git-lfs ffmpeg
git lfs install
```

If Homebrew is not installed, install Git LFS using another trusted package
manager before cloning the Seeed LeRobot fork. Terminal also needs Camera
permission before OpenCV can use USB cameras.

### Install the tested uv release

```bash
curl -LsSf https://astral.sh/uv/0.11.28/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv python install 3.10.20
```

## 3. Create the isolated runtime and clone exact sources

Choose one stable location. Do not put this runtime in a temporary directory.

```bash
export REBOT_ROOT="${HOME}/rebot-runtime"
export REBOT_SRC="${REBOT_ROOT}/src"
export REBOT_VENV="${REBOT_ROOT}/.venv"
export REBOT_STATE="${REBOT_ROOT}/state"
export HF_HOME="${REBOT_STATE}/huggingface"
export HF_LEROBOT_HOME="${REBOT_STATE}/lerobot"
export UV_CACHE_DIR="${REBOT_STATE}/uv-cache"
export TORCH_HOME="${REBOT_STATE}/torch"
mkdir -p "$REBOT_SRC" "$HF_HOME" "$HF_LEROBOT_HOME" "$UV_CACHE_DIR" "$TORCH_HOME"
```

Clone and detach at the tested commits:

```bash
git clone https://github.com/vectorBH6/reBotArm_control_py.git "$REBOT_SRC/reBotArm_control_py"
git clone https://github.com/Seeed-Projects/lerobot.git "$REBOT_SRC/lerobot"
git clone https://github.com/Seeed-Projects/lerobot-robot-seeed-b601.git "$REBOT_SRC/lerobot-robot-seeed-b601"
git clone https://github.com/Seeed-Projects/lerobot-teleoperator-rebot-arm-102.git "$REBOT_SRC/lerobot-teleoperator-rebot-arm-102"

git -C "$REBOT_SRC/reBotArm_control_py" checkout --detach 0324b734c86866ea23d6ea4c097833739329d6d3
git -C "$REBOT_SRC/lerobot" checkout --detach 0f392484458cb5ebca0310c0c4c47390a31c80ed
git -C "$REBOT_SRC/lerobot-robot-seeed-b601" checkout --detach 198bbf2b37a0c97be506ba778829ac915c6a2f6e
git -C "$REBOT_SRC/lerobot-teleoperator-rebot-arm-102" checkout --detach 4fc47f38f8ef8ee88cc286241e3e3bc1c88a996d
```

Confirm the captured source state:

```bash
git -C "$REBOT_SRC/reBotArm_control_py" rev-parse HEAD
git -C "$REBOT_SRC/lerobot" rev-parse HEAD
git -C "$REBOT_SRC/lerobot-robot-seeed-b601" rev-parse HEAD
git -C "$REBOT_SRC/lerobot-teleoperator-rebot-arm-102" rev-parse HEAD

test "$(git -C "$REBOT_SRC/reBotArm_control_py" rev-parse HEAD)" = 0324b734c86866ea23d6ea4c097833739329d6d3
test "$(git -C "$REBOT_SRC/lerobot" rev-parse HEAD)" = 0f392484458cb5ebca0310c0c4c47390a31c80ed
test "$(git -C "$REBOT_SRC/lerobot-robot-seeed-b601" rev-parse HEAD)" = 198bbf2b37a0c97be506ba778829ac915c6a2f6e
test "$(git -C "$REBOT_SRC/lerobot-teleoperator-rebot-arm-102" rev-parse HEAD)" = 4fc47f38f8ef8ee88cc286241e3e3bc1c88a996d
```

## 4. Apply the tested command-limit and latency patches

These patches are required to match the tested runtime. The first makes
`--robot.pos_vel_velocity` effective instead of silently sending a hard-coded
`32 rad/s` POS_VEL cap. The second rate-limits clamp warnings and removes
per-frame terminal output from the control loop.

```bash
git -C "$REBOT_SRC/lerobot-robot-seeed-b601" apply <<'PATCH'
diff --git a/lerobot_robot_seeed_b601/seeed_b601_follower.py b/lerobot_robot_seeed_b601/seeed_b601_follower.py
--- a/lerobot_robot_seeed_b601/seeed_b601_follower.py
+++ b/lerobot_robot_seeed_b601/seeed_b601_follower.py
@@ -362,4 +362,4 @@ class SeeedB601FollowerBase(Robot):
                         )
                     else:
-                        motor.send_pos_vel(pos_rad, 32)
+                        motor.send_pos_vel(pos_rad, vel_rad)
                         logger.debug(f"Sent POS_VEL command to {motor_name}: target={pos_rad:.2f},pos={position_degrees:.2f}°, vel={vel_deg_s:.2f}°/s")
PATCH
```

```bash
git -C "$REBOT_SRC/lerobot" apply <<'PATCH'
diff --git a/src/lerobot/robots/utils.py b/src/lerobot/robots/utils.py
--- a/src/lerobot/robots/utils.py
+++ b/src/lerobot/robots/utils.py
@@ -15,3 +15,4 @@
 import logging
+import time
 from pprint import pformat
 from typing import cast
@@ -22 +23,5 @@ from .config import RobotConfig
 from .robot import Robot
+
+
+_LAST_CLAMP_WARNING_AT = 0.0
+_CLAMP_WARNING_INTERVAL_S = 1.0
@@ -112 +117,8 @@ def ensure_safe_goal_position(
     if warnings_dict:
+        global _LAST_CLAMP_WARNING_AT
+        now = time.monotonic()
+        should_log = now - _LAST_CLAMP_WARNING_AT >= _CLAMP_WARNING_INTERVAL_S
+        if should_log:
+            _LAST_CLAMP_WARNING_AT = now
+
+    if warnings_dict and should_log:
diff --git a/src/lerobot/scripts/lerobot_teleoperate.py b/src/lerobot/scripts/lerobot_teleoperate.py
--- a/src/lerobot/scripts/lerobot_teleoperate.py
+++ b/src/lerobot/scripts/lerobot_teleoperate.py
@@ -154 +154,3 @@ def teleop_loop(
     start = time.perf_counter()
+    status_started_at = start
+    status_loop_count = 0
@@ -197,2 +199,8 @@ def teleop_loop(
-        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
-        move_cursor_up(1)
+        status_loop_count += 1
+        status_now = time.perf_counter()
+        status_elapsed = status_now - status_started_at
+        if not display_data and status_elapsed >= 1.0:
+            actual_hz = status_loop_count / status_elapsed
+            print(f"Teleop running: {actual_hz:.1f} Hz (last loop {loop_s * 1e3:.1f}ms)")
+            status_started_at = status_now
+            status_loop_count = 0
PATCH
```

Validate both patches before installing:

```bash
git -C "$REBOT_SRC/lerobot" diff --check
git -C "$REBOT_SRC/lerobot-robot-seeed-b601" diff --check
```

## 5. Build one dedicated robot-control virtual environment

This environment is only for the SDK, LeRobot, and the two hardware plugins.
The DeskPartner harness still uses the repository's own `.venv`.

```bash
uv venv --python 3.10.20 "$REBOT_VENV"
source "$REBOT_VENV/bin/activate"

uv pip install -e "$REBOT_SRC/reBotArm_control_py"
uv pip install -e "$REBOT_SRC/lerobot"
uv pip install -e "$REBOT_SRC/lerobot-robot-seeed-b601"
uv pip install -e "$REBOT_SRC/lerobot-teleoperator-rebot-arm-102"

uv pip install \
  motorbridge==0.4.9 \
  motorbridge-smart-servo==0.0.4 \
  pyserial==3.5 \
  python-can==4.6.1 \
  torch==2.7.1 \
  torchvision==0.22.1 \
  opencv-python-headless==4.12.0.88 \
  rerun-sdk==0.26.2 \
  av==15.1.0 \
  datasets==4.1.1 \
  draccus==0.10.0 \
  numpy==2.2.6
```

Check imports and command entry points:

```bash
export PYTHONPATH="${REBOT_SRC}/lerobot/src:${REBOT_SRC}/lerobot-robot-seeed-b601:${REBOT_SRC}/lerobot-teleoperator-rebot-arm-102${PYTHONPATH:+:${PYTHONPATH}}"
python --version
python -c 'import lerobot, motorbridge, serial, torch, cv2, rerun; print("lerobot", lerobot.__version__); print("torch", torch.__version__); print("mps", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()); print("opencv", cv2.__version__)'
command -v motorbridge-cli
command -v lerobot-calibrate
command -v lerobot-teleoperate
command -v lerobot-record
command -v lerobot-find-cameras
command -v lerobot-train
uv pip freeze > "$REBOT_STATE/resolved-requirements.txt"
```

Every new terminal must restore the same environment:

```bash
export REBOT_ROOT="${HOME}/rebot-runtime"
export REBOT_SRC="${REBOT_ROOT}/src"
export REBOT_VENV="${REBOT_ROOT}/.venv"
export REBOT_STATE="${REBOT_ROOT}/state"
export HF_HOME="${REBOT_STATE}/huggingface"
export HF_LEROBOT_HOME="${REBOT_STATE}/lerobot"
export UV_CACHE_DIR="${REBOT_STATE}/uv-cache"
export TORCH_HOME="${REBOT_STATE}/torch"
export PYTHONPATH="${REBOT_SRC}/lerobot/src:${REBOT_SRC}/lerobot-robot-seeed-b601:${REBOT_SRC}/lerobot-teleoperator-rebot-arm-102${PYTHONPATH:+:${PYTHONPATH}}"
source "$REBOT_VENV/bin/activate"
```

## 6. Discover the two serial devices by USB identity

Do not copy `/dev/ttyACM0`, `/dev/ttyUSB0`, or a macOS `/dev/cu.*` name from
another computer. Resolve exactly one follower and one leader every time the
USB topology changes.

The tested identities are:

| Device | USB VID:PID | Baud |
| --- | --- | --- |
| B601 follower Damiao bridge | `2e88:4603` | `921600` |
| reBot 102 leader CH340/UC-01 | `1a86:7523` | `1000000` |

```bash
export FOLLOWER_PORT="$(python -c 'from serial.tools import list_ports; h=[p.device for p in list_ports.comports() if p.vid==0x2E88 and p.pid==0x4603]; assert len(h)==1, h; print(h[0])')"
export LEADER_PORT="$(python -c 'from serial.tools import list_ports; h=[p.device for p in list_ports.comports() if p.vid==0x1A86 and p.pid==0x7523]; assert len(h)==1, h; print(h[0])')"
printf 'follower=%s\nleader=%s\n' "$FOLLOWER_PORT" "$LEADER_PORT"
```

The assertions intentionally fail if a device is absent or ambiguous.
The CH340 VID/PID is not unique to reBot hardware. If another CH340 device is
connected, disconnect it or extend the selector with the expected serial number
or USB location; never choose the first match silently. On Linux, prefer a
stable `/dev/serial/by-id/` alias after confirming the underlying VID/PID.

Read-only follower scan:

```bash
motorbridge-cli scan \
  --vendor damiao \
  --transport dm-serial \
  --serial-port "$FOLLOWER_PORT" \
  --serial-baud 921600
```

Read-only leader ping and position read:

```bash
python -c 'from motorbridge_smart_servo import FashionStarServo
import os
bus=FashionStarServo(os.environ["LEADER_PORT"],1_000_000)
try:
    online={i:bus.ping(i) for i in range(7)}
    state=bus.sync_monitor(list(range(7)))
    print("online", online)
    print("angles_deg", {i:(None if state[i] is None else round(state[i].angle_deg,2)) for i in range(7)})
    assert all(online.values()), online
finally:
    bus.close()'
```

Expected gate: seven follower motors found and seven leader servos online.

## 7. Calibrate or migrate calibration

For a new arm pair or an intentional zero reset, place both arms in the
documented folded zero pose, close both grippers, clear the workspace, and
calibrate using stable IDs. **These commands write the follower hardware zero
and leader servo origins as well as calibration files; do not run them from an
arbitrary pose.**

Do not use the SDK's `example/2_zero_and_read.py` as a portable shortcut. At
the pinned revision it reads a hardware YAML containing `/dev/ttyACM0`, and it
immediately writes the follower zero before entering free-drive mode.

```bash
lerobot-calibrate \
  --robot.type=seeed_b601_dm_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=follower1 \
  --robot.can_adapter=damiao
```

```bash
lerobot-calibrate \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=rebot_arm_102_leader
```

Verify that both calibration files exist:

```bash
test -f "$HF_LEROBOT_HOME/calibration/robots/seeed_b601_dm_follower/follower1.json"
test -f "$HF_LEROBOT_HOME/calibration/teleoperators/rebot_arm_102_leader/rebot_arm_102_leader.json"
```

Calibration belongs to the physical arm pair. When moving the **same** pair to
a second computer, skip the two calibration commands and transfer the complete
`$HF_LEROBOT_HOME/calibration/` directory after recording and checking a
SHA-256 hash. If either arm changes, recalibrate instead. Never copy a different
arm's hardware zero and never commit calibration JSON to this repository.

## 8. Five-second low-speed teleoperation gate

Keep a hand on the e-stop. This is the first command in this file that sends
motor targets.

```bash
lerobot-teleoperate \
  --robot.type=seeed_b601_dm_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=follower1 \
  --robot.can_adapter=damiao \
  --robot.max_relative_target=0.5 \
  --robot.pos_vel_velocity='[5,5,5,5,5,5,5]' \
  --robot.force_pos_torque_ration=0.05 \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=rebot_arm_102_leader \
  --fps=10 \
  --teleop_time_s=5
```

Stop immediately if a joint moves in the wrong direction, jumps, oscillates,
or does not stop when the leader stops.

## 9. Continuous and faster teleoperation profiles

Define the command once:

```bash
rebot_teleop() {
  local velocity="$1"
  local requested_fps="$2"
  local step_limit="$3"
  lerobot-teleoperate \
    --robot.type=seeed_b601_dm_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id=follower1 \
    --robot.can_adapter=damiao \
    --robot.max_relative_target="$step_limit" \
    --robot.pos_vel_velocity="$velocity" \
    --robot.force_pos_torque_ration=0.05 \
    --teleop.type=rebot_arm_102_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id=rebot_arm_102_leader \
    --fps="$requested_fps"
}
```

Safe continuous profile:

```bash
rebot_teleop '[15,15,15,15,15,15,15]' 15 1.0
```

The following `150°/s`, 30 Hz-requested, `5°/cycle` profile was the fastest
profile actually exercised on the physical arm during the captured bring-up:

```bash
rebot_teleop '[150,150,150,150,150,150,150]' 30 5.0
```

Only after the five-second gate, safe continuous profile, and the physically
tested `150°/s` profile all pass should the following candidates be bench
tested. They passed CLI parsing and configuration validation, but were **not**
physically motion-tested during the captured session:

```bash
# Faster response test: arm joints 600°/s cap, gripper 180°/s.
rebot_teleop '[600,600,600,600,600,600,180]' 50 12.0

# Fast continuous profile.
rebot_teleop '[1200,1200,1200,1200,1200,1200,240]' 60 20.0

# Upstream-equivalent 32 rad/s command cap for six arm joints.
rebot_teleop '[1833,1833,1833,1833,1833,1833,300]' 60 30.0
```

The last number is a command cap, not the mechanical speed. `32 rad/s` is
already above the joints' physical no-load limits, so increasing it further
does not make the arm faster. Requested `--fps` is also not proof of achieved
loop rate; use the once-per-second `Teleop running: ... Hz` output.

Use `Ctrl+C` for a clean stop. Before restarting, confirm both serial devices
are free:

```bash
lsof "$FOLLOWER_PORT" "$LEADER_PORT"
```

No output means no process currently owns either port.

## 10. Match DeskPartner configuration

On Ubuntu, write the discovered ports into `config/arm.yaml` and
`config/recording.yaml`. Also set `config/arm.yaml`'s `sdk.repo` to the absolute
path of `$REBOT_SRC/reBotArm_control_py`. Camera indexes belong in both
`config/recording.yaml` and `config/cameras.yaml` after camera discovery.

Keep the single-arm types and IDs exactly as follows:

```yaml
follower:
  type: seeed_b601_dm_follower
  port: /dev/ttyACM0
  id: follower1
  can_adapter: damiao
leader:
  type: rebot_arm_102_leader
  port: /dev/ttyUSB0
  id: rebot_arm_102_leader
```

Replace the example ports with the values discovered on that host. Do not add
left/right or bimanual channels.

## 11. Dual-camera preview and recording

Find the camera indexes after every reconnect:

```bash
lerobot-find-cameras opencv \
  --output-dir "$REBOT_STATE/camera-probe" \
  --record-time-s 6
export CAM_FRONT=0
export CAM_SIDE=1
```

`front` is the locked overhead camera and `side` is the locked 45-degree
camera. Before recording, confirm those names have not been swapped.
LeRobot reads the latest frame from each camera asynchronously; this is not
hardware stereo synchronization. Reject stale or visibly mismatched feeds
before collecting a production dataset.
Both cameras should sustain the same measured frame rate. If either cannot hold
30 fps, configure both recording views and `--dataset.fps` to 15 instead of
letting one stream silently lag.

Preview both views while teleoperating:

```bash
lerobot-teleoperate \
  --robot.type=seeed_b601_dm_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=follower1 \
  --robot.can_adapter=damiao \
  --robot.max_relative_target=1.0 \
  --robot.pos_vel_velocity='[15,15,15,15,15,15,15]' \
  --robot.force_pos_torque_ration=0.05 \
  --robot.cameras="{front: {type: opencv, index_or_path: ${CAM_FRONT}, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, side: {type: opencv, index_or_path: ${CAM_SIDE}, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}}" \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=rebot_arm_102_leader \
  --fps=30 \
  --display_data=true
```

Use a local dataset path and keep uploads disabled:

```bash
export DATASET_ROOT="$REBOT_ROOT/data/crumpled_paper_v1"
mkdir -p "$DATASET_ROOT"

lerobot-record \
  --robot.type=seeed_b601_dm_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=follower1 \
  --robot.can_adapter=damiao \
  --robot.max_relative_target=5.0 \
  --robot.pos_vel_velocity='[150,150,150,150,150,150,150]' \
  --robot.force_pos_torque_ration=0.05 \
  --robot.cameras="{front: {type: opencv, index_or_path: ${CAM_FRONT}, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, side: {type: opencv, index_or_path: ${CAM_SIDE}, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}}" \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=rebot_arm_102_leader \
  --display_data=true \
  --dataset.repo_id=local/deskpartner_crumpled_paper_v1 \
  --dataset.root="$DATASET_ROOT" \
  --dataset.fps=30 \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=20 \
  --dataset.single_task='Pick crumpled paper and drop in trash' \
  --dataset.push_to_hub=false
```

Five episodes are only the format/synchronization gate. Once they pass, start a
fresh production dataset and collect at least 50 clean successful episodes.
Failed, collided, or heavily corrected episodes must be re-recorded.

When operating from the DeskPartner checkout, use its fail-loud wrappers:

```bash
python -m p4_data_collection.check_camera_lock --save-ref
```

Use the raw `lerobot-record` command above for recording. At the pinned
DeskPartner revision, `p4_data_collection/config_loader.py` over-escapes the
camera `fourcc` value when it launches a subprocess, so the record/batch
wrappers must not be treated as reproducible until that separate issue is
fixed. After recording, `python -m p4_data_collection.verify_episode_format`
remains the required schema check for the **single throwaway episode**. At the
pinned DeskPartner revision its `max_frames=3000` check treats the total frame
count as one episode, so it can falsely reject a valid 50-episode dataset; do
not use that result as the final batch verdict until the verifier is fixed.

The camera keys, resolution, frame rate, physical poses, and task string must
remain unchanged between recording, training, and policy execution.

## 12. Optional local ACT baseline

This is a local imitation-learning baseline, separate from DeskPartner's
MolmoAct 2 bake-off:

```bash
export MODEL_ROOT="$REBOT_ROOT/models/act_crumpled_paper_v1"
export POLICY_DEVICE="$(python -c 'import torch; mps=getattr(torch.backends,"mps",None); print("cuda" if torch.cuda.is_available() else "mps" if mps and mps.is_available() else "cpu")')"
printf 'policy_device=%s\n' "$POLICY_DEVICE"

lerobot-train \
  --dataset.repo_id=local/deskpartner_crumpled_paper_v1 \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=act \
  --policy.device="$POLICY_DEVICE" \
  --policy.push_to_hub=false \
  --output_dir="$MODEL_ROOT" \
  --job_name=act_crumpled_paper_v1 \
  --batch_size=8 \
  --num_workers=0 \
  --steps=2000 \
  --save_freq=1000 \
  --wandb.enable=false
```

The selector prefers CUDA, then MPS, then CPU. Availability can change with the
host, Python build, and shell environment. CPU is a slow fallback. The captured
Mac reported MPS unavailable during the final audit, so this training command
was configuration-validated but not completed end to end. The `2000`-step run
is only a smoke test; start a separate 50k-100k run after it successfully loads
both camera streams and the seven-joint state/action schema.

Do not let a newly trained policy drive the arm immediately. Validate its
outputs offline, then shadow mode, then an empty workspace at 10-20% speed.
In LeRobot `0.4.4`, supplying a policy and a teleoperator does not provide live
Leader takeover: policy actions win inside the episode. `Ctrl+C` and the
independent hardware e-stop/power cut remain the stop paths until an explicit
supervisor is implemented and tested.

## 13. New-machine acceptance checklist

A second computer is considered reproduced only when all of these pass:

1. The four repository hashes match section 1.
2. `git diff --check` passes after applying the two patches.
3. The isolated environment imports LeRobot, motorbridge, both plugins, OpenCV,
   torch, and rerun.
4. USB discovery returns exactly one follower and exactly one leader.
5. The read-only scan finds seven Damiao motors.
6. The read-only leader check reports seven online servos.
7. Both calibration files exist under the dedicated `HF_LEROBOT_HOME`.
8. The five-second `5°/s` test moves every joint in the correct direction and
   stops cleanly.
9. The safe continuous profile runs without port contention or disconnects.
10. Both camera feeds are correctly named, fixed, current, and visible.
11. One throwaway episode passes `verify_episode_format` with no phantom second
    arm channels.

## 14. Common failures

- **No popup:** these are terminal commands. A GUI appears only when camera
  visualization is enabled; macOS may show a one-time Camera permission prompt.
- **`git-lfs filter-process: git-lfs: command not found`:** install Git LFS,
  run `git lfs install`, then re-clone the Seeed LeRobot fork.
- **Follower missing:** check 24 V follower power, the Damiao bridge, and USB
  identity `2e88:4603`.
- **Leader missing:** check 12 V leader power, UC-01 red LED, the first 3-pin bus
  cable, and USB identity `1a86:7523`.
- **Port permission denied on Ubuntu:** confirm the user is in `dialout`, then
  log out and back in.
- **Camera indexes swapped:** rerun `lerobot-find-cameras opencv`; never assume
  that index `0` or `1` survived a reconnect.
- **A port is busy:** stop the previous LeRobot process cleanly and use `lsof`
  before restarting. Never run teleoperation and autonomous control against the
  follower at the same time.
- **Follower moves opposite to the leader:** stop immediately. Recheck physical
  zero/calibration and the `joint_directions` in `config/arm.yaml`; do not keep
  increasing speed to mask a mapping error.
