# reBot teleop setup (macOS) — portable snapshot

Snapshot of the reBot **B601-DM** control + teleop environment, captured while switching
computers. Vendored code lives in `vendor/` (venvs/`.git` excluded — rebuild with `setup.sh`).

> Hardware: reBot **B601-DM** follower (Damiao CAN motors, USB2CAN "dm-serial" bridge) +
> reBot **102** leader (FashionStar UART servos, CH340 adapter). Off the SO-101 path.

## New-machine setup

Prereq: [`uv`](https://astral.sh/uv) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
cd rebot_setup
./setup.sh          # rebuilds both venvs from vendored code (offline-capable)
```

This creates:
- `vendor/reBotArm_control_py/.venv` — reBot SDK (direct follower control, Pinocchio FK/IK).
- `vendor/rebot_lerobot/.venv` — LeRobot + rebot plugins + motorbridge (leader→follower teleop).

## ⚠️ Ports change per machine

Serial device names are **not portable** — re-detect them on the new Mac:

```bash
ls /dev/cu.* | grep -iE 'usbmodem|usbserial'   # or: lerobot-find-port
```

- **Follower** (Damiao USB2CAN, HDSC/CDC): was `/dev/cu.usbmodem00000000050C1`.
- **Leader** (CH340 UART, VID 0x1a86): was `/dev/cu.usbserial-10`.

Update the follower port in `vendor/reBotArm_control_py/config/rebotarm_dm.yaml` (`channel:`)
and pass the right `--*.port=` on the LeRobot commands below.

## A) Direct follower control (SDK — no leader needed)

```bash
cd vendor/reBotArm_control_py
uv run motorbridge-cli scan --vendor damiao --transport dm-serial \
  --serial-port <FOLLOWER_PORT> --serial-baud 921600      # read-only: confirm 7 motors
uv run python example/2_zero_and_read.py rebotarm_dm.yaml  # zero + free-drive (hand-move, prints angles)
uv run python example/9_gravity_compensation.py            # weightless "floating" mode
```

## B) Leader → follower teleop (LeRobot — needs BOTH arms plugged in)

Device types (this Seeed-Projects build): `seeed_b601_dm_follower` / `rebot_arm_102_leader`.

```bash
cd vendor/rebot_lerobot
V=.venv/bin

# 1) calibrate leader (move to zero pose, gripper closed, press Enter)
$V/lerobot-calibrate --teleop.type=rebot_arm_102_leader \
  --teleop.port=<LEADER_PORT> --teleop.id=rebot_arm_102_leader

# 2) teleop — follower mirrors the leader (follower auto-calibrates; place at zero first)
$V/lerobot-teleoperate \
  --robot.type=seeed_b601_dm_follower --robot.port=<FOLLOWER_PORT> \
  --robot.id=follower1 --robot.can_adapter=damiao \
  --teleop.type=rebot_arm_102_leader --teleop.port=<LEADER_PORT> \
  --teleop.id=rebot_arm_102_leader
```

**Safety:** base C-clamped, hand on e-stop, follower at zero/rest pose — it mirrors the
leader live the instant teleop starts.

## C) Recording demos (adds cameras)

Recording needs **both arms + both cameras on at once** (overhead + 45°, locked, MJPG
640×480). With only 2 USB-C ports you need a **powered** hub. Add to the teleop command:

```
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, \
                     side:  {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}}" \
  --display_data=true
```
…and swap `lerobot-teleoperate` → `lerobot-record` with `--dataset.*` args (see the workshop deck).

## Notes
- RobStride (`00-arm-rs_asm-v3`) meshes were omitted (not used by the DM arm); re-clone
  `github.com/vectorBH6/reBotArm_control_py` if you switch to the RS variant.
- Upstream device names may differ on current HF LeRobot (`rebot_b601_follower` /
  `rebot_102_leader`); this vendored Seeed build uses the `seeed_b601_dm_follower` names above.
