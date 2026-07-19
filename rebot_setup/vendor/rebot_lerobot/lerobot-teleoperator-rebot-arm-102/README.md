# reBot Arm 102 Teleoperator for reBot B601

[中文版说明](./README.zh-CN.md)

This repository provides a LeRobot teleoperator integration for the reBot Arm 102 leader arm, intended to be paired with the Seeed reBot B601 follower arm.

The implementation is intentionally opinionated:

- joint names are aligned to reBot B601
- leader-side joint limits are taken directly from config
- follower-side direction mapping is configured in the follower integration
- each startup calibration sets the current servo origin to zero

## Supported Setup

- Leader: reBot Arm 102
- Follower: Seeed reBot B601
- Communication: UART for reBot Arm 102, CAN or Damiao serial bridge for B601

## Installation

Install LeRobot first, then install this package in editable mode:

```bash
cd lerobot-teleoperator-rebot-arm-102
pip install -e .
```

This package registers one teleoperator type:

- `rebot_arm_102_leader`

## Default Mapping

- `shoulder_pan` -> servo ID `0`
- `shoulder_lift` -> servo ID `1`
- `elbow_flex` -> servo ID `2`
- `wrist_flex` -> servo ID `3`
- `wrist_yaw` -> servo ID `4`
- `wrist_roll` -> servo ID `5`
- `gripper` -> servo ID `6`

Leader-side joint limits are defined in `lerobot_teleoperator_rebot_arm_102/config_rebot_arm_102_leader.py`.
Follower-side direction mapping is defined in the B601 follower configuration.

## Usage

Standard teleoperation:

```bash
lerobot-teleoperate \
  --teleop.type=rebot_arm_102_leader \
  --teleop.id=rebot_arm_102_leader \
  --teleop.port=/dev/ttyUSB0 \
  --robot.type=seeed_b601_dm_follower \
    --robot.id=follower1 \
    --robot.port=/dev/ttyACM4 \
    --robot.can_adapter=damiao
```

## Example Scripts

### `read_raw_angles.py`

Purpose:

- read raw reBot Arm 102 servo angles directly from the Fashion Star SDK
- verify servo ID to joint mapping
- check whether a given joint is actually changing at the hardware level

Usage:

```bash
cd lerobot-teleoperator-rebot-arm-102
python examples/read_raw_angles.py --port /dev/ttyUSB0
```

What to look for:

- move only one joint at a time
- confirm the expected joint column changes
- if raw values change but teleop behavior is wrong, the problem is usually the leader range config or the follower direction config, not the SDK readout

### `read_leader_follower_compare.py`

Purpose:

- read reBot Arm 102 leader output and B601 follower observation side by side
- compare `leader`, `follower`, and `delta` per joint
- debug direction mismatches safely without sending follower commands

Prerequisite:

- need a reBot B601 arm setup
- install the python integration for reBot B601, https://github.com/Seeed-Projects/lerobot-robot-seeed-b601

Behavior:

- after connecting the B601 follower, it disables torque so the arm can be moved by hand
- it does not send action commands to the follower

Usage with a Damiao follower on `can0`:

```bash
cd lerobot-teleoperator-rebot-arm-102
python examples/read_leader_follower_compare.py --leader-port /dev/ttyUSB0 --follower-port can0 --follower-type dm
```

Usage with a Damiao follower through a Damiao serial bridge:

```bash
cd lerobot-teleoperator-rebot-arm-102
python examples/read_leader_follower_compare.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyACM0 --follower-type dm --follower-can-adapter damiao
```

What to look for:

- move one joint on the leader
- move the same joint by hand on the follower if needed
- compare whether `mapped` and `follower` change in the same sign direction
- if one side increases while the other decreases, update that joint in the follower `joint_directions` config

## Notes

- Under the current implementation, startup calibration resets each reBot Arm 102 servo origin to the current pose.
- `joint_ranges` are taken from config, not from calibration data.
- If a joint appears stuck near one limit, check `joint_ranges` first.
