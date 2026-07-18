#!/usr/bin/env bash
# Single-arm teleop with overhead + 45° preview (MolmoAct 2 dataset views).
set -euo pipefail

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyUSB0}"
CAM_OVERHEAD="${CAM_OVERHEAD:-0}"
CAM_SIDE="${CAM_SIDE:-1}"

lerobot-teleoperate \
  --robot.type=seeed_b601_dm_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id=follower1 \
  --robot.can_adapter=damiao \
  --robot.cameras="{ front: {type: opencv, index_or_path: ${CAM_OVERHEAD}, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, side: {type: opencv, index_or_path: ${CAM_SIDE}, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}}" \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id=rebot_arm_102_leader \
  --display_data=true
