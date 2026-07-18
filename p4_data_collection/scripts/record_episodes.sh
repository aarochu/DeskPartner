#!/usr/bin/env bash
# Record MolmoAct 2 demos: single-arm + locked dual-cam (front=overhead, side=45°).
set -euo pipefail

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyUSB0}"
CAM_OVERHEAD="${CAM_OVERHEAD:-0}"
CAM_SIDE="${CAM_SIDE:-1}"
REPO_ID="${REPO_ID:-deskpartner/crumpled_paper_molmoact2}"
NUM_EPISODES="${NUM_EPISODES:-50}"
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"
RESET_TIME_S="${RESET_TIME_S:-20}"

lerobot-record \
  --robot.type=seeed_b601_dm_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id=follower1 \
  --robot.can_adapter=damiao \
  --robot.cameras="{ front: {type: opencv, index_or_path: ${CAM_OVERHEAD}, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, side: {type: opencv, index_or_path: ${CAM_SIDE}, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}}" \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id=rebot_arm_102_leader \
  --display_data=true \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.single_task="Pick crumpled paper and drop in trash" \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}"

echo
echo "Next: python p4_data_collection/scripts/verify_single_arm_dataset.py --repo-id ${REPO_ID}"
