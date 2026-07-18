#!/bin/zsh

set -euo pipefail
KIT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$KIT_ROOT/env.sh"

print -u2 -- "This legacy recorder is disabled: it does not carry the locked training-profile digest"
print -u2 -- "and can save actions in the wrong ReBot coordinate frame."
print -u2 -- "Open http://127.0.0.1:8765/training and use Start collection instead."
exit 2

print_support_warning() {
  print -- ""
  print -- "IMPORTANT: LeRobot releases follower torque when recording disconnects."
  print -- "Support the follower before stopping, resetting, or cutting motor power."
}
trap print_support_warning EXIT

[[ -f "$KIT_STATE_ROOT/normal_teleop_passed" ]] || {
  print -u2 -- "Complete and PASS all three teleop gates in 03_teleop.command first."
  exit 2
}

detect_ports
require_calibration
assert_ports_free

"$PYTHON_BIN" "$KIT_ROOT/dual_camera_check.py" \
  --front "$CAM_FRONT" \
  --side "$CAM_SIDE" \
  --excluded-screen "$CAM_EXCLUDED_SCREEN" \
  --fps "$CAMERA_FPS" \
  --minimum-fps "$MINIMUM_CAMERA_FPS" \
  --duration 8 \
  --output "$KIT_CAMERA_ROOT"

DATASET_ID="local/deskpartner_crumpled_paper_smoke_v1"
DATASET_ROOT="$KIT_DATA_ROOT/crumpled_paper_smoke_v1"

if [[ -d "$DATASET_ROOT" ]] && [[ -n "$(find "$DATASET_ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  print -u2 -- "Dataset path is not empty: $DATASET_ROOT"
  print -u2 -- "Move it aside or choose a new version name; this script will not overwrite data."
  exit 2
fi

mkdir -p "$DATASET_ROOT"

CAMERAS='{front: {type: opencv, index_or_path: '"$CAM_FRONT"', width: 640, height: 480, fps: '"$CAMERA_FPS"', fourcc: "MJPG"}, side: {type: opencv, index_or_path: '"$CAM_SIDE"', width: 1280, height: 720, fps: '"$CAMERA_FPS"', fourcc: "MJPG"}}'

print -- ""
print -- "Five-episode recording smoke test"
print -- "Dataset: $DATASET_ROOT"
print -- "Task: Pick crumpled paper and drop in trash"
print -- "Logitech overhead=index $CAM_FRONT, Innomaker wrist=index $CAM_SIDE, Mac webcam index $CAM_EXCLUDED_SCREEN excluded"
print -- ""
print -- "Confirm the follower is clamped, the e-stop is reachable, both camera mounts are locked,"
print -- "the task zone is clear, and the operator can support the follower when recording stops."
print -- "Keep the leader still while typing RECORD; control begins immediately afterward."
read -r "answer?Type RECORD to start five 30-second episodes: "
[[ "$answer" == "RECORD" ]] || { print -u2 -- "Recording confirmation not received."; exit 1; }

"$RECORD_BIN" \
  --robot.type=seeed_b601_dm_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=follower1 \
  --robot.can_adapter=damiao \
  --robot.max_relative_target=5.0 \
  --robot.pos_vel_velocity='[150,150,150,150,150,150,150]' \
  --robot.force_pos_torque_ration=0.05 \
  --robot.cameras="$CAMERAS" \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=rebot_arm_102_leader \
  --display_data=true \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.fps="$CAMERA_FPS" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=20 \
  --dataset.single_task='Pick crumpled paper and drop in trash' \
  --dataset.push_to_hub=false \
  --dataset.vcodec=h264

touch "$KIT_STATE_ROOT/record_smoke_completed"
print -- "Recording finished. Run 05_validate_dataset.command next."
read -r "?Press Return to close this window..."
