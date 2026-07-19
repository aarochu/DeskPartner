#!/bin/zsh

set -u
KIT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$KIT_ROOT/env.sh"

detect_ports || exit 1
require_calibration || exit 1
assert_ports_free || exit 1

print_support_warning() {
  print -- ""
  print -- "IMPORTANT: LeRobot releases follower torque when it disconnects."
  print -- "Support the follower before stopping or cutting motor power."
}
trap print_support_warning EXIT

confirm_safety() {
  print -- ""
  print -- "Before motor commands:"
  print -- "- The B601 follower base is rigidly C-clamped and powered by 24 V."
  print -- "- The reBot 102 leader is powered by 12 V. Never connect 24 V to it."
  print -- "- Both arms start in corresponding stable poses; the follower is supported."
  print -- "- The one-meter workspace is empty and the e-stop/power cut is reachable."
  print -- "- No camera tripod, cable, person, or loose object is inside the sweep volume."
  print -- "- Keep the leader still while typing SAFE; motion control begins immediately afterward."
  local answer
  read -r "answer?Type SAFE to enable the selected motion test: "
  [[ "$answer" == "SAFE" ]] || { print -u2 -- "Safety confirmation not received."; return 1; }
}

run_teleop() {
  local velocity="$1"
  local fps="$2"
  local step_limit="$3"
  local duration="$4"
  local duration_args=()
  if [[ "$duration" != "continuous" ]]; then
    duration_args=("--teleop_time_s=$duration")
  fi

  "$TELEOPERATE_BIN" \
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
    --fps="$fps" \
    "${duration_args[@]}"
}

print -- "ReBot leader-to-follower teleoperation"
print -- "Follower: $FOLLOWER_PORT"
print -- "Leader:   $LEADER_PORT"
print -- ""
print -- "1) Five-second low-speed gate (5 deg/s, 10 Hz, 0.5 deg/cycle)"
print -- "2) Safe continuous gate (15 deg/s, 15 Hz, 1 deg/cycle; Ctrl+C to stop)"
print -- "3) Tested responsive profile (150 deg/s, 30 Hz, 5 deg/cycle; Ctrl+C to stop)"
print -- "4) Hand-response profile (170 deg/s, 60 Hz, 2.8 deg/cycle; Ctrl+C to stop)"
print -- "5) Continuous 500 deg/s profile (60 Hz, 8.3 deg/cycle; Ctrl+C to stop)"
print -- "6) Low-latency 500 deg/s profile (120 Hz, 4.2 deg/cycle; Ctrl+C to stop)"
print -- "q) Quit"

choice="${1:-}"
if [[ -z "$choice" ]]; then
  read -r "choice?Select: "
fi

case "$choice" in
  1|low)
    confirm_safety || exit 1
    print -- "For five seconds, move only one leader joint a few degrees; test the gripper last."
    run_teleop '[5,5,5,5,5,5,5]' 10 0.5 5
    teleop_status=$?
    if [[ $teleop_status -ne 0 ]]; then
      print -u2 -- "Low-speed teleop failed (exit $teleop_status); PASS was not recorded."
      exit $teleop_status
    fi
    read -r "answer?If every joint moved correctly and stopped cleanly, type PASS: "
    [[ "$answer" == "PASS" ]] && touch "$KIT_STATE_ROOT/low_speed_passed"
    ;;
  2|safe)
    [[ -f "$KIT_STATE_ROOT/low_speed_passed" ]] || { print -u2 -- "Run and pass option 1 first."; exit 2; }
    confirm_safety || exit 1
    print -- "Move one joint at a time. Press Ctrl+C while supporting the follower to stop."
    set +e
    run_teleop '[15,15,15,15,15,15,15]' 15 1.0 continuous
    teleop_status=$?
    set -e
    [[ $teleop_status -eq 0 || $teleop_status -eq 130 ]] || exit $teleop_status
    read -r "answer?If tracking and stopping were correct, type PASS: "
    [[ "$answer" == "PASS" ]] && touch "$KIT_STATE_ROOT/safe_continuous_passed"
    ;;
  3|normal)
    [[ -f "$KIT_STATE_ROOT/safe_continuous_passed" ]] || { print -u2 -- "Run and pass options 1 and 2 first."; exit 2; }
    confirm_safety || exit 1
    print -- "Begin with small motions. Press Ctrl+C while supporting the follower to stop."
    set +e
    run_teleop '[150,150,150,150,150,150,150]' 30 5.0 continuous
    teleop_status=$?
    set -e
    [[ $teleop_status -eq 0 || $teleop_status -eq 130 ]] || exit $teleop_status
    read -r "answer?If responsive tracking and stopping were correct, type PASS: "
    [[ "$answer" == "PASS" ]] && touch "$KIT_STATE_ROOT/normal_teleop_passed"
    ;;
  4|hand|turbo)
    [[ -f "$KIT_STATE_ROOT/normal_teleop_passed" ]] || { print -u2 -- "Run and pass options 1, 2, and 3 first."; exit 2; }
    confirm_safety || exit 1
    print -- "Hand-response ceiling: 170 deg/s at 60 Hz. Begin with small motions."
    print -- "A literal 5x/750 deg/s command is blocked because it exceeds the ReBot joint envelope."
    set +e
    run_teleop '[170,170,170,170,170,170,170]' 60 2.8 continuous
    teleop_status=$?
    set -e
    [[ $teleop_status -eq 0 || $teleop_status -eq 130 ]] || exit $teleop_status
    read -r "answer?If hand-response tracking and stopping were correct, type PASS: "
    [[ "$answer" == "PASS" ]] && touch "$KIT_STATE_ROOT/hand_response_passed"
    ;;
  5|500|experimental)
    [[ -f "$KIT_STATE_ROOT/hand_response_passed" ]] || { print -u2 -- "Run and pass option 4 first."; exit 2; }
    confirm_safety || exit 1
    print -- "Continuous 500 deg/s teleop at 60 Hz. Press Ctrl+C to stop."
    set +e
    run_teleop '[500,500,500,500,500,500,500]' 60 8.3 continuous
    teleop_status=$?
    set -e
    [[ $teleop_status -eq 0 || $teleop_status -eq 130 ]] || exit $teleop_status
    ;;
  6|120hz|low-latency)
    [[ -f "$KIT_STATE_ROOT/experimental_500_passed" ]] || { print -u2 -- "Run and pass option 5 first."; exit 2; }
    confirm_safety || exit 1
    print -- "Continuous 500 deg/s teleop at 120 Hz. Press Ctrl+C to stop."
    set +e
    run_teleop '[500,500,500,500,500,500,500]' 120 4.2 continuous
    teleop_status=$?
    set -e
    [[ $teleop_status -eq 0 || $teleop_status -eq 130 ]] || exit $teleop_status
    ;;
  q|Q|quit|exit)
    exit 0
    ;;
  *)
    print -u2 -- "Unknown selection: $choice"
    exit 2
    ;;
esac

print -- "Done."
read -r "?Press Return to close this window..."
