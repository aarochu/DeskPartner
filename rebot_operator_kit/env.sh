#!/bin/zsh

if [[ -z "${RUNTIME_ROOT:-}" ]]; then
  for candidate in \
    "$KIT_ROOT/../rebot_setup/vendor/rebot_lerobot" \
    "$KIT_ROOT/../rebot-run" \
    "$KIT_ROOT/../../rebot-run"; do
    if [[ -d "$candidate/lerobot" ]]; then
      export RUNTIME_ROOT="$candidate"
      break
    fi
  done
fi
export RUNTIME_ROOT="${RUNTIME_ROOT:-$KIT_ROOT/../rebot_setup/vendor/rebot_lerobot}"
if [[ -d "$RUNTIME_ROOT/reBotArm_control_py/.venv" ]]; then
  export VENV="${VENV:-$RUNTIME_ROOT/reBotArm_control_py/.venv}"
else
  export VENV="${VENV:-$RUNTIME_ROOT/.venv}"
fi
export PYTHON_BIN="$VENV/bin/python"
export MOTORBRIDGE_BIN="$VENV/bin/motorbridge-cli"
export CALIBRATE_BIN="$VENV/bin/lerobot-calibrate"
export TELEOPERATE_BIN="$VENV/bin/lerobot-teleoperate"
export RECORD_BIN="$VENV/bin/lerobot-record"
export TRAIN_BIN="$VENV/bin/lerobot-train"

export HF_LEROBOT_HOME="$RUNTIME_ROOT/lerobot-home"
if [[ -x "$PYTHON_BIN" ]]; then
  PYTHON_SITE_PACKAGES="$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
else
  PYTHON_SITE_PACKAGES="$VENV/lib/python3.11/site-packages"
fi
export PYTHONPATH="$RUNTIME_ROOT/lerobot/src:$RUNTIME_ROOT/lerobot-robot-seeed-b601:$RUNTIME_ROOT/lerobot-teleoperator-rebot-arm-102:$PYTHON_SITE_PACKAGES/rerun_sdk${PYTHONPATH:+:$PYTHONPATH}"

export CAM_FRONT=0
export CAM_SIDE=1
export CAM_EXCLUDED_SCREEN=3
export CAMERA_FPS="${CAMERA_FPS:-30}"
export MINIMUM_CAMERA_FPS="${MINIMUM_CAMERA_FPS:-27}"

export KIT_STATE_ROOT="$KIT_ROOT/.state"
export KIT_DATA_ROOT="$KIT_ROOT/data"
export KIT_MODEL_ROOT="$KIT_ROOT/models"
export KIT_CAMERA_ROOT="$KIT_ROOT/camera-check"
export HF_HOME="$KIT_STATE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$KIT_STATE_ROOT" "$KIT_DATA_ROOT" "$KIT_MODEL_ROOT" "$KIT_CAMERA_ROOT" \
  "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

detect_ports() {
  local assignments
  assignments="$($PYTHON_BIN - <<'PY'
from serial.tools import list_ports
import shlex

ports = list(list_ports.comports())

def unique(vid, pid, label):
    hits = [p.device for p in ports if p.vid == vid and p.pid == pid]
    if len(hits) != 1:
        raise SystemExit(f"{label}: expected exactly one port, found {hits}")
    return hits[0]

print("FOLLOWER_PORT=" + shlex.quote(unique(0x2E88, 0x4603, "follower")))
print("LEADER_PORT=" + shlex.quote(unique(0x1A86, 0x7523, "leader")))
PY
)"
  eval "$assignments"
  export FOLLOWER_PORT LEADER_PORT
}

require_calibration() {
  local follower_cal="$HF_LEROBOT_HOME/calibration/robots/seeed_b601_dm_follower/follower1.json"
  local leader_cal="$HF_LEROBOT_HOME/calibration/teleoperators/rebot_arm_102_leader/rebot_arm_102_leader.json"

  [[ -f "$follower_cal" ]] || { print -u2 -- "Missing follower calibration: $follower_cal"; return 1; }
  [[ -f "$leader_cal" ]] || { print -u2 -- "Missing leader calibration: $leader_cal"; return 1; }
}

assert_ports_free() {
  local owners
  owners="$(lsof "$FOLLOWER_PORT" "$LEADER_PORT" 2>/dev/null || true)"
  if [[ -n "$owners" ]]; then
    print -u2 -- "A serial port is already in use:"
    print -u2 -- "$owners"
    return 1
  fi
}
