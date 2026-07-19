#!/usr/bin/env bash
# Rebuild the reBot teleop envs on a new machine from the vendored code.
# Requires: uv (https://astral.sh/uv). Works offline — venvs build from vendor/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

command -v uv >/dev/null 2>&1 || { echo "ERROR: install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

echo "==> [1/2] reBot SDK (direct follower control / kinematics)"
( cd "$HERE/vendor/reBotArm_control_py" && uv sync )

echo "==> [2/2] LeRobot rebot teleop stack (leader -> follower)"
cd "$HERE/vendor/rebot_lerobot"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  -e ./lerobot \
  -e ./lerobot-teleoperator-rebot-arm-102 \
  -e ./lerobot-robot-seeed-b601 \
  motorbridge

echo
echo "Done. Next: re-detect serial ports on this machine (ls /dev/cu.* | grep -iE 'usbmodem|usbserial'),"
echo "update config/rebotarm_dm.yaml channel:, then follow README.md (A direct / B teleop / C record)."
