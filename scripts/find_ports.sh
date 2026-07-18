#!/usr/bin/env bash
# Wrapper around LeRobot port discovery. Update config/arm.yaml after.
set -euo pipefail
echo "=== lerobot-find-port ==="
echo "Unplug/replug MotorsBus when prompted."
lerobot-find-port
echo
echo "=== serial devices ==="
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "(none)"
echo
echo "Write ports into DeskPartner/config/arm.yaml and ~/reBotArm_control_py/config/arm.yaml"
