#!/bin/zsh

set -euo pipefail
KIT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$KIT_ROOT/env.sh"

print -- "ReBot dual external-camera check"
print -- "Logitech overhead: index $CAM_FRONT"
print -- "Innomaker wrist on ReBot claw: index $CAM_SIDE"
print -- "excluded Mac webcam: index $CAM_EXCLUDED_SCREEN"
print -- ""

set +e
"$PYTHON_BIN" "$KIT_ROOT/dual_camera_check.py" \
  --front "$CAM_FRONT" \
  --side "$CAM_SIDE" \
  --excluded-screen "$CAM_EXCLUDED_SCREEN" \
  --fps "$CAMERA_FPS" \
  --minimum-fps "$MINIMUM_CAMERA_FPS" \
  --duration 8 \
  --output "$KIT_CAMERA_ROOT"
status=$?
set -e

print -- ""
if [[ $status -eq 0 ]]; then
  print -- "PASS: both external cameras are bright and meet the frame-rate gate."
else
  print -u2 -- "FAIL: inspect the report and saved frames before recording."
  print -u2 -- "Check the Logitech overhead framing and the Innomaker claw-camera lens, cable, and exposure."
fi
print -- "Saved frames: $KIT_CAMERA_ROOT"
read -r "?Press Return to close this window..."
exit $status
