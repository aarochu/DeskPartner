#!/bin/zsh

set -euo pipefail
KIT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$KIT_ROOT/env.sh"

detect_ports
require_calibration
assert_ports_free

print -- "ReBot hardware check"
print -- "Follower: $FOLLOWER_PORT (HDSC 2e88:4603, 921600 baud)"
print -- "Leader:   $LEADER_PORT (CH340 1a86:7523, 1000000 baud)"
print -- ""
print -- "[1/2] Scanning the seven follower motors..."

"$MOTORBRIDGE_BIN" scan \
  --vendor damiao \
  --transport dm-serial \
  --serial-port "$FOLLOWER_PORT" \
  --serial-baud 921600

print -- ""
print -- "[2/2] Pinging leader servo IDs 0 through 6..."

"$PYTHON_BIN" - "$LEADER_PORT" <<'PY'
from motorbridge_smart_servo import FashionStarServo
import sys

port = sys.argv[1]
bus = FashionStarServo(port, 1_000_000)
try:
    online = {i: bus.ping(i) for i in range(7)}
    state = bus.sync_monitor(list(range(7)))
    angles = {
        i: None if state[i] is None else round(state[i].angle_deg, 2)
        for i in range(7)
    }
    print("online:", online)
    print("angles_deg:", angles)
    if not all(online.values()):
        raise SystemExit(2)
finally:
    bus.close()
PY

print -- ""
print -- "PASS: follower motors 1-7 and leader servos 0-6 are reachable."
read -r "?Press Return to close this window..."
