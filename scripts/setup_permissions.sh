#!/usr/bin/env bash
# Grant serial access for follower (ACM) and leader (USB).
set -euo pipefail
sudo chmod 666 /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || {
  echo "No matching tty devices yet — plug arms in, then rerun."
  exit 1
}
echo "OK: serial ports writable"
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
