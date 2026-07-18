"""Capture current pose as out-of-FOV home into config/arm.yaml."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

from p1_arm_motion.arm_client import ArmConfig, make_arm_client


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--config", type=Path, default=Path("config/arm.yaml"))
    args = ap.parse_args()

    cfg = ArmConfig.from_yaml(args.config)
    arm = make_arm_client(cfg, dry_run=not args.live)
    arm.connect()
    try:
        tip = arm.get_tip_m()
        joints = arm.get_joints_rad()
    finally:
        arm.disconnect()

    raw = yaml.safe_load(args.config.read_text())
    home = raw.setdefault("home", {})
    home["xyz_m"] = [round(tip[0], 5), round(tip[1], 5), round(tip[2], 5)]
    home["rpy_rad"] = [round(tip[3], 5), round(tip[4], 5), round(tip[5], 5)]
    home["joints_deg"] = [round(math.degrees(v), 3) for v in joints]
    args.config.write_text(yaml.safe_dump(raw, sort_keys=False))
    print(f"Wrote home to {args.config}")
    print(f"  xyz_m={home['xyz_m']}")
    print(f"  joints_deg={home['joints_deg']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
