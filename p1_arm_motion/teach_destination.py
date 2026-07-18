"""Record current EE pose + joints into config/destinations.yaml.

Workflow: teleop/jog tip to the release pose over the container, then:
  python -m p1_arm_motion.teach_destination trash --live
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from p1_arm_motion.arm_client import ArmConfig, make_arm_client

DEST_PATH = Path("config/destinations.yaml")


def teach_destination(name: str, path: Path = DEST_PATH, live: bool = False) -> None:
    if name == "keep":
        raise SystemExit("cannot teach 'keep'")

    cfg = ArmConfig.from_yaml()
    arm = make_arm_client(cfg, dry_run=not live)
    arm.connect()
    try:
        tip = arm.get_tip_m()
        joints = arm.get_joints_rad()
    finally:
        arm.disconnect()

    x, y, z, r, p, yaw = tip
    data = yaml.safe_load(path.read_text()) if path.exists() else {"destinations": {}}
    entry = data.setdefault("destinations", {}).setdefault(name, {"label": name})
    entry["xyz_m"] = [round(x, 5), round(y, 5), round(z, 5)]
    entry["rpy_rad"] = [round(r, 5), round(p, 5), round(yaw, 5)]
    entry["joints_rad"] = [round(v, 5) for v in joints]
    entry["taught_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Saved destination '{name}' -> {path}")
    print(f"  xyz_m={entry['xyz_m']}")
    print(f"  joints_rad={entry['joints_rad']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=["trash", "pen_cup", "tray"])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--path", type=Path, default=DEST_PATH)
    args = ap.parse_args()
    teach_destination(args.name, path=args.path, live=args.live)
    return 0


if __name__ == "__main__":
    sys.exit(main())
