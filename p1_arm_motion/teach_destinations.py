"""Jog the tip to each container and write config/destinations.yaml.

Friday night / Saturday AM. Uses DryRun prompts until ReBotArmClient is wired.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DEST_NAMES = ("trash", "pen_cup", "tray")


def main() -> None:
    path = Path("config/destinations.yaml")
    data = yaml.safe_load(path.read_text())
    print("Teach destinations: for each, jog tip to release pose, then enter x y z (mm).")
    for name in DEST_NAMES:
        raw = input(f"{name} xyz_mm (e.g. 280 -180 80): ").strip()
        if not raw:
            print(f"  skip {name}")
            continue
        x, y, z = [float(v) for v in raw.split()]
        data["destinations"][name]["xyz_mm"] = [x, y, z]
        data["destinations"][name]["rpy_deg"] = [0.0, 0.0, 0.0]
        print(f"  saved {name}={data['destinations'][name]['xyz_mm']}")
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
