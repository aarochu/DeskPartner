"""Move tip to an (x,y) mm target — used by P2 click-to-verify exit criterion.

  python -m p1_arm_motion.verify_click_to_reach 180 40 --live
  python -m p1_arm_motion.verify_click_to_reach --x-mm 180 --y-mm 40 --live
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

from p1_arm_motion.arm_client import ArmConfig, make_arm_client

log = logging.getLogger("p1.verify")


def move_xy_mm(
    x_mm: float,
    y_mm: float,
    live: bool = False,
    desk_z_m: float = 0.0,
    hover: bool = True,
) -> tuple[float, float, float]:
    """Move tip to (x,y) mm; return actual tip (x_mm, y_mm, err_mm)."""
    cfg = ArmConfig.from_yaml()
    arm = make_arm_client(cfg, dry_run=not live)
    z_m = desk_z_m + (cfg.transit_height_mm if hover else cfg.grasp_z_mm("default")) / 1000.0
    arm.connect()
    try:
        ok = arm.move_tip_m(x_mm / 1000.0, y_mm / 1000.0, z_m)
        tip = arm.get_tip_m()
        ax, ay = tip[0] * 1000.0, tip[1] * 1000.0
        err = math.hypot(ax - x_mm, ay - y_mm)
        tol = cfg.click_verify_tolerance_mm
        status = "PASS" if ok and err <= tol else "FAIL"
        print(
            f"target_mm=({x_mm:.2f},{y_mm:.2f}) reached_mm=({ax:.2f},{ay:.2f}) "
            f"err={err:.2f}mm tol={tol}mm → {status}"
        )
        return ax, ay, err
    finally:
        arm.disconnect()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Move arm tip to plane XY (mm)")
    ap.add_argument("x", nargs="?", type=float, help="x mm")
    ap.add_argument("y", nargs="?", type=float, help="y mm")
    ap.add_argument("--x-mm", type=float)
    ap.add_argument("--y-mm", type=float)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--desk-z-m", type=float, default=0.0)
    args = ap.parse_args()

    x = args.x_mm if args.x_mm is not None else args.x
    y = args.y_mm if args.y_mm is not None else args.y
    if x is None or y is None:
        ap.error("provide x y (positional) or --x-mm --y-mm")

    cfg = ArmConfig.from_yaml()
    _, _, err = move_xy_mm(x, y, live=args.live, desk_z_m=args.desk_z_m)
    return 0 if err <= cfg.click_verify_tolerance_mm else 2


if __name__ == "__main__":
    sys.exit(main())
