"""Verify the arm can hover above each work-zone corner before taping.

  python -m p1_arm_motion.reach_check --corners 0,0 400,0 400,300 0,300 --live
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

from p1_arm_motion.arm_client import ArmConfig, make_arm_client

log = logging.getLogger("p1.reach")


def parse_corner(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


def reach_check(
    corners_xy_mm: list[tuple[float, float]],
    live: bool = False,
    desk_z_m: float = 0.0,
) -> int:
    cfg = ArmConfig.from_yaml()
    arm = make_arm_client(cfg, dry_run=not live)
    hover_z_m = desk_z_m + cfg.transit_height_mm / 1000.0
    tol = cfg.reach_tolerance_mm
    arm.connect()
    fails = 0
    try:
        arm.move_to_home()
        for i, (x_mm, y_mm) in enumerate(corners_xy_mm):
            ok = arm.move_tip_m(x_mm / 1000.0, y_mm / 1000.0, hover_z_m)
            tip = arm.get_tip_m()
            ax_mm, ay_mm = tip[0] * 1000.0, tip[1] * 1000.0
            err = math.hypot(ax_mm - x_mm, ay_mm - y_mm)
            status = "PASS" if ok and err <= tol else "FAIL"
            if status == "FAIL":
                fails += 1
            print(
                f"corner[{i}] target_mm=({x_mm:.1f},{y_mm:.1f}) "
                f"reached_mm=({ax_mm:.1f},{ay_mm:.1f}) err={err:.1f}mm → {status}"
            )
        arm.move_to_home()
    finally:
        arm.disconnect()

    print(f"reach_check: {len(corners_xy_mm) - fails}/{len(corners_xy_mm)} passed (tol={tol}mm)")
    return 1 if fails else 0


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corners",
        nargs=4,
        required=True,
        help="Four X,Y pairs in mm, e.g. 0,0 400,0 400,300 0,300",
    )
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--desk-z-m", type=float, default=0.0)
    args = ap.parse_args()
    corners = [parse_corner(c) for c in args.corners]
    return reach_check(corners, live=args.live, desk_z_m=args.desk_z_m)


if __name__ == "__main__":
    sys.exit(main())
