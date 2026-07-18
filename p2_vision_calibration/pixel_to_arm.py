"""P3 contract: pixel → arm mm in one call."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from p2_vision_calibration.calibration_io import DEFAULT_PATH, load
from p2_vision_calibration.pixel_to_plane import pixel_to_mm
from p2_vision_calibration.plane_to_arm_registration import plane_to_arm


class CalibrationChain:
    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self.path = path
        self.data = load(path)
        if not self.data.get("homography"):
            raise RuntimeError("Missing homography — run detect_aruco_corners + pixel_to_plane --fit")
        p2a = self.data.get("plane_to_arm") or {}
        if not p2a.get("A") or not p2a.get("b"):
            raise RuntimeError("Missing plane_to_arm — run plane_to_arm_registration")
        self.H = np.array(self.data["homography"], dtype=np.float64)
        self.A = np.array(p2a["A"], dtype=np.float64)
        self.b = np.array(p2a["b"], dtype=np.float64)

    def pixel_to_arm_coords(self, px: float, py: float) -> tuple[float, float]:
        x_mm, y_mm = pixel_to_mm(px, py, self.H)
        return plane_to_arm(x_mm, y_mm, self.A, self.b)

    # aliases used elsewhere
    def pixel_to_arm(self, u: float, v: float) -> tuple[float, float]:
        return self.pixel_to_arm_coords(u, v)


def pixel_to_arm_coords(px: float, py: float, path: Path = DEFAULT_PATH) -> tuple[float, float]:
    return CalibrationChain(path).pixel_to_arm_coords(px, py)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("px", type=float)
    ap.add_argument("py", type=float)
    ap.add_argument("--cal", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args()
    ax, ay = pixel_to_arm_coords(args.px, args.py, args.cal)
    print(f"pixel=({args.px},{args.py}) → arm_mm=({ax:.2f},{ay:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
