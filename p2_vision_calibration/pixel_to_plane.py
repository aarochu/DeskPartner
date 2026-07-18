"""Homography: camera pixel → desk-plane mm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from p2_vision_calibration.calibration_io import DEFAULT_PATH, load, save


def fit_homography_from_cal(data: dict) -> np.ndarray:
    centers = data["aruco"]["pixel_centers"]
    plane = data["aruco"]["plane_mm"]
    ids = [str(i) for i in data["aruco"]["ids"]]
    missing = [i for i in ids if i not in centers]
    if missing:
        raise RuntimeError(f"Missing pixel centers for ids {missing} — run detect_aruco_corners")
    img_pts = np.array([centers[i] for i in ids], dtype=np.float64)
    plane_pts = np.array([plane[i] for i in ids], dtype=np.float64)
    H, _ = cv2.findHomography(img_pts, plane_pts, method=0)
    if H is None:
        raise RuntimeError("findHomography failed")
    return H


def pixel_to_mm(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
    pt = np.array([px, py, 1.0], dtype=np.float64)
    mapped = H @ pt
    mapped /= mapped[2]
    return float(mapped[0]), float(mapped[1])


def load_H(path: Path = DEFAULT_PATH) -> np.ndarray:
    data = load(path)
    if not data.get("homography"):
        raise RuntimeError("No homography in calibration — run pixel_to_plane fit")
    return np.array(data["homography"], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--fit", action="store_true", help="Fit H from ArUco centers + plane_mm")
    ap.add_argument("--px", type=float)
    ap.add_argument("--py", type=float)
    # Optional override of plane corners (mm) as id:x,y
    ap.add_argument(
        "--plane",
        nargs="*",
        help="Override plane_mm e.g. 0:0,0 1:400,0 2:400,300 3:0,300",
    )
    args = ap.parse_args()

    data = load(args.cal)
    if args.plane:
        for item in args.plane:
            mid, xy = item.split(":")
            x, y = xy.split(",")
            data["aruco"]["plane_mm"][str(mid)] = [float(x), float(y)]

    if args.fit or not data.get("homography"):
        H = fit_homography_from_cal(data)
        data["homography"] = H.tolist()
        save(data, args.cal)
        print("Homography fitted and saved")
    else:
        H = np.array(data["homography"], dtype=np.float64)

    if args.px is not None and args.py is not None:
        x, y = pixel_to_mm(args.px, args.py, H)
        print(f"pixel=({args.px},{args.py}) → plane_mm=({x:.2f},{y:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
