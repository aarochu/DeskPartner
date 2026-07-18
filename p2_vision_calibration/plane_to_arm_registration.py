"""Interactive plane-mm → arm-mm registration (4 point pairs, least squares).

For each ArUco plane corner: jog tip to marker center, press Enter, we record FK tip.

  DESKPARTNER_DRY_RUN=0 python -m p2_vision_calibration.plane_to_arm_registration --live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from p2_vision_calibration.calibration_io import DEFAULT_PATH, load, save


def fit_affine(plane_xy: np.ndarray, arm_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """arm = A @ plane + b. Returns A (2x2), b (2,)."""
    # Prefer OpenCV estimateAffine2D when available
    aff, inliers = cv2.estimateAffine2D(plane_xy.reshape(-1, 1, 2), arm_xy.reshape(-1, 1, 2))
    if aff is not None:
        A = aff[:, :2]
        b = aff[:, 2]
        return A, b

    n = plane_xy.shape[0]
    M = np.zeros((2 * n, 6))
    y = np.zeros(2 * n)
    for i, ((px, py), (ax, ay)) in enumerate(zip(plane_xy, arm_xy)):
        M[2 * i] = [px, py, 1, 0, 0, 0]
        M[2 * i + 1] = [0, 0, 0, px, py, 1]
        y[2 * i] = ax
        y[2 * i + 1] = ay
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    A = np.array([[coef[0], coef[1]], [coef[3], coef[4]]])
    b = np.array([coef[2], coef[5]])
    return A, b


def plane_to_arm(x_mm: float, y_mm: float, A: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    p = A @ np.array([x_mm, y_mm]) + b
    return float(p[0]), float(p[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--live", action="store_true", help="Read tip from real arm FK")
    ap.add_argument(
        "--manual",
        action="store_true",
        help="Type arm x,y mm instead of reading FK (no arm needed)",
    )
    args = ap.parse_args()

    data = load(args.cal)
    plane = data["aruco"]["plane_mm"]
    ids = [str(i) for i in data["aruco"]["ids"]]

    arm_client = None
    if args.live and not args.manual:
        from p1_arm_motion.arm_client import ArmConfig, make_arm_client

        cfg = ArmConfig.from_yaml()
        arm_client = make_arm_client(cfg, dry_run=False)
        arm_client.connect()

    pairs = []
    try:
        for mid in ids:
            px, py = plane[mid]
            print(f"\n=== Marker id={mid} plane_mm=({px}, {py}) ===")
            print("Jog tip to the marker center, then press Enter.")
            input()
            if args.manual or arm_client is None:
                raw = input("Enter arm tip x_mm y_mm: ").strip()
                ax, ay = [float(v) for v in raw.split()]
            else:
                tip = arm_client.get_tip_m()
                ax, ay = tip[0] * 1000.0, tip[1] * 1000.0
                print(f"  FK tip_mm=({ax:.2f}, {ay:.2f})")
            pairs.append({"id": int(mid), "plane_mm": [px, py], "arm_mm": [ax, ay]})
    finally:
        if arm_client is not None:
            arm_client.disconnect()

    plane_xy = np.array([p["plane_mm"] for p in pairs], dtype=np.float64)
    arm_xy = np.array([p["arm_mm"] for p in pairs], dtype=np.float64)
    A, b = fit_affine(plane_xy, arm_xy)

    # residual report
    print("\nResiduals:")
    for p in pairs:
        est = plane_to_arm(p["plane_mm"][0], p["plane_mm"][1], A, b)
        err = np.hypot(est[0] - p["arm_mm"][0], est[1] - p["arm_mm"][1])
        print(f"  id={p['id']} err={err:.2f}mm")

    data["plane_to_arm"] = {
        "A": A.tolist(),
        "b": b.tolist(),
        "pairs": pairs,
    }
    save(data, args.cal)
    print("plane→arm transform saved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
