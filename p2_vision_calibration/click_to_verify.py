"""Hard exit criterion: click pixel → arm tip within 5 mm.

Chains pixel_to_arm → P1 verify_click_to_reach.

  python -m p2_vision_calibration.click_to_verify --camera 0 --live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from p2_vision_calibration.calibration_io import DEFAULT_PATH, load
from p2_vision_calibration.detect_aruco_corners import apply_exposure, open_camera
from p2_vision_calibration.pixel_to_arm import CalibrationChain
from p1_arm_motion.verify_click_to_reach import move_xy_mm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--cal", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--desk-z-m", type=float, default=0.0)
    args = ap.parse_args()

    data = load(args.cal)
    tol = float(data.get("click_verify_tolerance_mm", 5.0))
    cam_index = args.camera if args.camera is not None else int(data["camera"].get("index") or 0)

    try:
        chain = CalibrationChain(args.cal)
    except RuntimeError as exc:
        print(exc)
        return 1

    cap = open_camera(cam_index)
    apply_exposure(cap, data["camera"].get("lock"))
    ok, frame = cap.read()
    if not ok:
        print("Failed to grab frame")
        return 1

    win = "click_to_verify (click=move arm, q=quit)"
    clone = frame.copy()
    results: list[float] = []

    def on_click(event, x, y, _flags, _param):
        nonlocal clone
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        arm_x, arm_y = chain.pixel_to_arm_coords(float(x), float(y))
        print(f"click=({x},{y}) -> arm_mm=({arm_x:.2f},{arm_y:.2f}) - moving...")
        ax, ay, err = move_xy_mm(arm_x, arm_y, live=args.live, desk_z_m=args.desk_z_m)
        results.append(err)
        status = "PASS" if err <= tol else "FAIL"
        print(f"  reached=({ax:.2f},{ay:.2f}) err={err:.2f}mm tol={tol}mm -> {status}")
        color = (0, 255, 0) if err <= tol else (0, 0, 255)
        cv2.circle(clone, (x, y), 8, color, -1)
        cv2.putText(
            clone,
            f"{err:.1f}mm {status}",
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        cv2.imshow(win, clone)

    cv2.imshow(win, clone)
    cv2.setMouseCallback(win, on_click)
    print(f"Left-click targets. Exit criterion: <= {tol} mm. q to quit.")
    while True:
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

    if not results:
        print("No clicks recorded")
        return 1
    n_pass = sum(1 for e in results if e <= tol)
    print(f"Summary: {n_pass}/{len(results)} within {tol}mm")
    return 0 if n_pass >= min(3, len(results)) and all(e <= tol for e in results[-min(3, len(results)):]) else 2


if __name__ == "__main__":
    sys.exit(main())
