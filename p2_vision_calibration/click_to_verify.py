"""Exit criterion UI: click pixel → command arm tip → measure error ≤ 5 mm.

Stub: records click and prints target arm XY. Wire to P1 client for live move.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from p2_vision_calibration.camera import grab_frame
from p2_vision_calibration.plane_to_arm import Calibration


MAX_ERROR_MM = 5.0


def main() -> None:
    cal_dir = Path("data/calibration")
    if not (cal_dir / "homography.npy").exists():
        print("Run aruco_homography + plane_to_arm fit first.")
        return
    cal = Calibration.load(cal_dir)
    frame = grab_frame()
    clone = frame.copy()

    def on_click(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ax, ay = cal.pixel_to_arm(float(x), float(y))
        print(f"click=({x},{y}) → arm_xy_mm=({ax:.1f},{ay:.1f})")
        print(f"TODO: move tip here; measure error; pass if ≤ {MAX_ERROR_MM} mm")
        cv2.circle(clone, (x, y), 6, (0, 255, 0), -1)
        cv2.imshow("click_to_verify", clone)

    cv2.imshow("click_to_verify", clone)
    cv2.setMouseCallback("click_to_verify", on_click)
    print("Left-click targets. q to quit.")
    while True:
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
