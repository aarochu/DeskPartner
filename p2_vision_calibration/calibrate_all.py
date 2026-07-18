"""Friday-night calibration checklist runner."""

from __future__ import annotations

STEPS = [
    "1. Mount camera 60–75 cm, direct USB, lock exposure/WB",
    "2. Tape zone sized to reach; place ArUco ids 0–3",
    "3. python -m p2_vision_calibration.aruco_homography  (fit + save H)",
    "4. With P1: jog tip to each marker center → fit plane_to_arm → save",
    "5. python -m p2_vision_calibration.click_to_verify  (≤ 5 mm)",
    "6. Save empty-desk reference (clear zone first)",
]


def main() -> None:
    print("=== DeskPartner Friday calibration ===")
    for s in STEPS:
        print(s)
    print("\nExit criterion: click pixel → tip ≤ 5 mm. Do not proceed without it.")


if __name__ == "__main__":
    main()
