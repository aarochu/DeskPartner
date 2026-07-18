"""Friday-night calibration checklist (prints exact commands)."""

from __future__ import annotations

STEPS = """
=== DeskPartner Friday calibration ===

1) Reach-check corners BEFORE taping (P1):
   DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.reach_check --corners 0,0 400,0 400,300 0,300 --live

2) Tape zone + ArUco 0-3. Mount camera. Then:
   python -m p2_vision_calibration.detect_aruco_corners --camera 0 --show
   python -m p2_vision_calibration.pixel_to_plane --fit

3) Plane→arm with P1 (jog tip to each marker):
   DESKPARTNER_DRY_RUN=0 python -m p2_vision_calibration.plane_to_arm_registration --live

4) Exit criterion (must PASS ≤5mm):
   DESKPARTNER_DRY_RUN=0 python -m p2_vision_calibration.click_to_verify --camera 0 --live

5) Empty desk reference:
   python -m p2_vision_calibration.empty_desk_reference --camera 0

6) Teach home + destinations (P1):
   DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.teach_home --live
   DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.teach_destination trash --live
   DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.teach_destination pen_cup --live
   DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.teach_destination tray --live

7) One hardcoded paper pick:
   DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.pick_and_drop --x-mm 180 --y-mm 40 --item paper --dest trash --live

Shared truth: data/calibration/calibration.json
Recal anytime: see RECALIBRATION.md
"""


def main() -> None:
    print(STEPS)


if __name__ == "__main__":
    main()
