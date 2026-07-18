"""Capture empty-desk reference photo for CV fallback + lighting baseline.

  python -m p2_vision_calibration.empty_desk_reference --camera 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from p2_vision_calibration.calibration_io import DEFAULT_PATH, load, save
from p2_vision_calibration.detect_aruco_corners import apply_exposure, open_camera


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--cal", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = load(args.cal)
    cam_index = args.camera if args.camera is not None else int(data["camera"].get("index") or 0)
    out = args.out or Path(data.get("empty_desk_image") or "data/calibration/empty_desk.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Clear the work zone, then press Enter to capture.")
    input()

    cap = open_camera(cam_index)
    apply_exposure(cap, data["camera"].get("lock"))
    for _ in range(8):
        ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Capture failed")
        return 1

    cv2.imwrite(str(out), frame)
    data["empty_desk_image"] = str(out).replace("\\", "/")
    save(data, args.cal)
    print(f"Saved empty desk -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
