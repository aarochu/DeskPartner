"""Detect ArUco ids 0–3 and save pixel centers into calibration.json.

  python -m p2_vision_calibration.detect_aruco_corners --camera 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from p2_vision_calibration.calibration_io import DEFAULT_PATH, load, save


def open_camera(index: int) -> cv2.VideoCapture:
    # Windows: DSHOW; Linux: default V4L2
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    return cap


def lock_exposure(cap: cv2.VideoCapture) -> dict:
    """Best-effort lock. Drivers vary; we store whatever we can read back."""
    # Disable auto where supported
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual on many UVC
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    settings = {
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "white_balance": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
        "gain": cap.get(cv2.CAP_PROP_GAIN),
    }
    return settings


def apply_exposure(cap: cv2.VideoCapture, settings: dict | None) -> None:
    if not settings:
        return
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    for prop, key in [
        (cv2.CAP_PROP_EXPOSURE, "exposure"),
        (cv2.CAP_PROP_WB_TEMPERATURE, "white_balance"),
        (cv2.CAP_PROP_BRIGHTNESS, "brightness"),
        (cv2.CAP_PROP_GAIN, "gain"),
    ]:
        if settings.get(key) is not None:
            cap.set(prop, float(settings[key]))


def detect_centers(frame: np.ndarray, dict_name: str = "DICT_4X4_50") -> dict[int, tuple[float, float]]:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(frame)
    out: dict[int, tuple[float, float]] = {}
    if ids is None:
        return out
    for i, mid in enumerate(ids.flatten()):
        c = corners[i][0]
        out[int(mid)] = (float(np.mean(c[:, 0])), float(np.mean(c[:, 1])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="/dev/videoN or Windows index")
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--seconds", type=float, default=2.0, help="warm-up capture time")
    args = ap.parse_args()

    data = load(args.out)
    cap = open_camera(args.camera)
    t_end = time.time() + args.seconds
    frame = None
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Failed to grab frame")
    assert frame is not None

    settings = lock_exposure(cap)
    # one more frame after lock
    for _ in range(5):
        ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to grab frame after exposure lock")

    centers = detect_centers(frame, data["aruco"]["dictionary"])
    print("Detected:", centers)

    if args.show:
        vis = frame.copy()
        for mid, (u, v) in centers.items():
            cv2.circle(vis, (int(u), int(v)), 8, (0, 255, 0), -1)
            cv2.putText(vis, str(mid), (int(u) + 10, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("aruco", vis)
        print("Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cap.release()

    missing = [i for i in data["aruco"]["ids"] if i not in centers]
    data["camera"]["index"] = args.camera
    data["camera"].update({k: settings.get(k) for k in ("exposure", "white_balance", "width", "height")})
    data["camera"]["lock"] = settings
    data["aruco"]["pixel_centers"] = {str(k): [v[0], v[1]] for k, v in centers.items()}
    save(data, args.out)

    if missing:
        print(f"WARNING: missing marker ids {missing} — fix lighting/tape and rerun")
        return 2
    print("OK: all four markers found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
