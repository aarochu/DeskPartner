"""Pixel → desk-plane mm via four ArUco corners."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml


def detect_markers(frame: np.ndarray, dict_name: str = "DICT_4X4_50"):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(frame)
    return corners, ids


def fit_homography(
    image_pts: np.ndarray,
    plane_pts_mm: np.ndarray,
) -> np.ndarray:
    """image_pts (4,2) pixels → plane_pts_mm (4,2) mm. Returns 3x3 H."""
    H, _ = cv2.findHomography(image_pts, plane_pts_mm, method=0)
    if H is None:
        raise RuntimeError("Homography failed")
    return H


def pixel_to_plane(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    pt = np.array([u, v, 1.0], dtype=np.float64)
    mapped = H @ pt
    mapped /= mapped[2]
    return float(mapped[0]), float(mapped[1])


def save_H(H: np.ndarray, path: Path = Path("data/calibration/homography.npy")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, H)
    print(f"Saved homography → {path}")


def load_H(path: Path = Path("data/calibration/homography.npy")) -> np.ndarray:
    return np.load(path)


def main() -> None:
    """Stub CLI: expects you to fill plane corner mm after taping the zone."""
    from p2_vision_calibration.camera import grab_frame

    cfg = yaml.safe_load(Path("config/workspace.yaml").read_text())
    frame = grab_frame()
    corners, ids = detect_markers(frame, cfg["aruco"]["dictionary"])
    if ids is None or len(ids) < 4:
        print(f"Need 4 markers; got {0 if ids is None else len(ids)}. Check lighting/tape.")
        return
    print("Detected ids:", ids.flatten().tolist())
    # TODO: order corners by id 0..3 and set plane_pts_mm from measured zone
    print("Wire ordered image_pts + plane_pts_mm → fit_homography → save_H")


if __name__ == "__main__":
    main()
