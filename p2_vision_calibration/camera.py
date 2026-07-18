"""Overhead camera capture. Prefer locked exposure once lighting is final."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml


def load_camera_index(path: str | Path = "config/cameras.yaml") -> int:
    raw = yaml.safe_load(Path(path).read_text())
    idx = raw["overhead"]["index_or_path"]
    return int(idx)


def grab_frame(index: int | None = None) -> np.ndarray:
    index = load_camera_index() if index is None else index
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    # Warm-up
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to grab frame")
    return frame


def save_empty_desk_reference(out: Path = Path("data/calibration/empty_desk.png")) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = grab_frame()
    cv2.imwrite(str(out), frame)
    print(f"Saved empty-desk reference → {out}")
    return out
