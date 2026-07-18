"""Refine object centroid inside a VLM bounding box (local CV)."""

from __future__ import annotations

import cv2
import numpy as np


def refine_centroid(
    frame: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Return (u, v) pixel centroid; falls back to box center if empty."""
    x1, y1, x2, y2 = bbox_xyxy
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Prefer the foreground blob; invert if needed so object is white
    if np.mean(th) > 127:
        th = 255 - th
    moments = cv2.moments(th)
    if moments["m00"] < 1e-3:
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cu = moments["m10"] / moments["m00"] + x1
    cv_ = moments["m01"] / moments["m00"] + y1
    return float(cu), float(cv_)
