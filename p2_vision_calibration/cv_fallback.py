"""Classical CV detector vs empty-desk reference — no API, no network."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from shared.types import DetectedItem, Destination, PlanResult


def _load_rules(path: str | Path = "config/cv_color_rules.yaml") -> list[dict]:
    return yaml.safe_load(Path(path).read_text()).get("rules", [])


def detect(
    frame: np.ndarray,
    empty_ref_path: str | Path = "data/calibration/empty_desk.png",
    min_area: int = 400,
) -> PlanResult:
    ref_path = Path(empty_ref_path)
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing empty-desk reference: {ref_path}")

    ref = cv2.imread(str(ref_path))
    if ref is None:
        raise RuntimeError("Failed to read empty-desk reference")
    if ref.shape[:2] != frame.shape[:2]:
        ref = cv2.resize(ref, (frame.shape[1], frame.shape[0]))

    diff = cv2.absdiff(frame, ref)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rules = _load_rules()
    items: list[DetectedItem] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        dest = _classify_roi(frame[y : y + h, x : x + w], rules)
        items.append(
            DetectedItem(
                label="cv_blob",
                destination=dest,
                bbox_xyxy=(x, y, x + w, y + h),
                centroid_uv=(x + w / 2.0, y + h / 2.0),
            )
        )

    return PlanResult(desk_is_clean=len(items) == 0, items=items, raw_provider="cv_fallback")


def _classify_roi(roi: np.ndarray, rules: list[dict]) -> Destination:
    if roi.size == 0:
        return "tray"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    for rule in rules:
        lo, hi = rule.get("hsv_lower"), rule.get("hsv_upper")
        if lo is None or hi is None:
            continue
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        if np.count_nonzero(mask) > 0.15 * mask.size:
            return rule["destination"]  # type: ignore[return-value]
    return "tray"
