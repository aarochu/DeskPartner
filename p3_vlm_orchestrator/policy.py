"""Attach grasp heights; drop keep items from the action queue."""

from __future__ import annotations

from pathlib import Path

import yaml

from shared.types import DetectedItem, PlanResult


def load_grasp_heights(path: str | Path = "config/arm.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text()).get("grasp_heights_mm") or {}


def actionable_items(plan: PlanResult, grasp_heights: dict | None = None) -> list[DetectedItem]:
    heights = grasp_heights or load_grasp_heights()
    default_h = float(heights.get("default", 20))
    out: list[DetectedItem] = []
    for item in plan.items:
        if item.destination == "keep":
            continue
        key = item.label.lower().replace(" ", "_")
        # crude match on curated names
        h = default_h
        for name, val in heights.items():
            if name == "default":
                continue
            if name in key:
                h = float(val)
                break
        item.grasp_height_mm = h
        out.append(item)
    return out
