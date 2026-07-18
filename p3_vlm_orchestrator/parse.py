"""Robust JSON extraction from VLM text."""

from __future__ import annotations

import json
import re
from typing import Any

from shared.types import DetectedItem, Destination, PlanResult

ALLOWED: set[str] = {"trash", "pen_cup", "tray", "keep"}


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def to_plan(raw: dict[str, Any], provider: str = "vlm") -> PlanResult:
    items: list[DetectedItem] = []
    for it in raw.get("items") or []:
        dest = str(it.get("destination", "tray")).lower().strip()
        if dest not in ALLOWED:
            dest = "tray"
        bbox = it.get("bbox_xyxy") or [0, 0, 0, 0]
        if len(bbox) != 4:
            continue
        items.append(
            DetectedItem(
                label=str(it.get("label", "object")),
                destination=dest,  # type: ignore[arg-type]
                bbox_xyxy=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            )
        )
    clean = bool(raw.get("desk_is_clean", False))
    if not items:
        clean = True
    return PlanResult(desk_is_clean=clean, items=items, raw_provider=provider)
