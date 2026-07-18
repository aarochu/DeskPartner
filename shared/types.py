"""Handoff contracts between tracks.

P2 → P3: pixel in, arm coordinates out
P3 → P1: arm coordinates + category in, pick happens
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Destination = Literal["trash", "pen_cup", "tray", "keep"]


@dataclass
class ArmPose:
    """Tip pose in arm base frame (mm / degrees)."""

    x_mm: float
    y_mm: float
    z_mm: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class DetectedItem:
    label: str
    destination: Destination
    bbox_xyxy: tuple[int, int, int, int]  # pixel, image coords
    centroid_uv: tuple[float, float] | None = None  # refined by P2 CV
    arm_xy_mm: tuple[float, float] | None = None  # after P2 transform
    grasp_height_mm: float | None = None


@dataclass
class PlanResult:
    desk_is_clean: bool
    items: list[DetectedItem] = field(default_factory=list)
    raw_provider: str = ""
    image_path: str | None = None
