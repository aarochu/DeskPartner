"""Fake inputs so nobody blocks on a late handoff."""

from __future__ import annotations

from shared.types import ArmPose, DetectedItem, PlanResult


def fake_messy_plan() -> PlanResult:
    """P3 can drive orchestrator without a real camera/VLM."""
    return PlanResult(
        desk_is_clean=False,
        raw_provider="fixture",
        items=[
            DetectedItem(
                label="crumpled_paper",
                destination="trash",
                bbox_xyxy=(120, 200, 220, 300),
                centroid_uv=(170.0, 250.0),
                arm_xy_mm=(180.0, 40.0),
                grasp_height_mm=15.0,
            ),
            DetectedItem(
                label="blue_marker",
                destination="pen_cup",
                bbox_xyxy=(400, 180, 480, 320),
                centroid_uv=(440.0, 250.0),
                arm_xy_mm=(220.0, -30.0),
                grasp_height_mm=20.0,
            ),
            DetectedItem(
                label="phone",
                destination="keep",
                bbox_xyxy=(300, 350, 420, 520),
                centroid_uv=(360.0, 435.0),
                arm_xy_mm=(200.0, 80.0),
                grasp_height_mm=None,
            ),
        ],
    )


def fake_clean_plan() -> PlanResult:
    return PlanResult(desk_is_clean=True, items=[], raw_provider="fixture")


def fake_destination_pose(name: str) -> ArmPose:
    presets = {
        "trash": ArmPose(280, -180, 80),
        "pen_cup": ArmPose(300, 160, 90),
        "tray": ArmPose(320, 0, 70),
    }
    return presets[name]
