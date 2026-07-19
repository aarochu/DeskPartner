"""Live 3D arm view for Rerun recordings.

Draws a simple, intuitive skeleton of the reBot B601-DM from the seven joint
observations (degrees) and curates a minimal viewer layout, so the Rerun
window that pops up when recording starts shows a moving robot next to the
two cameras — instead of a wall of raw scalar lanes.

Deliberately an approximation, not a URDF: link lengths are real-ish, joint
conventions are chosen so the model moves the way the arm feels. The training
signal remains the logged joint scalars; this view is for humans.

Everything here fails soft. A missing joint, a NaN, or an old Rerun SDK must
never take down the 240 Hz collector.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping

import rerun as rr

log = logging.getLogger(__name__)

# ── geometry (metres, approximating the B601-DM's ~0.74 m reach) ──────────────
BASE_HEIGHT_M = 0.10
UPPER_ARM_M = 0.30
FOREARM_M = 0.28
WRIST_M = 0.16

# Gripper value (degrees) → visual jaw separation. 0° = closed on this rig.
GRIPPER_CLOSED_DEG = 0.0
GRIPPER_OPEN_DEG = 90.0
JAW_CLOSED_M = 0.008
JAW_OPEN_M = 0.06
JAW_LENGTH_M = 0.05

# Entity paths (single /world tree so one 3D view shows everything).
SCENE_FLOOR_ENTITY = "world/floor"
SCENE_REACH_ENTITY = "world/reach"
ARM_ENTITY = "world/robot/arm"
JOINTS_ENTITY = "world/robot/joints"
GRIPPER_ENTITY = "world/robot/gripper"

ARM_COLOR = (62, 207, 218)      # follower cyan, matching the operator kit
JOINT_COLOR = (240, 244, 248)
GRIPPER_COLOR = (255, 180, 84)  # command amber
FLOOR_COLOR = (90, 99, 92)

REQUIRED_JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
)
OPTIONAL_JOINTS = ("wrist_yaw.pos", "wrist_roll.pos", "gripper.pos")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def extract_joints(observation: Mapping[str, Any]) -> dict[str, float] | None:
    """Pull the joint angles (degrees) out of a robot observation.

    Camera frames and any other keys are ignored. Returns None when a
    required joint is missing or non-finite — callers skip the frame.
    """
    pose: dict[str, float] = {}
    for key in REQUIRED_JOINTS:
        value = _finite(observation.get(key))
        if value is None:
            return None
        pose[key] = value
    for key in OPTIONAL_JOINTS:
        value = _finite(observation.get(key))
        pose[key] = 0.0 if value is None else value
    return pose


def fk_points(observation: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    """Base→tip points of the simple skeleton for one pose (degrees in).

    Convention: all-zero pose stands straight up; ``shoulder_lift`` pitches
    the upper arm away from vertical; ``elbow_flex`` and ``wrist_flex`` are
    relative pitches; ``shoulder_pan`` spins everything about the base.
    """
    pose = extract_joints(observation)
    if pose is None:
        raise ValueError("observation is missing required joints")
    pan = math.radians(pose["shoulder_pan.pos"])
    pitches = (
        math.radians(pose["shoulder_lift.pos"]),
        math.radians(pose["shoulder_lift.pos"] + pose["elbow_flex.pos"]),
        math.radians(pose["shoulder_lift.pos"] + pose["elbow_flex.pos"] + pose["wrist_flex.pos"]),
    )
    lengths = (UPPER_ARM_M, FOREARM_M, WRIST_M)
    points: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0), (0.0, 0.0, BASE_HEIGHT_M)]
    x, y, z = points[-1]
    for pitch, length in zip(pitches, lengths):
        x += length * math.sin(pitch) * math.cos(pan)
        y += length * math.sin(pitch) * math.sin(pan)
        z += length * math.cos(pitch)
        points.append((x, y, z))
    return points


def jaw_separation_m(gripper_deg: float) -> float:
    """Visual jaw gap for a gripper joint value, clamped to the real range."""
    span = GRIPPER_OPEN_DEG - GRIPPER_CLOSED_DEG
    fraction = (float(gripper_deg) - GRIPPER_CLOSED_DEG) / span if span else 0.0
    fraction = min(1.0, max(0.0, fraction))
    return JAW_CLOSED_M + fraction * (JAW_OPEN_M - JAW_CLOSED_M)


def _gripper_strips(
    tip: tuple[float, float, float], pan_deg: float, wrist_yaw_deg: float, gripper_deg: float
) -> list[list[tuple[float, float, float]]]:
    angle = math.radians(pan_deg + wrist_yaw_deg + 90.0)
    px, py = math.cos(angle), math.sin(angle)
    half = jaw_separation_m(gripper_deg) / 2.0
    strips = []
    for side in (1.0, -1.0):
        jx, jy = tip[0] + side * px * half, tip[1] + side * py * half
        strips.append([(jx, jy, tip[2]), (jx, jy, max(0.0, tip[2] - JAW_LENGTH_M))])
    strips.append([(tip[0] - px * half, tip[1] - py * half, tip[2]), (tip[0] + px * half, tip[1] + py * half, tip[2])])
    return strips


def log_scene(recording: Any) -> bool:
    """Log the static scene (floor grid + reach ring) once per attempt."""
    try:
        extent, step = 0.8, 0.1
        lines = []
        ticks = int(extent / step)
        for i in range(-ticks, ticks + 1):
            v = i * step
            lines.append([(v, -extent, 0.0), (v, extent, 0.0)])
            lines.append([(-extent, v, 0.0), (extent, v, 0.0)])
        recording.log(
            SCENE_FLOOR_ENTITY,
            rr.LineStrips3D(lines, colors=[FLOOR_COLOR], radii=0.001),
            static=True,
        )
        reach = UPPER_ARM_M + FOREARM_M + WRIST_M
        ring = [
            (reach * math.cos(a / 48 * math.tau), reach * math.sin(a / 48 * math.tau), 0.002)
            for a in range(49)
        ]
        recording.log(
            SCENE_REACH_ENTITY,
            rr.LineStrips3D([ring], colors=[FLOOR_COLOR], radii=0.0015),
            static=True,
        )
        return True
    except Exception as exc:  # noqa: BLE001 — visual aid must never break recording
        log.warning("3D scene not logged: %s", exc)
        return False


def log_pose(recording: Any, observation: Mapping[str, Any]) -> bool:
    """Log one skeleton pose from a robot observation. Fails soft."""
    pose = extract_joints(observation)
    if pose is None:
        return False
    try:
        points = fk_points(pose)
        recording.log(
            ARM_ENTITY,
            rr.LineStrips3D([points], colors=[ARM_COLOR], radii=0.011),
        )
        recording.log(
            JOINTS_ENTITY,
            rr.Points3D(points[1:], colors=[JOINT_COLOR], radii=0.016),
        )
        recording.log(
            GRIPPER_ENTITY,
            rr.LineStrips3D(
                _gripper_strips(
                    points[-1], pose["shoulder_pan.pos"], pose["wrist_yaw.pos"], pose["gripper.pos"]
                ),
                colors=[GRIPPER_COLOR],
                radii=0.006,
            ),
        )
        return True
    except Exception as exc:  # noqa: BLE001 — visual aid must never break recording
        log.warning("3D pose not logged: %s", exc)
        return False


def send_simple_blueprint(recording: Any) -> bool:
    """Open the viewer on a small, curated layout: robot, cameras, two plots.

    Keeps the first look simple — everything else stays reachable through
    Rerun's own panels, which start collapsed.
    """
    try:
        import rerun.blueprint as rrb

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/world", name="Robot (live)"),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="/observation/front", name="Overhead camera"),
                    rrb.Spatial2DView(origin="/observation/side", name="Wrist camera"),
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(origin="/observation/gripper", name="Gripper"),
                    rrb.TimeSeriesView(
                        name="Arm joints",
                        origin="/observation",
                        contents=[
                            "+ /observation/shoulder_pan/**",
                            "+ /observation/shoulder_lift/**",
                            "+ /observation/elbow_flex/**",
                        ],
                    ),
                ),
                column_shares=[5, 3, 3],
            ),
            collapse_panels=True,
        )
        recording.send_blueprint(blueprint)
        return True
    except Exception as exc:  # noqa: BLE001 — layout is a nicety, recording is not
        log.warning("Simple blueprint not sent: %s", exc)
        return False
