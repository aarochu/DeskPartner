"""Fail-closed calibrated workspace validation for learned follower actions.

The learned seven-dimensional action is actuated by the LeRobot follower plugin.
The P1 ``ArmConfig`` and reBot SDK are used here only to forward-kinematics check
the first six physical follower joints; the Cartesian P1 client is not an
actuation path for policy actions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


DEFAULT_MAX_CALIBRATION_AGE_S = 12 * 60 * 60
MAX_FUTURE_SKEW_S = 5 * 60
DEFAULT_APPROACH_HEIGHT_MM = 40.0
POLYGON_TOLERANCE = 1e-6
DIMENSION_TOLERANCE_MM = 1e-3


class WorkspaceViolation(RuntimeError):
    """Raised when a proposed physical follower action is not workspace-safe."""


@dataclass(frozen=True)
class CalibratedWorkspaceGuard:
    """Validate predicted tip position against a calibrated arm-frame prism."""

    polygon_xy_mm: np.ndarray
    z_min_mm: float
    z_max_mm: float
    fk_deg_to_xyz_mm: Callable[[np.ndarray], Sequence[float] | np.ndarray]

    @classmethod
    def from_files(
        cls,
        *,
        arm_config_path: Path | str,
        workspace_config_path: Path | str,
        calibration_path: Path | str,
        fk_deg_to_xyz_mm: (
            Callable[[np.ndarray], Sequence[float] | np.ndarray] | None
        ) = None,
        current_utc: datetime | Callable[[], datetime] | None = None,
        max_calibration_age_s: float = DEFAULT_MAX_CALIBRATION_AGE_S,
    ) -> CalibratedWorkspaceGuard:
        """Load and authenticate the geometric inputs without touching hardware."""

        arm_path = Path(arm_config_path).expanduser().resolve()
        workspace_path = Path(workspace_config_path).expanduser().resolve()
        calibration_file = Path(calibration_path).expanduser().resolve()

        arm_raw = _load_mapping(arm_path, "arm config")
        workspace = _load_mapping(workspace_path, "workspace config")
        calibration = _load_mapping(calibration_file, "workspace calibration")

        now = current_utc() if callable(current_utc) else current_utc
        if now is None:
            now = datetime.now(timezone.utc)
        checked_now = _aware_utc(now, "Current UTC time")
        _validate_freshness(
            calibration.get("updated_at"),
            now=checked_now,
            max_age_s=max_calibration_age_s,
        )

        affine = calibration.get("plane_to_arm")
        if not isinstance(affine, Mapping):
            raise ValueError("Workspace calibration affine transform is missing")
        A = _finite_array(affine.get("A"), (2, 2), "affine A")
        b = _finite_array(affine.get("b"), (2,), "affine b")

        zone = workspace.get("zone")
        if not isinstance(zone, Mapping):
            raise ValueError("Workspace zone configuration is missing")
        width = _positive_finite(zone.get("width_mm"), "workspace zone width")
        depth = _positive_finite(zone.get("depth_mm"), "workspace zone depth")
        plane_corners = _plane_corners(calibration)
        _validate_zone_dimensions(plane_corners, width, depth)

        polygon = np.asarray([A @ corner + b for corner in plane_corners], dtype=float)
        _validate_convex_polygon(polygon)
        polygon.setflags(write=False)

        # ArmConfig is the single source of the P1 arm safety fields. The raw
        # mapping is read only for approach_height_mm because the existing
        # dataclass intentionally does not expose that field yet.
        from p1_arm_motion.arm_client import ArmConfig

        arm_config = ArmConfig.from_yaml(arm_path)
        grasp_heights = np.asarray(
            [float(value) for value in arm_config.grasp_heights_mm.values()],
            dtype=float,
        )
        if grasp_heights.size == 0 or not np.all(np.isfinite(grasp_heights)):
            raise ValueError("Arm grasp heights must contain finite values")
        safety_raw = arm_raw.get("safety")
        if safety_raw is None:
            safety_raw = {}
        if not isinstance(safety_raw, Mapping):
            raise ValueError("Arm safety config must be an object")
        approach = _finite_float(
            safety_raw.get("approach_height_mm", DEFAULT_APPROACH_HEIGHT_MM),
            "arm approach height",
        )
        z_min = float(np.min(grasp_heights)) - 10.0
        z_max = float(arm_config.transit_height_mm) + approach
        if not math.isfinite(z_min) or not math.isfinite(z_max) or z_min >= z_max:
            raise ValueError("Configured workspace Z range is invalid")

        fk = fk_deg_to_xyz_mm or _make_sdk_fk(arm_config)
        if not callable(fk):
            raise ValueError("Workspace forward kinematics adapter must be callable")
        return cls(
            polygon_xy_mm=polygon,
            z_min_mm=z_min,
            z_max_mm=z_max,
            fk_deg_to_xyz_mm=fk,
        )

    def validate(self, action_deg: np.ndarray) -> None:
        """Return normally only when the predicted physical action is in bounds."""

        try:
            action = np.asarray(action_deg, dtype=float).copy()
        except (TypeError, ValueError) as exc:
            raise WorkspaceViolation(
                "Workspace action must contain seven finite physical joint values"
            ) from exc
        if action.shape != (7,):
            raise WorkspaceViolation("Workspace action must have shape (7,)")
        if not np.all(np.isfinite(action)):
            raise WorkspaceViolation("Workspace action joint values must be finite")

        try:
            xyz = np.asarray(self.fk_deg_to_xyz_mm(action.copy()), dtype=float)
        except WorkspaceViolation:
            raise
        except Exception as exc:
            raise WorkspaceViolation(f"Workspace forward kinematics failed: {exc}") from exc
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise WorkspaceViolation(
                "Workspace forward kinematics must return three finite XYZ millimeters"
            )

        if not _point_in_convex_polygon(xyz[:2], self.polygon_xy_mm):
            raise WorkspaceViolation(
                f"Predicted tip XY is outside calibrated workspace: {xyz[:2].tolist()}"
            )
        if (
            xyz[2] < self.z_min_mm - POLYGON_TOLERANCE
            or xyz[2] > self.z_max_mm + POLYGON_TOLERANCE
        ):
            raise WorkspaceViolation(
                "Predicted tip Z is outside configured workspace range: "
                f"{float(xyz[2])} not in [{self.z_min_mm}, {self.z_max_mm}] mm"
            )


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        raw_text = path.read_text(encoding="utf-8")
        value = (
            json.loads(raw_text)
            if path.suffix.lower() == ".json"
            else yaml.safe_load(raw_text)
        )
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label.capitalize()} must be an object")
    return value


def _aware_utc(value: object, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _validate_freshness(
    timestamp: object,
    *,
    now: datetime,
    max_age_s: float,
) -> None:
    max_age = _finite_float(max_age_s, "Maximum calibration age")
    if max_age < 0:
        raise ValueError("Maximum calibration age must be nonnegative")
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("Workspace calibration timestamp is missing")
    try:
        normalized = timestamp.strip().replace("Z", "+00:00")
        updated_at = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("Workspace calibration timestamp is not parseable") from exc
    if updated_at.tzinfo is None:
        raise ValueError("Workspace calibration timestamp must include a timezone")
    age_s = (now - updated_at.astimezone(timezone.utc)).total_seconds()
    if age_s < -MAX_FUTURE_SKEW_S:
        raise ValueError("Workspace calibration timestamp is in the future")
    if age_s > max_age:
        raise ValueError("Workspace calibration is stale")


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_finite(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Workspace calibration {label} is malformed") from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"Workspace calibration {label} is malformed or nonfinite")
    return result


def _plane_corners(calibration: Mapping[str, Any]) -> np.ndarray:
    aruco = calibration.get("aruco")
    if not isinstance(aruco, Mapping):
        raise ValueError("Workspace calibration plane corners are missing")
    ids = aruco.get("ids")
    plane = aruco.get("plane_mm")
    if (
        not isinstance(ids, list)
        or len(ids) != 4
        or len(set(str(value) for value in ids)) != 4
        or not isinstance(plane, Mapping)
    ):
        raise ValueError("Workspace calibration must define four plane corners")
    try:
        corners = np.asarray([plane[str(marker_id)] for marker_id in ids], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Workspace calibration plane corners are malformed") from exc
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise ValueError("Workspace calibration plane corners are malformed or nonfinite")
    return corners


def _validate_zone_dimensions(corners: np.ndarray, width: float, depth: float) -> None:
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    expected = np.asarray((width, depth, width, depth), dtype=float)
    swapped = np.asarray((depth, width, depth, width), dtype=float)
    if not (
        np.allclose(edges, expected, rtol=1e-6, atol=DIMENSION_TOLERANCE_MM)
        or np.allclose(edges, swapped, rtol=1e-6, atol=DIMENSION_TOLERANCE_MM)
    ):
        raise ValueError(
            "Configured workspace zone dimensions are inconsistent with calibration plane corners"
        )


def _validate_convex_polygon(polygon: np.ndarray) -> None:
    if polygon.shape != (4, 2) or not np.all(np.isfinite(polygon)):
        raise ValueError("Calibrated workspace polygon is malformed")
    edge_a = np.roll(polygon, -1, axis=0) - polygon
    edge_b = np.roll(polygon, -2, axis=0) - np.roll(polygon, -1, axis=0)
    crosses = _cross_2d(edge_a, edge_b)
    if not (
        np.all(crosses > POLYGON_TOLERANCE)
        or np.all(crosses < -POLYGON_TOLERANCE)
    ):
        raise ValueError("Calibrated workspace polygon must be nondegenerate and convex")


def _point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = point - polygon
    crosses = _cross_2d(edges, offsets)
    return bool(
        np.all(crosses >= -POLYGON_TOLERANCE)
        or np.all(crosses <= POLYGON_TOLERANCE)
    )


def _cross_2d(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return row-wise 2D cross products without NumPy's deprecated 2D cross."""

    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _make_sdk_fk(arm_config: object) -> Callable[[np.ndarray], np.ndarray]:
    """Create a lazy SDK FK callable; this path never connects to the arm."""

    sdk_repo = Path(getattr(arm_config, "sdk_repo")).expanduser().resolve()
    if not sdk_repo.is_dir():
        raise FileNotFoundError(f"reBot SDK not found at {sdk_repo}")
    sdk_root = str(sdk_repo)
    if sdk_root not in sys.path:
        sys.path.insert(0, sdk_root)

    def fk_deg_to_xyz_mm(action_deg: np.ndarray) -> np.ndarray:
        # Pinocchio and the SDK remain lazy until a live validation is requested.
        from reBotArm_control_py.kinematics import joint_to_pose

        physical_six_rad = np.radians(np.asarray(action_deg[:6], dtype=float))
        position_m, _ = joint_to_pose(physical_six_rad)
        return np.asarray(position_m, dtype=float) * 1000.0

    return fk_deg_to_xyz_mm
