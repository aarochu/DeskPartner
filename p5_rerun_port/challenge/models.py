"""Typed, immutable records shared by the curation pipeline."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, TypeAlias


SourceRole: TypeAlias = Literal["success", "failure"]
TaskKey: TypeAlias = Literal["single_can", "two_can"]
Verdict: TypeAlias = Literal["PASS", "REVIEW", "REJECT"]


@dataclass(frozen=True)
class EpisodeIdentity:
    repo_id: str
    revision: str
    source_key: str

    @property
    def canonical(self) -> str:
        return f"{self.repo_id}@{self.revision}:{self.source_key}"


@dataclass(frozen=True)
class InventoryRow:
    identity: EpisodeIdentity
    role: SourceRole
    task_key: TaskKey
    source_path: str
    episode_index: int | None
    attempt_id: str | None
    frame_count: int
    captured_at: str


@dataclass(frozen=True)
class SourceSpec:
    repo_id: str
    revision: str
    role: SourceRole
    task_key: TaskKey
    expected_items: int
    expected_frames: int | None


@dataclass(frozen=True)
class RobotProfileConfig:
    """Authenticated robot coordinates and physical action limits."""

    path: Path
    file_sha256: str
    semantic_sha256: str
    profile_id: str
    profile_version: int
    coordinate_frame: str
    feature_names: tuple[str, ...]
    action_limits_deg: tuple[tuple[float, float], ...]
    normalization_spans_deg: tuple[float, ...]
    collection_max_step_deg: float

    def limit_violations(
        self,
        values: Sequence[float],
        *,
        tolerance_deg: float = 0.0,
    ) -> tuple[int, ...]:
        """Return joints outside inclusive limits, optionally widened for observations."""

        if len(values) != len(self.action_limits_deg):
            raise ValueError(f"expected {len(self.action_limits_deg)} joint values")
        if not math.isfinite(tolerance_deg) or tolerance_deg < 0:
            raise ValueError("limit tolerance must be finite and non-negative")
        violations: list[int] = []
        for index, (value, (lower, upper)) in enumerate(
            zip(values, self.action_limits_deg, strict=True)
        ):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("joint values must be finite")
            if number < lower - tolerance_deg or number > upper + tolerance_deg:
                violations.append(index)
        return tuple(violations)


@dataclass(frozen=True)
class QualityConfig:
    position_unit: str
    sample_period_s: float
    max_state_age_ns: int
    max_camera_age_ns: int
    observed_state_limit_tolerance_deg: float
    stationary_epsilon_deg_per_frame: float
    saturation_band_deg: float
    derivative_normalization_span_deg: tuple[float, ...]
    gripper_open_threshold_deg: float
    gripper_closed_threshold_deg: float
    lag_search_frames: int
    lag_min_normalized_motion_range: float
    lag_score_round_decimals: int
    calibration_fraction: float
    upper_quantile: float
    lower_quantile: float
    mad_multiplier: float
