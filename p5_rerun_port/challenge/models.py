"""Typed, immutable records shared by the curation pipeline."""

from dataclasses import dataclass
from typing import Literal, TypeAlias


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
class QualityConfig:
    max_state_age_ms: float
    max_camera_age_ms: float
    lag_search_frames: int
    calibration_fraction: float
    upper_quantile: float
    lower_quantile: float
    mad_multiplier: float
