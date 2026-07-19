"""Loading and validation for the checked-in Rerun Query source lock."""

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Any, Mapping, cast

import yaml

from .models import QualityConfig, SourceRole, SourceSpec, TaskKey


class ConfigError(ValueError):
    """The checked-in challenge source lock is malformed or unsafe."""


_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
    "wrist_yaw", "wrist_roll", "gripper",
)
_CAMERA_KEYS = ("front", "side")
_ROBOT_TYPE = "seeed_b601_dm_follower"


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be a mapping")
    return cast(Mapping[str, Any], value)


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-blank string")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{field} must be an integer of at least one")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{field} must be finite")
    return number


@dataclass(frozen=True)
class ChallengeConfig:
    sources: tuple[SourceSpec, ...]
    robot_type: str
    fps: int
    joint_names: tuple[str, ...]
    camera_keys: tuple[str, ...]
    destination_repo: str
    quality: QualityConfig

    @classmethod
    def load(cls, path: Path) -> "ChallengeConfig":
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ConfigError(f"unable to load {path}: {error}") from error
        data = _mapping(document, "root")
        if data.get("schema_version") != 1 or isinstance(data.get("schema_version"), bool):
            raise ConfigError("schema_version must be 1")

        raw_sources = data.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ConfigError("sources must be a non-empty list")
        sources = tuple(cls._source(item, index) for index, item in enumerate(raw_sources))
        repo_ids = [source.repo_id for source in sources]
        if len(repo_ids) != len(set(repo_ids)):
            raise ConfigError("sources must not contain duplicate repo_id values")

        robot_type = _string(data.get("robot_type"), "robot_type")
        if robot_type != _ROBOT_TYPE:
            raise ConfigError(f"robot_type must be {_ROBOT_TYPE}")
        fps = _positive_int(data.get("fps"), "fps")
        if fps != 30:
            raise ConfigError("fps must be 30")
        joint_names = cls._ordered_strings(data.get("joint_names"), "joint_names")
        if joint_names != _JOINT_NAMES:
            raise ConfigError("joint_names must match the authenticated reBot order")
        camera_keys = cls._ordered_strings(data.get("camera_keys"), "camera_keys")
        if camera_keys != _CAMERA_KEYS:
            raise ConfigError("camera_keys must be ordered as ('front', 'side')")

        destination_repo = _string(data.get("destination_repo"), "destination_repo")
        quality = cls._quality(data.get("quality"))
        return cls(sources, robot_type, fps, joint_names, camera_keys, destination_repo, quality)

    @staticmethod
    def _ordered_strings(value: Any, field: str) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise ConfigError(f"{field} must be a list")
        return tuple(_string(item, f"{field}[{index}]") for index, item in enumerate(value))

    @staticmethod
    def _source(value: Any, index: int) -> SourceSpec:
        data = _mapping(value, f"sources[{index}]")
        repo_id = _string(data.get("repo_id"), f"sources[{index}].repo_id")
        revision = _string(data.get("revision"), f"sources[{index}].revision")
        if not _REVISION_RE.fullmatch(revision):
            raise ConfigError(f"sources[{index}].revision must be a 40-character lowercase SHA")
        role = data.get("role")
        if role not in ("success", "failure"):
            raise ConfigError(f"sources[{index}].role must be success or failure")
        task_key = data.get("task_key")
        if task_key not in ("single_can", "two_can"):
            raise ConfigError(f"sources[{index}].task_key must be single_can or two_can")
        expected_frames = data.get("expected_frames")
        if expected_frames is not None:
            expected_frames = _positive_int(expected_frames, f"sources[{index}].expected_frames")
        return SourceSpec(
            repo_id=repo_id,
            revision=revision,
            role=cast(SourceRole, role),
            task_key=cast(TaskKey, task_key),
            expected_items=_positive_int(data.get("expected_items"), f"sources[{index}].expected_items"),
            expected_frames=expected_frames,
        )

    @staticmethod
    def _quality(value: Any) -> QualityConfig:
        data = _mapping(value, "quality")
        max_state_age_ms = _number(data.get("max_state_age_ms"), "quality.max_state_age_ms")
        max_camera_age_ms = _number(data.get("max_camera_age_ms"), "quality.max_camera_age_ms")
        lag_search_frames = _positive_int(data.get("lag_search_frames"), "quality.lag_search_frames")
        calibration_fraction = _number(data.get("calibration_fraction"), "quality.calibration_fraction")
        upper_quantile = _number(data.get("upper_quantile"), "quality.upper_quantile")
        lower_quantile = _number(data.get("lower_quantile"), "quality.lower_quantile")
        mad_multiplier = _number(data.get("mad_multiplier"), "quality.mad_multiplier")
        if max_state_age_ms <= 0 or max_camera_age_ms <= 0 or mad_multiplier <= 0:
            raise ConfigError("quality age limits and mad_multiplier must be positive")
        if not 0 < calibration_fraction <= 1:
            raise ConfigError("quality.calibration_fraction must be in (0, 1]")
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ConfigError("quality quantiles must satisfy 0 <= lower < upper <= 1")
        return QualityConfig(
            max_state_age_ms=max_state_age_ms,
            max_camera_age_ms=max_camera_age_ms,
            lag_search_frames=lag_search_frames,
            calibration_fraction=calibration_fraction,
            upper_quantile=upper_quantile,
            lower_quantile=lower_quantile,
            mad_multiplier=mad_multiplier,
        )
