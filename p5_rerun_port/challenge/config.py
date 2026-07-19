"""Loading and validation for the checked-in Rerun Query source lock."""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import yaml

from .models import QualityConfig, RobotProfileConfig, SourceRole, SourceSpec, TaskKey


class ConfigError(ValueError):
    """The checked-in challenge source lock is malformed or unsafe."""


_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
    "wrist_yaw", "wrist_roll", "gripper",
)
_CAMERA_KEYS = ("front", "side")
_ROBOT_TYPE = "seeed_b601_dm_follower"
_PROFILE_ID = "rebot-b601-dm-follower1-native7d-v5"
_PROFILE_VERSION = 5
_COORDINATE_FRAME = "follower_degrees_after_direction_limits_and_step_cap"


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


def _sha256(value: Any, field: str) -> str:
    digest = _string(value, field)
    if not _SHA256_RE.fullmatch(digest):
        raise ConfigError(f"{field} must be a 64-character lowercase SHA-256")
    return digest


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
    robot_profile: RobotProfileConfig
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
        robot_profile = cls._robot_profile(data.get("robot_profile"), path, robot_type)
        if tuple(name.removesuffix(".pos") for name in robot_profile.feature_names) != joint_names:
            raise ConfigError("robot profile feature order must match joint_names")
        quality = cls._quality(data.get("quality"), robot_profile)
        return cls(
            sources,
            robot_type,
            fps,
            joint_names,
            camera_keys,
            destination_repo,
            robot_profile,
            quality,
        )

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
    def _robot_profile(
        value: Any,
        config_path: Path,
        robot_type: str,
    ) -> RobotProfileConfig:
        data = _mapping(value, "robot_profile")
        configured_path = Path(_string(data.get("path"), "robot_profile.path")).expanduser()
        if configured_path.is_absolute():
            profile_path = configured_path
        else:
            profile_path = (config_path.resolve().parent.parent / configured_path).resolve()
        expected_file_sha = _sha256(data.get("file_sha256"), "robot_profile.file_sha256")
        expected_semantic_sha = _sha256(
            data.get("semantic_sha256"), "robot_profile.semantic_sha256"
        )
        try:
            raw = profile_path.read_bytes()
        except OSError as error:
            raise ConfigError(f"unable to load robot profile {profile_path}: {error}") from error
        file_sha = hashlib.sha256(raw).hexdigest()
        if file_sha != expected_file_sha:
            raise ConfigError("robot_profile.file_sha256 does not authenticate profile bytes")
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ConfigError("robot profile must be valid UTF-8 JSON") from error
        profile = _mapping(document, "robot profile")
        canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
        semantic_sha = hashlib.sha256(canonical).hexdigest()
        if semantic_sha != expected_semantic_sha:
            raise ConfigError("robot_profile.semantic_sha256 does not authenticate profile semantics")

        if profile.get("schema_version") != 1 or isinstance(profile.get("schema_version"), bool):
            raise ConfigError("robot profile schema_version must be 1")
        profile_id = _string(profile.get("profile_id"), "robot profile.profile_id")
        if profile_id != _PROFILE_ID:
            raise ConfigError(f"robot profile.profile_id must be {_PROFILE_ID}")
        profile_version = _positive_int(profile.get("profile_version"), "robot profile.profile_version")
        if profile_version != _PROFILE_VERSION:
            raise ConfigError(f"robot profile.profile_version must be {_PROFILE_VERSION}")

        calibration = _mapping(profile.get("calibration"), "robot profile.calibration")
        follower = _mapping(calibration.get("follower"), "robot profile.calibration.follower")
        if _string(follower.get("type"), "robot profile.calibration.follower.type") != robot_type:
            raise ConfigError("robot profile follower type must match robot_type")
        coordinate = _mapping(profile.get("coordinate_contract"), "robot profile.coordinate_contract")
        coordinate_frame = _string(coordinate.get("frame"), "robot profile.coordinate_contract.frame")
        if coordinate_frame != _COORDINATE_FRAME:
            raise ConfigError(f"robot profile coordinate frame must be {_COORDINATE_FRAME}")
        action_dimension = _positive_int(
            coordinate.get("action_dimension"), "robot profile.coordinate_contract.action_dimension"
        )
        raw_joints = coordinate.get("joints")
        if not isinstance(raw_joints, list) or len(raw_joints) != action_dimension:
            raise ConfigError("robot profile joints must match action_dimension")

        features: list[str] = []
        limits: list[tuple[float, float]] = []
        for index, item in enumerate(raw_joints):
            joint = _mapping(item, f"robot profile joints[{index}]")
            name = _string(joint.get("name"), f"robot profile joints[{index}].name")
            feature = _string(joint.get("feature"), f"robot profile joints[{index}].feature")
            if feature != f"{name}.pos":
                raise ConfigError(f"robot profile joints[{index}].feature must be {name}.pos")
            raw_limit = joint.get("soft_limit_degrees")
            if not isinstance(raw_limit, list) or len(raw_limit) != 2:
                raise ConfigError(f"robot profile joints[{index}].soft_limit_degrees must have two values")
            lower = _number(raw_limit[0], f"robot profile joints[{index}].soft_limit_degrees[0]")
            upper = _number(raw_limit[1], f"robot profile joints[{index}].soft_limit_degrees[1]")
            if lower >= upper:
                raise ConfigError(f"robot profile joints[{index}] limits must increase")
            features.append(feature)
            limits.append((lower, upper))

        collection = _mapping(profile.get("collection_defaults"), "robot profile.collection_defaults")
        collection_fps = _positive_int(collection.get("fps"), "robot profile.collection_defaults.fps")
        if collection_fps != 30:
            raise ConfigError("robot profile collection FPS must be 30")
        max_step = _number(
            collection.get("max_step"), "robot profile.collection_defaults.max_step"
        )
        if max_step <= 0:
            raise ConfigError("robot profile collection max_step must be positive")

        return RobotProfileConfig(
            path=profile_path,
            file_sha256=file_sha,
            semantic_sha256=semantic_sha,
            profile_id=profile_id,
            profile_version=profile_version,
            coordinate_frame=coordinate_frame,
            feature_names=tuple(features),
            action_limits_deg=tuple(limits),
            normalization_spans_deg=tuple(upper - lower for lower, upper in limits),
            collection_max_step_deg=max_step,
        )

    @staticmethod
    def _quality(value: Any, robot_profile: RobotProfileConfig) -> QualityConfig:
        data = _mapping(value, "quality")
        position_unit = _string(data.get("position_unit"), "quality.position_unit")
        sample_period_s = _number(data.get("sample_period_s"), "quality.sample_period_s")
        max_state_age_ns = _positive_int(data.get("max_state_age_ns"), "quality.max_state_age_ns")
        max_camera_age_ns = _positive_int(data.get("max_camera_age_ns"), "quality.max_camera_age_ns")
        state_tolerance = _number(
            data.get("observed_state_limit_tolerance_deg"),
            "quality.observed_state_limit_tolerance_deg",
        )
        stationary_epsilon = _number(
            data.get("stationary_epsilon_deg_per_frame"),
            "quality.stationary_epsilon_deg_per_frame",
        )
        saturation_band = _number(data.get("saturation_band_deg"), "quality.saturation_band_deg")
        spans = tuple(
            _number(item, f"quality.derivative_normalization_span_deg[{index}]")
            for index, item in enumerate(
                data.get("derivative_normalization_span_deg")
                if isinstance(data.get("derivative_normalization_span_deg"), list)
                else ()
            )
        )
        gripper_open = _number(
            data.get("gripper_open_threshold_deg"), "quality.gripper_open_threshold_deg"
        )
        gripper_closed = _number(
            data.get("gripper_closed_threshold_deg"), "quality.gripper_closed_threshold_deg"
        )
        lag_search_frames = _positive_int(data.get("lag_search_frames"), "quality.lag_search_frames")
        lag_motion = _number(
            data.get("lag_min_normalized_motion_range"),
            "quality.lag_min_normalized_motion_range",
        )
        lag_round = _positive_int(
            data.get("lag_score_round_decimals"), "quality.lag_score_round_decimals"
        )
        calibration_fraction = _number(data.get("calibration_fraction"), "quality.calibration_fraction")
        upper_quantile = _number(data.get("upper_quantile"), "quality.upper_quantile")
        lower_quantile = _number(data.get("lower_quantile"), "quality.lower_quantile")
        mad_multiplier = _number(data.get("mad_multiplier"), "quality.mad_multiplier")
        if position_unit != "degree":
            raise ConfigError("quality.position_unit must be degree")
        if sample_period_s != 1.0 / 30.0:
            raise ConfigError("quality.sample_period_s must be the nominal 30 FPS period")
        if max_state_age_ns != 100_000_000 or max_camera_age_ns != 66_666_667:
            raise ConfigError("quality age limits must match the authenticated latest-at contract")
        if state_tolerance != 0.25 or stationary_epsilon != 0.25 or saturation_band != 0.25:
            raise ConfigError("quality degree tolerances must match the authenticated contract")
        if spans != robot_profile.normalization_spans_deg:
            raise ConfigError("quality derivative spans must match robot profile limits")
        if gripper_open != -67.5 or gripper_closed != -202.5:
            raise ConfigError("quality gripper thresholds must match authenticated hysteresis")
        if lag_search_frames != 15 or lag_motion != 0.01 or lag_round != 12:
            raise ConfigError("quality lag settings must match the authenticated contract")
        if mad_multiplier != 5.0:
            raise ConfigError("quality.mad_multiplier must be 5.0")
        if not 0 < calibration_fraction <= 1:
            raise ConfigError("quality.calibration_fraction must be in (0, 1]")
        if lower_quantile != 0.01:
            raise ConfigError("quality.lower_quantile must be 0.01")
        if upper_quantile != 0.99:
            raise ConfigError("quality.upper_quantile must be 0.99")
        return QualityConfig(
            position_unit=position_unit,
            sample_period_s=sample_period_s,
            max_state_age_ns=max_state_age_ns,
            max_camera_age_ns=max_camera_age_ns,
            observed_state_limit_tolerance_deg=state_tolerance,
            stationary_epsilon_deg_per_frame=stationary_epsilon,
            saturation_band_deg=saturation_band,
            derivative_normalization_span_deg=spans,
            gripper_open_threshold_deg=gripper_open,
            gripper_closed_threshold_deg=gripper_closed,
            lag_search_frames=lag_search_frames,
            lag_min_normalized_motion_range=lag_motion,
            lag_score_round_decimals=lag_round,
            calibration_fraction=calibration_fraction,
            upper_quantile=upper_quantile,
            lower_quantile=lower_quantile,
            mad_multiplier=mad_multiplier,
        )
