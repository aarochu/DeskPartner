import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from p5_rerun_port.challenge.config import ChallengeConfig, ConfigError


CONFIG = Path("config/rerun_query_challenge.yaml")
PROFILE = Path("rebot_operator_kit/config/training_profile.json")

EXPECTED_LIMITS = (
    (-145.0, 145.0),
    (-170.0, 0.0),
    (-200.0, 0.0),
    (-80.0, 90.0),
    (-90.0, 90.0),
    (-90.0, 90.0),
    (-270.0, 0.0),
)


def test_checked_in_config_locks_real_sources_and_robot_contract() -> None:
    config = ChallengeConfig.load(CONFIG)
    assert [(s.repo_id, s.revision, s.role, s.expected_items) for s in config.sources] == [
        ("Cornerf/rebot-can-sort-stage1-v1-smoke", "74d1f300786d58b4f6f55e1798cbb1a1a48f5409", "success", 52),
        ("Cornerf/rebot-two-can-recycle-v2-smoke", "778d0bf5de1096a80b1cf355073e369faa1409da", "success", 25),
        ("Cornerf/rebot-can-sort-stage1-v1-failed", "4952b618a23f8f2e5b09f736cea0a490c62e57b4", "failure", 19),
        ("Cornerf/rebot-two-can-recycle-v2-failed", "2d9ea53cf8f4835fcfc1656b23d56308696b3e5b", "failure", 6),
    ]
    assert config.joint_names == (
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
        "wrist_yaw", "wrist_roll", "gripper",
    )
    assert config.camera_keys == ("front", "side")
    assert config.destination_repo == "Cornerf/rebot-cansort-rerun-curated"
    assert config.robot_profile.path == PROFILE.resolve()
    assert config.robot_profile.file_sha256 == (
        "82363ef023de870db5b22173837fc95fef2f8adb426bb1d65ed46eb96ace31b6"
    )
    assert config.robot_profile.semantic_sha256 == (
        "9370a9fc5d90df7ffb0174c0a035fcf043911c04bbef6820c9bf36ab04131208"
    )
    assert config.robot_profile.profile_id == "rebot-b601-dm-follower1-native7d-v5"
    assert config.robot_profile.profile_version == 5
    assert config.robot_profile.coordinate_frame == (
        "follower_degrees_after_direction_limits_and_step_cap"
    )
    assert config.robot_profile.feature_names == tuple(
        f"{name}.pos" for name in config.joint_names
    )
    assert config.robot_profile.action_limits_deg == EXPECTED_LIMITS
    assert config.robot_profile.normalization_spans_deg == (
        290.0, 170.0, 200.0, 170.0, 180.0, 180.0, 270.0
    )
    assert config.robot_profile.collection_max_step_deg == 33.6
    assert config.quality.position_unit == "degree"
    assert config.quality.sample_period_s == 1 / 30
    assert config.quality.max_state_age_ns == 100_000_000
    assert config.quality.max_camera_age_ns == 66_666_667
    assert config.quality.observed_state_limit_tolerance_deg == 0.25
    assert config.quality.stationary_epsilon_deg_per_frame == 0.25
    assert config.quality.saturation_band_deg == 0.25
    assert config.quality.derivative_normalization_span_deg == (
        290.0, 170.0, 200.0, 170.0, 180.0, 180.0, 270.0
    )
    assert config.quality.gripper_open_threshold_deg == -67.5
    assert config.quality.gripper_closed_threshold_deg == -202.5
    assert config.quality.lag_search_frames == 15
    assert config.quality.lag_min_normalized_motion_range == 0.01
    assert config.quality.lag_score_round_decimals == 12
    assert config.quality.calibration_fraction == 0.8
    assert config.quality.upper_quantile == 0.99
    assert config.quality.lower_quantile == 0.01
    assert config.quality.mad_multiplier == 5.0


def test_config_rejects_duplicate_repo_or_non_sha_revision(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nsources: []\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        ChallengeConfig.load(path)


def _config_document() -> dict[str, object]:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    profile = document["robot_profile"]
    assert isinstance(profile, dict)
    profile["path"] = str(PROFILE.resolve())
    return document


def _write_config(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "challenge.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("digest_key", ["file_sha256", "semantic_sha256"])
def test_config_rejects_robot_profile_digest_mismatch(
    digest_key: str, tmp_path: Path
) -> None:
    document = _config_document()
    profile = document["robot_profile"]
    assert isinstance(profile, dict)
    profile[digest_key] = "0" * 64

    with pytest.raises(ConfigError, match=digest_key):
        ChallengeConfig.load(_write_config(tmp_path, document))


def test_config_rejects_rehashed_profile_with_wrong_feature_contract(tmp_path: Path) -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    profile["coordinate_contract"]["joints"][0]["feature"] = "wrong.pos"
    profile_path = tmp_path / "profile.json"
    encoded = json.dumps(profile, indent=2).encode("utf-8")
    profile_path.write_bytes(encoded)

    document = _config_document()
    profile_lock = document["robot_profile"]
    assert isinstance(profile_lock, dict)
    profile_lock["path"] = str(profile_path)
    profile_lock["file_sha256"] = hashlib.sha256(encoded).hexdigest()
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    profile_lock["semantic_sha256"] = hashlib.sha256(canonical).hexdigest()

    with pytest.raises(ConfigError, match="feature"):
        ChallengeConfig.load(_write_config(tmp_path, document))


def test_action_and_observed_state_limit_boundaries_are_inclusive() -> None:
    config = ChallengeConfig.load(CONFIG)
    robot = config.robot_profile
    tolerance = config.quality.observed_state_limit_tolerance_deg

    for joint, (lower, upper) in enumerate(EXPECTED_LIMITS):
        at_lower = [0.0] * 7
        at_lower[joint] = lower
        at_upper = [0.0] * 7
        at_upper[joint] = upper
        assert robot.limit_violations(at_lower) == ()
        assert robot.limit_violations(at_upper) == ()

        below_action = at_lower.copy()
        below_action[joint] = np.nextafter(lower, -np.inf)
        above_action = at_upper.copy()
        above_action[joint] = np.nextafter(upper, np.inf)
        assert robot.limit_violations(below_action) == (joint,)
        assert robot.limit_violations(above_action) == (joint,)

        at_state_lower = at_lower.copy()
        at_state_lower[joint] = lower - tolerance
        at_state_upper = at_upper.copy()
        at_state_upper[joint] = upper + tolerance
        assert robot.limit_violations(at_state_lower, tolerance_deg=tolerance) == ()
        assert robot.limit_violations(at_state_upper, tolerance_deg=tolerance) == ()

        below_state = at_state_lower.copy()
        below_state[joint] = np.nextafter(lower - tolerance, -np.inf)
        above_state = at_state_upper.copy()
        above_state[joint] = np.nextafter(upper + tolerance, np.inf)
        assert robot.limit_violations(below_state, tolerance_deg=tolerance) == (joint,)
        assert robot.limit_violations(above_state, tolerance_deg=tolerance) == (joint,)


def _huggingface_cache_root() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home.expanduser() / "hub"


def test_pinned_success_parquet_profile_regression_when_cached() -> None:
    config = ChallengeConfig.load(CONFIG)
    cache_root = _huggingface_cache_root()
    episode_files: list[Path] = []
    for source in config.sources:
        if source.role != "success":
            continue
        owner, repo = source.repo_id.split("/", 1)
        snapshot = (
            cache_root
            / f"datasets--{owner}--{repo}"
            / "snapshots"
            / source.revision
        )
        files = sorted(snapshot.glob("data/**/*.parquet"))
        if len(files) != source.expected_items:
            pytest.skip("complete pinned success Parquet cache is not available")
        episode_files.extend(files)

    robot = config.robot_profile
    tolerance = config.quality.observed_state_limit_tolerance_deg
    frame_count = 0
    action_reject_episodes = 0
    exact_state_reject_episodes = 0
    tolerant_state_reject_episodes = 0
    for path in episode_files:
        frame = pd.read_parquet(path, columns=["action", "observation.state"])
        actions = np.stack(frame["action"].to_numpy()).astype(np.float64)
        states = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
        frame_count += len(frame)
        action_reject_episodes += any(
            robot.limit_violations(row) for row in actions
        )
        exact_state_reject_episodes += any(
            robot.limit_violations(row) for row in states
        )
        tolerant_state_reject_episodes += any(
            robot.limit_violations(row, tolerance_deg=tolerance) for row in states
        )

    assert len(episode_files) == 77
    assert frame_count == 51_207
    assert action_reject_episodes == 0
    assert exact_state_reject_episodes == 54
    assert tolerant_state_reject_episodes == 0
