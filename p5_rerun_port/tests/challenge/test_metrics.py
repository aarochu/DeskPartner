from __future__ import annotations

from dataclasses import fields
import math
from pathlib import Path

import numpy as np
import pytest

from p5_rerun_port.challenge.alignment import AlignedEpisode
from p5_rerun_port.challenge.canonical import CanonicalArtifact
from p5_rerun_port.challenge.config import ChallengeConfig
from p5_rerun_port.challenge.metrics import EpisodeMetrics, measure_aligned, measure_artifact


CONFIG = Path("config/rerun_query_challenge.yaml")
JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
    "wrist_yaw", "wrist_roll", "gripper",
)


def _episode(
    action: np.ndarray,
    state: np.ndarray,
    *,
    frame: np.ndarray | None = None,
    state_present: np.ndarray | None = None,
    front_present: np.ndarray | None = None,
    expected: int | None = None,
) -> AlignedEpisode:
    n = len(action)
    frame = np.arange(n, dtype=np.int64) if frame is None else frame
    times = np.rint(np.arange(n) * 1_000_000_000 / 30).astype(np.int64)
    state_present = np.ones(n, dtype=bool) if state_present is None else state_present
    front_present = np.ones(n, dtype=bool) if front_present is None else front_present
    return AlignedEpisode(
        identity="repo@" + "a" * 40 + ":0",
        segment_id="segment",
        task_key="single_can",
        expected_sample_count=n if expected is None else expected,
        joint_names=JOINTS,
        frame=frame,
        action_time_ns=times,
        action_present=np.ones(n, dtype=bool),
        action=np.asarray(action, dtype=np.float64),
        state_time_ns=np.where(state_present, times, -1),
        state_age_ns=np.where(state_present, 0, -1),
        state_present=state_present,
        state=np.asarray(state, dtype=np.float64),
        camera_time_ns={"front": np.where(front_present, times, -1), "side": times.copy()},
        camera_age_ns={"front": np.where(front_present, 0, -1), "side": np.zeros(n, dtype=np.int64)},
        camera_present={"front": front_present, "side": np.ones(n, dtype=bool)},
        query_columns=(),
    )


def test_exact_tracking_metrics_derivatives_and_units() -> None:
    config = ChallengeConfig.load(CONFIG)
    n = 40
    offsets = np.arange(1, 8, dtype=np.float64)
    action = np.tile(np.asarray([0, -80, -100, 0, 0, 0, -100.0]), (n, 1))
    state = action + offsets

    metrics = measure_aligned(_episode(action, state), config)

    assert metrics.per_joint["tracking_mae_deg"] == pytest.approx(offsets)
    assert metrics.per_joint["tracking_max_abs_error_deg"] == pytest.approx(offsets)
    assert metrics.values["tracking_rms_deg"] == pytest.approx(math.sqrt(np.mean(offsets**2)))
    assert metrics.values["action_jerk_max_deg_s3"] == 0.0
    assert metrics.values["state_jerk_p95_deg_s3"] == 0.0
    assert metrics.metric_units["tracking_mae_deg"] == "deg"
    assert metrics.metric_units["action_jerk_max_normalized_s3"] == "s^-3"
    assert metrics.hard_reasons == ()


def test_lag_plus_two_means_state_trails_action_and_constant_is_ineligible() -> None:
    config = ChallengeConfig.load(CONFIG)
    n = 90
    action = np.zeros((n, 7), dtype=np.float64)
    action[:, 0] = 30 * np.sin(np.linspace(0, 8, n))
    action[:, 1] = -80 + 20 * np.cos(np.linspace(0, 5, n))
    state = np.empty_like(action)
    state[:2] = action[0]
    state[2:] = action[:-2]

    metrics = measure_aligned(_episode(action, state), config)
    assert metrics.values["best_lag_frames"] == 2
    assert metrics.values["lag_status"] == "ok"

    constant = np.tile(np.asarray([0, -80, -100, 0, 0, 0, -100.0]), (n, 1))
    metrics = measure_aligned(_episode(constant, constant), config)
    assert metrics.values["best_lag_frames"] is None
    assert metrics.values["lag_status"] == "insufficient_motion"
    assert metrics.values["lag_corrected_rms_deg"] == metrics.values["zero_lag_rms_deg"]


def test_lag_tie_prefers_smaller_magnitude_then_negative_sign() -> None:
    config = ChallengeConfig.load(CONFIG)
    n = 90
    action = np.zeros((n, 7), dtype=np.float64)
    action[:, 0] = np.tile([30.0, -30.0], n // 2)
    state = -action

    metrics = measure_aligned(_episode(action, state), config)

    assert metrics.values["best_lag_frames"] == -1


def test_stationary_and_saturation_boundaries_are_inclusive_and_units_are_total() -> None:
    config = ChallengeConfig.load(CONFIG)
    lower = np.asarray(config.robot_profile.action_limits_deg)[:, 0]
    action = np.vstack([lower, lower + 0.25])
    metrics = measure_aligned(_episode(action, action), config)

    assert metrics.values["action_stationary_fraction"] == 1.0
    assert metrics.per_joint["action_saturation_fraction"] == pytest.approx([1.0] * 7)
    assert metrics.values["action_velocity_max_deg_s"] == pytest.approx(7.5)
    assert metrics.per_joint["action_velocity_max_normalized_per_joint"][0] == pytest.approx(
        7.5 / 290
    )
    assert (set(metrics.values) | set(metrics.per_joint)) <= set(metrics.metric_units)


def test_hard_reason_order_camera_gaps_and_gripper_hysteresis() -> None:
    config = ChallengeConfig.load(CONFIG)
    n = 8
    action = np.tile(np.asarray([0, -80, -100, 0, 0, 0, -100.0]), (n, 1))
    action[:, 6] = [0, -100, -220, -150, -50, -150, -220, -50]
    state = action.copy()
    state[2, 3] = np.nan
    frame = np.asarray([0, 1, 4, 5, 6, 7, 8, 9])
    front = np.asarray([True, False, False, True, False, True, True, True])

    metrics = measure_aligned(_episode(action, state, frame=frame, front_present=front), config)

    assert metrics.hard_reasons == (
        "FRAME_INDEX_NONCONTIGUOUS",
        "STATE_NONFINITE",
        "CAMERA_FRONT_MISSING_OR_STALE",
    )
    assert metrics.values["camera_front_gap_count"] == 2
    assert metrics.values["gripper_close_transition_count"] == 2
    assert metrics.values["gripper_open_transition_count"] == 2


def test_action_limits_are_exact_and_state_has_separate_quarter_degree_tolerance() -> None:
    config = ChallengeConfig.load(CONFIG)
    limits = np.asarray(config.robot_profile.action_limits_deg)
    lower, upper = limits[:, 0], limits[:, 1]
    action = np.vstack([lower, upper])
    state = np.vstack([lower - 0.25, upper + 0.25])
    assert measure_aligned(_episode(action, state), config).hard_reasons == ()

    action[0, 0] = np.nextafter(lower[0], -np.inf)
    assert "ACTION_LIMIT" in measure_aligned(_episode(action, state), config).hard_reasons
    action[0, 0] = lower[0]
    state[0, 6] = np.nextafter(lower[6] - 0.25, -np.inf)
    assert "STATE_LIMIT" in measure_aligned(_episode(action, state), config).hard_reasons


def test_rejected_artifact_yields_one_zero_sample_row_without_open(monkeypatch) -> None:
    config = ChallengeConfig.load(CONFIG)
    artifact = CanonicalArtifact("identity", "rejected", None, None, 12, ("RRD_VERIFY_FAILED",))

    def forbidden(*args, **kwargs):
        raise AssertionError("must not open rejected artifact")

    monkeypatch.setattr("p5_rerun_port.challenge.metrics.open_dataset_server", forbidden)
    metrics = measure_artifact(artifact, config)
    assert metrics.sample_count == 0
    assert metrics.hard_reasons == ("RRD_VERIFY_FAILED",)
    assert metrics.values == {}


def test_metric_types_are_label_blind() -> None:
    forbidden = {"role", "disposition", "failure_label", "training_eligible"}
    assert forbidden.isdisjoint(field.name for field in fields(AlignedEpisode))
    assert forbidden.isdisjoint(field.name for field in fields(EpisodeMetrics))
