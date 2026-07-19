"""Label-blind, deterministic quality metrics for aligned robot episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import AlignedEpisode, AlignmentError, extract_aligned_episode
from .canonical import CanonicalArtifact
from .config import ChallengeConfig
from p5_rerun_port.rerun_query import open_dataset_server


METRIC_SCHEMA_VERSION = "1"
HARD_REASON_ORDER = (
    "SEGMENT_COUNT",
    "IDENTITY_MISMATCH",
    "TASK_MISMATCH",
    "JOINT_SCHEMA_MISMATCH",
    "FRAME_COUNT_MISMATCH",
    "FRAME_INDEX_NONCONTIGUOUS",
    "ACTION_TIME_NONMONOTONIC",
    "ACTION_MISSING",
    "ACTION_DIMENSION",
    "ACTION_NONFINITE",
    "ACTION_LIMIT",
    "STATE_MISSING_OR_STALE",
    "STATE_DIMENSION",
    "STATE_NONFINITE",
    "STATE_LIMIT",
    "CAMERA_FRONT_MISSING_OR_STALE",
    "CAMERA_SIDE_MISSING_OR_STALE",
)


@dataclass(frozen=True)
class EpisodeMetrics:
    identity: str
    segment_id: str | None
    task_key: str | None
    metric_schema_version: str
    sample_count: int
    values: dict[str, float | int | str | None]
    per_joint: dict[str, tuple[float, ...]]
    metric_units: dict[str, str]
    hard_reasons: tuple[str, ...]


def _finite_rows(array: np.ndarray) -> np.ndarray:
    return np.isfinite(array).all(axis=1)


def _scalar_stats(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    return float(np.percentile(values, 95)), float(np.max(values))


def _joint_stats(values: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if values.ndim != 2 or values.shape[0] == 0:
        width = values.shape[1] if values.ndim == 2 else 7
        nan = tuple(float("nan") for _ in range(width))
        return nan, nan
    p95 = []
    maximum = []
    for column in values.T:
        finite = column[np.isfinite(column)]
        p95.append(float(np.percentile(finite, 95)) if len(finite) else float("nan"))
        maximum.append(float(np.max(finite)) if len(finite) else float("nan"))
    return tuple(p95), tuple(maximum)


def _differences(
    values: np.ndarray,
    valid: np.ndarray,
    frame: np.ndarray,
    order: int,
    dt: float,
) -> np.ndarray:
    if values.ndim != 2 or len(values) <= order:
        width = values.shape[1] if values.ndim == 2 else 7
        return np.empty((0, width), dtype=np.float64)
    difference = np.diff(values, n=order, axis=0) / (dt**order)
    keep = np.ones(len(difference), dtype=bool)
    for offset in range(order + 1):
        keep &= valid[offset : offset + len(difference)]
    # A derivative never bridges a missing or discontinuous frame.
    for offset in range(order):
        keep &= np.diff(frame)[offset : offset + len(difference)] == 1
    return difference[keep]


def _put_derivatives(
    values: dict[str, float | int | str | None],
    per_joint: dict[str, tuple[float, ...]],
    units: dict[str, str],
    prefix: str,
    trace: np.ndarray,
    valid: np.ndarray,
    frame: np.ndarray,
    spans: np.ndarray,
    dt: float,
) -> None:
    families = ((1, "velocity", "deg/s", "s^-1"), (2, "acceleration", "deg/s^2", "s^-2"), (3, "jerk", "deg/s^3", "s^-3"))
    for order, family, raw_unit, normalized_unit in families:
        raw = np.abs(_differences(trace, valid, frame, order, dt))
        normalized = raw / spans if len(raw) else raw
        raw_p95, raw_max = _scalar_stats(raw)
        norm_p95, norm_max = _scalar_stats(normalized)
        raw_joint_p95, raw_joint_max = _joint_stats(raw)
        norm_joint_p95, norm_joint_max = _joint_stats(normalized)
        entries = {
            f"{prefix}_{family}_p95_{'deg_s' if order == 1 else 'deg_s2' if order == 2 else 'deg_s3'}": (raw_p95, raw_unit),
            f"{prefix}_{family}_max_{'deg_s' if order == 1 else 'deg_s2' if order == 2 else 'deg_s3'}": (raw_max, raw_unit),
            f"{prefix}_{family}_p95_normalized_{'s1' if order == 1 else 's2' if order == 2 else 's3'}": (norm_p95, normalized_unit),
            f"{prefix}_{family}_max_normalized_{'s1' if order == 1 else 's2' if order == 2 else 's3'}": (norm_max, normalized_unit),
        }
        for name, (number, unit) in entries.items():
            values[name] = number
            units[name] = unit
        vectors = {
            f"{prefix}_{family}_p95_per_joint": (raw_joint_p95, raw_unit),
            f"{prefix}_{family}_max_per_joint": (raw_joint_max, raw_unit),
            f"{prefix}_{family}_p95_normalized_per_joint": (norm_joint_p95, normalized_unit),
            f"{prefix}_{family}_max_normalized_per_joint": (norm_joint_max, normalized_unit),
        }
        for name, (vector, unit) in vectors.items():
            per_joint[name] = vector
            units[name] = unit


def _missing_run_count(present: np.ndarray) -> int:
    missing = ~np.asarray(present, dtype=bool)
    return int(np.count_nonzero(missing & np.r_[True, ~missing[:-1]]))


def _age_metrics(
    values: dict[str, float | int | str | None],
    units: dict[str, str],
    prefix: str,
    ages_ns: np.ndarray,
    present: np.ndarray,
) -> None:
    ages_ms = np.asarray(ages_ns, dtype=np.float64)[present] / 1_000_000
    for suffix, function in (
        ("mean_age_ms", np.mean), ("p95_age_ms", lambda x: np.percentile(x, 95)), ("max_age_ms", np.max)
    ):
        name = f"{prefix}_{suffix}"
        values[name] = float(function(ages_ms)) if len(ages_ms) else float("nan")
        units[name] = "ms"


def _lag(
    action: np.ndarray,
    state: np.ndarray,
    valid: np.ndarray,
    spans: np.ndarray,
    config: ChallengeConfig,
    zero_raw: float,
    zero_normalized: float,
) -> tuple[int | None, str, float, float, tuple[float, ...]]:
    finite_action = action[valid]
    if not len(finite_action):
        return None, "insufficient_samples", zero_raw, zero_normalized, tuple([float("nan")] * 7)
    motion = np.ptp(finite_action, axis=0) / spans
    if float(np.sqrt(np.mean(motion**2))) < config.quality.lag_min_normalized_motion_range:
        error = action[valid] - state[valid]
        per_joint = tuple(np.sqrt(np.mean(error**2, axis=0))) if len(error) else tuple([float("nan")] * 7)
        return None, "insufficient_motion", zero_raw, zero_normalized, per_joint
    window = config.quality.lag_search_frames
    if len(action) < 2 * window + 1:
        return None, "insufficient_samples", zero_raw, zero_normalized, tuple([float("nan")] * 7)
    anchors = np.arange(window, len(action) - window)
    # Every lag is scored on exactly the same valid action anchors.
    common = valid[anchors].copy()
    for lag in range(-window, window + 1):
        common &= valid[anchors + lag]
    anchors = anchors[common]
    if not len(anchors):
        return None, "insufficient_samples", zero_raw, zero_normalized, tuple([float("nan")] * 7)
    candidates: list[tuple[tuple[float, int, int], int, float, float, tuple[float, ...]]] = []
    for lag in range(-window, window + 1):
        error = action[anchors] - state[anchors + lag]
        normalized_rms = float(np.sqrt(np.mean((error / spans) ** 2)))
        raw_rms = float(np.sqrt(np.mean(error**2)))
        per_joint = tuple(np.sqrt(np.mean(error**2, axis=0)))
        key = (round(normalized_rms, config.quality.lag_score_round_decimals), abs(lag), lag)
        candidates.append((key, lag, raw_rms, normalized_rms, per_joint))
    _, lag, raw_rms, normalized_rms, per_joint = min(candidates, key=lambda item: item[0])
    return lag, "ok", raw_rms, normalized_rms, per_joint


def _gripper_transitions(
    values: np.ndarray, valid: np.ndarray, open_threshold: float, closed_threshold: float
) -> tuple[int, int]:
    previous: str | None = None
    opens = closes = 0
    for value, is_valid in zip(values, valid, strict=True):
        if not is_valid:
            previous = None
            continue
        current = "open" if value >= open_threshold else "closed" if value <= closed_threshold else previous
        if current is not None and previous is not None and current != previous:
            if current == "open":
                opens += 1
            else:
                closes += 1
        previous = current
    return opens, closes


def _empty(identity: str, reasons: tuple[str, ...], task_key: str | None = None) -> EpisodeMetrics:
    return EpisodeMetrics(identity, None, task_key, METRIC_SCHEMA_VERSION, 0, {}, {}, {}, reasons)


def measure_aligned(episode: AlignedEpisode, config: ChallengeConfig) -> EpisodeMetrics:
    """Compute structural reasons and explainable physical metrics using NumPy only."""
    values: dict[str, float | int | str | None] = {}
    per_joint: dict[str, tuple[float, ...]] = {}
    units: dict[str, str] = {}
    reasons: set[str] = set()
    n = len(episode.frame)
    if episode.joint_names != config.joint_names:
        reasons.add("JOINT_SCHEMA_MISMATCH")
    if n != episode.expected_sample_count:
        reasons.add("FRAME_COUNT_MISMATCH")
    if episode.frame.shape != (n,) or not np.array_equal(episode.frame, np.arange(n)):
        reasons.add("FRAME_INDEX_NONCONTIGUOUS")
    if episode.action_time_ns.shape != (n,) or (n > 1 and np.any(np.diff(episode.action_time_ns) <= 0)):
        reasons.add("ACTION_TIME_NONMONOTONIC")
    action_dimension_ok = episode.action.shape == (n, 7)
    state_dimension_ok = episode.state.shape == (n, 7)
    if not action_dimension_ok:
        reasons.add("ACTION_DIMENSION")
    if not state_dimension_ok:
        reasons.add("STATE_DIMENSION")
    action_dimension_rows = (
        np.ones(n, dtype=bool)
        if episode.action_dimension_valid is None
        else np.asarray(episode.action_dimension_valid, dtype=bool)
    )
    state_dimension_rows = (
        np.ones(n, dtype=bool)
        if episode.state_dimension_valid is None
        else np.asarray(episode.state_dimension_valid, dtype=bool)
    )
    if action_dimension_rows.shape != (n,) or not np.all(action_dimension_rows):
        reasons.add("ACTION_DIMENSION")
        action_dimension_rows = np.zeros(n, dtype=bool) if action_dimension_rows.shape != (n,) else action_dimension_rows
    if state_dimension_rows.shape != (n,) or not np.all(state_dimension_rows):
        reasons.add("STATE_DIMENSION")
        state_dimension_rows = np.zeros(n, dtype=bool) if state_dimension_rows.shape != (n,) else state_dimension_rows
    action_present = np.asarray(episode.action_present, dtype=bool)
    state_present = np.asarray(episode.state_present, dtype=bool)
    if action_present.shape != (n,) or not np.all(action_present):
        reasons.add("ACTION_MISSING")
        action_present = np.zeros(n, dtype=bool) if action_present.shape != (n,) else action_present
    if state_present.shape != (n,) or not np.all(state_present):
        reasons.add("STATE_MISSING_OR_STALE")
        state_present = np.zeros(n, dtype=bool) if state_present.shape != (n,) else state_present
    if not action_dimension_ok or not state_dimension_ok:
        return EpisodeMetrics(
            episode.identity, episode.segment_id, episode.task_key, METRIC_SCHEMA_VERSION, n,
            values, per_joint, units, tuple(reason for reason in HARD_REASON_ORDER if reason in reasons),
        )
    action_finite = _finite_rows(episode.action)
    state_finite = _finite_rows(episode.state)
    if np.any(action_present & action_dimension_rows & ~action_finite):
        reasons.add("ACTION_NONFINITE")
    if np.any(state_present & state_dimension_rows & ~state_finite):
        reasons.add("STATE_NONFINITE")
    limits = np.asarray(config.robot_profile.action_limits_deg, dtype=np.float64)
    lower, upper = limits[:, 0], limits[:, 1]
    if np.any((episode.action < lower) | (episode.action > upper)):
        reasons.add("ACTION_LIMIT")
    tolerance = config.quality.observed_state_limit_tolerance_deg
    if np.any((episode.state < lower - tolerance) | (episode.state > upper + tolerance)):
        reasons.add("STATE_LIMIT")
    for key in config.camera_keys:
        present = np.asarray(episode.camera_present.get(key, np.zeros(n)), dtype=bool)
        if present.shape != (n,) or not np.all(present):
            reasons.add(f"CAMERA_{key.upper()}_MISSING_OR_STALE")

    valid_action = action_present & action_dimension_rows & action_finite
    valid_state = state_present & state_dimension_rows & state_finite
    paired = valid_action & valid_state
    error = episode.action[paired] - episode.state[paired]
    spans = np.asarray(config.quality.derivative_normalization_span_deg, dtype=np.float64)
    if len(error):
        absolute = np.abs(error)
        per_joint["tracking_mae_deg"] = tuple(np.mean(absolute, axis=0))
        per_joint["tracking_rms_deg"] = tuple(np.sqrt(np.mean(error**2, axis=0)))
        per_joint["tracking_max_abs_error_deg"] = tuple(np.max(absolute, axis=0))
        zero_raw = float(np.sqrt(np.mean(error**2)))
        zero_normalized = float(np.sqrt(np.mean((error / spans) ** 2)))
        values["tracking_mae_deg"] = float(np.mean(absolute))
        values["tracking_rms_deg"] = zero_raw
        values["tracking_max_abs_error_deg"] = float(np.max(absolute))
    else:
        nan7 = tuple([float("nan")] * 7)
        per_joint.update({"tracking_mae_deg": nan7, "tracking_rms_deg": nan7, "tracking_max_abs_error_deg": nan7})
        zero_raw = zero_normalized = float("nan")
        values.update({"tracking_mae_deg": float("nan"), "tracking_rms_deg": float("nan"), "tracking_max_abs_error_deg": float("nan")})
    for name in ("tracking_mae_deg", "tracking_rms_deg", "tracking_max_abs_error_deg"):
        units[name] = "deg"
    values["zero_lag_rms_deg"] = zero_raw
    units["zero_lag_rms_deg"] = "deg"
    values["zero_lag_normalized_rms"] = zero_normalized
    units["zero_lag_normalized_rms"] = "range"

    lag, lag_status, lag_raw, lag_normalized, lag_per_joint = _lag(
        episode.action, episode.state, paired, spans, config, zero_raw, zero_normalized
    )
    values["best_lag_frames"] = lag
    values["best_lag_seconds"] = lag * config.quality.sample_period_s if lag is not None else None
    values["lag_status"] = lag_status
    values["lag_corrected_rms_deg"] = lag_raw
    values["lag_corrected_normalized_rms"] = lag_normalized
    per_joint["lag_corrected_rms_deg"] = lag_per_joint
    units.update({"best_lag_frames": "frame", "best_lag_seconds": "s", "lag_status": "enum", "lag_corrected_rms_deg": "deg", "lag_corrected_normalized_rms": "range"})

    _put_derivatives(values, per_joint, units, "action", episode.action, valid_action, episode.frame, spans, config.quality.sample_period_s)
    _put_derivatives(values, per_joint, units, "state", episode.state, valid_state, episode.frame, spans, config.quality.sample_period_s)
    for prefix, trace, valid in (("action", episode.action, valid_action), ("state", episode.state, valid_state)):
        differences = np.abs(_differences(trace, valid, episode.frame, 1, 1.0))
        values[f"{prefix}_max_discontinuity_deg_per_frame"] = float(np.max(differences)) if differences.size else float("nan")
        values[f"{prefix}_max_discontinuity_range_per_frame"] = float(np.max(differences / spans)) if differences.size else float("nan")
        units[f"{prefix}_max_discontinuity_deg_per_frame"] = "deg/frame"
        units[f"{prefix}_max_discontinuity_range_per_frame"] = "range/frame"

        finite_trace = trace[valid]
        ranges = np.ptp(finite_trace, axis=0) if len(finite_trace) else np.full(7, np.nan)
        per_joint[f"{prefix}_range_deg"] = tuple(ranges)
        units[f"{prefix}_range_deg"] = "deg"
        delta = np.abs(_differences(trace, valid, episode.frame, 1, 1.0))
        if len(delta):
            stationary_joint = np.mean(delta <= config.quality.stationary_epsilon_deg_per_frame, axis=0)
            stationary = float(np.mean(np.all(delta <= config.quality.stationary_epsilon_deg_per_frame, axis=1)))
        else:
            stationary_joint = np.full(7, np.nan)
            stationary = float("nan")
        per_joint[f"{prefix}_stationary_fraction"] = tuple(stationary_joint)
        values[f"{prefix}_stationary_fraction"] = stationary
        units[f"{prefix}_stationary_fraction"] = "ratio"

    saturation = ((episode.action - lower) <= config.quality.saturation_band_deg) | ((upper - episode.action) <= config.quality.saturation_band_deg)
    saturation &= valid_action[:, None]
    denom = max(int(np.count_nonzero(valid_action)), 1)
    per_joint["action_saturation_fraction"] = tuple(np.sum(saturation, axis=0) / denom)
    values["action_saturation_fraction"] = float(np.sum(saturation) / (denom * 7))
    units["action_saturation_fraction"] = "ratio"

    values["episode_duration_s"] = float((episode.action_time_ns[-1] - episode.action_time_ns[0]) / 1e9) if n > 1 else 0.0
    values["expected_sample_count"] = episode.expected_sample_count
    units["episode_duration_s"] = "s"
    units["expected_sample_count"] = "count"
    gripper = episode.action[:, 6]
    opens, closes = _gripper_transitions(gripper, valid_action, config.quality.gripper_open_threshold_deg, config.quality.gripper_closed_threshold_deg)
    gripper_delta = np.abs(_differences(gripper[:, None], valid_action, episode.frame, 1, 1.0))
    values.update({"gripper_open_transition_count": opens, "gripper_close_transition_count": closes, "gripper_transition_count": opens + closes, "gripper_total_travel_deg": float(np.sum(gripper_delta)) if len(gripper_delta) else 0.0, "gripper_range_deg": float(np.ptp(gripper[valid_action])) if np.any(valid_action) else float("nan")})
    units.update({"gripper_open_transition_count": "count", "gripper_close_transition_count": "count", "gripper_transition_count": "count", "gripper_total_travel_deg": "deg", "gripper_range_deg": "deg"})

    _age_metrics(values, units, "state", episode.state_age_ns, state_present)
    for key in config.camera_keys:
        present = np.asarray(episode.camera_present.get(key, np.zeros(n)), dtype=bool)
        if present.shape != (n,):
            present = np.zeros(n, dtype=bool)
        count = int(np.count_nonzero(present))
        values[f"camera_{key}_coverage"] = count / n if n else 0.0
        values[f"camera_{key}_present_sample_count"] = count
        values[f"camera_{key}_missing_sample_count"] = n - count
        values[f"camera_{key}_gap_count"] = _missing_run_count(present)
        units.update({f"camera_{key}_coverage": "ratio", f"camera_{key}_present_sample_count": "count", f"camera_{key}_missing_sample_count": "count", f"camera_{key}_gap_count": "count"})
        ages = episode.camera_age_ns.get(key, np.full(n, -1, dtype=np.int64))
        _age_metrics(values, units, f"camera_{key}", ages, present)

    ordered = tuple(reason for reason in HARD_REASON_ORDER if reason in reasons)
    return EpisodeMetrics(episode.identity, episode.segment_id, episode.task_key, METRIC_SCHEMA_VERSION, n, values, per_joint, units, ordered)


def _source_task(identity: str, config: ChallengeConfig) -> str | None:
    return next((source.task_key for source in config.sources if identity.startswith(f"{source.repo_id}@{source.revision}:")), None)


def measure_artifact(artifact: CanonicalArtifact, config: ChallengeConfig) -> EpisodeMetrics:
    """Measure a canonical receipt, yielding one zero-row metric for every rejection."""
    task_key = _source_task(artifact.identity, config)
    if artifact.status == "rejected":
        return _empty(artifact.identity, artifact.reason_codes, task_key)
    if task_key is None:
        return _empty(artifact.identity, ("IDENTITY_MISMATCH",), None)
    if artifact.rrd_path is None:
        return _empty(artifact.identity, ("IDENTITY_MISMATCH",), task_key)
    try:
        with open_dataset_server(
            f"measure-{Path(artifact.rrd_path).stem}", rrd_paths=[artifact.rrd_path]
        ) as dataset:
            aligned = extract_aligned_episode(
                dataset,
                identity=artifact.identity,
                task_key=task_key,
                expected_sample_count=artifact.frame_count,
                config=config,
            )
    except AlignmentError as error:
        return _empty(artifact.identity, error.reason_codes, task_key)
    return measure_aligned(aligned, config)


compute_metrics = measure_aligned
