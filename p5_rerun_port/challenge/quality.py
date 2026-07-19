"""Outcome-blind threshold calibration and deterministic episode scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from typing import Any, Literal, cast

import numpy as np

from .config import ChallengeConfig
from .metrics import EpisodeMetrics, HARD_REASON_ORDER
from .models import TaskKey, Verdict


class QualityError(ValueError):
    """Metrics or thresholds do not satisfy the scoring contract."""


class CalibrationError(QualityError):
    """Success metrics cannot produce an authenticated threshold snapshot."""


@dataclass(frozen=True)
class CalibratedMetricSpec:
    metric: str
    units: str
    direction: Literal["upper", "lower"]
    physical_bound: float | None


@dataclass(frozen=True)
class CalibrationCapture:
    """Outcome-free source metadata used only for chronological calibration."""

    identity: str
    task_key: TaskKey
    captured_at: str


# Ordered explicitly so calibration rows and anomaly reasons never depend on maps.
# Structural integrity fields are deliberately absent: they are hard gates upstream.
CALIBRATED_METRICS = (
    CalibratedMetricSpec("tracking_rms_deg", "deg", "upper", None),
    CalibratedMetricSpec("lag_corrected_normalized_rms", "range", "upper", None),
    CalibratedMetricSpec("action_jerk_p95_normalized_s3", "s^-3", "upper", None),
    CalibratedMetricSpec("state_jerk_p95_normalized_s3", "s^-3", "upper", None),
    CalibratedMetricSpec(
        "action_max_discontinuity_range_per_frame", "range/frame", "upper", None
    ),
    CalibratedMetricSpec(
        "state_max_discontinuity_range_per_frame", "range/frame", "upper", None
    ),
    CalibratedMetricSpec("action_stationary_fraction", "ratio", "upper", None),
    CalibratedMetricSpec("action_saturation_fraction", "ratio", "upper", None),
    CalibratedMetricSpec("gripper_transition_count", "count", "lower", 0.0),
    CalibratedMetricSpec("gripper_total_travel_deg", "deg", "lower", 0.0),
    CalibratedMetricSpec("gripper_range_deg", "deg", "lower", 0.0),
)


@dataclass(frozen=True)
class MetricThreshold:
    metric: str
    units: str
    direction: Literal["upper", "lower"]
    q01: float
    q99: float
    median: float
    mad: float
    mad_bound: float
    physical_bound: float | None
    method: str
    sample_count: int
    value: float


@dataclass(frozen=True)
class ThresholdSnapshot:
    calibration_identities: tuple[str, ...]
    held_out_success_identities: tuple[str, ...]
    thresholds: tuple[MetricThreshold, ...]
    payload_digest: str


@dataclass(frozen=True)
class EpisodeVerdict:
    identity: str
    verdict: Verdict
    reason_codes: tuple[str, ...]
    metrics_digest: str
    threshold_digest: str
    verdict_digest: str


def _canonical(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+Infinity" if value > 0 else "-Infinity"
        return 0.0 if value == 0 else value
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def threshold_snapshot_digest(snapshot: ThresholdSnapshot) -> str:
    """Recompute the semantic digest over every frozen threshold field."""

    return _digest(
        {
            "calibration_identities": snapshot.calibration_identities,
            "held_out_success_identities": snapshot.held_out_success_identities,
            "thresholds": [asdict(threshold) for threshold in snapshot.thresholds],
        }
    )


def episode_verdict_digest(verdict: EpisodeVerdict) -> str:
    """Recompute the complete frozen verdict payload digest."""

    return _digest(
        {
            "identity": verdict.identity,
            "verdict": verdict.verdict,
            "reason_codes": verdict.reason_codes,
            "metrics_digest": verdict.metrics_digest,
            "threshold_digest": verdict.threshold_digest,
        }
    )


def _parse_timestamp(value: str, identity: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00" if value.endswith("Z") else value)
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"success inventory {identity} has invalid captured_at") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CalibrationError(f"success inventory {identity} captured_at must be timezone-aware")
    return parsed


def _unique_by_identity(items: tuple[Any, ...], identity_of: Any, name: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        identity = identity_of(item)
        if not isinstance(identity, str) or not identity.strip():
            raise CalibrationError(f"{name} identity must be non-blank")
        if identity in result:
            raise CalibrationError(f"{name} contains duplicate identity {identity}")
        result[identity] = item
    return result


def _numeric_metric(metrics: EpisodeMetrics, spec: CalibratedMetricSpec) -> float:
    value = metrics.values.get(spec.metric)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise CalibrationError(f"{metrics.identity} metric {spec.metric} must be numeric and finite")
    number = float(value)
    if not math.isfinite(number):
        raise CalibrationError(f"{metrics.identity} metric {spec.metric} must be finite")
    if metrics.metric_units.get(spec.metric) != spec.units:
        raise CalibrationError(
            f"{metrics.identity} metric {spec.metric} units must be {spec.units}"
        )
    return number


def calibrate_thresholds(
    metrics: tuple[EpisodeMetrics, ...],
    captures: tuple[CalibrationCapture, ...],
    config: ChallengeConfig,
) -> ThresholdSnapshot:
    """Freeze an exact time-ordered approved-source split and thresholds."""

    captures_by_identity = _unique_by_identity(
        captures, lambda capture: capture.identity, "calibration captures"
    )
    metrics_by_identity = _unique_by_identity(metrics, lambda row: row.identity, "metrics")
    if len(captures_by_identity) != 77:
        raise CalibrationError("calibration captures must contain exactly 77 identities")
    if set(metrics_by_identity) != set(captures_by_identity):
        missing = sorted(set(captures_by_identity) - set(metrics_by_identity))
        extra = sorted(set(metrics_by_identity) - set(captures_by_identity))
        raise CalibrationError(
            f"metrics identities must equal calibration capture identities; missing={missing}, extra={extra}"
        )
    for identity, capture in captures_by_identity.items():
        if capture.task_key not in ("single_can", "two_can"):
            raise CalibrationError(f"calibration capture {identity} has unknown task")
        if metrics_by_identity[identity].task_key != capture.task_key:
            raise CalibrationError(f"metrics task for {identity} must match calibration capture")

    required_counts = {"single_can": 52, "two_can": 25}
    calibration: list[str] = []
    held_out: list[str] = []
    for task in ("single_can", "two_can"):
        rows = [capture for capture in captures_by_identity.values() if capture.task_key == task]
        if len(rows) != required_counts[task]:
            raise CalibrationError(
                f"{task} calibration capture count {len(rows)} must equal {required_counts[task]}"
            )
        ordered = sorted(
            rows,
            key=lambda capture: (
                _parse_timestamp(capture.captured_at, capture.identity),
                capture.identity,
            ),
        )
        calibration_count = int(len(ordered) * config.quality.calibration_fraction)
        if calibration_count <= 0 or calibration_count >= len(ordered):
            raise CalibrationError(f"{task} must have both calibration and held-out successes")
        calibration.extend(capture.identity for capture in ordered[:calibration_count])
        held_out.extend(capture.identity for capture in ordered[calibration_count:])

    if len(calibration) != 61 or len(held_out) != 16:
        raise CalibrationError("locked split must contain exactly 61 calibration and 16 held-out successes")
    if set(calibration) | set(held_out) != set(captures_by_identity):
        raise CalibrationError("calibration and held-out split must cover every joined identity")
    if set(calibration) & set(held_out):
        raise CalibrationError("calibration and held-out split must be disjoint")

    thresholds: list[MetricThreshold] = []
    for spec in CALIBRATED_METRICS:
        values: list[float] = []
        for identity in calibration:
            episode = metrics_by_identity[identity]
            if episode.hard_reasons:
                raise CalibrationError(
                    f"calibration metric {identity} has hard integrity reasons"
                )
            values.append(_numeric_metric(episode, spec))
        array = np.asarray(values, dtype=np.float64)
        q01 = float(np.quantile(array, config.quality.lower_quantile, method="linear"))
        q99 = float(np.quantile(array, config.quality.upper_quantile, method="linear"))
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        signed_multiplier = (
            config.quality.mad_multiplier
            if spec.direction == "upper"
            else -config.quality.mad_multiplier
        )
        mad_bound = median + signed_multiplier * mad
        if spec.direction == "upper":
            value = max(q99, mad_bound)
        else:
            if spec.physical_bound is None:
                raise CalibrationError(f"lower metric {spec.metric} requires a physical bound")
            value = max(spec.physical_bound, min(q01, mad_bound))
        threshold = MetricThreshold(
            metric=spec.metric,
            units=spec.units,
            direction=spec.direction,
            q01=q01,
            q99=q99,
            median=median,
            mad=mad,
            mad_bound=mad_bound,
            physical_bound=spec.physical_bound,
            method="linear",
            sample_count=len(array),
            value=float(value),
        )
        if not all(
            math.isfinite(number)
            for number in (
                threshold.q01,
                threshold.q99,
                threshold.median,
                threshold.mad,
                threshold.mad_bound,
                threshold.value,
            )
        ):
            raise CalibrationError(f"threshold {spec.metric} contains a non-finite statistic")
        thresholds.append(threshold)

    snapshot = ThresholdSnapshot(tuple(calibration), tuple(held_out), tuple(thresholds), "")
    return ThresholdSnapshot(
        snapshot.calibration_identities,
        snapshot.held_out_success_identities,
        snapshot.thresholds,
        threshold_snapshot_digest(snapshot),
    )


def _ordered_hard_reasons(reasons: tuple[str, ...]) -> tuple[str, ...]:
    order = {reason: index for index, reason in enumerate(HARD_REASON_ORDER)}
    return tuple(sorted(set(reasons), key=lambda reason: (order.get(reason, len(order)), reason)))


def _metric_number(metrics: EpisodeMetrics, threshold: MetricThreshold) -> float | None:
    value = metrics.values.get(threshold.metric)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    number = float(value)
    if not math.isfinite(number) or metrics.metric_units.get(threshold.metric) != threshold.units:
        return None
    return number


def score_episode(metrics: EpisodeMetrics, thresholds: ThresholdSnapshot) -> EpisodeVerdict:
    """Score one episode using only metrics and a verified threshold snapshot."""

    if (
        not thresholds.payload_digest.strip()
        or threshold_snapshot_digest(thresholds) != thresholds.payload_digest
    ):
        raise CalibrationError("threshold snapshot digest does not match its frozen payload")
    metrics_digest = _digest(asdict(metrics))
    hard_reasons = _ordered_hard_reasons(metrics.hard_reasons)
    if hard_reasons:
        verdict = EpisodeVerdict(
            metrics.identity,
            "REJECT",
            hard_reasons,
            metrics_digest,
            thresholds.payload_digest,
            "",
        )
        return EpisodeVerdict(
            verdict.identity,
            verdict.verdict,
            verdict.reason_codes,
            verdict.metrics_digest,
            verdict.threshold_digest,
            episode_verdict_digest(verdict),
        )

    reasons: list[str] = []
    for threshold in thresholds.thresholds:
        number = _metric_number(metrics, threshold)
        anomalous = number is None or (
            number > threshold.value if threshold.direction == "upper" else number < threshold.value
        )
        if anomalous:
            reasons.append(f"ANOMALY_{threshold.metric.upper()}")
    verdict = cast(Verdict, "REVIEW" if reasons else "PASS")
    frozen = EpisodeVerdict(
        metrics.identity,
        verdict,
        tuple(reasons),
        metrics_digest,
        thresholds.payload_digest,
        "",
    )
    return EpisodeVerdict(
        frozen.identity,
        frozen.verdict,
        frozen.reason_codes,
        frozen.metrics_digest,
        frozen.threshold_digest,
        episode_verdict_digest(frozen),
    )
