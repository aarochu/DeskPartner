from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from p5_rerun_port.challenge.config import ChallengeConfig
from p5_rerun_port.challenge.metrics import EpisodeMetrics
from p5_rerun_port.challenge.models import EpisodeIdentity, InventoryRow
from p5_rerun_port.challenge.quality import (
    CALIBRATED_METRICS,
    CalibrationError,
    calibrate_thresholds,
    score_episode,
)


CONFIG = Path("config/rerun_query_challenge.yaml")


def _success_fixture() -> tuple[tuple[EpisodeMetrics, ...], tuple[InventoryRow, ...]]:
    metrics: list[EpisodeMetrics] = []
    inventory: list[InventoryRow] = []
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for task_key, count, repo in (
        ("single_can", 52, "owner/single"),
        ("two_can", 25, "owner/two"),
    ):
        for index in range(count):
            identity = EpisodeIdentity(repo, "a" * 40, str(index))
            captured = epoch + timedelta(seconds=index)
            # Tie the first pair to prove identity is the deterministic tie breaker.
            if index == 1:
                captured = epoch
            inventory.append(
                InventoryRow(
                    identity=identity,
                    role="success",
                    task_key=task_key,
                    source_path=f"data/{index}.parquet",
                    episode_index=index,
                    attempt_id=None,
                    frame_count=100,
                    captured_at=captured.isoformat(),
                )
            )
            values = {
                spec.metric: float(index + metric_index)
                for metric_index, spec in enumerate(CALIBRATED_METRICS)
            }
            units = {spec.metric: spec.units for spec in CALIBRATED_METRICS}
            metrics.append(
                EpisodeMetrics(
                    identity.canonical,
                    f"segment-{task_key}-{index}",
                    task_key,
                    "1",
                    100,
                    values,
                    {},
                    units,
                    (),
                )
            )
    return tuple(reversed(metrics)), tuple(reversed(inventory))


def test_calibration_uses_exact_capture_ordered_41_20_and_11_5_split() -> None:
    config = ChallengeConfig.load(CONFIG)
    metrics, inventory = _success_fixture()

    snapshot = calibrate_thresholds(metrics, inventory, config)

    calibration_tasks = [
        next(row.task_key for row in inventory if row.identity.canonical == identity)
        for identity in snapshot.calibration_identities
    ]
    held_tasks = [
        next(row.task_key for row in inventory if row.identity.canonical == identity)
        for identity in snapshot.held_out_success_identities
    ]
    assert calibration_tasks.count("single_can") == 41
    assert calibration_tasks.count("two_can") == 20
    assert held_tasks.count("single_can") == 11
    assert held_tasks.count("two_can") == 5
    assert snapshot.calibration_identities[:2] == (
        f"owner/single@{'a' * 40}:0",
        f"owner/single@{'a' * 40}:1",
    )


def test_thresholds_serialize_linear_quantiles_mad_bounds_and_physical_bounds() -> None:
    config = ChallengeConfig.load(CONFIG)
    metrics, inventory = _success_fixture()

    snapshot = calibrate_thresholds(metrics, inventory, config)
    upper = next(threshold for threshold in snapshot.thresholds if threshold.direction == "upper")
    lower = next(threshold for threshold in snapshot.thresholds if threshold.direction == "lower")
    calibration_values = np.asarray(
        [
            metric.values[upper.metric]
            for metric in metrics
            if metric.identity in snapshot.calibration_identities
        ],
        dtype=np.float64,
    )
    expected_median = float(np.median(calibration_values))
    expected_mad = float(np.median(np.abs(calibration_values - expected_median)))

    assert upper.q01 == pytest.approx(np.quantile(calibration_values, 0.01, method="linear"))
    assert upper.q99 == pytest.approx(np.quantile(calibration_values, 0.99, method="linear"))
    assert upper.median == pytest.approx(expected_median)
    assert upper.mad == pytest.approx(expected_mad)
    assert upper.mad_bound == pytest.approx(expected_median + 5 * expected_mad)
    assert upper.value == max(upper.q99, upper.mad_bound)
    assert upper.method == "linear"
    assert upper.sample_count == 61
    assert lower.value == max(lower.physical_bound, min(lower.q01, lower.mad_bound))
    assert len(snapshot.payload_digest) == 64


@pytest.mark.parametrize("problem", ["duplicate_metric", "missing_metric", "extra_metric"])
def test_calibration_rejects_duplicate_missing_and_extra_metric_identities(problem: str) -> None:
    config = ChallengeConfig.load(CONFIG)
    metrics, inventory = _success_fixture()
    changed = list(metrics)
    if problem == "duplicate_metric":
        changed[-1] = changed[0]
    elif problem == "missing_metric":
        changed.pop()
    else:
        changed.append(replace(changed[0], identity="extra@" + "b" * 40 + ":0"))

    with pytest.raises(CalibrationError, match="identit"):
        calibrate_thresholds(tuple(changed), inventory, config)


def test_calibration_rejects_nonfinite_values_and_hard_reasons() -> None:
    config = ChallengeConfig.load(CONFIG)
    metrics, inventory = _success_fixture()
    calibration_index = next(
        index for index, metric in enumerate(metrics) if metric.identity.endswith(":0") and "single" in metric.identity
    )
    bad_values = dict(metrics[calibration_index].values)
    bad_values[CALIBRATED_METRICS[0].metric] = math.inf
    nonfinite = list(metrics)
    nonfinite[calibration_index] = replace(metrics[calibration_index], values=bad_values)
    with pytest.raises(CalibrationError, match="finite"):
        calibrate_thresholds(tuple(nonfinite), inventory, config)

    hard = list(metrics)
    hard[calibration_index] = replace(metrics[calibration_index], hard_reasons=("ACTION_LIMIT",))
    with pytest.raises(CalibrationError, match="hard"):
        calibrate_thresholds(tuple(hard), inventory, config)


def test_calibration_rejects_metric_task_metadata_mismatch() -> None:
    config = ChallengeConfig.load(CONFIG)
    metrics, inventory = _success_fixture()
    changed = (replace(metrics[0], task_key="single_can"),) + metrics[1:]

    with pytest.raises(CalibrationError, match="task"):
        calibrate_thresholds(changed, inventory, config)


def test_score_signature_and_metric_type_are_label_isolated() -> None:
    assert tuple(inspect.signature(score_episode).parameters) == ("metrics", "thresholds")
    forbidden = {
        "role",
        "disposition",
        "failure_label",
        "operator_disposition",
        "training_eligible",
        "captured_at",
    }
    assert forbidden.isdisjoint(field.name for field in fields(EpisodeMetrics))


def test_hard_reasons_reject_anomaly_reviews_and_clean_metrics_pass_deterministically() -> None:
    config = ChallengeConfig.load(CONFIG)
    metrics, inventory = _success_fixture()
    snapshot = calibrate_thresholds(metrics, inventory, config)
    clean = metrics[0]
    clean_values = {
        threshold.metric: threshold.median for threshold in snapshot.thresholds
    }
    clean = replace(clean, values=clean_values)
    first = score_episode(clean, snapshot)
    assert first.verdict == "PASS"
    assert first.reason_codes == ()
    assert first == score_episode(replace(clean, values=dict(reversed(tuple(clean_values.items())))), snapshot)

    anomalous_values = dict(clean_values)
    first_threshold, second_threshold = snapshot.thresholds[:2]
    for threshold in (second_threshold, first_threshold):
        anomalous_values[threshold.metric] = (
            threshold.value + 1 if threshold.direction == "upper" else threshold.value - 1
        )
    review = score_episode(replace(clean, values=anomalous_values), snapshot)
    assert review.verdict == "REVIEW"
    assert review.reason_codes == tuple(
        f"ANOMALY_{threshold.metric.upper()}"
        for threshold in snapshot.thresholds[:2]
    )

    hard = score_episode(replace(clean, hard_reasons=("STATE_LIMIT", "ACTION_LIMIT")), snapshot)
    assert hard.verdict == "REJECT"
    assert hard.reason_codes == ("ACTION_LIMIT", "STATE_LIMIT")
    assert len(hard.metrics_digest) == 64


def test_snapshot_digest_is_stable_under_input_reordering_and_changes_with_configured_stats() -> None:
    config = ChallengeConfig.load(CONFIG)
    metrics, inventory = _success_fixture()
    first = calibrate_thresholds(metrics, inventory, config)
    second = calibrate_thresholds(tuple(reversed(metrics)), tuple(reversed(inventory)), config)
    assert first == second

    calibration_index = next(
        index for index, metric in enumerate(metrics) if metric.identity.endswith(":0") and "single" in metric.identity
    )
    changed_values = dict(metrics[calibration_index].values)
    changed_values[CALIBRATED_METRICS[0].metric] = float(changed_values[CALIBRATED_METRICS[0].metric]) + 7
    changed_metrics = list(metrics)
    changed_metrics[calibration_index] = replace(metrics[calibration_index], values=changed_values)
    changed = calibrate_thresholds(tuple(changed_metrics), inventory, config)
    assert changed.payload_digest != first.payload_digest
