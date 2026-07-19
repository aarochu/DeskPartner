from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path

import pytest

from p5_rerun_port.challenge.config import ChallengeConfig
from p5_rerun_port.challenge.evaluate import (
    EvaluationError,
    EvaluationLabel,
    build_evaluation_labels,
    evaluate_verdicts,
)
from p5_rerun_port.challenge.hub import HuggingFaceHubReader
from p5_rerun_port.challenge.metrics import EpisodeMetrics
from p5_rerun_port.challenge.models import EpisodeIdentity, InventoryRow
from p5_rerun_port.challenge.quality import (
    CALIBRATED_METRICS,
    CalibrationCapture,
    EpisodeVerdict,
    ThresholdSnapshot,
    calibrate_thresholds,
    episode_verdict_digest,
    score_episode,
    threshold_snapshot_digest,
)


CONFIG = Path("config/rerun_query_challenge.yaml")


def _snapshot_and_labels() -> tuple[ThresholdSnapshot, tuple[EvaluationLabel, ...]]:
    calibration = tuple(f"success:cal:{index:02d}" for index in range(61))
    held = tuple(f"success:held:{index:02d}" for index in range(16))
    failures = tuple(f"failure:{index:02d}" for index in range(25))
    snapshot = ThresholdSnapshot(calibration, held, (), "")
    snapshot = replace(snapshot, payload_digest=threshold_snapshot_digest(snapshot))
    labels = [
        EvaluationLabel(identity, "single_can" if index < 41 else "two_can", "success", None, None, None)
        for index, identity in enumerate(calibration)
    ]
    labels.extend(
        EvaluationLabel(identity, "single_can" if index < 11 else "two_can", "success", None, None, None)
        for index, identity in enumerate(held)
    )
    dispositions = ["aborted"] * 17 + ["collector_error"] * 5 + ["failed"] * 3
    operator_dispositions = ["aborted"] * 19 + ["collector_error"] * 3 + ["failed"] * 3
    failure_labels: list[str | None] = (
        [None] * 19
        + ["dropped_object"] * 2
        + ["hardware_or_control_problem"] * 2
        + ["test_or_setup"] * 2
    )
    labels.extend(
        EvaluationLabel(
            identity,
            "single_can" if index < 19 else "two_can",
            "failure",
            dispositions[index],
            operator_dispositions[index],
            failure_labels[index],
        )
        for index, identity in enumerate(failures)
    )
    return snapshot, tuple(labels)


def _verdict(identity: str, verdict: str, snapshot: ThresholdSnapshot) -> EpisodeVerdict:
    frozen = EpisodeVerdict(
        identity,
        verdict,
        () if verdict == "PASS" else ("reason",),
        hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        snapshot.payload_digest,
        "",
    )
    return replace(frozen, verdict_digest=episode_verdict_digest(frozen))


def test_evaluation_enforces_held_out_plus_all_failures_and_reports_cross_tabs() -> None:
    snapshot, labels = _snapshot_and_labels()
    held = snapshot.held_out_success_identities
    failures = tuple(label.identity for label in labels if label.actual_role == "failure")
    verdicts = tuple(
        [_verdict(identity, "REVIEW" if index < 2 else "PASS", snapshot) for index, identity in enumerate(held)]
        + [_verdict(identity, "REJECT" if index < 20 else "PASS", snapshot) for index, identity in enumerate(failures)]
    )

    report = evaluate_verdicts(snapshot, tuple(reversed(verdicts)), tuple(reversed(labels)))

    assert report.evaluated_identities == tuple(sorted(held + failures))
    assert report.confusion == {"tp": 20, "fp": 2, "tn": 14, "fn": 5}
    assert report.precision == pytest.approx(20 / 22)
    assert report.recall == pytest.approx(20 / 25)
    assert report.f1 == pytest.approx(40 / 47)
    assert report.false_rejection_rate == pytest.approx(2 / 16)
    assert report.by_task["single_can"]["total"] == 30
    assert report.by_task["two_can"]["total"] == 11
    assert report.by_disposition["aborted"]["total"] == 17
    assert report.by_failure_label["__unlabeled__"]["total"] == 19
    assert report.false_positive_identities == tuple(sorted(held[:2]))
    assert report.false_negative_identities == tuple(sorted(failures[20:]))


def test_evaluation_uses_zero_for_undefined_ratios() -> None:
    snapshot, labels = _snapshot_and_labels()
    evaluated = snapshot.held_out_success_identities + tuple(
        label.identity for label in labels if label.actual_role == "failure"
    )
    report = evaluate_verdicts(
        snapshot,
        tuple(_verdict(identity, "PASS", snapshot) for identity in evaluated),
        labels,
    )
    assert report.precision == 0.0
    assert report.f1 == 0.0
    assert report.by_failure_label["dropped_object"]["precision"] == 0.0


@pytest.mark.parametrize(
    "problem",
    ["labels_missing", "labels_duplicate", "verdict_extra", "calibration_verdict", "blank_digest"],
)
def test_evaluation_rejects_every_identity_set_or_digest_mismatch(problem: str) -> None:
    snapshot, labels = _snapshot_and_labels()
    failures = tuple(label.identity for label in labels if label.actual_role == "failure")
    verdicts = tuple(
        _verdict(identity, "PASS", snapshot)
        for identity in snapshot.held_out_success_identities + failures
    )
    changed_labels = labels
    changed_verdicts = verdicts
    if problem == "labels_missing":
        changed_labels = labels[:-1]
    elif problem == "labels_duplicate":
        changed_labels = labels[:-1] + (labels[0],)
    elif problem == "verdict_extra":
        changed_verdicts += (_verdict("extra", "PASS", snapshot),)
    elif problem == "calibration_verdict":
        changed_verdicts = changed_verdicts[:-1] + (
            _verdict(snapshot.calibration_identities[0], "PASS", snapshot),
        )
    else:
        changed_verdicts = (replace(verdicts[0], metrics_digest=""),) + verdicts[1:]

    with pytest.raises(EvaluationError):
        evaluate_verdicts(snapshot, changed_verdicts, changed_labels)


@pytest.mark.parametrize(
    "problem",
    [
        "task_counts",
        "disposition_counts",
        "operator_disposition_counts",
        "failure_label_counts",
        "invalid_verdict",
    ],
)
def test_evaluation_rejects_mutated_pinned_label_or_verdict_contract(problem: str) -> None:
    snapshot, labels = _snapshot_and_labels()
    failures = tuple(label.identity for label in labels if label.actual_role == "failure")
    verdicts = tuple(
        _verdict(identity, "PASS", snapshot)
        for identity in snapshot.held_out_success_identities + failures
    )
    changed_labels = list(labels)
    changed_verdicts = verdicts
    if problem == "task_counts":
        changed_labels[0] = replace(changed_labels[0], task_key="two_can")
    elif problem == "disposition_counts":
        index = next(index for index, label in enumerate(changed_labels) if label.disposition == "aborted")
        changed_labels[index] = replace(changed_labels[index], disposition="failed")
    elif problem == "operator_disposition_counts":
        index = next(
            index
            for index, label in enumerate(changed_labels)
            if label.operator_disposition == "aborted"
        )
        changed_labels[index] = replace(changed_labels[index], operator_disposition="failed")
    elif problem == "failure_label_counts":
        index = next(
            index
            for index, label in enumerate(changed_labels)
            if label.failure_label is None and label.actual_role == "failure"
        )
        changed_labels[index] = replace(changed_labels[index], failure_label="dropped_object")
    else:
        changed_verdicts = (replace(verdicts[0], verdict="UNKNOWN"),) + verdicts[1:]

    with pytest.raises(EvaluationError):
        evaluate_verdicts(snapshot, changed_verdicts, tuple(changed_labels))


def _calibrate_score_evaluate_flow() -> tuple[
    ThresholdSnapshot, tuple[EpisodeVerdict, ...], tuple[EvaluationLabel, ...]
]:
    config = ChallengeConfig.load(CONFIG)
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    metrics: list[EpisodeMetrics] = []
    captures: list[CalibrationCapture] = []
    for task_key, count in (("single_can", 52), ("two_can", 25)):
        for index in range(count):
            identity = f"flow:{task_key}:{index:02d}"
            values = {
                spec.metric: float(index + offset)
                for offset, spec in enumerate(CALIBRATED_METRICS)
            }
            metrics.append(
                EpisodeMetrics(
                    identity,
                    f"segment:{identity}",
                    task_key,
                    "1",
                    100,
                    values,
                    {},
                    {spec.metric: spec.units for spec in CALIBRATED_METRICS},
                    (),
                )
            )
            captures.append(
                CalibrationCapture(
                    identity,
                    task_key,
                    (epoch + timedelta(seconds=index)).isoformat(),
                )
            )
    snapshot = calibrate_thresholds(tuple(metrics), tuple(captures), config)
    metrics_by_identity = {metric.identity: metric for metric in metrics}
    labels = [
        EvaluationLabel(
            identity,
            "single_can" if identity.startswith("flow:single_can") else "two_can",
            "success",
            None,
            None,
            None,
        )
        for identity in snapshot.calibration_identities + snapshot.held_out_success_identities
    ]
    verdicts = [
        score_episode(metrics_by_identity[identity], snapshot)
        for identity in snapshot.held_out_success_identities
    ]
    dispositions = ["aborted"] * 17 + ["collector_error"] * 5 + ["failed"] * 3
    operator_dispositions = ["aborted"] * 19 + ["collector_error"] * 3 + ["failed"] * 3
    failure_labels = (
        [None] * 19
        + ["dropped_object"] * 2
        + ["hardware_or_control_problem"] * 2
        + ["test_or_setup"] * 2
    )
    clean_values = {threshold.metric: threshold.median for threshold in snapshot.thresholds}
    units = {threshold.metric: threshold.units for threshold in snapshot.thresholds}
    for index in range(25):
        identity = f"flow:failure:{index:02d}"
        task_key = "single_can" if index < 19 else "two_can"
        labels.append(
            EvaluationLabel(
                identity,
                task_key,
                "failure",
                dispositions[index],
                operator_dispositions[index],
                failure_labels[index],
            )
        )
        failure_metrics = EpisodeMetrics(
            identity,
            f"segment:{identity}",
            task_key,
            "1",
            100,
            dict(clean_values),
            {},
            dict(units),
            ("ACTION_LIMIT",),
        )
        verdicts.append(score_episode(failure_metrics, snapshot))
    return snapshot, tuple(verdicts), tuple(labels)


def test_labels_are_revealed_after_real_freeze_and_cannot_change_any_digest() -> None:
    snapshot, verdicts, labels = _calibrate_score_evaluate_flow()
    baseline_digests = (
        snapshot.payload_digest,
        tuple((verdict.metrics_digest, verdict.verdict_digest) for verdict in verdicts),
    )
    first = next(
        index
        for index, label in enumerate(labels)
        if label.failure_label is None and label.actual_role == "failure"
    )
    second = next(index for index, label in enumerate(labels) if label.failure_label == "dropped_object")
    changed = list(labels)
    changed[first] = replace(changed[first], failure_label=labels[second].failure_label)
    changed[second] = replace(changed[second], failure_label=labels[first].failure_label)

    evaluate_verdicts(snapshot, verdicts, tuple(changed))
    assert baseline_digests == (
        snapshot.payload_digest,
        tuple((verdict.metrics_digest, verdict.verdict_digest) for verdict in verdicts),
    )


@pytest.mark.parametrize(
    "problem",
    [
        "snapshot",
        "verdict",
        "reasons",
        "metrics_digest",
        "threshold_digest",
        "reused_digest",
        "reused_metrics_digest",
    ],
)
def test_evaluation_rejects_every_stale_or_mutated_frozen_payload(problem: str) -> None:
    snapshot, verdicts, labels = _calibrate_score_evaluate_flow()
    changed_snapshot = snapshot
    changed_verdicts = list(verdicts)
    if problem == "snapshot":
        threshold = replace(snapshot.thresholds[0], value=snapshot.thresholds[0].value + 1)
        changed_snapshot = replace(snapshot, thresholds=(threshold,) + snapshot.thresholds[1:])
    elif problem == "verdict":
        changed_verdicts[0] = replace(changed_verdicts[0], verdict="REVIEW")
    elif problem == "reasons":
        changed_verdicts[0] = replace(changed_verdicts[0], reason_codes=("changed",))
    elif problem == "metrics_digest":
        changed_verdicts[0] = replace(changed_verdicts[0], metrics_digest="c" * 64)
    elif problem == "threshold_digest":
        changed_verdicts[0] = replace(changed_verdicts[0], threshold_digest="d" * 64)
    elif problem == "reused_digest":
        changed_verdicts[0] = replace(
            changed_verdicts[0], verdict_digest=changed_verdicts[1].verdict_digest
        )
    else:
        changed_verdicts[0] = replace(
            changed_verdicts[0],
            metrics_digest=changed_verdicts[1].metrics_digest,
            verdict_digest="",
        )
        changed_verdicts[0] = replace(
            changed_verdicts[0],
            verdict_digest=episode_verdict_digest(changed_verdicts[0]),
        )

    with pytest.raises(EvaluationError, match="digest"):
        evaluate_verdicts(changed_snapshot, tuple(changed_verdicts), labels)


class _MetadataHub:
    def __init__(self, documents: dict[tuple[str, str], dict]) -> None:
        self.documents = documents

    def read_json(self, repo_id: str, revision: str, path: str) -> dict:
        assert path == "FAILURE_INDEX.json"
        return self.documents[(repo_id, revision)]


def test_build_labels_joins_attempts_exactly_and_preserves_operator_disposition() -> None:
    success = InventoryRow(
        EpisodeIdentity("o/s", "a" * 40, "0"),
        "success",
        "single_can",
        "x",
        0,
        None,
        1,
        "2026-01-01T00:00:00Z",
    )
    failure_id = EpisodeIdentity("o/f", "b" * 40, "attempt-1")
    failure = InventoryRow(failure_id, "failure", "single_can", "x", None, "attempt-1", 1, "2026-01-01T00:00:00Z")
    document = {
        "training_eligible": False,
        "attempt_count": 1,
        "attempts": [{
            "attempt_id": "attempt-1",
            "disposition": "collector_error",
            "operator_disposition": "aborted",
            "failure_label": None,
            "training_included": False,
        }],
    }
    hub = _MetadataHub({("o/f", "b" * 40): document})
    labels = build_evaluation_labels((failure, success), hub)
    assert [label.identity for label in labels] == sorted([success.identity.canonical, failure_id.canonical])
    failure_label = next(label for label in labels if label.actual_role == "failure")
    assert failure_label.disposition == "collector_error"
    assert failure_label.operator_disposition == "aborted"
    assert failure_label.failure_label is None

    document["attempts"].append(dict(document["attempts"][0], attempt_id="extra"))
    document["attempt_count"] = 2
    with pytest.raises(EvaluationError, match="join"):
        build_evaluation_labels((failure, success), hub)


def test_build_labels_requires_explicit_nullable_failure_label_key() -> None:
    failure_id = EpisodeIdentity("o/f", "b" * 40, "attempt-1")
    failure = InventoryRow(
        failure_id,
        "failure",
        "single_can",
        "x",
        None,
        "attempt-1",
        1,
        "2026-01-01T00:00:00Z",
    )
    document = {
        "training_eligible": False,
        "attempt_count": 1,
        "attempts": [{
            "attempt_id": "attempt-1",
            "disposition": "aborted",
            "operator_disposition": "aborted",
            "training_included": False,
        }],
    }
    hub = _MetadataHub({("o/f", "b" * 40): document})

    with pytest.raises(EvaluationError, match="failure_label"):
        build_evaluation_labels((failure,), hub)


def test_build_labels_requires_canonical_source_key_to_equal_attempt_id() -> None:
    failure = InventoryRow(
        EpisodeIdentity("o/f", "b" * 40, "wrong-source-key"),
        "failure",
        "single_can",
        "x",
        None,
        "attempt-1",
        1,
        "2026-01-01T00:00:00Z",
    )
    document = {
        "training_eligible": False,
        "attempt_count": 1,
        "attempts": [{
            "attempt_id": "attempt-1",
            "disposition": "aborted",
            "operator_disposition": "aborted",
            "failure_label": None,
            "training_included": False,
        }],
    }
    hub = _MetadataHub({("o/f", "b" * 40): document})

    with pytest.raises(EvaluationError, match="canonical"):
        build_evaluation_labels((failure,), hub)


@pytest.mark.skipif(os.environ.get("SKIP_PINNED_METADATA_SMOKE") == "1", reason="explicitly disabled")
def test_real_pinned_failure_json_counts_metadata_only() -> None:
    config = ChallengeConfig.load(CONFIG)
    hub = HuggingFaceHubReader()
    dispositions: Counter[str] = Counter()
    operator_dispositions: Counter[str] = Counter()
    failure_labels: Counter[str] = Counter()
    total = 0
    for source in config.sources:
        if source.role != "failure":
            continue
        document = hub.read_json(source.repo_id, source.revision, "FAILURE_INDEX.json")
        attempts = document["attempts"]
        assert len(attempts) == source.expected_items
        total += len(attempts)
        dispositions.update(attempt["disposition"] for attempt in attempts)
        operator_dispositions.update(attempt["operator_disposition"] for attempt in attempts)
        failure_labels.update(
            "__unlabeled__" if attempt["failure_label"] is None else attempt["failure_label"]
            for attempt in attempts
        )
    assert total == 25
    assert dispositions == {"aborted": 17, "collector_error": 5, "failed": 3}
    assert operator_dispositions == {"aborted": 19, "collector_error": 3, "failed": 3}
    assert failure_labels == {
        "__unlabeled__": 19,
        "dropped_object": 2,
        "hardware_or_control_problem": 2,
        "test_or_setup": 2,
    }
