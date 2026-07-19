"""Post-verdict label loading and held-out evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from .hub import HubReader
from .models import InventoryRow, SourceRole, TaskKey
from .quality import (
    EpisodeVerdict,
    ThresholdSnapshot,
    episode_verdict_digest,
    threshold_snapshot_digest,
)


Disposition = Literal["aborted", "collector_error", "failed"]
_DISPOSITIONS = frozenset({"aborted", "collector_error", "failed"})


class EvaluationError(ValueError):
    """Labels, verdicts, or frozen split identities are inconsistent."""


@dataclass(frozen=True)
class EvaluationLabel:
    identity: str
    task_key: TaskKey
    actual_role: SourceRole
    disposition: Disposition | None
    operator_disposition: Disposition | None
    failure_label: str | None

    def __post_init__(self) -> None:
        if not self.identity.strip():
            raise EvaluationError("evaluation label identity must be non-blank")
        if self.task_key not in ("single_can", "two_can"):
            raise EvaluationError("evaluation label task_key is invalid")
        if self.actual_role == "success":
            if any(
                value is not None
                for value in (self.disposition, self.operator_disposition, self.failure_label)
            ):
                raise EvaluationError("success labels cannot contain failure fields")
        elif self.actual_role == "failure":
            if self.disposition not in _DISPOSITIONS or self.operator_disposition not in _DISPOSITIONS:
                raise EvaluationError("failure labels require valid dispositions")
            if self.failure_label is not None and not self.failure_label.strip():
                raise EvaluationError("failure_label must be null or non-blank")
        else:
            raise EvaluationError("evaluation label actual_role is invalid")


@dataclass(frozen=True)
class EvaluationReport:
    evaluated_identities: tuple[str, ...]
    confusion: dict[str, int]
    precision: float
    recall: float
    f1: float
    false_rejection_rate: float
    by_task: dict[str, dict[str, float | int]]
    by_disposition: dict[str, dict[str, float | int]]
    by_failure_label: dict[str, dict[str, float | int]]
    false_positive_identities: tuple[str, ...]
    false_negative_identities: tuple[str, ...]


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise EvaluationError(f"{field} must be a sequence")
    return value


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{field} must be a non-blank string")
    return value


def build_evaluation_labels(
    inventory: tuple[InventoryRow, ...], hub: HubReader
) -> tuple[EvaluationLabel, ...]:
    """Join pinned FAILURE_INDEX metadata only after threshold/verdict freezing."""

    rows: dict[str, InventoryRow] = {}
    for row in inventory:
        identity = row.identity.canonical
        if identity in rows:
            raise EvaluationError(f"inventory contains duplicate identity {identity}")
        rows[identity] = row

    labels: list[EvaluationLabel] = [
        EvaluationLabel(identity, row.task_key, "success", None, None, None)
        for identity, row in rows.items()
        if row.role == "success"
    ]
    failures_by_source: dict[tuple[str, str], list[InventoryRow]] = defaultdict(list)
    for row in rows.values():
        if row.role == "failure":
            if row.attempt_id is None or row.identity.source_key != row.attempt_id:
                raise EvaluationError(
                    f"failure inventory canonical source key must equal attempt_id for {row.identity.canonical}"
                )
            failures_by_source[(row.identity.repo_id, row.identity.revision)].append(row)

    for (repo_id, revision), source_rows in sorted(failures_by_source.items()):
        document = _mapping(
            hub.read_json(repo_id, revision, "FAILURE_INDEX.json"),
            f"{repo_id}@{revision}:FAILURE_INDEX.json",
        )
        if document.get("training_eligible") is not False:
            raise EvaluationError(f"{repo_id} failure labels must be training-ineligible")
        raw_attempts = _sequence(document.get("attempts"), f"{repo_id} attempts")
        if document.get("attempt_count") != len(raw_attempts):
            raise EvaluationError(f"{repo_id} attempt_count does not match attempts")
        attempts: dict[str, Mapping[str, Any]] = {}
        for index, raw_attempt in enumerate(raw_attempts):
            attempt = _mapping(raw_attempt, f"{repo_id} attempts[{index}]")
            attempt_id = _nonblank(attempt.get("attempt_id"), f"{repo_id} attempt_id")
            if attempt_id in attempts:
                raise EvaluationError(f"{repo_id} label join has duplicate attempt_id {attempt_id}")
            attempts[attempt_id] = attempt
        expected_attempts = {
            row.attempt_id for row in source_rows if row.attempt_id is not None
        }
        if len(expected_attempts) != len(source_rows) or set(attempts) != expected_attempts:
            raise EvaluationError(
                f"{repo_id} label join must exactly match failure inventory attempts"
            )
        for row in source_rows:
            assert row.attempt_id is not None
            attempt = attempts[row.attempt_id]
            if attempt.get("training_included") is not False:
                raise EvaluationError(f"{repo_id} attempt {row.attempt_id} must be training-ineligible")
            disposition = _nonblank(
                attempt.get("disposition"), f"{repo_id} attempt {row.attempt_id}.disposition"
            )
            operator_disposition = _nonblank(
                attempt.get("operator_disposition"),
                f"{repo_id} attempt {row.attempt_id}.operator_disposition",
            )
            if "failure_label" not in attempt:
                raise EvaluationError(
                    f"{repo_id} attempt {row.attempt_id}.failure_label key must be present"
                )
            failure_label = attempt["failure_label"]
            if failure_label is not None:
                failure_label = _nonblank(
                    failure_label, f"{repo_id} attempt {row.attempt_id}.failure_label"
                )
            labels.append(
                EvaluationLabel(
                    row.identity.canonical,
                    row.task_key,
                    "failure",
                    cast(Disposition, disposition),
                    cast(Disposition, operator_disposition),
                    failure_label,
                )
            )

    label_identities = [label.identity for label in labels]
    if len(label_identities) != len(set(label_identities)) or set(label_identities) != set(rows):
        raise EvaluationError("evaluation label join must cover inventory exactly once")
    return tuple(sorted(labels, key=lambda label: label.identity))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _stats(records: Sequence[tuple[EvaluationLabel, EpisodeVerdict]]) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for label, verdict in records:
        actual = label.actual_role == "failure"
        predicted = verdict.verdict in ("REVIEW", "REJECT")
        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif not actual and not predicted:
            tn += 1
        else:
            fn += 1
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "total": len(records),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "predicted_questionable": tp + fp,
        "actual_questionable": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "false_rejection_rate": _ratio(fp, fp + tn),
    }


def _breakdown(
    records: Sequence[tuple[EvaluationLabel, EpisodeVerdict]], key: Any
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[tuple[EvaluationLabel, EpisodeVerdict]]] = defaultdict(list)
    for record in records:
        category = key(record[0])
        if category is not None:
            groups[str(category)].append(record)
    return {category: _stats(groups[category]) for category in sorted(groups)}


def evaluate_verdicts(
    snapshot: ThresholdSnapshot,
    verdicts: tuple[EpisodeVerdict, ...],
    labels: tuple[EvaluationLabel, ...],
) -> EvaluationReport:
    """Evaluate only held-out successes and every failure after scoring is frozen."""

    if (
        not snapshot.payload_digest.strip()
        or threshold_snapshot_digest(snapshot) != snapshot.payload_digest
    ):
        raise EvaluationError("threshold snapshot digest does not match its frozen payload")
    if len(set(snapshot.calibration_identities)) != len(snapshot.calibration_identities):
        raise EvaluationError("calibration identities must be unique")
    if len(set(snapshot.held_out_success_identities)) != len(snapshot.held_out_success_identities):
        raise EvaluationError("held-out identities must be unique")
    if set(snapshot.calibration_identities) & set(snapshot.held_out_success_identities):
        raise EvaluationError("calibration and held-out identities must be disjoint")
    if len(snapshot.calibration_identities) != 61 or len(snapshot.held_out_success_identities) != 16:
        raise EvaluationError("snapshot must contain the locked 61/16 success split")

    labels_by_identity: dict[str, EvaluationLabel] = {}
    for label in labels:
        if label.identity in labels_by_identity:
            raise EvaluationError(f"duplicate label identity {label.identity}")
        labels_by_identity[label.identity] = label
    success_identities = {
        label.identity for label in labels if label.actual_role == "success"
    }
    failure_identities = {
        label.identity for label in labels if label.actual_role == "failure"
    }
    frozen_successes = set(snapshot.calibration_identities) | set(
        snapshot.held_out_success_identities
    )
    if success_identities != frozen_successes or len(failure_identities) != 25:
        raise EvaluationError(
            "labels must contain exactly frozen successes and all 25 failures"
        )
    if len(labels_by_identity) != 102:
        raise EvaluationError("labels must contain exactly 102 unique inventory identities")
    calibration_tasks = Counter(
        labels_by_identity[identity].task_key for identity in snapshot.calibration_identities
    )
    held_out_tasks = Counter(
        labels_by_identity[identity].task_key for identity in snapshot.held_out_success_identities
    )
    failure_labels = [
        label for label in labels_by_identity.values() if label.actual_role == "failure"
    ]
    failure_tasks = Counter(label.task_key for label in failure_labels)
    dispositions = Counter(label.disposition for label in failure_labels)
    operator_dispositions = Counter(
        label.operator_disposition for label in failure_labels
    )
    semantic_labels = Counter(
        "__unlabeled__" if label.failure_label is None else label.failure_label
        for label in failure_labels
    )
    if calibration_tasks != {"single_can": 41, "two_can": 20}:
        raise EvaluationError("calibration labels must preserve the locked 41/20 task split")
    if held_out_tasks != {"single_can": 11, "two_can": 5}:
        raise EvaluationError("held-out labels must preserve the locked 11/5 task split")
    if failure_tasks != {"single_can": 19, "two_can": 6}:
        raise EvaluationError("failure labels must preserve the locked 19/6 task split")
    if dispositions != {"aborted": 17, "collector_error": 5, "failed": 3}:
        raise EvaluationError("failure labels must preserve pinned disposition counts")
    if operator_dispositions != {"aborted": 19, "collector_error": 3, "failed": 3}:
        raise EvaluationError("failure labels must preserve pinned operator_disposition counts")
    if semantic_labels != {
        "__unlabeled__": 19,
        "dropped_object": 2,
        "hardware_or_control_problem": 2,
        "test_or_setup": 2,
    }:
        raise EvaluationError("failure labels must preserve pinned failure-label counts")

    evaluated = set(snapshot.held_out_success_identities) | failure_identities
    verdicts_by_identity: dict[str, EpisodeVerdict] = {}
    metrics_digests: set[str] = set()
    verdict_digests: set[str] = set()
    for verdict in verdicts:
        if verdict.identity in verdicts_by_identity:
            raise EvaluationError(f"duplicate verdict identity {verdict.identity}")
        if not verdict.metrics_digest.strip():
            raise EvaluationError(f"verdict {verdict.identity} metrics digest must be non-blank")
        if verdict.verdict not in ("PASS", "REVIEW", "REJECT"):
            raise EvaluationError(f"verdict {verdict.identity} has an invalid value")
        if verdict.threshold_digest != snapshot.payload_digest:
            raise EvaluationError(
                f"verdict {verdict.identity} threshold digest does not match snapshot"
            )
        if (
            not verdict.verdict_digest.strip()
            or episode_verdict_digest(verdict) != verdict.verdict_digest
        ):
            raise EvaluationError(
                f"verdict {verdict.identity} digest does not match its frozen payload"
            )
        if verdict.metrics_digest in metrics_digests:
            raise EvaluationError(f"verdict {verdict.identity} reuses a metrics digest")
        if verdict.verdict_digest in verdict_digests:
            raise EvaluationError(f"verdict {verdict.identity} reuses a verdict digest")
        metrics_digests.add(verdict.metrics_digest)
        verdict_digests.add(verdict.verdict_digest)
        verdicts_by_identity[verdict.identity] = verdict
    if set(verdicts_by_identity) != evaluated or len(evaluated) != 41:
        raise EvaluationError(
            "verdict identities must equal 16 held-out successes plus all 25 failures"
        )

    records = [
        (labels_by_identity[identity], verdicts_by_identity[identity])
        for identity in sorted(evaluated)
    ]
    overall = _stats(records)
    confusion = {key: int(overall[key]) for key in ("tp", "fp", "tn", "fn")}
    false_positives = tuple(
        label.identity
        for label, verdict in records
        if label.actual_role == "success" and verdict.verdict in ("REVIEW", "REJECT")
    )
    false_negatives = tuple(
        label.identity
        for label, verdict in records
        if label.actual_role == "failure" and verdict.verdict == "PASS"
    )
    failure_records = [record for record in records if record[0].actual_role == "failure"]
    return EvaluationReport(
        evaluated_identities=tuple(sorted(evaluated)),
        confusion=confusion,
        precision=float(overall["precision"]),
        recall=float(overall["recall"]),
        f1=float(overall["f1"]),
        false_rejection_rate=float(overall["false_rejection_rate"]),
        by_task=_breakdown(records, lambda label: label.task_key),
        by_disposition=_breakdown(failure_records, lambda label: label.disposition),
        by_failure_label=_breakdown(
            failure_records,
            lambda label: "__unlabeled__" if label.failure_label is None else label.failure_label,
        ),
        false_positive_identities=false_positives,
        false_negative_identities=false_negatives,
    )
