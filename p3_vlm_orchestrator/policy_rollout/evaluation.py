"""Hardware-free held-out rollout validation, aggregation, and reporting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
from typing import Any


SUMMARY_FIELDS = (
    "checkpoint",
    "trials",
    "grasp_successes",
    "placement_successes",
    "safety_faults",
    "clamps",
    "mean_completion_s",
    "overall_success_rate",
)
MANIFEST_FIELDS = frozenset(
    ("schema_version", "checkpoint", "checkpoint_digest", "trials")
)
TRIAL_FIELDS = frozenset(
    (
        "checkpoint",
        "checkpoint_digest",
        "placement_id",
        "attempts_used",
        "grasp_success",
        "placement_success",
        "terminal_reason",
        "safety_faults",
        "clamps",
        "completion_s",
        "source_jsonl_paths",
    )
)
TERMINAL_REASONS = frozenset(
    ("operator_success", "operator_failure", "stopped", "timeout", "safety_fault")
)
_OUTPUT_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_PROTECTED_OUTPUT_MARKERS = (
    "credential",
    "secret",
    "calibration",
    "dataset",
    ".env",
)


@dataclass(frozen=True)
class Trial:
    checkpoint: str
    checkpoint_digest: str
    placement_id: str
    attempts_used: int
    grasp_success: bool
    placement_success: bool
    terminal_reason: str
    safety_faults: int
    clamps: int
    completion_s: float
    source_jsonl_paths: tuple[str, ...]


@dataclass(frozen=True)
class TrialManifest:
    source_path: Path
    checkpoint: str
    checkpoint_digest: str
    trials: tuple[Trial, ...]


@dataclass(frozen=True)
class ReportPaths:
    json_path: Path
    csv_path: Path


def load_trial_manifest(path: Path | str) -> TrialManifest:
    """Read and strictly validate one checkpoint's final held-out trials."""

    source = Path(path)
    try:
        document = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Cannot read trial manifest at {source}: {exc}") from exc
    if not isinstance(document, Mapping) or set(document) != MANIFEST_FIELDS:
        raise ValueError("Trial manifest schema is not exact")
    if document.get("schema_version") != 1:
        raise ValueError("Trial manifest schema_version must be exactly 1")

    checkpoint = _nonempty_string(document.get("checkpoint"), "checkpoint")
    digest = _checkpoint_digest(document.get("checkpoint_digest"))
    rows = document.get("trials")
    if not isinstance(rows, list) or not 10 <= len(rows) <= 15:
        raise ValueError("Trial manifest must contain 10 to 15 final trials")

    trials = tuple(
        _parse_trial(row, index=index, checkpoint=checkpoint, digest=digest)
        for index, row in enumerate(rows)
    )
    placement_ids = [trial.placement_id for trial in trials]
    if len(set(placement_ids)) != len(placement_ids):
        raise ValueError("Trials must use distinct placement IDs")
    return TrialManifest(
        source_path=source,
        checkpoint=checkpoint,
        checkpoint_digest=digest,
        trials=trials,
    )


def summarize(manifest: TrialManifest) -> dict[str, object]:
    """Aggregate one validated manifest using every final trial duration."""

    trials = manifest.trials
    count = len(trials)
    placement_successes = sum(trial.placement_success for trial in trials)
    return {
        "checkpoint": manifest.checkpoint,
        "trials": count,
        "grasp_successes": sum(trial.grasp_success for trial in trials),
        "placement_successes": placement_successes,
        "safety_faults": sum(trial.safety_faults for trial in trials),
        "clamps": sum(trial.clamps for trial in trials),
        "mean_completion_s": round(
            math.fsum(trial.completion_s for trial in trials) / count, 6
        ),
        "overall_success_rate": round(placement_successes / count, 6),
    }


def compare(manifests: Sequence[TrialManifest]) -> list[dict[str, object]]:
    """Rank checkpoints on an identical held-out placement set."""

    if len(manifests) < 2:
        raise ValueError("Checkpoint comparison requires at least two manifests")
    checkpoints = [(manifest.checkpoint, manifest.checkpoint_digest) for manifest in manifests]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Checkpoint comparison requires distinct checkpoint identities")
    expected_placements = frozenset(
        trial.placement_id for trial in manifests[0].trials
    )
    for manifest in manifests[1:]:
        placements = frozenset(trial.placement_id for trial in manifest.trials)
        if placements != expected_placements:
            raise ValueError(
                "Checkpoint comparison requires the same held-out placement IDs"
            )

    summaries = [summarize(manifest) for manifest in manifests]
    return sorted(
        summaries,
        key=lambda row: (
            -int(row["placement_successes"]),
            -float(row["overall_success_rate"]),
            int(row["safety_faults"]),
            int(row["clamps"]),
            float(row["mean_completion_s"]),
            str(row["checkpoint"]),
        ),
    )


def write_reports(
    manifests: Sequence[TrialManifest],
    *,
    output_name: str,
    repo_root: Path | str,
) -> ReportPaths:
    """Write deterministic JSON and CSV without overwriting prior reports."""

    report_root = _report_root(repo_root, output_name)
    json_path = report_root / f"{output_name}.json"
    csv_path = report_root / f"{output_name}.csv"
    for output in (json_path, csv_path):
        if output.exists() or output.is_symlink():
            raise ValueError(f"Report output already exists: {output}")

    if len(manifests) == 1:
        rows = [summarize(manifests[0])]
    else:
        rows = compare(manifests)
    _validate_summary_rows(rows)

    json_text = json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    csv_buffer = StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=SUMMARY_FIELDS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    csv_text = csv_buffer.getvalue()

    created: list[Path] = []
    try:
        for output, text in ((json_path, json_text), (csv_path, csv_text)):
            with output.open("x", encoding="utf-8", newline="") as handle:
                handle.write(text)
                handle.flush()
            created.append(output)
    except Exception:
        for output in created:
            output.unlink(missing_ok=True)
        raise
    return ReportPaths(json_path=json_path.resolve(), csv_path=csv_path.resolve())


def _parse_trial(
    value: object,
    *,
    index: int,
    checkpoint: str,
    digest: str,
) -> Trial:
    if not isinstance(value, Mapping) or set(value) != TRIAL_FIELDS:
        raise ValueError(f"Trial {index} schema is not exact")
    trial_checkpoint = _nonempty_string(value.get("checkpoint"), "trial checkpoint")
    trial_digest = _checkpoint_digest(value.get("checkpoint_digest"))
    if trial_checkpoint != checkpoint or trial_digest != digest:
        raise ValueError(f"Trial {index} has mixed checkpoint identity or digest")
    placement_id = _nonempty_string(value.get("placement_id"), "placement ID")
    attempts_used = _integer(value.get("attempts_used"), "attempts_used")
    if attempts_used not in (1, 2):
        raise ValueError("attempts_used must be exactly 1 or 2")
    grasp_success = _boolean(value.get("grasp_success"), "grasp_success")
    placement_success = _boolean(
        value.get("placement_success"), "placement_success"
    )
    terminal_reason = _nonempty_string(
        value.get("terminal_reason"), "terminal_reason"
    )
    if terminal_reason not in TERMINAL_REASONS:
        raise ValueError(f"Unrecognized terminal reason: {terminal_reason}")
    safety_faults = _nonnegative_integer(
        value.get("safety_faults"), "safety_faults"
    )
    clamps = _nonnegative_integer(value.get("clamps"), "clamps")
    completion_s = _nonnegative_finite(value.get("completion_s"), "completion_s")
    source_paths = _source_paths(value.get("source_jsonl_paths"))
    if len(source_paths) != attempts_used:
        raise ValueError(
            "source JSONL path count must exactly match attempts_used"
        )

    if placement_success and not grasp_success:
        raise ValueError("Placement success requires grasp success")
    if terminal_reason == "operator_success" and not placement_success:
        raise ValueError("Operator success requires placement success")
    if placement_success and terminal_reason != "operator_success":
        raise ValueError("Placement success requires operator_success terminal reason")
    if terminal_reason == "safety_fault":
        if grasp_success or placement_success:
            raise ValueError("Safety-fault terminal outcomes cannot be successful")
        if safety_faults < 1:
            raise ValueError("Safety-fault terminal outcome must count a safety fault")
    elif safety_faults != 0:
        raise ValueError("Safety fault counts require a safety_fault terminal outcome")

    return Trial(
        checkpoint=trial_checkpoint,
        checkpoint_digest=trial_digest,
        placement_id=placement_id,
        attempts_used=attempts_used,
        grasp_success=grasp_success,
        placement_success=placement_success,
        terminal_reason=terminal_reason,
        safety_faults=safety_faults,
        clamps=clamps,
        completion_s=completion_s,
        source_jsonl_paths=source_paths,
    )


def _report_root(repo_root: Path | str, output_name: str) -> Path:
    if not isinstance(output_name, str) or not _OUTPUT_NAME.fullmatch(output_name):
        raise ValueError("Report output name must be a lowercase slug")
    if any(marker in output_name for marker in _PROTECTED_OUTPUT_MARKERS):
        raise ValueError("Report output name cannot reference protected content")
    repository = Path(os.path.abspath(Path(repo_root).expanduser()))
    report_root = repository / "runs" / "policy" / "reports"
    for path in (
        repository,
        repository / "runs",
        repository / "runs" / "policy",
        report_root,
    ):
        if path.is_symlink():
            raise ValueError("Report output path must not use a symlink")
    report_root.mkdir(parents=True, exist_ok=True)
    if report_root.is_symlink():
        raise ValueError("Report output path must not use a symlink")
    resolved_root = report_root.resolve(strict=True)
    expected_root = (repository / "runs" / "policy" / "reports").resolve(strict=True)
    if resolved_root != expected_root:
        raise ValueError("Report output must remain under runs/policy/reports")
    return resolved_root


def _source_paths(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("source JSONL paths must be a nonempty list")
    paths = tuple(_nonempty_string(item, "source JSONL path") for item in value)
    if any(Path(path).suffix != ".jsonl" for path in paths):
        raise ValueError("source JSONL paths must name .jsonl files")
    return paths


def _validate_summary_rows(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        if tuple(row) != SUMMARY_FIELDS:
            raise ValueError("Summary schema is not exact")


def _checkpoint_digest(value: object) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError("checkpoint_digest must be a lowercase SHA-256 digest")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    result = _integer(value, label)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _nonnegative_finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and nonnegative")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Nonfinite JSON constant is not permitted: {value}")
