"""Hardware-free held-out rollout validation, aggregation, and reporting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
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


@dataclass(frozen=True)
class CheckpointIdentity:
    path: Path
    digest: str


def checkpoint_identity(path: Path | str) -> CheckpointIdentity:
    """Bind the accepted checkpoint layout to its resolved path and weights."""

    checkpoint = Path(path).expanduser().resolve()
    weights = checkpoint / "model.safetensors"
    if weights.is_symlink() or not weights.is_file():
        raise ValueError("Checkpoint model.safetensors is missing or not a regular file")
    digest = hashlib.sha256()
    try:
        with weights.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"Checkpoint model.safetensors cannot be read: {exc}") from exc
    return CheckpointIdentity(path=checkpoint, digest=digest.hexdigest())


@dataclass(frozen=True)
class _TerminalAudit:
    attempt: int
    terminal_reason: str
    clamps: int
    elapsed_seconds: float


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
    audit_sources = [
        source
        for trial in trials
        for source in trial.source_jsonl_paths
    ]
    if len(set(audit_sources)) != len(audit_sources):
        raise ValueError("source JSONL path is reused across placements")
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
    checkpoint_paths = [manifest.checkpoint for manifest in manifests]
    if len(set(checkpoint_paths)) != len(checkpoint_paths):
        raise ValueError("Checkpoint comparison requires distinct checkpoint identities")
    checkpoint_digests = [manifest.checkpoint_digest for manifest in manifests]
    if len(set(checkpoint_digests)) != len(checkpoint_digests):
        raise ValueError("Checkpoint comparison requires distinct checkpoint digests")
    expected_placements = frozenset(
        trial.placement_id for trial in manifests[0].trials
    )
    for manifest in manifests[1:]:
        placements = frozenset(trial.placement_id for trial in manifest.trials)
        if placements != expected_placements:
            raise ValueError(
                "Checkpoint comparison requires the same held-out placement IDs"
            )

    ranked = sorted(
        manifests,
        key=lambda manifest: (
            -sum(trial.placement_success for trial in manifest.trials),
            sum(trial.safety_faults for trial in manifest.trials),
            sum(trial.clamps for trial in manifest.trials),
            math.fsum(trial.completion_s for trial in manifest.trials)
            / len(manifest.trials),
            manifest.checkpoint,
        ),
    )
    return [summarize(manifest) for manifest in ranked]


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

    trial = Trial(
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
    _validate_trial_audits(trial)
    return trial


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
    if len(set(paths)) != len(paths):
        raise ValueError("source JSONL paths must be distinct")
    for source in paths:
        path = Path(source)
        if not path.is_absolute():
            raise ValueError("source JSONL paths must be absolute")
        if path.suffix != ".jsonl":
            raise ValueError("source JSONL paths must name .jsonl files")
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                "source JSONL paths must be existing regular non-symlink files"
            )
    return paths


def _validate_trial_audits(trial: Trial) -> None:
    terminal_rows: list[_TerminalAudit] = []
    for source in trial.source_jsonl_paths:
        terminal_rows.extend(
            _read_terminal_audits(
                Path(source),
                checkpoint=trial.checkpoint,
                checkpoint_digest=trial.checkpoint_digest,
            )
        )

    attempts = sorted(row.attempt for row in terminal_rows)
    expected_attempts = list(range(1, trial.attempts_used + 1))
    if attempts != expected_attempts:
        raise ValueError(
            "source JSONL terminal attempts must be exactly 1..attempts_used"
        )
    ordered = sorted(terminal_rows, key=lambda row: row.attempt)
    if any(row.terminal_reason != "operator_failure" for row in ordered[:-1]):
        raise ValueError(
            "source JSONL non-final attempts must end in operator_failure"
        )
    if ordered[-1].terminal_reason != trial.terminal_reason:
        raise ValueError(
            "source JSONL final terminal reason does not match the manifest"
        )

    audit_clamps = sum(row.clamps for row in ordered)
    if audit_clamps != trial.clamps:
        raise ValueError("source JSONL clamp count does not match the manifest")
    audit_safety_faults = sum(
        row.terminal_reason == "safety_fault" for row in ordered
    )
    if audit_safety_faults != trial.safety_faults:
        raise ValueError(
            "source JSONL safety fault count does not match the manifest"
        )
    audit_completion_s = math.fsum(row.elapsed_seconds for row in ordered)
    if not math.isclose(
        audit_completion_s,
        trial.completion_s,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "source JSONL completion seconds do not match the manifest"
        )


def _read_terminal_audits(
    path: Path,
    *,
    checkpoint: str,
    checkpoint_digest: str,
) -> list[_TerminalAudit]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read source JSONL at {path}: {exc}") from exc
    terminal_rows: list[_TerminalAudit] = []
    metadata_rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"Source JSONL {path}:{line_number} is blank")
        try:
            row = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Source JSONL {path}:{line_number} is invalid: {exc}"
            ) from exc
        if not isinstance(row, Mapping):
            raise ValueError(
                f"Source JSONL {path}:{line_number} must contain an object"
            )
        if row.get("event") == "rollout_metadata":
            metadata_rows.append(row)
            continue
        if row.get("event") not in ("terminal", "terminal_fallback"):
            continue
        attempt = _integer(row.get("attempt"), "source JSONL terminal attempt")
        if attempt < 1:
            raise ValueError("source JSONL terminal attempt must be positive")
        terminal_reason = _nonempty_string(
            row.get("terminal_reason"),
            "source JSONL terminal reason",
        )
        if terminal_reason not in TERMINAL_REASONS:
            raise ValueError("source JSONL terminal reason is unrecognized")
        terminal_rows.append(
            _TerminalAudit(
                attempt=attempt,
                terminal_reason=terminal_reason,
                clamps=_nonnegative_integer(
                    row.get("clamp_count"),
                    "source JSONL clamp count",
                ),
                elapsed_seconds=_nonnegative_finite(
                    row.get("elapsed_seconds"),
                    "source JSONL elapsed seconds",
                ),
            )
        )
    if len(metadata_rows) != 1:
        raise ValueError("source JSONL must contain exactly one rollout_metadata row")
    metadata = metadata_rows[0]
    if metadata.get("profile_authentication") != "checkpoint-sidecar-verified":
        raise ValueError("source JSONL does not contain an authenticated checkpoint")
    if metadata.get("checkpoint") != checkpoint:
        raise ValueError("source JSONL checkpoint does not match the manifest")
    if metadata.get("checkpoint_digest") != checkpoint_digest:
        raise ValueError("source JSONL checkpoint digest does not match the manifest")
    return terminal_rows


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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real JSON number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Nonfinite JSON constant is not permitted: {value}")
