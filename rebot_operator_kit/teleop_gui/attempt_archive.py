"""Persistent, reviewable attempt archives for ReBot data collection.

The LeRobot dataset is deliberately success-only.  Every physical attempt is
also written to a separate review archive so failed takes are not destroyed and
can be labelled without contaminating the policy-training dataset.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


FAILURE_LABELS: tuple[dict[str, str], ...] = (
    {"value": "failed_to_perform_task", "label": "Failed to perform task"},
    {"value": "test_or_setup", "label": "Test / setup"},
    {"value": "missed_grasp", "label": "Missed grasp"},
    {"value": "dropped_object", "label": "Dropped object"},
    {"value": "wrong_destination", "label": "Wrong destination"},
    {"value": "collision_or_safety_stop", "label": "Collision / safety stop"},
    {"value": "camera_problem", "label": "Camera problem"},
    {"value": "operator_intervention", "label": "Operator intervention"},
    {"value": "hardware_or_control_problem", "label": "Hardware / control problem"},
    {"value": "bad_start", "label": "Bad starting state"},
    {"value": "other", "label": "Other"},
)
FAILURE_LABEL_VALUES = frozenset(item["value"] for item in FAILURE_LABELS)
ATTEMPT_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}\.[0-9]{6}Z-[0-9a-f]{10}$")
DATASET_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,47}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_attempt_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}-{uuid4().hex[:10]}"


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate_failure_label(label: Any, note: Any = "") -> tuple[str, str]:
    normalized = str(label or "").strip().lower()
    normalized_note = str(note or "").strip()
    if normalized not in FAILURE_LABEL_VALUES:
        raise ValueError("Choose a failure reason before marking this attempt failed")
    if normalized == "other" and not normalized_note:
        raise ValueError("Add a note when the failure reason is Other")
    if len(normalized_note) > 500:
        raise ValueError("Failure note must be 500 characters or fewer")
    return normalized, normalized_note


def attempt_path(archive_root: Path, dataset: str, attempt_id: str) -> Path:
    if not DATASET_RE.fullmatch(dataset):
        raise ValueError("Invalid dataset name")
    if not ATTEMPT_ID_RE.fullmatch(attempt_id):
        raise ValueError("Invalid attempt ID")
    root = archive_root.resolve()
    candidate = (root / dataset / attempt_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Attempt path leaves the archive root") from exc
    return candidate


def _artifact_status(directory: Path, filename: str, committed: bool) -> dict[str, Any]:
    path = directory / filename
    available = bool(
        committed
        and path.is_file()
        and not path.is_symlink()
        and path.stat().st_size > 0
    )
    return {
        "name": filename,
        "available": available,
        "bytes": path.stat().st_size if available else 0,
    }


def attempt_inventory(archive_root: Path, dataset: str = "") -> list[dict[str, Any]]:
    if dataset and not DATASET_RE.fullmatch(dataset):
        raise ValueError("Invalid dataset name")
    roots = [archive_root / dataset] if dataset else list(archive_root.glob("*"))
    attempts: list[dict[str, Any]] = []
    for dataset_root in roots:
        if not dataset_root.is_dir() or not DATASET_RE.fullmatch(dataset_root.name):
            continue
        for metadata_path in dataset_root.glob("*/metadata.json"):
            directory = metadata_path.parent
            if not ATTEMPT_ID_RE.fullmatch(directory.name):
                continue
            try:
                metadata = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            committed = metadata.get("archive_complete") is True
            attempts.append(
                {
                    **metadata,
                    "dataset": dataset_root.name,
                    "attempt_id": directory.name,
                    "artifacts": {
                        "overhead": _artifact_status(directory, "overhead.mp4", committed),
                        "wrist": _artifact_status(directory, "wrist.mp4", committed),
                        "rerun": _artifact_status(directory, "attempt.rrd", committed),
                    },
                }
            )
    attempts.sort(
        key=lambda item: str(item.get("started_at") or item.get("attempt_id") or ""),
        reverse=True,
    )
    return attempts


def find_attempt(archive_root: Path, attempt_id: str) -> tuple[Path, dict[str, Any]]:
    if not ATTEMPT_ID_RE.fullmatch(str(attempt_id or "")):
        raise ValueError("Invalid attempt ID")
    matches = list(archive_root.glob(f"*/{attempt_id}/metadata.json"))
    if len(matches) != 1:
        raise FileNotFoundError("Attempt was not found")
    metadata_path = matches[0]
    root = archive_root.resolve()
    resolved = metadata_path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Attempt path leaves the archive root") from exc
    try:
        metadata = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt metadata is unreadable") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Attempt metadata is invalid")
    return resolved.parent, metadata


def update_failure_label(
    archive_root: Path,
    attempt_id: str,
    label: Any,
    note: Any = "",
) -> dict[str, Any]:
    directory, metadata = find_attempt(archive_root, attempt_id)
    if not (
        metadata.get("disposition") == "failed"
        or metadata.get("operator_disposition") == "failed"
    ):
        raise ValueError("Only failed attempts can have a failure label")
    normalized, normalized_note = validate_failure_label(label, note)
    metadata["failure_label"] = normalized
    metadata["failure_note"] = normalized_note
    metadata["label_updated_at"] = utc_now()
    atomic_write_json(directory / "metadata.json", metadata)
    return metadata


def artifact_path(archive_root: Path, attempt_id: str, artifact: str) -> Path:
    filenames = {
        "overhead": "overhead.mp4",
        "wrist": "wrist.mp4",
        "rerun": "attempt.rrd",
    }
    if artifact not in filenames:
        raise ValueError("Unknown attempt artifact")
    directory, metadata = find_attempt(archive_root, attempt_id)
    if metadata.get("archive_complete") is not True:
        raise FileNotFoundError("Attempt archive is not committed and verified")
    path = directory / filenames[artifact]
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(directory.resolve())
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(f"{artifact.title()} artifact is not safely available") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{artifact.title()} artifact is not available")
    return resolved
