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


def review_revision(metadata: dict[str, Any]) -> int:
    """Return the optimistic-concurrency revision for a review sidecar."""

    value = metadata.get("review_revision", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Attempt review revision is invalid")
    return value


def _require_review_revision(metadata: dict[str, Any], expected_revision: Any) -> int:
    current = review_revision(metadata)
    if expected_revision is None:
        return current
    if isinstance(expected_revision, bool):
        raise ValueError("Attempt review revision is invalid")
    try:
        expected = int(expected_revision)
    except (TypeError, ValueError) as exc:
        raise ValueError("Attempt review revision is invalid") from exc
    if expected != current:
        raise ValueError(
            "This attempt was reviewed in another tab; refresh the archive before saving"
        )
    return current


def _append_review_history(
    metadata: dict[str, Any],
    *,
    action: str,
    previous: dict[str, Any],
    label: str,
    note: str,
    revision: int,
) -> str:
    reviewed_at = utc_now()
    history = metadata.get("review_history")
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "revision": revision,
            "reviewed_at": reviewed_at,
            "action": action,
            "previous": previous,
            "failure_label": label,
            "failure_note": note,
        }
    )
    metadata["review_history"] = history
    metadata["review_revision"] = revision
    metadata["reviewed_at"] = reviewed_at
    return reviewed_at


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
    *,
    expected_revision: Any = None,
) -> dict[str, Any]:
    directory, metadata = find_attempt(archive_root, attempt_id)
    manually_failed = bool(
        metadata.get("disposition") == "failed"
        or metadata.get("operator_disposition") == "failed"
    )
    already_excluded = bool(
        metadata.get("archive_complete") is True
        and metadata.get("training_included") is False
        and metadata.get("disposition") not in {"kept", "recording"}
    )
    if not (manually_failed or already_excluded):
        raise ValueError(
            "Only failed or already-excluded attempts can have a failure label"
        )
    current_revision = _require_review_revision(metadata, expected_revision)
    normalized, normalized_note = validate_failure_label(label, note)
    previous = {
        "disposition": metadata.get("disposition"),
        "operator_disposition": metadata.get("operator_disposition"),
        "failure_label": metadata.get("failure_label"),
        "failure_note": metadata.get("failure_note"),
        "training_included": metadata.get("training_included"),
        "training_episode_index": metadata.get("training_episode_index"),
    }
    metadata["failure_label"] = normalized
    metadata["failure_note"] = normalized_note
    reviewed_at = _append_review_history(
        metadata,
        action="update_failure_label" if manually_failed else "classify_excluded_attempt",
        previous=previous,
        label=normalized,
        note=normalized_note,
        revision=current_revision + 1,
    )
    metadata["label_updated_at"] = reviewed_at
    atomic_write_json(directory / "metadata.json", metadata)
    return metadata


def mark_kept_attempt_failed(
    archive_root: Path,
    attempt_id: str,
    label: Any,
    note: Any = "",
    *,
    expected_revision: Any = None,
) -> dict[str, Any]:
    """Commit the sidecar half of a post-hoc LeRobot exclusion.

    The caller must remove and verify the corresponding physical LeRobot
    episode first. This function deliberately refuses metadata-only exclusion.
    """

    directory, metadata = find_attempt(archive_root, attempt_id)
    if metadata.get("archive_complete") is not True:
        raise ValueError("Only a completed attempt can be reviewed")
    if not (
        metadata.get("disposition") == "kept"
        and metadata.get("training_included") is True
        and isinstance(metadata.get("training_episode_index"), int)
    ):
        raise ValueError("Attempt is not an included kept LeRobot episode")
    current_revision = _require_review_revision(metadata, expected_revision)
    normalized, normalized_note = validate_failure_label(label, note)
    previous_episode_index = int(metadata["training_episode_index"])
    previous = {
        "disposition": metadata.get("disposition"),
        "operator_disposition": metadata.get("operator_disposition"),
        "failure_label": metadata.get("failure_label"),
        "failure_note": metadata.get("failure_note"),
        "training_included": True,
        "training_episode_index": previous_episode_index,
    }
    metadata.setdefault("recorded_disposition", metadata.get("disposition"))
    metadata.setdefault(
        "recorded_operator_disposition", metadata.get("operator_disposition")
    )
    metadata["disposition"] = "failed"
    metadata["operator_disposition"] = "failed"
    metadata["review_disposition"] = "failed"
    metadata["failure_label"] = normalized
    metadata["failure_note"] = normalized_note
    metadata["training_included"] = False
    metadata["training_episode_index"] = None
    metadata["previous_training_episode_index"] = previous_episode_index
    reviewed_at = _append_review_history(
        metadata,
        action="mark_failed_and_exclude_from_lerobot",
        previous=previous,
        label=normalized,
        note=normalized_note,
        revision=current_revision + 1,
    )
    metadata["label_updated_at"] = reviewed_at
    metadata["training_excluded_at"] = reviewed_at
    atomic_write_json(directory / "metadata.json", metadata)
    return metadata


def reindex_training_episode(
    archive_root: Path,
    attempt_id: str,
    *,
    old_index: int,
    new_index: int,
    review_attempt_id: str,
) -> dict[str, Any]:
    """Update an included attempt after a reviewed episode is compacted out."""

    directory, metadata = find_attempt(archive_root, attempt_id)
    if not (
        metadata.get("training_included") is True
        and metadata.get("training_episode_index") == old_index
    ):
        raise ValueError("Attempt-to-LeRobot episode mapping changed during review")
    changed_at = utc_now()
    history = metadata.get("training_reindex_history")
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "changed_at": changed_at,
            "old_index": old_index,
            "new_index": new_index,
            "caused_by_review_attempt_id": review_attempt_id,
        }
    )
    metadata["training_reindex_history"] = history
    metadata["training_episode_index"] = new_index
    metadata["training_reindexed_at"] = changed_at
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
