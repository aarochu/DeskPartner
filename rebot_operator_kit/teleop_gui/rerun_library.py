"""Read-only Rerun attempt catalog for the ReBot Operator Kit.

The current collector remains the sole owner of recording and robot hardware.
This module only reads committed per-attempt archives and returns JSON-safe
catalog rows for a browser.  In particular, it never imports or calls the
legacy ``p5_rerun_port.replay_episode`` physical-trajectory replay path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import quote


GOOD_EPISODE = "Good episode"
BAD_EPISODE = "Bad episode"
NEEDS_REVIEW = "Needs review"
RERUN_TAGS = (GOOD_EPISODE, BAD_EPISODE, NEEDS_REVIEW)

ATTEMPT_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}\.[0-9]{6}Z-[0-9a-f]{10}$")
DATASET_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,47}$")
EXPORT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,95}$")

JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_yaw.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

MEDIA_FILES = {
    "front": "overhead.mp4",
    "overhead": "overhead.mp4",
    "side": "wrist.mp4",
    "wrist": "wrist.mp4",
    "rerun": "attempt.rrd",
}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, number)


def _safe_artifact_status(directory: Path, filename: str, committed: bool) -> dict[str, Any]:
    """Describe one fixed-name artifact without following a symlink."""

    path = directory / filename
    available = False
    size = 0
    if committed:
        try:
            resolved = path.resolve(strict=True)
            available = bool(
                not path.is_symlink()
                and resolved.parent == directory.resolve(strict=True)
                and resolved.is_file()
                and resolved.stat().st_size > 0
            )
            if available:
                size = resolved.stat().st_size
        except OSError:
            available = False
    return {"name": filename, "available": available, "bytes": size}


def _raw_frames_preserved(directory: Path) -> bool:
    """Report a local recovery directory without enumerating or serving it."""

    for name in ("raw_frames", "raw_frame_recovery"):
        candidate = directory / name
        try:
            if (
                not candidate.is_symlink()
                and candidate.resolve(strict=True).parent == directory.resolve(strict=True)
                and candidate.is_dir()
            ):
                return True
        except OSError:
            continue
    return False


def _camera_freshness(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = metadata.get("camera_freshness")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for key in ("front", "side"):
        value = raw.get(key)
        if not isinstance(value, dict):
            continue
        result[key] = {
            "status": str(value.get("status") or "unknown"),
            "sample_attempts": _integer(value.get("sample_attempts")),
            "fresh_samples": _integer(value.get("fresh_samples")),
            "stale_samples_accepted": _integer(
                value.get("stale_samples_accepted", value.get("transient_jitter_samples"))
            ),
            "last_age_ms": _finite_number(value.get("last_age_ms")),
            "max_age_ms_seen": _finite_number(value.get("max_age_ms_seen")),
            "age_enforced": value.get("age_enforced") is True,
        }
    return result


def rerun_tag(metadata: dict[str, Any]) -> str:
    """Map collector outcomes to the three human-facing bounty tags."""

    disposition = str(metadata.get("disposition") or "").strip().lower()
    operator_disposition = str(metadata.get("operator_disposition") or "").strip().lower()
    included = metadata.get("training_included")
    committed = metadata.get("archive_complete") is True
    if committed and disposition == "kept" and included is True:
        return GOOD_EPISODE
    if disposition == "failed" or operator_disposition == "failed":
        return BAD_EPISODE
    return NEEDS_REVIEW


def normalize_attempt(
    metadata: dict[str, Any],
    directory: Path,
    *,
    dataset: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    """Convert one collector sidecar into a stable, JSON-safe catalog row."""

    dataset_name = str(dataset or metadata.get("dataset") or "")
    identifier = str(attempt_id or metadata.get("attempt_id") or "")
    committed = metadata.get("archive_complete") is True
    included = metadata.get("training_included") is True
    episode_index = metadata.get("training_episode_index")
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        episode_index = None

    artifacts = {
        "front": _safe_artifact_status(directory, "overhead.mp4", committed),
        "side": _safe_artifact_status(directory, "wrist.mp4", committed),
        "rerun": _safe_artifact_status(directory, "attempt.rrd", committed),
    }
    encoded = quote(identifier, safe="")
    artifacts["front"]["url"] = (
        f"/api/training/attempt/video?attempt_id={encoded}&camera=overhead"
        if artifacts["front"]["available"]
        else None
    )
    artifacts["side"]["url"] = (
        f"/api/training/attempt/video?attempt_id={encoded}&camera=wrist"
        if artifacts["side"]["available"]
        else None
    )

    session_home = metadata.get("session_home")
    if not isinstance(session_home, dict):
        session_home = {}
    start_positions = session_home.get("follower_positions_deg")
    if not isinstance(start_positions, dict):
        start_positions = {}
    normalized_positions = {
        joint: value
        for joint in JOINT_NAMES
        if (value := _finite_number(start_positions.get(joint))) is not None
    }

    duration = _finite_number(metadata.get("duration_s"))
    dataset_fps = _finite_number(metadata.get("dataset_fps"))
    control_hz = _finite_number(metadata.get("actual_control_hz"))
    tag = rerun_tag(metadata)
    disposition = str(metadata.get("disposition") or "unknown")
    row = {
        "dataset": dataset_name,
        "attempt_id": identifier,
        "episode_id": f"episode_{episode_index:06d}" if episode_index is not None else identifier,
        "training_episode_index": episode_index,
        "task": str(metadata.get("task") or ""),
        "tag": tag,
        "disposition": disposition,
        "operator_disposition": str(metadata.get("operator_disposition") or ""),
        "training_included": included,
        "archive_complete": committed,
        "raw_frames_preserved": _raw_frames_preserved(directory),
        "frames": _integer(metadata.get("samples")),
        "duration_s": duration if duration is not None and duration >= 0 else 0.0,
        "created_at": str(metadata.get("started_at") or identifier),
        "finished_at": str(metadata.get("finished_at") or ""),
        "failure_label": str(metadata.get("failure_label") or ""),
        "failure_note": str(metadata.get("failure_note") or ""),
        "review_revision": _integer(metadata.get("review_revision")),
        "artifacts": artifacts,
        "timing": {
            "dataset_fps": dataset_fps,
            "requested_control_hz": _finite_number(metadata.get("control_hz_requested")),
            "actual_control_hz": control_hz,
            "duration_s": duration if duration is not None and duration >= 0 else 0.0,
            "frames": _integer(metadata.get("samples")),
        },
        "joints": {
            "count": len(JOINT_NAMES),
            "names": list(JOINT_NAMES),
            "start_positions_deg": normalized_positions,
        },
        "camera_freshness": _camera_freshness(metadata),
        "profile": metadata.get("training_profile")
        if isinstance(metadata.get("training_profile"), dict)
        else {},
        "collection_contract_digest": str(metadata.get("collection_contract_digest") or ""),
    }
    row["search_text"] = " ".join(
        str(value)
        for value in (
            dataset_name,
            identifier,
            row["episode_id"],
            row["task"],
            tag,
            disposition,
            row["failure_label"],
            row["failure_note"],
        )
        if value
    ).casefold()
    return row


def _metadata_entries(archive_root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    """Yield secure metadata entries without following archive symlinks."""

    try:
        root = archive_root.resolve(strict=True)
    except OSError:
        return
    if not root.is_dir() or archive_root.is_symlink():
        return
    for dataset_root in root.iterdir():
        if (
            dataset_root.is_symlink()
            or not dataset_root.is_dir()
            or not DATASET_RE.fullmatch(dataset_root.name)
        ):
            continue
        for directory in dataset_root.iterdir():
            if (
                directory.is_symlink()
                or not directory.is_dir()
                or not ATTEMPT_ID_RE.fullmatch(directory.name)
            ):
                continue
            metadata_path = directory / "metadata.json"
            if metadata_path.is_symlink() or not metadata_path.is_file():
                continue
            try:
                resolved = metadata_path.resolve(strict=True)
                resolved.relative_to(root)
                if resolved.parent != directory.resolve(strict=True):
                    continue
                metadata = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(metadata, dict):
                yield directory, metadata


def filter_catalog(
    attempts: Iterable[dict[str, Any]],
    *,
    dataset: str = "",
    tag: str = "",
    search: str = "",
) -> list[dict[str, Any]]:
    """Apply exact dataset/tag filters and a case-insensitive text search."""

    if dataset and not DATASET_RE.fullmatch(dataset):
        raise ValueError("Invalid dataset filter")
    if tag and tag not in RERUN_TAGS:
        raise ValueError("Invalid Rerun tag filter")
    query = str(search or "").strip().casefold()
    if len(query) > 200:
        raise ValueError("Search must be 200 characters or fewer")
    result = []
    for attempt in attempts:
        if dataset and attempt.get("dataset") != dataset:
            continue
        if tag and attempt.get("tag") != tag:
            continue
        if query and query not in str(attempt.get("search_text") or ""):
            continue
        result.append(attempt)
    return sorted(
        result,
        key=lambda item: (str(item.get("created_at") or ""), str(item.get("attempt_id") or "")),
        reverse=True,
    )


def compact_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    """Return the small row used by the frequently refreshed library list."""

    artifacts = attempt.get("artifacts") if isinstance(attempt.get("artifacts"), dict) else {}
    return {
        key: attempt.get(key)
        for key in (
            "dataset",
            "attempt_id",
            "episode_id",
            "training_episode_index",
            "task",
            "tag",
            "disposition",
            "operator_disposition",
            "training_included",
            "archive_complete",
            "raw_frames_preserved",
            "frames",
            "duration_s",
            "created_at",
            "failure_label",
            "review_revision",
        )
    } | {
        "artifacts": {
            name: {
                "available": bool((artifacts.get(name) or {}).get("available")),
                "bytes": _integer((artifacts.get(name) or {}).get("bytes")),
            }
            for name in ("front", "side", "rerun")
        }
    }


def catalog_payload(
    archive_root: Path,
    *,
    dataset: str = "",
    tag: str = "",
    search: str = "",
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    """Build the API payload consumed by ``static/rerun.js``."""

    all_attempts = [
        normalize_attempt(
            metadata,
            directory,
            dataset=directory.parent.name,
            attempt_id=directory.name,
        )
        for directory, metadata in _metadata_entries(archive_root)
    ]
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("Catalog offset must be a non-negative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("Catalog limit must be between 1 and 100")
    filtered = filter_catalog(all_attempts, dataset=dataset, tag=tag, search=search)
    visible = filtered[offset : offset + limit]
    dataset_names = sorted({str(item["dataset"]) for item in all_attempts})
    counts = {label: sum(item["tag"] == label for item in all_attempts) for label in RERUN_TAGS}
    return {
        "schema_version": 1,
        "read_only": True,
        "physical_replay_available": False,
        "total": len(all_attempts),
        "visible": len(filtered),
        "returned": len(visible),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(visible) < len(filtered),
        "datasets": dataset_names,
        "tags": list(RERUN_TAGS),
        "tag_counts": counts,
        "filters": {"dataset": dataset, "tag": tag, "search": search},
        "attempts": [compact_attempt(item) for item in visible],
    }


def attempt_detail_payload(archive_root: Path, attempt_id: str) -> dict[str, Any]:
    """Return normalized detail for one validated archived attempt."""

    if not ATTEMPT_ID_RE.fullmatch(str(attempt_id or "")):
        raise ValueError("Invalid attempt ID")
    matches = [
        normalize_attempt(
            metadata,
            directory,
            dataset=directory.parent.name,
            attempt_id=directory.name,
        )
        for directory, metadata in _metadata_entries(archive_root)
        if directory.name == attempt_id
    ]
    if len(matches) != 1:
        raise FileNotFoundError("Attempt was not found")
    return {"schema_version": 1, "attempt": matches[0]}


def share_selection_preview(archive_root: Path, payload: Any) -> dict[str, Any]:
    """Validate a browser selection without copying, uploading, or mutating data."""

    if not isinstance(payload, dict):
        raise ValueError("Share preview must be a JSON object")
    raw_ids = payload.get("attempt_ids")
    if not isinstance(raw_ids, list) or not raw_ids or len(raw_ids) > 100:
        raise ValueError("Select between 1 and 100 attempts")
    attempt_ids: list[str] = []
    for value in raw_ids:
        identifier = str(value or "")
        if not ATTEMPT_ID_RE.fullmatch(identifier):
            raise ValueError("Share selection contains an invalid attempt ID")
        if identifier not in attempt_ids:
            attempt_ids.append(identifier)
    attempts = [attempt_detail_payload(archive_root, identifier)["attempt"] for identifier in attempt_ids]
    included = [item for item in attempts if item.get("training_included") is True]
    datasets = sorted({str(item.get("dataset") or "") for item in attempts})
    return {
        "schema_version": 1,
        "preview_only": True,
        "selected": len(attempts),
        "included": len(included),
        "excluded": len(attempts) - len(included),
        "datasets": datasets,
        "episode_indexes": sorted(
            item["training_episode_index"]
            for item in included
            if isinstance(item.get("training_episode_index"), int)
        ),
        "share_script": "08_share_dataset.command",
        "message": (
            f"Checked {len(attempts)} selected run(s): {len(included)} are included in the native "
            "LeRobot dataset. Sharing remains dataset-scoped and will run only from an idle, "
            "validated immutable snapshot."
        ),
    }


def resolve_media_path(archive_root: Path, attempt_id: str, media: str) -> Path:
    """Resolve one committed fixed-name artifact, rejecting traversal/symlinks."""

    if not ATTEMPT_ID_RE.fullmatch(str(attempt_id or "")):
        raise ValueError("Invalid attempt ID")
    filename = MEDIA_FILES.get(str(media or ""))
    if filename is None:
        raise ValueError("Unknown attempt media")

    matches = [
        (directory, metadata)
        for directory, metadata in _metadata_entries(archive_root)
        if directory.name == attempt_id
    ]
    if len(matches) != 1:
        raise FileNotFoundError("Attempt was not found")
    directory, metadata = matches[0]
    if metadata.get("archive_complete") is not True:
        raise FileNotFoundError("Attempt archive is not committed and verified")
    path = directory / filename
    if path.is_symlink():
        raise FileNotFoundError("Attempt media symlinks are not allowed")
    try:
        resolved = path.resolve(strict=True)
        if resolved.parent != directory.resolve(strict=True) or not resolved.is_file():
            raise FileNotFoundError("Attempt media is outside its archive")
        if resolved.stat().st_size <= 0:
            raise FileNotFoundError("Attempt media is empty")
    except OSError as exc:
        raise FileNotFoundError("Attempt media is not available") from exc
    return resolved


def safe_new_export_path(export_root: Path, export_name: str) -> Path:
    """Return a never-overwrite export destination or raise on collision."""

    if not EXPORT_NAME_RE.fullmatch(str(export_name or "")):
        raise ValueError("Invalid export name")
    if export_root.is_symlink():
        raise ValueError("Export root may not be a symlink")
    root = export_root.resolve(strict=False)
    candidate = (root / export_name).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Export path leaves the export root") from exc
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError("Export destination already exists; choose a new immutable name")
    return candidate
