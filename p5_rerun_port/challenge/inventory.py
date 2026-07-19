"""Build the deterministic inventory of pinned success and failure sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from numbers import Integral
from pathlib import Path, PurePosixPath
import re
from typing import Any

import pandas as pd

from .config import ChallengeConfig
from .hub import HubReader
from .models import EpisodeIdentity, InventoryRow, SourceSpec


class InventoryError(ValueError):
    """A remote source does not satisfy the immutable inventory contract."""


_EPISODE_METADATA = re.compile(r"meta/episodes/chunk-\d+/file-\d+\.parquet")
_ISO_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})"
)
_ALLOWED_DISPOSITIONS = frozenset({"aborted", "collector_error", "failed"})
_TASKS = {
    "single_can": "Pick up one can and place it in the taped sorting zone",
    "two_can": "Pull one of the two cans into the blue-taped recycling zone.",
}


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise InventoryError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise InventoryError(f"{field} must be a list")
    return tuple(value)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InventoryError(f"{field} must be a non-blank string")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise InventoryError(f"{field} must be an integer of at least {minimum}")
    return int(value)


def _timestamp(value: Any, field: str) -> tuple[str, datetime]:
    text = _text(value, field)
    if not _ISO_TIMESTAMP.fullmatch(text):
        raise InventoryError(f"{field} must be a timezone-aware ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(f"{text[:-1]}+00:00" if text.endswith("Z") else text)
    except ValueError as error:
        raise InventoryError(f"{field} must be a timezone-aware ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InventoryError(f"{field} must be a timezone-aware ISO-8601 timestamp")
    return text, parsed


def _safe_path(value: Any, field: str, prefix: str) -> str:
    path = _text(value, field)
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        raise InventoryError(f"{field} source path is unsafe: {path}")
    if not path.startswith(prefix) or str(parsed) != path:
        raise InventoryError(f"{field} source path must stay under {prefix}: {path}")
    return path


def _validate_info(config: ChallengeConfig, source: SourceSpec, info: Mapping[str, Any]) -> None:
    if info.get("codebase_version") != "v3.0":
        raise InventoryError(f"{source.repo_id} is not LeRobot v3.0")
    if info.get("robot_type") != config.robot_type:
        raise InventoryError(f"{source.repo_id} robot_type does not match the lock")
    if info.get("fps") != config.fps:
        raise InventoryError(f"{source.repo_id} fps does not match the lock")
    if info.get("total_episodes") != source.expected_items:
        raise InventoryError(f"{source.repo_id} total_episodes does not match the lock")
    if info.get("total_frames") != source.expected_frames:
        raise InventoryError(f"{source.repo_id} total_frames does not match the lock")

    features = _mapping(info.get("features"), f"{source.repo_id} features")
    expected_names = config.joint_names
    for key in ("action", "observation.state"):
        feature = _mapping(features.get(key), f"{source.repo_id} features.{key}")
        shape = tuple(_sequence(feature.get("shape"), f"{source.repo_id} features.{key}.shape"))
        names = tuple(_text(name, f"{source.repo_id} features.{key}.names") for name in _sequence(
            feature.get("names"), f"{source.repo_id} features.{key}.names"
        ))
        normalized = tuple(name[:-4] if name.endswith(".pos") else "" for name in names)
        if shape != (len(expected_names),) or normalized != expected_names:
            raise InventoryError(f"{source.repo_id} {key} joint order does not match the authenticated profile")
    for camera in config.camera_keys:
        key = f"observation.images.{camera}"
        if key not in features:
            raise InventoryError(f"{source.repo_id} is missing required camera feature {key}")


def _manifest_capture_times(manifest: Mapping[str, Any], repo_id: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for offset, raw in enumerate(_sequence(manifest.get("source_attempts"), f"{repo_id} source_attempts")):
        entry = _mapping(raw, f"{repo_id} source_attempts[{offset}]")
        index = _integer(entry.get("episode_index"), f"{repo_id} source_attempts[{offset}].episode_index")
        if index in result:
            raise InventoryError(f"{repo_id} has duplicate source_attempt episode_index {index}")
        started_text, started = _timestamp(
            entry.get("started_at_utc"), f"{repo_id} source_attempts[{offset}].started_at_utc"
        )
        _, finished = _timestamp(
            entry.get("finished_at_utc"), f"{repo_id} source_attempts[{offset}].finished_at_utc"
        )
        if finished < started:
            raise InventoryError(
                f"{repo_id} source_attempts[{offset}].finished_at must not precede started_at"
            )
        result[index] = started_text
    return result


def _success_rows(config: ChallengeConfig, source: SourceSpec, hub: HubReader) -> list[InventoryRow]:
    info = _mapping(hub.read_json(source.repo_id, source.revision, "meta/info.json"), "meta/info.json")
    _validate_info(config, source, info)
    manifest = _mapping(
        hub.read_json(source.repo_id, source.revision, "SHARE_MANIFEST.json"), "SHARE_MANIFEST.json"
    )
    capture_times = _manifest_capture_times(manifest, source.repo_id)

    metadata_paths: list[str] = []
    for offset, raw in enumerate(_sequence(manifest.get("files"), f"{source.repo_id} manifest files")):
        entry = _mapping(raw, f"{source.repo_id} manifest files[{offset}]")
        path = _safe_path(entry.get("path"), f"{source.repo_id} manifest files[{offset}].path", "")
        if _EPISODE_METADATA.fullmatch(path):
            metadata_paths.append(path)
    metadata_paths = sorted(set(metadata_paths))
    if not metadata_paths:
        raise InventoryError(f"{source.repo_id} has no episode metadata parquet files")

    data_template = _text(info.get("data_path"), f"{source.repo_id} data_path")
    raw_rows: list[Mapping[str, Any]] = []
    for path in metadata_paths:
        frame = hub.read_parquet(source.repo_id, source.revision, path)
        if not isinstance(frame, pd.DataFrame):
            raise InventoryError(f"{source.repo_id}:{path} did not return a dataframe")
        raw_rows.extend(frame.to_dict(orient="records"))

    seen: set[int] = set()
    rows: list[InventoryRow] = []
    expected_task = _TASKS[source.task_key]
    for offset, raw in enumerate(raw_rows):
        episode_index = _integer(raw.get("episode_index"), f"{source.repo_id} row[{offset}].episode_index")
        if episode_index in seen:
            raise InventoryError(f"{source.repo_id} has duplicate episode_index {episode_index}")
        seen.add(episode_index)
        tasks = tuple(_text(task, f"{source.repo_id} row[{offset}].tasks") for task in _sequence(
            raw.get("tasks"), f"{source.repo_id} row[{offset}].tasks"
        ))
        if tasks != (expected_task,):
            raise InventoryError(f"{source.repo_id} episode {episode_index} has an unknown task")
        length = _integer(raw.get("length"), f"{source.repo_id} row[{offset}].length", minimum=1)
        chunk = _integer(raw.get("data/chunk_index"), f"{source.repo_id} row[{offset}].data/chunk_index")
        file_index = _integer(raw.get("data/file_index"), f"{source.repo_id} row[{offset}].data/file_index")
        try:
            source_path = data_template.format(chunk_index=chunk, file_index=file_index)
        except (KeyError, ValueError) as error:
            raise InventoryError(f"{source.repo_id} data_path cannot be formatted") from error
        source_path = _safe_path(source_path, f"{source.repo_id} episode {episode_index}", "data/")
        if episode_index not in capture_times:
            raise InventoryError(f"{source.repo_id} episode {episode_index} has no capture time")
        rows.append(
            InventoryRow(
                identity=EpisodeIdentity(source.repo_id, source.revision, str(episode_index)),
                role="success",
                task_key=source.task_key,
                source_path=source_path,
                episode_index=episode_index,
                attempt_id=None,
                frame_count=length,
                captured_at=capture_times[episode_index],
            )
        )

    ordered = sorted(rows, key=lambda row: row.episode_index if row.episode_index is not None else -1)
    indexes = [row.episode_index for row in ordered]
    if indexes != list(range(source.expected_items)):
        raise InventoryError(f"{source.repo_id} episode indexes must be contiguous from zero")
    if len(capture_times) != source.expected_items:
        raise InventoryError(f"{source.repo_id} source_attempt count does not match the lock")
    if sum(row.frame_count for row in ordered) != source.expected_frames:
        raise InventoryError(f"{source.repo_id} observed frame total does not match the lock")
    return ordered


def _existing_attempt_paths(source: SourceSpec, attempt_id: str, available_paths: set[str]) -> list[str]:
    candidates: list[str] = []
    for filename in ("attempt.rrd", "attempt.partial.rrd"):
        path = f"failed_attempts/{attempt_id}/{filename}"
        _safe_path(path, f"{source.repo_id} attempt {attempt_id}", f"failed_attempts/{attempt_id}/")
        if path in available_paths:
            candidates.append(path)
    return candidates


def _failure_rows(source: SourceSpec, hub: HubReader) -> list[InventoryRow]:
    index = _mapping(
        hub.read_json(source.repo_id, source.revision, "FAILURE_INDEX.json"), "FAILURE_INDEX.json"
    )
    if index.get("training_eligible") is not False:
        raise InventoryError(f"{source.repo_id} root training_eligible must be false")
    if index.get("attempt_count") != source.expected_items:
        raise InventoryError(f"{source.repo_id} attempt_count does not match the lock")
    attempts = _sequence(index.get("attempts"), f"{source.repo_id} attempts")
    if len(attempts) != source.expected_items:
        raise InventoryError(f"{source.repo_id} observed attempt count does not match the lock")
    available_paths = set(hub.list_paths(source.repo_id, source.revision, "failed_attempts"))

    rows: list[InventoryRow] = []
    seen: set[str] = set()
    for offset, raw in enumerate(attempts):
        attempt = _mapping(raw, f"{source.repo_id} attempts[{offset}]")
        attempt_id = _text(attempt.get("attempt_id"), f"{source.repo_id} attempts[{offset}].attempt_id")
        if "/" in attempt_id or attempt_id in {".", ".."}:
            raise InventoryError(f"{source.repo_id} attempt_id cannot contain path separators")
        if attempt_id in seen:
            raise InventoryError(f"{source.repo_id} has duplicate attempt_id {attempt_id}")
        seen.add(attempt_id)
        disposition = _text(attempt.get("disposition"), f"{source.repo_id} attempt {attempt_id}.disposition")
        if disposition not in _ALLOWED_DISPOSITIONS:
            raise InventoryError(f"{source.repo_id} attempt {attempt_id} has unknown disposition")
        if attempt.get("training_included") is not False:
            raise InventoryError(f"{source.repo_id} attempt {attempt_id} training_included must be false")
        archive_complete = attempt.get("archive_complete")
        if not isinstance(archive_complete, bool):
            raise InventoryError(f"{source.repo_id} attempt {attempt_id} archive_complete must be boolean")
        samples = _integer(attempt.get("samples"), f"{source.repo_id} attempt {attempt_id}.samples", minimum=1)
        captured_at, started = _timestamp(
            attempt.get("started_at"), f"{source.repo_id} attempt {attempt_id}.started_at"
        )
        _, finished = _timestamp(
            attempt.get("finished_at"), f"{source.repo_id} attempt {attempt_id}.finished_at"
        )
        if finished < started:
            raise InventoryError(
                f"{source.repo_id} attempt {attempt_id}.finished_at must not precede started_at"
            )
        if _text(attempt.get("task"), f"{source.repo_id} attempt {attempt_id}.task") != _TASKS[source.task_key]:
            raise InventoryError(f"{source.repo_id} attempt {attempt_id} has an unknown task")
        paths = _existing_attempt_paths(source, attempt_id, available_paths)
        if len(paths) != 1:
            raise InventoryError(
                f"{source.repo_id} attempt {attempt_id} must have exactly one authoritative RRD; found {len(paths)}"
            )
        expected_filename = "attempt.rrd" if archive_complete else "attempt.partial.rrd"
        if PurePosixPath(paths[0]).name != expected_filename:
            raise InventoryError(
                f"{source.repo_id} attempt {attempt_id} archive_complete does not match RRD filename"
            )
        rows.append(
            InventoryRow(
                identity=EpisodeIdentity(source.repo_id, source.revision, attempt_id),
                role="failure",
                task_key=source.task_key,
                source_path=paths[0],
                episode_index=None,
                attempt_id=attempt_id,
                frame_count=samples,
                captured_at=captured_at,
            )
        )
    return sorted(rows, key=lambda row: row.attempt_id or "")


def build_inventory(config: ChallengeConfig, hub: HubReader) -> tuple[InventoryRow, ...]:
    """Validate every configured pin and return all source items exactly once."""

    rows: list[InventoryRow] = []
    for source in config.sources:
        resolved = hub.dataset_sha(source.repo_id, source.revision)
        if resolved != source.revision:
            raise InventoryError(
                f"{source.repo_id}@{source.revision} resolved SHA {resolved!r} does not match the lock"
            )
        rows.extend(_success_rows(config, source, hub) if source.role == "success" else _failure_rows(source, hub))
    identities = [row.identity.canonical for row in rows]
    if len(identities) != len(set(identities)):
        raise InventoryError("inventory contains duplicate canonical identities")
    expected = sum(source.expected_items for source in config.sources)
    if len(rows) != expected:
        raise InventoryError(f"inventory contains {len(rows)} rows; expected {expected}")
    return tuple(rows)


def _inventory_payload(inventory: Sequence[InventoryRow]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in inventory:
        value = asdict(row)
        value["canonical_identity"] = row.identity.canonical
        payload.append(value)
    return sorted(payload, key=lambda item: item["canonical_identity"])


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_source_lock_inventory(config: ChallengeConfig, inventory: Sequence[InventoryRow]) -> None:
    sources = {source.repo_id: source for source in config.sources}
    grouped: dict[str, list[InventoryRow]] = {source.repo_id: [] for source in config.sources}
    identities: set[str] = set()
    for row in inventory:
        repo_id = row.identity.repo_id
        source = sources.get(repo_id)
        if source is None:
            raise InventoryError(f"source-lock row repo_id {repo_id!r} is not configured")
        if row.identity.revision != source.revision:
            raise InventoryError(f"source-lock row {row.identity.canonical} revision does not match SourceSpec")
        if row.role != source.role:
            raise InventoryError(f"source-lock row {row.identity.canonical} role does not match SourceSpec")
        if row.task_key != source.task_key:
            raise InventoryError(f"source-lock row {row.identity.canonical} task does not match SourceSpec")
        if row.identity.canonical in identities:
            raise InventoryError(f"source-lock has duplicate identity {row.identity.canonical}")
        identities.add(row.identity.canonical)
        grouped[repo_id].append(row)

    for source in config.sources:
        rows = grouped[source.repo_id]
        if len(rows) != source.expected_items:
            raise InventoryError(
                f"source-lock {source.repo_id} count {len(rows)} does not match {source.expected_items}"
            )
        if source.role == "success" and sum(row.frame_count for row in rows) != source.expected_frames:
            raise InventoryError(f"source-lock {source.repo_id} frame count does not match SourceSpec")


def write_source_lock(
    path: Path,
    config: ChallengeConfig,
    inventory: Sequence[InventoryRow],
    *,
    config_path: Path = Path("config/rerun_query_challenge.yaml"),
) -> dict[str, Any]:
    """Write a stable source-lock whose digests exclude generation time."""

    _validate_source_lock_inventory(config, inventory)
    inventory_payload = _inventory_payload(inventory)
    expected_counts = {
        "success": sum(source.expected_items for source in config.sources if source.role == "success"),
        "failure": sum(source.expected_items for source in config.sources if source.role == "failure"),
    }
    expected_counts["total"] = expected_counts["success"] + expected_counts["failure"]
    observed_counts = {
        "success": sum(row.role == "success" for row in inventory),
        "failure": sum(row.role == "failure" for row in inventory),
    }
    observed_counts["total"] = observed_counts["success"] + observed_counts["failure"]
    sources = []
    for source in config.sources:
        source_rows = [row for row in inventory if row.identity.repo_id == source.repo_id]
        sources.append(
            {
                "repo_id": source.repo_id,
                "configured_revision": source.revision,
                "resolved_sha": source.revision,
                "role": source.role,
                "task_key": source.task_key,
                "expected_items": source.expected_items,
                "observed_items": len(source_rows),
                "expected_frames": source.expected_frames,
                "observed_frames": sum(row.frame_count for row in source_rows) if source.role == "success" else None,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "sources": sources,
        "expected_counts": expected_counts,
        "observed_counts": observed_counts,
        "inventory_payload_digest": _digest(inventory_payload),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
