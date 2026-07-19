"""Build and independently validate the Query-approved LeRobot derivative."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from .config import ChallengeConfig


_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TASK_TEXT = {
    "single_can": "Pick up one can and place it in the taped sorting zone",
    "two_can": "Pull one of the two cans into the blue-taped recycling zone.",
}
_VECTOR_KEYS = ("action", "observation.state")
_TIMESTAMP_TOLERANCE_S = 1e-5
_EMBEDDED_MANIFEST = Path("meta/rerun_query_selection_manifest.json")


class CurateError(ValueError):
    """The selection or derivative failed a fail-closed curation check."""


@dataclass(frozen=True)
class _Selection:
    identity: str
    repo_id: str
    revision: str
    role: str
    task_key: str
    episode_index: int
    frame_count: int
    verdict: str
    reason_codes: tuple[str, ...]


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CurateError(f"{field} must be an object")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CurateError(f"{field} must be a nonempty string")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise CurateError(f"{field} must be an integer >= {minimum}")
    return int(value)


def _load_manifest(path: Path, config: ChallengeConfig) -> tuple[dict[str, Any], tuple[_Selection, ...]]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CurateError(f"unable to read selection manifest: {error}") from error
    manifest = dict(_mapping(raw, "manifest"))
    if manifest.get("schema_version") != 1 or isinstance(manifest.get("schema_version"), bool):
        raise CurateError("manifest.schema_version must be 1")
    digest = _string(manifest.get("selection_payload_digest"), "selection_payload_digest")
    if not _SHA256.fullmatch(digest):
        raise CurateError("selection_payload_digest must be a lowercase SHA-256")

    if manifest.get("destination_repo") != config.destination_repo:
        raise CurateError("manifest destination does not match the challenge contract")
    source_lock = _mapping(manifest.get("source_lock"), "source_lock")
    if source_lock.get("destination_repo") != config.destination_repo:
        raise CurateError("source-lock destination does not match the challenge contract")
    if tuple(source_lock.get("joint_names", ())) != config.joint_names:
        raise CurateError("source-lock joint order does not match the challenge contract")
    if tuple(source_lock.get("camera_keys", ())) != config.camera_keys:
        raise CurateError("source-lock camera order does not match the challenge contract")
    schema = _mapping(manifest.get("data_schema"), "data_schema")
    expected_vector = {
        "dtype": "float32",
        "shape": [len(config.joint_names)],
        "names": list(config.joint_names),
    }
    if schema.get("action") != expected_vector or schema.get("state") != expected_vector:
        raise CurateError("manifest action/state schema does not match the challenge contract")
    if schema.get("cameras") != list(config.camera_keys):
        raise CurateError("manifest camera schema does not match the challenge contract")

    raw_sources = source_lock.get("sources")
    if not isinstance(raw_sources, list):
        raise CurateError("source_locks must be a list")
    locked = {
        (source.repo_id, source.revision, source.role, source.task_key)
        for source in config.sources
    }
    manifest_locks: set[tuple[str, str, str, str]] = set()
    for index, value in enumerate(raw_sources):
        item = _mapping(value, f"source_locks[{index}]")
        revision = _string(item.get("resolved_sha"), f"source_locks[{index}].resolved_sha")
        if not _SHA40.fullmatch(revision):
            raise CurateError(f"source_locks[{index}].revision must be a lowercase commit SHA")
        lock = (
            _string(item.get("repo_id"), f"source_locks[{index}].repo_id"),
            revision,
            _string(item.get("role"), f"source_locks[{index}].role"),
            next(
                (source.task_key for source in config.sources if source.repo_id == item.get("repo_id")),
                "",
            ),
        )
        if lock in manifest_locks:
            raise CurateError("source_locks contains a duplicate source")
        manifest_locks.add(lock)
    if manifest_locks != locked:
        raise CurateError("manifest source locks do not exactly match the challenge config")

    raw_items = manifest.get("items")
    raw_selected = manifest.get("selected_identities")
    if not isinstance(raw_items, list) or not isinstance(raw_selected, list) or not raw_selected:
        raise CurateError("manifest must contain items and at least one selected identity")
    if any(not isinstance(value, str) or not value for value in raw_selected):
        raise CurateError("selected_identities must contain nonempty strings")
    if len(raw_selected) != len(set(raw_selected)):
        raise CurateError("selected_identities contains a duplicate identity")

    items_by_identity: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(raw_items):
        item = _mapping(value, f"items[{index}]")
        identity = _string(item.get("identity"), f"items[{index}].identity")
        if identity in items_by_identity:
            raise CurateError("manifest items contains a duplicate identity")
        items_by_identity[identity] = item
    if any(identity not in items_by_identity for identity in raw_selected):
        raise CurateError("a selected identity is missing from manifest items")

    selections: list[_Selection] = []
    config_locks = {(s.repo_id, s.revision): s for s in config.sources}
    for identity in sorted(raw_selected):
        item = items_by_identity[identity]
        source_item = _mapping(item.get("source"), f"items[{identity}].source")
        repo_id = _string(source_item.get("repo_id"), f"items[{identity}].source.repo_id")
        revision = _string(source_item.get("revision"), f"items[{identity}].source.revision")
        role = _string(item.get("role"), f"items[{identity}].role")
        task_key = _string(item.get("task_key"), f"items[{identity}].task_key")
        verdict = _string(item.get("verdict"), f"items[{identity}].verdict")
        episode_index = _integer(item.get("episode_index"), f"items[{identity}].episode_index")
        frame_count = _integer(item.get("frame_count"), f"items[{identity}].frame_count", minimum=1)
        source_key = _string(source_item.get("source_key"), f"items[{identity}].source.source_key")
        expected_identity = f"{repo_id}@{revision}:{source_key}"
        if identity != expected_identity:
            raise CurateError(f"selected identity does not authenticate its source episode: {identity}")
        if source_key != str(episode_index):
            raise CurateError(f"selected source key is not its episode index: {identity}")
        source = config_locks.get((repo_id, revision))
        if source is None or source.role != role or source.task_key != task_key:
            raise CurateError(f"selected source does not match the challenge lock: {identity}")
        if item.get("selected") is not True or role != "success" or verdict != "PASS":
            raise CurateError(f"only PASS success episodes may be selected: {identity}")
        reasons = item.get("reason_codes")
        if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
            raise CurateError(f"items[{identity}].reason_codes must be a list of strings")
        selections.append(
            _Selection(
                identity, repo_id, revision, role, task_key, episode_index,
                frame_count, verdict, tuple(reasons),
            )
        )
    selected_from_items = {
        identity for identity, item in items_by_identity.items() if item.get("selected") is True
    }
    if selected_from_items != set(raw_selected):
        raise CurateError("selected_identities does not match item selection flags")
    if manifest.get("selected_episode_count") != len(selections):
        raise CurateError("selected_episode_count does not match selected identities")
    if manifest.get("selected_frame_count") != sum(item.frame_count for item in selections):
        raise CurateError("selected_frame_count does not match selected items")
    try:
        from .artifacts import ManifestError, selection_payload_digest

        computed_digest = selection_payload_digest(manifest)
    except ManifestError as error:
        raise CurateError(f"selection manifest is inconsistent: {error}") from error
    if computed_digest != digest:
        raise CurateError("selection_payload_digest does not authenticate the manifest")
    return manifest, tuple(selections)


def _dataset_class():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def _open_source(selection: _Selection):
    return _dataset_class()(
        repo_id=selection.repo_id,
        revision=selection.revision,
        episodes=[selection.episode_index],
        video_backend="pyav",
    )


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    array = np.asarray(value)
    if array.shape != ():
        raise CurateError(f"expected scalar, got {array.shape}")
    return array.item()


def _vector7(value: Any, field: str) -> np.ndarray:
    array = _numpy(value)
    if array.shape != (7,) or not np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_:
        raise CurateError(f"{field} must be a numeric (7,) vector")
    result = np.asarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise CurateError(f"{field} must be finite")
    return result


def _rgb_u8(value: Any, field: str) -> np.ndarray:
    array = _numpy(value)
    if array.ndim != 3:
        raise CurateError(f"{field} must be an RGB image")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3 or not np.issubdtype(array.dtype, np.number):
        raise CurateError(f"{field} must be an RGB image")
    if not np.isfinite(array).all():
        raise CurateError(f"{field} must be finite")
    if np.issubdtype(array.dtype, np.floating):
        if np.any(array < 0) or np.any(array > 1):
            raise CurateError(f"{field} floating pixels must be within [0, 1]")
        array = np.rint(array * 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        raise CurateError(f"{field} pixels must be float or uint8")
    return np.ascontiguousarray(array)


def _normalized_names(dataset: Any, key: str) -> tuple[str | None, ...]:
    try:
        names = dataset.features[key]["names"]
    except (AttributeError, KeyError, TypeError) as error:
        raise CurateError(f"source is missing {key} names") from error
    return tuple(name[:-4] if isinstance(name, str) and name.endswith(".pos") else None for name in names)


def _validate_source_schema(dataset: Any, config: ChallengeConfig) -> None:
    if getattr(dataset, "fps", None) != config.fps:
        raise CurateError("source dataset FPS must be 30")
    for key in _VECTOR_KEYS:
        if _normalized_names(dataset, key) != config.joint_names:
            raise CurateError(f"source {key} joint order does not match the challenge contract")
    for camera in config.camera_keys:
        key = f"observation.images.{camera}"
        try:
            feature = dataset.features[key]
        except (AttributeError, KeyError) as error:
            raise CurateError(f"source is missing required video {key}") from error
        if feature.get("dtype") not in ("video", "image"):
            raise CurateError(f"source {key} is not an image/video feature")


def _destination_features(dataset: Any, config: ChallengeConfig) -> dict[str, dict[str, Any]]:
    names = [f"{name}.pos" for name in config.joint_names]
    features: dict[str, dict[str, Any]] = {
        "action": {"dtype": "float32", "shape": (7,), "names": names},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": names},
    }
    for camera in config.camera_keys:
        key = f"observation.images.{camera}"
        source = dataset.features[key]
        shape = tuple(source.get("shape", ()))
        names_value = list(source.get("names", ()))
        if len(shape) != 3 or not names_value:
            raise CurateError(f"source {key} has an invalid declared shape")
        features[key] = {"dtype": "video", "shape": shape, "names": names_value}
    return features


def _feature_contract(features: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[Any, ...]]:
    return {
        key: (tuple(value.get("shape", ())), tuple(value.get("names", ())))
        for key, value in features.items()
        if key in _VECTOR_KEYS or key.startswith("observation.images.")
    }


def _copy_episode(destination: Any, source: Any, selection: _Selection, config: ChallengeConfig) -> None:
    if len(source) != selection.frame_count:
        raise CurateError(f"source frame count changed for {selection.identity}")
    for offset in range(len(source)):
        sample = source[offset]
        try:
            source_episode = _integer(_scalar(sample["episode_index"]), "episode_index")
            frame_index = _integer(_scalar(sample["frame_index"]), "frame_index")
            timestamp = float(_scalar(sample["timestamp"]))
        except (KeyError, TypeError, ValueError) as error:
            raise CurateError(f"source indexes are invalid for {selection.identity}") from error
        if source_episode != selection.episode_index or frame_index != offset:
            raise CurateError(f"source episode/frame indexes changed for {selection.identity}")
        if not math.isfinite(timestamp) or not math.isclose(
            timestamp, offset / config.fps, rel_tol=0.0, abs_tol=_TIMESTAMP_TOLERANCE_S
        ):
            raise CurateError(f"source timestamps are not 30 FPS for {selection.identity}")
        frame: dict[str, Any] = {
            "action": _vector7(sample["action"], "action"),
            "observation.state": _vector7(sample["observation.state"], "observation.state"),
            "task": _TASK_TEXT[selection.task_key],
        }
        for camera in config.camera_keys:
            key = f"observation.images.{camera}"
            if key not in sample:
                raise CurateError(f"decoded source sample is missing {key}")
            frame[key] = _rgb_u8(sample[key], key)
        destination.add_frame(frame)
    destination.save_episode(parallel_encoding=False)


def build_derivative(manifest_path: Path, config: ChallengeConfig, output_root: Path) -> Path:
    """Copy only manifest-selected PASS successes into a new LeRobot v3 tree."""

    manifest, selections = _load_manifest(Path(manifest_path), config)
    destination = Path(output_root) / config.destination_repo
    if destination.exists() and any(destination.iterdir()):
        raise CurateError(f"destination already exists and is nonempty: {destination}")
    if destination.exists():
        destination.rmdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    created: Any | None = None
    try:
        first = _open_source(selections[0])
        _validate_source_schema(first, config)
        features = _destination_features(first, config)
        created = _dataset_class().create(
            repo_id=config.destination_repo,
            fps=config.fps,
            features=features,
            root=temporary,
            robot_type=config.robot_type,
            use_videos=True,
            video_backend="pyav",
            vcodec="h264",
        )
        for index, selection in enumerate(selections):
            source = first if index == 0 else _open_source(selection)
            _validate_source_schema(source, config)
            if _feature_contract(source.features) != _feature_contract(features):
                raise CurateError(f"source feature schema changed for {selection.identity}")
            _copy_episode(created, source, selection, config)
        if hasattr(created, "finalize"):
            created.finalize()
        embedded = temporary / _EMBEDDED_MANIFEST
        embedded.parent.mkdir(parents=True, exist_ok=True)
        embedded.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _validate_in_process(dataset_root: Path, repo_id: str, expected_digest: str) -> dict[str, Any]:
    manifest_path = dataset_root / _EMBEDDED_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CurateError(f"embedded selection manifest is unreadable: {error}") from error
    if manifest.get("selection_payload_digest") != expected_digest:
        raise CurateError("embedded selection manifest digest does not match")
    try:
        from .artifacts import ManifestError, selection_payload_digest

        computed_digest = selection_payload_digest(manifest)
    except ManifestError as error:
        raise CurateError(f"embedded selection manifest is inconsistent: {error}") from error
    if computed_digest != expected_digest:
        raise CurateError("embedded selection manifest digest is not authentic")
    selected = manifest.get("selected_identities")
    items = manifest.get("items")
    schema = _mapping(manifest.get("data_schema"), "data_schema")
    if not isinstance(selected, list) or not isinstance(items, list):
        raise CurateError("embedded selection manifest is incomplete")
    by_identity = {_mapping(item, "item").get("identity"): item for item in items}
    ordered = [by_identity[identity] for identity in sorted(selected)]

    dataset = _dataset_class()(repo_id=repo_id, root=dataset_root, video_backend="pyav")
    if dataset.repo_id != repo_id or getattr(dataset, "fps", None) != 30:
        raise CurateError("derivative repository identity or FPS is invalid")
    expected_names = tuple(_mapping(schema.get("action"), "data_schema.action").get("names", ()))
    expected_cameras = tuple(schema.get("cameras", ()))
    if len(expected_names) != 7 or expected_cameras != ("front", "side"):
        raise CurateError(
            f"embedded schema is invalid: names={expected_names!r}, cameras={expected_cameras!r}"
        )
    for key in _VECTOR_KEYS:
        if _normalized_names(dataset, key) != expected_names:
            raise CurateError(f"derivative {key} names are invalid")
    for camera in expected_cameras:
        if f"observation.images.{camera}" not in dataset.features:
            raise CurateError(f"derivative is missing {camera} video")

    expected_frames = sum(_integer(item.get("frame_count"), "frame_count", minimum=1) for item in ordered)
    if dataset.meta.total_episodes != len(ordered) or dataset.meta.total_frames != expected_frames:
        raise CurateError("derivative episode/frame totals do not match the manifest")
    seen_frames = [0] * len(ordered)
    decoded_shapes: dict[str, list[int]] = {}
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        episode_index = _integer(_scalar(sample["episode_index"]), "episode_index")
        if episode_index >= len(ordered):
            raise CurateError("derivative episode indexes are not contiguous")
        frame_index = _integer(_scalar(sample["frame_index"]), "frame_index")
        if frame_index != seen_frames[episode_index]:
            raise CurateError("derivative frame indexes are not contiguous")
        timestamp = float(_scalar(sample["timestamp"]))
        if not math.isfinite(timestamp) or not math.isclose(
            timestamp, frame_index / 30, rel_tol=0.0, abs_tol=_TIMESTAMP_TOLERANCE_S
        ):
            raise CurateError("derivative timestamps are not 30 FPS")
        for key in _VECTOR_KEYS:
            _vector7(sample[key], key)
        expected_task = _TASK_TEXT[_string(ordered[episode_index].get("task_key"), "task_key")]
        if sample.get("task") != expected_task:
            raise CurateError("derivative task mapping is invalid")
        for camera in expected_cameras:
            key = f"observation.images.{camera}"
            image = _numpy(sample[key])
            if image.ndim != 3 or 3 not in (image.shape[0], image.shape[-1]) or not np.isfinite(image).all():
                raise CurateError(f"derivative {camera} video did not decode as RGB")
            decoded_shapes.setdefault(camera, list(image.shape))
        seen_frames[episode_index] += 1
    if seen_frames != [_integer(item.get("frame_count"), "frame_count", minimum=1) for item in ordered]:
        raise CurateError("derivative per-episode frame totals do not match provenance")
    return {
        "ok": True,
        "repo_id": repo_id,
        "selection_payload_digest": expected_digest,
        "episodes": len(ordered),
        "frames": expected_frames,
        "episode_indexes": list(range(len(ordered))),
        "source_identities": [str(item["identity"]) for item in ordered],
        "tasks": [_TASK_TEXT[str(item["task_key"])] for item in ordered],
        "camera_keys": list(expected_cameras),
        "decoded_shapes": decoded_shapes,
        "errors": [],
    }


def validate_derivative_fresh(
    dataset_root: Path,
    repo_id: str,
    expected_manifest_digest: str,
) -> dict[str, Any]:
    """Load and validate the derivative in a new pinned-Python process."""

    if not _SHA256.fullmatch(expected_manifest_digest):
        raise CurateError("expected_manifest_digest must be a lowercase SHA-256")
    report_path = Path(dataset_root).parent / f".{Path(dataset_root).name}.validation-{uuid.uuid4().hex}.json"
    command = [
        sys.executable,
        "-m",
        "p5_rerun_port.challenge.curate",
        "--validate-worker",
        str(Path(dataset_root).resolve()),
        "--repo-id",
        repo_id,
        "--expected-digest",
        expected_manifest_digest,
        "--report",
        str(report_path),
    ]
    environment = os.environ.copy()
    project_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (project_root, environment.get("PYTHONPATH", "")) if value
    )
    result = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CurateError(
            f"fresh derivative validation failed without a report (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        ) from error
    finally:
        report_path.unlink(missing_ok=True)
    if result.returncode != 0 or report.get("ok") is not True:
        errors = report.get("errors", [result.stderr.strip()])
        raise CurateError(f"fresh derivative validation failed: {errors}")
    return report


def _worker(args: argparse.Namespace) -> int:
    try:
        report = _validate_in_process(Path(args.validate_worker), args.repo_id, args.expected_digest)
        code = 0
    except Exception as error:
        report = {"ok": False, "errors": [str(error)]}
        code = 1
    Path(args.report).write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--validate-worker")
    parser.add_argument("--repo-id")
    parser.add_argument("--expected-digest")
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    if not all((args.validate_worker, args.repo_id, args.expected_digest, args.report)):
        parser.error("the validation worker requires all arguments")
    return _worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
