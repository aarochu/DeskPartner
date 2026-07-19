#!/usr/bin/env python3
"""High-rate ReBot teleoperation with synchronized LeRobot sampling.

Stock ``lerobot-record`` uses the dataset FPS for the motor-control loop and
stores the leader-space command instead of the follower-space action returned by
``robot.send_action``. That is not suitable for this ReBot: demonstrations need a
fast hand-following loop, and the learning target must be the exact clipped,
direction-corrected action actually sent to the follower.

This runner keeps those clocks separate:

* leader/follower control runs at ``--control-hz``;
* cameras and dataset samples are stored at ``--dataset-fps``;
* every sample pairs one follower observation with the follower-space action sent
  from that same control tick.

GUI controls are ordinary POSIX signals so no macOS Accessibility permission is
needed: SIGUSR1 keeps/finishes the episode, SIGUSR2 re-records it, and SIGHUP
stops the session and discards an incomplete current take.
"""

from __future__ import annotations

import argparse
import av
from contextlib import contextmanager
import hashlib
import json
import logging
import math
import numbers
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
from uuid import uuid4

import numpy as np
import rerun as rr
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import DEFAULT_FEATURES, build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager, encode_video_frames
from lerobot.processor import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from attempt_archive import (
    atomic_write_json,
    attempt_path,
    new_attempt_id,
    utc_now,
    validate_failure_label,
)
from lerobot_robot_seeed_b601 import (
    SeeedB601DMFollower,
    SeeedB601DMFollowerConfig,
)
from lerobot_teleoperator_rebot_arm_102 import (
    RebotArm102Leader,
    RebotArm102LeaderConfig,
)


# Camera buffers are sampled without waiting so the 30 FPS image clock cannot
# throttle the motor loop. A frame up to 250 ms old is normally fresh. macOS
# scheduling/USB jitter may briefly exceed that. At the 30 FPS dataset clock,
# allow up to eight consecutive samples in this bounded 250-500 ms jitter band;
# this covers the roughly 320 ms wrist-camera stalls observed on macOS without
# coupling camera reads back into the motor loop. A ninth consecutive jitter
# sample, or any frame older than 500 ms, fails closed as a stale/frozen camera.
CAMERA_FRESH_AGE_MS = 250.0
CAMERA_JITTER_MAX_AGE_MS = 500
CAMERA_MAX_CONSECUTIVE_JITTER_SAMPLES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--profile-path", type=Path, required=True)
    parser.add_argument("--profile-digest", required=True)
    parser.add_argument("--collection-contract-json", required=True)
    parser.add_argument("--collection-contract-digest", required=True)
    parser.add_argument("--follower-calibration", type=Path, required=True)
    parser.add_argument("--leader-calibration", type=Path, required=True)
    parser.add_argument("--follower-driver-contract", type=Path, required=True)
    parser.add_argument("--follower-base-implementation", type=Path, required=True)
    parser.add_argument("--follower-dm-implementation", type=Path, required=True)
    parser.add_argument("--leader-driver-contract", type=Path, required=True)
    parser.add_argument("--leader-implementation", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--episode-time-s", type=float, default=1000.0)
    parser.add_argument("--reset-time-s", type=float, default=20.0)
    parser.add_argument("--control-hz", type=int, default=240)
    parser.add_argument("--dataset-fps", type=int, default=30)
    parser.add_argument("--front-camera", type=int, default=0)
    parser.add_argument("--front-width", type=int, default=640)
    parser.add_argument("--front-height", type=int, default=480)
    parser.add_argument("--side-camera", type=int, default=1)
    parser.add_argument("--side-width", type=int, default=1280)
    parser.add_argument("--side-height", type=int, default=720)
    parser.add_argument("--excluded-camera", type=int, required=True)
    parser.add_argument("--max-step", type=float, default=33.6)
    parser.add_argument("--motor-velocity", type=float, default=2000.0)
    parser.add_argument("--gripper-force", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-rerun", action="store_true")
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--control-file", type=Path, required=True)
    args = parser.parse_args()
    if args.control_hz < args.dataset_fps:
        parser.error("--control-hz must be at least --dataset-fps")
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def collection_contract_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": args.dataset_root.name,
        "repo_id": args.repo_id,
        "task": args.task,
        "episode_time_s": args.episode_time_s,
        "reset_time_s": args.reset_time_s,
        "fps": args.dataset_fps,
        "control_hz": args.control_hz,
        "front_camera": args.front_camera,
        "front_width": args.front_width,
        "front_height": args.front_height,
        "side_camera": args.side_camera,
        "side_width": args.side_width,
        "side_height": args.side_height,
        "excluded_camera": args.excluded_camera,
        "max_step": args.max_step,
        "motor_velocity": args.motor_velocity,
        "gripper_force": args.gripper_force,
    }


def verify_collection_contract(args: argparse.Namespace) -> dict[str, Any]:
    try:
        supplied = json.loads(args.collection_contract_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Collection contract is not valid JSON") from exc
    if not isinstance(supplied, dict):
        raise RuntimeError("Collection contract must be an object")
    if canonical_digest(supplied) != args.collection_contract_digest:
        raise RuntimeError("Collection contract digest changed after collection was authorized")
    actual = collection_contract_from_args(args)
    if supplied != actual:
        changed = sorted(
            key for key in set(supplied) | set(actual) if supplied.get(key) != actual.get(key)
        )
        raise RuntimeError(
            "Collection command does not match its locked contract: " + ", ".join(changed)
        )
    return supplied


def verify_profile_lock(args: argparse.Namespace) -> dict[str, Any]:
    try:
        profile = json.loads(args.profile_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read training profile: {exc}") from exc
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise RuntimeError("Unsupported training profile schema")
    actual_profile_digest = canonical_digest(profile)
    if actual_profile_digest != args.profile_digest:
        raise RuntimeError(
            "Training profile digest changed after collection was authorized; reload the GUI"
        )

    calibration = profile.get("calibration")
    coordinates = profile.get("coordinate_contract")
    if not isinstance(calibration, dict) or not isinstance(coordinates, dict):
        raise RuntimeError("Training profile is missing calibration or coordinate contract")
    joints = coordinates.get("joints")
    if not isinstance(joints, list) or len(joints) != 7:
        raise RuntimeError("Training profile must contain exactly seven joints")
    joint_names = [str(joint.get("name", "")) for joint in joints if isinstance(joint, dict)]
    if len(joint_names) != 7 or len(set(joint_names)) != 7:
        raise RuntimeError("Training profile joint names are invalid")

    locked_files = {
        "follower": (args.follower_calibration, calibration.get("follower"), list(range(1, 8))),
        "leader": (args.leader_calibration, calibration.get("leader"), list(range(0, 7))),
    }
    for label, (path, entry, expected_ids) in locked_files.items():
        if not isinstance(entry, dict) or not path.is_file():
            raise RuntimeError(f"{label.title()} calibration file is missing")
        expected_hash = str(entry.get("sha256", ""))
        if sha256(path) != expected_hash:
            raise RuntimeError(f"{label.title()} calibration fingerprint changed")
        try:
            values = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{label.title()} calibration is not valid JSON") from exc
        if not isinstance(values, dict) or list(values) != joint_names:
            raise RuntimeError(f"{label.title()} calibration joint order does not match the profile")
        ids = []
        for name in joint_names:
            value = values.get(name)
            if not isinstance(value, dict):
                raise RuntimeError(f"{label.title()} calibration entry {name} is invalid")
            ids.append(value.get("id"))
            numeric = [
                value.get("drive_mode"),
                value.get("homing_offset"),
                value.get("range_min"),
                value.get("range_max"),
            ]
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in numeric
            ):
                raise RuntimeError(f"{label.title()} calibration entry {name} is non-finite")
        if ids != expected_ids:
            raise RuntimeError(
                f"{label.title()} calibration motor IDs changed: {ids} != {expected_ids}"
            )

    runtime_contracts = {
        "follower_driver_contract": args.follower_driver_contract,
        "follower_base_implementation": args.follower_base_implementation,
        "follower_dm_implementation": args.follower_dm_implementation,
        "leader_driver_contract": args.leader_driver_contract,
        "leader_implementation": args.leader_implementation,
    }
    for label, path in runtime_contracts.items():
        entry = calibration.get(label)
        if (
            not isinstance(entry, dict)
            or not path.is_file()
            or sha256(path) != str(entry.get("sha256", ""))
        ):
            raise RuntimeError(f"Runtime coordinate contract fingerprint changed: {label}")
    return profile


def profile_sidecar_path(args: argparse.Namespace) -> Path:
    return args.dataset_root / "meta" / "rebot_training_profile.json"


def verify_existing_sidecar(
    existing: Any,
    args: argparse.Namespace,
    contract: dict[str, Any],
) -> None:
    if not isinstance(existing, dict):
        raise RuntimeError("Dataset training-profile sidecar root is invalid")
    snapshot = existing.get("profile_snapshot")
    stored_contract = existing.get("collection_contract")
    if not isinstance(snapshot, dict) or not isinstance(stored_contract, dict):
        raise RuntimeError("Dataset profile or collection-contract snapshot is missing")
    if (
        canonical_digest(snapshot) != existing.get("training_profile_digest")
        or existing.get("training_profile_digest") != args.profile_digest
    ):
        raise RuntimeError("Dataset training-profile snapshot is modified or uses another profile")
    if (
        canonical_digest(stored_contract) != existing.get("collection_contract_digest")
        or existing.get("collection_contract_digest") != args.collection_contract_digest
        or stored_contract != contract
    ):
        raise RuntimeError(
            "Dataset uses different task, camera, clock, or motor-control settings"
        )
    if (
        existing.get("training_profile_id") != snapshot.get("profile_id")
        or existing.get("training_profile_version") != snapshot.get("profile_version")
    ):
        raise RuntimeError("Dataset profile identity does not match its authenticated snapshot")


def require_resume_profile(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    if not args.resume:
        if args.dataset_root.exists():
            raise RuntimeError(
                f"New dataset root must not already exist: {args.dataset_root}"
            )
        return
    sidecar = profile_sidecar_path(args)
    if not sidecar.is_file():
        raise RuntimeError("Resume dataset has no ReBot training-profile sidecar")
    try:
        existing = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Resume dataset training-profile sidecar is unreadable") from exc
    verify_existing_sidecar(existing, args, contract)


def write_profile_sidecar(
    args: argparse.Namespace,
    profile: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    sidecar = profile_sidecar_path(args)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "training_profile_id": profile["profile_id"],
        "training_profile_version": profile["profile_version"],
        "training_profile_digest": args.profile_digest,
        "collection_contract_digest": args.collection_contract_digest,
        "coordinate_frame": profile["coordinate_contract"]["frame"],
        "profile_snapshot": profile,
        "collection_contract": contract,
    }
    if sidecar.is_file():
        try:
            existing = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Dataset profile sidecar is unreadable") from exc
        verify_existing_sidecar(existing, args, contract)
        return
    sidecar.write_text(json.dumps(payload, indent=2) + "\n")


RERUN_VIEWER_PORT = 9876
RERUN_NATIVE_BIN = Path(rr.__file__).resolve().parents[1] / "rerun_cli" / "rerun"


def read_control_decision(args: argparse.Namespace, expected_action: str) -> dict[str, Any]:
    try:
        payload = json.loads(args.control_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not read the operator's episode decision") from exc
    if not isinstance(payload, dict) or payload.get("action") != expected_action:
        raise RuntimeError(
            f"Episode decision mismatch: expected {expected_action!r}; refusing to guess"
        )
    return payload


def consume_published_decision(
    control_file: Path,
    events: dict[str, bool],
    last_sequence: int | None,
) -> int | None:
    """Consume an atomic browser decision without relying solely on signals.

    POSIX signals remain the low-latency wakeup, while the atomic control file
    is the durable source of truth for Finish, Re-record, and an unopposed Stop.
    Polling it closes the gap if a supervisor is briefly unable to forward a
    signal.
    """

    try:
        payload = json.loads(control_file.read_text())
    except (OSError, json.JSONDecodeError):
        return last_sequence
    if not isinstance(payload, dict):
        return last_sequence
    sequence = payload.get("sequence")
    if not isinstance(sequence, int) or sequence == last_sequence:
        return last_sequence
    action = payload.get("action")
    if action in {"finish", "finish_and_stop"}:
        events["finish"] = True
        print(f"GUI_EVENT {action} source=control_file", flush=True)
    elif action == "rerecord":
        events["rerecord"] = True
        events["finish"] = True
        print("GUI_EVENT rerecord_current_episode source=control_file", flush=True)
    elif action == "stop":
        events["stop"] = True
        events["finish"] = True
        print("GUI_EVENT stop_and_finalize source=control_file", flush=True)
    else:
        return last_sequence
    return sequence


def resolve_attempt_disposition(
    events: dict[str, bool],
    published_action: str | None,
) -> str:
    """Give an atomically published operator decision priority over shutdown."""

    if events.get("rerecord") or published_action == "rerecord":
        return "failed"
    if events.get("stop") and published_action not in {"finish", "finish_and_stop"}:
        return "aborted"
    return "kept"


def start_rerun_viewer(enabled: bool) -> bool:
    if not enabled:
        return False

    def viewer_ready() -> bool:
        try:
            with socket.create_connection(("127.0.0.1", RERUN_VIEWER_PORT), timeout=0.2):
                return True
        except OSError:
            return False

    if viewer_ready():
        return True
    try:
        rr.init("rebot_training_collection")
        # Do not use rr.spawn here. A detached viewer inherits the collector's
        # stdout pipe, which prevents owned_process.py and the GUI from seeing
        # EOF after the robot has already saved and disconnected.
        subprocess.Popen(
            [
                str(RERUN_NATIVE_BIN),
                f"--port={RERUN_VIEWER_PORT}",
                "--memory-limit=10%",
                "--server-memory-limit=0B",
                "--expect-data-soon",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if viewer_ready():
                return True
            time.sleep(0.05)
        raise RuntimeError("Rerun viewer did not open its local port")
    except Exception as exc:
        logging.warning("Rerun viewer could not start; attempt RRD files will still be saved: %s", exc)
        return False


def begin_attempt(
    args: argparse.Namespace,
    profile: dict[str, Any],
    contract: dict[str, Any],
    *,
    candidate_episode_index: int,
    attempt_number: int,
    viewer_available: bool,
    session_home: dict[str, Any],
) -> tuple[Path, dict[str, Any], rr.RecordingStream]:
    attempt_id = new_attempt_id()
    directory = attempt_path(args.attempt_root, args.dataset_root.name, attempt_id)
    directory.mkdir(parents=True, exist_ok=False)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "dataset": args.dataset_root.name,
        "repo_id": args.repo_id,
        "task": args.task,
        "started_at": utc_now(),
        "finished_at": None,
        "disposition": "recording",
        "archive_complete": False,
        "training_included": False,
        "training_episode_index": None,
        "candidate_episode_index": candidate_episode_index,
        "attempt_number_in_session": attempt_number,
        "samples": 0,
        "duration_s": 0.0,
        "actual_control_hz": None,
        "dataset_fps": args.dataset_fps,
        "control_hz_requested": args.control_hz,
        "training_profile": {
            "id": profile["profile_id"],
            "version": profile["profile_version"],
            "digest": args.profile_digest,
        },
        "collection_contract_digest": args.collection_contract_digest,
        "collection_contract": contract,
        "session_home": session_home,
        "cameras": {
            "overhead": {
                "dataset_key": "observation.images.front",
                "index": args.front_camera,
                "width": args.front_width,
                "height": args.front_height,
                "video": "overhead.mp4",
            },
            "wrist": {
                "dataset_key": "observation.images.side",
                "index": args.side_camera,
                "width": args.side_width,
                "height": args.side_height,
                "video": "wrist.mp4",
            },
        },
        "rerun": "attempt.rrd",
    }
    atomic_write_json(directory / "metadata.json", metadata)
    recording = rr.RecordingStream(
        "rebot_training_attempt",
        recording_id=uuid4(),
    )
    sinks: list[Any] = [rr.FileSink(str(directory / "attempt.partial.rrd"))]
    if viewer_available:
        sinks.insert(0, rr.GrpcSink(f"rerun+http://127.0.0.1:{RERUN_VIEWER_PORT}/proxy"))
    recording.set_sinks(*sinks)
    recording.log(
        "attempt/metadata",
        rr.TextDocument(json.dumps(metadata, indent=2, ensure_ascii=False)),
        static=True,
    )
    return directory, metadata, recording


def _rerun_entity(namespace: str, key: str) -> str:
    return f"{namespace}/{str(key).replace('.', '/')}"


def log_attempt_sample(
    recording: rr.RecordingStream,
    observation: dict[str, Any],
    action: dict[str, Any],
    sample_index: int,
    fps: int,
) -> None:
    recording.set_time("attempt_frame", sequence=sample_index)
    recording.set_time("attempt_time", duration=sample_index / fps)
    for namespace, values in (("observation", observation), ("action", action)):
        for key, value in values.items():
            if value is None:
                continue
            entity = _rerun_entity(namespace, str(key))
            if isinstance(value, numbers.Real) or (
                isinstance(value, np.ndarray) and value.ndim == 0
            ):
                recording.log(entity, rr.Scalars(float(value)))
                continue
            if not isinstance(value, np.ndarray):
                continue
            array = value
            if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
                array = np.transpose(array, (1, 2, 0))
            if array.ndim == 3:
                recording.log(entity, rr.Image(array).compress(jpeg_quality=85))
            elif array.ndim == 1:
                recording.log(entity, rr.Scalars(array.astype(float, copy=False)))


class AttemptRerunWriter:
    """Keep JPEG/RRD serialization off the 240 Hz motor-control thread."""

    def __init__(self, recording: rr.RecordingStream, fps: int, max_pending: int = 16) -> None:
        self.recording = recording
        self.fps = fps
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_pending)
        self._error: Exception | None = None
        self._finished = False
        self._submitted = 0
        self._written = 0
        self._thread = threading.Thread(
            target=self._run,
            name="attempt-rerun-writer",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _snapshot(values: dict[str, Any]) -> dict[str, Any]:
        return {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            for key, value in values.items()
        }

    def submit(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        sample_index: int,
    ) -> None:
        if self._error is not None:
            raise RuntimeError(f"Rerun archive writer failed: {self._error}") from self._error
        if self._finished:
            raise RuntimeError("Rerun archive writer is already closed")
        item = (self._snapshot(observation), self._snapshot(action), sample_index)
        try:
            self._queue.put_nowait(item)
            self._submitted += 1
        except queue.Full as exc:
            error = RuntimeError(
                "Rerun archive writer fell behind; stopping instead of dropping training frames"
            )
            self._error = error
            raise error from exc

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self._error is not None:
                continue
            observation, action, sample_index = item
            try:
                log_attempt_sample(
                    self.recording,
                    observation,
                    action,
                    sample_index,
                    self.fps,
                )
                self._written += 1
            except Exception as exc:
                self._error = exc

    def finish(self, expected_samples: int | None = None) -> None:
        if self._finished:
            if self._error is not None:
                raise RuntimeError(f"Rerun archive writer failed: {self._error}") from self._error
            if expected_samples is not None and (
                self._submitted != expected_samples or self._written != expected_samples
            ):
                raise RuntimeError(
                    "Rerun sample count mismatch: "
                    f"submitted={self._submitted}, written={self._written}, expected={expected_samples}"
                )
            return
        self._finished = True
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError(f"Rerun archive writer failed: {self._error}") from self._error
        if expected_samples is not None and (
            self._submitted != expected_samples or self._written != expected_samples
        ):
            raise RuntimeError(
                "Rerun sample count mismatch: "
                f"submitted={self._submitted}, written={self._written}, expected={expected_samples}"
            )


def _episode_index(dataset: LeRobotDataset) -> int:
    value = dataset.episode_buffer["episode_index"]
    if isinstance(value, np.ndarray):
        return int(value.item() if value.size == 1 else value[0])
    return int(value)


def buffered_sample_count(dataset: LeRobotDataset) -> int:
    buffer = dataset.episode_buffer
    if not isinstance(buffer, dict):
        return 0
    return int(buffer.get("size", 0))


def quarantine_stale_candidate_frames(
    dataset: LeRobotDataset,
    attempt_root: Path,
    dataset_name: str,
    candidate_episode_index: int,
) -> dict[str, Any] | None:
    """Move leftover PNGs away before reusing an unsaved episode index.

    A failed encoder can leave one camera directory behind after the in-memory
    episode buffer is released.  The next take would otherwise append to that
    same ``episode-N`` directory and silently mix two physical attempts.  This
    guard is deliberately recoverable: it moves every leftover camera
    directory into an audit quarantine and never deletes recorded frames.
    """

    dataset._wait_image_writer()
    stale_sources = {
        camera_key: dataset._get_image_file_dir(candidate_episode_index, camera_key)
        for camera_key in dataset.meta.camera_keys
    }
    stale_sources = {
        camera_key: source
        for camera_key, source in stale_sources.items()
        if source.exists()
    }
    if not stale_sources:
        return None
    for source in stale_sources.values():
        if not source.is_dir() or source.is_symlink():
            raise RuntimeError(f"Unsafe stale camera-frame path: {source}")

    quarantine_id = new_attempt_id()
    quarantine = attempt_root / dataset_name / "_stale_frame_quarantine" / quarantine_id
    quarantine.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "quarantine_id": quarantine_id,
        "dataset": dataset_name,
        "candidate_episode_index": candidate_episode_index,
        "quarantined_at": utc_now(),
        "reason": "stale camera frames existed before a new attempt",
        "camera_directories": {},
        "complete": False,
    }
    atomic_write_json(quarantine / "manifest.json", manifest)
    try:
        for camera_key, source in stale_sources.items():
            destination = quarantine / camera_key.replace(".", "_")
            if destination.exists():
                raise RuntimeError(f"Stale-frame quarantine destination exists: {destination}")
            source.replace(destination)
            manifest["camera_directories"][camera_key] = {
                "path": str(destination.relative_to(quarantine)),
                "png_frames": len(list(destination.glob("*.png"))),
            }
            atomic_write_json(quarantine / "manifest.json", manifest)
    except Exception as exc:
        manifest["error"] = str(exc)
        atomic_write_json(quarantine / "manifest.json", manifest)
        raise RuntimeError(
            "Could not isolate stale frames from the prior attempt; refusing to record"
        ) from exc
    manifest["complete"] = True
    atomic_write_json(quarantine / "manifest.json", manifest)
    return manifest


def _camera_artifact_name(camera_key: str) -> str:
    suffix = camera_key.rsplit(".", 1)[-1]
    names = {"front": "overhead.mp4", "side": "wrist.mp4"}
    if suffix not in names:
        raise RuntimeError(f"Unexpected camera key in attempt buffer: {camera_key}")
    return names[suffix]


def archive_attempt_videos(
    dataset: LeRobotDataset,
    directory: Path,
    *,
    samples: int,
) -> dict[str, dict[str, Any]]:
    if samples <= 0:
        return {
            "overhead": {"path": "overhead.mp4", "available": False, "bytes": 0},
            "wrist": {"path": "wrist.mp4", "available": False, "bytes": 0},
        }
    dataset._wait_image_writer()
    episode_index = _episode_index(dataset)
    artifacts: dict[str, dict[str, Any]] = {}
    pending: list[tuple[Path, Path]] = []
    for camera_key in dataset.meta.camera_keys:
        filename = _camera_artifact_name(camera_key)
        camera_name = "overhead" if filename == "overhead.mp4" else "wrist"
        source = dataset._get_image_file_dir(episode_index, camera_key)
        destination = directory / filename
        partial = directory / filename.replace(".mp4", ".partial.mp4")
        encode_video_frames(
            source,
            partial,
            dataset.fps,
            vcodec="h264",
            overwrite=True,
        )
        with av.open(str(partial), "r") as container:
            if not container.streams.video:
                raise RuntimeError(f"Encoded attempt video has no video stream: {partial}")
            stream = container.streams.video[0]
            decoded_frames = sum(1 for _frame in container.decode(stream))
            encoded_fps = float(stream.average_rate or stream.base_rate or 0)
        if decoded_frames != samples:
            raise RuntimeError(
                f"Attempt video frame mismatch for {camera_key}: {decoded_frames} != {samples}"
            )
        if abs(encoded_fps - dataset.fps) > 0.05:
            raise RuntimeError(
                f"Attempt video FPS mismatch for {camera_key}: {encoded_fps} != {dataset.fps}"
            )
        pending.append((partial, destination))
        artifacts[camera_name] = {
            "path": filename,
            "available": True,
            "bytes": partial.stat().st_size,
            "sha256": sha256(partial),
            "frames": decoded_frames,
            "fps": encoded_fps,
            "dataset_key": camera_key,
        }
    if set(artifacts) != {"overhead", "wrist"}:
        raise RuntimeError("Attempt archive must contain exactly overhead and wrist videos")
    for partial, destination in pending:
        partial.replace(destination)
    return artifacts


@contextmanager
def reuse_attempt_videos_for_lerobot(
    dataset: LeRobotDataset,
    directory: Path,
):
    """Hand verified attempt MP4s to LeRobot instead of encoding them twice.

    ``finalize_attempt_archive`` has already encoded and frame-checked one MP4
    per camera.  LeRobot normally encodes the same PNG directories again in
    ``save_episode``.  Temporarily replace that encoder with a hard-link (or a
    copy when links are unavailable) to the verified archive video.  LeRobot
    is then free to move/concatenate its private link while the replay archive
    remains immutable beside ``attempt.rrd``.
    """

    original_encoder = dataset._encode_temporary_episode_video
    expected_keys = set(dataset.meta.video_keys)
    handed_off: set[str] = set()

    def archived_video(video_key: str, episode_index: int) -> Path:
        if video_key not in expected_keys:
            raise RuntimeError(f"Unexpected LeRobot video key: {video_key}")
        source = directory / _camera_artifact_name(video_key)
        if not source.is_file() or source.stat().st_size <= 0:
            raise RuntimeError(f"Verified attempt video is missing: {source}")

        temporary_dir = Path(
            tempfile.mkdtemp(prefix=".rebot-video-handoff-", dir=dataset.root)
        )
        temporary_video = temporary_dir / f"episode-{episode_index:06d}.mp4"
        try:
            os.link(source, temporary_video)
        except OSError:
            shutil.copy2(source, temporary_video)

        # This mirrors LeRobot's normal encoder worker: after the encoded
        # video exists, its temporary source PNGs are no longer needed.
        frame_directory = dataset._get_image_file_dir(episode_index, video_key)
        if frame_directory.is_dir():
            shutil.rmtree(frame_directory)
        handed_off.add(video_key)
        return temporary_video

    dataset._encode_temporary_episode_video = archived_video  # type: ignore[method-assign]
    try:
        yield
        if handed_off != expected_keys:
            missing = sorted(expected_keys - handed_off)
            raise RuntimeError(f"LeRobot did not consume archived camera videos: {missing}")
    finally:
        dataset._encode_temporary_episode_video = original_encoder  # type: ignore[method-assign]


def checkpoint_and_reopen_dataset(
    dataset: LeRobotDataset,
    *,
    expected_episode_index: int,
) -> LeRobotDataset:
    """Finalize one episode and prove a fresh LeRobot loader can read it.

    Reusing a finalized dataset object would reopen and overwrite its last
    parquet file.  A fresh object intentionally advances to new parquet/video
    files, so every subsequent attempt stays independently durable.
    """

    expected_episodes = expected_episode_index + 1
    expected_frames = int(dataset.meta.total_frames)
    expected_fps = int(dataset.fps)
    expected_video_keys = set(dataset.meta.video_keys)
    dataset._wait_image_writer()
    dataset.stop_image_writer()
    dataset.finalize()

    reopened = LeRobotDataset(
        dataset.repo_id,
        root=dataset.root,
        batch_encoding_size=1,
        vcodec=dataset.vcodec,
    )
    if reopened.num_episodes != expected_episodes:
        raise RuntimeError(
            "LeRobot durability check failed: "
            f"fresh loader sees {reopened.num_episodes} episodes, expected {expected_episodes}"
        )
    if reopened.num_frames != expected_frames:
        raise RuntimeError(
            "LeRobot durability check failed: "
            f"fresh loader sees {reopened.num_frames} frames, expected {expected_frames}"
        )
    if reopened.fps != expected_fps or set(reopened.meta.video_keys) != expected_video_keys:
        raise RuntimeError("LeRobot durability check failed: reopened schema does not match")
    reopened.start_image_writer(num_processes=0, num_threads=8)
    return reopened


def verify_and_commit_rerun(directory: Path) -> Path:
    partial = directory / "attempt.partial.rrd"
    final = directory / "attempt.rrd"
    if not partial.is_file() or partial.stat().st_size <= 0:
        raise RuntimeError("Rerun archive was not written")
    if not RERUN_NATIVE_BIN.is_file():
        raise RuntimeError(f"Rerun verifier is unavailable: {RERUN_NATIVE_BIN}")
    verified = subprocess.run(
        [str(RERUN_NATIVE_BIN), "rrd", "verify", str(partial)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    if verified.returncode != 0:
        raise RuntimeError(
            "Rerun archive verification failed: " + verified.stdout.strip()[-1000:]
        )
    partial.replace(final)
    return final


def rerun_result_snapshot(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return immutable-at-recording facts without pretending mutable fields are current."""

    disposition = str(metadata.get("operator_disposition") or metadata.get("disposition"))
    result: dict[str, Any] = {
        "schema_version": 1,
        "attempt_id": metadata.get("attempt_id"),
        "decision_at_recording": disposition,
        "samples": metadata.get("samples"),
        "duration_s": metadata.get("duration_s"),
        "metadata_authority": "metadata.json beside this RRD is authoritative for current labels and training inclusion",
        "training_status_at_close": "pending_dataset_save" if disposition == "kept" else "excluded",
    }
    if disposition == "failed":
        result["failure_label_at_recording"] = metadata.get("failure_label")
        result["failure_note_at_recording"] = metadata.get("failure_note", "")
    return result


def preserve_raw_attempt_frames(
    dataset: LeRobotDataset,
    directory: Path,
) -> dict[str, Any]:
    """Move unsaved source PNGs out of LeRobot's exception-cleanup path."""

    dataset._wait_image_writer()
    episode_index = _episode_index(dataset)
    raw_root = directory / "raw_frames"
    raw_root.mkdir(parents=True, exist_ok=True)
    moved: dict[str, str] = {}
    expected = list(dataset.meta.camera_keys)
    for camera_key in expected:
        source = dataset._get_image_file_dir(episode_index, camera_key)
        destination = raw_root / camera_key.replace(".", "_")
        if destination.is_dir() and not source.exists():
            moved[camera_key] = str(destination.relative_to(directory))
            continue
        if destination.exists():
            raise RuntimeError(f"Raw-frame recovery destination already exists: {destination}")
        if not source.is_dir():
            continue
        source.replace(destination)
        moved[camera_key] = str(destination.relative_to(directory))
    return {
        "preserved": set(moved) == set(expected),
        "camera_directories": moved,
        "expected_camera_keys": expected,
    }


def finalize_attempt_archive(
    *,
    dataset: LeRobotDataset,
    directory: Path,
    metadata: dict[str, Any],
    recording: rr.RecordingStream,
    rerun_writer: AttemptRerunWriter | None = None,
    disposition: str,
    samples: int,
    duration_s: float,
    actual_hz: float,
    failure_label: str = "",
    failure_note: str = "",
    archive_error: str = "",
) -> dict[str, Any]:
    # A segment can raise after frames were already accepted by LeRobot but
    # before returning its local counter. The dataset buffer is authoritative.
    samples = buffered_sample_count(dataset)
    if disposition == "failed":
        failure_label, failure_note = validate_failure_label(failure_label, failure_note)
    elif failure_label or failure_note:
        raise RuntimeError("Only failed attempts may have failure labels")
    metadata["operator_disposition"] = disposition
    if disposition == "failed":
        # Persist the human decision before any encoder/Rerun work can fail.
        metadata["failure_label"] = failure_label
        metadata["failure_note"] = failure_note
    try:
        if rerun_writer is not None:
            rerun_writer.finish(expected_samples=samples)
        videos = archive_attempt_videos(dataset, directory, samples=samples)
        metadata.update(
            {
                "finished_at": utc_now(),
                "disposition": disposition,
                "samples": samples,
                "duration_s": round(duration_s, 3),
                "actual_control_hz": round(actual_hz, 2),
                "videos": videos,
                "archive_complete": False,
            }
        )
        if archive_error:
            metadata["archive_error"] = archive_error
        if disposition == "failed":
            metadata["failure_label"] = failure_label
            metadata["failure_note"] = failure_note
        else:
            metadata.pop("failure_label", None)
            metadata.pop("failure_note", None)
        recording.log(
            "attempt/result",
            rr.TextDocument(
                json.dumps(rerun_result_snapshot(metadata), indent=2, ensure_ascii=False)
            ),
            static=True,
        )
        recording.flush()
        recording.disconnect()
        rrd_path = verify_and_commit_rerun(directory)
        metadata["rerun_artifact"] = {
            "path": "attempt.rrd",
            "available": True,
            "bytes": rrd_path.stat().st_size,
            "sha256": sha256(rrd_path),
        }
        metadata["archive_complete"] = True
        atomic_write_json(directory / "metadata.json", metadata)
        return metadata
    except Exception as exc:
        recovery: dict[str, Any]
        try:
            recovery = preserve_raw_attempt_frames(dataset, directory)
        except Exception as recovery_exc:
            recovery = {
                "preserved": False,
                "camera_directories": {},
                "error": str(recovery_exc),
            }
        metadata.update(
            {
                "finished_at": utc_now(),
                "disposition": "collector_error",
                "archive_complete": False,
                "training_included": False,
                "samples": samples,
                "duration_s": round(duration_s, 3),
                "actual_control_hz": round(actual_hz, 2),
                "archive_error": str(exc),
                "raw_frame_recovery": recovery,
            }
        )
        try:
            atomic_write_json(directory / "metadata.json", metadata)
        except OSError:
            logging.exception("Could not write collector-error attempt metadata")
        try:
            recording.log(
                "attempt/result",
                rr.TextDocument(json.dumps(rerun_result_snapshot(metadata), indent=2)),
                static=True,
            )
            recording.flush()
        except Exception:
            logging.exception("Could not append collector error to Rerun archive")
        try:
            recording.disconnect()
        except Exception:
            logging.exception("Could not disconnect failed Rerun recording")
        raise


def mark_attempt_in_training(directory: Path, episode_index: int) -> None:
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict) or metadata.get("disposition") != "kept":
        raise RuntimeError("Only a kept attempt can be linked to a training episode")
    metadata["training_included"] = True
    metadata["training_episode_index"] = episode_index
    metadata["training_saved_at"] = utc_now()
    metadata["training_commit_state"] = "durable"
    metadata["lerobot_fresh_load_verified"] = True
    atomic_write_json(path, metadata)


def mark_attempt_home_return(directory: Path, result: dict[str, Any]) -> None:
    """Persist proof that the reset gate completed before the next attempt."""

    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict) or metadata.get("disposition") == "recording":
        raise RuntimeError("Only a finalized attempt can record a home return")
    metadata["home_return"] = result
    atomic_write_json(path, metadata)


def episode_visible_in_disk_info(dataset_root: Path, episode_index: int) -> bool:
    """Return whether LeRobot's durable info file exposes this episode."""

    try:
        info = json.loads((dataset_root / "meta" / "info.json").read_text())
        total_episodes = int(info["total_episodes"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return total_episodes > episode_index


def mark_training_save_failure(
    directory: Path,
    metadata: dict[str, Any],
    dataset_root: Path,
    episode_index: int,
    error: Exception,
) -> None:
    """Record a truthful state when LeRobot raises during its commit sequence."""

    metadata["operator_disposition"] = "kept"
    metadata["training_save_error"] = str(error)
    metadata["training_episode_index"] = episode_index
    if episode_visible_in_disk_info(dataset_root, episode_index):
        # Data/video and info may be durable while stats or buffered episode
        # metadata failed.  Neither included nor excluded is truthful until the
        # dataset validator reconciles it, so block training as uncertain.
        metadata["disposition"] = "commit_uncertain"
        metadata["training_included"] = None
        metadata["training_commit_state"] = "uncertain"
    else:
        metadata["disposition"] = "collector_error"
        metadata["training_included"] = False
        metadata["training_commit_state"] = "not_committed"
    atomic_write_json(directory / "metadata.json", metadata)


def clear_unsaved_attempt(dataset: LeRobotDataset) -> None:
    if not dataset.episode_buffer:
        return
    dataset._wait_image_writer()
    episode_index = _episode_index(dataset)
    for camera_key in dataset.meta.camera_keys:
        directory = dataset._get_image_file_dir(episode_index, camera_key)
        if directory.is_dir():
            shutil.rmtree(directory)
    dataset.clear_episode_buffer(delete_images=False)


def release_unsaved_attempt_after_archive(
    dataset: LeRobotDataset,
    directory: Path,
    metadata: dict[str, Any],
) -> None:
    """Release LeRobot's buffer only after an archive or raw recovery exists."""

    if metadata.get("archive_complete"):
        clear_unsaved_attempt(dataset)
        return
    recovery = metadata.get("raw_frame_recovery")
    if not isinstance(recovery, dict) or not recovery.get("camera_directories"):
        recovery = preserve_raw_attempt_frames(dataset, directory)
        metadata["raw_frame_recovery"] = recovery
        try:
            atomic_write_json(directory / "metadata.json", metadata)
        except OSError:
            logging.exception("Could not update raw-frame recovery metadata")
    if dataset.episode_buffer:
        # The PNG directories now live inside the attempt archive. Reset only
        # the in-memory buffer and leave the recovered files untouched.
        dataset.clear_episode_buffer(delete_images=False)


class RecoverySafeVideoEncodingManager(VideoEncodingManager):
    """Finalize saved episodes without deleting an interrupted take on errors.

    The upstream context manager removes the current PNG directories whenever
    an exception escapes. That is unsafe when the exception itself is ENOSPC or
    an archive failure: recovery may also be unable to create a destination.
    On an error path, leave all interrupted source frames in place for the run
    quarantine/recovery workflow.
    """

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is None:
            return bool(super().__exit__(exc_type, exc_val, exc_tb))
        if self.dataset.episodes_since_last_encoding > 0:
            start_ep = self.dataset.num_episodes - self.dataset.episodes_since_last_encoding
            end_ep = self.dataset.num_episodes
            try:
                self.dataset._batch_save_episode_video(start_ep, end_ep)
            except Exception:
                logging.exception("Could not encode already-saved episodes during error cleanup")
        try:
            self.dataset.finalize()
        except Exception:
            logging.exception("Could not finalize dataset writers during error cleanup")
        logging.error(
            "Collector error: interrupted source PNGs are intentionally preserved under %s",
            self.dataset.root / "images",
        )
        return False


def install_signal_controls() -> dict[str, bool]:
    events = {"finish": False, "rerecord": False, "stop": False}

    def finish(_signum: int, _frame: Any) -> None:
        events["finish"] = True
        print("GUI_EVENT finish_current_episode", flush=True)

    def rerecord(_signum: int, _frame: Any) -> None:
        events["rerecord"] = True
        events["finish"] = True
        print("GUI_EVENT rerecord_current_episode", flush=True)

    def stop(_signum: int, _frame: Any) -> None:
        events["stop"] = True
        events["finish"] = True
        print("GUI_EVENT stop_and_finalize", flush=True)

    signal.signal(signal.SIGUSR1, finish)
    signal.signal(signal.SIGUSR2, rerecord)
    signal.signal(signal.SIGHUP, stop)
    signal.signal(signal.SIGTERM, stop)
    print(
        "GUI_CONTROL ready (finish=SIGUSR1 rerecord=SIGUSR2 stop=SIGHUP)",
        flush=True,
    )
    return events


def make_hardware(
    args: argparse.Namespace,
    profile: dict[str, Any],
) -> tuple[SeeedB601DMFollower, RebotArm102Leader]:
    calibration = profile["calibration"]
    joints = profile["coordinate_contract"]["joints"]
    expected_names = [joint["name"] for joint in joints]
    expected_limits = {
        joint["name"]: tuple(float(value) for value in joint["soft_limit_degrees"])
        for joint in joints
    }
    expected_directions = {
        joint["name"]: float(joint["leader_to_follower_scale"])
        for joint in joints
    }
    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=args.front_camera,
            fps=args.dataset_fps,
            width=args.front_width,
            height=args.front_height,
            fourcc="MJPG",
        ),
        "side": OpenCVCameraConfig(
            index_or_path=args.side_camera,
            fps=args.dataset_fps,
            width=args.side_width,
            height=args.side_height,
            fourcc="MJPG",
        ),
    }
    follower_config = SeeedB601DMFollowerConfig(
        port=args.follower_port,
        id=str(calibration["follower"]["id"]),
        calibration_dir=args.follower_calibration.parent,
        can_adapter="damiao",
        dm_serial_baud=921600,
        max_relative_target=args.max_step,
        pos_vel_velocity=[args.motor_velocity] * 7,
        force_pos_torque_ration=args.gripper_force,
        disable_torque_on_disconnect=True,
        cameras=cameras,
    )
    leader_config = RebotArm102LeaderConfig(
        port=args.leader_port,
        id=str(calibration["leader"]["id"]),
        calibration_dir=args.leader_calibration.parent,
        baudrate=1_000_000,
    )
    follower = SeeedB601DMFollower(follower_config)
    leader = RebotArm102Leader(leader_config)
    actual_limits = {
        name: tuple(float(value) for value in values)
        for name, values in follower_config.joint_limits.items()
    }
    actual_directions = {
        name: float(value) for name, value in follower_config.joint_directions.items()
    }
    if actual_limits != expected_limits or actual_directions != expected_directions:
        raise RuntimeError("Loaded follower directions/limits do not match the training profile")
    if list(leader_config.joint_ids) != expected_names or list(leader_config.joint_ids.values()) != list(range(7)):
        raise RuntimeError("Loaded leader joint order/IDs do not match the training profile")
    if follower.calibration_fpath.resolve() != args.follower_calibration.resolve():
        raise RuntimeError("Follower resolved a different calibration file than the profile")
    if leader.calibration_fpath.resolve() != args.leader_calibration.resolve():
        raise RuntimeError("Leader resolved a different calibration file than the profile")
    return follower, leader


def make_dataset(
    args: argparse.Namespace,
    robot: SeeedB601DMFollower,
    action_processor: Any,
    observation_processor: Any,
) -> LeRobotDataset:
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )
    if args.resume:
        dataset = LeRobotDataset(
            args.repo_id,
            root=args.dataset_root,
            batch_encoding_size=1,
            vcodec="h264",
        )
        if dataset.fps != args.dataset_fps:
            raise RuntimeError(
                f"Resume FPS mismatch: dataset={dataset.fps}, requested={args.dataset_fps}"
            )
        expected_features = {**features, **DEFAULT_FEATURES}
        if semantic_feature_schema(dataset.features) != semantic_feature_schema(
            expected_features
        ):
            raise RuntimeError("Resume feature schema does not match the connected ReBot/cameras")
        dataset.start_image_writer(num_processes=0, num_threads=8)
        return dataset
    return LeRobotDataset.create(
        args.repo_id,
        args.dataset_fps,
        root=args.dataset_root,
        robot_type=robot.name,
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=8,
        batch_encoding_size=1,
        vcodec="h264",
    )


def semantic_feature_schema(features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Normalize the stable schema fields used to authorize dataset resume.

    A fresh dataset declaration omits LeRobot-owned index fields and encoded
    video metadata.  A loaded dataset contains both.  Resume must compare the
    complete learning plus LeRobot-owned dtype/shape/name contract while
    ignoring derived codec details such as pix_fmt and has_audio.
    """

    def canonical(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return value

    return {
        key: canonical(
            {
                field: feature.get(field)
                for field in ("dtype", "shape", "names")
            }
        )
        for key, feature in sorted(features.items())
    }


def build_training_frame(
    dataset_features: dict[str, dict[str, Any]],
    observation: dict[str, Any],
    sent_action: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    """Build only declared learning features; LeRobot owns frame metadata.

    Pinned LeRobot validates a caller-supplied frame before it creates
    ``timestamp`` and ``frame_index``. Supplying either metadata key here makes
    the very first sample fail as an extra feature.
    """

    observation_frame = build_dataset_frame(dataset_features, observation, prefix=OBS_STR)
    action_frame = build_dataset_frame(dataset_features, sent_action, prefix=ACTION)
    frame = {**observation_frame, **action_frame, "task": task}
    supplied_metadata = sorted(set(frame).intersection(DEFAULT_FEATURES))
    if supplied_metadata:
        raise RuntimeError(
            "Collector frame must not supply LeRobot-owned metadata: "
            + ", ".join(supplied_metadata)
        )
    return frame


HOME_RETURN_SPEED_DEG_S = 120.0
HOME_FOLLOWER_TOLERANCE_DEG = 2.0
HOME_GRIPPER_TOLERANCE_DEG = 5.0
HOME_LEADER_MIN_TOLERANCE_DEG = 1.0
HOME_SETTLE_TIME_S = 0.25


def _joint_positions(
    values: dict[str, Any],
    feature_names: list[str],
) -> dict[str, float]:
    missing = [name for name in feature_names if name not in values]
    if missing:
        raise RuntimeError("Joint position snapshot is missing: " + ", ".join(missing))
    return {name: float(values[name]) for name in feature_names}


def read_follower_joint_observation(robot: SeeedB601DMFollower) -> dict[str, float]:
    """Read follower motor state without waiting on either 30 FPS camera.

    The follower driver's public ``get_observation`` also consumes a new frame
    from every camera. Calling it in the control loop therefore caps teleop at
    camera FPS. This motor-only read preserves the driver's feedback/error
    contract while letting the requested control clock run independently.
    """

    # Preserve the generic Robot protocol for simulations/test doubles. The
    # locked B601 runtime always takes the motor-only path below.
    if not hasattr(robot, "motors") or not hasattr(robot, "bus"):
        return {
            key: float(value)
            for key, value in robot.get_observation().items()
            if key.endswith((".pos", ".vel", ".torque"))
        }

    for motor in robot.motors.values():
        motor.request_feedback()
    try:
        robot.bus.poll_feedback_once()
    except Exception as exc:
        raise RuntimeError("Follower feedback poll failed; teleoperation stopped.") from exc

    observation: dict[str, float] = {}
    for motor_name, motor in robot.motors.items():
        state = motor.get_state()
        if state is None:
            raise RuntimeError(
                f"Follower motor {motor_name!r} has no feedback; teleoperation stopped."
            )
        observation[f"{motor_name}.pos"] = math.degrees(state.pos)
        observation[f"{motor_name}.vel"] = math.degrees(state.vel)
        observation[f"{motor_name}.torque"] = float(state.torq)
    return observation


def _camera_frame_age_ms(camera: Any) -> float | None:
    timestamp = getattr(camera, "latest_timestamp", None)
    if not isinstance(timestamp, numbers.Real) or isinstance(timestamp, bool):
        return None
    return max(0.0, (time.perf_counter() - float(timestamp)) * 1e3)


def _camera_freshness_entry(telemetry: dict[str, Any], camera_key: str) -> dict[str, Any]:
    return telemetry.setdefault(
        camera_key,
        {
            "fresh_limit_ms": CAMERA_FRESH_AGE_MS,
            "jitter_limit_ms": CAMERA_JITTER_MAX_AGE_MS,
            "max_consecutive_jitter_samples": CAMERA_MAX_CONSECUTIVE_JITTER_SAMPLES,
            "sample_attempts": 0,
            "fresh_samples": 0,
            "transient_jitter_samples": 0,
            "consecutive_jitter_samples": 0,
            "max_consecutive_jitter_seen": 0,
            "last_age_ms": None,
            "max_age_ms_seen": 0.0,
            "status": "waiting_for_sample",
        },
    )


def read_latest_camera_observation(
    robot: SeeedB601DMFollower,
    freshness_telemetry: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Peek at camera buffers with bounded jitter tolerance and no frame wait."""

    telemetry = freshness_telemetry if freshness_telemetry is not None else {}
    observation: dict[str, np.ndarray] = {}
    for camera_key, camera in robot.cameras.items():
        entry = _camera_freshness_entry(telemetry, camera_key)
        entry["sample_attempts"] += 1
        try:
            # read_latest only copies the current buffer. It never clears the
            # new-frame event or waits for camera hardware.
            frame = camera.read_latest(max_age_ms=CAMERA_JITTER_MAX_AGE_MS)
        except Exception as exc:
            age_ms = _camera_frame_age_ms(camera)
            if age_ms is not None:
                entry["last_age_ms"] = round(age_ms, 3)
                entry["max_age_ms_seen"] = round(
                    max(float(entry["max_age_ms_seen"]), age_ms), 3
                )
            entry["status"] = "hard_stale_or_unavailable"
            entry["error"] = str(exc)
            print(
                f"CAMERA_FRESHNESS key={camera_key} status=hard_stale_or_unavailable "
                f"age_ms={age_ms if age_ms is not None else 'unknown'}",
                flush=True,
            )
            raise

        age_ms = _camera_frame_age_ms(camera)
        if age_ms is None:
            # Test doubles and non-OpenCV cameras may not expose a timestamp;
            # the camera's own bounded read_latest contract remains authoritative.
            entry["fresh_samples"] += 1
            entry["consecutive_jitter_samples"] = 0
            entry["status"] = "fresh_age_reported_by_camera"
            observation[camera_key] = frame
            continue

        entry["last_age_ms"] = round(age_ms, 3)
        entry["max_age_ms_seen"] = round(
            max(float(entry["max_age_ms_seen"]), age_ms), 3
        )
        if age_ms <= CAMERA_FRESH_AGE_MS:
            entry["fresh_samples"] += 1
            entry["consecutive_jitter_samples"] = 0
            entry["status"] = "fresh"
            entry.pop("error", None)
        else:
            entry["transient_jitter_samples"] += 1
            entry["consecutive_jitter_samples"] += 1
            entry["max_consecutive_jitter_seen"] = max(
                int(entry["max_consecutive_jitter_seen"]),
                int(entry["consecutive_jitter_samples"]),
            )
            if (
                entry["consecutive_jitter_samples"]
                > CAMERA_MAX_CONSECUTIVE_JITTER_SAMPLES
            ):
                entry["status"] = "sustained_stale"
                entry["error"] = (
                    f"{camera_key} stayed older than {CAMERA_FRESH_AGE_MS:g} ms for "
                    f"{entry['consecutive_jitter_samples']} consecutive samples"
                )
                print(
                    f"CAMERA_FRESHNESS key={camera_key} status=sustained_stale "
                    f"age_ms={age_ms:.1f} consecutive="
                    f"{entry['consecutive_jitter_samples']}",
                    flush=True,
                )
                raise TimeoutError(entry["error"])
            entry["status"] = "transient_jitter_accepted"
            print(
                f"CAMERA_FRESHNESS key={camera_key} status=transient_jitter_accepted "
                f"age_ms={age_ms:.1f} consecutive={entry['consecutive_jitter_samples']} "
                f"limit={CAMERA_MAX_CONSECUTIVE_JITTER_SAMPLES}",
                flush=True,
            )
        observation[camera_key] = frame
    return observation


def _physical_positions_to_follower_input(
    robot: SeeedB601DMFollower,
    physical_positions: dict[str, float],
) -> dict[str, float]:
    action: dict[str, float] = {}
    for feature, physical_position in physical_positions.items():
        motor_name = feature.removesuffix(".pos")
        direction = float(robot.config.joint_directions.get(motor_name, 0.0))
        if direction == 0.0:
            raise RuntimeError(f"Cannot return {motor_name}: direction scale is zero")
        action[feature] = physical_position / direction
    return action


def capture_session_home(
    robot: SeeedB601DMFollower,
    leader: RebotArm102Leader,
) -> dict[str, Any]:
    """Capture the stationary pose that this collection session returns to."""

    feature_names = list(robot.action_features)
    follower_positions = _joint_positions(
        read_follower_joint_observation(robot),
        feature_names,
    )
    leader_positions = _joint_positions(leader.get_action(), feature_names)
    follower_input = _physical_positions_to_follower_input(robot, follower_positions)
    leader_tolerances: dict[str, float] = {}
    follower_tolerances: dict[str, float] = {}
    for feature in feature_names:
        motor_name = feature.removesuffix(".pos")
        direction = abs(float(robot.config.joint_directions[motor_name]))
        follower_tolerance = (
            HOME_GRIPPER_TOLERANCE_DEG
            if motor_name == "gripper"
            else HOME_FOLLOWER_TOLERANCE_DEG
        )
        follower_tolerances[feature] = follower_tolerance
        leader_tolerances[feature] = max(
            HOME_LEADER_MIN_TOLERANCE_DEG,
            follower_tolerance / direction,
        )
    return {
        "captured_at": utc_now(),
        "capture_event": "collection_process_connected_before_first_attempt",
        "follower_positions_deg": follower_positions,
        "follower_input_deg": follower_input,
        "leader_positions_deg": leader_positions,
        "follower_tolerance_deg": follower_tolerances,
        "leader_tolerance_deg": leader_tolerances,
        "return_speed_deg_s": HOME_RETURN_SPEED_DEG_S,
        "settle_time_s": HOME_SETTLE_TIME_S,
    }


def _inside_joint_tolerances(
    current: dict[str, float],
    target: dict[str, float],
    tolerances: dict[str, float],
) -> bool:
    return all(
        abs(float(current[name]) - float(target[name])) <= float(tolerances[name])
        for name in target
    )


def automatic_reset_to_session_home(
    *,
    args: argparse.Namespace,
    robot: SeeedB601DMFollower,
    leader: RebotArm102Leader,
    events: dict[str, bool],
    session_home: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the follower home, then gate the next take on leader alignment."""

    follower_target = dict(session_home["follower_positions_deg"])
    leader_target = dict(session_home["leader_positions_deg"])
    follower_tolerances = dict(session_home["follower_tolerance_deg"])
    leader_tolerances = dict(session_home["leader_tolerance_deg"])
    feature_names = list(follower_target)
    control_period = 1.0 / args.control_hz
    maximum_step = min(
        float(args.max_step),
        HOME_RETURN_SPEED_DEG_S / float(args.control_hz),
    )
    started = time.perf_counter()
    next_control = started
    stable_since: float | None = None
    last_status = started - 1.0
    loops = 0

    print(
        f"RESET auto_home active minimum_seconds={args.reset_time_s:g} "
        f"return_speed={HOME_RETURN_SPEED_DEG_S:g}deg_s",
        flush=True,
    )
    while True:
        if events["stop"]:
            print("RESET auto_home interrupted_by_stop", flush=True)
            return None

        observation = _joint_positions(
            read_follower_joint_observation(robot),
            feature_names,
        )
        intermediate_physical = {
            name: observation[name]
            + max(
                -maximum_step,
                min(maximum_step, follower_target[name] - observation[name]),
            )
            for name in feature_names
        }
        robot.send_action(
            _physical_positions_to_follower_input(robot, intermediate_physical)
        )
        leader_positions = _joint_positions(leader.get_action(), feature_names)
        now = time.perf_counter()
        loops += 1

        follower_aligned = _inside_joint_tolerances(
            observation,
            follower_target,
            follower_tolerances,
        )
        leader_aligned = _inside_joint_tolerances(
            leader_positions,
            leader_target,
            leader_tolerances,
        )
        if follower_aligned and leader_aligned:
            stable_since = stable_since or now
        else:
            stable_since = None

        elapsed = now - started
        stable = stable_since is not None and now - stable_since >= HOME_SETTLE_TIME_S
        if elapsed >= float(args.reset_time_s) and stable:
            result = {
                "completed_at": utc_now(),
                "elapsed_s": round(elapsed, 3),
                "control_loops": loops,
                "follower_aligned": True,
                "leader_aligned": True,
            }
            print(
                f"RESET auto_home complete elapsed={elapsed:.1f}s "
                "next_attempt_ready=true",
                flush=True,
            )
            return result

        if now - last_status >= 1.0:
            follower_delta = max(
                abs(observation[name] - follower_target[name]) for name in feature_names
            )
            leader_delta = max(
                abs(leader_positions[name] - leader_target[name]) for name in feature_names
            )
            waiting_for = []
            if not follower_aligned:
                waiting_for.append("follower")
            if not leader_aligned:
                waiting_for.append("leader")
            if elapsed < float(args.reset_time_s):
                waiting_for.append("reset_timer")
            print(
                f"RESET auto_home elapsed={elapsed:.1f}s "
                f"follower_delta={follower_delta:.1f}deg "
                f"leader_delta={leader_delta:.1f}deg "
                f"waiting_for={'+'.join(waiting_for) or 'settle'} "
                f"waiting_for_operator={str(not leader_aligned).lower()}",
                flush=True,
            )
            last_status = now

        next_control += control_period
        if next_control < now - control_period:
            next_control = now
        precise_sleep(max(next_control - time.perf_counter(), 0.0))


def control_segment(
    *,
    args: argparse.Namespace,
    robot: SeeedB601DMFollower,
    leader: RebotArm102Leader,
    dataset: LeRobotDataset,
    events: dict[str, bool],
    duration_s: float,
    record: bool,
    task: str,
    rerun_writer: AttemptRerunWriter | None = None,
    sample_offset: int = 0,
    camera_freshness: dict[str, Any] | None = None,
) -> tuple[int, float]:
    control_period = 1.0 / args.control_hz
    sample_period = 1.0 / args.dataset_fps
    started = time.perf_counter()
    next_control = started
    next_sample = started
    loops = 0
    samples = 0
    status_started = started
    status_loops = 0
    minimum_free_bytes = 5 * 1024**3
    next_decision_poll = started
    last_decision_sequence: int | None = None

    while time.perf_counter() - started < duration_s:
        now = time.perf_counter()
        if record and now >= next_decision_poll:
            last_decision_sequence = consume_published_decision(
                args.control_file,
                events,
                last_decision_sequence,
            )
            next_decision_poll = now + 0.05
        if events["finish"] or events["stop"]:
            break
        loop_started = time.perf_counter()
        try:
            joint_observation = read_follower_joint_observation(robot)
            leader_action = leader.get_action()
            sent_action = robot.send_action(leader_action)
        except Exception:
            # SIGHUP/SIGTERM can interrupt a native serial read that was already
            # in flight. Once the stop handler has acknowledged that request,
            # the resulting EINTR is expected shutdown—not a collector fault.
            # The same I/O failure without a stop request remains fatal.
            if events["stop"]:
                print("CONTROL stop_acknowledged_during_io", flush=True)
                break
            raise
        now = time.perf_counter()
        loops += 1
        status_loops += 1

        if record and now + 1e-9 >= next_sample:
            observation = {
                **joint_observation,
                **read_latest_camera_observation(robot, camera_freshness),
            }
            # The learning target is the exact follower-space action after directions,
            # limits, and per-tick clipping—not the raw leader command.
            # LeRobot adds frame_index and the exact frame_index / dataset_fps
            # timestamp after validating only the declared feature schema.
            dataset.add_frame(
                build_training_frame(dataset.features, observation, sent_action, task)
            )
            samples += 1
            while next_sample <= now:
                next_sample += sample_period
            if rerun_writer is not None:
                rerun_writer.submit(
                    observation,
                    sent_action,
                    sample_offset + samples - 1,
                )

        if now - status_started >= 1.0:
            free_bytes = shutil.disk_usage(dataset.root).free
            if record and free_bytes < minimum_free_bytes:
                raise RuntimeError(
                    "Free disk space fell below the 5 GiB collection reserve; "
                    "the attempt was stopped and will not enter training"
                )
            actual_hz = status_loops / max(now - status_started, 1e-6)
            phase = "record" if record else "reset"
            print(
                f"CONTROL {phase} actual_hz={actual_hz:.1f} samples={samples} "
                f"loop_ms={(now - loop_started) * 1000:.1f}",
                flush=True,
            )
            status_started = now
            status_loops = 0

        next_control += control_period
        if next_control < now - control_period:
            next_control = now
        precise_sleep(max(next_control - time.perf_counter(), 0.0))

    elapsed = time.perf_counter() - started
    return samples, loops / max(elapsed, 1e-6)


def main() -> int:
    args = parse_args()
    init_logging()
    profile = verify_profile_lock(args)
    contract = verify_collection_contract(args)
    require_resume_profile(args, contract)
    events = install_signal_controls()
    robot, leader = make_hardware(args, profile)
    action_processor, _robot_action_processor, observation_processor = make_default_processors()
    dataset: LeRobotDataset | None = None

    try:
        # Keep the active teleop's known cleanup order: leader connects first; the
        # follower disconnects (and releases torque) before the leader closes.
        leader.connect(calibrate=False)
        robot.connect(calibrate=False)
        if not leader.is_calibrated or not robot.is_calibrated:
            raise RuntimeError("Locked calibration did not load; refusing to collect")
        session_home = capture_session_home(robot, leader)
        print(
            "SESSION home_captured joints=7 "
            f"return_speed={session_home['return_speed_deg_s']:g}deg_s",
            flush=True,
        )
        dataset = make_dataset(args, robot, action_processor, observation_processor)
        write_profile_sidecar(args, profile, contract)
        # Opening Rerun now means hardware, calibration, and dataset creation all
        # succeeded.  Every take still gets an RRD even if the live viewer cannot
        # open, so review data is never coupled to the GUI process.
        viewer_available = start_rerun_viewer(not args.no_rerun)
        print(
            f"SESSION connected control_hz={args.control_hz} dataset_fps={args.dataset_fps} "
            f"existing_episodes={dataset.num_episodes}",
            flush=True,
        )

        completed_this_run = 0
        attempt_number = 0
        with RecoverySafeVideoEncodingManager(dataset) as encoding_manager:
            while completed_this_run < args.episodes and not events["stop"]:
                episode_index = dataset.num_episodes
                quarantine = quarantine_stale_candidate_frames(
                    dataset,
                    args.attempt_root,
                    args.dataset_root.name,
                    episode_index,
                )
                if quarantine is not None:
                    print(
                        "RECOVERY stale_candidate_frames_quarantined "
                        f"id={quarantine['quarantine_id']} episode={episode_index} "
                        f"cameras={len(quarantine['camera_directories'])}",
                        flush=True,
                    )
                attempt_number += 1
                events["finish"] = False
                events["rerecord"] = False
                directory, metadata, attempt_recording = begin_attempt(
                    args,
                    profile,
                    contract,
                    candidate_episode_index=episode_index,
                    attempt_number=attempt_number,
                    viewer_available=viewer_available,
                    session_home=session_home,
                )
                # Clear the prior take's decision before announcing this one as
                # recordable.  The backend writes a new decision atomically
                # before signalling, which lets Stop/shutdown honor a failure
                # label even if POSIX signals are delivered in the other order.
                atomic_write_json(
                    args.control_file,
                    {
                        "action": None,
                        "attempt_id": metadata["attempt_id"],
                        "created_at": utc_now(),
                    },
                )
                rerun_writer = AttemptRerunWriter(attempt_recording, args.dataset_fps)
                camera_freshness: dict[str, Any] = {}
                metadata["camera_freshness"] = camera_freshness
                attempt_started = time.perf_counter()
                samples = 0
                measured_rates: list[float] = []
                dataset_save_completed = False
                print(
                    f"ATTEMPT recording id={metadata['attempt_id']} index={episode_index} "
                    f"success={completed_this_run + 1}/{args.episodes}",
                    flush=True,
                )
                try:
                    segment_samples, actual_hz = control_segment(
                        args=args,
                        robot=robot,
                        leader=leader,
                        dataset=dataset,
                        events=events,
                        duration_s=args.episode_time_s,
                        record=True,
                        task=args.task,
                        rerun_writer=rerun_writer,
                        sample_offset=samples,
                        camera_freshness=camera_freshness,
                    )
                    samples += segment_samples
                    measured_rates.append(actual_hz)
                    duration_s = time.perf_counter() - attempt_started
                    mean_hz = sum(measured_rates) / max(len(measured_rates), 1)

                    if not events["finish"] and not events["stop"]:
                        print(
                            f"AWAITING_DECISION id={metadata['attempt_id']} nominal_seconds="
                            f"{args.episode_time_s:g} recording_continues=false "
                            "motion_paused=true",
                            flush=True,
                        )
                    # A browser can disappear or be left unattended.  Never
                    # spool unbounded RGB PNGs while waiting for a decision;
                    # hold the last follower pose and await Keep / Failed / Stop.
                    while not events["finish"] and not events["stop"]:
                        time.sleep(0.05)
                    try:
                        published_decision = json.loads(args.control_file.read_text())
                    except (OSError, json.JSONDecodeError):
                        published_decision = {}
                    published_action = (
                        published_decision.get("action")
                        if isinstance(published_decision, dict)
                        else None
                    )
                    resolved_disposition = resolve_attempt_disposition(
                        events,
                        published_action,
                    )

                    if resolved_disposition == "aborted":
                        finalize_attempt_archive(
                            dataset=dataset,
                            directory=directory,
                            metadata=metadata,
                            recording=attempt_recording,
                            rerun_writer=rerun_writer,
                            disposition="aborted",
                            samples=samples,
                            duration_s=duration_s,
                            actual_hz=mean_hz,
                        )
                        clear_unsaved_attempt(dataset)
                        print(
                            f"ATTEMPT archived_aborted id={metadata['attempt_id']} samples={samples}",
                            flush=True,
                        )
                        break

                    if resolved_disposition == "failed":
                        decision = read_control_decision(args, "rerecord")
                        failure_label, failure_note = validate_failure_label(
                            decision.get("failure_label"), decision.get("failure_note")
                        )
                        finalize_attempt_archive(
                            dataset=dataset,
                            directory=directory,
                            metadata=metadata,
                            recording=attempt_recording,
                            rerun_writer=rerun_writer,
                            disposition="failed",
                            samples=samples,
                            duration_s=duration_s,
                            actual_hz=mean_hz,
                            failure_label=failure_label,
                            failure_note=failure_note,
                        )
                        clear_unsaved_attempt(dataset)
                        print(
                            f"ATTEMPT archived_failed id={metadata['attempt_id']} "
                            f"label={failure_label} training_included=false",
                            flush=True,
                        )
                        if events["stop"]:
                            print("SESSION stop_after_failed_attempt", flush=True)
                            break
                        events["finish"] = False
                        events["rerecord"] = False
                        home_result = automatic_reset_to_session_home(
                            args=args,
                            robot=robot,
                            leader=leader,
                            events=events,
                            session_home=session_home,
                        )
                        if events["stop"]:
                            print("SESSION stop_requested_during_reset", flush=True)
                            break
                        if home_result is not None:
                            mark_attempt_home_return(directory, home_result)
                        continue

                    keep_decision = read_control_decision(
                        args,
                        "finish_and_stop" if published_action == "finish_and_stop" else "finish",
                    )
                    stop_after_save = keep_decision.get("action") == "finish_and_stop"
                    minimum_samples = max(2, int(args.dataset_fps * 1.0))
                    if samples < minimum_samples:
                        finalize_attempt_archive(
                            dataset=dataset,
                            directory=directory,
                            metadata=metadata,
                            recording=attempt_recording,
                            rerun_writer=rerun_writer,
                            disposition="collector_error",
                            samples=samples,
                            duration_s=duration_s,
                            actual_hz=mean_hz,
                            archive_error=(
                                f"Attempt captured only {samples} samples; minimum is {minimum_samples}"
                            ),
                        )
                        clear_unsaved_attempt(dataset)
                        raise RuntimeError(
                            f"Attempt {metadata['attempt_id']} captured only {samples} samples; "
                            "archived but excluded from training"
                        )

                    print(
                        f"ATTEMPT save_started id={metadata['attempt_id']} "
                        "phase=rerun_and_attempt_videos",
                        flush=True,
                    )
                    finalize_attempt_archive(
                        dataset=dataset,
                        directory=directory,
                        metadata=metadata,
                        recording=attempt_recording,
                        rerun_writer=rerun_writer,
                        disposition="kept",
                        samples=samples,
                        duration_s=duration_s,
                        actual_hz=mean_hz,
                    )
                    print(
                        f"ATTEMPT save_phase id={metadata['attempt_id']} phase=lerobot",
                        flush=True,
                    )
                    with reuse_attempt_videos_for_lerobot(dataset, directory):
                        dataset.save_episode(parallel_encoding=False)
                    dataset = checkpoint_and_reopen_dataset(
                        dataset,
                        expected_episode_index=episode_index,
                    )
                    encoding_manager.dataset = dataset
                    dataset_save_completed = True
                    mark_attempt_in_training(directory, episode_index)
                    completed_this_run += 1
                    print(
                        f"ATTEMPT lerobot_durable id={metadata['attempt_id']} "
                        f"episode={episode_index} total_episodes={dataset.num_episodes}",
                        flush=True,
                    )
                    print(
                        f"ATTEMPT archived_kept id={metadata['attempt_id']} "
                        f"episode={episode_index} samples={samples} control_hz={mean_hz:.1f} "
                        "training_included=true",
                        flush=True,
                    )
                    if stop_after_save:
                        print("SESSION finish_and_stop_after_durable_save", flush=True)
                        break
                    if events["stop"]:
                        print("SESSION stop_after_kept_attempt", flush=True)
                        break
                except KeyboardInterrupt:
                    if metadata.get("disposition") == "recording":
                        finalize_attempt_archive(
                            dataset=dataset,
                            directory=directory,
                            metadata=metadata,
                            recording=attempt_recording,
                            rerun_writer=rerun_writer,
                            disposition="aborted",
                            samples=samples,
                            duration_s=time.perf_counter() - attempt_started,
                            actual_hz=sum(measured_rates) / max(len(measured_rates), 1),
                        )
                    if dataset.episode_buffer and dataset.episode_buffer.get("size", 0) > 0:
                        release_unsaved_attempt_after_archive(dataset, directory, metadata)
                    raise
                except Exception as exc:
                    if metadata.get("disposition") == "recording":
                        try:
                            finalize_attempt_archive(
                                dataset=dataset,
                                directory=directory,
                                metadata=metadata,
                                recording=attempt_recording,
                                rerun_writer=rerun_writer,
                                disposition="collector_error",
                                samples=samples,
                                duration_s=time.perf_counter() - attempt_started,
                                actual_hz=sum(measured_rates) / max(len(measured_rates), 1),
                                archive_error=str(exc),
                            )
                        except Exception as archive_exc:
                            logging.error("Could not finalize failed attempt archive: %s", archive_exc)
                    elif dataset_save_completed:
                        # save_episode already committed this take to LeRobot.
                        # Never claim exclusion merely because sidecar sync failed.
                        metadata["disposition"] = "kept"
                        metadata["training_included"] = True
                        metadata["training_episode_index"] = episode_index
                        metadata["training_saved_at"] = utc_now()
                        metadata["metadata_sync_error"] = str(exc)
                        try:
                            atomic_write_json(directory / "metadata.json", metadata)
                        except OSError:
                            logging.exception(
                                "LeRobot episode was saved, but attempt metadata could not be reconciled"
                            )
                    elif (
                        metadata.get("disposition") == "kept"
                        and metadata.get("archive_complete")
                        and not metadata.get("training_included")
                    ):
                        try:
                            mark_training_save_failure(
                                directory,
                                metadata,
                                dataset.root,
                                episode_index,
                                exc,
                            )
                        except OSError:
                            logging.exception("Could not record training-save failure in metadata")
                    if dataset.episode_buffer and dataset.episode_buffer.get("size", 0) > 0:
                        release_unsaved_attempt_after_archive(dataset, directory, metadata)
                    raise

                events["finish"] = False
                home_result = automatic_reset_to_session_home(
                    args=args,
                    robot=robot,
                    leader=leader,
                    events=events,
                    session_home=session_home,
                )
                if events["stop"]:
                    print("SESSION stop_requested_during_reset", flush=True)
                    break
                if home_result is not None:
                    mark_attempt_home_return(directory, home_result)
                if completed_this_run >= args.episodes:
                    break

        print(
            f"SESSION complete added_episodes={completed_this_run} total_episodes={dataset.num_episodes}",
            flush=True,
        )
        return 0
    except KeyboardInterrupt:
        print("SESSION interrupted", flush=True)
        return 130
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        finally:
            if leader.is_connected:
                leader.disconnect()
        logging.info("ReBot collection cleanup complete")


if __name__ == "__main__":
    sys.exit(main())
