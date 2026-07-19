"""Isolated ReBot data-collection and training-workspace backend.

This module deliberately uses the already-hardened ReBot runtime beside the
Operator Kit. It does not import or execute DeskPartner's older vendored motor
drivers, and it never writes datasets or checkpoints into the Git repository.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable
from uuid import uuid4
from zoneinfo import ZoneInfo

from attempt_archive import (
    FAILURE_LABELS,
    artifact_path as archived_artifact_path,
    atomic_write_json,
    attempt_inventory as read_attempt_inventory,
    find_attempt,
    mark_kept_attempt_failed as write_reviewed_failure,
    reindex_training_episode as write_reindexed_episode,
    review_revision,
    update_failure_label as write_failure_label,
    utc_now,
    validate_failure_label,
)


KIT_ROOT = Path(__file__).resolve().parent.parent
GUI_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = Path(
    os.environ.get("RUNTIME_ROOT", KIT_ROOT.parent / "rebot_setup" / "vendor" / "rebot_lerobot")
)
VENV = Path(os.environ.get("VENV", RUNTIME_ROOT / ".venv"))
PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", VENV / "bin" / "python"))
RECORD_BIN = Path(os.environ.get("RECORD_BIN", VENV / "bin" / "lerobot-record"))
TRAIN_BIN = Path(os.environ.get("TRAIN_BIN", VENV / "bin" / "lerobot-train"))
HF_LEROBOT_HOME = Path(os.environ.get("HF_LEROBOT_HOME", RUNTIME_ROOT / "lerobot-home"))
DATA_ROOT = Path(os.environ.get("KIT_DATA_ROOT", KIT_ROOT / "data"))
MODEL_ROOT = Path(os.environ.get("KIT_MODEL_ROOT", KIT_ROOT / "models"))
CAMERA_ROOT = Path(os.environ.get("KIT_CAMERA_ROOT", KIT_ROOT / "camera-check"))
RUN_ROOT = Path(os.environ.get("KIT_RUN_ROOT", KIT_ROOT / "training-runs"))
ATTEMPT_ROOT = RUN_ROOT / "attempts"
DATASET_REVISION_ROOT = RUN_ROOT / "dataset-revisions"
CONTROL_ROOT = RUN_ROOT / "controls"
CONTROLLED_RECORD = GUI_ROOT / "controlled_record.py"
OWNED_PROCESS = GUI_ROOT / "owned_process.py"
CAMERA_CHECK = KIT_ROOT / "dual_camera_check.py"
DATASET_VALIDATOR = KIT_ROOT / "validate_dataset.py"
HACKATHON_ROOT = KIT_ROOT.parent / "Embodied_Metal_Hackathon" / "so100-hackathon"
NEWT_BIN = HACKATHON_ROOT / ".pixi" / "envs" / "default" / "bin" / "newt"
PROFILE_PATH = KIT_ROOT / "config" / "training_profile.json"


def _venv_site_packages() -> Path:
    matches = sorted((VENV / "lib").glob("python*/site-packages"))
    return matches[-1] if matches else VENV / "lib" / "python3.10" / "site-packages"


SITE_PACKAGES = _venv_site_packages()
RERUN_NATIVE_BIN = SITE_PACKAGES / "rerun_sdk" / "rerun_cli" / "rerun"

DATASET_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,47}$")
PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,95}$")
EXPECTED_JOINT_FEATURES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_yaw.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
COLLECTION_CONTRACT_KEYS = (
    "dataset",
    "repo_id",
    "task",
    "episode_time_s",
    "reset_time_s",
    "fps",
    "control_hz",
    "front_camera",
    "front_width",
    "front_height",
    "side_camera",
    "side_width",
    "side_height",
    "excluded_camera",
    "max_step",
    "motor_velocity",
    "gripper_force",
)


class TrainingConfigError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _runtime_profile_file(entry: Any, label: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise TrainingConfigError(f"Training profile {label} entry must be an object")
    relative = Path(str(entry.get("runtime_relative_path", "")))
    expected = str(entry.get("sha256", "")).lower()
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or not re.fullmatch(r"[0-9a-f]{64}", expected)
    ):
        raise TrainingConfigError(f"Training profile {label} path or SHA-256 is invalid")
    root = RUNTIME_ROOT.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TrainingConfigError(
            f"Training profile {label} path leaves the isolated runtime"
        ) from exc
    exists = path.is_file()
    actual = _sha256(path) if exists else None
    return {
        "path": str(path),
        "runtime_relative_path": relative.as_posix(),
        "exists": exists,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "matches": bool(exists and actual == expected),
    }


def training_profile() -> dict[str, Any]:
    """Load and verify the single source of truth for collection/training.

    This is read-only. It fingerprints both motor-zero calibration files and
    every runtime file that defines the leader/follower coordinate conversion
    without opening either arm or camera.
    """

    try:
        profile = json.loads(PROFILE_PATH.read_text())
    except FileNotFoundError as exc:
        raise TrainingConfigError(f"Training profile is missing: {PROFILE_PATH}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigError(f"Training profile cannot be read: {exc}") from exc
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise TrainingConfigError("Training profile schema_version must be 1")
    profile_id = str(profile.get("profile_id", ""))
    if not PROFILE_ID_RE.fullmatch(profile_id):
        raise TrainingConfigError("Training profile_id is invalid")
    profile_version = profile.get("profile_version")
    if isinstance(profile_version, bool) or not isinstance(profile_version, int) or profile_version < 1:
        raise TrainingConfigError("Training profile_version must be a positive integer")

    calibration = profile.get("calibration")
    hardware = profile.get("hardware_identity")
    coordinates = profile.get("coordinate_contract")
    collection = profile.get("collection_defaults")
    cameras = profile.get("camera_defaults")
    training = profile.get("training_defaults")
    if not all(isinstance(item, dict) for item in (calibration, hardware, coordinates, collection, cameras, training)):
        raise TrainingConfigError("Training profile is missing a required object")
    follower_identity = calibration.get("follower")
    leader_identity = calibration.get("leader")
    if (
        not isinstance(follower_identity, dict)
        or not isinstance(leader_identity, dict)
        or follower_identity.get("type") != "seeed_b601_dm_follower"
        or follower_identity.get("id") != "follower1"
        or leader_identity.get("type") != "rebot_arm_102_leader"
        or leader_identity.get("id") != "rebot_arm_102_leader"
    ):
        raise TrainingConfigError("Training profile arm type/ID contract is invalid")
    for label in ("follower_usb", "leader_usb"):
        identity = hardware.get(label)
        if not isinstance(identity, dict) or any(
            isinstance(identity.get(key), bool)
            or not isinstance(identity.get(key), int)
            or not 0 < identity[key] <= 0xFFFF
            for key in ("vid", "pid")
        ):
            raise TrainingConfigError(f"Training profile {label} identity is invalid")

    joints = coordinates.get("joints")
    if not isinstance(joints, list) or [joint.get("feature") for joint in joints if isinstance(joint, dict)] != EXPECTED_JOINT_FEATURES:
        raise TrainingConfigError("Training profile joint order must match the seven ReBot features")
    if coordinates.get("action_dimension") != len(EXPECTED_JOINT_FEATURES):
        raise TrainingConfigError("Training profile action_dimension must be 7")
    for joint in joints:
        scale = joint.get("leader_to_follower_scale")
        limits = joint.get("soft_limit_degrees")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) == 0
            or not isinstance(limits, list)
            or len(limits) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in limits)
            or not all(math.isfinite(float(value)) for value in limits)
            or float(limits[0]) >= float(limits[1])
        ):
            raise TrainingConfigError(f"Training profile joint contract is invalid for {joint.get('feature')}")

    front = cameras.get("front")
    side = cameras.get("side")
    if not isinstance(front, dict) or not isinstance(side, dict):
        raise TrainingConfigError("Training profile must define front and side cameras")
    if front.get("recording_key") != "observation.images.front" or side.get("recording_key") != "observation.images.side":
        raise TrainingConfigError("Training profile camera semantic keys are invalid")
    if front.get("fps") != collection.get("fps") or side.get("fps") != collection.get("fps"):
        raise TrainingConfigError("Training profile camera FPS must equal collection FPS")
    if training.get("action_dimension") != len(EXPECTED_JOINT_FEATURES):
        raise TrainingConfigError("Training profile policy action_dimension must be 7")
    if training.get("image_order") != [front["recording_key"], side["recording_key"]]:
        raise TrainingConfigError("Training profile image_order must be front then side")
    if (
        training.get("state_normalization") != "quantile"
        or training.get("action_normalization") != "quantile"
        or training.get("normalize_gripper") is not True
    ):
        raise TrainingConfigError(
            "Training profile must lock quantile state/action normalization and normalized gripper"
        )

    files = {
        "follower": _runtime_profile_file(calibration.get("follower"), "follower calibration"),
        "leader": _runtime_profile_file(calibration.get("leader"), "leader calibration"),
        "follower_driver_contract": _runtime_profile_file(
            calibration.get("follower_driver_contract"), "follower driver contract"
        ),
        "follower_base_implementation": _runtime_profile_file(
            calibration.get("follower_base_implementation"), "follower base implementation"
        ),
        "follower_dm_implementation": _runtime_profile_file(
            calibration.get("follower_dm_implementation"), "follower DM implementation"
        ),
        "leader_driver_contract": _runtime_profile_file(
            calibration.get("leader_driver_contract"), "leader driver contract"
        ),
        "leader_implementation": _runtime_profile_file(
            calibration.get("leader_implementation"), "leader implementation"
        ),
    }
    passed = all(item["matches"] for item in files.values())
    defaults = {
        **collection,
        "front_camera": front.get("index"),
        "front_width": front.get("width"),
        "front_height": front.get("height"),
        "side_camera": side.get("index"),
        "side_width": side.get("width"),
        "side_height": side.get("height"),
        "excluded_camera": cameras.get("excluded_screen_index"),
    }
    return {
        "passed": passed,
        "profile_id": profile_id,
        "profile_version": profile_version,
        "profile_digest": _canonical_digest(profile),
        "profile_path": str(PROFILE_PATH),
        "defaults": defaults,
        "calibration_files": files,
        "profile": profile,
        "error": None if passed else "A calibration or driver fingerprint changed; create a reviewed profile version before collecting or training.",
    }


def training_profile_status() -> dict[str, Any]:
    try:
        return training_profile()
    except TrainingConfigError as exc:
        return {
            "passed": False,
            "profile_path": str(PROFILE_PATH),
            "error": str(exc),
            "calibration_files": {},
        }


def require_training_profile() -> dict[str, Any]:
    result = training_profile()
    if not result["passed"]:
        mismatches = [
            name for name, status in result["calibration_files"].items() if not status["matches"]
        ]
        raise TrainingConfigError(
            "Training profile fingerprint mismatch: " + ", ".join(mismatches)
        )
    return result


def _number(
    value: Any,
    name: str,
    minimum: float,
    maximum: float,
    *,
    whole: bool = False,
) -> float | int:
    if isinstance(value, bool):
        raise TrainingConfigError(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingConfigError(f"{name} must be a number") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise TrainingConfigError(
            f"{name} must be between {minimum:g} and {maximum:g}"
        )
    if whole:
        if not result.is_integer():
            raise TrainingConfigError(f"{name} must be a whole number")
        return int(result)
    return result


def validate_collection_config(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TrainingConfigError("Request body must be a JSON object")

    profile_status = require_training_profile()
    defaults = profile_status["defaults"]
    supplied_digest = str(payload.get("profile_digest", "")).strip()
    if supplied_digest and supplied_digest != profile_status["profile_digest"]:
        raise TrainingConfigError(
            "The training profile changed after this page loaded; refresh and restore verified defaults"
        )

    slug = str(payload.get("dataset", defaults["dataset"])).strip().lower()
    if not DATASET_SLUG_RE.fullmatch(slug):
        raise TrainingConfigError(
            "Dataset name must be 3-48 lowercase letters, numbers, hyphens, or underscores"
        )
    task = " ".join(str(payload.get("task", defaults["task"])).strip().split())
    if len(task) < 12 or len(task) > 220:
        raise TrainingConfigError(
            "Task instruction must be a clear 12-220 character imperative sentence"
        )

    front = _number(
        payload.get("front_camera", defaults["front_camera"]),
        "Logitech overhead camera",
        0,
        32,
        whole=True,
    )
    side = _number(
        payload.get("side_camera", defaults["side_camera"]),
        "Innomaker wrist camera",
        0,
        32,
        whole=True,
    )
    excluded = _number(
        payload.get("excluded_camera", defaults["excluded_camera"]),
        "Excluded Mac camera",
        0,
        32,
        whole=True,
    )
    if front == side:
        raise TrainingConfigError("Overhead and wrist cameras must use different indices")
    if excluded in {front, side}:
        raise TrainingConfigError(
            "The Mac camera cannot also be a recording camera"
        )

    config = {
        "dataset": slug,
        "repo_id": f"local/{slug}",
        "task": task,
        "episodes": _number(payload.get("episodes", defaults["episodes"]), "Episodes this run", 1, 500, whole=True),
        "episode_time_s": _number(payload.get("episode_time_s", defaults["episode_time_s"]), "Episode time", 5, 3600),
        "reset_time_s": _number(payload.get("reset_time_s", defaults["reset_time_s"]), "Reset time", 3, 180),
        "fps": _number(payload.get("fps", defaults["fps"]), "Dataset FPS", 5, 60, whole=True),
        "control_hz": _number(payload.get("control_hz", defaults["control_hz"]), "Control Hz", 30, 240, whole=True),
        "front_width": _number(payload.get("front_width", defaults["front_width"]), "Front width", 320, 1920, whole=True),
        "front_height": _number(payload.get("front_height", defaults["front_height"]), "Front height", 240, 1080, whole=True),
        "side_width": _number(payload.get("side_width", defaults["side_width"]), "Side width", 320, 1920, whole=True),
        "side_height": _number(payload.get("side_height", defaults["side_height"]), "Side height", 240, 1080, whole=True),
        "front_camera": front,
        "side_camera": side,
        "excluded_camera": excluded,
        "resume": bool(payload.get("resume", defaults["resume"])),
        "max_step": _number(payload.get("max_step", defaults["max_step"]), "Maximum step", 0.01, 45),
        "motor_velocity": _number(payload.get("motor_velocity", defaults["motor_velocity"]), "Motor velocity", 0.1, 2000),
        "gripper_force": _number(payload.get("gripper_force", defaults["gripper_force"]), "Gripper force", 0, 1),
        "training_profile_id": profile_status["profile_id"],
        "training_profile_version": profile_status["profile_version"],
        "training_profile_digest": profile_status["profile_digest"],
    }
    if config["control_hz"] < config["fps"]:
        raise TrainingConfigError("Control Hz must be at least the dataset FPS")
    return config


def collection_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable semantic contract shared by all dataset episodes.

    ``episodes`` and ``resume`` are intentionally excluded: operators may add a
    new batch of episodes, but may not silently change the task, clocks, camera
    identity/shape, or actuator-control settings inside one dataset.
    """

    missing = [key for key in COLLECTION_CONTRACT_KEYS if key not in config]
    if missing:
        raise TrainingConfigError(
            "Collection configuration is missing contract fields: " + ", ".join(missing)
        )
    return {key: config[key] for key in COLLECTION_CONTRACT_KEYS}


def collection_contract_lock(config: dict[str, Any]) -> dict[str, Any]:
    snapshot = collection_contract(config)
    return {"digest": _canonical_digest(snapshot), "snapshot": snapshot}


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    hf_home = KIT_ROOT / ".state" / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    paths = [
        RUNTIME_ROOT / "lerobot" / "src",
        RUNTIME_ROOT / "lerobot-robot-seeed-b601",
        RUNTIME_ROOT / "lerobot-teleoperator-rebot-arm-102",
        SITE_PACKAGES / "rerun_sdk",
    ]
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in paths] + ([inherited] if inherited else [])
    )
    env["HF_LEROBOT_HOME"] = str(HF_LEROBOT_HOME)
    # Keep every dataset-loader cache inside the isolated Operator Kit instead
    # of modifying the user's shared ~/.cache/huggingface state.
    env["HF_HOME"] = str(hf_home)
    env["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    env["PATH"] = str(VENV / "bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def build_record_command(
    config: dict[str, Any],
    ports: dict[str, str],
    *,
    attempt_root: Path | None = None,
    control_file: Path | None = None,
) -> list[str]:
    profile_status = require_training_profile()
    calibration_files = profile_status["calibration_files"]
    contract = collection_contract_lock(config)
    root = DATA_ROOT / config["dataset"]
    attempt_root = attempt_root or ATTEMPT_ROOT
    control_file = control_file or CONTROL_ROOT / "record-control.json"
    command = [
        str(PYTHON_BIN),
        "-u",
        str(CONTROLLED_RECORD),
        "--follower-port",
        ports["follower"],
        "--leader-port",
        ports["leader"],
        "--repo-id",
        config["repo_id"],
        "--dataset-root",
        str(root),
        "--profile-path",
        profile_status["profile_path"],
        "--profile-digest",
        profile_status["profile_digest"],
        "--collection-contract-json",
        json.dumps(contract["snapshot"], sort_keys=True, separators=(",", ":")),
        "--collection-contract-digest",
        contract["digest"],
        "--follower-calibration",
        calibration_files["follower"]["path"],
        "--leader-calibration",
        calibration_files["leader"]["path"],
        "--follower-driver-contract",
        calibration_files["follower_driver_contract"]["path"],
        "--follower-base-implementation",
        calibration_files["follower_base_implementation"]["path"],
        "--follower-dm-implementation",
        calibration_files["follower_dm_implementation"]["path"],
        "--leader-driver-contract",
        calibration_files["leader_driver_contract"]["path"],
        "--leader-implementation",
        calibration_files["leader_implementation"]["path"],
        "--task",
        config["task"],
        "--episodes",
        str(config["episodes"]),
        "--episode-time-s",
        f"{config['episode_time_s']:g}",
        "--reset-time-s",
        f"{config['reset_time_s']:g}",
        "--control-hz",
        str(config["control_hz"]),
        "--dataset-fps",
        str(config["fps"]),
        "--front-camera",
        str(config["front_camera"]),
        "--front-width",
        str(config["front_width"]),
        "--front-height",
        str(config["front_height"]),
        "--side-camera",
        str(config["side_camera"]),
        "--side-width",
        str(config["side_width"]),
        "--side-height",
        str(config["side_height"]),
        "--excluded-camera",
        str(config["excluded_camera"]),
        "--max-step",
        f"{config['max_step']:g}",
        "--motor-velocity",
        f"{config['motor_velocity']:g}",
        "--gripper-force",
        f"{config['gripper_force']:g}",
        "--attempt-root",
        str(attempt_root),
        "--control-file",
        str(control_file),
    ]
    if config["resume"]:
        command.append("--resume")
    return command


def _collection_manifests(dataset_name: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if not RUN_ROOT.is_dir():
        return {"records": records, "errors": errors, "candidate_count": 0}
    paths = [
        path
        for path in sorted(RUN_ROOT.glob(f"{dataset_name}--*.json"))
        if not path.name.endswith("--validation.json")
    ]
    for path in paths:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue
        if not isinstance(value, dict):
            errors.append({"path": str(path), "error": "manifest root is not an object"})
            continue
        records.append({"path": str(path), **value})
    return {"records": records, "errors": errors, "candidate_count": len(paths)}


def _zero_frame_dataset_counts(root: Path) -> tuple[int, int] | None:
    """Return (episodes, frames) only when a dataset is safely retryable.

    Missing/empty roots and metadata that explicitly reports zero saved data
    are retryable. Unknown metadata or any saved parquet/video output is kept
    in place for manual recovery instead of being moved automatically.
    """

    if not root.exists():
        return (0, 0)
    if not root.is_dir() or root.is_symlink():
        return None
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        try:
            if any(
                path.is_file() and path.suffix.lower() in {".parquet", ".mp4"}
                for path in root.rglob("*")
            ):
                return None
        except OSError:
            return None
        return (0, 0)
    try:
        info = json.loads(info_path.read_text())
        episodes = info.get("total_episodes", 0)
        frames = info.get("total_frames", 0)
        if (
            not isinstance(info, dict)
            or isinstance(episodes, bool)
            or isinstance(frames, bool)
            or not isinstance(episodes, (int, float))
            or not isinstance(frames, (int, float))
            or not float(episodes).is_integer()
            or not float(frames).is_integer()
            or episodes < 0
            or frames < 0
        ):
            return None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return int(episodes), int(frames)


def _manifest_is_incomplete(path: Path) -> bool:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    lifecycle = manifest.get("lifecycle") if isinstance(manifest, dict) else None
    return not (
        isinstance(lifecycle, dict)
        and lifecycle.get("state") == "COMPLETE"
        and lifecycle.get("exit_code") == 0
    )


def _quarantine_zero_frame_attempt(
    dataset_name: str,
    *,
    manifest_paths: list[Path] | None = None,
    reason: str,
) -> Path | None:
    """Move a zero-frame attempt out of the active namespace, never delete it."""

    if not DATASET_SLUG_RE.fullmatch(dataset_name):
        return None
    root = DATA_ROOT / dataset_name
    counts = _zero_frame_dataset_counts(root)
    if counts is None or counts != (0, 0):
        return None

    explicit = {path.resolve() for path in (manifest_paths or []) if path.is_file()}
    candidates: set[Path] = set(explicit)
    for path in RUN_ROOT.glob(f"{dataset_name}--*.json") if RUN_ROOT.is_dir() else []:
        if path.name.endswith("--validation.json"):
            continue
        resolved = path.resolve()
        if root.exists() or resolved in explicit or _manifest_is_incomplete(path):
            candidates.add(resolved)
    if not root.exists() and not candidates:
        return None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination = RUN_ROOT / "quarantine" / f"{stamp}-{dataset_name}-zero-frame"
    data_destination = destination / "data"
    manifest_destination = destination / "manifests"
    data_destination.mkdir(parents=True, exist_ok=False)
    manifest_destination.mkdir(parents=True, exist_ok=False)

    moved_root: str | None = None
    if root.exists():
        target = data_destination / dataset_name
        shutil.move(str(root), str(target))
        moved_root = str(target)
    moved_manifests: list[str] = []
    for path in sorted(candidates):
        if not path.is_file():
            continue
        target = manifest_destination / path.name
        shutil.move(str(path), str(target))
        moved_manifests.append(str(target))

    audit = {
        "quarantined_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "reason": reason,
        "saved_episodes": counts[0],
        "saved_frames": counts[1],
        "moved_dataset_root": moved_root,
        "moved_manifests": moved_manifests,
        "recoverable": True,
    }
    (destination / "quarantine.json").write_text(json.dumps(audit, indent=2) + "\n")
    return destination


def _manifest_profile_compatibility(
    dataset_name: str,
    profile_digest: str,
    collection_digest: str | None = None,
) -> dict[str, Any]:
    result = _collection_manifests(dataset_name)
    manifests = result["records"]
    invalid = len(result["errors"])
    profile_digests: set[str] = set()
    contract_digests: set[str] = set()
    lifecycle_states: set[str] = set()
    for manifest in manifests:
        profile_lock = manifest.get("training_profile")
        contract_lock = manifest.get("collection_contract")
        lifecycle = manifest.get("lifecycle")
        if not isinstance(profile_lock, dict) or not isinstance(profile_lock.get("snapshot"), dict):
            invalid += 1
            continue
        declared_profile = str(profile_lock.get("digest", ""))
        calculated_profile = _canonical_digest(profile_lock["snapshot"])
        profile_digests.add(declared_profile)
        if not declared_profile or calculated_profile != declared_profile:
            invalid += 1
        if not isinstance(contract_lock, dict) or not isinstance(contract_lock.get("snapshot"), dict):
            invalid += 1
        else:
            declared_contract = str(contract_lock.get("digest", ""))
            calculated_contract = _canonical_digest(contract_lock["snapshot"])
            contract_digests.add(declared_contract)
            if not declared_contract or calculated_contract != declared_contract:
                invalid += 1
        lifecycle_exit = lifecycle.get("exit_code") if isinstance(lifecycle, dict) else None
        state = str(lifecycle.get("state", "")) if isinstance(lifecycle, dict) else ""
        lifecycle_states.add(state)
        durable_partial = (
            state == "PARTIAL_COMPLETE"
            and isinstance(lifecycle_exit, int)
            and not isinstance(lifecycle_exit, bool)
            and lifecycle_exit != 0
            and isinstance(lifecycle.get("durable_episodes_this_run"), int)
            and not isinstance(lifecycle.get("durable_episodes_this_run"), bool)
            and lifecycle["durable_episodes_this_run"] > 0
        ) if isinstance(lifecycle, dict) else False
        if not ((state == "COMPLETE" and lifecycle_exit == 0) or durable_partial):
            invalid += 1
    compatible = bool(
        result["candidate_count"]
        and invalid == 0
        and profile_digests == {profile_digest}
        and (collection_digest is None or contract_digests == {collection_digest})
    )
    return {
        "compatible": compatible,
        "manifest_count": result["candidate_count"],
        "valid_manifest_count": len(manifests),
        "invalid_manifest_count": invalid,
        "manifest_errors": result["errors"],
        "profile_digests": sorted(profile_digests),
        "collection_contract_digests": sorted(contract_digests),
        "lifecycle_states": sorted(lifecycle_states),
    }


def _dataset_profile_sidecar(
    root: Path,
    profile_digest: str | None,
    collection_digest: str | None = None,
) -> dict[str, Any]:
    path = root / "meta" / "rebot_training_profile.json"
    result: dict[str, Any] = {
        "path": str(path),
        "compatible": False,
        "profile_digest": None,
        "collection_contract_digest": None,
        "error": None,
    }
    if not path.is_file():
        result["error"] = "profile sidecar is missing"
        return result
    try:
        sidecar = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"profile sidecar is unreadable: {exc}"
        return result
    if not isinstance(sidecar, dict):
        result["error"] = "profile sidecar root is not an object"
        return result
    snapshot = sidecar.get("profile_snapshot")
    contract_snapshot = sidecar.get("collection_contract")
    declared_profile = str(sidecar.get("training_profile_digest") or "") or None
    declared_contract = str(sidecar.get("collection_contract_digest") or "") or None
    result["profile_digest"] = declared_profile
    result["collection_contract_digest"] = declared_contract
    if not isinstance(snapshot, dict) or not isinstance(contract_snapshot, dict):
        result["error"] = "profile or collection-contract snapshot is missing"
        return result
    calculated_profile = _canonical_digest(snapshot)
    calculated_contract = _canonical_digest(contract_snapshot)
    if declared_profile != calculated_profile:
        result["error"] = "profile snapshot digest is invalid"
        return result
    if declared_contract != calculated_contract:
        result["error"] = "collection-contract snapshot digest is invalid"
        return result
    if profile_digest and declared_profile != profile_digest:
        result["error"] = "dataset uses a different training profile"
        return result
    if collection_digest and declared_contract != collection_digest:
        result["error"] = "dataset uses different task, camera, clock, or motor settings"
        return result
    if (
        sidecar.get("training_profile_id") != snapshot.get("profile_id")
        or sidecar.get("training_profile_version") != snapshot.get("profile_version")
    ):
        result["error"] = "profile identity does not match its snapshot"
        return result
    result["compatible"] = True
    result["sidecar"] = sidecar
    return result


def _dataset_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return {
            "name": root.name,
            "root": str(root),
            "exists": root.exists(),
            "ready": False,
            "episodes": 0,
            "frames": 0,
            "fps": None,
            "camera_keys": [],
            "action_shape": None,
            "state_shape": None,
        }
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "name": root.name,
            "root": str(root),
            "exists": True,
            "ready": False,
            "error": str(exc),
        }
    features = info.get("features") if isinstance(info.get("features"), dict) else {}
    camera_keys = sorted(key for key in features if key.startswith("observation.images."))

    def shape(key: str) -> Any:
        feature = features.get(key, {})
        return feature.get("shape") if isinstance(feature, dict) else None

    stats_path = root / "meta" / "stats.json"
    quantile_ready = False
    if stats_path.is_file():
        try:
            stats = json.loads(stats_path.read_text())
            quantile_ready = all(
                isinstance(stats.get(key), dict)
                and all(name in stats[key] for name in ("q01", "q99"))
                for key in ("observation.state", "action")
            )
        except (OSError, json.JSONDecodeError):
            pass

    validation_path = RUN_ROOT / f"{root.name}--validation.json"
    validation: dict[str, Any] | None = None
    if validation_path.is_file():
        try:
            candidate = json.loads(validation_path.read_text())
            if isinstance(candidate, dict):
                validation = candidate
        except (OSError, json.JSONDecodeError):
            pass
    profile_status = training_profile_status()
    profile_digest = profile_status.get("profile_digest")
    sidecar_status = _dataset_profile_sidecar(root, str(profile_digest) if profile_digest else None)
    sidecar_digest = sidecar_status["profile_digest"]
    sidecar_contract_digest = sidecar_status["collection_contract_digest"]
    sidecar_compatible = bool(profile_digest and sidecar_status["compatible"])
    manifest_profile = (
        _manifest_profile_compatibility(
            root.name,
            str(profile_digest),
            str(sidecar_contract_digest) if sidecar_contract_digest else None,
        )
        if profile_digest
        else {
            "compatible": False,
            "manifest_count": 0,
            "valid_manifest_count": 0,
            "invalid_manifest_count": 0,
            "manifest_errors": [],
            "profile_digests": [],
            "collection_contract_digests": [],
            "lifecycle_states": [],
        }
    )
    validation_current = bool(
        validation
        and validation.get("passed") is True
        and validation.get("episodes") == info.get("total_episodes", 0)
        and validation.get("frames") == info.get("total_frames", 0)
        and profile_status.get("passed")
        and validation.get("training_profile_digest") == profile_digest
        and validation.get("collection_contract_digest") == sidecar_contract_digest
        and manifest_profile["compatible"]
        and sidecar_compatible
    )

    return {
        "name": root.name,
        "root": str(root),
        "repo_id": info.get("repo_id") or f"local/{root.name}",
        "exists": True,
        "ready": True,
        "episodes": info.get("total_episodes", 0),
        "frames": info.get("total_frames", 0),
        "fps": info.get("fps"),
        "camera_keys": camera_keys,
        "action_shape": shape("action"),
        "state_shape": shape("observation.state"),
        "robot_type": info.get("robot_type"),
        "quantile_stats": quantile_ready,
        "validation_passed": validation_current,
        "validation_report": str(validation_path),
        "training_profile_id": validation.get("training_profile_id") if validation else None,
        "training_profile_digest": validation.get("training_profile_digest") if validation else None,
        "profile_compatible": manifest_profile["compatible"],
        "profile_manifest": manifest_profile,
        "profile_sidecar": sidecar_status["path"],
        "profile_sidecar_digest": sidecar_digest,
        "collection_contract_digest": sidecar_contract_digest,
        "profile_sidecar_compatible": sidecar_compatible,
        "profile_sidecar_error": sidecar_status["error"],
    }


def dataset_inventory() -> list[dict[str, Any]]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return [
        _dataset_info(path)
        for path in sorted(DATA_ROOT.iterdir(), key=lambda value: value.name)
        if path.is_dir()
    ]


def camera_report() -> dict[str, Any]:
    report_path = CAMERA_ROOT / "camera_report.json"
    if not report_path.is_file():
        return {"available": False, "passed": False, "path": str(report_path)}
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": True, "passed": False, "path": str(report_path), "error": str(exc)}
    return {"available": True, "path": str(report_path), **report}


def camera_report_matches(config: dict[str, Any], report: dict[str, Any]) -> bool:
    if not report.get("passed"):
        return False
    cameras = report.get("cameras") if isinstance(report.get("cameras"), dict) else {}
    front = cameras.get("front") if isinstance(cameras.get("front"), dict) else {}
    side = cameras.get("side") if isinstance(cameras.get("side"), dict) else {}
    return bool(
        front.get("index") == config["front_camera"]
        and side.get("index") == config["side_camera"]
        and report.get("excluded_screen_camera_index") == config["excluded_camera"]
        and abs(float(report.get("requested_fps", 0)) - config["fps"]) < 0.01
        and front.get("width") == config["front_width"]
        and front.get("height") == config["front_height"]
        and side.get("width") == config["side_width"]
        and side.get("height") == config["side_height"]
        and not front.get("shape_mismatch", False)
        and not side.get("shape_mismatch", False)
    )


def preflight(devices: dict[str, Any]) -> dict[str, Any]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    report = camera_report()
    profile_status = training_profile_status()
    free_bytes = shutil.disk_usage(DATA_ROOT).free
    components = {
        "python": PYTHON_BIN.is_file(),
        "record": RECORD_BIN.is_file(),
        "controlled_record": CONTROLLED_RECORD.is_file(),
        "validator": DATASET_VALIDATOR.is_file(),
        "camera_check": CAMERA_CHECK.is_file(),
        "newt": NEWT_BIN.is_file(),
    }
    calibrations = devices.get("calibration", {}) if devices.get("ok") else {}
    ready = bool(
        devices.get("ok")
        and devices.get("ports_free")
        and calibrations.get("follower")
        and calibrations.get("leader")
        and profile_status.get("passed")
        and report.get("passed")
        and all(components[name] for name in ("python", "record", "controlled_record", "validator"))
        and free_bytes >= 5 * 1024**3
    )
    return {
        "ready_to_record": ready,
        "training_profile": profile_status,
        "devices": devices,
        "components": components,
        "camera_report": report,
        "disk": {"free_bytes": free_bytes, "data_root": str(DATA_ROOT)},
        "paths": {
            "data": str(DATA_ROOT),
            "models": str(MODEL_ROOT),
            "runs": str(RUN_ROOT),
            "deskpartner": str(KIT_ROOT.parent / "DeskPartner"),
        },
        "newt": {
            "installed": NEWT_BIN.is_file(),
            "authentication": "not rechecked by the GUI",
            "base": "molmoact2 / so101",
            "compatible": False,
            "reason": (
                "Mission/New Theory currently requires 6D SO-101 state/action, but this "
                "ReBot dataset is 7D (extra wrist_yaw). Managed upload stays disabled "
                "until a 7D ReBot base is provisioned or you explicitly approve a 6D adapter."
            ),
        },
        "official_molmoact2": {
            "compatible": True,
            "requires": "NVIDIA GPU; official LeRobot MolmoAct2 environment; clean 7D dataset",
            "checkpoint": (
                profile_status.get("profile", {})
                .get("training_defaults", {})
                .get("checkpoint", "allenai/MolmoAct2")
            ),
        },
        "datasets": dataset_inventory(),
    }


def training_recipe(dataset_name: str) -> dict[str, Any]:
    if not DATASET_SLUG_RE.fullmatch(dataset_name):
        raise TrainingConfigError("Choose a valid dataset name")
    profile_status = require_training_profile()
    profile = profile_status["profile"]
    training = profile["training_defaults"]
    dataset = _dataset_info(DATA_ROOT / dataset_name)
    if not dataset.get("ready"):
        raise TrainingConfigError("The selected dataset is not finalized")
    if not dataset.get("validation_passed"):
        raise TrainingConfigError(
            "Run Dataset validation under the current verified training profile after the final episode"
        )
    profile_compatibility = _manifest_profile_compatibility(
        dataset_name,
        profile_status["profile_digest"],
        dataset.get("collection_contract_digest"),
    )
    if not profile_compatibility["compatible"]:
        raise TrainingConfigError(
            "Dataset manifests do not all use the current training profile; do not mix calibrations"
        )
    image_keys = list(training["image_order"])
    if set(dataset.get("camera_keys") or []) != set(image_keys):
        raise TrainingConfigError("Dataset camera keys do not match the training profile")

    def truth(value: Any) -> str:
        return "true" if bool(value) else "false"

    normalization_mapping = {
        "VISUAL": "IDENTITY",
        "STATE": "QUANTILES",
        "ACTION": "QUANTILES",
    }
    gpu_dataset_root = f"rebot-training/data/{dataset_name}"
    gpu_output_root = f"rebot-training/models/{dataset_name}-molmoact2-rebot"
    gpu_profile_sidecar = f"{gpu_dataset_root}/meta/rebot_training_profile.json"
    profile_check = shlex.join(
        [
            "python",
            "-c",
            (
                "import hashlib,json; "
                f"p=json.load(open({gpu_profile_sidecar!r})); "
                "d=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(); "
                f"assert p['training_profile_digest']==d(p['profile_snapshot'])=={profile_status['profile_digest']!r}; "
                "assert p['collection_contract_digest']==d(p['collection_contract']); "
                "t=p['profile_snapshot']['training_defaults']; "
                "assert t['state_normalization']=='quantile' and t['action_normalization']=='quantile' and t['normalize_gripper'] is True; "
                "print('ReBot training profile verified:', p['training_profile_id'])"
            ),
        ]
    )
    official = [
        "accelerate",
        "launch",
        "--num_processes=1",
        "--mixed_precision=bf16",
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={dataset.get('repo_id') or f'local/{dataset_name}'}",
        f"--dataset.root={gpu_dataset_root}",
        "--dataset.video_backend=pyav",
        "--dataset.image_transforms.enable=true",
        f"--dataset.eval_split={training['eval_split']:g}",
        f"--policy.type={training['policy_type']}",
        f"--policy.checkpoint_path={training['checkpoint']}",
        f"--policy.device={training['device']}",
        f"--policy.action_mode={training['action_mode']}",
        f"--policy.chunk_size={training['chunk_size']}",
        f"--policy.n_action_steps={training['n_action_steps']}",
        "--policy.setup_type=single ReBot B601-DM seven-joint arm on a desk",
        f"--policy.control_mode={profile['coordinate_contract']['control_mode']}",
        "--policy.image_keys=" + json.dumps(image_keys, separators=(",", ":")),
        f"--policy.model_dtype={training['model_dtype']}",
        f"--policy.num_flow_timesteps={training['num_flow_timesteps']}",
        f"--policy.expected_max_action_dim={training['expected_max_action_dim']}",
        "--policy.normalization_mapping="
        + json.dumps(normalization_mapping, separators=(",", ":")),
        "--policy.gradient_checkpointing=true",
        f"--policy.train_action_expert_only={truth(training['train_action_expert_only'])}",
        "--policy.enable_lora_vlm=false",
        f"--policy.normalize_gripper={truth(training['normalize_gripper'])}",
        "--policy.freeze_embedding=true",
        "--policy.enable_knowledge_insulation=false",
        "--policy.push_to_hub=false",
        f"--job_name={dataset_name}-molmoact2-rebot",
        f"--output_dir={gpu_output_root}",
        f"--steps={training['steps']}",
        f"--policy.scheduler_warmup_steps={training['scheduler_warmup_steps']}",
        f"--policy.scheduler_decay_steps={training['scheduler_decay_steps']}",
        f"--batch_size={training['batch_size']}",
        "--num_workers=4",
        "--log_freq=20",
        "--env_eval_freq=-1",
        f"--eval_steps={training['eval_steps']}",
        "--max_eval_samples=128",
        "--save_checkpoint=true",
        f"--save_freq={training['save_freq']}",
        "--wandb.enable=false",
    ]
    return {
        "dataset": dataset,
        "route": training["route"],
        "training_profile": {
            "id": profile_status["profile_id"],
            "version": profile_status["profile_version"],
            "digest": profile_status["profile_digest"],
            "path": profile_status["profile_path"],
            "calibration_files": profile_status["calibration_files"],
            "snapshot": profile,
        },
        "gpu_paths": {
            "dataset": gpu_dataset_root,
            "output": gpu_output_root,
        },
        "host_setup": [
            "mkdir -p rebot-training/data rebot-training/models",
            profile_check,
        ],
        "command": official,
        "command_text": shlex.join(official),
        "post_training": [
            shlex.join(
                [
                    "cp",
                    gpu_profile_sidecar,
                    f"{gpu_output_root}/rebot_training_profile.json",
                ]
            )
        ],
        "notes": [
            "Start with action-expert-only continuous training for the first clean 7D run.",
            "Use a global batch size of 16-32 for fewer than 200 demonstrations.",
            "Keep the action expert fully trainable; do not full-fine-tune on the laptop.",
            "normalize_gripper=true is required because the ReBot gripper is stored in degrees.",
            "State/action quantile statistics and this profile digest are the calibration bridge into offline training.",
            "The optimizer does not reopen the physical-arm calibration JSON files or move either arm.",
            "This command is for an NVIDIA GPU host, not this Mac.",
        ],
        "newt": {
            "enabled": False,
            "would_be": [str(NEWT_BIN), "finetune", "--dataset", str(DATA_ROOT / dataset_name)],
            "blocker": "New Theory intake is 6D SO-101; ReBot data is 7D.",
        },
    }


def _update_run_manifest(path: Path | None, **lifecycle: Any) -> None:
    if path is None:
        return
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("manifest root is not an object")
        current = value.get("lifecycle") if isinstance(value.get("lifecycle"), dict) else {}
        value["lifecycle"] = {**current, **lifecycle}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not update run manifest {path}: {exc}") from exc


def attempt_archive_inventory(dataset: str = "") -> dict[str, Any]:
    try:
        attempts = read_attempt_inventory(ATTEMPT_ROOT, dataset.strip().lower())
    except ValueError as exc:
        raise TrainingConfigError(str(exc)) from exc
    return {
        "attempts": attempts,
        "failure_labels": list(FAILURE_LABELS),
        "archive_root": str(ATTEMPT_ROOT),
        "semantics": {
            "successes_are_unlabeled": True,
            "failed_attempts_are_excluded_from_training": True,
            "each_attempt_has_separate_video_and_rerun_files": True,
        },
    }


def update_attempt_failure_label(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TrainingConfigError("Request body must be a JSON object")
    try:
        metadata = write_failure_label(
            ATTEMPT_ROOT,
            str(payload.get("attempt_id", "")),
            payload.get("failure_label"),
            payload.get("failure_note", ""),
            expected_revision=payload.get("expected_revision"),
        )
    except (ValueError, FileNotFoundError) as exc:
        raise TrainingConfigError(str(exc)) from exc
    return {"attempt": metadata}


def _review_local_now() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")


def _review_transaction_update(transaction_root: Path, **updates: Any) -> dict[str, Any]:
    path = transaction_root / "review-transaction.json"
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Review transaction manifest is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Review transaction manifest is invalid: {path}")
    value.update(updates)
    atomic_write_json(path, value)
    return value


def _load_review_dataset(root: Path, repo_id: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id, root=root, batch_encoding_size=1)


def _prepare_lerobot_episode_exclusion(
    dataset_name: str,
    episode_index: int,
    attempt_id: str,
) -> dict[str, Any]:
    """Build and verify a recoverable replacement without touching the source."""

    if not DATASET_SLUG_RE.fullmatch(dataset_name):
        raise TrainingConfigError("Invalid dataset name")
    dataset_root = DATA_ROOT / dataset_name
    if not dataset_root.is_dir() or dataset_root.is_symlink():
        raise RuntimeError(f"LeRobot dataset is not safely available: {dataset_root}")

    repo_id = f"local/{dataset_name}"
    try:
        source = _load_review_dataset(dataset_root, repo_id)
    except Exception as exc:
        raise RuntimeError(f"LeRobot dataset could not be fresh-loaded: {exc}") from exc

    total_episodes = int(source.num_episodes)
    if isinstance(episode_index, bool) or not 0 <= episode_index < total_episodes:
        raise TrainingConfigError("Attempt points to an invalid LeRobot episode")
    episode_metadata = source.meta.episodes[episode_index]
    removed_frames = int(episode_metadata.get("length", 0))
    expected_frames = int(source.num_frames) - removed_frames
    remaining_episodes = total_episodes - 1
    if expected_frames < 0:
        raise RuntimeError("LeRobot episode frame metadata is inconsistent")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    transaction_root = (
        DATASET_REVISION_ROOT
        / dataset_name
        / f"{stamp}--exclude-{attempt_id}-{uuid4().hex[:8]}"
    )
    transaction_root.mkdir(parents=True, exist_ok=False)
    backup_root = transaction_root / "source-dataset"
    replacement_root = transaction_root / "replacement-dataset"
    validation_path = RUN_ROOT / f"{dataset_name}--validation.json"
    validation_backup = transaction_root / "previous-validation.json"
    manifest = {
        "schema_version": 1,
        "transaction_id": transaction_root.name,
        "state": "preparing",
        "created_at_utc": utc_now(),
        "created_at_local": _review_local_now(),
        "timezone": "America/Los_Angeles",
        "dataset": dataset_name,
        "source_path": str(dataset_root),
        "backup_path": str(backup_root),
        "replacement_path": str(replacement_root) if remaining_episodes else None,
        "attempt_id": attempt_id,
        "removed_episode_index": episode_index,
        "source_episodes": total_episodes,
        "source_frames": int(source.num_frames),
        "removed_frames": removed_frames,
        "remaining_episodes": remaining_episodes,
        "expected_remaining_frames": expected_frames,
    }
    atomic_write_json(transaction_root / "review-transaction.json", manifest)

    if remaining_episodes:
        try:
            from lerobot.datasets.dataset_tools import delete_episodes

            replacement = delete_episodes(
                source,
                [episode_index],
                output_dir=replacement_root,
                repo_id=repo_id,
            )
            profile_sidecar = dataset_root / "meta" / "rebot_training_profile.json"
            if profile_sidecar.is_file():
                destination = replacement_root / "meta" / profile_sidecar.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(profile_sidecar, destination)
            del replacement
            verified = _load_review_dataset(replacement_root, repo_id)
            if (
                int(verified.num_episodes) != remaining_episodes
                or int(verified.num_frames) != expected_frames
            ):
                raise RuntimeError(
                    "Replacement LeRobot episode/frame counts do not match the review plan"
                )
            del verified
        except Exception as exc:
            _review_transaction_update(
                transaction_root,
                state="prepare_failed",
                error=str(exc),
                failed_at_utc=utc_now(),
                failed_at_local=_review_local_now(),
            )
            raise RuntimeError(
                "Could not build a verified LeRobot dataset without this episode; "
                "the original dataset was not changed"
            ) from exc

    del source
    _review_transaction_update(
        transaction_root,
        state="prepared",
        prepared_at_utc=utc_now(),
        prepared_at_local=_review_local_now(),
    )
    return {
        **manifest,
        "transaction_root": transaction_root,
        "dataset_root": dataset_root,
        "backup_root": backup_root,
        "replacement_root": replacement_root if remaining_episodes else None,
        "validation_path": validation_path,
        "validation_backup": validation_backup,
        "repo_id": repo_id,
    }


def _commit_lerobot_episode_exclusion(plan: dict[str, Any]) -> None:
    dataset_root = Path(plan["dataset_root"])
    backup_root = Path(plan["backup_root"])
    replacement_root = (
        Path(plan["replacement_root"]) if plan.get("replacement_root") else None
    )
    validation_path = Path(plan["validation_path"])
    validation_backup = Path(plan["validation_backup"])
    transaction_root = Path(plan["transaction_root"])
    try:
        dataset_root.replace(backup_root)
        if replacement_root is not None:
            replacement_root.replace(dataset_root)
            verified = _load_review_dataset(dataset_root, str(plan["repo_id"]))
            if (
                int(verified.num_episodes) != int(plan["remaining_episodes"])
                or int(verified.num_frames) != int(plan["expected_remaining_frames"])
            ):
                raise RuntimeError("Active replacement failed fresh-load verification")
            del verified
        elif dataset_root.exists():
            raise RuntimeError("Zero-episode exclusion left an active dataset behind")
        if validation_path.is_file():
            validation_path.replace(validation_backup)
        _review_transaction_update(
            transaction_root,
            state="dataset_swapped",
            swapped_at_utc=utc_now(),
            swapped_at_local=_review_local_now(),
        )
    except Exception as exc:
        try:
            if dataset_root.exists():
                failed_replacement = transaction_root / "failed-active-replacement"
                dataset_root.replace(failed_replacement)
            if backup_root.exists():
                backup_root.replace(dataset_root)
            if validation_backup.exists() and not validation_path.exists():
                validation_backup.replace(validation_path)
        except Exception as rollback_exc:
            _review_transaction_update(
                transaction_root,
                state="rollback_failed",
                error=str(exc),
                rollback_error=str(rollback_exc),
                failed_at_utc=utc_now(),
                failed_at_local=_review_local_now(),
            )
            raise RuntimeError(
                "LeRobot review swap and automatic rollback both failed; validation is blocked"
            ) from rollback_exc
        _review_transaction_update(
            transaction_root,
            state="swap_failed_rolled_back",
            error=str(exc),
            failed_at_utc=utc_now(),
            failed_at_local=_review_local_now(),
        )
        raise RuntimeError(
            "LeRobot review swap failed and was rolled back; no training episode was removed"
        ) from exc


def _rollback_lerobot_episode_exclusion(plan: dict[str, Any], reason: str) -> None:
    dataset_root = Path(plan["dataset_root"])
    backup_root = Path(plan["backup_root"])
    validation_path = Path(plan["validation_path"])
    validation_backup = Path(plan["validation_backup"])
    transaction_root = Path(plan["transaction_root"])
    if dataset_root.exists():
        rolled_back_replacement = transaction_root / "rolled-back-replacement"
        dataset_root.replace(rolled_back_replacement)
    if not backup_root.exists():
        raise RuntimeError("Review rollback source dataset is missing")
    backup_root.replace(dataset_root)
    if validation_backup.exists() and not validation_path.exists():
        validation_backup.replace(validation_path)
    restored = _load_review_dataset(dataset_root, str(plan["repo_id"]))
    if int(restored.num_episodes) != int(plan["source_episodes"]):
        raise RuntimeError("Review rollback did not restore the source episode count")
    del restored
    _review_transaction_update(
        transaction_root,
        state="metadata_failed_rolled_back",
        error=reason,
        rolled_back_at_utc=utc_now(),
        rolled_back_at_local=_review_local_now(),
    )


def _dataset_attempt_mapping(dataset_name: str, expected_episodes: int) -> list[dict[str, Any]]:
    attempts = read_attempt_inventory(ATTEMPT_ROOT, dataset_name)
    included = [item for item in attempts if item.get("training_included") is True]
    raw_indices = [item.get("training_episode_index") for item in included]
    if any(isinstance(index, bool) or not isinstance(index, int) for index in raw_indices):
        raise RuntimeError(
            "Attempt-to-LeRobot episode mapping is inconsistent; review cannot safely mutate data"
        )
    indices = sorted(raw_indices)
    if indices != list(range(expected_episodes)):
        raise RuntimeError(
            "Attempt-to-LeRobot episode mapping is inconsistent; review cannot safely mutate data"
        )
    return included


def _restore_attempt_metadata(snapshots: dict[Path, dict[str, Any]]) -> None:
    for path, value in snapshots.items():
        atomic_write_json(path, value)


def attempt_artifact_path(attempt_id: str, artifact: str) -> Path:
    try:
        return archived_artifact_path(ATTEMPT_ROOT, attempt_id, artifact)
    except ValueError as exc:
        raise TrainingConfigError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc


def replay_attempt(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TrainingConfigError("Request body must be a JSON object")
    attempt_id = str(payload.get("attempt_id", ""))
    rrd_path = attempt_artifact_path(attempt_id, "rerun")
    try:
        _directory, metadata = find_attempt(ATTEMPT_ROOT, attempt_id)
    except (ValueError, FileNotFoundError) as exc:
        raise TrainingConfigError(str(exc)) from exc
    if not RERUN_NATIVE_BIN.is_file() or not os.access(RERUN_NATIVE_BIN, os.X_OK):
        raise RuntimeError(f"Rerun native viewer is unavailable: {RERUN_NATIVE_BIN}")
    process = subprocess.Popen(
        [str(RERUN_NATIVE_BIN), str(rrd_path)],
        cwd=str(rrd_path.parent),
        env=runtime_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {
        "attempt_id": attempt_id,
        "dataset": metadata.get("dataset"),
        "pid": process.pid,
        "replaying": True,
    }


class TrainingManager:
    _SHUTDOWN_WAIT_SLICE_S = 10.0
    _RECORD_ARCHIVE_GRACE_S = 120.0

    def __init__(
        self,
        simulate: bool = False,
        hardware_lock: threading.Lock | None = None,
    ) -> None:
        self.simulate = simulate
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._hardware_lock = hardware_lock or threading.Lock()
        self._hardware_token: object | None = None
        self._decision_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._state = "READY"
        self._kind: str | None = None
        self._config: dict[str, Any] | None = None
        self._started_at: float | None = None
        self._last_exit_code: int | None = None
        self._fault: str | None = None
        self._record_control_file: Path | None = None
        self._record_decision_pending = False
        self._record_phase: str | None = None
        self._seq = 0
        self._logs: deque[dict[str, Any]] = deque(maxlen=2500)
        self._append_log("INFO", "ReBot data and training workspace ready")

    def _append_log(self, level: str, message: str) -> None:
        with self._lock:
            self._seq += 1
            self._logs.append(
                {
                    "seq": self._seq,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "level": level,
                    "message": message,
                }
            )

    def _ensure_idle(self) -> None:
        with self._lock:
            if self._process is not None:
                phase = "running" if self._process.poll() is None else "finalizing"
                raise RuntimeError(
                    f"A {self._kind or 'workspace'} job is still {phase}"
                )

    def _run_start_operation(
        self,
        operation: Callable[..., dict[str, Any]],
        *args: Any,
    ) -> dict[str, Any]:
        """Reserve the whole validate/artifact/Popen transaction atomically."""

        if not self._lifecycle_lock.acquire(blocking=False):
            raise RuntimeError("Another training workspace start or stop is in progress")
        try:
            return operation(*args)
        finally:
            self._lifecycle_lock.release()

    def _claim_hardware(self) -> object:
        if not self._hardware_lock.acquire(blocking=False):
            raise RuntimeError(
                "The ReBot hardware is starting, running, or stopping in another session"
            )
        token = object()
        with self._lock:
            self._hardware_token = token
        return token

    def _release_hardware(self, token: object) -> None:
        with self._lock:
            if self._hardware_token is not token:
                return
            self._hardware_token = None
        self._hardware_lock.release()

    def _release_process_ownership_when_done(
        self,
        process: subprocess.Popen[str],
        owner_write_fd: int,
        hardware_token: object | None,
    ) -> None:
        try:
            process.wait()
        finally:
            try:
                os.close(owner_write_fd)
            except OSError:
                pass
            if hardware_token is not None:
                self._release_hardware(hardware_token)

    def _start_process(
        self,
        kind: str,
        command: list[str],
        *,
        config: dict[str, Any] | None = None,
        cwd: Path = KIT_ROOT,
        manifest_path: Path | None = None,
        record_control_file: Path | None = None,
    ) -> dict[str, Any]:
        self._ensure_idle()
        with self._lock:
            self._state = "STARTING"
            self._kind = kind
            self._config = config
            self._started_at = None
            self._last_exit_code = None
            self._fault = None
            self._record_control_file = record_control_file if kind == "record" else None
            self._record_decision_pending = kind == "record"
            self._record_phase = "starting" if kind == "record" else None
        self._append_log("INFO", f"Starting {kind}")
        owner_read_fd, owner_write_fd = os.pipe()
        owner_stop_signal = "SIGHUP" if kind == "record" else "SIGINT"
        wrapped_command = [
            str(PYTHON_BIN),
            str(OWNED_PROCESS),
            "--owner-fd",
            str(owner_read_fd),
            "--owner-stop-signal",
            owner_stop_signal,
            "--",
            *command,
        ]
        try:
            process = subprocess.Popen(
                wrapped_command,
                cwd=str(cwd),
                env=runtime_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
                pass_fds=(owner_read_fd,),
            )
        except Exception as exc:
            for descriptor in (owner_read_fd, owner_write_fd):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            try:
                _update_run_manifest(
                    manifest_path,
                    state="FAILED_TO_START",
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                    exit_code=None,
                    error=str(exc),
                )
            except RuntimeError as manifest_exc:
                self._append_log("ERROR", str(manifest_exc))
            if kind == "record" and config and config.get("dataset"):
                try:
                    quarantined = _quarantine_zero_frame_attempt(
                        str(config["dataset"]),
                        manifest_paths=[manifest_path] if manifest_path else None,
                        reason=f"record process failed to start: {exc}",
                    )
                    if quarantined:
                        self._append_log(
                            "WARN",
                            "Zero-frame attempt quarantined; the same dataset name is ready "
                            f"to retry: {quarantined}",
                        )
                except (OSError, RuntimeError) as quarantine_exc:
                    self._append_log(
                        "ERROR", f"Zero-frame quarantine failed: {quarantine_exc}"
                    )
            with self._lock:
                self._state = "FAULT"
                self._fault = str(exc)
                self._record_control_file = None
                self._record_decision_pending = False
                self._record_phase = None
            self._append_log("ERROR", f"{kind} failed to start: {exc}")
            raise
        os.close(owner_read_fd)
        with self._lock:
            self._process = process
            self._state = "RUNNING"
            self._started_at = time.monotonic()
        reader = threading.Thread(
            target=self._read_process,
            args=(process, kind, manifest_path, config),
            name=f"{kind}-log-reader",
            daemon=True,
        )
        with self._lock:
            self._reader_thread = reader
        with self._lock:
            hardware_token = self._hardware_token if kind == "record" else None
        if kind == "record" and hardware_token is None:
            raise RuntimeError("Record process started without a hardware ownership token")
        threading.Thread(
            target=self._release_process_ownership_when_done,
            args=(process, owner_write_fd, hardware_token),
            name=f"{kind}-process-owner",
            daemon=True,
        ).start()
        try:
            _update_run_manifest(
                manifest_path,
                state="RUNNING",
                started_at=datetime.now().isoformat(timespec="seconds"),
                pid=process.pid,
                exit_code=None,
                error=None,
            )
        except RuntimeError as exc:
            # Drain stdout before signalling so a verbose collector cannot fill
            # its PIPE while rollback waits for archive/recovery and disconnect.
            reader.start()
            self._wait_for_owned_process_exit(
                process,
                kind,
                reader,
                context="Startup rollback",
            )
            with self._lock:
                self._process = None
                self._reader_thread = None
                self._state = "FAULT"
                self._fault = str(exc)
                self._record_control_file = None
                self._record_decision_pending = False
                self._record_phase = None
            self._append_log("ERROR", str(exc))
            raise
        reader.start()
        return self.status()

    def _read_process(
        self,
        process: subprocess.Popen[str],
        kind: str,
        manifest_path: Path | None,
        config: dict[str, Any] | None,
    ) -> None:
        assert process.stdout is not None
        durable_episodes_this_run = 0
        durable_dataset_episodes = 0
        for raw in process.stdout:
            line = raw.rstrip("\r\n")
            if not line:
                continue
            lower = line.lower()
            if kind == "record":
                with self._lock:
                    if line.startswith("ATTEMPT recording"):
                        self._record_decision_pending = False
                        self._record_phase = "recording"
                    elif line.startswith("AWAITING_DECISION"):
                        self._record_phase = "awaiting_decision"
                    elif line.startswith("ATTEMPT save_started"):
                        self._record_phase = "saving_rerun"
                    elif line.startswith("ATTEMPT save_phase") and "phase=lerobot" in line:
                        self._record_phase = "saving_lerobot"
                    elif line.startswith("ATTEMPT lerobot_durable"):
                        self._record_phase = "returning_home"
                        durable_episodes_this_run += 1
                        total_match = re.search(r"\btotal_episodes=(\d+)\b", line)
                        if total_match:
                            durable_dataset_episodes = int(total_match.group(1))
                    elif line.startswith("RESET auto_home"):
                        self._record_phase = "returning_home"
                if line.startswith("ATTEMPT lerobot_durable"):
                    try:
                        _update_run_manifest(
                            manifest_path,
                            durable_episodes_this_run=durable_episodes_this_run,
                            durable_dataset_episodes=durable_dataset_episodes,
                        )
                    except RuntimeError as exc:
                        self._append_log("ERROR", str(exc))
            level = "INFO"
            if line.startswith("ATTEMPT archived_failed"):
                level = "INFO"
            elif "traceback" in lower or "error" in lower or "failed" in lower:
                level = "ERROR"
            elif "warning" in lower or "clamped" in lower:
                level = "WARN"
            self._append_log(level, line)
        exit_code = process.wait()
        lifecycle_state = (
            "COMPLETE"
            if exit_code == 0
            else "PARTIAL_COMPLETE"
            if durable_episodes_this_run > 0
            else "FAILED"
        )
        try:
            _update_run_manifest(
                manifest_path,
                state=lifecycle_state,
                finished_at=datetime.now().isoformat(timespec="seconds"),
                exit_code=exit_code,
                error=None if exit_code == 0 else f"{kind} exited with code {exit_code}",
                durable_episodes_this_run=durable_episodes_this_run,
                durable_dataset_episodes=durable_dataset_episodes,
            )
        except RuntimeError as exc:
            self._append_log("ERROR", str(exc))
            if exit_code == 0:
                exit_code = 74
        if kind == "record" and config and config.get("dataset"):
            try:
                quarantined = _quarantine_zero_frame_attempt(
                    str(config["dataset"]),
                    manifest_paths=[manifest_path] if manifest_path else None,
                    reason=f"record process finished with exit code {exit_code}",
                )
                if quarantined:
                    self._append_log(
                        "WARN",
                        "Zero-frame attempt quarantined; the same dataset name is ready "
                        f"to retry: {quarantined}",
                    )
            except (OSError, RuntimeError) as exc:
                self._append_log("ERROR", f"Zero-frame quarantine failed: {exc}")
        with self._lock:
            if self._process is not process:
                return
            self._process = None
            self._reader_thread = None
            self._last_exit_code = exit_code
            if kind == "record":
                self._record_control_file = None
                self._record_decision_pending = False
                self._record_phase = None
            if exit_code == 0:
                self._state = "COMPLETE"
                self._fault = None
            else:
                self._state = "FAULT"
                self._fault = f"{kind} exited with code {exit_code}"
        self._append_log(
            "INFO" if exit_code == 0 else "ERROR",
            f"{kind} finished (exit {exit_code})",
        )

    def start_camera_check(self, payload: Any) -> dict[str, Any]:
        return self._run_start_operation(self._start_camera_check_locked, payload)

    def _start_camera_check_locked(self, payload: Any) -> dict[str, Any]:
        config = validate_collection_config(payload)
        profile_status = require_training_profile()
        camera_defaults = profile_status["profile"]["camera_defaults"]
        minimum_fps = (
            camera_defaults["minimum_measured_fps"]
            if config["fps"] == profile_status["defaults"]["fps"]
            else max(5, config["fps"] - 3)
        )
        command = [
            str(PYTHON_BIN),
            str(CAMERA_CHECK),
            "--front",
            str(config["front_camera"]),
            "--side",
            str(config["side_camera"]),
            "--excluded-screen",
            str(config["excluded_camera"]),
            "--fps",
            str(config["fps"]),
            "--front-width",
            str(config["front_width"]),
            "--front-height",
            str(config["front_height"]),
            "--side-width",
            str(config["side_width"]),
            "--side-height",
            str(config["side_height"]),
            "--minimum-fps",
            str(minimum_fps),
            "--duration",
            "6",
            "--output",
            str(CAMERA_ROOT),
        ]
        return self._start_process("camera_check", command, config=config)

    def start_record(
        self,
        payload: Any,
        devices: dict[str, Any],
    ) -> dict[str, Any]:
        def start_with_hardware_claim() -> dict[str, Any]:
            hardware_token = self._claim_hardware()
            try:
                return self._start_record_locked(payload, devices)
            except BaseException:
                # Once Popen succeeds a watcher releases the claim only after
                # the collector is truly gone.  Before that point, this is the
                # rollback path for validation/artifact/start failures.
                self._release_hardware(hardware_token)
                raise

        return self._run_start_operation(start_with_hardware_claim)

    def _start_record_locked(
        self,
        payload: Any,
        devices: dict[str, Any],
    ) -> dict[str, Any]:
        config = validate_collection_config(payload)
        profile_status = require_training_profile()
        current_preflight = preflight(devices)
        if not current_preflight["ready_to_record"]:
            reasons = []
            if not devices.get("ok"):
                reasons.append(devices.get("error", "arms unavailable"))
            elif not devices.get("ports_free"):
                reasons.append("arm serial ports are busy; stop teleop first")
            if not current_preflight["camera_report"].get("passed"):
                reasons.append("dual-camera check has not passed")
            if not current_preflight["training_profile"].get("passed"):
                reasons.append(
                    current_preflight["training_profile"].get("error")
                    or "training profile is not verified"
                )
            if not reasons:
                reasons.append("preflight is incomplete")
            raise RuntimeError("; ".join(reasons))
        if not camera_report_matches(config, current_preflight["camera_report"]):
            raise RuntimeError(
                "Camera check does not match the selected indices, FPS, and actual frame sizes"
            )

        root = DATA_ROOT / config["dataset"]
        if not config["resume"]:
            quarantined = _quarantine_zero_frame_attempt(
                config["dataset"],
                reason="pre-start cleanup of an earlier zero-frame attempt",
            )
            if quarantined:
                self._append_log(
                    "WARN",
                    "Earlier zero-frame attempt quarantined; retry namespace is clean: "
                    f"{quarantined}",
                )
        if root.exists() and not root.is_dir():
            raise RuntimeError(f"Dataset path is not a directory: {root}")
        populated = root.exists() and any(root.iterdir())
        if populated and not config["resume"]:
            raise RuntimeError(
                f"Dataset already contains data: {root}. Enable Resume or choose a new name."
            )
        if root.exists() and not populated and not config["resume"]:
            raise RuntimeError(
                f"An empty dataset folder already exists: {root}. Choose a new dataset name "
                "or remove that empty folder before retrying."
            )
        if config["resume"] and not (root / "meta" / "info.json").is_file():
            raise RuntimeError("Resume requires an existing LeRobot dataset with meta/info.json")
        contract = collection_contract_lock(config)
        if config["resume"]:
            sidecar_status = _dataset_profile_sidecar(
                root,
                profile_status["profile_digest"],
                contract["digest"],
            )
            if not sidecar_status["compatible"]:
                raise RuntimeError(
                    "Resume is blocked: "
                    + str(sidecar_status["error"] or "dataset profile sidecar is incompatible")
                )
            compatibility = _manifest_profile_compatibility(
                config["dataset"],
                profile_status["profile_digest"],
                contract["digest"],
            )
            if not compatibility["compatible"]:
                raise RuntimeError(
                    "Resume is blocked because a prior run manifest is incomplete, malformed, "
                    "or uses different profile/collection settings"
                )
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        ATTEMPT_ROOT.mkdir(parents=True, exist_ok=True)
        CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
        run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        control_file = CONTROL_ROOT / f"{config['dataset']}--{run_stamp}.json"
        atomic_write_json(
            control_file,
            {
                "action": None,
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
                "dataset": config["dataset"],
            },
        )
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": config,
            "ports": devices.get("ports"),
            "data_root": str(root),
            "attempt_archive_root": str(ATTEMPT_ROOT / config["dataset"]),
            "control_file": str(control_file),
            "training_profile": {
                "id": profile_status["profile_id"],
                "version": profile_status["profile_version"],
                "digest": profile_status["profile_digest"],
                "path": profile_status["profile_path"],
                "calibration_files": profile_status["calibration_files"],
                "snapshot": profile_status["profile"],
            },
            "collection_contract": contract,
            "lifecycle": {
                "state": "AUTHORIZED",
                "authorized_at": datetime.now().isoformat(timespec="seconds"),
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "error": None,
            },
            "training_route": "official MolmoAct2 LeRobot 7D",
            "newt_compatibility": "blocked: current managed base is SO-101 6D",
        }
        manifest_name = f"{config['dataset']}--{run_stamp}.json"
        manifest_path = RUN_ROOT / manifest_name
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        self._append_log("INFO", f"Run manifest: {manifest_path}")
        return self._start_process(
            "record",
            build_record_command(
                config,
                devices["ports"],
                attempt_root=ATTEMPT_ROOT,
                control_file=control_file,
            ),
            config=config,
            cwd=RUNTIME_ROOT,
            manifest_path=manifest_path,
            record_control_file=control_file,
        )

    def start_validation(self, payload: Any) -> dict[str, Any]:
        return self._run_start_operation(self._start_validation_locked, payload)

    def _start_validation_locked(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TrainingConfigError("Request body must be a JSON object")
        name = str(payload.get("dataset", "")).strip().lower()
        if not DATASET_SLUG_RE.fullmatch(name):
            raise TrainingConfigError("Choose a valid dataset")
        profile_status = require_training_profile()
        root = DATA_ROOT / name
        sidecar_status = _dataset_profile_sidecar(root, profile_status["profile_digest"])
        if not sidecar_status["compatible"]:
            raise RuntimeError(
                "Dataset profile sidecar is invalid: "
                + str(sidecar_status["error"] or "unknown profile mismatch")
            )
        compatibility = _manifest_profile_compatibility(
            name,
            profile_status["profile_digest"],
            sidecar_status["collection_contract_digest"],
        )
        if not compatibility["compatible"]:
            raise RuntimeError(
                "Dataset manifests do not all use the current verified training profile"
            )
        minimum = _number(
            payload.get("minimum_episodes", 5),
            "Minimum episodes",
            1,
            500,
            whole=True,
        )
        if not (root / "meta" / "info.json").is_file():
            raise RuntimeError(f"Dataset is not ready: {root}")
        step_caps: list[float] = []
        for manifest in _collection_manifests(name)["records"]:
            value = manifest.get("config", {}).get("max_step")
            if value is not None:
                try:
                    step_caps.append(float(value))
                except (ValueError, TypeError) as exc:
                    raise RuntimeError("Run manifest contains an invalid max_step") from exc
        profile_step = float(profile_status["defaults"]["max_step"])
        maximum_delta = max(step_caps, default=profile_step) + 0.5
        command = [
            str(PYTHON_BIN),
            str(DATASET_VALIDATOR),
            "--repo-id",
            f"local/{name}",
            "--root",
            str(root),
            "--minimum-episodes",
            str(minimum),
            "--max-action-state-delta",
            f"{maximum_delta:g}",
            "--profile",
            str(PROFILE_PATH),
            "--report",
            str(RUN_ROOT / f"{name}--validation.json"),
        ]
        return self._start_process(
            "validate",
            command,
            config={"dataset": name, "minimum_episodes": minimum},
        )

    def review_attempt(self, payload: Any) -> dict[str, Any]:
        if not self._lifecycle_lock.acquire(blocking=False):
            raise RuntimeError("Another training workspace operation is in progress")
        try:
            self._ensure_idle()
            return self._review_attempt_locked(payload)
        finally:
            self._lifecycle_lock.release()

    def _review_attempt_locked(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TrainingConfigError("Request body must be a JSON object")
        attempt_id = str(payload.get("attempt_id", ""))
        action = str(payload.get("action", "")).strip().lower()
        if action not in {"", "mark_failed", "label_excluded"}:
            raise TrainingConfigError("Unknown attempt review action")
        try:
            directory, metadata = find_attempt(ATTEMPT_ROOT, attempt_id)
            label, note = validate_failure_label(
                payload.get("failure_label"), payload.get("failure_note", "")
            )
            current_revision = review_revision(metadata)
        except (ValueError, FileNotFoundError) as exc:
            raise TrainingConfigError(str(exc)) from exc
        if metadata.get("archive_complete") is not True:
            raise TrainingConfigError("Only a completed archived attempt can be reviewed")
        expected_revision = payload.get("expected_revision")
        if expected_revision is not None:
            if isinstance(expected_revision, bool):
                raise TrainingConfigError("Attempt review revision is invalid")
            try:
                expected = int(expected_revision)
            except (TypeError, ValueError) as exc:
                raise TrainingConfigError("Attempt review revision is invalid") from exc
            if expected != current_revision:
                raise TrainingConfigError(
                    "This attempt was reviewed in another tab; refresh the archive before saving"
                )

        is_included_kept = bool(
            metadata.get("disposition") == "kept"
            and metadata.get("training_included") is True
            and isinstance(metadata.get("training_episode_index"), int)
            and not isinstance(metadata.get("training_episode_index"), bool)
        )
        if not is_included_kept:
            if action == "mark_failed":
                raise TrainingConfigError("This attempt is already excluded from LeRobot")
            try:
                updated = write_failure_label(
                    ATTEMPT_ROOT,
                    attempt_id,
                    label,
                    note,
                    expected_revision=expected_revision,
                )
            except (ValueError, FileNotFoundError) as exc:
                raise TrainingConfigError(str(exc)) from exc
            self._append_log(
                "INFO",
                f"Reviewed excluded attempt {attempt_id}: {label}",
            )
            return {
                "attempt": updated,
                "review_action": "label_excluded",
                "training_changed": False,
                "message": "Review label saved; this attempt remains excluded from LeRobot.",
            }

        if action == "label_excluded":
            raise TrainingConfigError(
                "A kept attempt must be marked failed and removed from LeRobot"
            )
        dataset_name = str(metadata.get("dataset", ""))
        episode_index = int(metadata["training_episode_index"])
        info_path = DATA_ROOT / dataset_name / "meta" / "info.json"
        try:
            info = json.loads(info_path.read_text())
            expected_episodes = int(info["total_episodes"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("LeRobot dataset metadata is unavailable or invalid") from exc
        included = _dataset_attempt_mapping(dataset_name, expected_episodes)
        matching = [item for item in included if item.get("attempt_id") == attempt_id]
        if len(matching) != 1 or matching[0].get("training_episode_index") != episode_index:
            raise RuntimeError(
                "Finished attempt does not map uniquely to its LeRobot episode"
            )

        reindexed = sorted(
            (
                item
                for item in included
                if int(item["training_episode_index"]) > episode_index
            ),
            key=lambda item: int(item["training_episode_index"]),
        )
        affected_ids = [attempt_id, *(str(item["attempt_id"]) for item in reindexed)]
        snapshots: dict[Path, dict[str, Any]] = {}
        for affected_id in affected_ids:
            affected_directory, affected_metadata = find_attempt(ATTEMPT_ROOT, affected_id)
            snapshots[affected_directory / "metadata.json"] = affected_metadata

        plan = _prepare_lerobot_episode_exclusion(
            dataset_name,
            episode_index,
            attempt_id,
        )
        _commit_lerobot_episode_exclusion(plan)
        try:
            updated = write_reviewed_failure(
                ATTEMPT_ROOT,
                attempt_id,
                label,
                note,
                expected_revision=expected_revision,
            )
            updated["training_exclusion_transaction"] = Path(
                plan["transaction_root"]
            ).name
            updated["training_dataset_revision_backup"] = str(plan["backup_root"])
            updated["training_dataset_remaining_episodes"] = int(
                plan["remaining_episodes"]
            )
            atomic_write_json(directory / "metadata.json", updated)

            for item in reindexed:
                old_index = int(item["training_episode_index"])
                changed = write_reindexed_episode(
                    ATTEMPT_ROOT,
                    str(item["attempt_id"]),
                    old_index=old_index,
                    new_index=old_index - 1,
                    review_attempt_id=attempt_id,
                )
                changed["training_reindex_transaction"] = Path(
                    plan["transaction_root"]
                ).name
                changed_directory, _ = find_attempt(
                    ATTEMPT_ROOT, str(item["attempt_id"])
                )
                atomic_write_json(changed_directory / "metadata.json", changed)

            _dataset_attempt_mapping(
                dataset_name,
                int(plan["remaining_episodes"]),
            )
            _review_transaction_update(
                Path(plan["transaction_root"]),
                state="complete",
                failure_label=label,
                failure_note=note,
                completed_at_utc=utc_now(),
                completed_at_local=_review_local_now(),
            )
        except Exception as exc:
            metadata_restore_error: Exception | None = None
            try:
                _restore_attempt_metadata(snapshots)
            except Exception as restore_exc:
                metadata_restore_error = restore_exc
            try:
                _rollback_lerobot_episode_exclusion(plan, str(exc))
            except Exception as rollback_exc:
                raise RuntimeError(
                    "Post-review metadata failed and the LeRobot rollback needs manual recovery"
                ) from rollback_exc
            if metadata_restore_error is not None:
                raise RuntimeError(
                    "LeRobot was restored but attempt metadata needs manual recovery"
                ) from metadata_restore_error
            raise RuntimeError(
                "Post-review metadata failed; the original LeRobot dataset was restored"
            ) from exc

        remaining = int(plan["remaining_episodes"])
        self._append_log(
            "INFO",
            f"Attempt {attempt_id} marked failed and removed from LeRobot episode "
            f"{episode_index}; {remaining} episode(s) remain",
        )
        return {
            "attempt": updated,
            "review_action": "mark_failed",
            "training_changed": True,
            "removed_episode_index": episode_index,
            "remaining_episodes": remaining,
            "dataset_empty": remaining == 0,
            "backup_root": str(plan["backup_root"]),
            "message": (
                "Failure label saved and the episode was removed from LeRobot. "
                + (
                    "The dataset now has no kept episodes; start the next run with Resume off."
                    if remaining == 0
                    else f"{remaining} kept episode(s) remain and were fresh-load verified."
                )
            ),
        }

    def record_control(self, payload: Any) -> dict[str, Any]:
        with self._decision_lock:
            return self._record_control_locked(payload)

    def _record_control_locked(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, str):
            payload = {"action": payload}
        if not isinstance(payload, dict):
            raise TrainingConfigError("Request body must be a JSON object")
        action = str(payload.get("action", ""))
        signals = {
            "finish": signal.SIGUSR1,
            "rerecord": signal.SIGUSR2,
            "stop": signal.SIGHUP,
        }
        if action not in signals:
            raise TrainingConfigError("Unknown recording action")
        decision: dict[str, Any] = {
            "action": action,
            "requested_at": datetime.now().isoformat(timespec="milliseconds"),
            "sequence": time.time_ns(),
        }
        if action == "rerecord":
            try:
                label, note = validate_failure_label(
                    payload.get("failure_label"), payload.get("failure_note", "")
                )
            except ValueError as exc:
                raise TrainingConfigError(str(exc)) from exc
            decision["failure_label"] = label
            decision["failure_note"] = note
        elif payload.get("failure_label") or payload.get("failure_note"):
            raise TrainingConfigError("Only failed attempts may have a failure label")
        with self._lock:
            process = self._process
            control_file = self._record_control_file
            if self._kind != "record" or process is None or process.poll() is not None:
                raise RuntimeError("No recording session is running")
            if control_file is None:
                raise RuntimeError("Recording decision channel is unavailable")
            if self._record_decision_pending and action != "stop":
                raise RuntimeError("The previous decision is still being archived")
            was_pending = self._record_decision_pending
            self._record_decision_pending = True
        try:
            # Stop never overwrites a finish/failure decision that the collector
            # may not have consumed yet.  With no pending decision, publish Stop
            # too so the collector has a durable fallback if signal forwarding
            # is delayed.
            if action != "stop" or not was_pending:
                atomic_write_json(control_file, decision)
            os.kill(process.pid, signals[action])
        except Exception as exc:
            with self._lock:
                if self._process is process:
                    self._record_decision_pending = was_pending
            if isinstance(exc, ProcessLookupError):
                raise RuntimeError("Recording process already exited") from exc
            if isinstance(exc, OSError):
                raise RuntimeError(f"Could not send recording decision: {exc}") from exc
            raise
        detail = f" ({decision['failure_label']})" if action == "rerecord" else ""
        self._append_log("INFO", f"Operator requested: {action}{detail}")
        return self.status()

    def stop_job(self) -> dict[str, Any]:
        if not self._lifecycle_lock.acquire(blocking=False):
            raise RuntimeError("Another training workspace start or stop is in progress")
        try:
            with self._lock:
                process = self._process
                kind = self._kind
                if process is None or process.poll() is not None:
                    return self.status()
            if kind == "record":
                return self.record_control({"action": "stop"})
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            self._append_log("INFO", f"Stop requested for {kind}")
            return self.status()
        finally:
            self._lifecycle_lock.release()

    def _wait_for_owned_process_exit(
        self,
        process: subprocess.Popen[str],
        kind: str | None,
        reader: threading.Thread | None,
        *,
        context: str,
    ) -> None:
        """Stop, drain, and retain ownership until a child is truly gone."""

        if process.poll() is not None:
            if reader is not None and reader is not threading.current_thread():
                reader.join()
            return
        try:
            if kind == "record":
                # SIGHUP is a graceful collector stop: finish the current raw
                # buffer, commit or recover its per-attempt archive, then run
                # the follower/leader disconnect sequence.  Never let the GUI
                # exit while this child still owns the serial ports.
                # Serialize with finish/failure file+signal publication so a
                # concurrent shutdown cannot overtake a committed label.
                with self._decision_lock:
                    os.kill(process.pid, signal.SIGHUP)
                self._append_log(
                    "INFO",
                    f"{context} requested; waiting for the active attempt to archive "
                    "and both arms to disconnect",
                )
            else:
                os.killpg(process.pid, signal.SIGINT)
                self._append_log(
                    "INFO", f"{context} requested; waiting for {kind} to exit"
                )
        except ProcessLookupError:
            pass

        waited_s = 0.0
        recovery_interrupt_sent = False
        while process.poll() is None:
            try:
                process.wait(timeout=self._SHUTDOWN_WAIT_SLICE_S)
            except subprocess.TimeoutExpired:
                waited_s += self._SHUTDOWN_WAIT_SLICE_S
                if (
                    kind == "record"
                    and not recovery_interrupt_sent
                    and waited_s >= self._RECORD_ARCHIVE_GRACE_S
                ):
                    # A normal dual-camera encode can take longer than the old
                    # 30 second timeout.  After a generous grace period, SIGINT
                    # reaches the whole collector session (including an active
                    # encoder).  controlled_record catches KeyboardInterrupt,
                    # preserves raw frames if finalization cannot complete, and
                    # still disconnects both arms in its finally block.
                    try:
                        os.killpg(process.pid, signal.SIGINT)
                    except ProcessLookupError:
                        continue
                    recovery_interrupt_sent = True
                    self._append_log(
                        "WARN",
                        "Collector archive exceeded the shutdown grace period; "
                        "requested recovery-safe interruption and will keep the GUI "
                        "alive until the arms disconnect",
                    )
                else:
                    activity = (
                        "collector archive and arm disconnect"
                        if kind == "record"
                        else f"{kind} shutdown"
                    )
                    self._append_log(
                        "INFO",
                        f"Still waiting for {activity} ({int(waited_s)}s); "
                        "the GUI will not orphan the child process",
                    )

        # The log reader owns the final state/manifest transition.  Once the
        # process has exited it should drain immediately, but do not abandon it
        # at interpreter shutdown and lose the final recovery result.
        if reader is not None and reader is not threading.current_thread():
            reader.join()

    def shutdown_cleanup(self) -> None:
        # A signal may arrive while a request thread is still validating or
        # publishing its child.  Wait for that lifecycle transaction first, so
        # shutdown cannot miss a process that is about to be spawned.
        with self._lifecycle_lock:
            with self._lock:
                process = self._process
                kind = self._kind
                reader = self._reader_thread
            if process is None:
                return
            self._wait_for_owned_process_exit(
                process,
                kind,
                reader,
                context="GUI shutdown",
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            finalizing = self._process is not None and not running
            return {
                "state": "FINALIZING" if finalizing else self._state,
                "running": running,
                "finalizing": finalizing,
                "kind": self._kind,
                "pid": self._process.pid if running and self._process else None,
                "runtime_s": (
                    time.monotonic() - self._started_at
                    if running and self._started_at is not None
                    else 0
                ),
                "config": self._config,
                "fault": self._fault,
                "decision_pending": self._record_decision_pending if running and self._kind == "record" else False,
                "record_phase": self._record_phase if running and self._kind == "record" else None,
                "last_exit_code": self._last_exit_code,
                "latest_seq": self._seq,
                "simulate": self.simulate,
            }

    def logs_since(self, since: int) -> list[dict[str, Any]]:
        with self._lock:
            return [entry for entry in self._logs if entry["seq"] > since]
