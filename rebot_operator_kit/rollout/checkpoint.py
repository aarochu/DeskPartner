"""Dependency-free validation of the Person 3 checkpoint handoff."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


PROFILE_FILENAME = "rebot_training_profile.json"
REQUIRED_CONFIG_FILENAMES = (
    "config.json",
    "preprocessor_config.json",
    "postprocessor_config.json",
)
EXPECTED_IMAGE_ORDER = (
    "observation.images.front",
    "observation.images.side",
)


class CheckpointError(ValueError):
    """Raised when a checkpoint handoff does not satisfy the rollout contract."""


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _required_object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise CheckpointError(f"Checkpoint profile {key} must be an object")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CheckpointError(f"Checkpoint {label} must be a positive integer")
    return value


@dataclass(frozen=True)
class CheckpointBundle:
    path: Path
    task: str
    action_dimension: int
    chunk_size: int
    action_steps: int
    image_order: tuple[str, str]
    profile_digest: str
    profile_snapshot: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> CheckpointBundle:
        checkpoint = Path(path).expanduser().resolve()
        if not checkpoint.is_dir():
            raise CheckpointError(f"Checkpoint directory is missing: {checkpoint}")

        for filename in REQUIRED_CONFIG_FILENAMES:
            if not (checkpoint / filename).is_file():
                raise CheckpointError(f"Checkpoint {filename} is missing")

        if not (checkpoint / "model.safetensors").is_file():
            raise CheckpointError("Checkpoint model weights are missing")

        profile_path = checkpoint / PROFILE_FILENAME
        if not profile_path.is_file():
            raise CheckpointError(f"Checkpoint {PROFILE_FILENAME} is missing")
        try:
            sidecar = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CheckpointError(
                f"Checkpoint {PROFILE_FILENAME} cannot be read: {exc}"
            ) from exc
        if not isinstance(sidecar, dict):
            raise CheckpointError("Checkpoint profile sidecar must be an object")

        profile = _required_object(sidecar, "profile_snapshot")
        declared_profile_digest = sidecar.get("training_profile_digest")
        calculated_profile_digest = _canonical_digest(profile)
        if declared_profile_digest != calculated_profile_digest:
            raise CheckpointError("Checkpoint profile digest does not match its snapshot")

        collection_contract = _required_object(sidecar, "collection_contract")
        declared_collection_digest = sidecar.get("collection_contract_digest")
        if declared_collection_digest != _canonical_digest(collection_contract):
            raise CheckpointError(
                "Checkpoint collection contract digest does not match its snapshot"
            )

        collection_defaults = _required_object(profile, "collection_defaults")
        locked_task = collection_defaults.get("task")
        dataset_task = collection_contract.get("task")
        if (
            not isinstance(locked_task, str)
            or not locked_task.strip()
            or not isinstance(dataset_task, str)
            or not dataset_task.strip()
        ):
            raise CheckpointError("Checkpoint task must be nonempty")
        if locked_task != dataset_task:
            raise CheckpointError(
                "Checkpoint collection task does not match the locked profile task"
            )

        coordinates = _required_object(profile, "coordinate_contract")
        training = _required_object(profile, "training_defaults")
        coordinate_dimension = coordinates.get("action_dimension")
        training_dimension = training.get("action_dimension")
        if (
            isinstance(coordinate_dimension, bool)
            or not isinstance(coordinate_dimension, int)
            or coordinate_dimension != 7
            or isinstance(training_dimension, bool)
            or not isinstance(training_dimension, int)
            or training_dimension != 7
        ):
            raise CheckpointError("Checkpoint action dimension must be 7")

        chunk_size = _positive_integer(training.get("chunk_size"), "chunk size")
        action_steps = _positive_integer(
            training.get("n_action_steps"), "action steps"
        )
        image_order = training.get("image_order")
        if image_order != list(EXPECTED_IMAGE_ORDER):
            raise CheckpointError("Checkpoint image order must be front then side")

        return cls(
            path=checkpoint,
            task=locked_task,
            action_dimension=training_dimension,
            chunk_size=chunk_size,
            action_steps=action_steps,
            image_order=EXPECTED_IMAGE_ORDER,
            profile_digest=calculated_profile_digest,
            profile_snapshot=profile,
        )
