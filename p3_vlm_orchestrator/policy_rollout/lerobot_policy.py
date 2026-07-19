"""Lazy, policy-generic adapter for saved LeRobot checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import platform
from threading import Lock
from typing import Any

import numpy as np

from rebot_operator_kit.rollout.checkpoint import CheckpointBundle
from rebot_operator_kit.rollout.contracts import RolloutObservation


class LeRobotCompatibilityError(RuntimeError):
    """The installed LeRobot runtime cannot read a checkpoint artifact."""


_PLUGIN_REGISTRATION_LOCK = Lock()


def _register_policy_plugins(registrar: Callable[[], None]) -> None:
    """Run LeRobot's registrar without importing offline-irrelevant hardware."""

    import importlib.metadata

    with _PLUGIN_REGISTRATION_LOCK:
        discover = importlib.metadata.distributions
        policy_distributions = tuple(
            distribution
            for distribution in discover()
            if isinstance(distribution.metadata.get("Name"), str)
            and distribution.metadata["Name"].replace("-", "_").startswith(
                "lerobot_policy_"
            )
        )

        def discover_policies(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
            return policy_distributions

        importlib.metadata.distributions = discover_policies
        try:
            registrar()
        finally:
            importlib.metadata.distributions = discover


@dataclass(frozen=True)
class _LeRobotAPI:
    """Small injectable boundary around LeRobot's versioned public APIs."""

    config_from_pretrained: Callable[..., Any]
    get_policy_class: Callable[[str], type]
    make_pre_post_processors: Callable[..., tuple[Any, Any]]
    prepare_observation: Callable[[dict[str, np.ndarray], str, str], dict[str, Any]]
    inference_mode: Callable[[], Any]
    runtime_description: str


@dataclass(frozen=True)
class _LoadedBackend:
    config: Any
    policy: Any
    preprocessor: Any
    postprocessor: Any
    prepare_observation: Callable[[dict[str, np.ndarray], str, str], dict[str, Any]]
    inference_mode: Callable[[], Any]


def _import_lerobot_api() -> _LeRobotAPI:
    """Import all optional inference dependencies at the loading boundary."""

    try:
        import importlib.metadata

        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import (
            get_policy_class,
            make_pre_post_processors,
        )
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot.utils.import_utils import register_third_party_plugins

        _register_policy_plugins(register_third_party_plugins)
    except Exception as exc:
        raise LeRobotCompatibilityError(
            "LeRobot inference dependencies are unavailable. Install the same "
            "LeRobot checkout and policy plugins used for training. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    runtime = f"LeRobot {version} on Python {platform.python_version()}"

    return _LeRobotAPI(
        config_from_pretrained=PreTrainedConfig.from_pretrained,
        get_policy_class=get_policy_class,
        make_pre_post_processors=make_pre_post_processors,
        prepare_observation=prepare_observation_for_inference,
        inference_mode=torch.inference_mode,
        runtime_description=runtime,
    )


def _compatibility_error(
    *, stage: str, checkpoint: Path, api: _LeRobotAPI, error: Exception
) -> LeRobotCompatibilityError:
    return LeRobotCompatibilityError(
        f"Checkpoint compatibility error during {stage}: {api.runtime_description} "
        f"cannot read {checkpoint}. Use the same official LeRobot/Python environment "
        f"that produced this checkpoint. Original error: "
        f"{type(error).__name__}: {error}"
    )


def _saved_preprocessor_device_overrides(
    checkpoint: Path, device: str
) -> dict[str, dict[str, str]]:
    """Retarget saved processor steps that explicitly carry a device setting."""

    config_path = checkpoint / "preprocessor_config.json"
    if not config_path.is_file():
        return {}
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(document, dict) or not isinstance(document.get("steps"), list):
        return {}

    overrides: dict[str, dict[str, str]] = {}
    for entry in document["steps"]:
        if not isinstance(entry, dict):
            continue
        saved_config = entry.get("config")
        if not isinstance(saved_config, dict) or "device" not in saved_config:
            continue
        key = entry.get("registry_name")
        if not isinstance(key, str):
            class_path = entry.get("class")
            if isinstance(class_path, str):
                key = class_path.rsplit(".", 1)[-1]
        if isinstance(key, str):
            overrides[key] = {"device": device}
    return overrides


def _validate_local_processor_artifacts(checkpoint: Path) -> None:
    """Reject incomplete local pipelines before LeRobot can try a Hub fallback."""

    checkpoint_root = checkpoint.resolve()
    for filename in ("preprocessor_config.json", "postprocessor_config.json"):
        config_path = checkpoint / filename
        if not config_path.is_file():
            continue
        try:
            document = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"saved processor config {filename} is unreadable: {exc}") from exc
        if not isinstance(document, dict) or not isinstance(document.get("steps"), list):
            raise ValueError(f"saved processor config {filename} has an invalid steps schema")
        for index, entry in enumerate(document["steps"]):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"saved processor config {filename} step {index} must be an object"
                )
            state_file = entry.get("state_file")
            if state_file is None:
                continue
            if not isinstance(state_file, str) or not state_file:
                raise ValueError(
                    f"saved processor config {filename} step {index} has an invalid state_file"
                )
            state_path = (checkpoint / state_file).resolve()
            if not state_path.is_relative_to(checkpoint_root):
                raise ValueError(
                    f"saved processor state file must stay inside checkpoint: {state_file}"
                )
            if not state_path.is_file():
                raise ValueError(
                    f"saved processor state file is missing: {state_file}"
                )


def _reset_if_supported(component: Any) -> None:
    reset = getattr(component, "reset", None)
    if callable(reset):
        reset()


def _load_lerobot_backend(checkpoint: Path, device: str) -> _LoadedBackend:
    api = _import_lerobot_api()
    checkpoint_text = str(checkpoint)

    try:
        config = api.config_from_pretrained(
            checkpoint_text,
            local_files_only=True,
        )
    except Exception as exc:
        raise _compatibility_error(
            stage="policy configuration loading",
            checkpoint=checkpoint,
            api=api,
            error=exc,
        ) from exc

    try:
        config.device = device
        policy_class = api.get_policy_class(config.type)
    except Exception as exc:
        raise _compatibility_error(
            stage=f"registered policy {getattr(config, 'type', None)!r} resolution",
            checkpoint=checkpoint,
            api=api,
            error=exc,
        ) from exc

    try:
        policy = policy_class.from_pretrained(
            checkpoint_text,
            config=config,
            local_files_only=True,
        )
        to_device = getattr(policy, "to", None)
        if callable(to_device):
            to_device(device)
        evaluate = getattr(policy, "eval", None)
        if callable(evaluate):
            evaluate()
        _reset_if_supported(policy)
    except Exception as exc:
        raise _compatibility_error(
            stage="policy weight loading",
            checkpoint=checkpoint,
            api=api,
            error=exc,
        ) from exc

    processor_kwargs: dict[str, Any] = {
        "policy_cfg": config,
        "pretrained_path": checkpoint_text,
        "preprocessor_config_filename": "preprocessor_config.json",
        "postprocessor_config_filename": "postprocessor_config.json",
    }
    device_overrides = _saved_preprocessor_device_overrides(checkpoint, device)
    if device_overrides:
        processor_kwargs["preprocessor_overrides"] = device_overrides
    try:
        _validate_local_processor_artifacts(checkpoint)
        preprocessor, postprocessor = api.make_pre_post_processors(
            **processor_kwargs
        )
        _reset_if_supported(preprocessor)
        _reset_if_supported(postprocessor)
    except Exception as exc:
        raise _compatibility_error(
            stage=(
                "saved processor schema loading "
                "(preprocessor_config.json and postprocessor_config.json)"
            ),
            checkpoint=checkpoint,
            api=api,
            error=exc,
        ) from exc

    return _LoadedBackend(
        config=config,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        prepare_observation=api.prepare_observation,
        inference_mode=api.inference_mode,
    )


class LeRobotPolicyAdapter:
    """Implement the rollout PolicyAdapter protocol for a LeRobot policy."""

    def __init__(
        self,
        *,
        bundle: CheckpointBundle,
        device: str,
        backend: _LoadedBackend,
    ) -> None:
        self.bundle = bundle
        self.device = device
        self.config = backend.config
        self.policy = backend.policy
        self.preprocessor = backend.preprocessor
        self.postprocessor = backend.postprocessor
        self._prepare_observation = backend.prepare_observation
        self._inference_mode = backend.inference_mode

    @classmethod
    def from_checkpoint(
        cls, bundle: CheckpointBundle, device: str
    ) -> LeRobotPolicyAdapter:
        backend = _load_lerobot_backend(bundle.path, device)
        return cls(bundle=bundle, device=device, backend=backend)

    def predict(self, observation: RolloutObservation) -> np.ndarray:
        state = np.asarray(observation.state_deg)
        expected_state_shape = (self.bundle.action_dimension,)
        if state.shape != expected_state_shape:
            raise ValueError(
                "LeRobot observation state must have shape "
                f"{expected_state_shape}; received {state.shape}"
            )

        raw_observation = {
            "observation.images.front": observation.front,
            "observation.images.side": observation.side,
            "observation.state": observation.state_deg,
        }
        with self._inference_mode():
            prepared = self._prepare_observation(
                raw_observation,
                self.device,
                self.bundle.task,
            )
            processed_observation = self.preprocessor(prepared)
            action_chunk = self.policy.predict_action_chunk(processed_observation)
            postprocessed = self.postprocessor(action_chunk)

        value = postprocessed
        for method_name in ("detach", "cpu"):
            method = getattr(value, method_name, None)
            if callable(method):
                value = method()
        to_numpy = getattr(value, "numpy", None)
        if callable(to_numpy):
            value = to_numpy()
        try:
            array = np.array(value, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "LeRobot postprocessed action chunk must be numeric"
            ) from exc

        if array.ndim == 3:
            if array.shape[0] != 1:
                raise ValueError(
                    "LeRobot postprocessed action chunk batch dimension must be 1; "
                    f"received {array.shape}"
                )
            array = array[0]
        if array.ndim != 2 or array.shape[1] != self.bundle.action_dimension:
            raise ValueError(
                "LeRobot postprocessed action chunk shape must be [steps, 7]; "
                f"received {array.shape}"
            )
        if array.shape[0] < 1:
            raise ValueError(
                "LeRobot postprocessed action chunk must contain at least one step"
            )
        if not np.isfinite(array).all():
            raise ValueError(
                "LeRobot postprocessed action chunk contains nonfinite values"
            )
        return np.array(array, dtype=np.float64, copy=True)
