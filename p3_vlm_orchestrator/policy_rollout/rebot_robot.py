"""Authenticated LeRobot hardware adapter for policy rollout.

All serial, camera, LeRobot, and motor-driver imports are behind the default
backend seam. Importing this module is therefore safe on machines without the
physical runtime installed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from hashlib import sha256
import math
from pathlib import Path
import sys
import time
from typing import Any, Protocol

import numpy as np

from rebot_operator_kit.rollout.contracts import RolloutObservation


JOINT_COUNT = 7
FIRST_LIVE_SPEED_SCALE_MIN = 0.10
FIRST_LIVE_SPEED_SCALE_MAX = 0.20
MAX_RELATIVE_TARGET_DEG = 1.5
ACTION_MATCH_TOLERANCE_DEG = 1e-9
FOLLOWER_RUNTIME_CONTRACTS = (
    "follower_driver_contract",
    "follower_base_implementation",
    "follower_dm_implementation",
)
EXPECTED_CAMERA_KEYS = ("front", "side")
EXPECTED_RECORDING_KEYS = (
    "observation.images.front",
    "observation.images.side",
)


class _Backend(Protocol):
    def make_camera_config(self, **kwargs: object) -> object: ...

    def make_follower_config(self, **kwargs: object) -> object: ...

    def make_follower(self, config: object) -> object: ...


class ReBotPolicyRobot:
    """RobotAdapter over the verified seven-joint LeRobot follower plugin."""

    def __init__(
        self,
        *,
        follower: object,
        joint_names: Sequence[str],
        task: str,
        camera_contract: Mapping[str, Mapping[str, object]],
        calibration_path: Path,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind an already verified plugin; ``from_profile`` performs verification."""

        self.follower = follower
        self.joint_names = tuple(joint_names)
        self.feature_names = tuple(f"{name}.pos" for name in self.joint_names)
        self.task = task
        self.camera_contract = {
            key: dict(value) for key, value in camera_contract.items()
        }
        self.calibration_path = calibration_path
        self.monotonic_clock = monotonic_clock
        self._connected = False

    @classmethod
    def from_profile(
        cls,
        *,
        profile_snapshot: Mapping[str, Any],
        runtime_root: Path | str,
        speed_scale: float,
        monotonic_clock: Callable[[], float] = time.monotonic,
        backend: _Backend | None = None,
        serial_ports_provider: Callable[[], Iterable[object]] | None = None,
    ) -> ReBotPolicyRobot:
        """Validate the authenticated profile and build without connecting."""

        checked_speed = _finite_float(speed_scale, "speed_scale")
        if not (
            FIRST_LIVE_SPEED_SCALE_MIN
            <= checked_speed
            <= FIRST_LIVE_SPEED_SCALE_MAX
        ):
            raise ValueError(
                "speed_scale must be within [0.10, 0.20] for a live-capable follower"
            )
        if not isinstance(profile_snapshot, Mapping):
            raise ValueError("Authenticated profile snapshot must be an object")

        root = Path(runtime_root).expanduser().resolve()
        calibration_section = _required_mapping(
            profile_snapshot, "calibration", "profile"
        )
        follower_profile = _required_mapping(
            calibration_section, "follower", "profile calibration"
        )
        follower_type = _required_text(
            follower_profile.get("type"), "Follower calibration type"
        )
        follower_id = _required_text(
            follower_profile.get("id"), "Follower calibration id"
        )
        calibration_path = _verified_runtime_file(
            root, follower_profile, "Follower calibration"
        )
        for contract_name in FOLLOWER_RUNTIME_CONTRACTS:
            entry = _required_mapping(
                calibration_section, contract_name, "profile calibration"
            )
            _verified_runtime_file(root, entry, contract_name)

        coordinates = _required_mapping(
            profile_snapshot, "coordinate_contract", "profile"
        )
        joints = _profile_joints(coordinates)
        joint_names = tuple(joint["name"] for joint in joints)
        expected_directions = {
            joint["name"]: joint["direction"] for joint in joints
        }
        expected_limits = {joint["name"]: joint["limits"] for joint in joints}
        expected_features = tuple(joint["feature"] for joint in joints)
        if expected_features != tuple(f"{name}.pos" for name in joint_names):
            raise ValueError("Profile joint features must match the seven physical joints")

        calibration_values = _load_calibration(calibration_path, joint_names)
        camera_contract = _camera_contract(profile_snapshot)
        task = _required_text(
            _required_mapping(
                profile_snapshot, "collection_defaults", "profile"
            ).get("task"),
            "Authenticated rollout task",
        )
        hardware = _required_mapping(
            profile_snapshot, "hardware_identity", "profile"
        )
        follower_usb = _required_mapping(
            hardware, "follower_usb", "profile hardware identity"
        )
        expected_vid = _usb_id(follower_usb.get("vid"), "follower USB VID")
        expected_pid = _usb_id(follower_usb.get("pid"), "follower USB PID")

        if serial_ports_provider is None:
            serial_ports_provider = _default_serial_ports
        try:
            ports = list(serial_ports_provider())
        except Exception as exc:
            raise ValueError(f"Follower USB discovery failed: {exc}") from exc
        matches = [
            port
            for port in ports
            if getattr(port, "vid", None) == expected_vid
            and getattr(port, "pid", None) == expected_pid
        ]
        if len(matches) != 1:
            devices = [str(getattr(port, "device", "<unknown>")) for port in matches]
            raise ValueError(
                "Follower USB discovery expected exactly one authenticated device; "
                f"found {devices}"
            )
        device = getattr(matches[0], "device", None)
        if not isinstance(device, str) or not device:
            raise ValueError("Authenticated follower USB device has no usable port")

        checked_backend: _Backend = backend or _DefaultBackend(root)
        cameras = {
            key: checked_backend.make_camera_config(
                index_or_path=contract["index"],
                fps=contract["fps"],
                width=contract["width"],
                height=contract["height"],
                fourcc="MJPG",
            )
            for key, contract in camera_contract.items()
        }
        collection = _required_mapping(
            profile_snapshot, "collection_defaults", "profile"
        )
        base_velocity = _positive_finite(
            collection.get("motor_velocity", 2000.0),
            "Profile motor velocity",
        )
        gripper_force = _finite_float(
            collection.get("gripper_force", 0.05),
            "Profile gripper force",
        )
        if not 0.0 <= gripper_force <= 1.0:
            raise ValueError("Profile gripper force must be in [0, 1]")

        expected_velocity = [base_velocity * checked_speed] * JOINT_COUNT
        follower_config = checked_backend.make_follower_config(
            port=device,
            id=follower_id,
            calibration_dir=calibration_path.parent,
            can_adapter="damiao",
            dm_serial_baud=921600,
            max_relative_target=MAX_RELATIVE_TARGET_DEG,
            pos_vel_velocity=expected_velocity,
            force_pos_torque_ration=gripper_force,
            disable_torque_on_disconnect=True,
            cameras=cameras,
        )
        follower = checked_backend.make_follower(follower_config)
        _verify_plugin_binding(
            follower=follower,
            follower_type=follower_type,
            joint_names=joint_names,
            expected_directions=expected_directions,
            expected_limits=expected_limits,
            camera_contract=camera_contract,
            calibration_path=calibration_path,
            calibration_values=calibration_values,
            expected_velocity=expected_velocity,
        )

        return cls(
            follower=follower,
            joint_names=joint_names,
            task=task,
            camera_contract=camera_contract,
            calibration_path=calibration_path,
            monotonic_clock=monotonic_clock,
        )

    def connect(self) -> None:
        if self._connected:
            raise RuntimeError("ReBot policy follower is already connected")
        try:
            self.follower.connect(calibrate=False)
        except TypeError as exc:
            raise RuntimeError(
                "Follower plugin does not support fail-closed calibrate=False connection"
            ) from exc
        if not bool(getattr(self.follower, "is_connected", False)):
            raise RuntimeError("Follower plugin did not report a connected state")
        self._connected = True

    def disconnect(self) -> None:
        if not self._connected and not bool(
            getattr(self.follower, "is_connected", False)
        ):
            return
        try:
            self.follower.disconnect()
        finally:
            self._connected = False

    def observe(self) -> RolloutObservation:
        self._require_connected()
        try:
            raw = self.follower.get_observation()
        except Exception:
            raise
        if not isinstance(raw, Mapping):
            raise ValueError("Follower observation must be a mapping")

        images: dict[str, np.ndarray] = {}
        for key in EXPECTED_CAMERA_KEYS:
            if key not in raw:
                raise ValueError(f"Follower observation is missing camera {key!r}")
            try:
                image = np.asarray(raw[key])
            except Exception as exc:
                raise ValueError(f"Follower observation camera {key!r} is malformed") from exc
            contract = self.camera_contract[key]
            expected_shape = (
                int(contract["height"]),
                int(contract["width"]),
                3,
            )
            if image.shape != expected_shape or not np.issubdtype(
                image.dtype, np.number
            ):
                raise ValueError(
                    f"Follower observation camera {key!r} does not match {expected_shape}"
                )
            images[key] = image.copy()

        try:
            state = np.asarray([raw[name] for name in self.feature_names], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Follower observation is missing or has malformed physical joint positions"
            ) from exc
        if state.shape != (JOINT_COUNT,) or not np.all(np.isfinite(state)):
            raise ValueError(
                "Follower observation physical joint positions must be seven finite values"
            )
        captured_monotonic_s = _finite_float(
            self.monotonic_clock(),
            "Follower observation capture monotonic time",
        )
        return RolloutObservation(
            front=images["front"],
            side=images["side"],
            state_deg=state.copy(),
            task=self.task,
            captured_monotonic_s=captured_monotonic_s,
        )

    def send_action(self, action_deg: np.ndarray) -> np.ndarray:
        """Invert plugin scales so its returned action is requested physical space."""

        self._require_connected()
        try:
            requested = np.asarray(action_deg, dtype=float).copy()
        except (TypeError, ValueError) as exc:
            raise ValueError("Physical follower action must be seven finite values") from exc
        if requested.shape != (JOINT_COUNT,) or not np.all(np.isfinite(requested)):
            raise ValueError("Physical follower action must be seven finite values")

        directions = getattr(self.follower.config, "joint_directions", None)
        limits = getattr(self.follower.config, "joint_limits", None)
        if not isinstance(directions, Mapping):
            raise ValueError("Follower plugin direction mapping is missing")
        if not isinstance(limits, Mapping):
            raise ValueError("Follower plugin joint limits are missing")
        command: dict[str, float] = {}
        for index, (joint_name, feature_name) in enumerate(
            zip(self.joint_names, self.feature_names, strict=True)
        ):
            direction = _finite_float(
                directions.get(joint_name),
                f"Follower plugin direction for {joint_name}",
            )
            if direction == 0.0:
                raise ValueError(
                    f"Follower plugin direction for {joint_name} must be nonzero"
                )
            joint_limit = limits.get(joint_name)
            try:
                low, high = (float(value) for value in joint_limit)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Follower plugin limit for {joint_name} is malformed"
                ) from exc
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(
                    f"Follower plugin limit for {joint_name} is invalid"
                )
            if requested[index] < low or requested[index] > high:
                raise ValueError(
                    f"Physical follower action for {joint_name} is outside plugin limits"
                )
            driver_value = float(requested[index]) / direction
            if not math.isfinite(driver_value):
                raise ValueError(
                    f"Physical follower action for {joint_name} cannot be safely inverted"
                )
            command[feature_name] = driver_value

        returned_raw = self.follower.send_action(command)
        if not isinstance(returned_raw, Mapping) or set(returned_raw) != set(
            self.feature_names
        ):
            raise ValueError(
                "Follower plugin returned action must contain exactly seven joint keys"
            )
        try:
            returned = np.asarray(
                [returned_raw[name] for name in self.feature_names], dtype=float
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Follower plugin returned action is malformed") from exc
        if returned.shape != (JOINT_COUNT,) or not np.all(np.isfinite(returned)):
            raise ValueError("Follower plugin returned action must be seven finite values")
        if not np.allclose(
            returned,
            requested,
            rtol=0.0,
            atol=ACTION_MATCH_TOLERANCE_DEG,
        ):
            raise ValueError(
                "Follower plugin returned action disagrees with requested physical action"
            )
        return returned.copy()

    def _require_connected(self) -> None:
        if not self._connected or not bool(
            getattr(self.follower, "is_connected", False)
        ):
            raise RuntimeError("ReBot policy follower is disconnected")


class _DefaultBackend:
    """Lazy import seam for the real LeRobot follower and OpenCV camera config."""

    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = runtime_root

    def _ensure_runtime_paths(self) -> None:
        for path in (
            self.runtime_root / "lerobot-robot-seeed-b601",
            self.runtime_root / "lerobot" / "src",
        ):
            resolved = str(path.resolve())
            if path.is_dir() and resolved not in sys.path:
                sys.path.insert(0, resolved)

    def make_camera_config(self, **kwargs: object) -> object:
        self._ensure_runtime_paths()
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        return OpenCVCameraConfig(**kwargs)

    def make_follower_config(self, **kwargs: object) -> object:
        self._ensure_runtime_paths()
        from lerobot_robot_seeed_b601 import SeeedB601DMFollowerConfig

        return SeeedB601DMFollowerConfig(**kwargs)

    def make_follower(self, config: object) -> object:
        self._ensure_runtime_paths()
        from lerobot_robot_seeed_b601 import SeeedB601DMFollower

        return SeeedB601DMFollower(config)


def _default_serial_ports() -> Iterable[object]:
    from serial.tools import list_ports

    return list_ports.comports()


def _required_mapping(
    parent: Mapping[str, Any], key: str, label: str
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Authenticated {label} {key} must be an object")
    return value


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty")
    return value


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_finite(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _usb_id(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFF:
        raise ValueError(f"Authenticated {label} is invalid")
    return value


def _verified_runtime_file(
    runtime_root: Path,
    entry: Mapping[str, Any],
    label: str,
) -> Path:
    relative = entry.get("runtime_relative_path")
    expected_digest = entry.get("sha256")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} runtime path is invalid")
    path = (runtime_root / relative).resolve()
    try:
        path.relative_to(runtime_root)
    except ValueError as exc:
        raise ValueError(f"{label} runtime path escapes the authenticated root") from exc
    if not path.is_file():
        raise ValueError(f"{label} fingerprint cannot be verified; file is missing")
    actual_digest = sha256(path.read_bytes()).hexdigest()
    if not isinstance(expected_digest, str) or actual_digest != expected_digest:
        raise ValueError(f"{label} fingerprint does not match the authenticated profile")
    return path


def _profile_joints(coordinates: Mapping[str, Any]) -> list[dict[str, Any]]:
    if coordinates.get("action_dimension") != JOINT_COUNT:
        raise ValueError("Profile action dimension must be seven")
    raw_joints = coordinates.get("joints")
    if not isinstance(raw_joints, list) or len(raw_joints) != JOINT_COUNT:
        raise ValueError("Profile must define exactly seven physical follower joints")
    joints: list[dict[str, Any]] = []
    for raw in raw_joints:
        if not isinstance(raw, Mapping):
            raise ValueError("Profile physical follower joint entry is malformed")
        name = _required_text(raw.get("name"), "Profile joint name")
        feature = _required_text(raw.get("feature"), f"Profile feature for {name}")
        direction = _finite_float(
            raw.get("leader_to_follower_scale"),
            f"Profile direction for {name}",
        )
        if direction == 0.0:
            raise ValueError(f"Profile direction for {name} must be nonzero")
        raw_limits = raw.get("soft_limit_degrees")
        try:
            low, high = (float(value) for value in raw_limits)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Profile limits for {name} are malformed") from exc
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError(f"Profile limits for {name} are invalid")
        joints.append(
            {
                "name": name,
                "feature": feature,
                "direction": direction,
                "limits": (low, high),
            }
        )
    names = [joint["name"] for joint in joints]
    if len(set(names)) != JOINT_COUNT:
        raise ValueError("Profile physical follower joint names must be unique")
    return joints


def _load_calibration(
    path: Path, joint_names: Sequence[str]
) -> Mapping[str, object]:
    import json

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Follower calibration is not valid JSON") from exc
    if not isinstance(value, dict) or list(value) != list(joint_names):
        raise ValueError(
            "Follower calibration joint order does not match the authenticated profile"
        )
    for joint_name in joint_names:
        entry = value[joint_name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Follower calibration entry {joint_name} is malformed")
        for field in ("id", "drive_mode", "homing_offset", "range_min", "range_max"):
            if isinstance(entry.get(field), bool):
                raise ValueError(
                    f"Follower calibration {joint_name}.{field} must not be boolean"
                )
            _finite_float(entry.get(field), f"Follower calibration {joint_name}.{field}")
    return value


def _camera_contract(
    profile: Mapping[str, Any],
) -> dict[str, dict[str, int | str]]:
    cameras = _required_mapping(profile, "camera_defaults", "profile")
    if tuple(cameras)[:2] != EXPECTED_CAMERA_KEYS or set(cameras).intersection(
        EXPECTED_CAMERA_KEYS
    ) != set(EXPECTED_CAMERA_KEYS):
        raise ValueError("Profile camera contract must contain front then side")
    result: dict[str, dict[str, int | str]] = {}
    for key, expected_recording_key in zip(
        EXPECTED_CAMERA_KEYS, EXPECTED_RECORDING_KEYS, strict=True
    ):
        raw = cameras.get(key)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Profile camera {key} must be an object")
        if raw.get("recording_key") != expected_recording_key:
            raise ValueError(f"Profile camera {key} recording key is invalid")
        index = raw.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"Profile camera {key} index is invalid")
        result[key] = {
            "recording_key": expected_recording_key,
            "index": index,
            "width": _positive_integer(raw.get("width"), f"{key} camera width"),
            "height": _positive_integer(raw.get("height"), f"{key} camera height"),
            "fps": _positive_integer(raw.get("fps"), f"{key} camera FPS"),
        }
    return result


def _verify_plugin_binding(
    *,
    follower: object,
    follower_type: str,
    joint_names: Sequence[str],
    expected_directions: Mapping[str, float],
    expected_limits: Mapping[str, tuple[float, float]],
    camera_contract: Mapping[str, Mapping[str, object]],
    calibration_path: Path,
    calibration_values: Mapping[str, object],
    expected_velocity: Sequence[float],
) -> None:
    if getattr(follower, "name", None) != follower_type:
        raise ValueError("Instantiated follower plugin type does not match the profile")
    if list(getattr(follower, "motor_names", ())) != list(joint_names):
        raise ValueError("Instantiated follower plugin joint order does not match the profile")

    plugin_config = getattr(follower, "config", None)
    actual_directions_raw = getattr(plugin_config, "joint_directions", None)
    actual_limits_raw = getattr(plugin_config, "joint_limits", None)
    if not isinstance(actual_directions_raw, Mapping) or not isinstance(
        actual_limits_raw, Mapping
    ):
        raise ValueError("Instantiated follower plugin direction or limit contract is missing")
    actual_directions = {
        name: _finite_float(value, f"Follower plugin direction for {name}")
        for name, value in actual_directions_raw.items()
    }
    try:
        actual_limits = {
            name: tuple(float(value) for value in values)
            for name, values in actual_limits_raw.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("Instantiated follower plugin limits are malformed") from exc
    if actual_directions != dict(expected_directions):
        raise ValueError("Instantiated follower plugin directions do not match the profile")
    if actual_limits != dict(expected_limits):
        raise ValueError("Instantiated follower plugin limits do not match the profile")
    try:
        actual_velocity = np.asarray(
            getattr(plugin_config, "pos_vel_velocity"), dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Instantiated follower plugin velocity is malformed") from exc
    if (
        actual_velocity.shape != (JOINT_COUNT,)
        or not np.all(np.isfinite(actual_velocity))
        or not np.array_equal(actual_velocity, np.asarray(expected_velocity, dtype=float))
    ):
        raise ValueError("Instantiated follower plugin velocity does not match the profile")

    actual_calibration = Path(
        getattr(follower, "calibration_fpath", "")
    ).expanduser().resolve()
    if actual_calibration != calibration_path:
        raise ValueError("Follower plugin calibration path does not match the profile")
    loaded_calibration = getattr(follower, "calibration", None)
    if not isinstance(loaded_calibration, Mapping) or list(loaded_calibration) != list(
        calibration_values
    ):
        raise ValueError("Follower plugin calibration did not load the verified joint order")

    follower_cameras = getattr(follower, "cameras", None)
    config_cameras = getattr(plugin_config, "cameras", None)
    if not isinstance(follower_cameras, Mapping) or list(follower_cameras) != list(
        EXPECTED_CAMERA_KEYS
    ):
        raise ValueError("Instantiated follower plugin camera keys do not match the profile")
    if not isinstance(config_cameras, Mapping) or list(config_cameras) != list(
        EXPECTED_CAMERA_KEYS
    ):
        raise ValueError("Follower plugin camera config does not match the profile")
    for key in EXPECTED_CAMERA_KEYS:
        actual = config_cameras[key]
        expected = camera_contract[key]
        for field, actual_field in (
            ("index", "index_or_path"),
            ("width", "width"),
            ("height", "height"),
            ("fps", "fps"),
        ):
            if getattr(actual, actual_field, None) != expected[field]:
                raise ValueError(
                    f"Follower plugin camera {key} {field} does not match the profile"
                )
    if bool(getattr(follower, "is_connected", False)):
        raise ValueError("Follower plugin connected during construction validation")


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value
