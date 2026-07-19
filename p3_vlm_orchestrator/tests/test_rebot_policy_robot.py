from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from p3_vlm_orchestrator.policy_rollout.rebot_robot import ReBotPolicyRobot


JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)
DIRECTIONS = (-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -6.0)
LIMITS = (
    (-145.0, 145.0),
    (-170.0, 0.0),
    (-200.0, 0.0),
    (-80.0, 90.0),
    (-90.0, 90.0),
    (-90.0, 90.0),
    (-270.0, 0.0),
)
TASK = "Pick up one can and place it in the taped sorting zone"


class FakeFollower:
    name = "seeed_b601_dm_follower"

    def __init__(self, config, *, motor_order=JOINTS, camera_keys=("front", "side")):
        self.config = config
        self.motor_names = list(motor_order)
        self.cameras = {key: config.cameras[key] for key in camera_keys}
        self.calibration_fpath = config.calibration_dir / f"{config.id}.json"
        self.calibration = json.loads(self.calibration_fpath.read_text())
        self.is_connected = False
        self.connect_count = 0
        self.disconnect_count = 0
        self.last_command = None
        self.observation = {
            "front": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
            "side": np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
            **{f"{name}.pos": float(index) for index, name in enumerate(JOINTS)},
        }
        self.return_override = None

    def connect(self, *, calibrate=True):
        self.connect_count += 1
        self.connect_calibrate = calibrate
        self.is_connected = True

    def disconnect(self):
        self.disconnect_count += 1
        self.is_connected = False

    def get_observation(self):
        if not self.is_connected:
            raise RuntimeError("disconnected")
        return self.observation

    def send_action(self, command):
        if not self.is_connected:
            raise RuntimeError("disconnected")
        self.last_command = dict(command)
        if self.return_override is not None:
            return self.return_override
        return {
            f"{name}.pos": command[f"{name}.pos"]
            * self.config.joint_directions[name]
            for name in JOINTS
        }


class FakeBackend:
    def __init__(self):
        self.direction_override = None
        self.limit_override = None
        self.motor_order = JOINTS
        self.camera_keys = ("front", "side")
        self.calibration_path_override = None
        self.velocity_override = None
        self.follower_direction_override = None
        self.follower = None
        self.config_kwargs = None

    def make_camera_config(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def make_follower_config(self, **kwargs):
        self.config_kwargs = kwargs
        directions = dict(zip(JOINTS, DIRECTIONS, strict=True))
        limits = dict(zip(JOINTS, LIMITS, strict=True))
        if self.direction_override is not None:
            directions = self.direction_override
        if self.limit_override is not None:
            limits = self.limit_override
        config = SimpleNamespace(
            **kwargs,
            motor_can_ids={name: (index + 1, index + 17) for index, name in enumerate(JOINTS)},
            joint_directions=directions,
            joint_limits=limits,
        )
        if self.velocity_override is not None:
            config.pos_vel_velocity = self.velocity_override
        return config

    def make_follower(self, config):
        follower = FakeFollower(
            config,
            motor_order=self.motor_order,
            camera_keys=self.camera_keys,
        )
        if self.calibration_path_override is not None:
            follower.calibration_fpath = self.calibration_path_override
        if self.follower_direction_override is not None:
            follower.config = SimpleNamespace(**vars(config))
            follower.config.joint_directions = self.follower_direction_override
        self.follower = follower
        return follower


class ReBotPolicyRobotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.runtime_root = Path(self.tempdir.name)
        self.backend = FakeBackend()
        self.profile = self._make_profile()
        self.ports = [SimpleNamespace(device="/dev/fake-follower", vid=0x2E88, pid=0x4603)]

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_runtime_file(self, relative_path: str, payload: bytes) -> str:
        path = self.runtime_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return sha256(payload).hexdigest()

    def _make_profile(self) -> dict:
        calibration_relative = (
            "lerobot-home/calibration/robots/seeed_b601_dm_follower/follower1.json"
        )
        calibration = {
            name: {
                "id": index + 1,
                "drive_mode": 0,
                "homing_offset": 0.0,
                "range_min": -180.0,
                "range_max": 180.0,
            }
            for index, name in enumerate(JOINTS)
        }
        calibration_digest = self._write_runtime_file(
            calibration_relative,
            json.dumps(calibration).encode(),
        )
        runtime_entries = {}
        for key, relative in (
            ("follower_driver_contract", "driver/config.py"),
            ("follower_base_implementation", "driver/base.py"),
            ("follower_dm_implementation", "driver/dm.py"),
        ):
            runtime_entries[key] = {
                "runtime_relative_path": relative,
                "sha256": self._write_runtime_file(relative, key.encode()),
            }
        return {
            "calibration": {
                "follower": {
                    "type": "seeed_b601_dm_follower",
                    "id": "follower1",
                    "runtime_relative_path": calibration_relative,
                    "sha256": calibration_digest,
                },
                **runtime_entries,
            },
            "hardware_identity": {"follower_usb": {"vid": 0x2E88, "pid": 0x4603}},
            "coordinate_contract": {
                "action_dimension": 7,
                "joints": [
                    {
                        "name": name,
                        "feature": f"{name}.pos",
                        "leader_to_follower_scale": direction,
                        "soft_limit_degrees": list(limit),
                    }
                    for name, direction, limit in zip(JOINTS, DIRECTIONS, LIMITS, strict=True)
                ],
            },
            "collection_defaults": {
                "task": TASK,
                "motor_velocity": 2000.0,
                "gripper_force": 0.05,
            },
            "camera_defaults": {
                "front": {
                    "recording_key": "observation.images.front",
                    "index": 0,
                    "width": 2,
                    "height": 2,
                    "fps": 30,
                },
                "side": {
                    "recording_key": "observation.images.side",
                    "index": 1,
                    "width": 4,
                    "height": 3,
                    "fps": 30,
                },
            },
        }

    def _robot(self, **kwargs) -> ReBotPolicyRobot:
        return ReBotPolicyRobot.from_profile(
            profile_snapshot=self.profile,
            runtime_root=self.runtime_root,
            speed_scale=0.10,
            monotonic_clock=lambda: 123.456,
            backend=self.backend,
            serial_ports_provider=lambda: list(self.ports),
            **kwargs,
        )

    def test_discovers_one_authenticated_follower_and_builds_locked_config_without_connecting(
        self,
    ) -> None:
        robot = self._robot()

        self.assertEqual(self.backend.follower.connect_count, 0)
        self.assertFalse(self.backend.follower.is_connected)
        self.assertEqual(self.backend.config_kwargs["port"], "/dev/fake-follower")
        self.assertEqual(self.backend.config_kwargs["max_relative_target"], 1.5)
        self.assertEqual(self.backend.config_kwargs["pos_vel_velocity"], [200.0] * 7)
        self.assertTrue(self.backend.config_kwargs["disable_torque_on_disconnect"])
        self.assertEqual(list(self.backend.config_kwargs["cameras"]), ["front", "side"])
        self.assertEqual(robot.task, TASK)

    def test_wrong_no_or_multiple_usb_matches_reject(self) -> None:
        cases = (
            [],
            [SimpleNamespace(device="/dev/wrong", vid=1, pid=2)],
            self.ports + [SimpleNamespace(device="/dev/duplicate", vid=0x2E88, pid=0x4603)],
        )
        for ports in cases:
            with self.subTest(ports=ports):
                self.ports = ports
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    self._robot()

    def test_rejects_speed_outside_first_live_range(self) -> None:
        for speed in (0.099, 0.201, np.nan):
            with self.subTest(speed=speed), self.assertRaisesRegex(ValueError, "speed_scale"):
                ReBotPolicyRobot.from_profile(
                    profile_snapshot=self.profile,
                    runtime_root=self.runtime_root,
                    speed_scale=speed,
                    backend=self.backend,
                    serial_ports_provider=lambda: self.ports,
                )

    def test_calibration_or_driver_fingerprint_mismatch_rejects(self) -> None:
        entries = (
            "follower",
            "follower_driver_contract",
            "follower_base_implementation",
            "follower_dm_implementation",
        )
        for entry in entries:
            with self.subTest(entry=entry):
                original = self.profile["calibration"][entry]["sha256"]
                self.profile["calibration"][entry]["sha256"] = "0" * 64
                try:
                    with self.assertRaisesRegex(ValueError, "fingerprint"):
                        self._robot()
                finally:
                    self.profile["calibration"][entry]["sha256"] = original

    def test_rejects_nonintegral_camera_dimensions_without_truncating_profile(self) -> None:
        self.profile["camera_defaults"]["front"]["width"] = 2.5

        with self.assertRaisesRegex(ValueError, "width"):
            self._robot()

    def test_rejects_boolean_calibration_numbers_even_with_matching_fingerprint(self) -> None:
        entry = self.profile["calibration"]["follower"]
        path = self.runtime_root / entry["runtime_relative_path"]
        calibration = json.loads(path.read_text())
        calibration["shoulder_pan"]["homing_offset"] = True
        payload = json.dumps(calibration).encode()
        path.write_bytes(payload)
        entry["sha256"] = sha256(payload).hexdigest()

        with self.assertRaisesRegex(ValueError, "calibration"):
            self._robot()

    def test_rejects_instantiated_velocity_mismatch_before_connect(self) -> None:
        self.backend.velocity_override = [199.0] * 7

        with self.assertRaisesRegex(ValueError, "velocity"):
            self._robot()

        self.assertEqual(self.backend.follower.connect_count, 0)

    def test_rejects_direction_mismatch_on_follower_actual_config(self) -> None:
        directions = dict(zip(JOINTS, DIRECTIONS, strict=True))
        directions["wrist_yaw"] = -1.0
        self.backend.follower_direction_override = directions

        with self.assertRaisesRegex(ValueError, "plugin directions"):
            self._robot()

    def test_rejects_nonfinite_capture_clock(self) -> None:
        robot = ReBotPolicyRobot.from_profile(
            profile_snapshot=self.profile,
            runtime_root=self.runtime_root,
            speed_scale=0.10,
            monotonic_clock=lambda: np.nan,
            backend=self.backend,
            serial_ports_provider=lambda: list(self.ports),
        )
        robot.connect()

        with self.assertRaisesRegex(ValueError, "capture"):
            robot.observe()

    def test_plugin_binding_mismatch_rejects_before_connect(self) -> None:
        cases = ("direction", "limit", "order", "calibration", "camera")
        for case in cases:
            with self.subTest(case=case):
                self.backend = FakeBackend()
                if case == "direction":
                    self.backend.direction_override = dict(zip(JOINTS, DIRECTIONS, strict=True))
                    self.backend.direction_override["wrist_yaw"] = -1.0
                elif case == "limit":
                    self.backend.limit_override = dict(zip(JOINTS, LIMITS, strict=True))
                    self.backend.limit_override["wrist_yaw"] = (-80.0, 80.0)
                elif case == "order":
                    self.backend.motor_order = tuple(reversed(JOINTS))
                elif case == "calibration":
                    self.backend.calibration_path_override = self.runtime_root / "wrong.json"
                else:
                    self.backend.camera_keys = ("front",)
                with self.assertRaisesRegex(ValueError, "plugin|calibration|camera"):
                    self._robot()
                self.assertEqual(self.backend.follower.connect_count, 0)

    def test_observation_copies_locked_images_and_physical_joint_order(self) -> None:
        robot = self._robot()
        robot.connect()
        observation = robot.observe()

        self.assertEqual(self.backend.follower.connect_calibrate, False)
        self.assertEqual(observation.task, TASK)
        self.assertEqual(observation.captured_monotonic_s, 123.456)
        np.testing.assert_array_equal(observation.state_deg, np.arange(7, dtype=float))
        np.testing.assert_array_equal(
            observation.front,
            self.backend.follower.observation["front"],
        )
        np.testing.assert_array_equal(observation.side, self.backend.follower.observation["side"])
        self.assertFalse(
            np.shares_memory(
                observation.front,
                self.backend.follower.observation["front"],
            )
        )
        self.assertFalse(
            np.shares_memory(
                observation.side,
                self.backend.follower.observation["side"],
            )
        )

    def test_missing_malformed_or_nonfinite_observation_rejects(self) -> None:
        cases = ("front", "side", "joint", "nonfinite")
        for case in cases:
            with self.subTest(case=case):
                self.backend = FakeBackend()
                robot = self._robot()
                robot.connect()
                if case in ("front", "side"):
                    self.backend.follower.observation.pop(case)
                elif case == "joint":
                    self.backend.follower.observation.pop("wrist_yaw.pos")
                else:
                    self.backend.follower.observation["gripper.pos"] = np.inf
                with self.assertRaisesRegex(ValueError, "observation"):
                    robot.observe()

    def test_inverts_all_seven_physical_actions_before_plugin_direction_mapping(self) -> None:
        robot = self._robot()
        robot.connect()
        requested = np.array((-10.0, -20.0, -30.0, 40.0, 50.0, -60.0, -120.0))

        returned = robot.send_action(requested)

        expected_command = {
            f"{name}.pos": requested[index] / DIRECTIONS[index]
            for index, name in enumerate(JOINTS)
        }
        self.assertEqual(self.backend.follower.last_command, expected_command)
        self.assertIn("wrist_yaw.pos", self.backend.follower.last_command)
        np.testing.assert_allclose(returned, requested, rtol=0, atol=1e-12)

    def test_zero_direction_or_malformed_nonfinite_disagreeing_plugin_return_rejects(self) -> None:
        robot = self._robot()
        robot.connect()
        requested = np.zeros(7)
        feature_keys = [f"{name}.pos" for name in JOINTS]
        cases = (
            {key: 0.0 for key in feature_keys[:-1]},
            {key: (np.nan if key == "gripper.pos" else 0.0) for key in feature_keys},
            {key: (1.0 if key == "gripper.pos" else 0.0) for key in feature_keys},
        )
        for value in cases:
            with self.subTest(value=value):
                self.backend.follower.return_override = value
                with self.assertRaisesRegex(ValueError, "returned action"):
                    robot.send_action(requested)

        self.backend.follower.return_override = None
        self.backend.follower.config.joint_directions["gripper"] = 0.0
        with self.assertRaisesRegex(ValueError, "direction"):
            robot.send_action(requested)

    def test_mutated_nonfinite_limit_or_overflowing_inversion_rejects_before_send(self) -> None:
        for case in ("limit", "direction"):
            with self.subTest(case=case):
                self.backend = FakeBackend()
                robot = self._robot()
                robot.connect()
                requested = np.zeros(7)
                if case == "limit":
                    self.backend.follower.config.joint_limits["gripper"] = (
                        -270.0,
                        np.nan,
                    )
                else:
                    requested[-1] = -1.0
                    self.backend.follower.config.joint_directions["gripper"] = 1e-320

                with self.assertRaisesRegex(ValueError, "limit|invert"):
                    robot.send_action(requested)

                self.assertIsNone(self.backend.follower.last_command)


if __name__ == "__main__":
    unittest.main()
