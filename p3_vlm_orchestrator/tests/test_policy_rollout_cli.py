from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest

import numpy as np

from p3_vlm_orchestrator.policy_rollout.cli import (
    CliDependencies,
    PreflightRobotAdapter,
    build_parser,
    canonical_profile_digest,
    default_serial_port_is_free,
    main,
)
from rebot_operator_kit.rollout.contracts import RolloutObservation


TASK = "Pick up one can and place it in the taped sorting zone"
NOW = datetime(2026, 7, 18, 21, 22, 23, tzinfo=timezone.utc)


def make_profile() -> dict[str, object]:
    names = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "gripper",
    )
    limits = (
        (-145.0, 145.0),
        (-170.0, 0.0),
        (-200.0, 0.0),
        (-80.0, 90.0),
        (-90.0, 90.0),
        (-90.0, 90.0),
        (-270.0, 0.0),
    )
    return {
        "schema_version": 1,
        "profile_id": "rebot-test-profile",
        "profile_version": 1,
        "collection_defaults": {
            "task": TASK,
            "fps": 30,
            "motor_velocity": 2000.0,
            "gripper_force": 0.05,
        },
        "training_defaults": {
            "action_dimension": 7,
            "chunk_size": 10,
            "n_action_steps": 10,
            "state_normalization": "quantile",
            "action_normalization": "quantile",
            "normalize_gripper": True,
            "image_order": [
                "observation.images.front",
                "observation.images.side",
            ],
        },
        "coordinate_contract": {
            "frame": "follower_degrees_after_direction_limits_and_step_cap",
            "control_mode": "absolute joint pose",
            "action_dimension": 7,
            "joints": [
                {
                    "name": name,
                    "feature": f"{name}.pos",
                    "leader_to_follower_scale": -1.0 if index < 2 else 1.0,
                    "soft_limit_degrees": list(limits[index]),
                }
                for index, name in enumerate(names)
            ],
        },
        "camera_defaults": {
            "front": {
                "recording_key": "observation.images.front",
                "index": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
            },
            "side": {
                "recording_key": "observation.images.side",
                "index": 1,
                "width": 1280,
                "height": 720,
                "fps": 30,
            },
            "excluded_screen_index": 3,
            "excluded_screen_name": "MacBook camera",
            "minimum_measured_fps": 27,
        },
        "calibration": {
            "follower": {
                "type": "seeed_b601_dm_follower",
                "id": "follower1",
                "runtime_relative_path": "calibration/follower.json",
                "sha256": "a" * 64,
            },
            "leader": {
                "type": "rebot_arm_102_leader",
                "id": "rebot_arm_102_leader",
                "runtime_relative_path": "calibration/leader.json",
                "sha256": "e" * 64,
            },
            "follower_driver_contract": {
                "runtime_relative_path": "driver/config.py",
                "sha256": "b" * 64,
            },
            "follower_base_implementation": {
                "runtime_relative_path": "driver/base.py",
                "sha256": "c" * 64,
            },
            "follower_dm_implementation": {
                "runtime_relative_path": "driver/dm.py",
                "sha256": "d" * 64,
            },
            "leader_driver_contract": {
                "runtime_relative_path": "driver/leader-config.py",
                "sha256": "f" * 64,
            },
            "leader_implementation": {
                "runtime_relative_path": "driver/leader.py",
                "sha256": "1" * 64,
            },
        },
        "hardware_identity": {
            "follower_usb": {"vid": 0x2E88, "pid": 0x4603},
            "leader_usb": {"vid": 0x1A86, "pid": 0x7523},
            "serial_port_policy": "Discover exactly one device; never persist a path.",
        },
    }


def make_observation(
    *,
    task: str = TASK,
    state: object | None = None,
    front_shape: tuple[int, ...] = (480, 640, 3),
    side_shape: tuple[int, ...] = (720, 1280, 3),
) -> RolloutObservation:
    return RolloutObservation(
        front=np.zeros(front_shape, dtype=np.uint8),
        side=np.zeros(side_shape, dtype=np.uint8),
        state_deg=np.zeros(7) if state is None else state,
        task=task,
        captured_monotonic_s=10.0,
    )


class FakeRobot:
    follower_port = "/dev/fake-follower"

    def __init__(self, observations: list[RolloutObservation] | None = None) -> None:
        self.observations = observations or [make_observation(), make_observation()]
        self.connect_count = 0
        self.disconnect_count = 0
        self.observe_count = 0
        self.send_count = 0

    def connect(self) -> None:
        self.connect_count += 1

    def disconnect(self) -> None:
        self.disconnect_count += 1

    def observe(self) -> RolloutObservation:
        result = self.observations[min(self.observe_count, len(self.observations) - 1)]
        self.observe_count += 1
        return result

    def send_action(self, action: np.ndarray) -> np.ndarray:
        self.send_count += 1
        return np.asarray(action).copy()


class FakeKeyboardStop:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.exited = True


@dataclass
class Summary:
    cycles_completed: int = 1
    actions_attempted: int = 1
    actions_confirmed: int = 1
    terminal_reason: str = "max_cycles"
    primary_fault_reason: str | None = None
    cleanup_fault_reason: str | None = None
    audit_fault_reason: str | None = None


class ExercisingRunner:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def run(self, cycles: int) -> Summary:
        robot = self.kwargs["robot"]
        robot.connect()
        robot.observe()  # Must be fresh; preflight consumed the first observation.
        robot.disconnect()
        return Summary(
            cycles_completed=cycles,
            actions_attempted=cycles if self.kwargs["mode"] == "live" else 0,
            actions_confirmed=cycles if self.kwargs["mode"] == "live" else 0,
        )


class PolicyRolloutCliTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stdout = StringIO()
        self.stderr = StringIO()
        self.profile = make_profile()
        self.bundle = SimpleNamespace(
            path=self.root / "checkpoint",
            task=TASK,
            action_dimension=7,
            chunk_size=10,
            action_steps=10,
            image_order=("observation.images.front", "observation.images.side"),
            profile_digest=canonical_profile_digest(self.profile),
            profile_snapshot=self.profile,
        )

    def dependencies(self, **overrides) -> CliDependencies:
        values = {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "input_fn": lambda prompt: "",
            "utc_now": lambda: NOW,
            "monotonic_clock": lambda: 10.0,
            "checkpoint_loader": lambda path: self.bundle,
            "offline_evaluator": lambda *args, **kwargs: [object()],
            "policy_factory": lambda bundle, device: object(),
            "dummy_policy_factory": lambda: object(),
            "safety_factory": lambda profile, mode: SimpleNamespace(mode=mode),
            "guard_factory": lambda **kwargs: object(),
            "robot_factory": lambda **kwargs: FakeRobot(),
            "runner_factory": lambda **kwargs: ExercisingRunner(**kwargs),
            "keyboard_stop_factory": FakeKeyboardStop,
            "serial_port_is_free": lambda port: True,
        }
        values.update(overrides)
        return CliDependencies(**values)

    def write_profile(self, profile: dict[str, object] | None = None) -> Path:
        path = self.root / "training_profile.json"
        path.write_text(json.dumps(profile or self.profile), encoding="utf-8")
        return path

    def write_checkpoint(self, *, digest: str | None = None) -> Path:
        checkpoint = self.root / "real-checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(
            json.dumps({"type": "molmoact2", "device": "cuda"}),
            encoding="utf-8",
        )
        (checkpoint / "preprocessor_config.json").write_text(
            json.dumps({"steps": []}), encoding="utf-8"
        )
        (checkpoint / "postprocessor_config.json").write_text(
            json.dumps({"steps": []}), encoding="utf-8"
        )
        (checkpoint / "model.safetensors").write_bytes(b"not-loaded-by-inspect")
        collection_contract = {"task": TASK}
        (checkpoint / "rebot_training_profile.json").write_text(
            json.dumps(
                {
                    "training_profile_digest": digest
                    or canonical_profile_digest(self.profile),
                    "profile_snapshot": self.profile,
                    "collection_contract": collection_contract,
                    "collection_contract_digest": canonical_profile_digest(
                        collection_contract
                    ),
                }
            ),
            encoding="utf-8",
        )
        return checkpoint

    def rollout_args(self, command: str = "live") -> list[str]:
        return [
            command,
            "--checkpoint",
            str(self.root / "checkpoint"),
            "--runtime-root",
            str(self.root / "runtime"),
            "--arm-config",
            str(self.root / "arm.yaml"),
            "--workspace-config",
            str(self.root / "workspace.yaml"),
            "--workspace-calibration",
            str(self.root / "calibration.json"),
            "--log-path",
            str(self.root / "rollout.jsonl"),
        ]

    def test_help_documents_all_staged_gates_and_physical_estop_warning(self) -> None:
        help_text = build_parser().format_help()

        for gate in ("Gate A", "Gate B", "Gate C", "Gate D"):
            self.assertIn(gate, help_text)
        self.assertIn("physical e-stop", help_text.lower())
        self.assertIn("offline --checkpoint CHECKPOINT", help_text)
        self.assertIn("live --checkpoint CHECKPOINT --cycles 5", help_text)

    def test_inspect_validates_and_prints_contract_without_policy_factory(self) -> None:
        checkpoint = self.write_checkpoint()
        deps = CliDependencies(
            stdout=self.stdout,
            stderr=self.stderr,
            policy_factory=lambda bundle, device: self.fail("policy weights loaded"),
            robot_factory=lambda **kwargs: self.fail("robot factory called"),
        )

        status = main(
            ["inspect", "--checkpoint", str(checkpoint)], dependencies=deps
        )

        self.assertEqual(status, 0)
        output = self.stdout.getvalue()
        self.assertIn(f"locked_task={TASK}", output)
        self.assertIn("policy_type=molmoact2", output)
        self.assertIn("dimension=7", output)
        self.assertIn("observation.images.front,observation.images.side", output)
        self.assertIn("preprocessor_config.json,postprocessor_config.json", output)
        self.assertIn(canonical_profile_digest(self.profile), output)

    def test_inspect_rejects_a_mismatched_checkpoint_profile_digest(self) -> None:
        checkpoint = self.write_checkpoint(digest="0" * 64)

        status = main(
            ["inspect", "--checkpoint", str(checkpoint)],
            dependencies=CliDependencies(stdout=self.stdout, stderr=self.stderr),
        )

        self.assertEqual(status, 2)
        self.assertIn("digest does not match", self.stderr.getvalue())

    def test_offline_delegates_directly_without_touching_hardware(self) -> None:
        calls: list[tuple[object, Path, int, str]] = []

        def evaluate(bundle, dataset, *, episodes, device, output):
            calls.append((bundle, dataset, episodes, device))
            self.assertIs(output, self.stdout)
            return [object()]

        deps = self.dependencies(
            offline_evaluator=evaluate,
            robot_factory=lambda **kwargs: self.fail("hardware factory called"),
        )
        status = main(
            [
                "offline",
                "--checkpoint",
                str(self.root / "checkpoint"),
                "--dataset",
                str(self.root / "dataset"),
                "--episodes",
                "3",
                "--device",
                "cpu",
            ],
            dependencies=deps,
        )

        self.assertEqual(status, 0)
        self.assertEqual(calls, [(self.bundle, self.root / "dataset", 3, "cpu")])

    def test_live_rejects_missing_live_gate_or_out_of_range_speed_before_hardware(self) -> None:
        deps = self.dependencies(
            robot_factory=lambda **kwargs: self.fail("hardware factory called")
        )

        self.assertEqual(main(self.rollout_args(), dependencies=deps), 2)
        self.assertIn("--live", self.stderr.getvalue())

        self.stderr.seek(0)
        self.stderr.truncate()
        args = self.rollout_args() + ["--live", "--speed-scale", "0.21"]
        self.assertEqual(main(args, dependencies=deps), 2)
        self.assertIn("[0.10, 0.20]", self.stderr.getvalue())

    def test_live_uses_exact_prompts_in_order_then_preflights_once(self) -> None:
        prompts: list[str] = []
        replies = iter(("I HAVE AN E-STOP OPERATOR", "WORKSPACE IS EMPTY"))
        robot = FakeRobot()
        runner_calls: list[dict[str, object]] = []

        def make_runner(**kwargs):
            runner_calls.append(kwargs)
            return ExercisingRunner(**kwargs)

        deps = self.dependencies(
            input_fn=lambda prompt: prompts.append(prompt) or next(replies),
            robot_factory=lambda **kwargs: robot,
            runner_factory=make_runner,
        )
        status = main(
            self.rollout_args()
            + ["--live", "--cycles", "2", "--speed-scale", "0.10"],
            dependencies=deps,
        )

        self.assertEqual(status, 0)
        self.assertEqual(
            prompts,
            [
                'Type exactly "I HAVE AN E-STOP OPERATOR": ',
                'Type exactly "WORKSPACE IS EMPTY": ',
            ],
        )
        self.assertEqual(robot.connect_count, 1)
        self.assertEqual(robot.disconnect_count, 1)
        self.assertEqual(robot.observe_count, 2)
        self.assertEqual(len(runner_calls), 1)
        self.assertIs(runner_calls[0]["stop_requested"], runner_calls[0]["stop_requested"])
        self.assertIn("actions_attempted=2", self.stdout.getvalue())
        self.assertIn("primary_fault=none", self.stdout.getvalue())

    def test_live_rejects_either_incorrect_phrase_before_connect(self) -> None:
        cases = (
            ("wrong", "WORKSPACE IS EMPTY"),
            ("I HAVE AN E-STOP OPERATOR", "wrong"),
        )
        for replies in cases:
            with self.subTest(replies=replies):
                robot = FakeRobot()
                answers = iter(replies)
                self.stderr.seek(0)
                self.stderr.truncate()
                deps = self.dependencies(
                    input_fn=lambda prompt: next(answers),
                    robot_factory=lambda **kwargs: robot,
                )
                status = main(
                    self.rollout_args() + ["--live"], dependencies=deps
                )
                self.assertEqual(status, 2)
                self.assertEqual(robot.connect_count, 0)

    def test_serial_ownership_fails_closed_before_prompt_or_connect(self) -> None:
        robot = FakeRobot()
        checked: list[str] = []
        deps = self.dependencies(
            input_fn=lambda prompt: self.fail("prompted before serial gate"),
            robot_factory=lambda **kwargs: robot,
            serial_port_is_free=lambda port: checked.append(port) or False,
        )

        status = main(self.rollout_args() + ["--live"], dependencies=deps)

        self.assertEqual(status, 2)
        self.assertEqual(checked, ["/dev/fake-follower"])
        self.assertEqual(robot.connect_count, 0)

    def test_dummy_shadow_validates_and_logs_canonical_digest_without_auth_claim(self) -> None:
        profile_path = self.write_profile()
        robot = FakeRobot()
        captured_profiles: list[dict[str, object]] = []
        deps = self.dependencies(
            robot_factory=lambda **kwargs: captured_profiles.append(
                kwargs["profile_snapshot"]
            )
            or robot,
        )
        log_path = self.root / "dummy.jsonl"

        status = main(
            [
                "shadow",
                "--dummy-hold",
                "--profile",
                str(profile_path),
                "--cycles",
                "2",
                "--runtime-root",
                str(self.root / "runtime"),
                "--arm-config",
                str(self.root / "arm.yaml"),
                "--workspace-config",
                str(self.root / "workspace.yaml"),
                "--workspace-calibration",
                str(self.root / "calibration.json"),
                "--log-path",
                str(log_path),
            ],
            dependencies=deps,
        )

        self.assertEqual(status, 0)
        self.assertEqual(captured_profiles, [self.profile])
        metadata = json.loads(log_path.read_text().splitlines()[0])
        self.assertEqual(metadata["profile_digest"], canonical_profile_digest(self.profile))
        self.assertEqual(metadata["profile_authentication"], "standalone-untrusted")
        self.assertNotIn("verified", metadata["profile_authentication"])
        self.assertEqual(robot.send_count, 0)

    def test_invalid_dummy_profile_is_rejected_before_any_rollout_factory(self) -> None:
        profile = make_profile()
        profile["camera_defaults"]["front"]["width"] = 641  # type: ignore[index]
        profile_path = self.write_profile(profile)
        called: list[str] = []
        deps = self.dependencies(
            robot_factory=lambda **kwargs: called.append("robot"),
            guard_factory=lambda **kwargs: called.append("guard"),
            safety_factory=lambda *args, **kwargs: called.append("safety"),
        )

        status = main(
            [
                "shadow",
                "--dummy-hold",
                "--profile",
                str(profile_path),
            ],
            dependencies=deps,
        )

        self.assertEqual(status, 2)
        self.assertEqual(called, [])
        self.assertIn("640x480", self.stderr.getvalue())

    def test_runner_construction_failure_disconnects_robot(self) -> None:
        robot = FakeRobot()
        deps = self.dependencies(
            input_fn=lambda prompt: (
                "I HAVE AN E-STOP OPERATOR"
                if "E-STOP" in prompt
                else "WORKSPACE IS EMPTY"
            ),
            robot_factory=lambda **kwargs: robot,
            runner_factory=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("runner construction exploded")
            ),
        )

        status = main(self.rollout_args() + ["--live"], dependencies=deps)

        self.assertEqual(status, 2)
        self.assertEqual(robot.disconnect_count, 1)
        self.assertIn("runner construction exploded", self.stderr.getvalue())


class PreflightRobotAdapterTest(unittest.TestCase):
    def test_success_connects_preflights_then_observes_fresh_and_disconnects_once(self) -> None:
        first = make_observation()
        second = make_observation()
        robot = FakeRobot([first, second])
        adapter = PreflightRobotAdapter(
            robot=robot,
            expected_task=TASK,
            profile_snapshot=make_profile(),
        )

        adapter.connect()
        self.assertIs(adapter.observe(), second)
        adapter.disconnect()
        adapter.disconnect()

        self.assertEqual(robot.connect_count, 1)
        self.assertEqual(robot.observe_count, 2)
        self.assertEqual(robot.disconnect_count, 1)

    def test_each_preflight_mismatch_fails_closed_and_disconnects(self) -> None:
        malformed = (
            make_observation(task="different"),
            make_observation(state=np.zeros(6)),
            make_observation(state=np.array([0, 0, 0, 0, 0, 0, np.nan])),
            make_observation(front_shape=(640, 480, 3)),
            make_observation(side_shape=(1280, 720, 3)),
            make_observation(front_shape=(480, 640)),
        )
        for observation in malformed:
            with self.subTest(
                task=observation.task,
                state_shape=np.asarray(observation.state_deg).shape,
                front_shape=observation.front.shape,
                side_shape=observation.side.shape,
            ):
                robot = FakeRobot([observation])
                adapter = PreflightRobotAdapter(
                    robot=robot,
                    expected_task=TASK,
                    profile_snapshot=make_profile(),
                )
                with self.assertRaises(ValueError):
                    adapter.connect()
                adapter.disconnect()
                self.assertEqual(robot.connect_count, 1)
                self.assertEqual(robot.disconnect_count, 1)


class SerialOwnershipTest(unittest.TestCase):
    def test_lsof_argv_and_fail_closed_result_handling(self) -> None:
        calls: list[list[str]] = []

        def result(returncode: int, stdout: str = "", stderr: str = ""):
            def run(argv, **kwargs):
                calls.append(argv)
                return SimpleNamespace(
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                )

            return run

        self.assertTrue(
            default_serial_port_is_free("/dev/fake", runner=result(1))
        )
        self.assertFalse(
            default_serial_port_is_free(
                "/dev/fake", runner=result(0, "python 1 user /dev/fake")
            )
        )
        self.assertFalse(
            default_serial_port_is_free(
                "/dev/fake", runner=result(2, stderr="lsof failed")
            )
        )
        self.assertFalse(
            default_serial_port_is_free(
                "/dev/fake",
                runner=lambda argv, **kwargs: (_ for _ in ()).throw(OSError("missing")),
            )
        )
        self.assertEqual(calls[0], ["lsof", "/dev/fake"])


if __name__ == "__main__":
    unittest.main()
