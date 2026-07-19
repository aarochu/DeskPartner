from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
import hashlib
import json
from pathlib import Path
import subprocess
import sys
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
MANUAL_RESET_PHRASE = "I RESET THE CAN AND CLEARED THE WORKSPACE"


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


class FailingConnectRobot(FakeRobot):
    def __init__(self, *, cleanup_error: Exception | None = None) -> None:
        super().__init__()
        self.cleanup_error = cleanup_error

    def connect(self) -> None:
        self.connect_count += 1
        raise RuntimeError("connect exploded")

    def disconnect(self) -> None:
        self.disconnect_count += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


class StopOnSecondObservationRobot(FakeRobot):
    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__()
        self.stop_event = stop_event

    def observe(self) -> RolloutObservation:
        observation = super().observe()
        if self.observe_count == 2:
            self.stop_event.set()
        return observation


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

    def verdict(self) -> str | None:
        return None


@dataclass
class Summary:
    cycles_completed: int = 1
    actions_attempted: int = 1
    actions_confirmed: int = 1
    terminal_reason: str = "max_cycles"
    primary_fault_reason: str | None = None
    cleanup_fault_reason: str | None = None
    audit_fault_reason: str | None = None


@dataclass
class EpisodeSummary:
    attempt: int
    terminal_reason: str
    cycles_completed: int = 1
    actions_attempted: int = 0
    actions_confirmed: int = 0
    primary_fault_reason: str | None = None
    cleanup_fault_reason: str | None = None
    audit_fault_reason: str | None = None
    elapsed_seconds: float = 1.25
    clamp_count: int = 0


class ScriptedEpisodeRunner:
    def __init__(self, *, outcomes: list[str], attempts: list[int], **kwargs) -> None:
        self.outcomes = outcomes
        self.attempts = attempts
        self.robot = kwargs["robot"]

    def run_episode(self, *, attempt: int) -> EpisodeSummary:
        self.attempts.append(attempt)
        self.robot.connect()
        self.robot.disconnect()
        outcome = self.outcomes.pop(0)
        return EpisodeSummary(
            attempt=attempt,
            terminal_reason=outcome,
            primary_fault_reason=(
                "simulated safety fault" if outcome == "safety_fault" else None
            ),
        )


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


class ConnectThenPropagateRunner:
    def __init__(self, **kwargs) -> None:
        self.robot = kwargs["robot"]

    def run(self, cycles: int) -> Summary:
        del cycles
        self.robot.connect()
        raise AssertionError("connect was expected to fail")


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
        self.bundle.path.mkdir()
        (self.bundle.path / "model.safetensors").write_bytes(b"test weights")
        self.write_processor_configs(self.bundle.path)

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
            "repo_root": self.root,
        }
        values.update(overrides)
        return CliDependencies(**values)

    @staticmethod
    def write_processor_configs(checkpoint: Path) -> None:
        (checkpoint / "preprocessor_config.json").write_text(
            json.dumps({"steps": []}), encoding="utf-8"
        )
        (checkpoint / "postprocessor_config.json").write_text(
            json.dumps({"steps": []}), encoding="utf-8"
        )

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
        self.write_processor_configs(checkpoint)
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
            str(self.root / "runs" / "policy" / "rollout.jsonl"),
        ]

    def test_help_documents_all_staged_gates_and_physical_estop_warning(self) -> None:
        help_text = build_parser().format_help()

        for gate in ("Gate A", "Gate B", "Gate C", "Gate D"):
            self.assertIn(gate, help_text)
        self.assertIn("physical e-stop", help_text.lower())
        self.assertIn("offline --checkpoint CHECKPOINT", help_text)
        self.assertIn("live --checkpoint CHECKPOINT --cycles 5", help_text)
        self.assertIn("s = success", help_text)
        self.assertIn("f = failure", help_text)
        self.assertIn("q/x/Esc = stop", help_text)
        self.assertIn("30.0-second", help_text)
        self.assertIn("300 confirmed live actions", help_text)
        self.assertIn("manual reset", help_text.lower())
        self.assertIn("--retry-on-failure", help_text)

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
        self.assertIn(
            "joint_order=shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,"
            "wrist_yaw,wrist_roll,gripper",
            output,
        )
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
        keyboard = FakeKeyboardStop()
        runner_calls: list[dict[str, object]] = []

        def make_runner(**kwargs):
            runner_calls.append(kwargs)
            return ExercisingRunner(**kwargs)

        deps = self.dependencies(
            input_fn=lambda prompt: prompts.append(prompt) or next(replies),
            robot_factory=lambda **kwargs: robot,
            runner_factory=make_runner,
            keyboard_stop_factory=lambda: keyboard,
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
        self.assertIs(runner_calls[0]["stop_requested"], keyboard.event)
        self.assertIn("actions_attempted=2", self.stdout.getvalue())
        self.assertIn("primary_fault=none", self.stdout.getvalue())
        metadata = json.loads(
            (self.root / "runs" / "policy" / "rollout.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        self.assertEqual(metadata["checkpoint"], str(self.bundle.path.resolve()))
        self.assertEqual(
            metadata["checkpoint_digest"],
            hashlib.sha256(b"test weights").hexdigest(),
        )

    def test_episode_default_has_no_retry_and_prints_attempt_metrics(self) -> None:
        robots: list[FakeRobot] = []
        attempts: list[int] = []
        outcomes = ["operator_failure"]
        prompts: list[str] = []

        def input_fn(prompt: str) -> str:
            prompts.append(prompt)
            return (
                "I HAVE AN E-STOP OPERATOR"
                if "E-STOP" in prompt
                else "WORKSPACE IS EMPTY"
            )

        deps = self.dependencies(
            input_fn=input_fn,
            robot_factory=lambda **kwargs: robots.append(FakeRobot()) or robots[-1],
            runner_factory=lambda **kwargs: ScriptedEpisodeRunner(
                outcomes=outcomes,
                attempts=attempts,
                **kwargs,
            ),
        )

        status = main(
            self.rollout_args() + ["--live", "--episode"],
            dependencies=deps,
        )

        self.assertEqual(status, 0)
        self.assertEqual(attempts, [1])
        self.assertEqual(len(robots), 1)
        self.assertFalse(any("manual reset" in prompt.lower() for prompt in prompts))
        summary = self.stdout.getvalue()
        self.assertIn("attempt=1", summary)
        self.assertIn("elapsed_seconds=1.250", summary)
        self.assertIn("clamp_count=0", summary)
        self.assertIn("terminal_reason=operator_failure", summary)

    def test_episode_failure_without_exact_reset_ack_is_not_retried(self) -> None:
        robots: list[FakeRobot] = []
        attempts: list[int] = []
        outcomes = ["operator_failure"]
        prompts: list[str] = []

        def input_fn(prompt: str) -> str:
            prompts.append(prompt)
            if "manual reset" in prompt.lower():
                return "not acknowledged"
            return (
                "I HAVE AN E-STOP OPERATOR"
                if "E-STOP" in prompt
                else "WORKSPACE IS EMPTY"
            )

        deps = self.dependencies(
            input_fn=input_fn,
            robot_factory=lambda **kwargs: robots.append(FakeRobot()) or robots[-1],
            runner_factory=lambda **kwargs: ScriptedEpisodeRunner(
                outcomes=outcomes,
                attempts=attempts,
                **kwargs,
            ),
        )

        status = main(
            self.rollout_args()
            + ["--live", "--episode", "--retry-on-failure"],
            dependencies=deps,
        )

        self.assertEqual(status, 0)
        self.assertEqual(attempts, [1])
        self.assertEqual(len(robots), 1)
        self.assertEqual(
            sum("manual reset" in prompt.lower() for prompt in prompts),
            1,
        )

    def test_acknowledged_failure_retries_once_with_fresh_robot_lifecycle(self) -> None:
        robots: list[FakeRobot] = []
        attempts: list[int] = []
        outcomes = ["operator_failure", "operator_success"]

        def input_fn(prompt: str) -> str:
            if "manual reset" in prompt.lower():
                return MANUAL_RESET_PHRASE
            return (
                "I HAVE AN E-STOP OPERATOR"
                if "E-STOP" in prompt
                else "WORKSPACE IS EMPTY"
            )

        deps = self.dependencies(
            input_fn=input_fn,
            robot_factory=lambda **kwargs: robots.append(FakeRobot()) or robots[-1],
            runner_factory=lambda **kwargs: ScriptedEpisodeRunner(
                outcomes=outcomes,
                attempts=attempts,
                **kwargs,
            ),
        )

        status = main(
            self.rollout_args()
            + ["--live", "--episode", "--retry-on-failure"],
            dependencies=deps,
        )

        self.assertEqual(status, 0)
        self.assertEqual(attempts, [1, 2])
        self.assertEqual(len(robots), 2)
        self.assertTrue(all(robot.connect_count == 1 for robot in robots))
        self.assertTrue(all(robot.disconnect_count == 1 for robot in robots))
        self.assertIn("attempt=2", self.stdout.getvalue())
        self.assertIn("terminal_reason=operator_success", self.stdout.getvalue())

    def test_each_live_episode_attempt_resets_policy_before_robot_use(self) -> None:
        class StatefulPolicy:
            def __init__(self) -> None:
                self.state = 99
                self.reset_calls = 0

            def reset(self) -> None:
                self.reset_calls += 1
                self.state = 0

        policy = StatefulPolicy()
        states_at_attempt: list[int] = []
        outcomes = ["operator_failure", "operator_success"]

        class StatefulRunner:
            def __init__(self, **kwargs) -> None:
                self.policy = kwargs["policy"]
                self.robot = kwargs["robot"]

            def run_episode(self, *, attempt: int) -> EpisodeSummary:
                states_at_attempt.append(self.policy.state)
                self.policy.state += 1
                self.robot.connect()
                self.robot.disconnect()
                return EpisodeSummary(attempt=attempt, terminal_reason=outcomes.pop(0))

        def input_fn(prompt: str) -> str:
            if "manual reset" in prompt.lower():
                return MANUAL_RESET_PHRASE
            return "I HAVE AN E-STOP OPERATOR" if "E-STOP" in prompt else "WORKSPACE IS EMPTY"

        robots: list[FakeRobot] = []
        status = main(
            self.rollout_args() + ["--live", "--episode", "--retry-on-failure"],
            dependencies=self.dependencies(
                input_fn=input_fn,
                policy_factory=lambda bundle, device: policy,
                robot_factory=lambda **kwargs: robots.append(FakeRobot()) or robots[-1],
                runner_factory=lambda **kwargs: StatefulRunner(**kwargs),
            ),
        )

        self.assertEqual(status, 0)
        self.assertEqual(policy.reset_calls, 2)
        self.assertEqual(states_at_attempt, [0, 0])
        self.assertEqual(len(robots), 2)

    def test_second_operator_failure_never_produces_a_third_attempt(self) -> None:
        robots: list[FakeRobot] = []
        attempts: list[int] = []
        outcomes = ["operator_failure", "operator_failure"]

        def input_fn(prompt: str) -> str:
            if "manual reset" in prompt.lower():
                return MANUAL_RESET_PHRASE
            return (
                "I HAVE AN E-STOP OPERATOR"
                if "E-STOP" in prompt
                else "WORKSPACE IS EMPTY"
            )

        deps = self.dependencies(
            input_fn=input_fn,
            robot_factory=lambda **kwargs: robots.append(FakeRobot()) or robots[-1],
            runner_factory=lambda **kwargs: ScriptedEpisodeRunner(
                outcomes=outcomes,
                attempts=attempts,
                **kwargs,
            ),
        )

        status = main(
            self.rollout_args()
            + ["--live", "--episode", "--retry-on-failure"],
            dependencies=deps,
        )

        self.assertEqual(status, 0)
        self.assertEqual(attempts, [1, 2])
        self.assertEqual(len(robots), 2)

    def test_safety_fault_is_never_retried(self) -> None:
        robots: list[FakeRobot] = []
        attempts: list[int] = []
        outcomes = ["safety_fault"]
        prompts: list[str] = []

        def input_fn(prompt: str) -> str:
            prompts.append(prompt)
            return (
                "I HAVE AN E-STOP OPERATOR"
                if "E-STOP" in prompt
                else "WORKSPACE IS EMPTY"
            )

        deps = self.dependencies(
            input_fn=input_fn,
            robot_factory=lambda **kwargs: robots.append(FakeRobot()) or robots[-1],
            runner_factory=lambda **kwargs: ScriptedEpisodeRunner(
                outcomes=outcomes,
                attempts=attempts,
                **kwargs,
            ),
        )

        status = main(
            self.rollout_args()
            + ["--live", "--episode", "--retry-on-failure"],
            dependencies=deps,
        )

        self.assertEqual(status, 1)
        self.assertEqual(attempts, [1])
        self.assertEqual(len(robots), 1)
        self.assertFalse(any("manual reset" in prompt.lower() for prompt in prompts))

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
        log_path = self.root / "runs" / "policy" / "dummy.jsonl"

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
        self.assertIsNone(metadata["checkpoint"])
        self.assertIsNone(metadata["checkpoint_digest"])
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

    def test_runner_construction_failure_is_balanced_and_prints_fault_summary(self) -> None:
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

        self.assertEqual(status, 1)
        self.assertEqual(robot.connect_count, 0)
        self.assertEqual(robot.disconnect_count, 0)
        self.assertEqual(robot.send_count, 0)
        summary = self.stdout.getvalue()
        self.assertIn("terminal_reason=fault", summary)
        self.assertIn("primary_fault=runner construction failed: runner construction exploded", summary)
        self.assertIn("cleanup_fault=none", summary)
        self.assertIn("audit_fault=none", summary)

    def test_runner_run_exception_prints_summary_without_motion(self) -> None:
        robot = FakeRobot()
        deps = self.dependencies(
            input_fn=lambda prompt: (
                "I HAVE AN E-STOP OPERATOR"
                if "E-STOP" in prompt
                else "WORKSPACE IS EMPTY"
            ),
            robot_factory=lambda **kwargs: robot,
            runner_factory=lambda **kwargs: SimpleNamespace(
                run=lambda cycles: (_ for _ in ()).throw(
                    RuntimeError("runner run exploded")
                )
            ),
        )

        status = main(self.rollout_args() + ["--live"], dependencies=deps)

        self.assertEqual(status, 1)
        self.assertEqual(robot.connect_count, 0)
        self.assertEqual(robot.disconnect_count, 0)
        self.assertEqual(robot.send_count, 0)
        self.assertIn(
            "primary_fault=runner execution failed: runner run exploded",
            self.stdout.getvalue(),
        )

    def test_preflight_connect_and_cleanup_faults_are_separated_in_summary(self) -> None:
        robot = FailingConnectRobot(
            cleanup_error=RuntimeError("disconnect exploded")
        )
        deps = self.dependencies(
            input_fn=lambda prompt: (
                "I HAVE AN E-STOP OPERATOR"
                if "E-STOP" in prompt
                else "WORKSPACE IS EMPTY"
            ),
            robot_factory=lambda **kwargs: robot,
            runner_factory=lambda **kwargs: ConnectThenPropagateRunner(**kwargs),
        )

        status = main(self.rollout_args() + ["--live"], dependencies=deps)

        self.assertEqual(status, 1)
        self.assertEqual(robot.connect_count, 1)
        self.assertEqual(robot.disconnect_count, 1)
        self.assertEqual(robot.send_count, 0)
        summary = self.stdout.getvalue()
        self.assertIn("primary_fault=runner execution failed: connect exploded", summary)
        self.assertIn("cleanup_fault=disconnect exploded", summary)
        self.assertIn("audit_fault=none", summary)

    def test_real_rollout_processor_artifacts_block_policy_and_hardware(self) -> None:
        cases = ("missing_config", "missing_state")
        for case in cases:
            with self.subTest(case=case):
                self.write_processor_configs(self.bundle.path)
                if case == "missing_config":
                    (self.bundle.path / "postprocessor_config.json").unlink()
                else:
                    (self.bundle.path / "preprocessor_config.json").write_text(
                        json.dumps(
                            {"steps": [{"state_file": "state/missing.bin"}]}
                        ),
                        encoding="utf-8",
                    )
                calls: list[str] = []
                self.stderr.seek(0)
                self.stderr.truncate()
                deps = self.dependencies(
                    input_fn=lambda prompt: self.fail("prompted after processor fault"),
                    policy_factory=lambda bundle, device: calls.append("policy"),
                    robot_factory=lambda **kwargs: calls.append("robot"),
                )

                status = main(
                    self.rollout_args() + ["--live"], dependencies=deps
                )

                self.assertEqual(status, 2)
                self.assertEqual(calls, [])
                self.assertRegex(
                    self.stderr.getvalue(),
                    "postprocessor_config.json|state/missing.bin",
                )

    def test_explicit_log_path_rejects_traversal_protected_roots_and_symlinks(self) -> None:
        allowed = self.root / "runs" / "policy"
        allowed.mkdir(parents=True)
        credential = self.root / "config" / "credentials.env"
        credential.parent.mkdir()
        credential.write_text("SECRET=unchanged", encoding="utf-8")
        symlink = allowed / "linked.jsonl"
        symlink.symlink_to(credential)
        disallowed = (
            self.root / "data" / "episodes.jsonl",
            self.root / "models" / "checkpoint.jsonl",
            self.root / "config" / "calibration.jsonl",
            allowed / ".." / "escaped.jsonl",
            symlink,
            allowed / "credentials.jsonl",
            allowed / ".env.jsonl",
        )
        for path in disallowed:
            with self.subTest(path=path):
                calls: list[str] = []
                self.stderr.seek(0)
                self.stderr.truncate()
                args = self.rollout_args()
                args[args.index("--log-path") + 1] = str(path)
                deps = self.dependencies(
                    policy_factory=lambda bundle, device: calls.append("policy"),
                    robot_factory=lambda **kwargs: calls.append("robot"),
                )

                status = main(args + ["--live"], dependencies=deps)

                self.assertEqual(status, 2)
                self.assertEqual(calls, [])
                self.assertIn("runs/policy", self.stderr.getvalue())
                self.assertEqual(
                    credential.read_text(encoding="utf-8"), "SECRET=unchanged"
                )

    def test_default_log_path_is_timestamped_under_injected_runs_policy(self) -> None:
        args = self.rollout_args()
        index = args.index("--log-path")
        del args[index : index + 2]
        deps = self.dependencies(
            input_fn=lambda prompt: (
                "I HAVE AN E-STOP OPERATOR"
                if "E-STOP" in prompt
                else "WORKSPACE IS EMPTY"
            )
        )

        status = main(args + ["--live"], dependencies=deps)

        expected = (
            self.root / "runs" / "policy" / "20260718T212223Z.jsonl"
        ).resolve()
        self.assertEqual(status, 0)
        self.assertTrue(expected.is_file())
        self.assertIn(f"jsonl_path={expected}", self.stdout.getvalue())

    def test_existing_rollout_log_is_refused_without_overwrite(self) -> None:
        log_path = self.root / "runs" / "policy" / "rollout.jsonl"
        log_path.parent.mkdir(parents=True)
        log_path.write_text("existing audit\n", encoding="utf-8")
        deps = self.dependencies(
            input_fn=lambda prompt: (
                "I HAVE AN E-STOP OPERATOR"
                if "E-STOP" in prompt
                else "WORKSPACE IS EMPTY"
            )
        )

        status = main(self.rollout_args() + ["--live"], dependencies=deps)

        self.assertEqual(status, 2)
        self.assertEqual(log_path.read_text(encoding="utf-8"), "existing audit\n")

    def test_shared_stop_event_set_after_preflight_prevents_policy_and_send(self) -> None:
        from p3_vlm_orchestrator.policy_rollout.runner import RolloutRunner

        keyboard = FakeKeyboardStop()
        robot = StopOnSecondObservationRobot(keyboard.event)
        policy_calls: list[str] = []

        class Policy:
            def predict(self, observation):
                policy_calls.append("predict")
                return np.zeros((10, 7))

        runner_calls: list[dict[str, object]] = []

        def make_runner(**kwargs):
            runner_calls.append(kwargs)
            return RolloutRunner(**kwargs)

        deps = self.dependencies(
            input_fn=lambda prompt: (
                "I HAVE AN E-STOP OPERATOR"
                if "E-STOP" in prompt
                else "WORKSPACE IS EMPTY"
            ),
            policy_factory=lambda bundle, device: Policy(),
            robot_factory=lambda **kwargs: robot,
            runner_factory=make_runner,
            keyboard_stop_factory=lambda: keyboard,
        )

        status = main(self.rollout_args() + ["--live"], dependencies=deps)

        self.assertEqual(status, 0)
        self.assertIs(runner_calls[0]["stop_requested"], keyboard.event)
        self.assertTrue(keyboard.event.is_set())
        self.assertEqual(policy_calls, [])
        self.assertEqual(robot.send_count, 0)
        self.assertEqual(robot.connect_count, 1)
        self.assertEqual(robot.disconnect_count, 1)
        self.assertIn("terminal_reason=stop_requested", self.stdout.getvalue())

    def test_fresh_process_inspect_and_offline_keep_hardware_modules_unloaded(self) -> None:
        checkpoint = self.write_checkpoint()
        repo_root = Path(__file__).resolve().parents[2]
        commands = (
            (["inspect", "--checkpoint", str(checkpoint)], 0),
            (
                [
                    "offline",
                    "--checkpoint",
                    str(checkpoint),
                    "--dataset",
                    str(self.root / "missing-dataset"),
                    "--episodes",
                    "1",
                ],
                2,
            ),
        )
        for argv, expected_status in commands:
            with self.subTest(command=argv[0]):
                script = f"""
import sys
from p3_vlm_orchestrator.policy_rollout.cli import main
status = main({argv!r})
assert status == {expected_status}, status
forbidden = []
for name in sys.modules:
    lower = name.lower()
    if (
        name == 'serial' or name.startswith('serial.')
        or name == 'cv2' or name.startswith('cv2.')
        or name == 'pinocchio' or name.startswith('pinocchio.')
        or name == 'lerobot' or name.startswith('lerobot.')
        or name == 'torch' or name.startswith('torch.')
        or 'rebot_robot' in lower
        or lower.startswith('rebotarm_control_py')
        or lower.startswith('p1_arm_motion')
    ):
        forbidden.append(name)
assert not forbidden, forbidden
"""
                completed = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    completed.stdout + completed.stderr,
                )


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
            make_observation(front_shape=(480, 640, 1)),
            make_observation(front_shape=(480, 640, 4)),
            make_observation(front_shape=(480, 640, 0)),
            make_observation(side_shape=(720, 1280, 1)),
            make_observation(side_shape=(720, 1280, 4)),
            make_observation(side_shape=(720, 1280, 0)),
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
