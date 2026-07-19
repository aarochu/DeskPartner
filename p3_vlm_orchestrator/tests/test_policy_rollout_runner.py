from __future__ import annotations

import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np

from p3_vlm_orchestrator.policy_rollout.dummy_policy import (
    HoldPositionPolicy,
    UnsafePolicy,
)
from p3_vlm_orchestrator.policy_rollout.runner import RolloutRunner
from p3_vlm_orchestrator.policy_rollout.workspace_guard import WorkspaceViolation
from rebot_operator_kit.rollout.contracts import RolloutObservation
from rebot_operator_kit.rollout.safety import SafetyGovernor


TASK = "Pick one can and place it in the taped sorting zone"
LIMITS = np.repeat(np.array([[-200.0, 200.0]]), 7, axis=0)
LOG_FIELDS = {
    "timestamp_utc",
    "monotonic_s",
    "event",
    "mode",
    "cycle",
    "task",
    "current_state_deg",
    "predicted_first_action_deg",
    "safety_result",
    "send_result_deg",
    "inference_latency_s",
    "actions_attempted",
    "actions_confirmed",
    "attempt",
    "elapsed_seconds",
    "clamp_count",
    "terminal_reason",
    "fault_reason",
    "primary_fault_reason",
    "cleanup_fault_reason",
    "audit_fault_reason",
}


class AdvancingClock:
    def __init__(self, start: float = 100.0, step: float = 0.01) -> None:
        self.value = start
        self.step = step

    def __call__(self) -> float:
        value = self.value
        self.value += self.step
        return value


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.last = values[-1]

    def __call__(self) -> float:
        try:
            self.last = next(self.values)
        except StopIteration:
            self.last += 0.01
        return self.last


class FakeRobot:
    """Stateful in-memory implementation of the complete robot boundary."""

    def __init__(
        self,
        *,
        state_deg: np.ndarray | None = None,
        captured_monotonic_s: float = 100.0,
        stop_after_observation: Event | None = None,
        disconnect_error: Exception | None = None,
        send_error_after_motion: Exception | None = None,
        audit_path: Path | None = None,
    ) -> None:
        self.state_deg = (
            np.zeros(7, dtype=float)
            if state_deg is None
            else np.asarray(state_deg, dtype=float).copy()
        )
        self.captured_monotonic_s = captured_monotonic_s
        self.stop_after_observation = stop_after_observation
        self.disconnect_error = disconnect_error
        self.send_error_after_motion = send_error_after_motion
        self.audit_path = audit_path
        self.connected = False
        self.connect_count = 0
        self.disconnect_count = 0
        self.observation_states: list[np.ndarray] = []
        self.sent_actions: list[np.ndarray] = []
        self.event_seen_before_send: str | None = None
        self.events_seen_before_disconnect: list[str] = []

    def connect(self) -> None:
        if self.connected:
            raise RuntimeError("robot is already connected")
        self.connected = True
        self.connect_count += 1

    def disconnect(self) -> None:
        if self.audit_path is not None and self.audit_path.exists():
            rows = [
                json.loads(line) for line in self.audit_path.read_text().splitlines()
            ]
            self.events_seen_before_disconnect = [row["event"] for row in rows]
        self.connected = False
        self.disconnect_count += 1
        if self.disconnect_error is not None:
            raise self.disconnect_error

    def observe(self) -> RolloutObservation:
        if not self.connected:
            raise RuntimeError("robot is disconnected")
        state = self.state_deg.copy()
        self.observation_states.append(state.copy())
        if self.stop_after_observation is not None:
            self.stop_after_observation.set()
        return RolloutObservation(
            front=np.zeros((2, 2, 3), dtype=np.uint8),
            side=np.zeros((2, 2, 3), dtype=np.uint8),
            state_deg=state,
            task=TASK,
            captured_monotonic_s=self.captured_monotonic_s,
        )

    def send_action(self, action_deg: np.ndarray) -> np.ndarray:
        if not self.connected:
            raise RuntimeError("robot is disconnected")
        if self.audit_path is not None:
            rows = [
                json.loads(line) for line in self.audit_path.read_text().splitlines()
            ]
            self.event_seen_before_send = rows[-1]["event"]
        actual = np.asarray(action_deg, dtype=float).copy()
        self.sent_actions.append(actual.copy())
        self.state_deg = actual.copy()
        if self.send_error_after_motion is not None:
            raise self.send_error_after_motion
        return actual


class MalformedStateRobot(FakeRobot):
    def observe(self) -> RolloutObservation:
        observation = super().observe()
        return RolloutObservation(
            front=observation.front,
            side=observation.side,
            state_deg=np.array(["not-a-number"] * 7, dtype=object),
            task=observation.task,
            captured_monotonic_s=observation.captured_monotonic_s,
        )


class FailingAuditFile:
    """File-like audit sink that fails one write or flush operation."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        self.failure_pending = True
        self.pending: list[str] = []
        self.durable: list[str] = []

    def __enter__(self) -> FailingAuditFile:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def write(self, text: str) -> int:
        if self.operation == "write" and self.failure_pending:
            self.failure_pending = False
            raise OSError("simulated audit write failure")
        if self.operation == "terminal_write" and self.failure_pending:
            row = json.loads(text)
            if row.get("event") == "terminal":
                self.failure_pending = False
                raise OSError("simulated terminal audit write failure")
        self.pending.append(text)
        return len(text)

    def flush(self) -> None:
        if self.operation == "flush" and self.failure_pending:
            self.failure_pending = False
            self.pending.clear()
            raise OSError("simulated audit flush failure")
        self.durable.extend(self.pending)
        self.pending.clear()

    def close(self) -> None:
        self.flush()

    def rows(self) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for chunk in self.durable
            for line in chunk.splitlines()
        ]


class ArrayPolicy:
    def __init__(self, prediction: np.ndarray) -> None:
        self.prediction = prediction
        self.calls = 0

    def predict(self, observation: RolloutObservation) -> np.ndarray:
        self.calls += 1
        return self.prediction


class RecedingHorizonPolicy:
    def __init__(self, first_action_deltas: list[float]) -> None:
        self.first_action_deltas = first_action_deltas
        self.observed_states: list[np.ndarray] = []

    def predict(self, observation: RolloutObservation) -> np.ndarray:
        state = observation.state_deg.copy()
        self.observed_states.append(state)
        prediction = np.repeat(state[None, :], 10, axis=0)
        prediction[0] = state + self.first_action_deltas[len(self.observed_states) - 1]
        prediction[1:] = state + 100.0
        return prediction


class RaisingPolicy:
    def predict(self, observation: RolloutObservation) -> np.ndarray:
        raise RuntimeError("inference exploded")


class MutatingPolicy:
    def predict(self, observation: RolloutObservation) -> np.ndarray:
        observation.state_deg[:] = 100.0
        return np.repeat(observation.state_deg[None, :], 10, axis=0)


class RecordingActionGuard:
    def __init__(
        self,
        events: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.events = events
        self.error = error
        self.actions: list[np.ndarray] = []

    def validate(self, action_deg: np.ndarray) -> None:
        if self.events is not None:
            self.events.append("workspace")
        self.actions.append(np.asarray(action_deg, dtype=float).copy())
        action_deg[:] = 999.0
        if self.error is not None:
            raise self.error


class RolloutRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.log_path = Path(temporary.name) / "rollout.jsonl"

    def make_runner(
        self,
        *,
        policy: object,
        robot: FakeRobot,
        mode: str = "shadow",
        stop_requested: Event | None = None,
        clock: AdvancingClock | SequenceClock | None = None,
        action_guard: object | None = None,
    ) -> RolloutRunner:
        return RolloutRunner(
            policy=policy,
            robot=robot,
            safety=SafetyGovernor(LIMITS, mode=mode),
            mode=mode,
            log_path=self.log_path,
            monotonic_clock=clock or AdvancingClock(),
            stop_requested=stop_requested or Event(),
            action_guard=action_guard,
        )

    def read_log(self) -> list[dict[str, object]]:
        return [json.loads(line) for line in self.log_path.read_text().splitlines()]

    def assert_full_fallback_schema(
        self,
        row: dict[str, object],
        *,
        event: str,
    ) -> None:
        self.assertEqual(set(row), LOG_FIELDS | {"failed_event"})
        self.assertEqual(row["event"], event)
        self.assertTrue(str(row["timestamp_utc"]).endswith("Z"))
        for field in (
            "monotonic_s",
            "task",
            "current_state_deg",
            "predicted_first_action_deg",
            "safety_result",
            "send_result_deg",
            "inference_latency_s",
            "primary_fault_reason",
            "cleanup_fault_reason",
        ):
            with self.subTest(field=field):
                self.assertIsNone(row[field])
        self.assertIsNotNone(row["fault_reason"])
        self.assertIsNotNone(row["audit_fault_reason"])

    def test_shadow_mode_never_sends_an_action(self) -> None:
        robot = FakeRobot()
        runner = self.make_runner(policy=HoldPositionPolicy(), robot=robot)

        summary = runner.run(max_cycles=2)

        self.assertEqual(summary.cycles_completed, 2)
        self.assertEqual(summary.actions_sent, 0)
        self.assertEqual(robot.sent_actions, [])

    def test_public_run_rejects_unbounded_or_nonpositive_cycle_limits(self) -> None:
        invalid_limits = (None, True, False, 0, -1, 1.0, "1")
        for invalid in invalid_limits:
            with self.subTest(max_cycles=invalid):
                stop = Event()
                stop.set()
                robot = FakeRobot()
                runner = self.make_runner(
                    policy=HoldPositionPolicy(),
                    robot=robot,
                    stop_requested=stop,
                )

                with self.assertRaisesRegex(ValueError, "positive integer"):
                    runner.run(invalid)  # type: ignore[arg-type]

                self.assertEqual(robot.connect_count, 0)
                self.assertEqual(robot.disconnect_count, 0)

    def test_shadow_invokes_workspace_guard_on_copy_before_safety_and_never_sends(self) -> None:
        events: list[str] = []
        guard = RecordingActionGuard(events)
        robot = FakeRobot()
        runner = self.make_runner(
            policy=HoldPositionPolicy(),
            robot=robot,
            action_guard=guard,
        )
        original_validate = runner.safety.validate

        def safety_validate(*args, **kwargs):
            events.append("safety")
            return original_validate(*args, **kwargs)

        runner.safety.validate = safety_validate  # type: ignore[method-assign]

        summary = runner.run(max_cycles=1)

        self.assertEqual(events, ["workspace", "safety"])
        self.assertEqual(len(guard.actions), 1)
        np.testing.assert_array_equal(guard.actions[0], np.zeros(7))
        self.assertEqual(summary.cycles_completed, 1)
        self.assertEqual(robot.sent_actions, [])

    def test_workspace_rejection_faults_shadow_and_live_before_safety_or_send(self) -> None:
        for mode in ("shadow", "live"):
            with self.subTest(mode=mode):
                robot = FakeRobot()
                guard = RecordingActionGuard(
                    error=WorkspaceViolation("tip outside calibrated polygon")
                )
                runner = self.make_runner(
                    policy=HoldPositionPolicy(),
                    robot=robot,
                    mode=mode,
                    action_guard=guard,
                )
                safety_calls = 0

                def unexpected_safety(*args, **kwargs):
                    nonlocal safety_calls
                    safety_calls += 1
                    raise AssertionError("safety must run after workspace validation")

                runner.safety.validate = unexpected_safety  # type: ignore[method-assign]

                summary = runner.run(max_cycles=1)

                self.assertEqual(summary.terminal_reason, "fault")
                self.assertIn("workspace safety", summary.primary_fault_reason or "")
                self.assertEqual(safety_calls, 0)
                self.assertEqual(robot.sent_actions, [])
                self.assertIn(
                    "workspace_safety_fault",
                    [row["event"] for row in self.read_log()],
                )
                self.log_path.unlink()

    def test_invalid_prediction_shape_does_not_invoke_workspace_guard(self) -> None:
        guard = RecordingActionGuard()
        robot = FakeRobot()
        runner = self.make_runner(
            policy=ArrayPolicy(np.zeros((1, 6))),
            robot=robot,
            action_guard=guard,
        )

        summary = runner.run(max_cycles=1)

        self.assertIn("prediction shape", summary.primary_fault_reason or "")
        self.assertEqual(guard.actions, [])

    def test_hold_position_policy_completes_multiple_fresh_observation_cycles(
        self,
    ) -> None:
        robot = FakeRobot()
        runner = self.make_runner(policy=HoldPositionPolicy(), robot=robot)

        summary = runner.run(max_cycles=3)

        self.assertEqual(summary.terminal_reason, "max_cycles")
        self.assertIsNone(summary.fault_reason)
        self.assertEqual(summary.cycles_completed, 3)
        self.assertEqual(len(robot.observation_states), 3)
        self.assertEqual(robot.disconnect_count, 1)

    def test_wrong_shape_or_empty_prediction_faults_before_motion(self) -> None:
        bad_predictions = (
            np.zeros(7),
            np.zeros((1, 6)),
            np.zeros((0, 7)),
            np.zeros((1, 7, 1)),
        )

        for prediction in bad_predictions:
            with self.subTest(shape=prediction.shape):
                robot = FakeRobot()
                runner = self.make_runner(
                    policy=ArrayPolicy(prediction),
                    robot=robot,
                )

                summary = runner.run(max_cycles=1)

                self.assertEqual(summary.terminal_reason, "fault")
                self.assertIn("prediction shape", summary.fault_reason or "")
                self.assertEqual(robot.sent_actions, [])
                self.assertEqual(robot.disconnect_count, 1)

    def test_stale_observation_faults_before_motion(self) -> None:
        robot = FakeRobot(captured_monotonic_s=0.0)
        runner = self.make_runner(
            policy=HoldPositionPolicy(),
            robot=robot,
            mode="live",
        )

        summary = runner.run(max_cycles=1)

        self.assertEqual(summary.terminal_reason, "fault")
        self.assertIn("stale", summary.fault_reason or "")
        self.assertEqual(robot.sent_actions, [])

    def test_policy_cannot_mutate_current_state_to_bypass_live_safety(self) -> None:
        robot = FakeRobot()
        runner = self.make_runner(
            policy=MutatingPolicy(),
            robot=robot,
            mode="live",
        )

        summary = runner.run(max_cycles=1)

        self.assertEqual(summary.terminal_reason, "fault")
        self.assertIn("delta", summary.primary_fault_reason or "")
        self.assertEqual(summary.actions_attempted, 0)
        self.assertEqual(robot.sent_actions, [])
        observation_row = next(
            row for row in self.read_log() if row["event"] == "observation"
        )
        self.assertEqual(observation_row["current_state_deg"], [0.0] * 7)

    def test_freshness_is_sampled_immediately_before_initial_safety(self) -> None:
        clock = SequenceClock([100.0, 100.0, 100.05, 100.30, 100.31])
        robot = FakeRobot()
        runner = self.make_runner(
            policy=HoldPositionPolicy(),
            robot=robot,
            mode="live",
            clock=clock,
        )

        summary = runner.run(max_cycles=1)

        self.assertIn("stale", summary.primary_fault_reason or "")
        self.assertEqual(summary.actions_attempted, 0)
        self.assertEqual(robot.sent_actions, [])

    def test_live_send_boundary_rechecks_observation_freshness(self) -> None:
        clock = SequenceClock(
            [100.0, 100.0, 100.05, 100.10, 100.20, 100.30, 100.31]
        )
        robot = FakeRobot()
        runner = self.make_runner(
            policy=HoldPositionPolicy(),
            robot=robot,
            mode="live",
            clock=clock,
        )

        summary = runner.run(max_cycles=1)

        self.assertIn("stale", summary.primary_fault_reason or "")
        self.assertEqual(summary.actions_attempted, 0)
        self.assertEqual(summary.actions_confirmed, 0)
        self.assertEqual(robot.sent_actions, [])
        events = [row["event"] for row in self.read_log()]
        self.assertIn("send_intent", events)
        self.assertIn("send_cancelled", events)

    def test_stop_event_exits_and_disconnects_cleanly(self) -> None:
        stop_requested = Event()
        stop_requested.set()
        robot = FakeRobot()
        runner = self.make_runner(
            policy=HoldPositionPolicy(),
            robot=robot,
            stop_requested=stop_requested,
        )

        summary = runner.run(max_cycles=5)

        self.assertEqual(summary.terminal_reason, "stop_requested")
        self.assertIsNone(summary.fault_reason)
        self.assertEqual(summary.cycles_completed, 0)
        self.assertEqual(robot.observation_states, [])
        self.assertEqual(robot.disconnect_count, 1)

    def test_stop_event_after_observation_prevents_inference_and_motion(self) -> None:
        stop_requested = Event()
        policy = ArrayPolicy(np.zeros((10, 7)))
        robot = FakeRobot(stop_after_observation=stop_requested)
        runner = self.make_runner(
            policy=policy,
            robot=robot,
            mode="live",
            stop_requested=stop_requested,
        )

        summary = runner.run(max_cycles=5)

        self.assertEqual(summary.terminal_reason, "stop_requested")
        self.assertEqual(policy.calls, 0)
        self.assertEqual(robot.sent_actions, [])
        self.assertEqual(robot.disconnect_count, 1)

    def test_live_mode_sends_only_an_accepted_first_action(self) -> None:
        policy = RecedingHorizonPolicy([1.0, 2.0])
        robot = FakeRobot()
        runner = self.make_runner(policy=policy, robot=robot, mode="live")

        summary = runner.run(max_cycles=2)

        self.assertEqual(summary.actions_sent, 1)
        self.assertEqual(summary.actions_attempted, 1)
        self.assertEqual(summary.actions_confirmed, 1)
        self.assertEqual(summary.cycles_completed, 1)
        self.assertIn("safety rejected", summary.fault_reason or "")
        self.assertEqual(len(robot.sent_actions), 1)
        np.testing.assert_array_equal(robot.sent_actions[0], np.ones(7))
        np.testing.assert_array_equal(policy.observed_states[0], np.zeros(7))
        np.testing.assert_array_equal(policy.observed_states[1], np.ones(7))

    def test_disconnect_runs_when_inference_raises(self) -> None:
        robot = FakeRobot()
        runner = self.make_runner(policy=RaisingPolicy(), robot=robot)

        summary = runner.run(max_cycles=1)

        self.assertEqual(summary.terminal_reason, "fault")
        self.assertIn("inference exploded", summary.fault_reason or "")
        self.assertEqual(robot.disconnect_count, 1)
        self.assertEqual(robot.sent_actions, [])
        fault_row = next(row for row in self.read_log() if row["event"] == "fault")
        self.assertEqual(fault_row["event"], "fault")
        self.assertIsNotNone(fault_row["inference_latency_s"])

    def test_send_intent_is_durable_before_adapter_failure_after_motion(self) -> None:
        robot = FakeRobot(
            send_error_after_motion=RuntimeError("transport acknowledgement lost"),
            audit_path=self.log_path,
        )
        runner = self.make_runner(
            policy=HoldPositionPolicy(),
            robot=robot,
            mode="live",
        )

        summary = runner.run(max_cycles=1)

        self.assertEqual(robot.event_seen_before_send, "send_intent")
        self.assertEqual(len(robot.sent_actions), 1)
        self.assertEqual(summary.actions_attempted, 1)
        self.assertEqual(summary.actions_confirmed, 0)
        self.assertEqual(summary.actions_sent, 0)
        events = [row["event"] for row in self.read_log()]
        self.assertLess(events.index("send_intent"), events.index("send_failed"))
        self.assertEqual(events[-1], "terminal")
        self.assertIn("acknowledgement lost", summary.primary_fault_reason or "")

    def test_jsonl_events_have_full_schema_and_do_not_mutate_arrays(self) -> None:
        state = np.zeros(7)
        prediction = np.repeat(state[None, :], 10, axis=0)
        untouched_state = state.copy()
        untouched_prediction = prediction.copy()
        robot = FakeRobot(state_deg=state)
        runner = self.make_runner(
            policy=ArrayPolicy(prediction),
            robot=robot,
            mode="live",
        )

        runner.run(max_cycles=1)

        rows = self.read_log()
        self.assertEqual(
            [row["event"] for row in rows],
            [
                "observation",
                "prediction",
                "safety",
                "send_intent",
                "send_boundary_safety",
                "send_confirmed",
                "terminal",
            ],
        )
        for row in rows:
            self.assertEqual(set(row), LOG_FIELDS)
            self.assertTrue(str(row["timestamp_utc"]).endswith("Z"))
            self.assertEqual(row["mode"], "live")
        self.assertEqual(rows[1]["predicted_first_action_deg"], [0.0] * 7)
        self.assertEqual(rows[2]["safety_result"]["accepted"], True)
        self.assertEqual(rows[5]["send_result_deg"], [0.0] * 7)
        self.assertEqual(rows[5]["actions_attempted"], 1)
        self.assertEqual(rows[5]["actions_confirmed"], 1)
        self.assertEqual(rows[-1]["terminal_reason"], "max_cycles")
        np.testing.assert_array_equal(state, untouched_state)
        np.testing.assert_array_equal(prediction, untouched_prediction)

    def test_unsafe_dummy_policy_is_rejected_without_invalid_json(self) -> None:
        robot = FakeRobot()
        runner = self.make_runner(policy=UnsafePolicy(), robot=robot, mode="live")

        summary = runner.run(max_cycles=1)

        self.assertEqual(summary.terminal_reason, "fault")
        self.assertIn("finite", summary.fault_reason or "")
        self.assertEqual(robot.sent_actions, [])
        prediction_row = next(
            row for row in self.read_log() if row["event"] == "prediction"
        )
        self.assertEqual(prediction_row["predicted_first_action_deg"], [None] * 7)

    def test_malformed_observation_state_returns_a_structured_fault(self) -> None:
        robot = MalformedStateRobot()
        runner = self.make_runner(
            policy=HoldPositionPolicy(),
            robot=robot,
            mode="live",
        )

        summary = runner.run(max_cycles=1)

        self.assertEqual(summary.terminal_reason, "fault")
        self.assertIn("observation state", summary.primary_fault_reason or "")
        self.assertEqual(robot.sent_actions, [])
        self.assertEqual(robot.disconnect_count, 1)
        rows = self.read_log()
        self.assertEqual(rows[-1]["event"], "terminal")
        self.assertEqual(rows[-1]["current_state_deg"], None)

    def test_audit_write_or_flush_failure_uses_guarded_fallback(self) -> None:
        for operation in ("write", "flush"):
            with self.subTest(operation=operation):
                audit_file = FailingAuditFile(operation)
                robot = FakeRobot()
                runner = self.make_runner(
                    policy=HoldPositionPolicy(),
                    robot=robot,
                    mode="live",
                )

                with patch.object(Path, "open", return_value=audit_file):
                    summary = runner.run(max_cycles=1)

                self.assertEqual(summary.terminal_reason, "fault")
                self.assertIn("audit log", summary.audit_fault_reason or "")
                self.assertEqual(summary.actions_attempted, 0)
                self.assertEqual(robot.sent_actions, [])
                self.assertEqual(robot.disconnect_count, 1)
                rows = audit_file.rows()
                fallback_row = next(
                    row for row in rows if row["event"] == "fault_fallback"
                )
                self.assert_full_fallback_schema(
                    fallback_row,
                    event="fault_fallback",
                )
                self.assertEqual(rows[-1]["event"], "terminal")

    def test_terminal_fallback_retains_the_full_safe_schema(self) -> None:
        audit_file = FailingAuditFile("terminal_write")
        robot = FakeRobot()
        runner = self.make_runner(policy=HoldPositionPolicy(), robot=robot)

        with patch.object(Path, "open", return_value=audit_file):
            summary = runner.run(max_cycles=1)

        self.assertEqual(summary.terminal_reason, "fault")
        fallback_row = next(
            row
            for row in audit_file.rows()
            if row["event"] == "terminal_fallback"
        )
        self.assert_full_fallback_schema(
            fallback_row,
            event="terminal_fallback",
        )
        self.assertEqual(fallback_row["terminal_reason"], "fault")

    def test_disconnect_fault_is_separate_and_terminal_is_after_cleanup(self) -> None:
        robot = FakeRobot(
            disconnect_error=RuntimeError("disconnect transport failed"),
            audit_path=self.log_path,
        )
        runner = self.make_runner(policy=RaisingPolicy(), robot=robot)

        summary = runner.run(max_cycles=1)

        self.assertIn("inference exploded", summary.primary_fault_reason or "")
        self.assertIn("disconnect transport", summary.cleanup_fault_reason or "")
        self.assertIn("inference exploded", summary.fault_reason or "")
        self.assertEqual(robot.disconnect_count, 1)
        self.assertNotIn("terminal", robot.events_seen_before_disconnect)
        rows = self.read_log()
        terminal_rows = [row for row in rows if row["event"] == "terminal"]
        self.assertEqual(len(terminal_rows), 1)
        self.assertIs(rows[-1], terminal_rows[0])
        self.assertIn(
            "disconnect transport", terminal_rows[0]["cleanup_fault_reason"]
        )


class DummyPolicyTest(unittest.TestCase):
    def test_hold_position_returns_ten_independent_state_copies(self) -> None:
        state = np.arange(7, dtype=float)
        observation = RolloutObservation(
            front=np.zeros((1, 1, 3)),
            side=np.zeros((1, 1, 3)),
            state_deg=state,
            task=TASK,
            captured_monotonic_s=1.0,
        )

        prediction = HoldPositionPolicy().predict(observation)

        self.assertEqual(prediction.shape, (10, 7))
        np.testing.assert_array_equal(prediction, np.repeat(state[None, :], 10, axis=0))
        prediction[0, 0] = -999.0
        self.assertEqual(state[0], 0.0)
        self.assertEqual(prediction[1, 0], 0.0)

    def test_unsafe_policy_returns_ten_nan_actions(self) -> None:
        observation = RolloutObservation(
            front=np.zeros((1, 1, 3)),
            side=np.zeros((1, 1, 3)),
            state_deg=np.zeros(7),
            task=TASK,
            captured_monotonic_s=1.0,
        )

        prediction = UnsafePolicy().predict(observation)

        self.assertEqual(prediction.shape, (10, 7))
        self.assertTrue(np.all(np.isnan(prediction)))


if __name__ == "__main__":
    unittest.main()
