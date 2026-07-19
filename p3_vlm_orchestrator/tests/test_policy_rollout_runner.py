from __future__ import annotations

import json
from pathlib import Path
import tempfile
from threading import Event
import unittest

import numpy as np

from p3_vlm_orchestrator.policy_rollout.dummy_policy import (
    HoldPositionPolicy,
    UnsafePolicy,
)
from p3_vlm_orchestrator.policy_rollout.runner import RolloutRunner
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
    "terminal_reason",
    "fault_reason",
}


class AdvancingClock:
    def __init__(self, start: float = 100.0, step: float = 0.01) -> None:
        self.value = start
        self.step = step

    def __call__(self) -> float:
        value = self.value
        self.value += self.step
        return value


class FakeRobot:
    """Stateful in-memory implementation of the complete robot boundary."""

    def __init__(
        self,
        *,
        state_deg: np.ndarray | None = None,
        captured_monotonic_s: float = 100.0,
        stop_after_observation: Event | None = None,
    ) -> None:
        self.state_deg = (
            np.zeros(7, dtype=float)
            if state_deg is None
            else np.asarray(state_deg, dtype=float).copy()
        )
        self.captured_monotonic_s = captured_monotonic_s
        self.stop_after_observation = stop_after_observation
        self.connected = False
        self.connect_count = 0
        self.disconnect_count = 0
        self.observation_states: list[np.ndarray] = []
        self.sent_actions: list[np.ndarray] = []

    def connect(self) -> None:
        if self.connected:
            raise RuntimeError("robot is already connected")
        self.connected = True
        self.connect_count += 1

    def disconnect(self) -> None:
        self.connected = False
        self.disconnect_count += 1

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
        actual = np.asarray(action_deg, dtype=float).copy()
        self.sent_actions.append(actual.copy())
        self.state_deg = actual.copy()
        return actual


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
        clock: AdvancingClock | None = None,
    ) -> RolloutRunner:
        return RolloutRunner(
            policy=policy,
            robot=robot,
            safety=SafetyGovernor(LIMITS, mode=mode),
            mode=mode,
            log_path=self.log_path,
            monotonic_clock=clock or AdvancingClock(),
            stop_requested=stop_requested or Event(),
        )

    def read_log(self) -> list[dict[str, object]]:
        return [json.loads(line) for line in self.log_path.read_text().splitlines()]

    def test_shadow_mode_never_sends_an_action(self) -> None:
        robot = FakeRobot()
        runner = self.make_runner(policy=HoldPositionPolicy(), robot=robot)

        summary = runner.run(max_cycles=2)

        self.assertEqual(summary.cycles_completed, 2)
        self.assertEqual(summary.actions_sent, 0)
        self.assertEqual(robot.sent_actions, [])

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
        fault_row = self.read_log()[-1]
        self.assertEqual(fault_row["event"], "fault")
        self.assertIsNotNone(fault_row["inference_latency_s"])

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
            ["observation", "prediction", "safety", "send", "stop"],
        )
        for row in rows:
            self.assertEqual(set(row), LOG_FIELDS)
            self.assertTrue(str(row["timestamp_utc"]).endswith("Z"))
            self.assertEqual(row["mode"], "live")
        self.assertEqual(rows[1]["predicted_first_action_deg"], [0.0] * 7)
        self.assertEqual(rows[2]["safety_result"]["accepted"], True)
        self.assertEqual(rows[3]["send_result_deg"], [0.0] * 7)
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
