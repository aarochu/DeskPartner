from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from p3_vlm_orchestrator.policy_rollout.runner import RolloutRunner
from rebot_operator_kit.rollout.contracts import RolloutObservation
from rebot_operator_kit.rollout.safety import SafetyGovernor


TASK = "Pick up one can and place it in the taped sorting zone"
LIMITS = np.repeat(np.array([[-200.0, 200.0]]), 7, axis=0)


class MutableClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class SequenceVerdicts:
    def __init__(self, *values: str | None) -> None:
        self.values = iter(values)

    def __call__(self) -> str | None:
        return next(self.values, None)


class EpisodePolicy:
    def __init__(self, delta: float = 0.0) -> None:
        self.delta = delta
        self.calls = 0

    def predict(self, observation: RolloutObservation) -> np.ndarray:
        self.calls += 1
        action = np.asarray(observation.state_deg, dtype=float) + self.delta
        return np.repeat(action[None, :], 10, axis=0)


class EpisodeRobot:
    def __init__(
        self,
        clock: MutableClock,
        *,
        after_observe=None,
    ) -> None:
        self.clock = clock
        self.after_observe = after_observe
        self.state = np.zeros(7, dtype=float)
        self.connect_count = 0
        self.disconnect_count = 0
        self.observe_count = 0
        self.sent_actions: list[np.ndarray] = []

    def connect(self) -> None:
        self.connect_count += 1

    def disconnect(self) -> None:
        self.disconnect_count += 1

    def observe(self) -> RolloutObservation:
        self.observe_count += 1
        if self.after_observe is not None:
            self.after_observe()
        return RolloutObservation(
            front=np.zeros((2, 2, 3), dtype=np.uint8),
            side=np.zeros((2, 2, 3), dtype=np.uint8),
            state_deg=self.state.copy(),
            task=TASK,
            captured_monotonic_s=self.clock(),
        )

    def send_action(self, action_deg: np.ndarray) -> np.ndarray:
        action = np.asarray(action_deg, dtype=float).copy()
        self.sent_actions.append(action)
        self.state = action.copy()
        return action


class TerminalFailingAudit:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failed = False

    def write(self, text: str) -> int:
        row = json.loads(text)
        if row.get("event") == "terminal" and not self.failed:
            self.failed = True
            raise OSError("terminal audit failed")
        self.lines.append(text)
        return len(text)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def rows(self) -> list[dict[str, object]]:
        return [json.loads(line) for line in self.lines]


class EpisodeControlTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.log_path = Path(temporary.name) / "episode.jsonl"

    def make_runner(
        self,
        *,
        mode: str,
        policy: EpisodePolicy,
        robot: EpisodeRobot,
        clock: MutableClock,
        verdicts,
        stop: threading.Event | None = None,
    ) -> RolloutRunner:
        return RolloutRunner(
            policy=policy,
            robot=robot,
            safety=SafetyGovernor(LIMITS, mode=mode),
            mode=mode,
            log_path=self.log_path,
            monotonic_clock=clock,
            stop_requested=stop or threading.Event(),
            operator_verdict=verdicts,
        )

    def test_success_before_inference_sends_nothing_further(self) -> None:
        clock = MutableClock()
        robot = EpisodeRobot(clock)
        policy = EpisodePolicy()
        runner = self.make_runner(
            mode="live",
            policy=policy,
            robot=robot,
            clock=clock,
            verdicts=SequenceVerdicts("success"),
        )

        summary = runner.run_episode(attempt=1)

        self.assertEqual(summary.terminal_reason, "operator_success")
        self.assertEqual(policy.calls, 0)
        self.assertEqual(robot.sent_actions, [])
        self.assertEqual(robot.disconnect_count, 1)

    def test_failure_consumed_immediately_before_send_prevents_motion(self) -> None:
        clock = MutableClock()
        robot = EpisodeRobot(clock)
        policy = EpisodePolicy()
        runner = self.make_runner(
            mode="live",
            policy=policy,
            robot=robot,
            clock=clock,
            verdicts=SequenceVerdicts(None, None, "failure"),
        )

        summary = runner.run_episode(attempt=1)

        self.assertEqual(summary.terminal_reason, "operator_failure")
        self.assertEqual(policy.calls, 1)
        self.assertEqual(robot.sent_actions, [])

    def test_failure_arriving_during_final_safety_check_prevents_send(self) -> None:
        clock = MutableClock()
        robot = EpisodeRobot(clock)
        policy = EpisodePolicy()
        runner = self.make_runner(
            mode="live",
            policy=policy,
            robot=robot,
            clock=clock,
            verdicts=SequenceVerdicts(None, None, None, None, "failure"),
        )

        summary = runner.run_episode(attempt=1)

        self.assertEqual(summary.terminal_reason, "operator_failure")
        self.assertEqual(summary.actions_attempted, 0)
        self.assertEqual(robot.sent_actions, [])

    def test_stop_racing_success_wins_and_sends_nothing(self) -> None:
        clock = MutableClock()
        stop = threading.Event()
        robot = EpisodeRobot(clock)
        policy = EpisodePolicy()

        def success_and_stop() -> str:
            stop.set()
            return "success"

        runner = self.make_runner(
            mode="live",
            policy=policy,
            robot=robot,
            clock=clock,
            verdicts=success_and_stop,
            stop=stop,
        )

        summary = runner.run_episode(attempt=1)

        self.assertEqual(summary.terminal_reason, "stopped")
        self.assertEqual(policy.calls, 0)
        self.assertEqual(robot.sent_actions, [])

    def test_timeout_boundary_prevents_inference_and_motion_at_30_seconds(self) -> None:
        clock = MutableClock()
        robot = EpisodeRobot(clock, after_observe=lambda: setattr(clock, "value", 30.0))
        policy = EpisodePolicy()
        runner = self.make_runner(
            mode="live",
            policy=policy,
            robot=robot,
            clock=clock,
            verdicts=SequenceVerdicts(None, None),
        )

        summary = runner.run_episode(attempt=1)

        self.assertEqual(summary.terminal_reason, "timeout")
        self.assertEqual(summary.elapsed_seconds, 30.0)
        self.assertEqual(policy.calls, 0)
        self.assertEqual(robot.sent_actions, [])

    def test_action_cap_counts_confirmed_live_sends_and_never_sends_301(self) -> None:
        clock = MutableClock()
        robot = EpisodeRobot(clock)
        policy = EpisodePolicy()
        runner = self.make_runner(
            mode="live",
            policy=policy,
            robot=robot,
            clock=clock,
            verdicts=SequenceVerdicts(),
        )

        summary = runner.run_episode(attempt=1)

        self.assertEqual(summary.terminal_reason, "timeout")
        self.assertEqual(summary.actions_confirmed, 300)
        self.assertEqual(len(robot.sent_actions), 300)

    def test_safety_fault_is_publicly_distinguishable(self) -> None:
        clock = MutableClock()
        robot = EpisodeRobot(clock)
        policy = EpisodePolicy(delta=2.0)
        runner = self.make_runner(
            mode="live",
            policy=policy,
            robot=robot,
            clock=clock,
            verdicts=SequenceVerdicts(None, None),
        )

        summary = runner.run_episode(attempt=1)

        self.assertEqual(summary.terminal_reason, "safety_fault")
        self.assertIn("safety rejected", summary.primary_fault_reason or "")
        self.assertEqual(robot.sent_actions, [])

    def test_audit_open_fault_uses_the_episode_safety_fault_reason(self) -> None:
        clock = MutableClock()
        robot = EpisodeRobot(clock)
        policy = EpisodePolicy()
        runner = self.make_runner(
            mode="live",
            policy=policy,
            robot=robot,
            clock=clock,
            verdicts=SequenceVerdicts("success"),
        )
        runner.log_path = self.log_path.parent

        summary = runner.run_episode(attempt=1)

        self.assertEqual(summary.terminal_reason, "safety_fault")
        self.assertIn("audit log open", summary.audit_fault_reason or "")
        self.assertEqual(robot.sent_actions, [])

    def test_terminal_audit_fault_keeps_exact_episode_terminal_reason(self) -> None:
        clock = MutableClock()
        robot = EpisodeRobot(clock)
        policy = EpisodePolicy()
        audit = TerminalFailingAudit()
        runner = self.make_runner(
            mode="live",
            policy=policy,
            robot=robot,
            clock=clock,
            verdicts=SequenceVerdicts("success"),
        )

        with patch.object(Path, "open", return_value=audit):
            summary = runner.run_episode(attempt=1)

        self.assertEqual(summary.terminal_reason, "safety_fault")
        fallback = next(
            row for row in audit.rows() if row["event"] == "terminal_fallback"
        )
        self.assertEqual(fallback["terminal_reason"], "safety_fault")

    def test_shadow_episode_stays_send_free(self) -> None:
        clock = MutableClock()
        robot = EpisodeRobot(clock)
        policy = EpisodePolicy()
        runner = self.make_runner(
            mode="shadow",
            policy=policy,
            robot=robot,
            clock=clock,
            verdicts=SequenceVerdicts(None, None, "success"),
        )

        summary = runner.run_episode(attempt=2)

        self.assertEqual(summary.terminal_reason, "operator_success")
        self.assertEqual(policy.calls, 1)
        self.assertEqual(robot.sent_actions, [])

    def test_terminal_summary_and_audit_include_attempt_elapsed_and_clamps(self) -> None:
        clock = MutableClock()
        robot = EpisodeRobot(clock, after_observe=lambda: setattr(clock, "value", 1.0))
        policy = EpisodePolicy(delta=2.0)
        runner = self.make_runner(
            mode="shadow",
            policy=policy,
            robot=robot,
            clock=clock,
            verdicts=SequenceVerdicts(None, None, "success"),
        )

        summary = runner.run_episode(attempt=2)

        self.assertEqual(summary.attempt, 2)
        self.assertEqual(summary.elapsed_seconds, 1.0)
        self.assertEqual(summary.clamp_count, 1)
        terminal = json.loads(self.log_path.read_text().splitlines()[-1])
        self.assertEqual(terminal["event"], "terminal")
        self.assertEqual(terminal["attempt"], 2)
        self.assertEqual(terminal["elapsed_seconds"], 1.0)
        self.assertEqual(terminal["clamp_count"], 1)
        self.assertEqual(terminal["terminal_reason"], "operator_success")


if __name__ == "__main__":
    unittest.main()
