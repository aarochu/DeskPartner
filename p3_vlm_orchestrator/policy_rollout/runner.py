"""One-step receding-horizon policy rollout independent of concrete hardware."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import IO, Protocol

import numpy as np

from rebot_operator_kit.rollout.contracts import (
    PolicyAdapter,
    RobotAdapter,
    RolloutObservation,
)
from rebot_operator_kit.rollout.safety import SafetyDecision, SafetyGovernor


EXPECTED_ACTION_DIMENSION = 7


class _StopEvent(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class RunSummary:
    """Terminal outcome of one rollout run."""

    mode: str
    cycles_completed: int
    actions_sent: int
    terminal_reason: str
    fault_reason: str | None


@dataclass
class _CycleContext:
    cycle: int = 0
    observation: RolloutObservation | None = None
    predicted_first_action: np.ndarray | None = None
    safety_result: SafetyDecision | None = None
    send_result: np.ndarray | None = None
    inference_latency_s: float | None = None


class RolloutRunner:
    """Observe, infer a chunk, validate its first action, and optionally send it."""

    def __init__(
        self,
        *,
        policy: PolicyAdapter,
        robot: RobotAdapter,
        safety: SafetyGovernor,
        mode: str,
        log_path: Path | str,
        monotonic_clock: Callable[[], float],
        stop_requested: Callable[[], bool] | _StopEvent,
    ) -> None:
        if mode not in ("shadow", "live"):
            raise ValueError("Rollout mode must be exactly 'shadow' or 'live'")
        if safety.mode != mode:
            raise ValueError("Rollout mode must match the safety governor mode")

        self.policy = policy
        self.robot = robot
        self.safety = safety
        self.mode = mode
        self.log_path = Path(log_path)
        self.monotonic_clock = monotonic_clock
        self.stop_requested = (
            stop_requested if callable(stop_requested) else stop_requested.is_set
        )

    def run(self, max_cycles: int) -> RunSummary:
        """Run until the cycle limit, a stop request, or a fail-closed fault."""

        cycles_completed = 0
        actions_sent = 0
        terminal_reason = "max_cycles"
        fault_reason: str | None = None
        phase = "connect"
        context = _CycleContext()

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log_file:

            def emit(event: str, *, monotonic_s: float | None = None) -> None:
                self._write_event(
                    log_file,
                    event=event,
                    context=context,
                    terminal_reason=(
                        terminal_reason if event in ("stop", "fault") else None
                    ),
                    fault_reason=fault_reason if event == "fault" else None,
                    monotonic_s=monotonic_s,
                )

            try:
                self.robot.connect()

                while cycles_completed < max_cycles:
                    context.cycle = cycles_completed
                    phase = "stop check"
                    if self.stop_requested():
                        terminal_reason = "stop_requested"
                        emit("stop")
                        break

                    context = _CycleContext(cycle=cycles_completed)
                    phase = "observation"
                    context.observation = self.robot.observe()
                    observation_logged_at = self.monotonic_clock()
                    emit("observation", monotonic_s=observation_logged_at)

                    phase = "stop check"
                    if self.stop_requested():
                        terminal_reason = "stop_requested"
                        emit("stop")
                        break

                    phase = "inference"
                    inference_started_at = self.monotonic_clock()
                    try:
                        raw_prediction = self.policy.predict(context.observation)
                    except Exception as exc:
                        inference_finished_at = self.monotonic_clock()
                        context.inference_latency_s = max(
                            0.0, inference_finished_at - inference_started_at
                        )
                        fault_reason = f"inference failed: {exc}"
                        terminal_reason = "fault"
                        emit("fault", monotonic_s=inference_finished_at)
                        break

                    inference_finished_at = self.monotonic_clock()
                    context.inference_latency_s = max(
                        0.0, inference_finished_at - inference_started_at
                    )

                    try:
                        prediction = np.asarray(raw_prediction, dtype=float)
                    except (TypeError, ValueError) as exc:
                        fault_reason = f"prediction must be a numeric array: {exc}"
                        terminal_reason = "fault"
                        emit("fault", monotonic_s=inference_finished_at)
                        break

                    if prediction.ndim == 2 and prediction.shape[0] >= 1:
                        context.predicted_first_action = prediction[0].copy()
                    emit("prediction", monotonic_s=inference_finished_at)

                    if (
                        prediction.ndim != 2
                        or prediction.shape[0] < 1
                        or prediction.shape[1] != EXPECTED_ACTION_DIMENSION
                    ):
                        fault_reason = (
                            "prediction shape must be [steps, 7] with at least "
                            f"one step; received {prediction.shape}"
                        )
                        terminal_reason = "fault"
                        emit("fault", monotonic_s=inference_finished_at)
                        break

                    phase = "stop check"
                    if self.stop_requested():
                        terminal_reason = "stop_requested"
                        emit("stop")
                        break

                    phase = "safety validation"
                    context.safety_result = self.safety.validate(
                        context.observation.state_deg,
                        context.predicted_first_action,
                        inference_finished_at,
                        context.observation.captured_monotonic_s,
                    )
                    emit("safety", monotonic_s=inference_finished_at)

                    if (
                        not context.safety_result.accepted
                        or context.safety_result.action_deg is None
                    ):
                        fault_reason = (
                            "safety rejected action: "
                            f"{context.safety_result.reason}"
                        )
                        terminal_reason = "fault"
                        emit("fault")
                        break

                    phase = "stop check"
                    if self.stop_requested():
                        terminal_reason = "stop_requested"
                        emit("stop")
                        break

                    if self.mode == "live":
                        phase = "send"
                        context.send_result = np.asarray(
                            self.robot.send_action(
                                context.safety_result.action_deg.copy()
                            ),
                            dtype=float,
                        ).copy()
                        actions_sent += 1
                        emit("send")

                    cycles_completed += 1
                else:
                    context.cycle = cycles_completed
                    terminal_reason = "max_cycles"
                    emit("stop")
            except Exception as exc:
                terminal_reason = "fault"
                fault_reason = f"{phase} failed: {exc}"
                emit("fault")
            finally:
                try:
                    self.robot.disconnect()
                except Exception as exc:
                    terminal_reason = "fault"
                    fault_reason = f"disconnect failed: {exc}"
                    emit("fault")

        return RunSummary(
            mode=self.mode,
            cycles_completed=cycles_completed,
            actions_sent=actions_sent,
            terminal_reason=terminal_reason,
            fault_reason=fault_reason,
        )

    def _write_event(
        self,
        log_file: IO[str],
        *,
        event: str,
        context: _CycleContext,
        terminal_reason: str | None = None,
        fault_reason: str | None = None,
        monotonic_s: float | None = None,
    ) -> None:
        checked_monotonic_s = (
            self.monotonic_clock() if monotonic_s is None else monotonic_s
        )
        observation = context.observation
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "monotonic_s": float(checked_monotonic_s),
            "event": event,
            "mode": self.mode,
            "cycle": context.cycle,
            "task": None if observation is None else observation.task,
            "current_state_deg": self._json_array(
                None if observation is None else observation.state_deg
            ),
            "predicted_first_action_deg": self._json_array(
                context.predicted_first_action
            ),
            "safety_result": self._json_safety_result(context.safety_result),
            "send_result_deg": self._json_array(context.send_result),
            "inference_latency_s": context.inference_latency_s,
            "terminal_reason": terminal_reason,
            "fault_reason": fault_reason,
        }
        log_file.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        log_file.flush()

    @staticmethod
    def _json_array(value: np.ndarray | None) -> list[float | None] | None:
        if value is None:
            return None
        values = np.asarray(value, dtype=float).reshape(-1).tolist()
        return [float(item) if np.isfinite(item) else None for item in values]

    @classmethod
    def _json_safety_result(
        cls, decision: SafetyDecision | None
    ) -> dict[str, object] | None:
        if decision is None:
            return None
        return {
            "accepted": decision.accepted,
            "action_deg": cls._json_array(decision.action_deg),
            "clamped": decision.clamped,
            "reason": decision.reason,
        }
