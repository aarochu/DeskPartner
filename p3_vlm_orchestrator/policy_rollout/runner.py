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
    actions_attempted: int
    actions_confirmed: int
    terminal_reason: str
    primary_fault_reason: str | None
    cleanup_fault_reason: str | None
    audit_fault_reason: str | None

    @property
    def actions_sent(self) -> int:
        """Backward-compatible alias for confirmed sends."""

        return self.actions_confirmed

    @property
    def fault_reason(self) -> str | None:
        """Return the primary fault without hiding a cleanup-only fault."""

        return (
            self.primary_fault_reason
            or self.audit_fault_reason
            or self.cleanup_fault_reason
        )


@dataclass
class _CycleContext:
    cycle: int = 0
    observation: RolloutObservation | None = None
    current_state_deg: np.ndarray | None = None
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
        actions_attempted = 0
        actions_confirmed = 0
        terminal_reason = "max_cycles"
        primary_fault_reason: str | None = None
        cleanup_fault_reason: str | None = None
        audit_fault_reason: str | None = None
        phase = "audit log open"
        context = _CycleContext()
        log_file: IO[str] | None = None

        def emit(event: str, *, monotonic_s: float | None = None) -> bool:
            nonlocal audit_fault_reason
            nonlocal primary_fault_reason
            nonlocal terminal_reason

            if log_file is None:
                return False
            error = self._write_event_guarded(
                log_file,
                event=event,
                context=context,
                actions_attempted=actions_attempted,
                actions_confirmed=actions_confirmed,
                terminal_reason=(terminal_reason if event == "terminal" else None),
                primary_fault_reason=primary_fault_reason,
                cleanup_fault_reason=cleanup_fault_reason,
                audit_fault_reason=audit_fault_reason,
                monotonic_s=monotonic_s,
            )
            if error is None:
                return True

            audit_fault_reason = f"audit log {event} failed: {error}"
            if primary_fault_reason is None:
                primary_fault_reason = audit_fault_reason
            terminal_reason = "fault"
            self._write_fallback_fault(
                log_file,
                failed_event=event,
                context=context,
                actions_attempted=actions_attempted,
                actions_confirmed=actions_confirmed,
                fault_reason=audit_fault_reason,
            )
            return False

        try:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = self.log_path.open("a", encoding="utf-8")
            except Exception as exc:
                audit_fault_reason = (
                    "audit log open failed: " + self._exception_text(exc)
                )
                primary_fault_reason = audit_fault_reason
                terminal_reason = "fault"

            if log_file is not None:
                phase = "connect"
                self.robot.connect()

                while cycles_completed < max_cycles:
                    context = _CycleContext(cycle=cycles_completed)
                    phase = "stop check"
                    if self.stop_requested():
                        terminal_reason = "stop_requested"
                        emit("stop_requested")
                        break

                    phase = "observation"
                    context.observation = self.robot.observe()
                    try:
                        current_state = np.asarray(
                            context.observation.state_deg, dtype=float
                        ).copy()
                    except (TypeError, ValueError) as exc:
                        primary_fault_reason = (
                            "observation state must be numeric: "
                            + self._exception_text(exc)
                        )
                        terminal_reason = "fault"
                        observation_logged_at = self.monotonic_clock()
                        if emit(
                            "observation", monotonic_s=observation_logged_at
                        ):
                            emit("fault", monotonic_s=observation_logged_at)
                        break

                    current_state.setflags(write=False)
                    context.current_state_deg = current_state
                    observation_logged_at = self.monotonic_clock()
                    if not emit(
                        "observation", monotonic_s=observation_logged_at
                    ):
                        break

                    phase = "stop check"
                    if self.stop_requested():
                        terminal_reason = "stop_requested"
                        emit("stop_requested")
                        break

                    policy_observation = RolloutObservation(
                        front=context.observation.front,
                        side=context.observation.side,
                        state_deg=current_state.copy(),
                        task=context.observation.task,
                        captured_monotonic_s=(
                            context.observation.captured_monotonic_s
                        ),
                    )
                    phase = "inference"
                    inference_started_at = self.monotonic_clock()
                    try:
                        raw_prediction = self.policy.predict(policy_observation)
                    except Exception as exc:
                        inference_finished_at = self.monotonic_clock()
                        context.inference_latency_s = max(
                            0.0, inference_finished_at - inference_started_at
                        )
                        primary_fault_reason = (
                            "inference failed: " + self._exception_text(exc)
                        )
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
                        primary_fault_reason = (
                            "prediction must be a numeric array: "
                            + self._exception_text(exc)
                        )
                        terminal_reason = "fault"
                        emit("fault", monotonic_s=inference_finished_at)
                        break

                    if prediction.ndim == 2 and prediction.shape[0] >= 1:
                        context.predicted_first_action = prediction[0].copy()
                    if not emit(
                        "prediction", monotonic_s=inference_finished_at
                    ):
                        break

                    if (
                        prediction.ndim != 2
                        or prediction.shape[0] < 1
                        or prediction.shape[1] != EXPECTED_ACTION_DIMENSION
                    ):
                        primary_fault_reason = (
                            "prediction shape must be [steps, 7] with at least "
                            f"one step; received {prediction.shape}"
                        )
                        terminal_reason = "fault"
                        emit("fault", monotonic_s=inference_finished_at)
                        break

                    phase = "stop check"
                    if self.stop_requested():
                        terminal_reason = "stop_requested"
                        emit("stop_requested")
                        break

                    phase = "safety validation"
                    safety_checked_at = self.monotonic_clock()
                    context.safety_result = self.safety.validate(
                        context.current_state_deg,
                        context.predicted_first_action,
                        safety_checked_at,
                        context.observation.captured_monotonic_s,
                    )
                    if not emit("safety", monotonic_s=safety_checked_at):
                        break

                    if (
                        not context.safety_result.accepted
                        or context.safety_result.action_deg is None
                    ):
                        primary_fault_reason = (
                            "safety rejected action: "
                            f"{context.safety_result.reason}"
                        )
                        terminal_reason = "fault"
                        emit("fault")
                        break

                    phase = "stop check"
                    if self.stop_requested():
                        terminal_reason = "stop_requested"
                        emit("stop_requested")
                        break

                    if self.mode == "live":
                        if not emit("send_intent"):
                            break
                        if self.stop_requested():
                            terminal_reason = "stop_requested"
                            emit("send_cancelled")
                            break

                        phase = "send boundary safety validation"
                        send_checked_at = self.monotonic_clock()
                        context.safety_result = self.safety.validate(
                            context.current_state_deg,
                            context.safety_result.action_deg,
                            send_checked_at,
                            context.observation.captured_monotonic_s,
                        )
                        if (
                            not context.safety_result.accepted
                            or context.safety_result.action_deg is None
                        ):
                            emit(
                                "send_boundary_safety",
                                monotonic_s=send_checked_at,
                            )
                            primary_fault_reason = (
                                "safety rejected action at live send boundary: "
                                f"{context.safety_result.reason}"
                            )
                            terminal_reason = "fault"
                            emit("send_cancelled")
                            emit("fault")
                            break

                        actions_attempted += 1
                        phase = "send"
                        try:
                            raw_send_result = self.robot.send_action(
                                context.safety_result.action_deg.copy()
                            )
                        except Exception as exc:
                            primary_fault_reason = (
                                "send failed: " + self._exception_text(exc)
                            )
                            terminal_reason = "fault"
                            emit(
                                "send_boundary_safety",
                                monotonic_s=send_checked_at,
                            )
                            emit("send_failed")
                            emit("fault")
                            break

                        try:
                            context.send_result = np.asarray(
                                raw_send_result, dtype=float
                            ).copy()
                        except (TypeError, ValueError) as exc:
                            primary_fault_reason = (
                                "send result must be numeric: "
                                + self._exception_text(exc)
                            )
                            terminal_reason = "fault"
                            emit(
                                "send_boundary_safety",
                                monotonic_s=send_checked_at,
                            )
                            emit("send_failed")
                            emit("fault")
                            break

                        actions_confirmed += 1
                        if not emit(
                            "send_boundary_safety",
                            monotonic_s=send_checked_at,
                        ):
                            break
                        if not emit("send_confirmed"):
                            break

                    cycles_completed += 1
        except Exception as exc:
            if primary_fault_reason is None:
                primary_fault_reason = (
                    f"{phase} failed: " + self._exception_text(exc)
                )
            terminal_reason = "fault"
            emit("fault")
        finally:
            try:
                self.robot.disconnect()
            except Exception as exc:
                cleanup_fault_reason = (
                    "disconnect failed: " + self._exception_text(exc)
                )
                terminal_reason = "fault"

            if log_file is not None:
                context.cycle = cycles_completed
                emit("terminal")
                try:
                    log_file.close()
                except Exception:
                    pass

        return RunSummary(
            mode=self.mode,
            cycles_completed=cycles_completed,
            actions_attempted=actions_attempted,
            actions_confirmed=actions_confirmed,
            terminal_reason=terminal_reason,
            primary_fault_reason=primary_fault_reason,
            cleanup_fault_reason=cleanup_fault_reason,
            audit_fault_reason=audit_fault_reason,
        )

    def _write_event_guarded(
        self,
        log_file: IO[str],
        **event_fields: object,
    ) -> str | None:
        try:
            self._write_event(log_file, **event_fields)
        except Exception as exc:
            return self._exception_text(exc)
        return None

    def _write_event(
        self,
        log_file: IO[str],
        *,
        event: str,
        context: _CycleContext,
        actions_attempted: int,
        actions_confirmed: int,
        terminal_reason: str | None,
        primary_fault_reason: str | None,
        cleanup_fault_reason: str | None,
        audit_fault_reason: str | None,
        monotonic_s: float | None = None,
    ) -> None:
        checked_monotonic_s = (
            self.monotonic_clock() if monotonic_s is None else monotonic_s
        )
        fault_reason = (
            primary_fault_reason or audit_fault_reason or cleanup_fault_reason
        )
        row = {
            "timestamp_utc": self._utc_timestamp(),
            "monotonic_s": float(checked_monotonic_s),
            "event": event,
            "mode": self.mode,
            "cycle": context.cycle,
            "task": (
                None if context.observation is None else context.observation.task
            ),
            "current_state_deg": self._json_array(context.current_state_deg),
            "predicted_first_action_deg": self._json_array(
                context.predicted_first_action
            ),
            "safety_result": self._json_safety_result(context.safety_result),
            "send_result_deg": self._json_array(context.send_result),
            "inference_latency_s": context.inference_latency_s,
            "actions_attempted": actions_attempted,
            "actions_confirmed": actions_confirmed,
            "terminal_reason": terminal_reason,
            "fault_reason": fault_reason,
            "primary_fault_reason": primary_fault_reason,
            "cleanup_fault_reason": cleanup_fault_reason,
            "audit_fault_reason": audit_fault_reason,
        }
        log_file.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        log_file.flush()

    def _write_fallback_fault(
        self,
        log_file: IO[str],
        *,
        failed_event: str,
        context: _CycleContext,
        actions_attempted: int,
        actions_confirmed: int,
        fault_reason: str,
    ) -> None:
        try:
            row = {
                "timestamp_utc": self._utc_timestamp(),
                "monotonic_s": None,
                "event": (
                    "terminal_fallback"
                    if failed_event == "terminal"
                    else "fault_fallback"
                ),
                "mode": self.mode,
                "cycle": context.cycle,
                "task": None,
                "current_state_deg": None,
                "predicted_first_action_deg": None,
                "safety_result": None,
                "send_result_deg": None,
                "inference_latency_s": None,
                "actions_attempted": actions_attempted,
                "actions_confirmed": actions_confirmed,
                "failed_event": failed_event,
                "fault_reason": fault_reason,
                "primary_fault_reason": None,
                "cleanup_fault_reason": None,
                "audit_fault_reason": fault_reason,
                "terminal_reason": (
                    "fault" if failed_event == "terminal" else None
                ),
            }
            log_file.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            log_file.flush()
        except Exception:
            pass

    @staticmethod
    def _exception_text(exc: Exception) -> str:
        try:
            return str(exc)
        except Exception:
            return type(exc).__name__

    @staticmethod
    def _utc_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

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
