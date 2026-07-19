"""Fail-closed validation for rollout actions before robot execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


JOINT_COUNT = 7
DEFAULT_MAX_OBSERVATION_AGE_S = 0.250
DEFAULT_MAX_ACTION_DELTA_DEG = 1.5
DEFAULT_MAX_CONSECUTIVE_INTERVENTIONS = 3


@dataclass(frozen=True)
class SafetyDecision:
    accepted: bool
    action_deg: np.ndarray | None
    reason: str
    clamped: bool = False


class SafetyFault(RuntimeError):
    """Raised once repeated safety interventions latch the governor."""


class SafetyGovernor:
    """Validate seven-joint actions against an authenticated profile."""

    def __init__(
        self,
        hard_limits_deg: Sequence[Sequence[float]] | np.ndarray,
        *,
        mode: str,
        max_observation_age_s: float = DEFAULT_MAX_OBSERVATION_AGE_S,
        max_action_delta_deg: float = DEFAULT_MAX_ACTION_DELTA_DEG,
        max_consecutive_interventions: int = DEFAULT_MAX_CONSECUTIVE_INTERVENTIONS,
    ) -> None:
        if mode not in ("shadow", "live"):
            raise ValueError("Safety governor mode must be exactly 'shadow' or 'live'")

        try:
            limits = np.asarray(hard_limits_deg, dtype=float).copy()
        except (TypeError, ValueError) as exc:
            raise ValueError("Safety hard limits must be numeric") from exc
        if limits.shape != (JOINT_COUNT, 2):
            raise ValueError("Safety hard limits must have shape (7, 2)")
        if not np.all(np.isfinite(limits)):
            raise ValueError("Safety hard limits must be finite")
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("Each safety hard-limit minimum must be below its maximum")

        if not np.isfinite(max_observation_age_s) or max_observation_age_s < 0:
            raise ValueError("Maximum observation age must be finite and nonnegative")
        if not np.isfinite(max_action_delta_deg) or max_action_delta_deg <= 0:
            raise ValueError("Maximum action delta must be finite and positive")
        if (
            isinstance(max_consecutive_interventions, bool)
            or not isinstance(max_consecutive_interventions, int)
            or max_consecutive_interventions <= 0
        ):
            raise ValueError("Maximum consecutive interventions must be positive")

        self.mode = mode
        self.hard_limits_deg = limits
        self.max_observation_age_s = float(max_observation_age_s)
        self.max_action_delta_deg = float(max_action_delta_deg)
        self.max_consecutive_interventions = max_consecutive_interventions
        self._consecutive_interventions = 0
        self._fault: SafetyFault | None = None

    @classmethod
    def from_profile(
        cls,
        profile_snapshot: Mapping[str, Any],
        *,
        mode: str,
        max_observation_age_s: float = DEFAULT_MAX_OBSERVATION_AGE_S,
        max_action_delta_deg: float = DEFAULT_MAX_ACTION_DELTA_DEG,
        max_consecutive_interventions: int = DEFAULT_MAX_CONSECUTIVE_INTERVENTIONS,
    ) -> SafetyGovernor:
        """Build a governor from the checkpoint's authenticated profile snapshot."""

        try:
            coordinate_contract = profile_snapshot["coordinate_contract"]
            joints = coordinate_contract["joints"]
            limits = [joint["soft_limit_degrees"] for joint in joints]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Profile must contain coordinate_contract joint soft limits"
            ) from exc

        return cls(
            limits,
            mode=mode,
            max_observation_age_s=max_observation_age_s,
            max_action_delta_deg=max_action_delta_deg,
            max_consecutive_interventions=max_consecutive_interventions,
        )

    def validate(
        self,
        current: np.ndarray,
        proposed: np.ndarray,
        now: float,
        observed_at: float,
    ) -> SafetyDecision:
        """Return an executable copy or reject and count a safety intervention."""

        if self._fault is not None:
            raise self._fault

        try:
            current_deg = np.asarray(current, dtype=float).copy()
            proposed_deg = np.asarray(proposed, dtype=float).copy()
        except (TypeError, ValueError):
            return self._reject("current and proposed joint values must be finite numbers")

        if current_deg.shape != (JOINT_COUNT,) or proposed_deg.shape != (JOINT_COUNT,):
            return self._reject("current and proposed actions must each have shape (7,)")
        if not np.all(np.isfinite(current_deg)) or not np.all(np.isfinite(proposed_deg)):
            return self._reject("current and proposed joint values must be finite")

        try:
            checked_now = float(now)
            checked_observed_at = float(observed_at)
        except (TypeError, ValueError):
            return self._reject("observation timestamps must be finite")
        if not np.isfinite(checked_now) or not np.isfinite(checked_observed_at):
            return self._reject("observation timestamps must be finite")

        observation_age = checked_now - checked_observed_at
        if observation_age < 0:
            return self._reject("observation timestamp is in the future")
        if observation_age > self.max_observation_age_s:
            return self._reject("observation is stale")

        lower = self.hard_limits_deg[:, 0]
        upper = self.hard_limits_deg[:, 1]
        if (
            np.any(current_deg < lower)
            or np.any(current_deg > upper)
            or np.any(proposed_deg < lower)
            or np.any(proposed_deg > upper)
        ):
            return self._reject("current or proposed action is outside hard limits")

        delta = proposed_deg - current_deg
        if np.any(np.abs(delta) > self.max_action_delta_deg):
            if self.mode == "live":
                return self._reject("action delta exceeds the live step cap")
            clamped = current_deg + np.clip(
                delta,
                -self.max_action_delta_deg,
                self.max_action_delta_deg,
            )
            return self._intervene(
                SafetyDecision(
                    accepted=True,
                    action_deg=clamped,
                    reason="action delta clamped to the live step cap",
                    clamped=True,
                )
            )

        self._consecutive_interventions = 0
        return SafetyDecision(
            accepted=True,
            action_deg=proposed_deg,
            reason="accepted",
        )

    def _reject(self, reason: str) -> SafetyDecision:
        return self._intervene(
            SafetyDecision(accepted=False, action_deg=None, reason=reason)
        )

    def _intervene(self, decision: SafetyDecision) -> SafetyDecision:
        self._consecutive_interventions += 1
        if self._consecutive_interventions >= self.max_consecutive_interventions:
            self._fault = SafetyFault(
                "Safety governor latched after "
                f"{self._consecutive_interventions} consecutive interventions"
            )
            raise self._fault
        return decision
