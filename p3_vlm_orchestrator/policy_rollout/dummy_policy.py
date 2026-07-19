"""Small deterministic policies for exercising the rollout harness."""

from __future__ import annotations

import numpy as np

from rebot_operator_kit.rollout.contracts import RolloutObservation


class HoldPositionPolicy:
    """Return a ten-step chunk that holds the observed follower pose."""

    def predict(self, observation: RolloutObservation) -> np.ndarray:
        return np.repeat(observation.state_deg[None, :], 10, axis=0)


class UnsafePolicy:
    """Return intentionally invalid actions for fail-closed smoke tests."""

    def predict(self, observation: RolloutObservation) -> np.ndarray:
        return np.full((10, 7), np.nan)
