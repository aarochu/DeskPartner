"""Pure interfaces shared by rollout policies, robots, and runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RolloutObservation:
    front: np.ndarray
    side: np.ndarray
    state_deg: np.ndarray
    task: str
    captured_monotonic_s: float


class PolicyAdapter(Protocol):
    def predict(self, observation: RolloutObservation) -> np.ndarray: ...


class RobotAdapter(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def observe(self) -> RolloutObservation: ...

    def send_action(self, action_deg: np.ndarray) -> np.ndarray: ...
