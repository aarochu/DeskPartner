from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from rebot_operator_kit.rollout.safety import (
    SafetyDecision,
    SafetyFault,
    SafetyGovernor,
)


LIMITS = np.array(
    [
        [-145.0, 145.0],
        [-170.0, 0.0],
        [-200.0, 0.0],
        [-80.0, 90.0],
        [-90.0, 90.0],
        [-90.0, 90.0],
        [-270.0, 0.0],
    ]
)
IN_RANGE = np.array([0.0, -80.0, -100.0, 0.0, 0.0, 0.0, -100.0])


def profile_snapshot() -> dict[str, object]:
    return {
        "coordinate_contract": {
            "joints": [
                {"soft_limit_degrees": bounds.tolist()} for bounds in LIMITS
            ]
        }
    }


class SafetyGovernorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.shadow = SafetyGovernor.from_profile(profile_snapshot(), mode="shadow")
        self.live = SafetyGovernor.from_profile(profile_snapshot(), mode="live")

    def validate(
        self,
        governor: SafetyGovernor,
        current: np.ndarray = IN_RANGE,
        proposed: np.ndarray = IN_RANGE,
        *,
        now: float = 10.0,
        observed_at: float = 10.0,
    ) -> SafetyDecision:
        return governor.validate(current, proposed, now, observed_at)

    def assert_rejected_without_replacement(self, decision: SafetyDecision) -> None:
        self.assertFalse(decision.accepted)
        self.assertIsNone(decision.action_deg)
        self.assertFalse(decision.clamped)

    def test_rejects_current_or_proposed_shape_other_than_seven_vector(self) -> None:
        bad_shapes = (np.zeros(6), np.zeros((1, 7)), np.zeros(8))

        for field in ("current", "proposed"):
            for bad_value in bad_shapes:
                with self.subTest(field=field, shape=bad_value.shape):
                    governor = SafetyGovernor(LIMITS, mode="live")
                    values = {"current": IN_RANGE, "proposed": IN_RANGE}
                    values[field] = bad_value

                    decision = self.validate(governor, **values)

                    self.assert_rejected_without_replacement(decision)
                    self.assertIn("shape (7,)", decision.reason)

    def test_rejects_nan_or_infinity_without_replacement(self) -> None:
        for field in ("current", "proposed"):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(field=field, value=value):
                    governor = SafetyGovernor(LIMITS, mode="shadow")
                    values = {
                        "current": IN_RANGE.copy(),
                        "proposed": IN_RANGE.copy(),
                    }
                    values[field][0] = value

                    decision = self.validate(governor, **values)

                    self.assert_rejected_without_replacement(decision)
                    self.assertIn("finite", decision.reason)

    def test_rejects_observation_older_than_250_milliseconds(self) -> None:
        decision = self.validate(self.live, now=10.251, observed_at=10.0)

        self.assert_rejected_without_replacement(decision)
        self.assertIn("stale", decision.reason)

    def test_accepts_observation_at_250_millisecond_boundary(self) -> None:
        decision = self.validate(self.live, now=10.250, observed_at=10.0)

        self.assertTrue(decision.accepted)

    def test_rejects_current_or_proposed_value_outside_hard_limits(self) -> None:
        for field in ("current", "proposed"):
            with self.subTest(field=field):
                governor = SafetyGovernor(LIMITS, mode="shadow")
                values = {
                    "current": IN_RANGE.copy(),
                    "proposed": IN_RANGE.copy(),
                }
                values[field][0] = LIMITS[0, 1] + 0.1

                decision = self.validate(governor, **values)

                self.assert_rejected_without_replacement(decision)
                self.assertIn("hard limits", decision.reason)

    def test_shadow_mode_clamps_excessive_delta_and_accepts_copy(self) -> None:
        proposed = IN_RANGE.copy()
        proposed[[0, 3]] += np.array([3.0, -2.0])
        untouched_proposed = proposed.copy()

        decision = self.validate(self.shadow, proposed=proposed)

        self.assertTrue(decision.accepted)
        self.assertTrue(decision.clamped)
        np.testing.assert_allclose(
            decision.action_deg,
            IN_RANGE + np.array([1.5, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0]),
        )
        np.testing.assert_array_equal(proposed, untouched_proposed)
        self.assertIsNot(decision.action_deg, proposed)

    def test_live_mode_rejects_excessive_delta_without_replacement(self) -> None:
        proposed = IN_RANGE.copy()
        proposed[0] += 1.5001

        decision = self.validate(self.live, proposed=proposed)

        self.assert_rejected_without_replacement(decision)
        self.assertIn("delta", decision.reason)

    def test_valid_in_range_action_is_accepted_as_copy(self) -> None:
        proposed = IN_RANGE + np.array([1.5, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0])

        decision = self.validate(self.live, proposed=proposed)

        self.assertTrue(decision.accepted)
        self.assertFalse(decision.clamped)
        self.assertEqual(decision.reason, "accepted")
        np.testing.assert_array_equal(decision.action_deg, proposed)
        self.assertIsNot(decision.action_deg, proposed)

    def test_profile_soft_limits_are_enforced_as_hard_rollout_bounds(self) -> None:
        proposed = IN_RANGE.copy()
        proposed[6] = LIMITS[6, 0] - 0.1

        decision = self.validate(self.live, proposed=proposed)

        self.assert_rejected_without_replacement(decision)
        self.assertIn("hard limits", decision.reason)

    def test_third_consecutive_rejection_latches_fault(self) -> None:
        stale = {"now": 10.251, "observed_at": 10.0}
        for _ in range(2):
            self.assertFalse(self.validate(self.live, **stale).accepted)

        with self.assertRaises(SafetyFault):
            self.validate(self.live, **stale)
        with self.assertRaises(SafetyFault):
            self.validate(self.live)

    def test_third_consecutive_shadow_clamp_latches_fault(self) -> None:
        proposed = IN_RANGE.copy()
        proposed[0] += 2.0
        for _ in range(2):
            self.assertTrue(self.validate(self.shadow, proposed=proposed).clamped)

        with self.assertRaises(SafetyFault):
            self.validate(self.shadow, proposed=proposed)

    def test_normal_acceptance_resets_consecutive_interventions(self) -> None:
        stale = {"now": 10.251, "observed_at": 10.0}
        for _ in range(2):
            self.assertFalse(self.validate(self.live, **stale).accepted)

        self.assertTrue(self.validate(self.live).accepted)

        for _ in range(2):
            self.assertFalse(self.validate(self.live, **stale).accepted)
        self.assertTrue(self.validate(self.live).accepted)

    def test_rejects_unsupported_execution_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "shadow.*live"):
            SafetyGovernor(LIMITS, mode="offline")

    def test_safety_decision_is_frozen(self) -> None:
        decision = self.validate(self.live)

        with self.assertRaises(FrozenInstanceError):
            decision.accepted = False


if __name__ == "__main__":
    unittest.main()
