"""Home pose helpers — photos only when arm is home."""

from __future__ import annotations

from p1_arm_motion.arm_client import DryRunArmClient


def go_home(arm: DryRunArmClient) -> None:
    arm.go_home()


def ensure_home_before_photo(arm: DryRunArmClient) -> None:
    """Contract with P3: never capture with arm in frame."""
    go_home(arm)
