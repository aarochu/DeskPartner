"""Home pose helpers — photos only when arm is home."""

from __future__ import annotations

from p1_arm_motion.arm_client import DryRunArmClient, RebotArmClient
from p1_arm_motion.pick_and_drop import move_to_home

Arm = DryRunArmClient | RebotArmClient


def go_home(arm: Arm) -> None:
    move_to_home(arm)


def ensure_home_before_photo(arm: Arm) -> None:
    """Contract with P3: never capture with arm in frame."""
    go_home(arm)
