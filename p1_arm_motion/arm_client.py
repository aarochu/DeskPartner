"""Thin client over reBotArm_control_py.

Fill in against the SDK once the laptop is on Ubuntu with the arm live.
Until then, DryRunArmClient lets P3 integrate without hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from shared.types import ArmPose


@dataclass
class ArmConfig:
    follower_port: str
    home_joints_deg: list[float] | None
    transit_height_mm: float
    approach_height_mm: float
    max_speed: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> ArmConfig:
        raw = yaml.safe_load(Path(path).read_text())
        home = raw.get("home") or {}
        safety = raw.get("safety") or {}
        return cls(
            follower_port=raw["follower"]["port"],
            home_joints_deg=home.get("joints_deg"),
            transit_height_mm=float(safety.get("transit_height_mm", 120)),
            approach_height_mm=float(safety.get("approach_height_mm", 40)),
            max_speed=float(safety.get("max_speed", 0.15)),
        )


class DryRunArmClient:
    """Logs intended motions; no hardware."""

    def __init__(self, cfg: ArmConfig) -> None:
        self.cfg = cfg

    def connect(self) -> None:
        print(f"[dry-run] connect follower {self.cfg.follower_port}")

    def go_home(self) -> None:
        print(f"[dry-run] go_home joints={self.cfg.home_joints_deg}")

    def move_tip(self, pose: ArmPose, duration_s: float = 2.0) -> None:
        print(f"[dry-run] move_tip {pose} in {duration_s}s (max_speed={self.cfg.max_speed})")

    def set_gripper(self, closed: bool) -> None:
        print(f"[dry-run] gripper {'CLOSE' if closed else 'OPEN'}")

    def disconnect(self) -> None:
        print("[dry-run] disconnect")


class ReBotArmClient:
    """TODO: wrap ~/reBotArm_control_py IK + trajectory controllers."""

    def __init__(self, cfg: ArmConfig) -> None:
        self.cfg = cfg
        raise NotImplementedError(
            "Wire to reBotArm_control_py (7_arm_ik_control / 8_arm_traj_control). "
            "Use DryRunArmClient until hardware is up."
        )
