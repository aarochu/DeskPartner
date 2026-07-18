"""Arm client wrapping reBotArm_control_py RebotArmEndPose.

Real hardware path uses move_to_ik / move_to_traj / open_gripper / close_gripper.
DryRunArmClient lets P2/P3 integrate without the arm.

SDK units: meters + radians. DeskPartner configs often use mm — convert at the boundary.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

log = logging.getLogger("p1.arm")


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


@dataclass
class ArmConfig:
    sdk_repo: Path
    hardware_yaml: str
    arm_control_mode: str
    home_joints_deg: list[float] | None
    home_xyz_m: list[float] | None
    home_rpy_rad: list[float]
    speed_multiplier: float
    base_duration_s: float
    slow_descend_duration_s: float
    transit_height_mm: float
    grasp_heights_mm: dict
    reach_tolerance_mm: float
    click_verify_tolerance_mm: float

    @classmethod
    def from_yaml(cls, path: str | Path = "config/arm.yaml") -> ArmConfig:
        raw = yaml.safe_load(Path(path).read_text())
        sdk = raw.get("sdk") or {}
        home = raw.get("home") or {}
        safety = raw.get("safety") or {}
        return cls(
            sdk_repo=_expand(sdk.get("repo", "~/reBotArm_control_py")),
            hardware_yaml=str(sdk.get("hardware_yaml", "rebotarm_dm.yaml")),
            arm_control_mode=str(sdk.get("arm_control_mode", "mit")),
            home_joints_deg=home.get("joints_deg"),
            home_xyz_m=home.get("xyz_m"),
            home_rpy_rad=list(home.get("rpy_rad") or [0.0, 0.0, 0.0]),
            speed_multiplier=float(safety.get("speed_multiplier", 0.5)),
            base_duration_s=float(safety.get("base_duration_s", 2.5)),
            slow_descend_duration_s=float(safety.get("slow_descend_duration_s", 3.5)),
            transit_height_mm=float(safety.get("transit_height_mm", 120)),
            grasp_heights_mm=dict(raw.get("grasp_heights_mm") or {"default": 20}),
            reach_tolerance_mm=float(safety.get("reach_tolerance_mm", 15)),
            click_verify_tolerance_mm=float(safety.get("click_verify_tolerance_mm", 5)),
        )

    def duration(self, slow: bool = False) -> float:
        base = self.slow_descend_duration_s if slow else self.base_duration_s
        mult = max(self.speed_multiplier, 0.05)
        return base / mult

    def grasp_z_mm(self, item_type: str) -> float:
        key = (item_type or "default").lower().replace(" ", "_")
        heights = self.grasp_heights_mm
        if key in heights:
            return float(heights[key])
        for name, val in heights.items():
            if name != "default" and name in key:
                return float(val)
        return float(heights.get("default", 20))


class DryRunArmClient:
    """Logs intended motions; no hardware."""

    def __init__(self, cfg: ArmConfig) -> None:
        self.cfg = cfg
        self._tip = [0.2, 0.0, 0.3]
        self._q = [0.0] * 6

    def connect(self) -> None:
        log.info("[dry-run] connect sdk=%s", self.cfg.sdk_repo)

    def disconnect(self) -> None:
        log.info("[dry-run] disconnect")

    def move_tip_m(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        duration_s: float | None = None,
        use_traj: bool = True,
    ) -> bool:
        dur = duration_s if duration_s is not None else self.cfg.duration()
        log.info(
            "[dry-run] move_tip_m xyz=(%.3f,%.3f,%.3f) rpy=(%.2f,%.2f,%.2f) dur=%.2fs traj=%s",
            x, y, z, roll, pitch, yaw, dur, use_traj,
        )
        self._tip = [x, y, z]
        return True

    def move_tip_mm(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        **kwargs,
    ) -> bool:
        return self.move_tip_m(x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0, **kwargs)

    def open_gripper(self) -> None:
        log.info("[dry-run] gripper OPEN")

    def close_gripper(self) -> None:
        log.info("[dry-run] gripper CLOSE")

    def get_tip_m(self) -> tuple[float, float, float, float, float, float]:
        return (*self._tip, 0.0, 0.0, 0.0)  # type: ignore[return-value]

    def get_joints_rad(self) -> list[float]:
        return list(self._q)

    def move_to_home(self) -> bool:
        if self.cfg.home_xyz_m:
            x, y, z = self.cfg.home_xyz_m
            r, p, yaw = self.cfg.home_rpy_rad
            return self.move_tip_m(x, y, z, r, p, yaw)
        log.info("[dry-run] move_to_home joints_deg=%s", self.cfg.home_joints_deg)
        return True


class RebotArmClient:
    """Live wrapper: RebotArm + RebotArmEndPose (no custom IK math)."""

    def __init__(self, cfg: ArmConfig) -> None:
        self.cfg = cfg
        self._arm = None
        self._ctrl = None
        self._ensure_sdk_on_path()

    def _ensure_sdk_on_path(self) -> None:
        repo = self.cfg.sdk_repo
        if not repo.exists():
            raise FileNotFoundError(
                f"reBot SDK not found at {repo}. Clone "
                "https://github.com/vectorBH6/reBotArm_control_py"
            )
        root = str(repo)
        if root not in sys.path:
            sys.path.insert(0, root)

    def connect(self) -> None:
        from reBotArm_control_py.actuator import RebotArm
        from reBotArm_control_py.controllers import RebotArmEndPose

        # RebotArm() loads config/rebotarm.yaml relative to CWD — run from SDK or set cwd
        prev = Path.cwd()
        try:
            os.chdir(self.cfg.sdk_repo)
            self._arm = RebotArm(self.cfg.hardware_yaml)
            self._ctrl = RebotArmEndPose(
                self._arm, arm_control_mode=self.cfg.arm_control_mode
            )
            self._ctrl.start()
        finally:
            os.chdir(prev)
        log.info("RebotArmEndPose started (mode=%s)", self.cfg.arm_control_mode)

    def disconnect(self) -> None:
        if self._ctrl is not None:
            self._ctrl.end()
            self._ctrl = None
            self._arm = None
            log.info("RebotArmEndPose ended")

    def move_tip_m(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        duration_s: float | None = None,
        use_traj: bool = True,
    ) -> bool:
        assert self._ctrl is not None
        if use_traj:
            dur = duration_s if duration_s is not None else self.cfg.duration()
            return bool(
                self._ctrl.move_to_traj(
                    x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, duration=dur
                )
            )
        return bool(self._ctrl.move_to_ik(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw))

    def move_tip_mm(self, x_mm: float, y_mm: float, z_mm: float, **kwargs) -> bool:
        return self.move_tip_m(x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0, **kwargs)

    def open_gripper(self) -> None:
        assert self._ctrl is not None
        self._ctrl.open_gripper()
        time.sleep(0.3)

    def close_gripper(self) -> None:
        assert self._ctrl is not None
        self._ctrl.close_gripper()
        time.sleep(0.4)

    def get_tip_m(self) -> tuple[float, float, float, float, float, float]:
        assert self._arm is not None
        from reBotArm_control_py.kinematics import joint_to_pose

        q, _, _ = self._arm.get_state()
        pos, rpy = joint_to_pose(q)
        return (
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(rpy[0]), float(rpy[1]), float(rpy[2]),
        )

    def get_joints_rad(self) -> list[float]:
        assert self._arm is not None
        q, _, _ = self._arm.get_state()
        n = min(6, len(q))
        return [float(q[i]) for i in range(n)]

    def move_to_home(self) -> bool:
        """Return to out-of-FOV home via SDK SE(3) trajectory (move_to_traj)."""
        if self.cfg.home_xyz_m:
            x, y, z = (float(v) for v in self.cfg.home_xyz_m)
            r, p, yaw = (float(v) for v in self.cfg.home_rpy_rad)
            return self.move_tip_m(x, y, z, r, p, yaw, use_traj=True)

        if not self.cfg.home_joints_deg:
            raise RuntimeError(
                "home.joints_deg and home.xyz_m are both null — teach home first "
                "(python -m p1_arm_motion.teach_home)"
            )

        # FK of taught home joints → tip, then geodesic traj (reuses trajectory/)
        assert self._arm is not None
        from reBotArm_control_py.kinematics import joint_to_pose
        import numpy as np

        q = np.array([math.radians(v) for v in self.cfg.home_joints_deg], dtype=float)
        # pad gripper if FK expects 7
        q_full, _, _ = self._arm.get_state()
        if len(q_full) > len(q):
            q = np.concatenate([q, q_full[len(q) :]])
        pos, rpy = joint_to_pose(q)
        return self.move_tip_m(
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(rpy[0]), float(rpy[1]), float(rpy[2]),
            use_traj=True,
        )


def make_arm_client(cfg: ArmConfig | None = None, dry_run: bool | None = None) -> DryRunArmClient | RebotArmClient:
    cfg = cfg or ArmConfig.from_yaml()
    if dry_run is None:
        dry_run = os.environ.get("DESKPARTNER_DRY_RUN", "1") != "0"
    if dry_run:
        return DryRunArmClient(cfg)
    return RebotArmClient(cfg)
