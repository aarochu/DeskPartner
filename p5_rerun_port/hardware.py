"""Follower/leader observation + optional teleop (fake or LeRobot/Seeed)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from p5_rerun_port.constants import FOLLOWER, JOINT_NAMES, LEADER


class ArmSession(Protocol):
    def start(self) -> None: ...
    def read(self) -> tuple[list[float], list[float] | None]:
        """Return (follower_state, follower_goal_or_None). Degrees-ish floats, len=7."""
        ...

    def send_goal(self, goal: list[float]) -> None: ...
    def close(self) -> None: ...


@dataclass
class FakeArmSession:
    """Sine-wave joints for dry-run without hardware."""

    fps: float = 30.0
    teleop: bool = True
    _t0: float = field(default=0.0, init=False)

    def start(self) -> None:
        self._t0 = time.time()
        print("hardware: FAKE arm session (no serial / CAN)", flush=True)

    def _pose(self, t: float, phase: float = 0.0) -> list[float]:
        vals = []
        for i, _name in enumerate(JOINT_NAMES):
            if _name == "gripper":
                vals.append(50.0 + 40.0 * math.sin(t * 0.8 + phase))
            else:
                amp = 15.0 + 5.0 * (i % 3)
                vals.append(amp * math.sin(t * (0.6 + 0.05 * i) + phase + i * 0.3))
        return vals

    def read(self) -> tuple[list[float], list[float] | None]:
        t = time.time() - self._t0
        state = self._pose(t)
        goal = self._pose(t, phase=0.15) if self.teleop else None
        return state, goal

    def send_goal(self, goal: list[float]) -> None:
        _ = goal  # no-op

    def close(self) -> None:
        print("hardware: fake session closed", flush=True)


@dataclass
class LeRobotArmSession:
    """Live session via LeRobot Seeed reBot types (venue Ubuntu path)."""

    robot_cfg: dict[str, Any]
    teleop_cfg: dict[str, Any] | None = None
    teleop: bool = True
    robot: Any = field(default=None, init=False, repr=False)
    teleop_device: Any = field(default=None, init=False, repr=False)

    def start(self) -> None:
        try:
            from lerobot.robots import make_robot_from_config  # type: ignore
        except Exception:
            try:
                from lerobot.common.robots.utils import make_robot_from_config  # type: ignore
            except Exception as exc:
                raise SystemExit(
                    "FAIL: LeRobot not importable. Use --fake for dry-run, or install "
                    "Seeed LeRobot + lerobot-robot-seeed-b601 on the venue machine."
                ) from exc

        self.robot = self._make_robot(make_robot_from_config)
        self.robot.connect()
        print(
            f"hardware: follower {self.robot_cfg.get('type')} on {self.robot_cfg.get('port')}",
            flush=True,
        )

        if self.teleop and self.teleop_cfg:
            self.teleop_device = self._make_teleop()
            if self.teleop_device is not None:
                self.teleop_device.connect()
                print(
                    f"hardware: leader {self.teleop_cfg.get('type')} on {self.teleop_cfg.get('port')}",
                    flush=True,
                )

    def _make_robot(self, factory: Any) -> Any:
        # Config shapes differ across LeRobot versions; try dataclass-style then kwargs.
        rtype = self.robot_cfg["type"]
        try:
            from lerobot.robots import RobotConfig  # type: ignore

            cfg = RobotConfig(
                type=rtype,
                port=self.robot_cfg["port"],
                id=self.robot_cfg.get("id"),
            )
            if "can_adapter" in self.robot_cfg:
                try:
                    cfg.can_adapter = self.robot_cfg["can_adapter"]
                except Exception:
                    pass
            return factory(cfg)
        except Exception:
            pass

        # Fallback: instantiate registered class if present
        try:
            from lerobot.robots.utils import make_robot_config  # type: ignore

            cfg = make_robot_config(
                rtype,
                port=self.robot_cfg["port"],
                id=self.robot_cfg.get("id", "follower1"),
            )
            if hasattr(cfg, "can_adapter") and "can_adapter" in self.robot_cfg:
                cfg.can_adapter = self.robot_cfg["can_adapter"]
            return factory(cfg)
        except Exception as exc:
            raise SystemExit(
                f"FAIL: could not construct robot type={rtype!r}. "
                f"Check Seeed LeRobot install. Underlying error: {exc}"
            ) from exc

    def _make_teleop(self) -> Any | None:
        assert self.teleop_cfg is not None
        try:
            from lerobot.teleoperators import make_teleoperator_from_config  # type: ignore
        except Exception:
            try:
                from lerobot.common.teleoperators.utils import make_teleoperator_from_config  # type: ignore
            except Exception as exc:
                print(f"WARN: teleop unavailable ({exc}); logging follower state only", flush=True)
                return None
        ttype = self.teleop_cfg["type"]
        try:
            from lerobot.teleoperators import TeleoperatorConfig  # type: ignore

            cfg = TeleoperatorConfig(
                type=ttype,
                port=self.teleop_cfg["port"],
                id=self.teleop_cfg.get("id"),
            )
            return make_teleoperator_from_config(cfg)
        except Exception:
            try:
                from lerobot.teleoperators.utils import make_teleoperator_config  # type: ignore

                cfg = make_teleoperator_config(
                    ttype,
                    port=self.teleop_cfg["port"],
                    id=self.teleop_cfg.get("id", "rebot_arm_102_leader"),
                )
                return make_teleoperator_from_config(cfg)
            except Exception as exc:
                print(f"WARN: teleop construct failed ({exc})", flush=True)
                return None

    def _obs_to_joints(self, obs: dict[str, Any]) -> list[float]:
        # Prefer ordered JOINT_NAMES keys; fall back to sorted *.pos
        vals: list[float] = []
        for name in JOINT_NAMES:
            key = f"{name}.pos"
            if key in obs:
                vals.append(float(obs[key]))
            elif name in obs:
                vals.append(float(obs[name]))
        if len(vals) == len(JOINT_NAMES):
            return vals
        # Generic fallback
        pos_keys = sorted(k for k in obs if str(k).endswith(".pos"))
        if pos_keys:
            return [float(obs[k]) for k in pos_keys]
        raise RuntimeError(f"cannot parse joint state from observation keys: {list(obs)[:20]}")

    def read(self) -> tuple[list[float], list[float] | None]:
        assert self.robot is not None
        obs = self.robot.get_observation()
        state = self._obs_to_joints(obs)
        goal: list[float] | None = None
        if self.teleop and self.teleop_device is not None:
            action = self.teleop_device.get_action()
            goal = self._obs_to_joints(action) if isinstance(action, dict) else list(map(float, action))
            # Drive follower
            try:
                self.robot.send_action(action)
            except Exception as exc:
                print(f"WARN: send_action failed: {exc}", flush=True)
        return state, goal

    def send_goal(self, goal: list[float]) -> None:
        assert self.robot is not None
        if len(goal) != len(JOINT_NAMES):
            raise ValueError(f"expected {len(JOINT_NAMES)} joints, got {len(goal)}")
        action = {f"{name}.pos": float(v) for name, v in zip(JOINT_NAMES, goal, strict=True)}
        self.robot.send_action(action)

    def close(self) -> None:
        for device in (self.teleop_device, self.robot):
            if device is None:
                continue
            try:
                if hasattr(device, "disconnect"):
                    device.disconnect()
            except Exception as exc:
                print(f"WARN: disconnect: {exc}", flush=True)


def make_session(
    *,
    fake: bool,
    teleop: bool,
    recording_cfg: dict[str, Any] | None = None,
) -> ArmSession:
    if fake:
        return FakeArmSession(teleop=teleop)
    if recording_cfg is None:
        from p5_rerun_port.config import load_recording_config

        recording_cfg = load_recording_config()
    return LeRobotArmSession(
        robot_cfg=recording_cfg["robot"],
        teleop_cfg=recording_cfg.get("teleop"),
        teleop=teleop,
    )


def entity_paths(include_leader: bool = False) -> dict[str, str]:
    paths = {
        "follower_position": f"{FOLLOWER}/position",
        "follower_goal": f"{FOLLOWER}/goal",
    }
    if include_leader:
        paths["leader_position"] = f"{LEADER}/position"
    return paths
