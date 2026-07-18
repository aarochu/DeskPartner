"""Top-down pick-and-drop primitive (Skill A).

Sequence: hover → descend → close → lift → transit → release over destination.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from p1_arm_motion.arm_client import ArmConfig, DryRunArmClient
from shared.types import ArmPose, Destination


def load_destination_pose(name: Destination, path: str | Path = "config/destinations.yaml") -> ArmPose:
    if name == "keep":
        raise ValueError("keep must never be picked")
    raw = yaml.safe_load(Path(path).read_text())
    entry = raw["destinations"][name]
    xyz = entry.get("xyz_mm")
    if xyz is None:
        raise RuntimeError(f"Destination '{name}' not taught yet — run teach_destinations.py")
    rpy = entry.get("rpy_deg") or [0, 0, 0]
    return ArmPose(xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2])


def pick_and_drop(
    arm: DryRunArmClient,
    cfg: ArmConfig,
    target_xy_mm: tuple[float, float],
    grasp_height_mm: float,
    destination: Destination,
    destinations_path: str | Path = "config/destinations.yaml",
) -> None:
    """Execute one scripted pick. Caller returns home for the next photo."""
    x, y = target_xy_mm
    hover = ArmPose(x, y, cfg.approach_height_mm)
    grasp = ArmPose(x, y, grasp_height_mm)
    lift = ArmPose(x, y, cfg.transit_height_mm)
    dest = load_destination_pose(destination, destinations_path)
    dest_hover = ArmPose(dest.x_mm, dest.y_mm, cfg.transit_height_mm, dest.roll_deg, dest.pitch_deg, dest.yaw_deg)

    arm.set_gripper(closed=False)
    arm.move_tip(hover)
    arm.move_tip(grasp, duration_s=1.5)
    arm.set_gripper(closed=True)
    arm.move_tip(lift)
    arm.move_tip(dest_hover)
    arm.move_tip(dest, duration_s=1.5)
    arm.set_gripper(closed=False)
    arm.move_tip(dest_hover)


if __name__ == "__main__":
    # Hardcoded crumpled-paper smoke test (Friday night exit item)
    cfg = ArmConfig.from_yaml("config/arm.yaml")
    arm = DryRunArmClient(cfg)
    arm.connect()
    arm.go_home()
    # Replace with a real in-zone XY after calibration
    pick_and_drop(arm, cfg, target_xy_mm=(180.0, 40.0), grasp_height_mm=15.0, destination="trash")
    arm.go_home()
    arm.disconnect()
