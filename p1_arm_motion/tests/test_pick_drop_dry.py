"""Dry-run smoke: pick_and_drop runs without hardware."""

from pathlib import Path

from p1_arm_motion.arm_client import ArmConfig, DryRunArmClient
from p1_arm_motion.pick_and_drop import pick_and_drop


def test_dry_pick_trash(tmp_path: Path) -> None:
    dest_yaml = tmp_path / "destinations.yaml"
    dest_yaml.write_text(
        """
destinations:
  trash:
    label: trash
    xyz_m: [0.28, -0.18, 0.08]
    rpy_rad: [0, 0, 0]
  pen_cup:
    label: pen_cup
    xyz_m: [0.30, 0.16, 0.09]
    rpy_rad: [0, 0, 0]
  tray:
    label: tray
    xyz_m: [0.32, 0.0, 0.07]
    rpy_rad: [0, 0, 0]
"""
    )
    cfg = ArmConfig.from_yaml("config/arm.yaml")
    # dry home needs something set
    cfg.home_xyz_m = [0.15, 0.25, 0.35]
    arm = DryRunArmClient(cfg)
    arm.connect()
    pick_and_drop(
        arm,
        cfg,
        target_xy_mm=(180.0, 40.0),
        item_type="paper",
        destination_name="trash",
        destinations_path=dest_yaml,
    )
    arm.disconnect()
