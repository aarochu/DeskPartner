"""Dry-run smoke: pick_drop loads and runs without hardware."""

from p1_arm_motion.arm_client import ArmConfig, DryRunArmClient
from p1_arm_motion.pick_drop import pick_and_drop
from shared.fixtures import fake_destination_pose


def test_dry_pick_trash(tmp_path, monkeypatch):
    # destinations.yaml may have null poses — stub load via monkeypatch path
    dest_yaml = tmp_path / "destinations.yaml"
    dest_yaml.write_text(
        """
destinations:
  trash:
    label: trash
    xyz_mm: [280, -180, 80]
    rpy_deg: [0, 0, 0]
  pen_cup:
    label: pen_cup
    xyz_mm: [300, 160, 90]
    rpy_deg: [0, 0, 0]
  tray:
    label: tray
    xyz_mm: [320, 0, 70]
    rpy_deg: [0, 0, 0]
  keep:
    label: keep
    action: never_touch
"""
    )
    cfg = ArmConfig(
        follower_port="/dev/null",
        home_joints_deg=[0, 0, 0, 0, 0, 0],
        transit_height_mm=120,
        approach_height_mm=40,
        max_speed=0.15,
    )
    arm = DryRunArmClient(cfg)
    arm.connect()
    pick_and_drop(
        arm,
        cfg,
        target_xy_mm=(180.0, 40.0),
        grasp_height_mm=15.0,
        destination="trash",
        destinations_path=dest_yaml,
    )
    arm.disconnect()
    assert fake_destination_pose("trash").x_mm == 280
