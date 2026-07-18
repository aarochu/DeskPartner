"""Fallback tier 3: pre-staged canned pick sequence (no perception)."""

from __future__ import annotations

from p1_arm_motion.arm_client import ArmConfig, DryRunArmClient
from p1_arm_motion.pick_drop import pick_and_drop
from shared.fixtures import fake_messy_plan


def main() -> None:
    cfg = ArmConfig.from_yaml("config/arm.yaml")
    arm = DryRunArmClient(cfg)
    arm.connect()
    arm.go_home()
    plan = fake_messy_plan()
    for item in plan.items:
        if item.destination == "keep" or item.arm_xy_mm is None:
            continue
        print(f"[canned] {item.label} → {item.destination}")
        pick_and_drop(
            arm,
            cfg,
            target_xy_mm=item.arm_xy_mm,
            grasp_height_mm=item.grasp_height_mm or 20.0,
            destination=item.destination,
        )
        arm.go_home()
    arm.disconnect()
    print("[canned] done — replace fixtures with taught poses before demo day")


if __name__ == "__main__":
    main()
