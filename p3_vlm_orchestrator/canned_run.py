"""Fallback tier 3: pre-staged canned pick sequence (no perception)."""

from __future__ import annotations

from p1_arm_motion.arm_client import ArmConfig, DryRunArmClient
from p1_arm_motion.pick_and_drop import pick_and_drop
from shared.fixtures import fake_messy_plan


def main() -> None:
    cfg = ArmConfig.from_yaml("config/arm.yaml")
    cfg.home_xyz_m = cfg.home_xyz_m or [0.15, 0.25, 0.35]
    arm = DryRunArmClient(cfg)
    arm.connect()
    arm.move_to_home()
    plan = fake_messy_plan()
    for item in plan.items:
        if item.destination == "keep" or item.arm_xy_mm is None:
            continue
        print(f"[canned] {item.label} → {item.destination}")
        # destinations must be taught for live; dry-run needs stub yaml with xyz_m
        try:
            pick_and_drop(
                arm,
                cfg,
                target_xy_mm=item.arm_xy_mm,
                item_type=item.label,
                destination_name=item.destination,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[canned] skip {item.label}: {exc}")
        arm.move_to_home()
    arm.disconnect()
    print("[canned] done — replace fixtures with taught poses before demo day")


if __name__ == "__main__":
    main()
