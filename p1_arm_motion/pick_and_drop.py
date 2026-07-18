"""Top-down pick-and-drop primitive (Skill A).

Uses RebotArmEndPose.move_to_traj / move_to_ik / gripper APIs — no custom IK math.

Sequence (each named for closed-loop retry logging):
  hover → descend → grasp → lift → transit → release → return_home
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import yaml

from p1_arm_motion.arm_client import ArmConfig, DryRunArmClient, RebotArmClient, make_arm_client

log = logging.getLogger("p1.pick")

DEST_PATH = Path("config/destinations.yaml")
Arm = DryRunArmClient | RebotArmClient


class StepError(RuntimeError):
    def __init__(self, step: str, message: str) -> None:
        self.step = step
        super().__init__(f"[{step}] {message}")


def _run_step(step: str, fn: Callable[[], None]) -> None:
    log.info("STEP start: %s", step)
    try:
        fn()
    except StepError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("STEP failed: %s", step)
        raise StepError(step, str(exc)) from exc
    log.info("STEP ok: %s", step)


def load_destination(name: str, path: Path = DEST_PATH) -> dict:
    if name == "keep":
        raise StepError("load_destination", "keep must never be picked")
    raw = yaml.safe_load(path.read_text())
    entry = (raw.get("destinations") or {}).get(name)
    if not entry:
        raise StepError("load_destination", f"unknown destination '{name}'")
    if entry.get("xyz_m") is None:
        raise StepError(
            "load_destination",
            f"destination '{name}' not taught — run: python -m p1_arm_motion.teach_destination {name}",
        )
    return entry


def move_to_home(arm: Arm) -> None:
    def _go() -> None:
        ok = arm.move_to_home()
        if not ok:
            raise RuntimeError("move_to_home returned False")

    _run_step("return_home", _go)


def pick_and_drop(
    arm: Arm,
    cfg: ArmConfig,
    target_xy_mm: tuple[float, float],
    item_type: str,
    destination_name: str,
    desk_z_m: float = 0.0,
    destinations_path: Path = DEST_PATH,
) -> None:
    """Execute one top-down pick. Raises StepError with .step name on failure."""
    tx, ty = target_xy_mm
    hover_z_mm = cfg.transit_height_mm
    grasp_z_mm = cfg.grasp_z_mm(item_type)
    # Arm Z: desk plane at desk_z_m; tip heights are above desk
    hover_z_m = desk_z_m + hover_z_mm / 1000.0
    grasp_z_m = desk_z_m + grasp_z_mm / 1000.0
    tx_m, ty_m = tx / 1000.0, ty / 1000.0

    dest = load_destination(destination_name, destinations_path)
    dx, dy, dz = (float(v) for v in dest["xyz_m"])
    rpy = dest.get("rpy_rad") or [0.0, 0.0, 0.0]
    dr, dp, dyaw = (float(v) for v in rpy)

    def hover() -> None:
        ok = arm.move_tip_m(tx_m, ty_m, hover_z_m, duration_s=cfg.duration())
        if not ok:
            raise RuntimeError("IK/traj failed at hover")

    def descend() -> None:
        ok = arm.move_tip_m(
            tx_m, ty_m, grasp_z_m, duration_s=cfg.duration(slow=True), use_traj=True
        )
        if not ok:
            raise RuntimeError("slow descend failed")

    def grasp() -> None:
        arm.close_gripper()

    def lift() -> None:
        ok = arm.move_tip_m(tx_m, ty_m, hover_z_m, duration_s=cfg.duration())
        if not ok:
            raise RuntimeError("lift failed")

    def transit() -> None:
        # Stay at safe Z while moving over destination XY, then descend to taught Z
        ok = arm.move_tip_m(dx, dy, max(hover_z_m, dz + 0.04), duration_s=cfg.duration())
        if not ok:
            raise RuntimeError("transit XY failed")
        ok = arm.move_tip_m(dx, dy, dz, dr, dp, dyaw, duration_s=cfg.duration(slow=True))
        if not ok:
            raise RuntimeError("transit descend to destination failed")

    def release() -> None:
        arm.open_gripper()
        time.sleep(0.4)

    def return_home() -> None:
        ok = arm.move_to_home()
        if not ok:
            raise RuntimeError("return home failed")

    _run_step("hover", hover)
    _run_step("descend", descend)
    _run_step("grasp", grasp)
    _run_step("lift", lift)
    _run_step("transit", transit)
    _run_step("release", release)
    _run_step("return_home", return_home)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Hardcoded / CLI pick-and-drop")
    p.add_argument("--x-mm", type=float, default=180.0)
    p.add_argument("--y-mm", type=float, default=40.0)
    p.add_argument("--item", default="paper")
    p.add_argument("--dest", default="trash")
    p.add_argument("--live", action="store_true", help="Use real arm (DESKPARTNER_DRY_RUN=0)")
    p.add_argument("--desk-z-m", type=float, default=0.0)
    args = p.parse_args(argv)

    cfg = ArmConfig.from_yaml()
    arm = make_arm_client(cfg, dry_run=not args.live)
    arm.connect()
    try:
        move_to_home(arm)
        pick_and_drop(
            arm,
            cfg,
            target_xy_mm=(args.x_mm, args.y_mm),
            item_type=args.item,
            destination_name=args.dest,
            desk_z_m=args.desk_z_m,
        )
    except StepError as exc:
        log.error("pick_and_drop aborted at step=%s: %s", exc.step, exc)
        return 2
    finally:
        arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
