"""Closed-loop: home → photo → plan → pick → drop → repeat."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import yaml

from p1_arm_motion.arm_client import ArmConfig, DryRunArmClient
from p1_arm_motion.home import ensure_home_before_photo
from p1_arm_motion.pick_and_drop import DEST_PATH, FIXTURE_DEST_PATH, pick_and_drop
from p3_vlm_orchestrator.logging_ui import RunLogger
from p3_vlm_orchestrator.policy import actionable_items
from p3_vlm_orchestrator.vlm_client import plan_from_image
from shared.types import DetectedItem, PlanResult


class Orchestrator:
    def __init__(
        self,
        arm: DryRunArmClient,
        arm_cfg: ArmConfig,
        workspace_path: str | Path = "config/workspace.yaml",
        use_fixtures: bool = False,
    ) -> None:
        self.arm = arm
        self.arm_cfg = arm_cfg
        self.workspace = yaml.safe_load(Path(workspace_path).read_text())
        self.max_retries = int(self.workspace["closed_loop"]["max_retries_per_object"])
        self.retries: dict[str, int] = defaultdict(int)
        self.logger = RunLogger(self.workspace["closed_loop"].get("log_dir", "runs"))
        self.use_fixtures = use_fixtures
        # Offline demo reads fake taught destinations; live reads the real ones.
        self._dest_path = FIXTURE_DEST_PATH if use_fixtures else DEST_PATH
        # Objects we've given up on (hit max retries or unreachable) — excluded
        # from selection so a stuck item can't deadlock the loop.
        self.skipped: set[str] = set()
        self.perception = os.environ.get(
            "DESKPARTNER_PERCEPTION",
            self.workspace["perception"].get("mode", "vlm"),
        )

    def _capture(self):
        ensure_home_before_photo(self.arm)
        if self.use_fixtures:
            import numpy as np

            return np.zeros((480, 640, 3), dtype="uint8")
        from p2_vision_calibration.camera import grab_frame

        return grab_frame()

    def _plan(self, frame) -> PlanResult:
        if self.use_fixtures:
            from shared.fixtures import fake_messy_plan

            return fake_messy_plan()
        if self.perception == "cv":
            from p2_vision_calibration.cv_fallback import detect

            return detect(frame)
        return plan_from_image(frame)

    def _attach_arm_coords(self, plan: PlanResult, frame) -> None:
        """Refine centroids and convert pixels → arm mm (P2 contract)."""
        if self.use_fixtures:
            return
        from p2_vision_calibration.centroid_refine import refine_centroid
        from p2_vision_calibration.pixel_to_arm import CalibrationChain

        cal = CalibrationChain()
        for item in plan.items:
            u, v = refine_centroid(frame, item.bbox_xyxy)
            item.centroid_uv = (u, v)
            item.arm_xy_mm = cal.pixel_to_arm_coords(u, v)

    @staticmethod
    def _item_key(item: DetectedItem) -> str:
        """Stable identity for retry/skip accounting.

        Must survive per-frame jitter in the refined centroid, so we quantize
        arm coords to a ~2 cm grid rather than keying on the raw float.
        """
        if item.arm_xy_mm is not None:
            gx = round(item.arm_xy_mm[0] / 20.0)
            gy = round(item.arm_xy_mm[1] / 20.0)
            return f"{item.label}:{gx},{gy}"
        x0, y0, x1, y1 = item.bbox_xyxy
        return f"{item.label}:bbox{round((x0 + x1) / 40)},{round((y0 + y1) / 40)}"

    def run(self, max_cycles: int = 20) -> None:
        self.arm.connect()
        try:
            for _ in range(max_cycles):
                frame = self._capture()
                plan = self._plan(frame)
                self._attach_arm_coords(plan, frame)
                self.logger.log_cycle(frame, plan)

                if plan.desk_is_clean:
                    print("[orch] desk clean — done")
                    break

                # First actionable object we haven't given up on. One per cycle
                # so the next photo can heal a failed grasp.
                item = next(
                    (it for it in actionable_items(plan) if self._item_key(it) not in self.skipped),
                    None,
                )
                if item is None:
                    print("[orch] nothing actionable left (clean or all skipped) — done")
                    break

                key = self._item_key(item)
                if self.retries[key] >= self.max_retries:
                    print(f"[orch] skip {item.label} after {self.max_retries} retries")
                    self.skipped.add(key)
                    continue
                if item.arm_xy_mm is None or item.grasp_height_mm is None:
                    print(f"[orch] missing coords for {item.label}; skip")
                    self.skipped.add(key)
                    continue

                print(f"[orch] pick {item.label} → {item.destination} @ {item.arm_xy_mm}")
                try:
                    pick_and_drop(
                        self.arm,
                        self.arm_cfg,
                        target_xy_mm=item.arm_xy_mm,
                        item_type=item.label,
                        destination_name=item.destination,
                        grasp_height_mm=item.grasp_height_mm,
                        destinations_path=self._dest_path,
                    )
                except Exception as exc:  # noqa: BLE001 — demo harness: log and retry
                    print(f"[orch] pick failed: {exc}")
                self.retries[key] += 1
            else:
                print("[orch] max_cycles reached")
        finally:
            self.arm.move_to_home()
            self.arm.disconnect()
