"""Persist photos + decisions for debugging and the judge screen."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from shared.types import PlanResult


class RunLogger:
    def __init__(self, root: str | Path = "runs") -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = Path(root) / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        print(f"[log] {self.dir}")

    def log_cycle(self, frame: np.ndarray, plan: PlanResult) -> Path:
        self.step += 1
        img_path = self.dir / f"{self.step:03d}_frame.jpg"
        json_path = self.dir / f"{self.step:03d}_plan.json"
        cv2.imwrite(str(img_path), frame)
        payload = {
            "desk_is_clean": plan.desk_is_clean,
            "raw_provider": plan.raw_provider,
            "items": [asdict(i) for i in plan.items],
            "image": img_path.name,
        }
        json_path.write_text(json.dumps(payload, indent=2))
        plan.image_path = str(img_path)
        print(f"[log] step={self.step} clean={plan.desk_is_clean} n={len(plan.items)} → {json_path.name}")
        return json_path
