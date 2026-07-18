"""Single shared calibration.json for P1 / P2 / P3."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("data/calibration/calibration.json")

EMPTY: dict[str, Any] = {
    "version": 1,
    "updated_at": None,
    "camera": {
        "index": 0,
        "exposure": None,
        "white_balance": None,
        "width": None,
        "height": None,
    },
    "aruco": {
        "dictionary": "DICT_4X4_50",
        "marker_size_mm": 50,
        "ids": [0, 1, 2, 3],
        "pixel_centers": {},  # "0": [u, v], ...
        "plane_mm": {         # id → known plane mm
            "0": [0.0, 0.0],
            "1": [400.0, 0.0],
            "2": [400.0, 300.0],
            "3": [0.0, 300.0],
        },
    },
    "homography": None,       # 3x3 row-major
    "plane_to_arm": {         # arm = A @ plane + b  (mm)
        "A": None,            # 2x2
        "b": None,            # 2
        "pairs": [],          # [{plane_mm, arm_mm}, ...]
    },
    "empty_desk_image": "data/calibration/empty_desk.png",
    "click_verify_tolerance_mm": 5.0,
}


def load(path: Path = DEFAULT_PATH) -> dict[str, Any]:
    if not path.exists():
        return deepcopy(EMPTY)
    return json.loads(path.read_text())


def save(data: dict[str, Any], path: Path = DEFAULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {path}")
    return path
