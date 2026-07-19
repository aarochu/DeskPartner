"""Load DeskPartner arm/recording YAML for the Rerun port."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from p5_rerun_port.constants import DEFAULT_URDF_CANDIDATES, REPO_ROOT

DEFAULT_ARM = REPO_ROOT / "config" / "arm.yaml"
DEFAULT_RECORDING = REPO_ROOT / "config" / "recording.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_arm_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or DEFAULT_ARM)


def load_recording_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or DEFAULT_RECORDING)


def resolve_urdf_path(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    for candidate in DEFAULT_URDF_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def camera_specs(recording_cfg: dict[str, Any]) -> list[tuple[str, int, int, int]]:
    """Return [(lerobot_name, index, width, height), ...] in front/side order when present."""
    cams = recording_cfg.get("cameras") or {}
    order = [name for name in ("front", "side") if name in cams]
    for name in cams:
        if name not in order:
            order.append(name)
    out: list[tuple[str, int, int, int]] = []
    for name in order:
        cam = cams[name]
        idx = cam.get("index_or_path", 0)
        if isinstance(idx, str) and idx.isdigit():
            idx = int(idx)
        if not isinstance(idx, int):
            # Device path — leave index as -1 and let OpenCV try the path string later.
            continue
        out.append((name, idx, int(cam.get("width", 640)), int(cam.get("height", 480))))
    return out
