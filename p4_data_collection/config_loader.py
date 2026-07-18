"""Load config/recording.yaml (+ optional arm.yaml port sync warnings)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDING = ROOT / "config" / "recording.yaml"
DEFAULT_ARM = ROOT / "config" / "arm.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return yaml.safe_load(path.read_text()) or {}


def load_recording_config(path: Path | None = None) -> dict[str, Any]:
    cfg = load_yaml(path or DEFAULT_RECORDING)
    # Warn if arm.yaml ports drifted
    arm_path = DEFAULT_ARM
    if arm_path.exists():
        arm = load_yaml(arm_path)
        f_port = (arm.get("follower") or {}).get("port")
        l_port = (arm.get("leader") or {}).get("port")
        if f_port and f_port != cfg["robot"]["port"]:
            print(
                f"WARN: recording.yaml robot.port={cfg['robot']['port']} "
                f"!= arm.yaml follower.port={f_port} — fix drift before recording"
            )
        if l_port and l_port != cfg["teleop"]["port"]:
            print(
                f"WARN: recording.yaml teleop.port={cfg['teleop']['port']} "
                f"!= arm.yaml leader.port={l_port} — fix drift before recording"
            )
    return cfg


def cameras_cli_dict(cfg: dict[str, Any]) -> str:
    """Build lerobot --robot.cameras='{...}' string from config."""
    parts = []
    for name, cam in (cfg.get("cameras") or {}).items():
        idx = cam["index_or_path"]
        # quote path strings; leave ints bare
        if isinstance(idx, str) and not idx.isdigit():
            idx_s = f'"{idx}"'
        else:
            idx_s = str(idx)
        fourcc = cam.get("fourcc", "MJPG")
        parts.append(
            f'{name}: {{type: {cam.get("type", "opencv")}, index_or_path: {idx_s}, '
            f'width: {cam["width"]}, height: {cam["height"]}, fps: {cam["fps"]}, '
            f'fourcc: \\"{fourcc}\\"}}'
        )
    return "{" + ", ".join(parts) + "}"


def default_dataset_root(repo_id: str) -> Path:
    home = Path.home()
    candidates = [
        home / ".cache" / "huggingface" / "lerobot" / repo_id,
        home / ".cache" / "huggingface" / "lerobot" / repo_id.replace("/", "_"),
        ROOT / "data" / "lerobot" / repo_id,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]
