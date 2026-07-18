"""Fail-loud schema check after EVERY episode (especially the first throwaway).

  python -m p4_data_collection.verify_episode_format
  python -m p4_data_collection.verify_episode_format --dataset-root /path/to/ds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from p4_data_collection.config_loader import default_dataset_root, load_recording_config


def _load_info(root: Path) -> dict[str, Any]:
    for rel in ("meta/info.json", "info.json"):
        p = root / rel
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"No info.json under {root}")


def _flatten_keys(obj: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            keys.append(path)
            keys.extend(_flatten_keys(v, path))
    return keys


def _feature_names(info: dict[str, Any]) -> list[str]:
    feats = info.get("features") or {}
    if isinstance(feats, dict):
        return list(feats.keys()) + _flatten_keys(feats)
    return _flatten_keys(info)


def _count_frames(root: Path, info: dict[str, Any]) -> int | None:
    # Prefer meta totals
    for key in ("total_frames", "total_episodes", "num_frames"):
        if key in info and isinstance(info[key], int):
            if key == "total_episodes":
                continue
            return int(info[key])
    # Fall back: count parquet / mp4 under data/
    data = root / "data"
    if data.exists():
        n = 0
        for p in data.rglob("*.parquet"):
            try:
                import pyarrow.parquet as pq  # optional

                n += pq.read_table(p).num_rows
            except Exception:  # noqa: BLE001
                pass
        if n:
            return n
    return None


def _episode_count(root: Path, info: dict[str, Any]) -> int | None:
    if "total_episodes" in info:
        return int(info["total_episodes"])
    ep = root / "meta" / "episodes.jsonl"
    if ep.exists():
        return sum(1 for line in ep.read_text().splitlines() if line.strip())
    return None


def verify(root: Path, cfg: dict[str, Any]) -> int:
    expect = cfg.get("expect") or {}
    print(f"=== verify_episode_format ===\nroot={root}")

    if not root.exists():
        print(f"FAIL: dataset root does not exist: {root}")
        return 2

    try:
        info = _load_info(root)
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}")
        return 2

    keys = _feature_names(info)
    key_blob = " ".join(keys).lower() + " " + json.dumps(info).lower()
    failures: list[str] = []
    warnings: list[str] = []

    # 1) Single-arm channels
    forbidden = [s.lower() for s in expect.get("forbidden_channel_substrings") or []]
    offenders = sorted({k for k in keys if any(f in k.lower() for f in forbidden)})
    # also scan raw json text for bimanual robot types
    if "bi_" in key_blob or "bimanual" in key_blob:
        offenders.append("(metadata contains bi_/bimanual)")
    if offenders:
        failures.append(
            "Second-arm / bimanual channels detected:\n  - " + "\n  - ".join(offenders)
        )
    else:
        print("PASS: no phantom second-arm channel names in features/meta")

    # 2) Camera keys present
    cam_keys = expect.get("camera_keys") or ["front", "side"]
    feat_names = set(keys)
    missing_cams = []
    for ck in cam_keys:
        # LeRobot often uses observation.images.front etc.
        if not any(ck in k for k in feat_names) and ck not in json.dumps(info):
            missing_cams.append(ck)
    if missing_cams:
        failures.append(f"Missing camera keys in dataset meta: {missing_cams}")
    else:
        print(f"PASS: camera keys present ({cam_keys})")

    # Non-empty video dirs if present
    for ck in cam_keys:
        vid_dirs = list(root.glob(f"**/videos/**/{ck}/**")) + list(root.glob(f"**/*{ck}*.mp4"))
        # soft: if videos exist, ensure at least one file > 0 bytes
        files = [p for p in root.rglob("*.mp4") if ck in str(p)]
        if files and all(p.stat().st_size == 0 for p in files):
            failures.append(f"Camera '{ck}' video files are empty (0 bytes)")
        elif files:
            print(f"PASS: camera '{ck}' has non-empty video ({len(files)} file(s))")
        else:
            warnings.append(f"No mp4 found for camera '{ck}' yet (ok if images-only store)")

    # 3) Control / robot metadata
    robot_hint = (expect.get("robot_type_contains") or "").lower()
    info_s = json.dumps(info).lower()
    if robot_hint and robot_hint not in info_s:
        warnings.append(
            f"Did not find robot type hint '{robot_hint}' in meta — confirm control mode manually"
        )
    else:
        print(f"PASS: robot/control metadata mentions '{robot_hint}'")

    # 4) Episode length sanity
    frames = _count_frames(root, info)
    eps = _episode_count(root, info)
    min_f = int(expect.get("min_frames", 15))
    max_f = int(expect.get("max_frames", 3000))
    print(f"INFO: total_episodes={eps} total_frames={frames}")

    if frames is not None:
        if frames < min_f:
            failures.append(f"Too few frames: {frames} < min_frames={min_f} (truncated?)")
        elif frames > max_f:
            failures.append(f"Absurdly long: {frames} > max_frames={max_f}")
        else:
            print(f"PASS: frame count sane ({frames} in [{min_f},{max_f}])")
    else:
        warnings.append("Could not determine total_frames — inspect episode manually")

    if eps is not None and eps < 1:
        failures.append("total_episodes < 1")

    for w in warnings:
        print(f"WARN: {w}")

    if failures:
        print("\nFAIL — do NOT trust this dataset for training:")
        for f in failures:
            print(f"  * {f}")
        return 2

    print("\nPASS — format check OK (still spot-check grasp quality visually)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--repo-id", default=None)
    args = ap.parse_args()

    cfg = load_recording_config(args.config)
    repo_id = args.repo_id or cfg["dataset"]["repo_id"]
    root = args.dataset_root or default_dataset_root(repo_id)
    return verify(root, cfg)


if __name__ == "__main__":
    sys.exit(main())
