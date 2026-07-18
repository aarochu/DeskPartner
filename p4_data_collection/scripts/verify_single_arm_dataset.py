"""Friday-night gate: dataset must be single-arm (no phantom bimanual channels).

Run on the throwaway episode before Saturday data collection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Heuristic forbidden substrings — expand once organizers confirm schema
BIMANUAL_MARKERS = (
    "left_arm",
    "right_arm",
    "left.",
    "right.",
    "bi_",
    "bimanual",
    "arm_left",
    "arm_right",
)


def _candidate_roots(repo_id: str) -> list[Path]:
    home = Path.home()
    return [
        home / ".cache" / "huggingface" / "lerobot" / repo_id,
        home / ".cache" / "huggingface" / "lerobot" / repo_id.replace("/", "_"),
        Path("data") / "lerobot" / repo_id,
    ]


def _load_info(root: Path) -> dict:
    for name in ("meta/info.json", "info.json"):
        p = root / name
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"No info.json under {root}")


def _flatten_keys(obj, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            keys.append(path)
            keys.extend(_flatten_keys(v, path))
    return keys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="deskpartner/crumpled_paper_molmoact2")
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args()

    root = args.dataset_root
    if root is None:
        for cand in _candidate_roots(args.repo_id):
            if cand.exists():
                root = cand
                break
    if root is None or not root.exists():
        print("FAIL: dataset root not found. Pass --dataset-root explicitly.")
        print("Tried:", *[str(c) for c in _candidate_roots(args.repo_id)], sep="\n  ")
        return 1

    info = _load_info(root)
    keys = _flatten_keys(info)
    # Also check features if present
    features = info.get("features") or info.get("data_config") or {}
    keys.extend(_flatten_keys(features))

    offenders = sorted({k for k in keys if any(m in k.lower() for m in BIMANUAL_MARKERS)})
    cam_ok = any("front" in k.lower() for k in keys) or "front" in str(info).lower()

    print(f"dataset_root={root}")
    print(f"info_keys_sample={keys[:40]}")
    if offenders:
        print("FAIL: possible bimanual / second-arm channels:")
        for o in offenders:
            print(f"  - {o}")
        print("Fix record config to single-arm follower only, then re-record throwaway.")
        return 2

    print("PASS: no obvious phantom second-arm channels in meta.")
    if not cam_ok:
        print("WARN: did not see 'front' camera key in meta — confirm camera keys match config/cameras.yaml")
    print("Still manually inspect one parquet/episode for action dim == single arm DOF.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
