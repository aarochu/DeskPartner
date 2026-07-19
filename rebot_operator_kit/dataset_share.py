#!/usr/bin/env python3
"""Prepare and verify a self-describing LeRobot dataset share."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from urllib.request import urlopen
from zoneinfo import ZoneInfo


MANIFEST_NAME = "SHARE_MANIFEST.json"
CARD_NAME = "README.md"
LOCAL_TIMEZONE = ZoneInfo("America/Los_Angeles")
REQUIRED_PATHS = (
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes",
    "data",
    "videos/observation.images.front",
    "videos/observation.images.side",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def check_layout(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    missing = [item for item in REQUIRED_PATHS if not (root / item).exists()]
    if missing:
        raise ValueError("Incomplete LeRobot dataset; missing: " + ", ".join(missing))
    info = read_json(root / "meta" / "info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Only the reviewed LeRobot v3.0 dataset may be shared")
    if int(info.get("total_episodes", 0)) < 1 or int(info.get("total_frames", 0)) < 1:
        raise ValueError("Dataset has no shareable episodes or frames")
    return info


def current_git_commit() -> str | None:
    configured = os.environ.get("REBOT_EXPORTER_GIT_COMMIT", "").strip()
    if configured:
        return configured
    script_root = Path(__file__).resolve().parent
    for candidate in (script_root, script_root.parent / "DeskPartner"):
        result = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    return None


def included_attempts(attempt_root: Path | None, dataset_name: str) -> list[dict[str, Any]]:
    if attempt_root is None:
        return []
    dataset_attempts = attempt_root / dataset_name
    if not dataset_attempts.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for metadata_path in sorted(dataset_attempts.glob("*/metadata.json")):
        metadata = read_json(metadata_path)
        if metadata.get("training_included") is not True:
            continue
        records.append(
            {
                "attempt_id": metadata.get("attempt_id", metadata_path.parent.name),
                "episode_index": metadata.get("training_episode_index"),
                "started_at_utc": metadata.get("started_at"),
                "finished_at_utc": metadata.get("finished_at"),
                "metadata_sha256": sha256(metadata_path),
            }
        )
    return sorted(records, key=lambda item: int(item.get("episode_index", -1)))


def write_dataset_card(root: Path, info: dict[str, Any], destination: str) -> None:
    features = info.get("features", {})
    action = features.get("action", {}) if isinstance(features, dict) else {}
    names = action.get("names", []) if isinstance(action, dict) else []
    card = f"""---
tags:
- lerobot
- robotics
- imitation-learning
task_categories:
- robotics
---

# ReBot can sorting Stage 1

Reviewed success-only LeRobot v3 dataset for: **Pick up one can and place it in
the taped sorting zone**.

- Episodes: {int(info['total_episodes'])}
- Frames: {int(info['total_frames'])}
- FPS: {int(info['fps'])}
- Robot: `{info.get('robot_type', 'unknown')}`
- Cameras: `observation.images.front` (Logitech overhead) and
  `observation.images.side` (Innomaker wrist/claw)
- Action order: {', '.join(f'`{name}`' for name in names)}
- Intended destination: `{destination}`

`SHARE_MANIFEST.json` contains timestamps, source attempt IDs, exact SHA-256
checksums, exporter commit, and the validation/share contract. Use an immutable
Hub revision for training runs rather than relying only on a moving `main`.
"""
    (root / CARD_NAME).write_text(card)


def dataset_files(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(root).as_posix()
        if "/.cache/" in f"/{relative}/" or relative.startswith(".cache/"):
            continue
        files.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    return files


def prepare(args: argparse.Namespace) -> None:
    root = args.dataset_root.resolve()
    info = check_layout(root)
    dataset_name = root.name
    write_dataset_card(root, info, args.destination)
    attempts = included_attempts(args.attempt_root, dataset_name)
    if args.attempt_root is not None:
        expected_indices = list(range(int(info["total_episodes"])))
        actual_indices = [item.get("episode_index") for item in attempts]
        if actual_indices != expected_indices:
            raise ValueError(
                "Included attempt-to-episode mapping does not match the LeRobot dataset"
            )
    now = datetime.now(timezone.utc)
    started = [item["started_at_utc"] for item in attempts if item.get("started_at_utc")]
    finished = [item["finished_at_utc"] for item in attempts if item.get("finished_at_utc")]
    manifest = {
        "schema_version": 1,
        "dataset_name": dataset_name,
        "destination": args.destination,
        "visibility": args.visibility,
        "timezone": "America/Los_Angeles",
        "exported_at_utc": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "exported_at_local": now.astimezone(LOCAL_TIMEZONE).isoformat(timespec="milliseconds"),
        "captured_at_utc": {
            "start": min(started) if started else None,
            "end": max(finished) if finished else None,
        },
        "exporter_git_commit": current_git_commit(),
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "source_attempts": attempts,
        "files": dataset_files(root),
        "validation_command": (
            f"python validate_dataset.py --repo-id local/{dataset_name} "
            f"--root <dataset-root> --minimum-episodes {int(info['total_episodes'])}"
        ),
        "known_limitations": [
            "Stage-1 smoke/review checkpoint; success-only demonstrations.",
            "Camera mounts, task text, calibration profile, joint order, and FPS must remain unchanged.",
        ],
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"Prepared {dataset_name}: {manifest['total_episodes']} episode(s), "
        f"{manifest['total_frames']} frame(s), {len(manifest['files'])} checksummed file(s)."
    )


def verify(args: argparse.Namespace) -> None:
    root = args.dataset_root.resolve()
    info = check_layout(root)
    manifest = read_json(root / MANIFEST_NAME)
    if manifest.get("dataset_name") != root.name:
        raise ValueError("Share manifest dataset name does not match the directory")
    if int(manifest.get("total_episodes", -1)) != int(info["total_episodes"]):
        raise ValueError("Share manifest episode count does not match meta/info.json")
    if int(manifest.get("total_frames", -1)) != int(info["total_frames"]):
        raise ValueError("Share manifest frame count does not match meta/info.json")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Share manifest contains no file checksums")
    for entry in files:
        path = root / str(entry["path"])
        if not path.is_file():
            raise ValueError(f"Shared file is missing: {entry['path']}")
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"Shared file size changed: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Shared file checksum failed: {entry['path']}")
    print(
        f"Verified {root.name}: {manifest['total_episodes']} episode(s), "
        f"{manifest['total_frames']} frame(s), {len(files)} file checksum(s) passed."
    )


def api_json(base_url: str, path: str) -> dict[str, Any]:
    with urlopen(base_url.rstrip("/") + path, timeout=5) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"Unexpected response from {path}")
    return value


def guard(args: argparse.Namespace) -> None:
    status = api_json(args.base_url, "/api/training/status")
    if status.get("running") or status.get("finalizing"):
        raise ValueError(
            "Sharing is blocked while collection, saving, validation, or finalization is active"
        )
    if status.get("decision_pending") or status.get("record_phase"):
        raise ValueError("Sharing is blocked while a recording decision or save is pending")
    inventory = api_json(args.base_url, "/api/training/datasets").get("datasets")
    if not isinstance(inventory, list):
        raise ValueError("Training GUI returned an invalid dataset inventory")
    matches = [item for item in inventory if item.get("name") == args.dataset]
    if len(matches) != 1:
        raise ValueError(f"Dataset is not uniquely present in the GUI inventory: {args.dataset}")
    dataset = matches[0]
    if dataset.get("ready") is not True or dataset.get("validation_passed") is not True:
        raise ValueError(
            "Sharing requires the collector to be idle/finalized and the current dataset validation to PASS"
        )
    print(
        f"Share guard passed: collector idle, dataset finalized, validation current "
        f"({dataset.get('episodes')} episode(s), {dataset.get('frames')} frame(s))."
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--dataset-root", type=Path, required=True)
    prepare_parser.add_argument("--attempt-root", type=Path)
    prepare_parser.add_argument("--destination", required=True)
    prepare_parser.add_argument("--visibility", choices=("private", "team"), default="private")
    prepare_parser.set_defaults(handler=prepare)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--dataset-root", type=Path, required=True)
    verify_parser.set_defaults(handler=verify)
    guard_parser = commands.add_parser("guard")
    guard_parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    guard_parser.add_argument("--dataset", required=True)
    guard_parser.set_defaults(handler=guard)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Share blocked: {exc}") from exc


if __name__ == "__main__":
    main()
