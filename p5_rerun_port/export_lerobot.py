"""Export tagged episodes to LeRobot v3-shaped dataset (reBot schema).

Reads the local catalog + ``.traj.npz`` (+ optional ``frames/{front,side}`` JPEGs)
written by ``record_episode``. Prefer LeRobot's ``LeRobotDataset.create`` when
installed; otherwise writes a compatible folder (parquet + meta + images).

Examples::

    python -m p5_rerun_port.export_lerobot --dataset cans --tag "Good episode"
    python -m p5_rerun_port.export_lerobot --dataset cans --tag "" --repo-id deskpartner/cans_rerun
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from p5_rerun_port.catalog import query_episodes
from p5_rerun_port.constants import (
    DEFAULT_CATALOG,
    DEFAULT_DATASETS_DIR,
    DEFAULT_RECORDINGS_DIR,
    JOINT_NAMES,
    ROBOT_TYPE,
)


def _load_episode_arrays(traj_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(traj_path, allow_pickle=True)
    return data["times"], data["state"].astype(np.float32), data["action"].astype(np.float32)


def _frame_dirs_for(rrd_path: str) -> dict[str, Path]:
    root = Path(rrd_path).with_suffix("")
    return {"front": root / "frames" / "front", "side": root / "frames" / "side"}


def _export_with_lerobot(
    episodes: list,
    *,
    repo_id: str,
    root: Path,
    fps: float,
) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    # Probe first episode for image shape
    cam_names = ["front", "side"]
    features: dict = {
        "action": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": list(JOINT_NAMES)},
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
    }
    sample_dirs = _frame_dirs_for(episodes[0].rrd_path)
    import cv2

    for cam in cam_names:
        jpgs = sorted(sample_dirs[cam].glob("*.jpg")) if sample_dirs[cam].is_dir() else []
        if not jpgs:
            continue
        img = cv2.imread(str(jpgs[0]))
        if img is None:
            continue
        h, w, c = img.shape
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
        }

    out = root / repo_id
    if out.exists():
        raise SystemExit(f"FAIL: export path exists: {out} (remove or change --repo-id)")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        features=features,
        root=out,
        robot_type=ROBOT_TYPE,
        use_videos=False,
    )

    for ep in episodes:
        _times, state, action = _load_episode_arrays(Path(ep.traj_path))
        dirs = _frame_dirs_for(ep.rrd_path)
        for i in range(len(action)):
            frame = {
                "action": action[i],
                "observation.state": state[i],
                "task": ep.task or "rebot_task",
            }
            for cam in cam_names:
                key = f"observation.images.{cam}"
                if key not in features:
                    continue
                jpg = dirs[cam] / f"{i:06d}.jpg"
                if jpg.exists():
                    bgr = cv2.imread(str(jpg))
                    frame[key] = bgr[:, :, ::-1]  # RGB
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"exported episode {ep.episode} ({len(action)} frames)", flush=True)

    if hasattr(dataset, "finalize"):
        dataset.finalize()
    return out


def _export_fallback(
    episodes: list,
    *,
    repo_id: str,
    root: Path,
    fps: float,
) -> Path:
    """Write a LeRobot-v3-shaped folder without the lerobot package."""
    out = root / repo_id
    if out.exists():
        raise SystemExit(f"FAIL: export path exists: {out}")
    data_dir = out / "data" / "chunk-000"
    meta_dir = out / "meta"
    video_dir = out / "videos" / "chunk-000"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    episodes_meta = []
    tasks: dict[str, int] = {}
    total_frames = 0

    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore

    for ep_idx, ep in enumerate(episodes):
        times, state, action = _load_episode_arrays(Path(ep.traj_path))
        n = len(action)
        task = ep.task or "rebot_task"
        if task not in tasks:
            tasks[task] = len(tasks)
        rows = {
            "episode_index": np.full(n, ep_idx, dtype=np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
            "timestamp": times.astype(np.float32),
            "task_index": np.full(n, tasks[task], dtype=np.int64),
        }
        for j, name in enumerate(JOINT_NAMES):
            rows[f"observation.state.{name}"] = state[:, j]
            rows[f"action.{name}"] = action[:, j]

        stem = f"episode_{ep_idx:06d}"
        if pd is not None:
            df = pd.DataFrame(rows)
            parquet_path = data_dir / f"{stem}.parquet"
            df.to_parquet(parquet_path, index=False)
        else:
            np.savez_compressed(data_dir / f"{stem}.npz", **rows)

        # Copy / note image frames
        dirs = _frame_dirs_for(ep.rrd_path)
        for cam, src in dirs.items():
            if not src.is_dir():
                continue
            dst = video_dir / f"observation.images.{cam}" / stem
            dst.mkdir(parents=True, exist_ok=True)
            for jpg in sorted(src.glob("*.jpg")):
                target = dst / jpg.name
                if not target.exists():
                    target.write_bytes(jpg.read_bytes())

        episodes_meta.append(
            {
                "episode_index": ep_idx,
                "tasks": [task],
                "length": n,
                "rerun_episode": ep.episode,
                "tag": ep.tag,
            }
        )
        total_frames += n
        print(f"exported episode {ep.episode} ({n} frames)", flush=True)

    info = {
        "codebase_version": "v3.0",
        "robot_type": ROBOT_TYPE,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [len(JOINT_NAMES)],
                "names": list(JOINT_NAMES),
            },
            "action": {
                "dtype": "float32",
                "shape": [len(JOINT_NAMES)],
                "names": list(JOINT_NAMES),
            },
            "observation.images.front": {"dtype": "image"},
            "observation.images.side": {"dtype": "image"},
        },
        "note": (
            "Exported by p5_rerun_port from Rerun recordings. "
            "Units follow the values logged at record time (Seeed/LeRobot wire units when live; "
            "synthetic degrees-like values when --fake). Do not mix with SO ±100 without conversion."
        ),
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episodes_meta:
            f.write(json.dumps(row) + "\n")
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for task, idx in tasks.items():
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    readme = out / "README_EXPORT.md"
    readme.write_text(
        f"# {repo_id}\n\n"
        f"robot_type: `{ROBOT_TYPE}`\n"
        f"joints ({len(JOINT_NAMES)}): {', '.join(JOINT_NAMES)}\n"
        f"cameras: front, side\n"
        f"episodes: {len(episodes)}  frames: {total_frames}\n"
        "\nSource: Rerun `.rrd` + local catalog via `p5_rerun_port`.\n",
        encoding="utf-8",
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export Rerun episodes to LeRobot v3 (reBot)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", default="Good episode", help='Filter tag; pass "" for all')
    ap.add_argument("--repo-id", default=None, help="Default: deskpartner/<dataset>_rerun")
    ap.add_argument("--out", type=Path, default=DEFAULT_DATASETS_DIR)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR)
    ap.add_argument("--fallback", action="store_true", help="Skip LeRobotDataset even if installed")
    args = ap.parse_args(argv)

    tag = args.tag
    rows = query_episodes(
        dataset=args.dataset,
        tag=tag if tag != "" else None,
        catalog_path=args.catalog,
        recordings_dir=args.recordings_dir,
    )
    if tag == "":
        # re-query without tag filter already done via None
        pass
    if not rows:
        raise SystemExit(f"FAIL: no episodes for dataset={args.dataset!r} tag={args.tag!r}")

    repo_id = args.repo_id or f"deskpartner/{args.dataset}_rerun"
    print(f"exporting {len(rows)} episode(s) -> {args.out / repo_id}", flush=True)

    if not args.fallback:
        try:
            out = _export_with_lerobot(rows, repo_id=repo_id, root=args.out, fps=args.fps)
            print(f"OK: LeRobot dataset at {out}", flush=True)
            return 0
        except SystemExit:
            raise
        except Exception as exc:
            print(f"WARN: LeRobotDataset export failed ({exc}); using fallback writer", flush=True)

    out = _export_fallback(rows, repo_id=repo_id, root=args.out, fps=args.fps)
    print(f"OK: fallback dataset at {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
