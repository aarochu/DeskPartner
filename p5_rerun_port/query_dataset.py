"""List / filter / inspect episodes in the local catalog (SO ``query-dataset``).

Examples::

    python -m p5_rerun_port.query_dataset
    python -m p5_rerun_port.query_dataset --dataset cans
    python -m p5_rerun_port.query_dataset --dataset cans --tag "Good episode"
    python -m p5_rerun_port.query_dataset --dataset cans --episode episode_01 --entity follower/position
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from p5_rerun_port.catalog import query_episodes
from p5_rerun_port.constants import DEFAULT_CATALOG, DEFAULT_RECORDINGS_DIR


def _print_table(rows: list) -> None:
    if not rows:
        print("(no episodes)")
        return
    headers = ("dataset", "episode", "tag", "frames", "dur_s", "task")
    print(f"{headers[0]:16} {headers[1]:14} {headers[2]:14} {headers[3]:>6} {headers[4]:>6}  {headers[5]}")
    print("-" * 90)
    for e in rows:
        print(
            f"{e.dataset:16} {e.episode:14} {e.tag:14} {e.n_frames:6d} {e.duration_s:6.1f}  {e.task[:40]}"
        )


def _inspect_entity(traj_path: str, entity: str) -> None:
    path = Path(traj_path)
    if not path.exists():
        raise SystemExit(f"FAIL: missing traj sidecar {path} (re-record episode)")
    data = np.load(path, allow_pickle=True)
    times = data["times"]
    if entity.endswith("/goal") or entity.endswith("goal"):
        series = data["action"]
        label = "action/goal"
    else:
        series = data["state"]
        label = "state/position"
    joints = list(data["joint_names"]) if "joint_names" in data else [f"j{i}" for i in range(series.shape[1])]
    print(f"entity≈{entity}  source={label}  frames={len(times)}  joints={joints}")
    print(f"t[0]={times[0]:.3f}  t[-1]={times[-1]:.3f}")
    print(f"first: {np.array2string(series[0], precision=2, suppress_small=True)}")
    print(f"last:  {np.array2string(series[-1], precision=2, suppress_small=True)}")
    print(f"mean:  {np.array2string(series.mean(axis=0), precision=2, suppress_small=True)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Query local Rerun episode catalog")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--episode", default=None)
    ap.add_argument("--entity", default=None, help="Inspect traj for this entity (e.g. follower/position)")
    ap.add_argument(
        "--rerun-api",
        action="store_true",
        help="Use Rerun Query API (rr.server.Server + reader) instead of traj.npz sidecar",
    )
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR)
    args = ap.parse_args(argv)

    if args.rerun_api:
        if not args.dataset:
            raise SystemExit("--rerun-api requires --dataset")
        from p5_rerun_port.query_api_cli import main as query_api_main

        api_argv = ["--dataset", args.dataset]
        if args.tag:
            api_argv += ["--tag", args.tag]
        if args.episode:
            api_argv += ["--episode", args.episode]
        if args.entity:
            api_argv += ["--entity", args.entity]
        else:
            api_argv += ["--schema"]
        api_argv += ["--recordings-dir", str(args.recordings_dir), "--catalog", str(args.catalog)]
        return query_api_main(api_argv)

    rows = query_episodes(
        dataset=args.dataset,
        tag=args.tag,
        episode=args.episode,
        catalog_path=args.catalog,
        recordings_dir=args.recordings_dir,
    )
    _print_table(rows)

    if args.entity:
        if len(rows) != 1:
            raise SystemExit("--entity requires exactly one matching episode (pass --dataset and --episode)")
        _inspect_entity(rows[0].traj_path, args.entity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
