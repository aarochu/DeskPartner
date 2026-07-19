"""Replay a recorded ``follower/goal`` trajectory on the reBot follower.

Examples::

    python -m p5_rerun_port.replay_episode --dataset cans --episode episode_01 --fake --speed 0.5
    python -m p5_rerun_port.replay_episode --dataset cans --episode episode_01 --speed 0.5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from p5_rerun_port.catalog import query_episodes
from p5_rerun_port.config import load_recording_config
from p5_rerun_port.constants import DEFAULT_CATALOG, DEFAULT_RECORDINGS_DIR, FOLLOWER, JOINT_NAMES
from p5_rerun_port.hardware import make_session
from p5_rerun_port.takes import begin_recording, finish_recording, log_scalars, set_timeline


def _load_traj(traj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not traj_path.exists():
        raise SystemExit(f"FAIL: missing {traj_path}")
    data = np.load(traj_path, allow_pickle=True)
    return data["times"].astype(np.float64), data["action"].astype(np.float64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay a Rerun episode trajectory on reBot")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--speed", type=float, default=0.5, help="Playback speed (keep <=1 first run)")
    ap.add_argument("--ramp-seconds", type=float, default=2.0)
    ap.add_argument("--fake", action="store_true", help="Dry-run: print goals, no hardware")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR)
    args = ap.parse_args(argv)

    if not 0.0 < args.speed <= 2.0:
        raise SystemExit("--speed must be in (0, 2]")

    rows = query_episodes(
        dataset=args.dataset,
        episode=args.episode,
        catalog_path=args.catalog,
        recordings_dir=args.recordings_dir,
    )
    if not rows:
        raise SystemExit(f"FAIL: episode not found: {args.dataset}/{args.episode}")
    ep = rows[0]
    times, goals = _load_traj(Path(ep.traj_path))
    if len(goals) == 0:
        raise SystemExit("FAIL: empty trajectory")
    if goals.shape[1] != len(JOINT_NAMES):
        print(
            f"WARN: traj has {goals.shape[1]} joints; expected {len(JOINT_NAMES)} ({JOINT_NAMES})",
            flush=True,
        )

    print(
        f"replay: {ep.episode}  frames={len(goals)}  dur={times[-1]:.1f}s  speed={args.speed}",
        flush=True,
    )

    cfg = load_recording_config()
    session = make_session(fake=args.fake, teleop=False, recording_cfg=cfg)
    session.start()

    live_rrd = Path(ep.rrd_path).parent / f"{ep.episode}_replay.rrd"
    rec = begin_recording(
        live_rrd,
        episode=f"{ep.episode}_replay",
        dataset=ep.dataset,
        task=f"replay:{ep.task}",
        spawn_viewer=not args.no_viewer,
        save_only=args.no_viewer,
    )

    # Ramp to start
    start = goals[0]
    ramp_n = max(int(args.ramp_seconds * 30), 1)
    try:
        try:
            current, _ = session.read()
        except Exception:
            current = list(start)
        for i in range(ramp_n):
            alpha = (i + 1) / ramp_n
            goal = [float(c + alpha * (s - c)) for c, s in zip(current, start, strict=False)]
            # pad/truncate
            if len(goal) < len(start):
                goal = list(start)
            session.send_goal(goal[: goals.shape[1]])
            set_timeline(rec, time.time())
            log_scalars(rec, f"{FOLLOWER}/goal", goal[: goals.shape[1]])
            time.sleep(args.ramp_seconds / ramp_n)

        t0_wall = time.time()
        t0_traj = times[0]
        for i, (t, goal) in enumerate(zip(times, goals, strict=True)):
            target_wall = t0_wall + (t - t0_traj) / args.speed
            delay = target_wall - time.time()
            if delay > 0:
                time.sleep(delay)
            session.send_goal(list(map(float, goal)))
            set_timeline(rec, time.time())
            log_scalars(rec, f"{FOLLOWER}/goal", goal)
            if i % 30 == 0:
                print(f"  frame {i}/{len(goals)}", flush=True)
    except KeyboardInterrupt:
        print("\nstopping replay", flush=True)
    finally:
        finish_recording(rec, dataset=ep.dataset, task=ep.task, tag="replay")
        session.close()

    print("OK: replay finished (torque/disconnect handled by session.close)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
