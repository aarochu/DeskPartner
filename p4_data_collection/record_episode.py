"""Run lerobot-record with locked config/recording.yaml (fail-loud if missing)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from p4_data_collection.config_loader import cameras_cli_dict, load_recording_config


def build_cmd(cfg: dict, task: str, num_episodes: int) -> list[str]:
    robot = cfg["robot"]
    teleop = cfg["teleop"]
    ds = cfg["dataset"]
    cams = cameras_cli_dict(cfg)

    cmd = [
        "lerobot-record",
        f"--robot.type={robot['type']}",
        f"--robot.port={robot['port']}",
        f"--robot.id={robot['id']}",
        f"--robot.can_adapter={robot['can_adapter']}",
        f"--robot.cameras={cams}",
        f"--teleop.type={teleop['type']}",
        f"--teleop.port={teleop['port']}",
        f"--teleop.id={teleop['id']}",
        f"--display_data={'true' if ds.get('display_data', True) else 'false'}",
        f"--dataset.repo_id={ds['repo_id']}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.single_task={task}",
        f"--dataset.push_to_hub={'true' if ds.get('push_to_hub') else 'false'}",
        f"--dataset.episode_time_s={ds['episode_time_s']}",
        f"--dataset.reset_time_s={ds['reset_time_s']}",
    ]
    return cmd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Record LeRobot episodes (single-arm, dual-cam)")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument(
        "--task",
        default=None,
        help='single_task string (default from recording.yaml)',
    )
    ap.add_argument("--num", type=int, default=1, help="num_episodes for this invocation")
    ap.add_argument("--dry-print", action="store_true", help="Print command only, do not run")
    args = ap.parse_args(argv)

    cfg = load_recording_config(args.config)
    task = args.task or cfg["dataset"]["single_task"]
    if not task.strip():
        raise SystemExit("FAIL: empty --task / single_task")

    # Hard guard: never silently record bi_* robot types
    rtype = cfg["robot"]["type"]
    if "bi_" in rtype or "bimanual" in rtype.lower():
        raise SystemExit(f"FAIL: robot.type looks bimanual: {rtype}")

    cmd = build_cmd(cfg, task=task, num_episodes=args.num)
    print("CMD:", " ".join(cmd))
    print(f"task={task!r} cameras={list(cfg['cameras'])} repo_id={cfg['dataset']['repo_id']}")

    if args.dry_print:
        return 0

    if not shutil.which("lerobot-record"):
        raise SystemExit(
            "FAIL: lerobot-record not on PATH. Install Seeed LeRobot + "
            "lerobot-robot-seeed-b601 first (see root README)."
        )

    # Ensure serial perms hint
    if sys.platform.startswith("linux"):
        for port in (cfg["robot"]["port"], cfg["teleop"]["port"]):
            if not Path(port).exists():
                print(f"WARN: port missing {port}")

    env = os.environ.copy()
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        print(f"FAIL: lerobot-record exited {rc}")
        return rc
    print("OK: lerobot-record finished — run verify_episode_format next")
    return 0


if __name__ == "__main__":
    sys.exit(main())
