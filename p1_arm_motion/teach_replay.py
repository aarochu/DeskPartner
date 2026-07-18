"""Teach-by-grabbing: gravity-comp record → replay.

Stub. Real path:
  1. Run SDK example/9_gravity_compensation.py (judge floats the arm)
  2. Log joint samples while they drag a motion
  3. Replay with trajectory controller (example/8_arm_traj_control.py)
"""

from __future__ import annotations

from pathlib import Path


def record_stub(out: Path = Path("data/teach_replay/last.jsonl")) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# TODO: stream joint angles while gravity-comp is active\n")
    print(f"Stub record → {out}")


def replay_stub(path: Path = Path("data/teach_replay/last.jsonl")) -> None:
    print(f"Stub replay ← {path} (wire to 8_arm_traj_control)")


if __name__ == "__main__":
    record_stub()
    replay_stub()
