"""Run N MolmoAct 2 trials on the arm for the bake-off.

Stub: wire to Modal-served (or local) MolmoAct 2 inference + seeed_b601_dm_follower.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local ckpt or Modal volume path")
    parser.add_argument("--endpoint", default=None, help="Optional Modal inference URL")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    print(
        f"TODO: serve MolmoAct 2 from {args.endpoint or args.checkpoint}, "
        f"run {args.trials} single-arm trials (action chunks → follower)"
    )
    print("Score Y/N per trial into runs/bakeoff_*.md")


if __name__ == "__main__":
    main()
