"""Eval a MolmoAct2 checkpoint for N live trials; log CSV for bake-off.

  python -m p5_training.eval_checkpoint --checkpoint /path/or/modal --trials 10 --live

Inference: Modal endpoint OR local stub. Action chunks => per-call latency OK.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["timestamp", "method", "trial", "object", "success", "notes"]
        )
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_policy_trial(trial: int, object_name: str, endpoint: str | None, checkpoint: Path) -> tuple[bool, str]:
    """Return (success, notes). Wire to Modal/chunk inference + arm at venue."""
    # Placeholder: operator marks success after watching the arm.
    print(f"\n--- Policy trial {trial}: stage '{object_name}', then press y=success n=fail ---")
    print(f"checkpoint={checkpoint} endpoint={endpoint}")
    # TODO: call Modal inference -> action chunks -> follower
    while True:
        ans = input("success? [y/n/q]: ").strip().lower()
        if ans == "q":
            raise SystemExit("aborted")
        if ans in {"y", "n"}:
            return ans == "y", "operator_labeled"
        print("enter y, n, or q")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--endpoint", default=None, help="Modal inference URL if serving remote")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--object", default="crumpled_paper")
    ap.add_argument("--csv", type=Path, default=Path("runs/bakeoff_trials.csv"))
    ap.add_argument("--live", action="store_true", help="reserved: require real arm")
    args = ap.parse_args()

    if not args.checkpoint.exists() and args.endpoint is None:
        print(f"WARN: checkpoint path {args.checkpoint} missing — continuing if endpoint set")

    successes = 0
    for i in range(1, args.trials + 1):
        ok, notes = run_policy_trial(i, args.object, args.endpoint, args.checkpoint)
        successes += int(ok)
        _append_csv(
            args.csv,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": "molmoact2",
                "trial": i,
                "object": args.object,
                "success": int(ok),
                "notes": notes,
            },
        )
        print(f"trial {i}: {'SUCCESS' if ok else 'FAIL'}")

    rate = successes / args.trials if args.trials else 0.0
    print(f"\nPOLICY RESULT: {successes}/{args.trials} = {rate:.0%}")
    print(f"csv={args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
