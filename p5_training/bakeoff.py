"""Sunday morning go/no-go: scripted P1 vs MolmoAct2 policy, same staging.

  python -m p5_training.bakeoff --trials 10 --checkpoint PATH [--live]

Prints a single unambiguous VERDICT line.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

from p5_training.eval_checkpoint import _append_csv, run_policy_trial


def run_scripted_trial(trial: int, object_name: str, live: bool) -> tuple[bool, str]:
    print(f"\n--- Scripted trial {trial}: stage '{object_name}' ---")
    if live:
        # Optional auto-run of P1 primitive when coords known; default operator label
        try:
            from p1_arm_motion.arm_client import ArmConfig, make_arm_client
            from p1_arm_motion.pick_and_drop import pick_and_drop

            print("Live scripted path available. Enter target x_mm y_mm (or 'skip' to label only):")
            line = input().strip()
            if line and line.lower() != "skip":
                x_s, y_s = line.split()
                cfg = ArmConfig.from_yaml()
                arm = make_arm_client(cfg, dry_run=False)
                arm.connect()
                try:
                    pick_and_drop(
                        arm,
                        cfg,
                        target_xy_mm=(float(x_s), float(y_s)),
                        item_type="paper",
                        destination_name="trash",
                    )
                finally:
                    arm.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"scripted auto-run error: {exc} — fall back to label")

    while True:
        ans = input("scripted success? [y/n/q]: ").strip().lower()
        if ans == "q":
            raise SystemExit("aborted")
        if ans in {"y", "n"}:
            return ans == "y", "operator_labeled_scripted"
        print("enter y, n, or q")


def _rate(rows: list[dict], method: str) -> tuple[int, int, float]:
    sub = [r for r in rows if r["method"] == method]
    n = len(sub)
    s = sum(int(r["success"]) for r in sub)
    return s, n, (s / n if n else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--object", default="crumpled_paper")
    ap.add_argument("--csv", type=Path, default=Path("runs/bakeoff_trials.csv"))
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"BAKEOFF start {stamp} trials={args.trials} object={args.object}")
    print("Same staging for both methods. Scripted first, then policy.\n")

    # Scripted block
    for i in range(1, args.trials + 1):
        ok, notes = run_scripted_trial(i, args.object, live=args.live)
        _append_csv(
            args.csv,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": "scripted",
                "trial": i,
                "object": args.object,
                "success": int(ok),
                "notes": notes,
            },
        )

    # Policy block
    for i in range(1, args.trials + 1):
        ok, notes = run_policy_trial(i, args.object, args.endpoint, args.checkpoint)
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

    # Summarize from CSV
    rows = list(csv.DictReader(args.csv.open()))
    # only this object's recent methods — filter by object
    rows = [r for r in rows if r.get("object") == args.object]
    s_ok, s_n, s_rate = _rate(rows, "scripted")
    p_ok, p_n, p_rate = _rate(rows, "molmoact2")

    # Prefer last N of each method
    scripted = [r for r in rows if r["method"] == "scripted"][-args.trials :]
    policy = [r for r in rows if r["method"] == "molmoact2"][-args.trials :]
    s_ok = sum(int(r["success"]) for r in scripted)
    p_ok = sum(int(r["success"]) for r in policy)
    s_n = len(scripted)
    p_n = len(policy)
    s_rate = s_ok / s_n if s_n else 0.0
    p_rate = p_ok / p_n if p_n else 0.0

    print("\n======== BAKEOFF SUMMARY ========")
    print(f"scripted : {s_ok}/{s_n} = {s_rate:.0%}")
    print(f"molmoact2: {p_ok}/{p_n} = {p_rate:.0%}")

    if p_ok > s_ok:
        winner = "molmoact2"
    elif s_ok > p_ok:
        winner = "scripted"
    else:
        winner = "scripted"  # tie -> prefer reliability (GROUND_TRUTH)

    print(f"VERDICT: WINNER={winner} (scripted={s_ok}/{s_n}, molmoact2={p_ok}/{p_n})")
    print(f"csv={args.csv}")

    out = Path("runs") / f"bakeoff_{stamp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# Bakeoff {stamp}\n\n"
        f"- scripted: {s_ok}/{s_n}\n"
        f"- molmoact2: {p_ok}/{p_n}\n"
        f"- VERDICT: WINNER={winner}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
