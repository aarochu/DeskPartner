"""Record N episodes with manual ready keypress + verify after each.

  python -m p4_data_collection.batch_record --num 50
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from p4_data_collection.config_loader import default_dataset_root, load_recording_config
from p4_data_collection.record_episode import main as record_main
from p4_data_collection.verify_episode_format import verify


def _ready(prompt: str) -> bool:
    try:
        line = input(prompt).strip().lower()
    except EOFError:
        return False
    if line in {"q", "quit", "abort"}:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--task", default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--skip-verify", action="store_true", help="DANGEROUS — do not use tonight")
    args = ap.parse_args()

    if args.skip_verify:
        print("FAIL: --skip-verify is disabled for fail-loud collection. Remove the flag.")
        return 2

    cfg = load_recording_config(args.config)
    task = args.task or cfg["dataset"]["single_task"]
    repo_id = cfg["dataset"]["repo_id"]
    root = default_dataset_root(repo_id)

    passed: list[int] = []
    failed: list[int] = []

    print(f"batch_record: N={args.num} task={task!r} repo_id={repo_id}")
    print("Cameras MUST stay locked. Run check_camera_lock before starting.\n")

    for i in range(1, args.num + 1):
        print(f"\n======== Episode {i}/{args.num} ========")
        if not _ready(
            "Reset desk (paper staged consistently). Type Enter when READY (q=abort): "
        ):
            print("Aborted by operator")
            break

        rc = record_main(["--task", task, "--num", "1"])
        if rc != 0:
            print(f"FAIL: recorder exited {rc} for episode slot {i}")
            failed.append(i)
            if not _ready("Continue to next episode anyway? Enter=yes q=stop: "):
                break
            continue

        vrc = verify(root, cfg)
        if vrc == 0:
            print(f"PASS episode slot {i}")
            passed.append(i)
        else:
            print(f"FAIL episode slot {i} — fix before continuing the batch")
            failed.append(i)
            if not _ready("Continue anyway? Enter=yes q=stop: "):
                break

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = Path("runs") / f"batch_record_{stamp}.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"passed={len(passed)}/{len(passed)+len(failed)}",
        f"passed_ids={passed}",
        f"failed_ids={failed}",
        f"task={task}",
        f"repo_id={repo_id}",
        f"dataset_root={root}",
    ]
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n======== SUMMARY ========")
    for line in lines:
        print(line)
    print(f"wrote {summary_path}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
