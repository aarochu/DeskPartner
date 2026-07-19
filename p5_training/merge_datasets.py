"""Merge two+ LeRobot datasets into one on Modal, and push the result to the Hub.

Datasets must share fps / robot_type / features (checked by LeRobot). Sources are
pulled from the Hub with the hf-write token; the merged dataset is pushed to a new
repo you own, so training can then pull it by repo_id.

  ./.venv/bin/python -m p5_training.merge_datasets \
      --sources Cornerf/rebot-can-sort-stage1-v1-smoke Cornerf/rebot-two-can-recycle-v2-smoke \
      --merged  Cornerf/rebot-cansort-recycle-merged
"""

from __future__ import annotations

import argparse

import modal

from p5_training.modal_finetune import image  # reuse the cached lerobot image

app = modal.App("deskpartner-merge-datasets")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("hf-write")],
    timeout=3600,
)
def merge(sources: list[str], merged: str) -> str:
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dss = []
    for s in sources:
        # revision="main": these repos aren't tagged with the lerobot codebase
        # version, so the default revision resolution raises RevisionNotFoundError.
        d = LeRobotDataset(s, revision="main")  # downloads from Hub (private -> HF_TOKEN)
        print(f"loaded {s}: {d.meta.total_episodes} eps, {d.meta.total_frames} frames", flush=True)
        dss.append(d)

    print(f"merging {len(dss)} datasets -> {merged}", flush=True)
    out = merge_datasets(dss, output_repo_id=merged)
    print(f"merged: {out.meta.total_episodes} eps, {out.meta.total_frames} frames", flush=True)

    out.push_to_hub()  # uploads the merged dataset to the Hub
    print(f"DONE: https://huggingface.co/datasets/{merged}", flush=True)
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="source dataset repo ids")
    ap.add_argument("--merged", required=True, help="output merged dataset repo id")
    args = ap.parse_args()
    with modal.enable_output():
        with app.run():
            print("RESULT:", merge.remote(args.sources, args.merged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
