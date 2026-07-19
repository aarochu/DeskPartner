"""Push a MolmoAct2 checkpoint from the Modal ckpt volume to a HF model repo.

Cloud-to-cloud: reads the checkpoint off the Modal volume and uploads it to the
Hub, so the multi-GB weights never touch your laptop.

Setup (write token for the account that OWNS the target repo):
    ./.venv/bin/modal secret create hf-write HF_TOKEN=hf_your_write_token

Run:
    ./.venv/bin/python -m p5_training.push_checkpoint \
        --exp rebot_smoke5_molmoact2 \
        --repo-id Cornerf/rebot-can-sort-molmoact2-5-examples

Load the result later with:  --policy.path=<repo-id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import modal
except ImportError:  # pragma: no cover
    modal = None

CKPT_VOLUME = "deskpartner-molmoact2-ckpts"

if modal is not None:
    app = modal.App("deskpartner-push-checkpoint")
    ckpt_vol = modal.Volume.from_name(CKPT_VOLUME, create_if_missing=True)
    image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub")

    @app.function(
        image=image,
        volumes={"/ckpts": ckpt_vol},
        secrets=[modal.Secret.from_name("hf-write")],  # HF_TOKEN with write access
        timeout=3600,
    )
    def push(exp_name: str, repo_id: str, checkpoint: str, private: bool) -> str:
        import os

        from huggingface_hub import HfApi

        token = os.environ["HF_TOKEN"]
        base = Path("/ckpts") / exp_name / "checkpoints"
        if not base.exists():
            raise SystemExit(f"FAIL: no checkpoints under {base}")

        if checkpoint == "last":
            steps = sorted(p.name for p in base.iterdir() if p.name.isdigit())
            if not steps:
                raise SystemExit(f"FAIL: no numbered checkpoints in {base}")
            step = steps[-1]
        else:
            step = checkpoint
        folder = base / step / "pretrained_model"
        if not folder.exists():
            raise SystemExit(f"FAIL: no pretrained_model at {folder}")

        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
        print(f"uploading {folder} -> {repo_id}", flush=True)
        api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"MolmoAct2 checkpoint {exp_name}/{step}",
        )
        url = f"https://huggingface.co/{repo_id}"
        print(f"DONE: {url}", flush=True)
        return url


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="exp_name (checkpoint dir on the volume)")
    ap.add_argument("--repo-id", required=True, help="target HF model repo, e.g. user/model")
    ap.add_argument("--checkpoint", default="last", help="'last' or a step like 000050")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    args = ap.parse_args()

    if modal is None:
        raise SystemExit("modal not installed")
    with modal.enable_output():
        with app.run():
            print("RESULT:", push.remote(args.exp, args.repo_id, args.checkpoint, args.private))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
