"""Modal MolmoAct2 fine-tune via LeRobot's official `lerobot-train`.

MolmoAct2 is now a first-class LeRobot policy, so we DON'T need a bespoke Ai2
script — we run the standard training entrypoint on a cloud GPU:

    accelerate launch -m lerobot.scripts.lerobot_train --policy.type=molmoact2 ...

Docs: https://huggingface.co/docs/lerobot/main/en/molmoact2

Warm-start options (set `base_checkpoint` in the train yaml):
  * allenai/MolmoAct2            (generic base; builds a fresh processor from OUR
                                  dataset metadata — most robust for a custom
                                  reBot dataset; this is the DEFAULT)
  * lerobot/MolmoAct2-SO100_101-LeRobot  (the published "SO-101 base"; closer
                                  embodiment but expects cam0/cam1 keys + carries
                                  an SO-101 joint-frame convention — try as a
                                  second experiment)

MolmoAct2 is large: LoRA / action-expert-only fine-tune needs ~16-40 GiB VRAM,
so this always runs on a cloud A100/H100 (never the local GPU).

  # show the exact command, spend nothing:
  python -m p5_training.modal_finetune --dry-print

  # real run (dataset must be uploaded to the Modal 'deskpartner-data' volume):
  python -m p5_training.modal_finetune --dataset-repo-id deskpartner/cans_single

Requires: pip install modal + modal token new.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

try:
    import modal
except ImportError:  # pragma: no cover
    modal = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CFG = ROOT / "p5_training" / "configs" / "molmoact2_single_arm.yaml"

# ---------------------------------------------------------------------------
# Knobs (overridable via train yaml / CLI / env)
# ---------------------------------------------------------------------------
# L4 (24GB) is allowed on Modal WITHOUT a payment method and fits action-expert-only
# fine-tuning (~16.5 GiB). Set DESKPARTNER_MODAL_GPU=A100-80GB / H100 for speed or LoRA
# (those require a Modal payment method / team credits).
GPU = os.environ.get("DESKPARTNER_MODAL_GPU", "L4")
WALL_CLOCK_TIMEOUT_S = 6 * 60 * 60  # hard stop — don't eat all of Saturday night
BASE_CHECKPOINT = "allenai/MolmoAct2"  # generic base; fresh processor from our data
CKPT_VOLUME = "deskpartner-molmoact2-ckpts"
DATA_VOLUME = "deskpartner-data"


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def build_train_argv(
    cfg: dict, dataset_repo_id: str, dataset_root: str, output_dir: str
) -> list[str]:
    """Construct the `lerobot-train` argv for a MolmoAct2 fine-tune.

    Returned as a list (no shell), so values with spaces (control_mode,
    setup_type) are passed through intact.
    """
    p = cfg.get("policy", {}) or {}
    t = cfg.get("train", {}) or {}
    ft = cfg.get("finetune", {}) or {}

    mode = ft.get("mode", "action_expert_only")  # action_expert_only | lora
    if ft.get("full_finetune") or mode == "full":
        raise RuntimeError("FAIL: full fine-tune is forbidden for this hackathon job")

    base = cfg.get("base_checkpoint", BASE_CHECKPOINT)
    image_keys = p.get(
        "image_keys",
        ["observation.images.front", "observation.images.side"],
    )

    argv = [
        "accelerate",
        "launch",
        "--num_processes=1",
        "--mixed_precision=bf16",
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={dataset_repo_id}",
        # Local root when the dataset is on the Modal 'data' volume; omit to let
        # LeRobot download the dataset straight from the HF Hub by repo_id.
        *([f"--dataset.root={dataset_root}"] if dataset_root else []),
        # Pin the exact dataset commit when pulling from the Hub. When training from
        # a local volume copy (dataset_root set), the on-disk files ARE that revision,
        # so we skip the flag to avoid a Hub round-trip / auth check.
        *(
            [f"--dataset.revision={cfg['dataset_revision']}"]
            if cfg.get("dataset_revision") and not dataset_root
            else []
        ),
        "--dataset.video_backend=pyav",  # known gotcha: default backend can choke
        "--dataset.image_transforms.enable=true",
        "--policy.type=molmoact2",
        # Generic HF base -> build a fresh processor from OUR dataset metadata.
        f"--policy.checkpoint_path={base}",
        "--policy.device=cuda",
        f"--policy.chunk_size={p.get('chunk_size', 10)}",
        f"--policy.n_action_steps={p.get('n_action_steps', 10)}",
        f"--policy.control_mode={p.get('control_mode', 'absolute joint pose')}",
        f"--policy.setup_type={p.get('setup_type', 'single rebot b601 arm on a desk')}",
        f"--policy.image_keys={json.dumps(image_keys)}",
        "--policy.model_dtype=bfloat16",
        f"--policy.num_flow_timesteps={p.get('num_flow_timesteps', 8)}",
        "--policy.gradient_checkpointing=true",
        # raw teleop gripper values aren't in [-1,1]; include them in normalization
        f"--policy.normalize_gripper={str(p.get('normalize_gripper', True)).lower()}",
        f"--output_dir={output_dir}",
        f"--steps={t.get('steps', 4000)}",
        f"--batch_size={t.get('batch_size', 16)}",
        f"--num_workers={t.get('num_workers', 4)}",
        f"--save_freq={t.get('save_freq', 1000)}",
        "--save_checkpoint=true",
        "--log_freq=20",
        "--env_eval_freq=-1",
    ]

    # Fine-tune mode (small-data recommendation: keep action expert trainable).
    if mode == "action_expert_only":
        argv += ["--policy.action_mode=continuous", "--policy.train_action_expert_only=true"]
    elif mode == "lora":
        argv += ["--policy.action_mode=both", "--policy.enable_lora_vlm=true"]
    else:
        raise RuntimeError(f"FAIL: unknown finetune mode {mode!r} (action_expert_only|lora)")

    # Per-checkpoint push to the HF Hub (optional). When on, lerobot uploads each
    # saved checkpoint to hub.repo_id as it trains — a durable backup of every step.
    hub = cfg.get("hub") or {}
    if hub.get("save_checkpoint_to_hub") and hub.get("repo_id"):
        argv += [
            "--policy.push_to_hub=true",
            f"--policy.repo_id={hub['repo_id']}",
            "--save_checkpoint_to_hub=true",
        ]
    else:
        argv.append("--policy.push_to_hub=false")

    # Normalization: quantile is MolmoAct2's default but needs pre-computed stats.
    # mean_std works out-of-the-box on a fresh dataset (simplest first run).
    if (cfg.get("normalization", "mean_std")) == "mean_std":
        argv.append(
            '--policy.normalization_mapping={"ACTION": "MEAN_STD", '
            '"STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'
        )

    wandb = cfg.get("wandb", {}) or {}
    if wandb.get("enable"):
        argv.append("--wandb.enable=true")
        if wandb.get("project"):
            argv.append(f"--wandb.project={wandb['project']}")
        if wandb.get("entity"):
            argv.append(f"--wandb.entity={wandb['entity']}")
    else:
        argv.append("--wandb.enable=false")

    return argv


if modal is not None:
    app = modal.App("deskpartner-molmoact2")
    ckpt_vol = modal.Volume.from_name(CKPT_VOLUME, create_if_missing=True)
    data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.12")  # lerobot requires >=3.12
        .apt_install("git", "ffmpeg")
        .pip_install("torch", "torchvision", "accelerate", "wandb")
        # MolmoAct2 policy lives in LeRobot main; install with the molmoact2 extra.
        .pip_install(
            # molmoact2 = the policy; dataset = HF datasets loader (LeRobotDataset).
            "lerobot[molmoact2,dataset] @ git+https://github.com/huggingface/lerobot.git"
        )
        .env({"HF_HOME": "/ckpts/hf"})  # cache base checkpoint in the volume
    )

    @app.function(
        image=image,
        gpu=GPU,
        timeout=WALL_CLOCK_TIMEOUT_S,
        volumes={"/ckpts": ckpt_vol, "/data": data_vol},
        # HF_TOKEN for pulling the private dataset from the Hub (never in code/logs).
        # hf-write is the known-good token (read+write access to Cornerf repos).
        # wandb secret (WANDB_API_KEY) powers the live loss dashboard.
        secrets=[modal.Secret.from_name("hf-write"), modal.Secret.from_name("wandb")],
    )
    def train(argv: list[str], exp_name: str, resume: bool = False) -> dict:
        """Run `lerobot-train` for MolmoAct2; checkpoints land in /ckpts volume."""
        import shutil
        import subprocess

        start = time.time()
        out_dir = Path("/ckpts") / exp_name
        # Fresh run: lerobot-train refuses a pre-existing output dir, so clear a stale
        # one. RESUME run: keep the dir — lerobot loads the last checkpoint from it.
        if out_dir.exists() and not resume:
            print(f"clearing existing output dir {out_dir}", flush=True)
            shutil.rmtree(out_dir)

        print("RUN:", " ".join(argv), flush=True)
        remaining = WALL_CLOCK_TIMEOUT_S - 120
        try:
            subprocess.run(argv, check=True, timeout=max(60, remaining))
        except subprocess.TimeoutExpired:
            print("TIMEOUT: wall-clock hit — checkpoints so far are committed", flush=True)
            ckpt_vol.commit()
            raise
        ckpt_vol.commit()
        return {"out_dir": str(out_dir), "exp": exp_name, "elapsed_s": round(time.time() - start, 1)}

else:
    app = None


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune MolmoAct2 on Modal via lerobot-train")
    ap.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CFG)
    ap.add_argument("--dataset-repo-id", default=None)
    ap.add_argument("--dataset-root", default=None, help="default /data/<repo_id> on the Modal volume")
    ap.add_argument("--dry-print", action="store_true", help="print the command, don't launch")
    ap.add_argument(
        "--detach",
        action="store_true",
        help="run detached on Modal: keeps training after this process/laptop disconnects",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume from the last checkpoint of exp_name (keeps optimizer state + step)",
    )
    args = ap.parse_args()

    cfg = _load_yaml(args.train_config) if args.train_config.exists() else {}
    repo_id = args.dataset_repo_id or cfg.get("dataset_repo_id") or "deskpartner/cans_single"
    # None -> default to the Modal data volume; "" -> omit root (download from Hub).
    dataset_root = f"/data/{repo_id}" if args.dataset_root is None else args.dataset_root
    exp = cfg.get("exp_name", "deskpartner_molmoact2")
    output_dir = f"/ckpts/{exp}"

    if args.resume:
        # Resume: lerobot restores model + optimizer + step counter + config from the
        # last checkpoint under output_dir and continues to --steps. Do NOT rebuild the
        # full arg list (the saved config is authoritative).
        steps = (cfg.get("train") or {}).get("steps", 8000)
        argv = [
            "accelerate", "launch", "--num_processes=1", "--mixed_precision=bf16",
            "-m", "lerobot.scripts.lerobot_train",
            f"--config_path={output_dir}/checkpoints/last/pretrained_model",
            "--resume=true",
            f"--steps={steps}",
        ]
    else:
        argv = build_train_argv(cfg, repo_id, dataset_root, output_dir)

    print(f"GPU:             {GPU}")
    print(f"mode:            {'RESUME from last checkpoint' if args.resume else 'fresh fine-tune'}")
    print(f"base_checkpoint: {cfg.get('base_checkpoint', BASE_CHECKPOINT)}")
    print(f"finetune mode:   {(cfg.get('finetune') or {}).get('mode', 'action_expert_only')}")
    print(f"dataset:         {repo_id}  (root {dataset_root})")
    print(f"output_dir:      {output_dir}")
    print("\nlerobot-train command:\n  " + " ".join(argv) + "\n")

    if args.dry_print or modal is None:
        if modal is None:
            print("(modal not installed — dry print only. pip install modal)")
        return 0

    if args.detach:
        # TRUE detach: spawn() submits the job and returns immediately, so there is
        # NO long-lived client connection to keep alive. app.run(detach=True) leaves
        # the app running after this process exits.
        # NOTE: an earlier version used the BLOCKING train.remote() here, which stays
        # tied to the live connection — when that connection dropped mid-run the call
        # was cancelled and the "detached" job died. spawn() fixes that.
        with app.run(detach=True):
            call = train.spawn(argv, exp, args.resume)
            app_id = getattr(app, "app_id", None)
        print(f"DETACHED: app_id={app_id} call_id={call.object_id}")
        print("Training runs independently of this client (laptop can sleep/close).")
        print("Watch the loss on wandb (project 'deskpartner-momo').")
        if app_id:
            print(f"Live logs: modal app logs {app_id}")
        return 0

    with modal.enable_output():  # attached run: stream the container's training logs
        with app.run():
            result = train.remote(argv, exp, args.resume)
    print("Modal finished:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
