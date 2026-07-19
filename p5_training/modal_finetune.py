"""Modal MolmoAct2 LoRA / action-expert fine-tune.

Wraps Ai2/LeRobot train entrypoints — does NOT reinvent the training loop.
Confirm newt vs Ai2 scripts with organizers Friday night; set TRAIN_ENTRY accordingly.

  modal run p5_training/modal_finetune.py --mixture p5_training/configs/mixture_deskpartner.yaml

Requires: `pip install modal` + `modal token new`
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
DEFAULT_MIXTURE = ROOT / "p5_training" / "configs" / "mixture_deskpartner.yaml"
DEFAULT_TRAIN_CFG = ROOT / "p5_training" / "configs" / "molmoact2_single_arm.yaml"

# ---------------------------------------------------------------------------
# Config knobs (also overridable via mixture / train yaml)
# ---------------------------------------------------------------------------
GPU = os.environ.get("DESKPARTNER_MODAL_GPU", "A100-40GB")  # cheap override for wiring tests, e.g. T4; default A100 for real training
WALL_CLOCK_TIMEOUT_S = 6 * 60 * 60  # hard stop — do not eat all of Saturday night
SAVE_INTERVAL_STEPS = 500
MAX_STEPS = 5000  # small-data LoRA; raise only with more demos
BASE_CHECKPOINT = "allenai/MolmoAct2"


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text()) or {}


if modal is not None:
    app = modal.App("deskpartner-molmoact2")
    vol = modal.Volume.from_name("deskpartner-molmoact2-ckpts", create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch",
            "torchvision",
            "transformers",
            "accelerate",
            "datasets",
            "huggingface_hub",
            "pyyaml",
            "numpy",
            "wandb",
        )
        # NOTE: pin/install MolmoAct2 + LeRobot molmoact2-policy at venue once organizers confirm path
        .env({"HF_HOME": "/ckpts/hf"})
    )

    @app.function(
        image=image,
        gpu=GPU,
        timeout=WALL_CLOCK_TIMEOUT_S,
        volumes={"/ckpts": vol},
    )
    def train(mixture: dict, train_cfg: dict) -> dict:
        """Run LoRA fine-tune; save periodic checkpoints under /ckpts."""
        import os
        import subprocess

        start = time.time()
        exp = train_cfg.get("exp_name") or mixture.get("name") or "deskpartner_lora"
        out_dir = Path("/ckpts") / exp
        out_dir.mkdir(parents=True, exist_ok=True)

        base = train_cfg.get("base_checkpoint") or BASE_CHECKPOINT
        mode = (train_cfg.get("finetune") or mixture.get("finetune") or {}).get("mode", "lora")
        if mode == "full" or (train_cfg.get("finetune") or {}).get("full_finetune"):
            raise RuntimeError("FAIL: full fine-tune is forbidden for this hackathon job")

        # Prefer organizer-provided entrypoint; default documents Ai2 script shape.
        entry = train_cfg.get("train_entrypoint") or "ai2_train_lerobot"
        log_path = out_dir / "train_log.jsonl"

        def log(event: dict) -> None:
            event = {**event, "elapsed_s": round(time.time() - start, 1)}
            with log_path.open("a") as f:
                f.write(json.dumps(event) + "\n")
            print(event, flush=True)

        log({"event": "start", "gpu": GPU, "base": base, "mode": mode, "entry": entry})

        # Placeholder command — swap for real Ai2/LeRobot once confirmed at venue.
        # This keeps Modal wiring / timeout / checkpoint dir real even before the exact CLI lands.
        train_cmd = train_cfg.get("train_cmd")
        if not train_cmd:
            mixture_name = mixture.get("name", "deskpartner_rebot_b601_single")
            train_cmd = [
                "python",
                "-c",
                (
                    "import json,time,pathlib;"
                    f"out=pathlib.Path({str(out_dir)!r});"
                    "out.mkdir(parents=True,exist_ok=True);"
                    f"steps={int(train_cfg.get('max_steps', MAX_STEPS))};"
                    f"save_every={int(train_cfg.get('save_interval_steps', SAVE_INTERVAL_STEPS))};"
                    "log=out/'train_log.jsonl';"
                    "\nfor i in range(1,steps+1):\n"
                    "  loss=1.0/i;\n"
                    "  open(log,'a').write(json.dumps({'step':i,'loss':loss})+'\\n');\n"
                    "  if i%save_every==0 or i==steps:\n"
                    "    (out/f'checkpoint-{i}').mkdir(exist_ok=True);\n"
                    "    (out/f'checkpoint-{i}'/'READY').write_text('ok');\n"
                    "    print('saved',i,flush=True);\n"
                    "  time.sleep(0.01)\n"
                    f"print('DONE mixture={mixture_name} base={base}')"
                ),
            ]
            log(
                {
                    "event": "warn_placeholder_trainer",
                    "msg": "Using placeholder trainer. Set train_cmd or install Ai2 script at venue.",
                }
            )

        # Wall clock guard around subprocess
        remaining = WALL_CLOCK_TIMEOUT_S - 120
        try:
            subprocess.run(train_cmd, check=True, timeout=max(60, remaining))
        except subprocess.TimeoutExpired as exc:
            log({"event": "TIMEOUT", "error": str(exc)})
            raise
        except subprocess.CalledProcessError as exc:
            log({"event": "train_failed", "code": exc.returncode})
            raise

        vol.commit()
        log({"event": "done", "out_dir": str(out_dir)})
        return {"out_dir": str(out_dir), "exp": exp}

else:
    app = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixture", type=Path, default=DEFAULT_MIXTURE)
    ap.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CFG)
    ap.add_argument("--gpu", default=None, help=f"Override GPU (default {GPU})")
    ap.add_argument("--dry-print", action="store_true")
    args = ap.parse_args()

    if not args.mixture.exists():
        if args.dry_print:
            print(f"WARN: mixture missing ({args.mixture}) — dry-print only")
            mixture = {"name": "MISSING", "image_keys": [], "finetune": {"mode": "lora"}}
        else:
            raise SystemExit(
                f"FAIL: mixture missing: {args.mixture} - run build_dataset_mixture first"
            )
    else:
        mixture = _load_yaml(args.mixture)
    train_cfg = _load_yaml(args.train_config) if args.train_config.exists() else {}
    if args.gpu:
        train_cfg["gpu"] = args.gpu
        print(f"GPU override requested: {args.gpu} (edit GPU constant / Modal decorator if needed)")

    print("mixture:", mixture.get("name"))
    print("image_keys:", mixture.get("image_keys"))
    print("finetune:", mixture.get("finetune"))
    print(f"wall_clock_timeout_s={WALL_CLOCK_TIMEOUT_S} save_interval={SAVE_INTERVAL_STEPS}")

    if args.dry_print or modal is None:
        if modal is None:
            print("WARN: modal not installed - dry print only. pip install modal")
        print("Would call Modal train() with mixture + train_cfg")
        return 0

    with app.run():
        result = train.remote(mixture, train_cfg)
    print("Modal finished:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
