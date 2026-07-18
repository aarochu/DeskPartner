"""Build a MolmoAct2 fine-tune mixture config from a verified LeRobot dataset.

Reads camera keys + feature names from dataset meta (not hardcoded).
Fails loudly if channels look bimanual or cameras mismatch recording.yaml.

Wraps the Ai2 pattern: build_single_lerobot_mixture / data_mixtures.py registration.
Exact setup_type / tag strings still need organizer confirm — see mixture YAML notes.

  python -m p5_training.build_dataset_mixture --dataset-root ~/.cache/huggingface/lerobot/deskpartner/crumpled_paper_molmoact2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_info(root: Path) -> dict[str, Any]:
    for rel in ("meta/info.json", "info.json"):
        p = root / rel
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"No info.json under {root}")


def _feature_keys(info: dict[str, Any]) -> list[str]:
    feats = info.get("features") or {}
    if isinstance(feats, dict):
        return list(feats.keys())
    return []


def _detect_camera_keys(feature_keys: list[str], info: dict[str, Any]) -> list[str]:
    cams = []
    for k in feature_keys:
        # observation.images.front / front / images.front
        parts = k.replace("observation.images.", "").replace("images.", "").split(".")
        name = parts[0]
        if any(x in k.lower() for x in ("image", "rgb", "cam")):
            if name and name not in cams and name not in ("observation", "images"):
                cams.append(name)
    # fallback: scan json
    blob = json.dumps(info)
    for guess in ("front", "side", "overhead", "wrist"):
        if guess in blob and guess not in cams:
            cams.append(guess)
    return cams


def _detect_state_action(feature_keys: list[str]) -> tuple[list[str], list[str]]:
    state, action = [], []
    for k in feature_keys:
        kl = k.lower()
        if "action" in kl:
            action.append(k)
        if "state" in kl or "qpos" in kl or "joint" in kl:
            if "action" not in kl:
                state.append(k)
    return state or ["observation.state"], action or ["action"]


def validate_single_arm(feature_keys: list[str], info: dict[str, Any]) -> None:
    blob = " ".join(feature_keys).lower() + " " + json.dumps(info).lower()
    bad = [
        s
        for s in (
            "left_arm",
            "right_arm",
            "bimanual",
            "bi_",
            "arm_left",
            "arm_right",
        )
        if s in blob
    ]
    if bad:
        raise SystemExit(
            f"FAIL: dataset looks bimanual / dual-arm ({bad}). "
            "Refuse to build mixture — fix P4 recording first."
        )


def build_mixture(
    dataset_root: Path,
    out_path: Path,
    mixture_name: str = "deskpartner_rebot_b601_single",
    setup_descriptor: str = "single rebot b601 arm on a desk",
) -> dict[str, Any]:
    if not dataset_root.exists():
        raise SystemExit(f"FAIL: dataset root missing: {dataset_root}")

    try:
        info = _load_info(dataset_root)
    except FileNotFoundError as exc:
        raise SystemExit(f"FAIL: {exc}") from exc
    feats = _feature_keys(info)
    validate_single_arm(feats, info)

    cam_keys = _detect_camera_keys(feats, info)
    if not cam_keys:
        raise SystemExit("FAIL: could not detect camera keys from dataset meta")

    # Cross-check recording.yaml expectations
    rec_path = ROOT / "config" / "recording.yaml"
    if rec_path.exists():
        rec = yaml.safe_load(rec_path.read_text())
        expect = (rec.get("expect") or {}).get("camera_keys") or []
        for ck in expect:
            if ck not in cam_keys and ck not in json.dumps(info):
                raise SystemExit(
                    f"FAIL: recording.yaml expects camera '{ck}' but dataset meta lacks it. "
                    f"detected={cam_keys}"
                )

    state_keys, action_keys = _detect_state_action(feats)

    # Ai2-style mixture descriptor (to paste into data_mixtures.py or pass to train)
    # CONFIRMED PATTERN from allenai/molmoact2 experiments/README:
    #   build_single_lerobot_mixture(name=..., data_urls=..., image_keys=..., state_keys=..., ...)
    # setup_type / action convention: CONFIRM WITH ORGANIZERS (pose vs joint).
    mixture = {
        "name": mixture_name,
        "backend": "molmoact2_build_single_lerobot_mixture",
        "setup_descriptor": setup_descriptor,
        "setup_type_TODO_confirm_with_organizers": "joint_pose_or_absolute_joint",
        "embodiment": "single_arm_rebot_b601_dm",
        "base_checkpoint_default": "allenai/MolmoAct2",
        "note": (
            "No zero-shot SO-100/DROID transfer assumed. "
            "Prefer allenai/MolmoAct2 base unless organizers give a better start."
        ),
        "dataset": {
            "root": str(dataset_root),
            "repo_id": info.get("repo_id") or dataset_root.name,
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
        },
        "image_keys": cam_keys,
        "state_keys": state_keys,
        "action_keys": action_keys,
        "feature_keys_all": feats,
        "finetune": {
            "mode": "lora",  # or action_expert_only
            "lora_enable": True,
            "ft_vlm": True,
            "ft_action_expert": True,
            "full_finetune": False,
        },
        "ai2_train_hint": {
            "script": "launch_scripts/train_lerobot.py",
            "example": (
                "torchrun launch_scripts/train_lerobot.py allenai/MolmoAct2 "
                f"{mixture_name} --lora_enable=true --ft_action_expert=true ..."
            ),
            "register_in": "launch_scripts/data_mixtures.py -> MOLMOACT2_LEROBOT_MIXTURES",
        },
    }

    # Validation: action dim must exist
    if not action_keys:
        raise SystemExit("FAIL: no action keys found in dataset features")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(mixture, sort_keys=False))
    print(f"PASS: wrote mixture -> {out_path}")
    print(f"  image_keys={cam_keys}")
    print(f"  state_keys={state_keys}")
    print(f"  action_keys={action_keys}")
    print(
        "NOTE: confirm setup_type / action convention with organizers before kicking Modal."
    )
    return mixture


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "p5_training" / "configs" / "mixture_deskpartner.yaml",
    )
    ap.add_argument("--name", default="deskpartner_rebot_b601_single")
    ap.add_argument(
        "--setup-descriptor",
        default="single rebot b601 arm on a desk",
    )
    args = ap.parse_args()
    build_mixture(args.dataset_root, args.out, args.name, args.setup_descriptor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
