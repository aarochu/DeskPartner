"""Convert a LeRobot MolmoAct2 checkpoint into the layout the arm rollout harness
(`p3_vlm_orchestrator/policy_rollout`, validated by
`rebot_operator_kit/rollout/checkpoint.py`) requires.

`lerobot-train` saves: config.json, model.safetensors, policy_preprocessor.json,
policy_postprocessor.json, train_config.json, ...
The harness's CheckpointBundle additionally requires, in the same directory:
  - preprocessor_config.json, postprocessor_config.json
  - rebot_training_profile.json  (a sidecar whose profile_snapshot / collection_contract
                                  each carry a matching SHA-256 digest)

This tool adds those files IN PLACE and never touches the weights. It mirrors the
digest algorithm in checkpoint.py exactly so the produced checkpoint passes
`CheckpointBundle.load()`.

    python -m p5_training.make_rollout_checkpoint \
        --checkpoint /path/to/pretrained_model \
        --task "Pick up the can and place it in the taped sorting zone"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

# Contract constants — must match rebot_operator_kit/rollout/checkpoint.py.
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)
IMAGE_ORDER = ["observation.images.front", "observation.images.side"]
COORDINATE_FRAME = "follower_degrees_after_direction_limits_and_step_cap"
CONTROL_MODE = "absolute joint pose"
PROFILE_FILENAME = "rebot_training_profile.json"


def canonical_digest(value: Any) -> str:
    """Identical to checkpoint.py::_canonical_digest so digests match on load."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ensure_processor_configs(ckpt: Path) -> list[str]:
    """Provide preprocessor_config.json / postprocessor_config.json.

    This lerobot build writes policy_preprocessor.json / policy_postprocessor.json;
    copy them into the harness-expected names when the expected files are absent.
    """
    notes: list[str] = []
    for expected, source in (
        ("preprocessor_config.json", "policy_preprocessor.json"),
        ("postprocessor_config.json", "policy_postprocessor.json"),
    ):
        if (ckpt / expected).is_file():
            notes.append(f"{expected}: already present")
            continue
        if (ckpt / source).is_file():
            shutil.copyfile(ckpt / source, ckpt / expected)
            notes.append(f"{expected}: copied from {source}")
        else:
            notes.append(f"{expected}: MISSING and no {source} to copy — inspect will fail")
    return notes


def build_profile(task: str, chunk_size: int, n_action_steps: int) -> dict[str, Any]:
    profile_snapshot = {
        "collection_defaults": {"task": task},
        "coordinate_contract": {
            "frame": COORDINATE_FRAME,
            "control_mode": CONTROL_MODE,
            "action_dimension": 7,
            "joints": [{"name": name} for name in JOINT_NAMES],
        },
        "training_defaults": {
            "action_dimension": 7,
            "chunk_size": chunk_size,
            "n_action_steps": n_action_steps,
            "image_order": IMAGE_ORDER,
        },
    }
    collection_contract = {"task": task}
    return {
        "profile_snapshot": profile_snapshot,
        "training_profile_digest": canonical_digest(profile_snapshot),
        "collection_contract": collection_contract,
        "collection_contract_digest": canonical_digest(collection_contract),
    }


def convert(ckpt: Path, task: str, chunk_size: int, n_action_steps: int) -> Path:
    ckpt = ckpt.expanduser().resolve()
    if not ckpt.is_dir():
        raise SystemExit(f"FAIL: checkpoint is not a directory: {ckpt}")
    if not (ckpt / "model.safetensors").is_file():
        raise SystemExit(f"FAIL: model.safetensors missing in {ckpt}")
    if not (ckpt / "config.json").is_file():
        raise SystemExit(f"FAIL: config.json missing in {ckpt}")
    if not task.strip():
        raise SystemExit("FAIL: --task must be nonempty")

    for note in ensure_processor_configs(ckpt):
        print(" ", note)

    profile = build_profile(task, chunk_size, n_action_steps)
    out = ckpt / PROFILE_FILENAME
    out.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"  task={task!r} chunk_size={chunk_size} n_action_steps={n_action_steps}")
    print(f"  training_profile_digest={profile['training_profile_digest']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True, help="checkpoint dir (pretrained_model)")
    ap.add_argument("--task", required=True, help="task string (must match the dataset's task)")
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--n-action-steps", type=int, default=10)
    args = ap.parse_args()
    convert(args.checkpoint, args.task, args.chunk_size, args.n_action_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
