"""Package a plain lerobot-train SmolVLA checkpoint into the format the Person 4
rollout harness (rebot_operator_kit/rollout/checkpoint.py) requires.

What it does (mechanical repackage — it invents nothing):
  1. Copies model.safetensors + config.json unchanged.
  2. Renames the processor configs to the names the harness loads:
       policy_preprocessor.json  -> preprocessor_config.json
       policy_postprocessor.json -> postprocessor_config.json
     (their schema already matches; state_file references are preserved) and
     copies the referenced *.safetensors normalization state files.
  3. Attaches the AUTHENTIC rebot_training_profile.json lifted from a dataset's
     meta/. It verifies the profile's stored digests are self-consistent with the
     harness's canonical digest before trusting them — it never fabricates a
     profile or silently "fixes" a digest.
  4. Validates the output against the repo's real CheckpointBundle.load.

Usage:
  python scripts/convert_smolvla_to_rollout_checkpoint.py \
    --checkpoint outputs/smolvla/rebot_combined_v1/checkpoints/last/pretrained_model \
    --profile-dataset ~/rebot-training/data/rebot-can-sort-stage1-v1-smoke-52ep \
    --out outputs/smolvla/rollout_checkpoint

Note: the profile you attach defines the checkpoint's declared task and rig
contract. Use a dataset whose profile is self-consistent and whose rig (7 joints,
front/side cameras, coordinate frame) matches how the weights were trained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_FILENAME = "rebot_training_profile.json"


def _canonical_digest(value) -> str:
    """Same canonical digest the rollout contract uses."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_checkpoint_validator():
    """Import the repo's real (dependency-free) checkpoint contract."""
    path = REPO_ROOT / "rebot_operator_kit" / "rollout" / "checkpoint.py"
    if not path.is_file():
        raise SystemExit(f"FAIL: cannot find rollout contract at {path}")
    spec = importlib.util.spec_from_file_location("_rollout_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def convert(checkpoint: Path, profile_dataset: Path, out: Path) -> None:
    checkpoint = checkpoint.expanduser().resolve()
    profile_dataset = profile_dataset.expanduser().resolve()
    out = out.expanduser().resolve()

    if not checkpoint.is_dir():
        raise SystemExit(f"FAIL: checkpoint dir missing: {checkpoint}")
    for required in ("model.safetensors", "config.json",
                     "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (checkpoint / required).is_file():
            raise SystemExit(f"FAIL: checkpoint is missing {required}")

    profile_path = profile_dataset / "meta" / PROFILE_FILENAME
    if not profile_path.is_file():
        raise SystemExit(f"FAIL: no authentic profile at {profile_path}")

    # --- sanity: the weights must actually be a 7-dim action policy ---
    cfg = _read_json(checkpoint / "config.json")
    action_shape = (cfg.get("output_features") or {}).get("action", {}).get("shape")
    if action_shape != [7]:
        raise SystemExit(
            f"FAIL: checkpoint action shape is {action_shape}, expected [7]. "
            "This profile/harness only accepts a 7-joint policy."
        )

    # --- verify the profile's digests are AUTHENTIC (self-consistent) ---
    sidecar = _read_json(profile_path)
    snapshot = sidecar.get("profile_snapshot")
    contract = sidecar.get("collection_contract")
    if not isinstance(snapshot, dict) or not isinstance(contract, dict):
        raise SystemExit("FAIL: profile sidecar missing profile_snapshot/collection_contract")
    if sidecar.get("training_profile_digest") != _canonical_digest(snapshot):
        raise SystemExit(
            "FAIL: profile_snapshot digest does not match the stored value. "
            "Refusing to alter it — supply a clean authentic profile."
        )
    if sidecar.get("collection_contract_digest") != _canonical_digest(contract):
        raise SystemExit(
            "FAIL: collection_contract digest does not match the stored value. "
            "Refusing to alter it — supply a clean authentic profile."
        )
    locked_task = (snapshot.get("collection_defaults") or {}).get("task")
    if contract.get("task") != locked_task:
        raise SystemExit(
            f"FAIL: this dataset's profile is internally inconsistent "
            f"(locked task {locked_task!r} != contract task {contract.get('task')!r}). "
            "It cannot back a valid single-task checkpoint."
        )

    # --- build the output checkpoint ---
    if out.exists():
        raise SystemExit(f"FAIL: output dir already exists, refusing to overwrite: {out}")
    out.mkdir(parents=True)

    shutil.copy2(checkpoint / "model.safetensors", out / "model.safetensors")
    shutil.copy2(checkpoint / "config.json", out / "config.json")

    for src_name, dst_name in (
        ("policy_preprocessor.json", "preprocessor_config.json"),
        ("policy_postprocessor.json", "postprocessor_config.json"),
    ):
        doc = _read_json(checkpoint / src_name)
        for step in doc.get("steps", []):
            state_file = step.get("state_file")
            if state_file:
                src_state = checkpoint / state_file
                if not src_state.is_file():
                    raise SystemExit(f"FAIL: processor state file missing: {src_state}")
                shutil.copy2(src_state, out / state_file)
        (out / dst_name).write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    shutil.copy2(profile_path, out / PROFILE_FILENAME)

    # --- validate against the repo's real contract ---
    validator = _load_checkpoint_validator()
    try:
        bundle = validator.CheckpointBundle.load(out)
    except validator.CheckpointError as exc:
        raise SystemExit(f"FAIL: converted checkpoint rejected by contract: {exc}") from exc

    print(f"PASS: wrote rollout-ready checkpoint -> {out}")
    print(f"  task:            {bundle.task}")
    print(f"  action_dim:      {bundle.action_dimension}")
    print(f"  chunk_size:      {bundle.chunk_size}")
    print(f"  n_action_steps:  {bundle.action_steps}")
    print(f"  image_order:     {list(bundle.image_order)}")
    print(f"  profile_digest:  {bundle.profile_digest}")
    print("  files:", sorted(p.name for p in out.iterdir()))
    print("NOTE: profile provenance = "
          f"{profile_dataset.name}; confirm this matches the intended task.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="lerobot-train SmolVLA checkpoint dir (has model.safetensors)")
    ap.add_argument("--profile-dataset", type=Path, required=True,
                    help="dataset root containing meta/rebot_training_profile.json")
    ap.add_argument("--out", type=Path, required=True,
                    help="output dir for the rollout-ready checkpoint")
    args = ap.parse_args()
    convert(args.checkpoint, args.profile_dataset, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
