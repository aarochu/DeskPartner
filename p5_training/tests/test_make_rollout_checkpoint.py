"""The converter's output must satisfy the harness validator (CheckpointBundle).

Pure stdlib — no lerobot, GPU, or hardware. Run directly:
    python p5_training/tests/test_make_rollout_checkpoint.py
or under pytest.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from p5_training.make_rollout_checkpoint import convert  # noqa: E402
from rebot_operator_kit.rollout.checkpoint import (  # noqa: E402
    CheckpointBundle,
    CheckpointError,
)

TASK = "Pick up the can and place it in the taped sorting zone"


def _fake_lerobot_checkpoint(d: Path) -> None:
    """Mimic what lerobot-train writes for a MolmoAct2 checkpoint (minimal)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\x00")
    (d / "policy_preprocessor.json").write_text('{"stub": "pre"}', encoding="utf-8")
    (d / "policy_postprocessor.json").write_text('{"stub": "post"}', encoding="utf-8")


def test_converted_checkpoint_loads() -> None:
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "pretrained_model"
        _fake_lerobot_checkpoint(ckpt)

        # Before conversion the harness must reject it (missing required files).
        try:
            CheckpointBundle.load(ckpt)
            raise AssertionError("validator should reject an un-converted checkpoint")
        except CheckpointError:
            pass

        convert(ckpt, task=TASK, chunk_size=10, n_action_steps=10)

        for f in ("preprocessor_config.json", "postprocessor_config.json", "rebot_training_profile.json"):
            assert (ckpt / f).is_file(), f"converter did not produce {f}"

        bundle = CheckpointBundle.load(ckpt)  # raises CheckpointError on any mismatch
        assert bundle.task == TASK
        assert bundle.action_dimension == 7
        assert bundle.chunk_size == 10
        assert bundle.action_steps == 10
        assert bundle.image_order == ("observation.images.front", "observation.images.side")
        return bundle


if __name__ == "__main__":
    b = test_converted_checkpoint_loads()
    print("PASS: converted checkpoint passes CheckpointBundle.load()")
    print("  task           :", b.task)
    print("  action_dim     :", b.action_dimension)
    print("  chunk_size     :", b.chunk_size)
    print("  action_steps   :", b.action_steps)
    print("  image_order    :", b.image_order)
    print("  profile_digest :", b.profile_digest)
