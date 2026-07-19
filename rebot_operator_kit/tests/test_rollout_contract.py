from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rebot_operator_kit.rollout.checkpoint import CheckpointBundle, CheckpointError
from rebot_operator_kit.rollout.contracts import (
    PolicyAdapter,
    RobotAdapter,
    RolloutObservation,
)


FRONT_IMAGE_KEY = "observation.images.front"
SIDE_IMAGE_KEY = "observation.images.side"
TASK = "Pick up the crumpled paper ball and place it in the trash bin"
COORDINATE_FRAME = "follower_degrees_after_direction_limits_and_step_cap"
CONTROL_MODE = "absolute joint pose"
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
]


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RolloutObservationTest(unittest.TestCase):
    def test_observation_is_immutable(self) -> None:
        observation = RolloutObservation(
            front=np.zeros((2, 2, 3)),
            side=np.zeros((2, 2, 3)),
            state_deg=np.zeros(7),
            task=TASK,
            captured_monotonic_s=123.0,
        )

        with self.assertRaises(FrozenInstanceError):
            observation.task = "changed"

    def test_adapter_protocols_expose_the_rollout_boundary(self) -> None:
        self.assertTrue(hasattr(PolicyAdapter, "predict"))
        for method in ("connect", "disconnect", "observe", "send_action"):
            with self.subTest(method=method):
                self.assertTrue(hasattr(RobotAdapter, method))


class CheckpointBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.checkpoint = Path(temporary.name)
        self.profile = {
            "schema_version": 1,
            "profile_id": "rebot-test-profile",
            "profile_version": 1,
            "collection_defaults": {"task": TASK},
            "coordinate_contract": {
                "frame": COORDINATE_FRAME,
                "control_mode": CONTROL_MODE,
                "action_dimension": 7,
                "joints": [{"name": name} for name in JOINT_NAMES],
            },
            "training_defaults": {
                "action_dimension": 7,
                "chunk_size": 10,
                "n_action_steps": 10,
                "image_order": [FRONT_IMAGE_KEY, SIDE_IMAGE_KEY],
            },
        }
        self.collection_contract = {"task": TASK}
        self._write_required_files()
        self._write_profile_sidecar()

    def _write_json(self, name: str, payload: object) -> None:
        (self.checkpoint / name).write_text(json.dumps(payload))

    def _write_required_files(self) -> None:
        self._write_json("config.json", {"policy_type": "molmoact2"})
        self._write_json("preprocessor_config.json", {})
        self._write_json("postprocessor_config.json", {})
        (self.checkpoint / "model.safetensors").write_bytes(b"test weights")

    def _write_profile_sidecar(self, *, digest: str | None = None) -> None:
        self._write_json(
            "rebot_training_profile.json",
            {
                "schema_version": 1,
                "training_profile_id": self.profile["profile_id"],
                "training_profile_version": self.profile["profile_version"],
                "training_profile_digest": digest or canonical_digest(self.profile),
                "profile_snapshot": self.profile,
                "collection_contract": self.collection_contract,
                "collection_contract_digest": canonical_digest(self.collection_contract),
            },
        )

    def assert_rejected(self, expected_message: str) -> None:
        with self.assertRaisesRegex(CheckpointError, expected_message):
            CheckpointBundle.load(self.checkpoint)

    def test_accepts_complete_checkpoint_manifest(self) -> None:
        bundle = CheckpointBundle.load(self.checkpoint)

        self.assertEqual(bundle.path, self.checkpoint.resolve())
        self.assertEqual(bundle.task, TASK)
        self.assertEqual(bundle.action_dimension, 7)
        self.assertEqual(bundle.chunk_size, 10)
        self.assertEqual(bundle.action_steps, 10)
        self.assertEqual(bundle.image_order, (FRONT_IMAGE_KEY, SIDE_IMAGE_KEY))
        self.assertEqual(bundle.profile_digest, canonical_digest(self.profile))
        self.assertEqual(bundle.profile_snapshot, self.profile)

    def test_rejects_missing_profile(self) -> None:
        (self.checkpoint / "rebot_training_profile.json").unlink()

        self.assert_rejected("rebot_training_profile.json.*missing")

    def test_rejects_profile_digest_mismatch(self) -> None:
        self._write_profile_sidecar(digest="0" * 64)

        self.assert_rejected("digest.*match")

    def test_rejects_collection_contract_digest_mismatch(self) -> None:
        self.collection_contract["task"] = TASK + " safely"
        sidecar = json.loads(
            (self.checkpoint / "rebot_training_profile.json").read_text()
        )
        sidecar["collection_contract"] = self.collection_contract
        self._write_json("rebot_training_profile.json", sidecar)

        self.assert_rejected("collection contract digest.*match")

    def test_rejects_task_mismatch(self) -> None:
        self.collection_contract["task"] = "Perform a different task"
        self._write_profile_sidecar()

        self.assert_rejected("task.*match")

    def test_rejects_blank_task(self) -> None:
        self.profile["collection_defaults"]["task"] = "   "
        self.collection_contract["task"] = "   "
        self._write_profile_sidecar()

        self.assert_rejected("task.*nonempty")

    def test_rejects_action_dimension_other_than_seven(self) -> None:
        self.profile["training_defaults"]["action_dimension"] = 6
        self.profile["coordinate_contract"]["action_dimension"] = 6
        self._write_profile_sidecar()

        self.assert_rejected("action dimension.*7")

    def test_rejects_profile_action_dimension_disagreement(self) -> None:
        self.profile["coordinate_contract"]["action_dimension"] = 6
        self._write_profile_sidecar()

        self.assert_rejected("action dimension.*7")

    def test_rejects_reordered_joint_names(self) -> None:
        joints = self.profile["coordinate_contract"]["joints"]
        joints[0], joints[1] = joints[1], joints[0]
        self._write_profile_sidecar()

        self.assert_rejected("joint names.*order")

    def test_rejects_missing_coordinate_frame(self) -> None:
        self.profile["coordinate_contract"].pop("frame")
        self._write_profile_sidecar()

        self.assert_rejected("coordinate frame")

    def test_rejects_wrong_coordinate_frame(self) -> None:
        self.profile["coordinate_contract"]["frame"] = "leader_degrees"
        self._write_profile_sidecar()

        self.assert_rejected("coordinate frame")

    def test_rejects_missing_control_mode(self) -> None:
        self.profile["coordinate_contract"].pop("control_mode")
        self._write_profile_sidecar()

        self.assert_rejected("control mode")

    def test_rejects_wrong_control_mode(self) -> None:
        self.profile["coordinate_contract"]["control_mode"] = "velocity"
        self._write_profile_sidecar()

        self.assert_rejected("control mode")

    def test_rejects_nonpositive_chunk_size(self) -> None:
        self.profile["training_defaults"]["chunk_size"] = 0
        self._write_profile_sidecar()

        self.assert_rejected("chunk size.*positive")

    def test_rejects_nonpositive_action_steps(self) -> None:
        self.profile["training_defaults"]["n_action_steps"] = 0
        self._write_profile_sidecar()

        self.assert_rejected("action steps.*positive")

    def test_rejects_wrong_image_order(self) -> None:
        self.profile["training_defaults"]["image_order"] = [
            SIDE_IMAGE_KEY,
            FRONT_IMAGE_KEY,
        ]
        self._write_profile_sidecar()

        self.assert_rejected("image order.*front then side")

    def test_rejects_missing_processor_config(self) -> None:
        for filename in ("preprocessor_config.json", "postprocessor_config.json"):
            with self.subTest(filename=filename):
                path = self.checkpoint / filename
                path.unlink()
                self.assert_rejected(f"{filename}.*missing")
                self._write_json(filename, {})

    def test_rejects_missing_policy_config(self) -> None:
        (self.checkpoint / "config.json").unlink()

        self.assert_rejected("config.json.*missing")

    def test_rejects_missing_model_weights(self) -> None:
        (self.checkpoint / "model.safetensors").unlink()

        self.assert_rejected("model weights.*missing")

    def test_rejects_unrecognized_binary_as_model_weights(self) -> None:
        (self.checkpoint / "model.safetensors").unlink()
        (self.checkpoint / "optimizer.bin").write_bytes(b"not policy weights")

        self.assert_rejected("model weights.*missing")


if __name__ == "__main__":
    unittest.main()
