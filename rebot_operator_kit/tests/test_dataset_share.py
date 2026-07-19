from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dataset_share  # noqa: E402


class DatasetShareTest(unittest.TestCase):
    def make_dataset(self, parent: Path) -> Path:
        root = parent / "rebot-can-sort-stage1-v1-smoke"
        for directory in (
            "meta/episodes",
            "data/chunk-000",
            "videos/observation.images.front/chunk-000",
            "videos/observation.images.side/chunk-000",
        ):
            (root / directory).mkdir(parents=True)
        (root / "meta/info.json").write_text(
            json.dumps(
                {
                    "codebase_version": "v3.0",
                    "robot_type": "seeed_b601_dm_follower",
                    "total_episodes": 1,
                    "total_frames": 3,
                    "fps": 30,
                    "features": {"action": {"names": ["shoulder_pan.pos"]}},
                }
            )
        )
        for relative in (
            "meta/stats.json",
            "meta/tasks.parquet",
            "meta/episodes/chunk-000.parquet",
            "data/chunk-000/file-000.parquet",
            "videos/observation.images.front/chunk-000/file-000.mp4",
            "videos/observation.images.side/chunk-000/file-000.mp4",
        ):
            (root / relative).write_bytes(relative.encode())
        return root

    def test_prepare_and_verify_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_dataset(Path(temporary))
            dataset_share.prepare(
                argparse.Namespace(
                    dataset_root=root,
                    attempt_root=None,
                    destination="team/rebot-can-sort-stage1-v1",
                    visibility="private",
                )
            )
            dataset_share.verify(argparse.Namespace(dataset_root=root))
            self.assertTrue((root / "README.md").is_file())
            manifest = json.loads((root / "SHARE_MANIFEST.json").read_text())
            self.assertEqual(manifest["total_frames"], 3)

            (root / "meta/stats.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Shared file"):
                dataset_share.verify(argparse.Namespace(dataset_root=root))

    def test_guard_requires_idle_and_current_validation(self) -> None:
        args = argparse.Namespace(base_url="http://example.invalid", dataset="smoke")
        responses = [
            {"running": False, "finalizing": False},
            {
                "datasets": [
                    {
                        "name": "smoke",
                        "ready": True,
                        "validation_passed": True,
                        "episodes": 10,
                        "frames": 900,
                    }
                ]
            },
        ]
        with patch.object(dataset_share, "api_json", side_effect=responses):
            dataset_share.guard(args)

        with patch.object(
            dataset_share,
            "api_json",
            return_value={"running": True, "finalizing": False},
        ), self.assertRaisesRegex(ValueError, "blocked"):
            dataset_share.guard(args)


if __name__ == "__main__":
    unittest.main()
