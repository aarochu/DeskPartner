from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


KIT_ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = KIT_ROOT / "teleop_gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

import rerun_library as library  # noqa: E402


class RerunLibraryTest(unittest.TestCase):
    attempt_ids = (
        "20260719T010101.000001Z-1111111111",
        "20260719T010102.000002Z-2222222222",
        "20260719T010103.000003Z-3333333333",
    )

    def write_attempt(
        self,
        root: Path,
        attempt_id: str,
        *,
        disposition: str,
        included: bool,
        complete: bool = True,
        task: str = "Pick up one can",
    ) -> Path:
        directory = root / "can-recycling-test" / attempt_id
        directory.mkdir(parents=True)
        metadata = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "dataset": "can-recycling-test",
            "task": task,
            "started_at": attempt_id.split("-")[0],
            "finished_at": "2026-07-19T01:02:00.000Z",
            "disposition": disposition,
            "operator_disposition": disposition,
            "archive_complete": complete,
            "training_included": included,
            "training_episode_index": 4 if included else None,
            "samples": 90,
            "duration_s": 3.0,
            "dataset_fps": 30,
            "control_hz_requested": 240,
            "actual_control_hz": 201.5,
            "session_home": {
                "follower_positions_deg": {name: index + 0.25 for index, name in enumerate(library.JOINT_NAMES)}
            },
            "camera_freshness": {
                "front": {"status": "fresh", "sample_attempts": 90, "fresh_samples": 90, "max_age_ms_seen": 45.2},
                "side": {"status": "stale_frame_accepted_manual_mode", "sample_attempts": 90, "fresh_samples": 80, "stale_samples_accepted": 10, "max_age_ms_seen": 620.0},
            },
        }
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        if complete:
            (directory / "overhead.mp4").write_bytes(b"front")
            (directory / "wrist.mp4").write_bytes(b"side")
            (directory / "attempt.rrd").write_bytes(b"rrd")
        return directory

    def test_catalog_normalizes_bounty_tags_and_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_attempt(root, self.attempt_ids[0], disposition="kept", included=True)
            failed = self.write_attempt(root, self.attempt_ids[1], disposition="failed", included=False)
            metadata_path = failed / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["failure_label"] = "missed_grasp"
            metadata_path.write_text(json.dumps(metadata))
            self.write_attempt(root, self.attempt_ids[2], disposition="collector_error", included=False)

            payload = library.catalog_payload(root)
            by_id = {item["attempt_id"]: item for item in payload["attempts"]}
            self.assertEqual(by_id[self.attempt_ids[0]]["tag"], library.GOOD_EPISODE)
            self.assertEqual(by_id[self.attempt_ids[1]]["tag"], library.BAD_EPISODE)
            self.assertEqual(by_id[self.attempt_ids[2]]["tag"], library.NEEDS_REVIEW)
            good = library.attempt_detail_payload(root, self.attempt_ids[0])["attempt"]
            self.assertEqual(good["episode_id"], "episode_000004")
            self.assertEqual(good["joints"]["count"], 7)
            self.assertEqual(good["timing"]["dataset_fps"], 30.0)
            self.assertEqual(good["camera_freshness"]["side"]["stale_samples_accepted"], 10)
            self.assertTrue(good["artifacts"]["front"]["url"].startswith("/api/training/attempt/video"))
            self.assertFalse(payload["physical_replay_available"])

    def test_catalog_is_compact_paginated_and_detail_is_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for attempt_id in self.attempt_ids:
                self.write_attempt(root, attempt_id, disposition="kept", included=True)
            payload = library.catalog_payload(root, offset=1, limit=1)
            self.assertEqual(payload["returned"], 1)
            self.assertEqual(payload["visible"], 3)
            self.assertTrue(payload["has_more"])
            self.assertNotIn("camera_freshness", payload["attempts"][0])
            detail = library.attempt_detail_payload(root, payload["attempts"][0]["attempt_id"])
            self.assertIn("camera_freshness", detail["attempt"])

    def test_filters_dataset_tag_and_case_insensitive_search(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_attempt(root, self.attempt_ids[0], disposition="kept", included=True, task="Move RED can")
            self.write_attempt(root, self.attempt_ids[1], disposition="failed", included=False, task="Move blue can")
            payload = library.catalog_payload(root, tag=library.GOOD_EPISODE, search="red CAN")
            self.assertEqual([item["attempt_id"] for item in payload["attempts"]], [self.attempt_ids[0]])
            self.assertEqual(payload["visible"], 1)
            with self.assertRaisesRegex(ValueError, "Invalid Rerun tag"):
                library.catalog_payload(root, tag="Delete it")

    def test_zero_attempts_and_bad_metadata_are_resilient(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = library.catalog_payload(root)
            self.assertEqual(payload["attempts"], [])
            self.assertEqual(payload["total"], 0)

            bad = root / "can-recycling-test" / self.attempt_ids[0]
            bad.mkdir(parents=True)
            (bad / "metadata.json").write_text("{not json")
            self.assertEqual(library.catalog_payload(root)["attempts"], [])

    def test_incomplete_attempt_reports_preserved_raw_frames_without_media(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.write_attempt(
                root,
                self.attempt_ids[0],
                disposition="collector_error",
                included=False,
                complete=False,
            )
            (directory / "raw_frames" / "observation_images_front").mkdir(parents=True)
            row = library.catalog_payload(root)["attempts"][0]
            self.assertTrue(row["raw_frames_preserved"])
            self.assertFalse(row["archive_complete"])
            self.assertFalse(row["artifacts"]["front"]["available"])

    def test_media_resolution_rejects_missing_traversal_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "attempts"
            directory = self.write_attempt(root, self.attempt_ids[0], disposition="kept", included=True)
            self.assertEqual(
                library.resolve_media_path(root, self.attempt_ids[0], "front"),
                (directory / "overhead.mp4").resolve(),
            )
            with self.assertRaisesRegex(ValueError, "Invalid attempt ID"):
                library.resolve_media_path(root, "../../etc/passwd", "front")
            with self.assertRaisesRegex(ValueError, "Unknown attempt media"):
                library.resolve_media_path(root, self.attempt_ids[0], "trajectory")

            (directory / "wrist.mp4").unlink()
            with self.assertRaises(FileNotFoundError):
                library.resolve_media_path(root, self.attempt_ids[0], "side")

            outside = Path(temporary) / "outside.mp4"
            outside.write_bytes(b"outside")
            os.symlink(outside, directory / "wrist.mp4")
            with self.assertRaisesRegex(FileNotFoundError, "symlinks"):
                library.resolve_media_path(root, self.attempt_ids[0], "side")

    def test_catalog_ignores_symlinked_attempt_directory_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "attempts"
            outside = Path(temporary) / "outside"
            outside_attempt = self.write_attempt(outside, self.attempt_ids[0], disposition="kept", included=True)
            dataset = root / "can-recycling-test"
            dataset.mkdir(parents=True)
            os.symlink(outside_attempt, dataset / self.attempt_ids[0])
            self.assertEqual(library.catalog_payload(root)["attempts"], [])

            directory = dataset / self.attempt_ids[1]
            directory.mkdir()
            os.symlink(outside_attempt / "metadata.json", directory / "metadata.json")
            self.assertEqual(library.catalog_payload(root)["attempts"], [])

    def test_export_destination_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exports"
            root.mkdir()
            expected = root / "rebot-can-sort-reviewed-0001"
            self.assertEqual(
                library.safe_new_export_path(root, "rebot-can-sort-reviewed-0001"),
                expected.resolve(),
            )
            expected.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                library.safe_new_export_path(root, "rebot-can-sort-reviewed-0001")
            with self.assertRaisesRegex(ValueError, "Invalid export name"):
                library.safe_new_export_path(root, "../overwrite")

    def test_share_selection_is_preview_only_and_never_mutates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_attempt(root, self.attempt_ids[0], disposition="kept", included=True)
            self.write_attempt(root, self.attempt_ids[1], disposition="failed", included=False)
            before = sorted(path.relative_to(root) for path in root.rglob("*"))
            preview = library.share_selection_preview(
                root,
                {"attempt_ids": [self.attempt_ids[0], self.attempt_ids[1]]},
            )
            self.assertTrue(preview["preview_only"])
            self.assertEqual(preview["included"], 1)
            self.assertEqual(preview["excluded"], 1)
            self.assertEqual(before, sorted(path.relative_to(root) for path in root.rglob("*")))

    def test_ui_contract_is_read_only_and_has_no_physical_replay(self) -> None:
        html = (GUI_ROOT / "static" / "rerun.html").read_text(encoding="utf-8")
        javascript = (GUI_ROOT / "static" / "rerun.js").read_text(encoding="utf-8")
        module = (GUI_ROOT / "rerun_library.py").read_text(encoding="utf-8")
        server_source = (GUI_ROOT / "server.py").read_text(encoding="utf-8")
        for element_id in (
            "attempt-list",
            "attempt-detail",
            "front-video",
            "side-video",
            "open-rerun-button",
            "share-button",
        ):
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn("/api/rerun/catalog", javascript)
        self.assertIn("/api/rerun/attempt", javascript)
        self.assertIn("/api/rerun/share-preview", javascript)
        self.assertIn("/api/training/attempt/replay", javascript)
        self.assertIn("/rerun-flowchart.png", html)
        self.assertIn('parsed.path in {"/rerun", "/rerun.html"}', server_source)
        self.assertIn('parsed.path == "/api/rerun/catalog"', server_source)
        self.assertNotIn("p5_rerun_port.replay_episode", javascript)
        self.assertNotIn("from p5_rerun_port", module)


if __name__ == "__main__":
    unittest.main()
