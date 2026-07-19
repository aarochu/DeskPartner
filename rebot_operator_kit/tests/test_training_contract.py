from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from datasets import config as datasets_config


KIT_ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = KIT_ROOT / "teleop_gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

import controlled_record  # noqa: E402
import attempt_archive  # noqa: E402
import server  # noqa: E402
import training_workspace as workspace  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import DEFAULT_FEATURES  # noqa: E402


EXPECTED = {
    "control_hz": 240,
    "motor_velocity": 2000.0,
    "max_step": 8.4,
    "gripper_force": 0.05,
}


def command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class TrainingDefaultsTest(unittest.TestCase):
    def test_profile_api_form_and_command_share_exact_defaults(self) -> None:
        status = workspace.require_training_profile()
        self.assertEqual(status["profile_id"], "rebot-b601-dm-follower1-native7d-v4")
        self.assertEqual(status["profile_version"], 4)
        for key, expected in EXPECTED.items():
            self.assertEqual(status["defaults"][key], expected)

        config = workspace.validate_collection_config({})
        command = workspace.build_record_command(
            config,
            {"follower": "/dev/test-follower", "leader": "/dev/test-leader"},
        )
        self.assertEqual(command_value(command, "--control-hz"), "240")
        self.assertEqual(command_value(command, "--motor-velocity"), "2000")
        self.assertEqual(command_value(command, "--max-step"), "8.4")
        self.assertEqual(command_value(command, "--gripper-force"), "0.05")
        self.assertEqual(command_value(command, "--attempt-root"), str(workspace.ATTEMPT_ROOT))
        self.assertTrue(command_value(command, "--control-file").endswith("record-control.json"))

        training_html = (GUI_ROOT / "static" / "training.html").read_text()
        self.assertIn('id="control-hz-input" type="number" value="240"', training_html)
        self.assertIn('id="velocity-input" type="number" value="2000"', training_html)
        self.assertIn('id="max-step-input" type="number" value="8.4"', training_html)

    def test_stage_one_can_smoke_defaults_and_finish_notice_are_explicit(self) -> None:
        status = workspace.require_training_profile()
        defaults = status["defaults"]
        self.assertEqual(
            defaults["task"],
            "Pick up one can and place it in the taped sorting zone",
        )
        self.assertEqual(defaults["dataset"], "rebot-can-sort-stage1-v1-smoke")
        self.assertEqual(defaults["episodes"], 10)

        training_html = (GUI_ROOT / "static" / "training.html").read_text()
        self.assertIn(
            'value="Pick up one can and place it in the taped sorting zone"',
            training_html,
        )
        self.assertIn('value="rebot-can-sort-stage1-v1-smoke"', training_html)
        self.assertIn('id="episodes-input" type="number" value="10"', training_html)

        training_js = (GUI_ROOT / "static" / "training.js").read_text()
        self.assertIn("Save accepted. Writing this episode to Rerun and LeRobot now.", training_js)
        self.assertIn("Do not press Stop; wait until the next attempt is ready.", training_js)

    def test_manual_gui_default_matches_collection_profile(self) -> None:
        preset = server.PRESETS["hand_tracking"]
        self.assertEqual(
            {
                "control_hz": preset["hz"],
                "motor_velocity": preset["velocity"],
                "max_step": preset["max_step"],
                "gripper_force": preset["gripper_force"],
            },
            EXPECTED,
        )
        validated = server.validate_config({})
        self.assertEqual(validated["hz"], 240)
        self.assertEqual(validated["max_step"], 8.4)
        self.assertEqual(validated["gripper_force"], 0.05)
        self.assertEqual(set(validated["velocities"].values()), {2000.0})
        self.assertEqual(validated["tracking_cap"], 2016.0)
        app_js = (GUI_ROOT / "static" / "app.js").read_text()
        self.assertIn('const STORAGE_KEY = "rebot.teleop.draft.v3";', app_js)
        self.assertIn('const DEFAULT_PRESET = "hand_tracking";', app_js)

    def test_follower_config_receives_2000_for_all_seven_joints(self) -> None:
        status = workspace.require_training_profile()
        calibration = status["calibration_files"]
        args = SimpleNamespace(
            follower_port="/dev/test-follower",
            leader_port="/dev/test-leader",
            front_camera=0,
            front_width=640,
            front_height=480,
            side_camera=1,
            side_width=1280,
            side_height=720,
            dataset_fps=30,
            max_step=8.4,
            motor_velocity=2000.0,
            gripper_force=0.05,
            follower_calibration=Path(calibration["follower"]["path"]),
            leader_calibration=Path(calibration["leader"]["path"]),
        )

        class FakeFollower:
            def __init__(self, config):
                self.config = config
                self.calibration_fpath = args.follower_calibration

        class FakeLeader:
            def __init__(self, config):
                self.config = config
                self.calibration_fpath = args.leader_calibration

        with (
            patch.object(controlled_record, "SeeedB601DMFollower", FakeFollower),
            patch.object(controlled_record, "RebotArm102Leader", FakeLeader),
        ):
            follower, _leader = controlled_record.make_hardware(args, status["profile"])
        self.assertEqual(follower.config.pos_vel_velocity, [2000.0] * 7)
        self.assertEqual(follower.config.max_relative_target, 8.4)


class CollectorSchemaTest(unittest.TestCase):
    def test_lerobot_owns_timestamp_and_frame_index(self) -> None:
        fps = 30
        names = [f"j{index}" for index in range(7)]
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": names,
            },
            "action": {"dtype": "float32", "shape": (7,), "names": names},
        }
        observation = {name: 0.0 for name in names}
        action = {name: 1.0 for name in names}

        with tempfile.TemporaryDirectory() as temporary:
            dataset = LeRobotDataset.create(
                "local/schema-test",
                fps,
                features,
                root=Path(temporary) / "dataset",
                use_videos=False,
            )
            frame = controlled_record.build_training_frame(
                dataset.features, observation, action, "schema test"
            )
            self.assertTrue(set(frame).isdisjoint(DEFAULT_FEATURES))
            dataset.add_frame(dict(frame))
            dataset.add_frame(dict(frame))
            self.assertEqual(dataset.episode_buffer["size"], 2)
            self.assertEqual(dataset.episode_buffer["frame_index"], [0, 1])
            np.testing.assert_allclose(
                dataset.episode_buffer["timestamp"], [0.0, 1.0 / fps]
            )
            with self.assertRaisesRegex(ValueError, "Extra features.*timestamp"):
                dataset.add_frame({**frame, "timestamp": np.float32(2.0 / fps)})


class SessionHomeReturnTest(unittest.TestCase):
    FEATURES = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_yaw.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    DIRECTIONS = {
        "shoulder_pan": -1.0,
        "shoulder_lift": 1.0,
        "elbow_flex": 1.0,
        "wrist_flex": 1.0,
        "wrist_yaw": 1.0,
        "wrist_roll": 1.0,
        "gripper": -6.0,
    }

    class FakeRobot:
        def __init__(self, features: list[str], directions: dict[str, float]) -> None:
            self.action_features = {name: float for name in features}
            self.config = SimpleNamespace(joint_directions=directions)
            self.current = {
                name: float((index + 1) * 3) for index, name in enumerate(features)
            }
            self.send_count = 0

        def get_observation(self) -> dict[str, float]:
            return dict(self.current)

        def send_action(self, action: dict[str, float]) -> dict[str, float]:
            self.send_count += 1
            for feature, value in action.items():
                motor = feature.removesuffix(".pos")
                self.current[feature] = float(value) * self.config.joint_directions[motor]
            return dict(self.current)

    class FakeLeader:
        def __init__(self, positions: dict[str, float]) -> None:
            self.positions = positions

        def get_action(self) -> dict[str, float]:
            return dict(self.positions)

    def test_capture_inverts_driver_directions_and_reset_returns_home(self) -> None:
        robot = self.FakeRobot(self.FEATURES, self.DIRECTIONS)
        expected_home = dict(robot.current)
        leader_home = {
            feature: expected_home[feature] / self.DIRECTIONS[feature.removesuffix(".pos")]
            for feature in self.FEATURES
        }
        leader = self.FakeLeader(leader_home)
        home = controlled_record.capture_session_home(robot, leader)
        self.assertEqual(home["follower_positions_deg"], expected_home)
        self.assertEqual(home["leader_positions_deg"], leader_home)
        self.assertEqual(home["follower_input_deg"]["gripper.pos"], -3.5)

        robot.current = {name: value + 4.0 for name, value in expected_home.items()}
        args = SimpleNamespace(control_hz=100, max_step=8.4, reset_time_s=0.0)
        with patch.object(controlled_record, "HOME_SETTLE_TIME_S", 0.01):
            result = controlled_record.automatic_reset_to_session_home(
                args=args,
                robot=robot,
                leader=leader,
                events={"stop": False},
                session_home=home,
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result["follower_aligned"])
        self.assertTrue(result["leader_aligned"])
        self.assertGreater(robot.send_count, 0)
        for feature, target in expected_home.items():
            self.assertAlmostEqual(robot.current[feature], target)

    def test_stop_interrupts_home_return_without_sending_motion(self) -> None:
        robot = self.FakeRobot(self.FEATURES, self.DIRECTIONS)
        leader = self.FakeLeader({name: 0.0 for name in self.FEATURES})
        home = controlled_record.capture_session_home(robot, leader)
        result = controlled_record.automatic_reset_to_session_home(
            args=SimpleNamespace(control_hz=100, max_step=8.4, reset_time_s=0.0),
            robot=robot,
            leader=leader,
            events={"stop": True},
            session_home=home,
        )
        self.assertIsNone(result)
        self.assertEqual(robot.send_count, 0)

    def test_completed_home_return_is_persisted_in_attempt_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            attempt_archive.atomic_write_json(
                directory / "metadata.json",
                {"attempt_id": "attempt-test", "disposition": "kept"},
            )
            result = {
                "completed_at": "2026-07-18T00:00:00Z",
                "follower_aligned": True,
                "leader_aligned": True,
            }
            controlled_record.mark_attempt_home_return(directory, result)
            stored = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(stored["home_return"], result)

    def test_stop_interrupted_serial_read_is_clean_shutdown(self) -> None:
        events = {"finish": False, "rerecord": False, "stop": False}

        class Robot:
            def get_observation(self) -> dict[str, float]:
                return {}

            def send_action(self, _action: dict[str, float]) -> dict[str, float]:
                raise AssertionError("No follower command should follow a stopped leader read")

        class InterruptedLeader:
            def get_action(self) -> dict[str, float]:
                events["stop"] = True
                events["finish"] = True
                raise RuntimeError("Leader position read failed; teleoperation stopped")

        samples, _actual_hz = controlled_record.control_segment(
            args=SimpleNamespace(control_hz=100, dataset_fps=30),
            robot=Robot(),
            leader=InterruptedLeader(),
            dataset=SimpleNamespace(),
            events=events,
            duration_s=1.0,
            record=False,
            task="shutdown test",
        )
        self.assertEqual(samples, 0)

    def test_serial_read_failure_without_stop_remains_fatal(self) -> None:
        class Robot:
            def get_observation(self) -> dict[str, float]:
                return {}

        class BrokenLeader:
            def get_action(self) -> dict[str, float]:
                raise RuntimeError("serial read failed")

        with self.assertRaisesRegex(RuntimeError, "serial read failed"):
            controlled_record.control_segment(
                args=SimpleNamespace(control_hz=100, dataset_fps=30),
                robot=Robot(),
                leader=BrokenLeader(),
                dataset=SimpleNamespace(),
                events={"finish": False, "rerecord": False, "stop": False},
                duration_s=1.0,
                record=False,
                task="failure test",
            )


class ZeroFrameQuarantineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.data_root = root / "data"
        self.run_root = root / "runs"
        self.data_root.mkdir()
        self.run_root.mkdir()
        self.globals = patch.multiple(
            workspace,
            DATA_ROOT=self.data_root,
            RUN_ROOT=self.run_root,
        )
        self.globals.start()

    def tearDown(self) -> None:
        self.globals.stop()
        self.temporary.cleanup()

    def write_manifest(self, name: str, state: str, exit_code: int | None) -> Path:
        path = self.run_root / f"{name}--20260718-120000.json"
        path.write_text(
            json.dumps({"lifecycle": {"state": state, "exit_code": exit_code}})
        )
        return path

    def write_info(self, name: str, episodes: int, frames: int) -> Path:
        root = self.data_root / name
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(
            json.dumps({"total_episodes": episodes, "total_frames": frames})
        )
        return root

    def test_metadata_zero_frame_attempt_is_recoverably_moved(self) -> None:
        name = "retry-zero"
        root = self.write_info(name, 0, 0)
        manifest = self.write_manifest(name, "FAILED", 1)
        destination = workspace._quarantine_zero_frame_attempt(
            name, manifest_paths=[manifest], reason="test failure"
        )
        self.assertIsNotNone(destination)
        self.assertFalse(root.exists())
        self.assertFalse(manifest.exists())
        self.assertTrue((destination / "data" / name / "meta" / "info.json").is_file())
        self.assertTrue((destination / "manifests" / manifest.name).is_file())
        self.assertFalse((self.data_root / name).exists())

    def test_failed_before_root_is_removed_from_active_manifest_namespace(self) -> None:
        name = "retry-no-root"
        manifest = self.write_manifest(name, "FAILED_TO_START", None)
        destination = workspace._quarantine_zero_frame_attempt(
            name, reason="pre-start cleanup"
        )
        self.assertIsNotNone(destination)
        self.assertFalse(manifest.exists())
        self.assertTrue((destination / "manifests" / manifest.name).is_file())

    def test_clean_zero_episode_stop_is_also_retryable(self) -> None:
        name = "retry-clean-stop"
        self.write_info(name, 0, 0)
        manifest = self.write_manifest(name, "COMPLETE", 0)
        destination = workspace._quarantine_zero_frame_attempt(
            name, manifest_paths=[manifest], reason="clean stop before first episode"
        )
        self.assertIsNotNone(destination)
        self.assertFalse(manifest.exists())

    def test_nonzero_dataset_is_never_moved(self) -> None:
        name = "keep-one-episode"
        root = self.write_info(name, 1, 30)
        manifest = self.write_manifest(name, "FAILED", 1)
        destination = workspace._quarantine_zero_frame_attempt(
            name, manifest_paths=[manifest], reason="must preserve saved data"
        )
        self.assertIsNone(destination)
        self.assertTrue(root.is_dir())
        self.assertTrue(manifest.is_file())


class AttemptArchiveTest(unittest.TestCase):
    class FakeDataset:
        def __init__(self, source_root: Path) -> None:
            self.source_root = source_root
            self.fps = 10
            self.episode_buffer = {"episode_index": 0, "size": 3}
            self.meta = SimpleNamespace(
                camera_keys=["observation.images.front", "observation.images.side"]
            )

        def _wait_image_writer(self) -> None:
            return None

        def _get_image_file_dir(self, _episode_index: int, camera_key: str) -> Path:
            return self.source_root / camera_key

        def clear_episode_buffer(self, delete_images: bool = True) -> None:
            self.episode_buffer = {"episode_index": 0, "size": 0}

    def write_source_frames(self, source_root: Path) -> None:
        for camera_key, channel in (
            ("observation.images.front", 0),
            ("observation.images.side", 1),
        ):
            directory = source_root / camera_key
            directory.mkdir(parents=True)
            for index in range(3):
                frame = np.zeros((24, 32, 3), dtype=np.uint8)
                frame[:, :, channel] = 80 + index * 40
                Image.fromarray(frame).save(directory / f"frame-{index:06d}.png")

    def make_recording(self, directory: Path, seed: int):
        recording = controlled_record.rr.RecordingStream(
            "rebot_attempt_test",
            recording_id=controlled_record.uuid4(),
        )
        recording.set_sinks(
            controlled_record.rr.FileSink(str(directory / "attempt.partial.rrd"))
        )
        image = np.full((24, 32, 3), seed, dtype=np.uint8)
        controlled_record.log_attempt_sample(
            recording,
            {"front": image, "side": image, "joint": 1.0},
            {"joint": 2.0},
            0,
            10,
        )
        return recording

    def test_success_failure_success_create_three_replayable_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_root = root / "attempts"
            source_root = root / "frames"
            self.write_source_frames(source_root)
            dataset = self.FakeDataset(source_root)
            outcomes = ["kept", "failed", "kept"]
            ids: list[str] = []
            saved_episode_index = 0
            for index, disposition in enumerate(outcomes):
                attempt_id = attempt_archive.new_attempt_id()
                ids.append(attempt_id)
                directory = attempt_archive.attempt_path(
                    archive_root, "archive-contract-test", attempt_id
                )
                directory.mkdir(parents=True)
                metadata = {
                    "schema_version": 1,
                    "attempt_id": attempt_id,
                    "dataset": "archive-contract-test",
                    "started_at": attempt_archive.utc_now(),
                    "disposition": "recording",
                    "training_included": False,
                }
                attempt_archive.atomic_write_json(directory / "metadata.json", metadata)
                recording = self.make_recording(directory, 40 + index * 40)
                controlled_record.finalize_attempt_archive(
                    dataset=dataset,
                    directory=directory,
                    metadata=metadata,
                    recording=recording,
                    disposition=disposition,
                    samples=3,
                    duration_s=0.3,
                    actual_hz=220.0,
                    failure_label="failed_to_perform_task" if disposition == "failed" else "",
                )
                if disposition == "kept":
                    controlled_record.mark_attempt_in_training(directory, saved_episode_index)
                    saved_episode_index += 1

                for filename in ("overhead.mp4", "wrist.mp4", "attempt.rrd"):
                    self.assertGreater((directory / filename).stat().st_size, 0)
                verified = subprocess.run(
                    [str(workspace.RERUN_NATIVE_BIN), "rrd", "verify", str(directory / "attempt.rrd")],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(verified.returncode, 0, verified.stderr or verified.stdout)

            inventory = attempt_archive.attempt_inventory(archive_root)
            self.assertEqual(len(inventory), 3)
            self.assertEqual(sum(bool(item["training_included"]) for item in inventory), 2)
            self.assertEqual(
                sorted(
                    item["training_episode_index"]
                    for item in inventory
                    if item["training_included"]
                ),
                [0, 1],
            )
            failed = next(item for item in inventory if item["disposition"] == "failed")
            self.assertEqual(failed["failure_label"], "failed_to_perform_task")
            self.assertNotIn(
                "failure_label", next(item for item in inventory if item["disposition"] == "kept")
            )
            attempt_archive.update_failure_label(
                archive_root, failed["attempt_id"], "test_or_setup", "camera setup trial"
            )
            updated = attempt_archive.find_attempt(archive_root, failed["attempt_id"])[1]
            self.assertEqual(updated["failure_label"], "test_or_setup")
            with self.assertRaisesRegex(ValueError, "Only failed"):
                attempt_archive.update_failure_label(
                    archive_root, ids[0], "failed_to_perform_task"
                )

    def test_failure_label_is_required_and_other_needs_note(self) -> None:
        with self.assertRaisesRegex(ValueError, "Choose a failure reason"):
            attempt_archive.validate_failure_label("")
        with self.assertRaisesRegex(ValueError, "Add a note"):
            attempt_archive.validate_failure_label("other", "")
        self.assertEqual(
            attempt_archive.validate_failure_label("other", "lighting trial"),
            ("other", "lighting trial"),
        )

    def test_archive_failure_moves_raw_frames_before_buffer_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "frames"
            self.write_source_frames(source_root)
            dataset = self.FakeDataset(source_root)
            attempt_id = attempt_archive.new_attempt_id()
            directory = attempt_archive.attempt_path(root / "attempts", "raw-recovery-test", attempt_id)
            directory.mkdir(parents=True)
            metadata = {
                "attempt_id": attempt_id,
                "dataset": "raw-recovery-test",
                "disposition": "recording",
                "archive_complete": False,
                "training_included": False,
            }
            attempt_archive.atomic_write_json(directory / "metadata.json", metadata)
            recording = self.make_recording(directory, 80)
            with (
                patch.object(controlled_record, "encode_video_frames", side_effect=RuntimeError("encoder failed")),
                self.assertRaisesRegex(RuntimeError, "encoder failed"),
            ):
                controlled_record.finalize_attempt_archive(
                    dataset=dataset,
                    directory=directory,
                    metadata=metadata,
                    recording=recording,
                    disposition="failed",
                    samples=0,
                    duration_s=0.2,
                    actual_hz=220.0,
                    failure_label="camera_problem",
                )
            self.assertFalse(metadata["archive_complete"])
            self.assertEqual(metadata["disposition"], "collector_error")
            self.assertEqual(metadata["operator_disposition"], "failed")
            self.assertEqual(metadata["failure_label"], "camera_problem")
            recovery = metadata["raw_frame_recovery"]
            self.assertTrue(recovery["preserved"])
            for relative in recovery["camera_directories"].values():
                self.assertEqual(len(list((directory / relative).glob("*.png"))), 3)
            incomplete = attempt_archive.attempt_inventory(root / "attempts")[0]
            self.assertTrue(
                all(not artifact["available"] for artifact in incomplete["artifacts"].values())
            )
            controlled_record.release_unsaved_attempt_after_archive(
                dataset, directory, metadata
            )
            self.assertEqual(dataset.episode_buffer["size"], 0)
            for relative in recovery["camera_directories"].values():
                self.assertEqual(len(list((directory / relative).glob("*.png"))), 3)

    def test_background_rerun_writer_commits_every_submitted_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recording = controlled_record.rr.RecordingStream(
                "rebot_background_writer_test",
                recording_id=controlled_record.uuid4(),
            )
            recording.set_sinks(
                controlled_record.rr.FileSink(str(directory / "attempt.partial.rrd"))
            )
            writer = controlled_record.AttemptRerunWriter(recording, fps=30)
            for index in range(3):
                image = np.full((24, 32, 3), index * 60, dtype=np.uint8)
                writer.submit(
                    {"front": image, "side": image, "joint": float(index)},
                    {"joint": float(index + 1)},
                    index,
                )
            writer.finish(expected_samples=3)
            recording.flush()
            recording.disconnect()
            final = controlled_record.verify_and_commit_rerun(directory)
            self.assertTrue(final.is_file())

    def test_exception_cleanup_never_deletes_interrupted_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "images" / "observation.images.front" / "episode-000000"
            image_dir.mkdir(parents=True)
            (image_dir / "frame-000000.png").write_bytes(b"recover me")

            class FakeEncodingDataset:
                episodes_since_last_encoding = 0
                num_episodes = 0

                def __init__(self):
                    self.root = root
                    self.finalized = False

                def finalize(self):
                    self.finalized = True

            dataset = FakeEncodingDataset()
            with self.assertRaisesRegex(RuntimeError, "simulated ENOSPC"):
                with controlled_record.RecoverySafeVideoEncodingManager(dataset):
                    raise RuntimeError("simulated ENOSPC")
            self.assertTrue(dataset.finalized)
            self.assertEqual((image_dir / "frame-000000.png").read_bytes(), b"recover me")

    def test_verified_attempt_videos_are_handed_to_lerobot_without_reencoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "attempt"
            dataset_root = root / "dataset"
            archive.mkdir()
            dataset_root.mkdir()
            sources = {
                "observation.images.front": archive / "overhead.mp4",
                "observation.images.side": archive / "wrist.mp4",
            }
            sources["observation.images.front"].write_bytes(b"overhead-video")
            sources["observation.images.side"].write_bytes(b"wrist-video")

            frame_directories: dict[str, Path] = {}
            for key in sources:
                frame_directory = dataset_root / "images" / key / "episode-000000"
                frame_directory.mkdir(parents=True)
                (frame_directory / "frame-000000.png").write_bytes(b"frame")
                frame_directories[key] = frame_directory

            original_encoder = Mock(side_effect=AssertionError("must not re-encode"))

            class FakeDataset:
                root = dataset_root
                meta = SimpleNamespace(video_keys=list(sources))
                _encode_temporary_episode_video = original_encoder

                @staticmethod
                def _get_image_file_dir(_episode_index: int, video_key: str) -> Path:
                    return frame_directories[video_key]

            dataset = FakeDataset()
            original_bound_encoder = dataset._encode_temporary_episode_video
            handed_off: dict[str, Path] = {}
            with controlled_record.reuse_attempt_videos_for_lerobot(dataset, archive):
                for key, source in sources.items():
                    temporary_video = dataset._encode_temporary_episode_video(key, 0)
                    handed_off[key] = temporary_video
                    self.assertEqual(temporary_video.read_bytes(), source.read_bytes())
                    self.assertTrue(source.is_file())
                    self.assertFalse(frame_directories[key].exists())

            original_encoder.assert_not_called()
            self.assertIs(dataset._encode_temporary_episode_video, original_bound_encoder)
            for key, source in sources.items():
                self.assertEqual(source.read_bytes(), handed_off[key].read_bytes())

    def test_checkpoint_is_fresh_loadable_after_each_of_two_no_video_episodes(self) -> None:
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["joint_a", "joint_b"],
            },
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["joint_a", "joint_b"],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            cache = Path(temporary) / "huggingface-cache"
            with patch.object(datasets_config, "HF_DATASETS_CACHE", cache):
                dataset = LeRobotDataset.create(
                    "local/two-episode-durability-test",
                    30,
                    features,
                    root=root,
                    use_videos=False,
                    batch_encoding_size=1,
                )

                for episode_index in range(2):
                    for frame_index in range(3):
                        offset = float(episode_index * 10 + frame_index)
                        dataset.add_frame(
                            {
                                "observation.state": np.array(
                                    [offset, offset + 0.5], dtype=np.float32
                                ),
                                "action": np.array(
                                    [offset + 1.0, offset + 1.5], dtype=np.float32
                                ),
                                "task": "durability test",
                            }
                        )
                    dataset.save_episode(parallel_encoding=False)
                    dataset = controlled_record.checkpoint_and_reopen_dataset(
                        dataset,
                        expected_episode_index=episode_index,
                    )

                    fresh = LeRobotDataset(
                        "local/two-episode-durability-test",
                        root=root,
                        batch_encoding_size=1,
                        vcodec=dataset.vcodec,
                    )
                    self.assertEqual(fresh.num_episodes, episode_index + 1)
                    self.assertEqual(fresh.num_frames, (episode_index + 1) * 3)

                self.assertEqual(dataset.num_episodes, 2)
                self.assertEqual(dataset.num_frames, 6)
                dataset.stop_image_writer()
                dataset.finalize()

                final = LeRobotDataset(
                    "local/two-episode-durability-test",
                    root=root,
                    batch_encoding_size=1,
                    vcodec=dataset.vcodec,
                )
                self.assertEqual(final.num_episodes, 2)
                self.assertEqual(final.num_frames, 6)

    def test_two_camera_episodes_are_durable_separate_videos_without_reencoding(self) -> None:
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["joint_a", "joint_b"],
            },
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["joint_a", "joint_b"],
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": (16, 16, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.side": {
                "dtype": "video",
                "shape": (16, 16, 3),
                "names": ["height", "width", "channels"],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            cache = Path(temporary) / "huggingface-cache"
            with patch.object(datasets_config, "HF_DATASETS_CACHE", cache):
                dataset = LeRobotDataset.create(
                    "local/two-camera-durability-test",
                    30,
                    features,
                    root=root,
                    use_videos=True,
                    image_writer_threads=2,
                    batch_encoding_size=1,
                    vcodec="h264",
                )
                archived_paths: list[tuple[Path, Path]] = []
                for episode_index in range(2):
                    for frame_index in range(3):
                        value = episode_index * 40 + frame_index * 10
                        image = np.full((16, 16, 3), value, dtype=np.uint8)
                        dataset.add_frame(
                            {
                                "observation.state": np.array(
                                    [value, value + 0.5], dtype=np.float32
                                ),
                                "action": np.array(
                                    [value + 1.0, value + 1.5], dtype=np.float32
                                ),
                                "observation.images.front": image,
                                "observation.images.side": 255 - image,
                                "task": "video durability test",
                            }
                        )

                    archive = Path(temporary) / f"attempt-{episode_index}"
                    archive.mkdir()
                    artifacts = controlled_record.archive_attempt_videos(
                        dataset,
                        archive,
                        samples=3,
                    )
                    self.assertEqual(artifacts["overhead"]["frames"], 3)
                    self.assertEqual(artifacts["wrist"]["frames"], 3)
                    archived_paths.append((archive / "overhead.mp4", archive / "wrist.mp4"))

                    with controlled_record.reuse_attempt_videos_for_lerobot(dataset, archive):
                        dataset.save_episode(parallel_encoding=False)
                    dataset = controlled_record.checkpoint_and_reopen_dataset(
                        dataset,
                        expected_episode_index=episode_index,
                    )

                    for camera_key in (
                        "observation.images.front",
                        "observation.images.side",
                    ):
                        video_files = sorted((root / "videos" / camera_key).rglob("*.mp4"))
                        self.assertEqual(len(video_files), episode_index + 1)
                    for overhead, wrist in archived_paths:
                        self.assertGreater(overhead.stat().st_size, 0)
                        self.assertGreater(wrist.stat().st_size, 0)

                fresh = LeRobotDataset(
                    "local/two-camera-durability-test",
                    root=root,
                    batch_encoding_size=1,
                    vcodec="h264",
                )
                self.assertEqual(fresh.num_episodes, 2)
                self.assertEqual(fresh.num_frames, 6)
                dataset.stop_image_writer()
                dataset.finalize()

    def test_record_save_log_lines_drive_visible_workspace_phases(self) -> None:
        manager = workspace.TrainingManager()
        observed: list[str | None] = []

        def output_lines():
            yield "ATTEMPT recording id=attempt-1\n"
            observed.append(manager._record_phase)
            yield "AWAITING_DECISION id=attempt-1\n"
            observed.append(manager._record_phase)
            yield "ATTEMPT save_started id=attempt-1 phase=rerun_and_attempt_videos\n"
            observed.append(manager._record_phase)
            yield "ATTEMPT save_phase id=attempt-1 phase=lerobot\n"
            observed.append(manager._record_phase)
            yield "ATTEMPT lerobot_durable id=attempt-1 episode=0 total_episodes=1\n"
            observed.append(manager._record_phase)

        class FakeProcess:
            pid = 31337
            stdout = output_lines()

            @staticmethod
            def wait() -> int:
                return 0

            @staticmethod
            def poll() -> int:
                return 0

        process = FakeProcess()
        manager._kind = "record"
        manager._process = process
        manager._reader_thread = threading.current_thread()
        manager._read_process(process, "record", None, None)
        self.assertEqual(
            observed,
            [
                "recording",
                "awaiting_decision",
                "saving_rerun",
                "saving_lerobot",
                "returning_home",
            ],
        )
        self.assertIsNone(manager._record_phase)

    def test_record_control_writes_label_before_signalling(self) -> None:
        class FakeProcess:
            pid = 424242

            @staticmethod
            def poll():
                return None

        with tempfile.TemporaryDirectory() as temporary:
            control_file = Path(temporary) / "control.json"
            manager = workspace.TrainingManager()
            manager._kind = "record"
            manager._process = FakeProcess()
            manager._record_control_file = control_file
            with self.assertRaisesRegex(workspace.TrainingConfigError, "Choose a failure reason"):
                manager.record_control({"action": "rerecord"})
            with patch.object(workspace.os, "kill") as send_signal:
                manager.record_control(
                    {
                        "action": "rerecord",
                        "failure_label": "test_or_setup",
                        "failure_note": "dry run",
                    }
                )
            decision = json.loads(control_file.read_text())
            self.assertEqual(decision["failure_label"], "test_or_setup")
            self.assertEqual(decision["failure_note"], "dry run")
            send_signal.assert_called_once_with(FakeProcess.pid, workspace.signal.SIGUSR2)
            with self.assertRaisesRegex(RuntimeError, "still being archived"):
                manager.record_control({"action": "finish"})
            before_stop = control_file.read_text()
            with patch.object(workspace.os, "kill") as send_stop:
                manager.record_control({"action": "stop"})
            self.assertEqual(control_file.read_text(), before_stop)
            send_stop.assert_called_once_with(FakeProcess.pid, workspace.signal.SIGHUP)

    def test_shutdown_waits_for_record_archive_and_escalates_recovery_safely(self) -> None:
        class FakeProcess:
            pid = 515151

            def __init__(self):
                self.running = True
                self.wait_calls = 0

            def poll(self):
                return None if self.running else 130

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls < 3:
                    raise subprocess.TimeoutExpired("collector", timeout)
                self.running = False
                return 130

        process = FakeProcess()
        reader = Mock()
        manager = workspace.TrainingManager()
        manager._kind = "record"
        manager._process = process
        manager._reader_thread = reader
        manager._SHUTDOWN_WAIT_SLICE_S = 1.0
        manager._RECORD_ARCHIVE_GRACE_S = 2.0

        with (
            patch.object(workspace.os, "kill") as graceful_stop,
            patch.object(workspace.os, "killpg") as recovery_interrupt,
        ):
            manager.shutdown_cleanup()

        graceful_stop.assert_called_once_with(process.pid, workspace.signal.SIGHUP)
        recovery_interrupt.assert_called_once_with(process.pid, workspace.signal.SIGINT)
        self.assertFalse(process.running)
        reader.join.assert_called_once_with()
        messages = [entry["message"] for entry in manager.logs_since(0)]
        self.assertTrue(any("recovery-safe interruption" in message for message in messages))

    def test_failure_decision_wins_over_simultaneous_stop(self) -> None:
        self.assertEqual(
            controlled_record.resolve_attempt_disposition(
                {"stop": True, "rerecord": True},
                "rerecord",
            ),
            "failed",
        )
        self.assertEqual(
            controlled_record.resolve_attempt_disposition(
                {"stop": True, "rerecord": False},
                None,
            ),
            "aborted",
        )
        self.assertEqual(
            controlled_record.resolve_attempt_disposition(
                {"stop": True, "rerecord": False},
                "finish",
            ),
            "kept",
        )

    def test_save_exception_records_uncertain_if_disk_info_advanced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            directory = Path(temporary) / "attempt"
            (root / "meta").mkdir(parents=True)
            directory.mkdir()
            (root / "meta" / "info.json").write_text('{"total_episodes":1}')
            metadata = {
                "disposition": "kept",
                "archive_complete": True,
                "training_included": False,
            }
            controlled_record.mark_training_save_failure(
                directory,
                metadata,
                root,
                0,
                RuntimeError("stats write failed"),
            )
            saved = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(saved["disposition"], "commit_uncertain")
            self.assertEqual(saved["training_commit_state"], "uncertain")
            self.assertIsNone(saved["training_included"])

    def test_hardware_ownership_token_cannot_release_a_new_owner(self) -> None:
        shared = threading.Lock()
        training = workspace.TrainingManager(hardware_lock=shared)
        teleop = server.TeleopManager(hardware_lock=shared)
        old_token = training._claim_hardware()
        training._release_hardware(old_token)
        current_token = training._claim_hardware()
        training._release_hardware(old_token)
        with self.assertRaisesRegex(RuntimeError, "hardware is starting"):
            teleop._claim_hardware()
        training._release_hardware(current_token)
        teleop_token = teleop._claim_hardware()
        teleop._release_hardware(teleop_token)

    def test_training_manager_treats_exited_child_as_finalizing(self) -> None:
        class ExitedProcess:
            @staticmethod
            def poll():
                return 0

        manager = workspace.TrainingManager()
        manager._kind = "record"
        manager._process = ExitedProcess()
        with self.assertRaisesRegex(RuntimeError, "still finalizing"):
            manager._ensure_idle()
        self.assertTrue(manager.status()["finalizing"])

    def test_start_reservation_precedes_record_artifacts(self) -> None:
        manager = workspace.TrainingManager()
        manager._lifecycle_lock.acquire()
        try:
            with (
                patch.object(manager, "_start_record_locked") as start_locked,
                self.assertRaisesRegex(RuntimeError, "start or stop is in progress"),
            ):
                manager.start_record({}, {})
            start_locked.assert_not_called()
        finally:
            manager._lifecycle_lock.release()

    def test_owned_process_stops_worker_when_gui_pipe_closes(self) -> None:
        owner_read, owner_write = os.pipe()
        worker_code = (
            "import signal,time\n"
            "running=True\n"
            "def stop(*_):\n global running\n running=False\n"
            "signal.signal(signal.SIGINT, stop)\n"
            "print('WORKER_READY', flush=True)\n"
            "while running: time.sleep(0.01)\n"
            "print('WORKER_STOPPED', flush=True)\n"
        )
        process = subprocess.Popen(
            [
                sys.executable,
                str(GUI_ROOT / "owned_process.py"),
                "--owner-fd",
                str(owner_read),
                "--owner-stop-signal",
                "SIGINT",
                "--",
                sys.executable,
                "-u",
                "-c",
                worker_code,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            pass_fds=(owner_read,),
            start_new_session=True,
        )
        os.close(owner_read)
        assert process.stdout is not None
        self.assertEqual(process.stdout.readline().strip(), "WORKER_READY")
        os.close(owner_write)
        output = process.communicate(timeout=5)[0]
        self.assertEqual(process.returncode, 0)
        self.assertIn("OWNER_LOST forwarding=SIGINT", output)
        self.assertIn("WORKER_STOPPED", output)

    def test_owned_process_cleanup_survives_broken_log_pipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / "ready"
            stopped = Path(temporary) / "stopped"
            owner_read, owner_write = os.pipe()
            worker_code = (
                "import pathlib,signal,sys,time\n"
                "running=True\n"
                "def stop(*_):\n global running\n print('STOP_SIGNAL', flush=True)\n running=False\n"
                "signal.signal(signal.SIGINT, stop)\n"
                "pathlib.Path(sys.argv[1]).write_text('ready')\n"
                "while running: time.sleep(0.01)\n"
                "pathlib.Path(sys.argv[2]).write_text('stopped')\n"
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(GUI_ROOT / "owned_process.py"),
                    "--owner-fd",
                    str(owner_read),
                    "--owner-stop-signal",
                    "SIGINT",
                    "--",
                    sys.executable,
                    "-c",
                    worker_code,
                    str(ready),
                    str(stopped),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                pass_fds=(owner_read,),
                start_new_session=True,
            )
            os.close(owner_read)
            for _ in range(300):
                if ready.is_file():
                    break
                time.sleep(0.01)
            self.assertTrue(ready.is_file())
            assert process.stdout is not None
            process.stdout.close()
            os.close(owner_write)
            self.assertEqual(process.wait(timeout=5), 0)
            self.assertEqual(stopped.read_text(), "stopped")


class FileStreamingTest(unittest.TestCase):
    def test_static_file_supports_bounded_http_range(self) -> None:
        handler = object.__new__(server.ReBotHandler)
        handler.headers = {"Range": "bytes=0-15"}
        handler.wfile = io.BytesIO()
        statuses: list[int] = []
        headers: dict[str, str] = {}
        handler.send_response = lambda status: statuses.append(int(status))
        handler.send_header = lambda name, value: headers.__setitem__(name, value)
        handler.end_headers = lambda: None
        handler._send_file(GUI_ROOT / "static" / "training.js", "text/javascript")
        self.assertEqual(statuses, [206])
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        self.assertEqual(headers["Content-Range"].split("/")[0], "bytes 0-15")
        self.assertEqual(len(handler.wfile.getvalue()), 16)

    def test_all_responses_deny_framing(self) -> None:
        handler = object.__new__(server.ReBotHandler)
        handler.request_version = "HTTP/1.1"
        handler.command = "GET"
        handler.requestline = "GET /training HTTP/1.1"
        handler.wfile = io.BytesIO()
        handler._headers_buffer = []
        handler.send_response(200)
        handler.end_headers()
        headers = handler.wfile.getvalue().decode("latin-1")
        self.assertIn("X-Frame-Options: DENY", headers)
        self.assertIn("frame-ancestors 'none'", headers)


if __name__ == "__main__":
    unittest.main()
