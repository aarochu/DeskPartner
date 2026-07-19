"""Tests for the live 3D arm view logged into Rerun during recording.

Written before the implementation (arm3d.py). Contract under test:

- A simple kinematic skeleton of the B601-DM drawn from the 7 joint
  observations (degrees, keys like ``shoulder_pan.pos``) — an intuitive
  approximation, not a URDF-accurate model.
- Logged live into the attempt RecordingStream so the shared viewer that
  pops up on record shows a moving arm, not just raw scalar lanes.
- A curated, minimal blueprint (3D + two cameras + a couple of plots) so
  the first thing an operator sees is simple, not overwhelming.
- Everything fails soft: a missing joint or an old Rerun SDK must never
  break recording.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

KIT_ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = KIT_ROOT / "teleop_gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

import arm3d  # noqa: E402


def joints(pan=0.0, lift=0.0, elbow=0.0, wrist_flex=0.0, wrist_yaw=0.0, wrist_roll=0.0, gripper=0.0):
    return {
        "shoulder_pan.pos": pan,
        "shoulder_lift.pos": lift,
        "elbow_flex.pos": elbow,
        "wrist_flex.pos": wrist_flex,
        "wrist_yaw.pos": wrist_yaw,
        "wrist_roll.pos": wrist_roll,
        "gripper.pos": gripper,
    }


def dist(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


class FakeRecording:
    """Captures (entity_path, archetype-class-name, static) without Rerun I/O."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []
        self.blueprints: list[object] = []

    def log(self, entity, archetype, *, static=False):
        self.calls.append((entity, type(archetype).__name__, bool(static)))

    def send_blueprint(self, blueprint):
        self.blueprints.append(blueprint)

    def entities(self):
        return [entity for entity, _, _ in self.calls]


class ForwardKinematicsTest(unittest.TestCase):
    def test_link_lengths_are_preserved_for_any_pose(self) -> None:
        for pose in (
            joints(),
            joints(pan=45, lift=30, elbow=60, wrist_flex=-40),
            joints(pan=-120, lift=85, elbow=-70, wrist_flex=90),
        ):
            points = arm3d.fk_points(pose)
            self.assertEqual(len(points), 5)  # base, shoulder, elbow, wrist, tip
            expected = (arm3d.BASE_HEIGHT_M, arm3d.UPPER_ARM_M, arm3d.FOREARM_M, arm3d.WRIST_M)
            for index, length in enumerate(expected):
                self.assertAlmostEqual(dist(points[index], points[index + 1]), length, places=6)

    def test_zero_pose_stands_upright(self) -> None:
        points = arm3d.fk_points(joints())
        tip = points[-1]
        self.assertAlmostEqual(tip[0], 0.0, places=6)
        self.assertAlmostEqual(tip[1], 0.0, places=6)
        self.assertAlmostEqual(
            tip[2],
            arm3d.BASE_HEIGHT_M + arm3d.UPPER_ARM_M + arm3d.FOREARM_M + arm3d.WRIST_M,
            places=6,
        )

    def test_pan_spins_the_arm_without_changing_height_or_radius(self) -> None:
        bent = dict(lift=50, elbow=40, wrist_flex=-20)
        tip_a = arm3d.fk_points(joints(pan=0, **bent))[-1]
        tip_b = arm3d.fk_points(joints(pan=90, **bent))[-1]
        radius_a = math.hypot(tip_a[0], tip_a[1])
        radius_b = math.hypot(tip_b[0], tip_b[1])
        self.assertAlmostEqual(radius_a, radius_b, places=6)
        self.assertAlmostEqual(tip_a[2], tip_b[2], places=6)
        self.assertGreater(radius_a, 0.1)  # genuinely bent, not a degenerate zero-radius case

    def test_lifting_the_shoulder_lowers_the_tip_and_reaches_out(self) -> None:
        upright = arm3d.fk_points(joints())[-1]
        reaching = arm3d.fk_points(joints(lift=90))[-1]
        self.assertLess(reaching[2], upright[2])
        self.assertGreater(math.hypot(reaching[0], reaching[1]), 0.3)

    def test_gripper_jaw_separation_is_monotonic_and_clamped(self) -> None:
        closed = arm3d.jaw_separation_m(arm3d.GRIPPER_CLOSED_DEG)
        opened = arm3d.jaw_separation_m(arm3d.GRIPPER_OPEN_DEG)
        self.assertLess(closed, opened)
        self.assertGreater(closed, 0.0)  # jaws never fully merge visually
        middle = arm3d.jaw_separation_m((arm3d.GRIPPER_CLOSED_DEG + arm3d.GRIPPER_OPEN_DEG) / 2)
        self.assertTrue(closed < middle < opened)
        beyond_open = arm3d.jaw_separation_m(arm3d.GRIPPER_OPEN_DEG + 500)
        beyond_closed = arm3d.jaw_separation_m(arm3d.GRIPPER_CLOSED_DEG - 500)
        self.assertAlmostEqual(beyond_open, opened, places=6)
        self.assertAlmostEqual(beyond_closed, closed, places=6)


class LoggingContractTest(unittest.TestCase):
    def test_scene_is_logged_once_and_static(self) -> None:
        recording = FakeRecording()
        arm3d.log_scene(recording)
        self.assertTrue(recording.calls)
        for entity, _, static in recording.calls:
            self.assertTrue(entity.startswith("world/"), entity)
            self.assertTrue(static, f"{entity} must be logged as static scene data")

    def test_pose_logs_skeleton_joints_and_gripper_under_world(self) -> None:
        recording = FakeRecording()
        self.assertTrue(arm3d.log_pose(recording, joints(pan=30, lift=45, elbow=30, gripper=40)))
        entities = recording.entities()
        self.assertIn(arm3d.ARM_ENTITY, entities)
        self.assertIn(arm3d.JOINTS_ENTITY, entities)
        self.assertIn(arm3d.GRIPPER_ENTITY, entities)
        kinds = {entity: kind for entity, kind, _ in recording.calls}
        self.assertEqual(kinds[arm3d.ARM_ENTITY], "LineStrips3D")
        self.assertEqual(kinds[arm3d.JOINTS_ENTITY], "Points3D")
        self.assertEqual(kinds[arm3d.GRIPPER_ENTITY], "LineStrips3D")
        for entity, _, static in recording.calls:
            self.assertFalse(static, f"{entity} is per-frame pose data, not static")

    def test_pose_extraction_tolerates_camera_frames_in_observation(self) -> None:
        observation = joints(lift=20)
        observation["front"] = object()  # stands in for a camera ndarray
        observation["side"] = object()
        recording = FakeRecording()
        self.assertTrue(arm3d.log_pose(recording, observation))

    def test_pose_fails_soft_when_joints_are_missing_or_bad(self) -> None:
        recording = FakeRecording()
        self.assertFalse(arm3d.log_pose(recording, {}))
        self.assertFalse(arm3d.log_pose(recording, {"shoulder_pan.pos": 10.0}))
        broken = joints()
        broken["elbow_flex.pos"] = float("nan")
        self.assertFalse(arm3d.log_pose(recording, broken))
        self.assertEqual(recording.calls, [])

    def test_pose_never_raises_even_if_recording_log_explodes(self) -> None:
        class ExplodingRecording:
            def log(self, *args, **kwargs):
                raise RuntimeError("sink gone")

        self.assertFalse(arm3d.log_pose(ExplodingRecording(), joints()))


class BlueprintTest(unittest.TestCase):
    def test_simple_blueprint_is_sent_to_the_recording(self) -> None:
        recording = FakeRecording()
        self.assertTrue(arm3d.send_simple_blueprint(recording))
        self.assertEqual(len(recording.blueprints), 1)
        self.assertIsNotNone(recording.blueprints[0])

    def test_blueprint_failure_is_soft(self) -> None:
        class NoBlueprintRecording:
            def send_blueprint(self, blueprint):
                raise RuntimeError("viewer too old")

        self.assertFalse(arm3d.send_simple_blueprint(NoBlueprintRecording()))


class RecorderWiringTest(unittest.TestCase):
    """Source contract, mirroring test_rerun_library's UI-contract style."""

    def test_controlled_record_wires_the_live_3d_view(self) -> None:
        source = (GUI_ROOT / "controlled_record.py").read_text(encoding="utf-8")
        self.assertIn("import arm3d", source)
        self.assertIn("arm3d.log_scene(", source)
        self.assertIn("arm3d.send_simple_blueprint(", source)
        self.assertIn("arm3d.log_pose(", source)


if __name__ == "__main__":
    unittest.main()
