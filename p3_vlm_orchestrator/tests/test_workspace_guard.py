from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from p3_vlm_orchestrator.policy_rollout.workspace_guard import (
    CalibratedWorkspaceGuard,
    WorkspaceViolation,
)


NOW = datetime(2026, 7, 18, 20, 0, tzinfo=timezone.utc)


class CalibratedWorkspaceGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.arm_path = self.root / "arm.yaml"
        self.workspace_path = self.root / "workspace.yaml"
        self.calibration_path = self.root / "calibration.json"
        self._write_files()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_files(
        self,
        *,
        updated_at: str | None = None,
        affine_A: object = ((1.0, 0.0), (0.0, 1.0)),
        affine_b: object = (10.0, 20.0),
        plane_corners: dict[str, list[float]] | None = None,
        width_mm: float = 400.0,
        depth_mm: float = 300.0,
        transit_height_mm: float = 120.0,
        approach_height_mm: float | None = 40.0,
    ) -> None:
        arm = {
            "sdk": {"repo": str(self.root / "sdk")},
            "safety": {"transit_height_mm": transit_height_mm},
            "grasp_heights_mm": {"low": 15.0, "high": 30.0},
        }
        if approach_height_mm is not None:
            arm["safety"]["approach_height_mm"] = approach_height_mm
        self.arm_path.write_text(json.dumps(arm), encoding="utf-8")
        self.workspace_path.write_text(
            json.dumps({"zone": {"width_mm": width_mm, "depth_mm": depth_mm}}),
            encoding="utf-8",
        )
        calibration = {
            "updated_at": updated_at or (NOW - timedelta(hours=1)).isoformat(),
            "aruco": {
                "ids": [0, 1, 2, 3],
                "plane_mm": plane_corners
                or {
                    "0": [0.0, 0.0],
                    "1": [400.0, 0.0],
                    "2": [400.0, 300.0],
                    "3": [0.0, 300.0],
                },
            },
            "plane_to_arm": {"A": affine_A, "b": affine_b},
        }
        self.calibration_path.write_text(json.dumps(calibration), encoding="utf-8")

    def _guard(self, fk=lambda action: (210.0, 170.0, 50.0), **kwargs):
        return CalibratedWorkspaceGuard.from_files(
            arm_config_path=self.arm_path,
            workspace_config_path=self.workspace_path,
            calibration_path=self.calibration_path,
            fk_deg_to_xyz_mm=fk,
            current_utc=NOW,
            **kwargs,
        )

    def test_accepts_inside_corner_and_reversed_winding_polygon(self) -> None:
        self._guard().validate(np.zeros(7))
        self._guard(fk=lambda action: (10.0, 20.0, 5.0)).validate(np.zeros(7))

        reversed_corners = {
            "0": [0.0, 0.0],
            "1": [0.0, 300.0],
            "2": [400.0, 300.0],
            "3": [400.0, 0.0],
        }
        self._write_files(plane_corners=reversed_corners)
        self._guard().validate(np.zeros(7))

    def test_rejects_tip_outside_xy_or_z_range(self) -> None:
        for xyz in ((410.1, 170.0, 50.0), (210.0, 170.0, 4.9), (210.0, 170.0, 160.1)):
            with self.subTest(xyz=xyz), self.assertRaises(WorkspaceViolation):
                self._guard(fk=lambda action, xyz=xyz: xyz).validate(np.zeros(7))

    def test_default_z_bounds_use_grasp_transit_and_approach_config(self) -> None:
        guard = self._guard()
        self.assertEqual(guard.z_min_mm, 5.0)
        self.assertEqual(guard.z_max_mm, 160.0)

        self._write_files(approach_height_mm=None)
        self.assertEqual(self._guard().z_max_mm, 160.0)

    def test_rejects_invalid_z_range(self) -> None:
        self._write_files(transit_height_mm=-50.0, approach_height_mm=40.0)
        with self.assertRaisesRegex(ValueError, "Z range"):
            self._guard()

    def test_rejects_missing_malformed_or_nonfinite_affine(self) -> None:
        for A, b in ((None, None), ([[1, 0]], [0, 0]), ([[1, 0], [0, 1]], [0, np.nan])):
            with self.subTest(A=A, b=b):
                self._write_files(affine_A=A, affine_b=b)
                with self.assertRaisesRegex(ValueError, "affine"):
                    self._guard()

    def test_rejects_missing_unparseable_expired_or_future_calibration_time(self) -> None:
        timestamps = (
            None,
            "not-a-time",
            (NOW - timedelta(hours=12, seconds=1)).isoformat(),
            (NOW + timedelta(minutes=5, seconds=1)).isoformat(),
        )
        for timestamp in timestamps:
            with self.subTest(timestamp=timestamp):
                self._write_files(updated_at=timestamp or "")
                if timestamp is None:
                    data = json.loads(self.calibration_path.read_text())
                    data.pop("updated_at")
                    self.calibration_path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "timestamp|stale|future"):
                    self._guard()

    def test_accepts_configurable_calibration_age_and_callable_clock(self) -> None:
        self._write_files(updated_at=(NOW - timedelta(hours=24)).isoformat())
        guard = CalibratedWorkspaceGuard.from_files(
            arm_config_path=self.arm_path,
            workspace_config_path=self.workspace_path,
            calibration_path=self.calibration_path,
            fk_deg_to_xyz_mm=lambda action: (210.0, 170.0, 50.0),
            current_utc=lambda: NOW,
            max_calibration_age_s=25 * 60 * 60,
        )
        guard.validate(np.zeros(7))

    def test_rejects_workspace_dimensions_inconsistent_with_plane_corners(self) -> None:
        self._write_files(width_mm=401.0)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self._guard()

    def test_default_sdk_fk_is_lazy_converts_first_six_degrees_and_returns_mm(self) -> None:
        sdk_repo = self.root / "sdk"
        sdk_repo.mkdir()
        captured = []
        package = ModuleType("reBotArm_control_py")
        package.__path__ = []  # type: ignore[attr-defined]
        kinematics = ModuleType("reBotArm_control_py.kinematics")

        def joint_to_pose(q):
            captured.append(np.asarray(q).copy())
            return np.array([0.21, 0.17, 0.05]), np.zeros(3)

        kinematics.joint_to_pose = joint_to_pose  # type: ignore[attr-defined]
        guard = CalibratedWorkspaceGuard.from_files(
            arm_config_path=self.arm_path,
            workspace_config_path=self.workspace_path,
            calibration_path=self.calibration_path,
            current_utc=NOW,
        )
        self.assertNotIn("reBotArm_control_py.kinematics", sys.modules)

        with patch.dict(
            sys.modules,
            {
                "reBotArm_control_py": package,
                "reBotArm_control_py.kinematics": kinematics,
            },
        ):
            guard.validate(np.array([0.0, 90.0, -90.0, 45.0, -45.0, 180.0, 999.0]))

        np.testing.assert_allclose(
            captured[0],
            np.radians([0.0, 90.0, -90.0, 45.0, -45.0, 180.0]),
        )

    def test_rejects_malformed_action_or_fk_return(self) -> None:
        with self.assertRaisesRegex(WorkspaceViolation, "shape"):
            self._guard().validate(np.zeros(6))
        with self.assertRaisesRegex(WorkspaceViolation, "finite"):
            self._guard().validate(np.array([0, 0, 0, 0, 0, 0, np.nan]))
        with self.assertRaisesRegex(WorkspaceViolation, "forward kinematics"):
            self._guard(fk=lambda action: (1.0, 2.0)).validate(np.zeros(7))


if __name__ == "__main__":
    unittest.main()
