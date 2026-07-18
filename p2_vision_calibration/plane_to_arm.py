"""Back-compat: Calibration.load(...).pixel_to_arm(u,v) → CalibrationChain."""

from __future__ import annotations

from pathlib import Path

from p2_vision_calibration.calibration_io import DEFAULT_PATH
from p2_vision_calibration.pixel_to_arm import CalibrationChain
from p2_vision_calibration.plane_to_arm_registration import fit_affine, plane_to_arm

__all__ = ["Calibration", "CalibrationChain", "fit_affine", "plane_to_arm"]


class Calibration(CalibrationChain):
    """Alias matching earlier STRUCTURE.md contract."""

    @classmethod
    def load(cls, directory: str | Path = "data/calibration") -> Calibration:
        d = Path(directory)
        path = d / "calibration.json" if d.is_dir() else d
        if not path.exists() and d.is_dir():
            path = DEFAULT_PATH
        return cls(path)
