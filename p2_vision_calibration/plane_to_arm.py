"""Plane mm → arm mm (2D affine / similarity from 4 point pairs) + combined Calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from p2_vision_calibration.aruco_homography import load_H, pixel_to_plane


@dataclass
class PlaneToArm:
    """Maps desk-plane (x,y) mm → arm base (x,y) mm."""

    A: np.ndarray  # 2x2
    b: np.ndarray  # 2

    def transform(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        p = self.A @ np.array([x_mm, y_mm]) + self.b
        return float(p[0]), float(p[1])

    def save(self, path: Path = Path("data/calibration/plane_to_arm.json")) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"A": self.A.tolist(), "b": self.b.tolist()}, indent=2)
        )

    @classmethod
    def load(cls, path: Path = Path("data/calibration/plane_to_arm.json")) -> PlaneToArm:
        raw = json.loads(path.read_text())
        return cls(A=np.array(raw["A"]), b=np.array(raw["b"]))


def fit_plane_to_arm(
    plane_xy: np.ndarray,
    arm_xy: np.ndarray,
) -> PlaneToArm:
    """Least squares affine: arm = A @ plane + b. plane_xy/arm_xy shape (N,2), N>=3."""
    n = plane_xy.shape[0]
    M = np.zeros((2 * n, 6))
    y = np.zeros(2 * n)
    for i, ((px, py), (ax, ay)) in enumerate(zip(plane_xy, arm_xy)):
        M[2 * i] = [px, py, 1, 0, 0, 0]
        M[2 * i + 1] = [0, 0, 0, px, py, 1]
        y[2 * i] = ax
        y[2 * i + 1] = ay
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    A = np.array([[coef[0], coef[1]], [coef[3], coef[4]]])
    b = np.array([coef[2], coef[5]])
    return PlaneToArm(A=A, b=b)


@dataclass
class Calibration:
    H: np.ndarray
    plane_to_arm: PlaneToArm

    @classmethod
    def load(cls, directory: str | Path = "data/calibration") -> Calibration:
        d = Path(directory)
        return cls(H=load_H(d / "homography.npy"), plane_to_arm=PlaneToArm.load(d / "plane_to_arm.json"))

    def pixel_to_arm(self, u: float, v: float) -> tuple[float, float]:
        px, py = pixel_to_plane(self.H, u, v)
        return self.plane_to_arm.transform(px, py)
