"""OpenCV camera helpers for Rerun logging."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from p5_rerun_port.constants import CAMERA_TO_RERUN


@dataclass
class CameraStream:
    name: str
    index: int
    width: int = 640
    height: int = 480
    cap: object | None = field(default=None, repr=False)

    @property
    def entity(self) -> str:
        return CAMERA_TO_RERUN.get(self.name, f"camera/{self.name}")

    def open(self) -> None:
        import cv2

        self.cap = cv2.VideoCapture(self.index)
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Prefer MJPG on UVC cams (Seeed / innomaker guidance)
        try:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open camera index {self.index} ({self.name})")

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_cameras(specs: list[tuple[str, int, int, int]]) -> list[CameraStream]:
    streams: list[CameraStream] = []
    for name, index, width, height in specs:
        cam = CameraStream(name=name, index=index, width=width, height=height)
        try:
            cam.open()
            streams.append(cam)
            print(f"camera: {name} index={index} -> {cam.entity}", flush=True)
        except Exception as exc:
            print(f"WARN: skip camera {name} index={index}: {exc}", flush=True)
    return streams


def fake_frame(width: int = 640, height: int = 480, t: float = 0.0, label: str = "") -> np.ndarray:
    """Synthetic BGR frame for dry-run / --fake."""
    import cv2

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (40, 40, 40)
    x = int((0.5 + 0.4 * np.sin(t)) * width)
    y = int((0.5 + 0.3 * np.cos(t * 0.7)) * height)
    cv2.circle(img, (x, y), 40, (80, 180, 255), -1)
    cv2.putText(img, label or "fake", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
    return img
