#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraResult:
    label: str
    index: int
    opened: bool = False
    successful_frames: int = 0
    failed_reads: int = 0
    measured_fps: float = 0.0
    width: int = 0
    height: int = 0
    reported_width: int = 0
    reported_height: int = 0
    reported_fps: float = 0.0
    shape_mismatch: bool = False
    shape_changed: bool = False
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    dark_or_covered: bool = True
    image_path: str | None = None
    error: str | None = None


def capture_camera(
    label: str,
    index: int,
    requested_fps: float,
    requested_width: int,
    requested_height: int,
    duration_s: float,
    output_dir: Path,
    start_event: threading.Event,
    result: CameraResult,
) -> None:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    try:
        result.opened = cap.isOpened()
        if not result.opened:
            result.error = "camera did not open"
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
        cap.set(cv2.CAP_PROP_FPS, requested_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        result.reported_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result.reported_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result.reported_fps = float(cap.get(cv2.CAP_PROP_FPS))

        start_event.wait()
        started = time.perf_counter()
        deadline = started + duration_s
        last_frame = None

        while time.perf_counter() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                result.successful_frames += 1
                last_frame = frame
                frame_height, frame_width = frame.shape[:2]
                if result.width and (result.width != frame_width or result.height != frame_height):
                    result.shape_changed = True
                result.width = int(frame_width)
                result.height = int(frame_height)
            else:
                result.failed_reads += 1

        elapsed = max(time.perf_counter() - started, 1e-6)
        result.measured_fps = result.successful_frames / elapsed

        if last_frame is not None:
            result.shape_mismatch = (
                result.width != requested_width or result.height != requested_height
            )
            gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
            result.brightness_mean = float(np.mean(gray))
            result.brightness_std = float(np.std(gray))
            result.dark_or_covered = (
                result.brightness_mean < 12.0 or result.brightness_std < 4.0
            )
            image_path = output_dir / f"{label}_index{index}.png"
            cv2.imwrite(str(image_path), last_frame)
            result.image_path = str(image_path)
        else:
            result.error = "no frames were captured"
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        cap.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", type=int, default=0)
    parser.add_argument("--side", type=int, default=1)
    parser.add_argument("--excluded-screen", type=int, default=3)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--front-width", type=int, default=640)
    parser.add_argument("--front-height", type=int, default=480)
    parser.add_argument("--side-width", type=int, default=1280)
    parser.add_argument("--side-height", type=int, default=720)
    parser.add_argument("--minimum-fps", type=float, default=27.0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    start_event = threading.Event()
    results = {
        "front": CameraResult("front", args.front),
        "side": CameraResult("side", args.side),
    }
    requested_sizes = {
        "front": (args.front_width, args.front_height),
        "side": (args.side_width, args.side_height),
    }
    threads = [
        threading.Thread(
            target=capture_camera,
            args=(
                result.label,
                result.index,
                args.fps,
                requested_sizes[result.label][0],
                requested_sizes[result.label][1],
                args.duration,
                args.output,
                start_event,
                result,
            ),
            daemon=True,
        )
        for result in results.values()
    ]

    for thread in threads:
        thread.start()
    start_event.set()
    for thread in threads:
        thread.join(timeout=args.duration + 12.0)

    passed = True
    for result in results.values():
        if (
            not result.opened
            or result.successful_frames == 0
            or result.measured_fps < args.minimum_fps
            or result.dark_or_covered
            or result.shape_mismatch
            or result.shape_changed
            or result.error is not None
        ):
            passed = False

    report = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requested_fps": args.fps,
        "requested_sizes": {
            "front": {"width": args.front_width, "height": args.front_height},
            "side": {"width": args.side_width, "height": args.side_height},
        },
        "minimum_fps": args.minimum_fps,
        "duration_s": args.duration,
        "excluded_screen_camera_index": args.excluded_screen,
        "cameras": {key: asdict(value) for key, value in results.items()},
        "passed": passed,
    }
    report_path = args.output / "camera_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report: {report_path}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
