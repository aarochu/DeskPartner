"""Smoke-test: stream reBot joints + cameras + URDF into Rerun (SO ``log-so100`` equivalent).

Examples::

    python -m p5_rerun_port.log_rebot --fake --seconds 10
    python -m p5_rerun_port.log_rebot --fake --teleop --seconds 5
    # venue (arms plugged in):
    python -m p5_rerun_port.log_rebot --seconds 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from p5_rerun_port.cameras import CameraStream, fake_frame, open_cameras
from p5_rerun_port.config import camera_specs, load_recording_config, resolve_urdf_path
from p5_rerun_port.constants import CAMERA_TO_RERUN, FOLLOWER, JOINT_NAMES
from p5_rerun_port.hardware import make_session
from p5_rerun_port.takes import begin_recording, finish_recording, log_image_bgr, log_scalars, set_timeline
from p5_rerun_port.urdf_log import log_urdf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Log reBot telemetry + cameras + URDF to Rerun")
    ap.add_argument("--fake", action="store_true", help="No hardware; synthetic joints/cams")
    ap.add_argument("--teleop", action="store_true", help="Mirror leader->follower and log follower/goal")
    ap.add_argument("--seconds", type=float, default=None, help="Stop after N seconds (default: Ctrl-C)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--save", type=Path, default=None, help="Also write this .rrd path")
    ap.add_argument("--no-viewer", action="store_true", help="Do not spawn Rerun viewer")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--urdf", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_recording_config()
    save_path = args.save or (Path("recordings") / "_smoke" / "live.rrd")

    rec = begin_recording(
        save_path,
        episode="live",
        dataset="_smoke",
        task="log_rebot smoke",
        spawn_viewer=not args.no_viewer,
        save_only=args.no_viewer,
    )

    log_urdf(rec, resolve_urdf_path(args.urdf))
    try:
        import rerun as rr

        rec.log(
            f"{FOLLOWER}/joint_names",
            rr.TextDocument(", ".join(JOINT_NAMES)),
            static=True,
        )
    except Exception:
        pass

    session = make_session(fake=args.fake, teleop=args.teleop, recording_cfg=cfg)
    session.start()

    cam_streams: list[CameraStream] = []
    use_fake_cams = args.fake and not args.no_cameras
    if not args.no_cameras and not args.fake:
        cam_streams = open_cameras(camera_specs(cfg))

    period = 1.0 / max(args.fps, 1.0)
    t0 = time.time()
    n = 0
    print(
        f"logging: follower/position"
        f"{' + follower/goal' if args.teleop else ''}"
        f" @ {args.fps} Hz - Ctrl-C to stop",
        flush=True,
    )
    try:
        while True:
            loop_t = time.time()
            if args.seconds is not None and (loop_t - t0) >= args.seconds:
                break
            state, goal = session.read()
            set_timeline(rec, loop_t)
            log_scalars(rec, f"{FOLLOWER}/position", state)
            if goal is not None:
                log_scalars(rec, f"{FOLLOWER}/goal", goal)

            if use_fake_cams:
                for name in ("front", "side"):
                    log_image_bgr(
                        rec,
                        CAMERA_TO_RERUN[name],
                        fake_frame(640, 480, loop_t - t0, name),
                    )
            for cam in cam_streams:
                frame = cam.read()
                if frame is not None:
                    log_image_bgr(rec, cam.entity, frame)

            n += 1
            sleep = period - (time.time() - loop_t)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        finish_recording(rec, dataset="_smoke", task="log_rebot smoke", tag="")
        session.close()
        for cam in cam_streams:
            cam.close()

    print(f"OK: logged {n} frames ({time.time() - t0:.1f}s) -> {save_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
