"""Record one teleop episode to ``recordings/<dataset>/<episode>.rrd`` + catalog.

SO ``record-episode`` equivalent for reBot.

Examples::

    python -m p5_rerun_port.record_episode --fake --dataset cans --task "Pick can" --tag "Good episode" --seconds 5
    python -m p5_rerun_port.record_episode --dataset cans --task "Pick can" --tag "Good episode"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from p5_rerun_port.cameras import CameraStream, fake_frame, open_cameras
from p5_rerun_port.catalog import SEGMENT_TAGS
from p5_rerun_port.config import camera_specs, load_recording_config, resolve_urdf_path
from p5_rerun_port.constants import CAMERA_TO_RERUN, DEFAULT_RECORDINGS_DIR, FOLLOWER, JOINT_NAMES
from p5_rerun_port.hardware import make_session
from p5_rerun_port.takes import (
    begin_recording,
    finish_recording,
    log_image_bgr,
    log_scalars,
    prepare_episode_paths,
    register_take,
    save_trajectory,
    set_timeline,
)
from p5_rerun_port.urdf_log import log_urdf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Record a reBot teleop episode to Rerun (.rrd)")
    ap.add_argument("--dataset", default="cans")
    ap.add_argument("--episode", default=None, help="Default: episode_NN auto-increment")
    ap.add_argument("--task", default="Pick one can and place in taped zone")
    ap.add_argument("--tag", default=SEGMENT_TAGS[0], help=f"One of {list(SEGMENT_TAGS)} (or free-form)")
    ap.add_argument("--seconds", type=float, default=None, help="Stop after N seconds (else Enter)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--fake", action="store_true", help="No hardware; synthetic episode")
    ap.add_argument("--no-teleop", action="store_true", help="Log follower state only (no goal)")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR)
    ap.add_argument("--urdf", type=Path, default=None)
    args = ap.parse_args(argv)

    teleop = not args.no_teleop
    cfg = load_recording_config()
    episode, rrd_path, traj_path = prepare_episode_paths(args.dataset, args.episode, args.recordings_dir)

    rec = begin_recording(
        rrd_path,
        episode=episode,
        dataset=args.dataset,
        task=args.task,
        spawn_viewer=not args.no_viewer,
        save_only=args.no_viewer,
    )
    log_urdf(rec, resolve_urdf_path(args.urdf))
    try:
        import rerun as rr

        rec.log(f"{FOLLOWER}/joint_names", rr.TextDocument(", ".join(JOINT_NAMES)), static=True)
    except Exception:
        pass

    session = make_session(fake=args.fake, teleop=teleop, recording_cfg=cfg)
    session.start()

    cam_streams: list[CameraStream] = []
    fake_cam_specs: list[tuple[str, int, int, int]] = []
    if not args.no_cameras:
        if args.fake:
            fake_cam_specs = [("front", 0, 640, 480), ("side", 1, 640, 480)]
        else:
            cam_streams = open_cameras(camera_specs(cfg))

    frames_root = rrd_path.with_suffix("")  # episode_01/
    frame_dirs = {
        "front": frames_root / "frames" / "front",
        "side": frames_root / "frames" / "side",
    }
    for d in frame_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    period = 1.0 / max(args.fps, 1.0)
    times: list[float] = []
    states: list[list[float]] = []
    actions: list[list[float]] = []
    t0 = time.time()
    print(f"recording: {rrd_path}", flush=True)
    print("press Enter to stop" if args.seconds is None else f"recording for {args.seconds}s...", flush=True)

    stop = False

    def _should_stop() -> bool:
        nonlocal stop
        if stop:
            return True
        if args.seconds is not None and (time.time() - t0) >= args.seconds:
            return True
        return False

    # Non-blocking Enter on Unix; on Windows with --seconds we skip input thread.
    enter_thread = None
    if args.seconds is None:
        import threading

        def _wait_enter() -> None:
            nonlocal stop
            try:
                input()
            except EOFError:
                pass
            stop = True

        enter_thread = threading.Thread(target=_wait_enter, daemon=True)
        enter_thread.start()

    try:
        while not _should_stop():
            loop_t = time.time()
            state, goal = session.read()
            action = goal if goal is not None else state
            set_timeline(rec, loop_t)
            log_scalars(rec, f"{FOLLOWER}/position", state)
            log_scalars(rec, f"{FOLLOWER}/goal", action)

            frame_i = len(times)
            if fake_cam_specs:
                import cv2

                for name, _i, w, h in fake_cam_specs:
                    bgr = fake_frame(w, h, loop_t - t0, name)
                    log_image_bgr(rec, CAMERA_TO_RERUN[name], bgr)
                    cv2.imwrite(str(frame_dirs[name] / f"{frame_i:06d}.jpg"), bgr)
            for cam in cam_streams:
                frame = cam.read()
                if frame is not None:
                    import cv2

                    log_image_bgr(rec, cam.entity, frame)
                    if cam.name in frame_dirs:
                        cv2.imwrite(str(frame_dirs[cam.name] / f"{frame_i:06d}.jpg"), frame)

            times.append(loop_t - t0)
            states.append(list(map(float, state)))
            actions.append(list(map(float, action)))

            sleep = period - (time.time() - loop_t)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        finish_recording(rec, dataset=args.dataset, task=args.task, tag=args.tag)
        session.close()
        for cam in cam_streams:
            cam.close()

    if not times:
        print("FAIL: no frames recorded", flush=True)
        return 1

    save_trajectory(traj_path, times=times, states=states, actions=actions, joint_names=JOINT_NAMES)
    record = register_take(
        dataset=args.dataset,
        episode=episode,
        task=args.task,
        tag=args.tag,
        rrd_path=rrd_path,
        traj_path=traj_path,
        fps=args.fps,
        n_frames=len(times),
        duration_s=times[-1] if times else 0.0,
    )
    print(
        f"OK: {record.episode}  frames={record.n_frames}  tag={record.tag!r}\n"
        f"    rrd={record.rrd_path}\n"
        f"    traj={record.traj_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
