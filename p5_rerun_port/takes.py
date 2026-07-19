"""Begin/finish Rerun takes and write trajectory sidecars for export/replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from p5_rerun_port.catalog import EpisodeRecord, episode_rrd_path, sanitize_name, upsert_episode
from p5_rerun_port.constants import APP_ID, DEFAULT_CATALOG, DEFAULT_RECORDINGS_DIR, ROBOT_TYPE


def _import_rerun():
    try:
        import rerun as rr
    except ImportError as exc:
        raise SystemExit(
            "FAIL: rerun-sdk not installed. pip install 'rerun-sdk>=0.22' "
            "(see p5_rerun_port/README.md)"
        ) from exc
    return rr


def begin_recording(
    path: Path,
    *,
    episode: str,
    dataset: str,
    task: str,
    spawn_viewer: bool = True,
    save_only: bool = False,
) -> Any:
    """Create a RecordingStream writing to ``path`` (and optionally spawning a viewer)."""
    rr = _import_rerun()
    path.parent.mkdir(parents=True, exist_ok=True)

    blueprint = None
    if spawn_viewer and not save_only:
        try:
            from p5_rerun_port.viewer_blueprint import build_hackathon_blueprint

            blueprint = build_hackathon_blueprint()
        except Exception as exc:
            # Recording must remain usable if a particular SDK build lacks a
            # blueprint feature. The FileSink is the durable source of truth.
            print(f"WARN: Viewer blueprint unavailable ({exc}); using automatic layout", flush=True)

    # Prefer modern multi-sink API; fall back to init+save for older SDKs.
    try:
        rec = rr.RecordingStream(APP_ID, recording_id=f"{sanitize_name(dataset)}-{path.stem}")
        if spawn_viewer and not save_only and hasattr(rec, "spawn"):
            try:
                # Spawn without connecting first: set_sinks below then tees the
                # stream to both the Viewer and the durable .rrd FileSink.
                rec.spawn(
                    connect=False,
                    hide_welcome_screen=True,
                    default_blueprint=blueprint,
                )
            except Exception as exc:
                print(f"WARN: Rerun Viewer did not start ({exc}); recording to file", flush=True)
        sinks: list[Any] = [rr.FileSink(str(path))]
        if spawn_viewer and not save_only:
            try:
                sinks.insert(0, rr.GrpcSink())
            except Exception:
                pass
        rec.set_sinks(*sinks, default_blueprint=blueprint)
        if blueprint is not None and hasattr(rec, "send_blueprint"):
            try:
                rec.send_blueprint(blueprint, make_active=True, make_default=True)
            except Exception as exc:
                print(f"WARN: Viewer blueprint was not activated ({exc})", flush=True)
        if hasattr(rec, "send_recording_name"):
            rec.send_recording_name(episode)
        if hasattr(rec, "send_property"):
            try:
                rec.send_property("episode", rr.AnyValues(dataset=dataset, task=task, tag=""))
            except Exception:
                pass
        rec.log("/task", rr.TextDocument(task or ""), static=True)
        return rec
    except Exception:
        rr.init(APP_ID, spawn=spawn_viewer and not save_only)
        rr.save(str(path))
        if blueprint is not None and hasattr(rr, "send_blueprint"):
            rr.send_blueprint(blueprint, make_active=True, make_default=True)
        rr.log("/task", rr.TextDocument(task or ""), static=True)
        return rr


def set_timeline(rec: Any, t_s: float) -> None:
    rr = _import_rerun()
    if hasattr(rec, "set_time"):
        try:
            rec.set_time("time", timestamp=t_s)
            return
        except TypeError:
            pass
    if hasattr(rec, "set_time_seconds"):
        rec.set_time_seconds("time", t_s)
        return
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("time", t_s)


def log_scalars(rec: Any, entity: str, values: list[float] | np.ndarray) -> None:
    rr = _import_rerun()
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if hasattr(rr, "Scalars"):
        rec.log(entity, rr.Scalars(arr))
    else:
        # Very old SDKs
        for i, v in enumerate(arr):
            rec.log(f"{entity}/{i}", rr.Scalar(float(v)))


def log_image_bgr(rec: Any, entity: str, bgr: np.ndarray, jpeg_quality: int = 75) -> None:
    rr = _import_rerun()
    rgb = bgr[:, :, ::-1].copy()
    try:
        rec.log(entity, rr.Image(rgb).compress(jpeg_quality=jpeg_quality))
    except Exception:
        rec.log(entity, rr.Image(rgb))


def finish_recording(rec: Any, *, dataset: str, task: str, tag: str) -> None:
    rr = _import_rerun()
    if hasattr(rec, "send_property"):
        try:
            rec.send_property("episode", rr.AnyValues(dataset=dataset, task=task, tag=tag))
        except Exception:
            pass
    # Drop file sink / flush
    if hasattr(rec, "disconnect"):
        rec.disconnect()
    elif hasattr(rr, "disconnect"):
        rr.disconnect()


def save_trajectory(
    traj_path: Path,
    *,
    times: list[float],
    states: list[list[float]],
    actions: list[list[float]],
    joint_names: tuple[str, ...] | list[str],
) -> None:
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        traj_path,
        times=np.asarray(times, dtype=np.float64),
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        joint_names=np.asarray(list(joint_names)),
    )


def register_take(
    *,
    dataset: str,
    episode: str,
    task: str,
    tag: str,
    rrd_path: Path,
    traj_path: Path,
    fps: float,
    n_frames: int,
    duration_s: float,
    catalog_path: Path = DEFAULT_CATALOG,
) -> EpisodeRecord:
    meta_path = rrd_path.with_suffix(".meta.json")
    record = EpisodeRecord(
        dataset=sanitize_name(dataset),
        episode=sanitize_name(episode),
        task=task,
        tag=tag,
        rrd_path=str(rrd_path.resolve()),
        traj_path=str(traj_path.resolve()),
        meta_path=str(meta_path.resolve()),
        fps=fps,
        n_frames=n_frames,
        duration_s=duration_s,
        robot_type=ROBOT_TYPE,
    )
    upsert_episode(record, catalog_path=catalog_path)
    return record


def prepare_episode_paths(
    dataset: str,
    episode: str | None,
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
) -> tuple[str, Path, Path]:
    from p5_rerun_port.catalog import next_episode_name

    ep_name = episode or next_episode_name(recordings_dir, dataset)
    rrd = episode_rrd_path(recordings_dir, dataset, ep_name)
    # Final episode id is the rrd stem (may gain -2 suffix on collision).
    return rrd.stem, rrd, rrd.with_suffix(".traj.npz")
