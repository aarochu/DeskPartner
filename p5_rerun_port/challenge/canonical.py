"""Materialize revision-locked LeRobot success episodes as canonical RRDs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Literal
import uuid

import numpy as np

from .config import ChallengeConfig
from .models import EpisodeIdentity, InventoryRow


_APPLICATION_ID = "deskpartner-rerun-query"
_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)
_TASKS = {
    "single_can": "Pick up one can and place it in the taped sorting zone",
    "two_can": "Pull one of the two cans into the blue-taped recycling zone.",
}
_TIMESTAMP_TOLERANCE_S = 1e-5


@dataclass(frozen=True)
class CanonicalArtifact:
    """The fail-closed result of one canonical materialization attempt."""

    identity: str
    status: Literal["ready", "rejected"]
    rrd_path: Path | None
    sha256: str | None
    frame_count: int
    reason_codes: tuple[str, ...]


class _CanonicalReject(ValueError):
    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


def _rerun_module():
    import rerun as rr

    return rr


def _lerobot_dataset_class():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def _episode_cache_root(identity: EpisodeIdentity) -> Path:
    """Return an external, per-episode cache root without embedding huge inputs in Git."""

    base = Path(
        os.environ.get(
            "DESKPARTNER_LEROBOT_CACHE",
            Path.home() / ".cache" / "deskpartner" / "rerun-query" / "lerobot",
        )
    ).expanduser()
    digest = hashlib.sha256(identity.canonical.encode("utf-8")).hexdigest()
    return base / digest


def _recording_id(identity: EpisodeIdentity) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", identity.canonical).strip("_")
    digest = hashlib.sha256(identity.canonical.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:96]}-{digest}"


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.rrd")


def _rerun_executable() -> str | None:
    beside_python = Path(sys.executable).with_name("rerun")
    if beside_python.is_file():
        return str(beside_python)
    return shutil.which("rerun")


def _verify_rrd(path: Path) -> bool:
    executable = _rerun_executable()
    if executable is None:
        return False
    try:
        result = subprocess.run(
            [executable, "rrd", "verify", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def _disconnect_recording(recording: Any) -> None:
    recording.disconnect()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any) -> int | float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"expected scalar, got {array.shape}")
    return array.item()


def _vector7(value: Any) -> np.ndarray:
    raw = _numpy(value)
    if (
        raw.shape != (7,)
        or not np.issubdtype(raw.dtype, np.number)
        or raw.dtype == np.bool_
    ):
        raise ValueError("expected finite (7,) vector")
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("expected finite (7,) vector")
    return array


def _rgb_u8(image: Any) -> np.ndarray:
    array = _numpy(image)
    if array.ndim != 3:
        raise ValueError(f"expected HWC or CHW RGB, got {array.shape}")
    if array.shape[-1] == 3:
        pass
    elif array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    else:
        raise ValueError(f"expected HWC RGB, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("non-finite image")
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError("floating RGB image must stay within [0, 1]")
        array = np.rint(array * 255.0).astype(np.uint8)
    elif array.dtype == np.uint8:
        array = array.astype(np.uint8, copy=False)
    else:
        raise ValueError(f"expected float or uint8 RGB image, got {array.dtype}")
    return np.ascontiguousarray(array)


def _integral_scalar(value: Any) -> int:
    scalar = _scalar(value)
    if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, Integral):
        raise ValueError(f"expected integral scalar, got {type(scalar).__name__}")
    return int(scalar)


def _rejected(identity: EpisodeIdentity, frame_count: int, reason_code: str) -> CanonicalArtifact:
    return CanonicalArtifact(
        identity=identity.canonical,
        status="rejected",
        rrd_path=None,
        sha256=None,
        frame_count=frame_count,
        reason_codes=(reason_code,),
    )


class CanonicalEpisodeWriter:
    """Write one canonical recording, verify it, then publish it atomically."""

    def __init__(self, path: Path, identity: EpisodeIdentity, task: str, fps: int):
        self.path = Path(path)
        self.identity = identity
        self.task = task
        self.fps = fps
        self.frame_count = 0
        self._closed = False
        self._recording: Any | None = None
        self._temporary_path = _temporary_sibling(self.path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rr = _rerun_module()
            self._recording = rr.RecordingStream(
                _APPLICATION_ID,
                recording_id=_recording_id(identity),
            )
            self._recording.set_sinks(rr.FileSink(str(self._temporary_path)))
            self._log_static_documents(rr)
        except Exception:
            self.abort()
            raise

    def _log_static_documents(self, rr: Any) -> None:
        static_documents = {
            "/episode/source": self.identity.canonical,
            "/episode/repo_id": self.identity.repo_id,
            "/episode/revision": self.identity.revision,
            "/episode/source_index": self.identity.source_key,
            "/episode/task": self.task,
            "/episode/fps": str(self.fps),
            "/episode/joint_names": ",".join(_JOINT_NAMES),
        }
        for entity, text in static_documents.items():
            self._recording.log(entity, rr.TextDocument(text), static=True)

    def add(
        self,
        frame: int,
        timestamp_s: float,
        action: np.ndarray,
        state: np.ndarray,
        front_rgb: np.ndarray,
        side_rgb: np.ndarray,
    ) -> None:
        if self._closed:
            raise RuntimeError("canonical writer is already closed")
        if frame != self.frame_count:
            raise ValueError(f"frame {frame} is not contiguous after {self.frame_count - 1}")
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp must be finite")

        action = _vector7(action)
        state = _vector7(state)
        front_rgb = _rgb_u8(front_rgb)
        side_rgb = _rgb_u8(side_rgb)

        rr = _rerun_module()
        self._recording.set_time("frame", sequence=frame)
        self._recording.set_time("time", duration=timestamp_s)
        self._recording.log("/follower/goal", rr.Scalars(action))
        self._recording.log("/follower/position", rr.Scalars(state))
        self._recording.log("/camera/cam0", rr.Image(front_rgb).compress(jpeg_quality=85))
        self._recording.log("/camera/cam1", rr.Image(side_rgb).compress(jpeg_quality=85))
        self.frame_count += 1

    def abort(self) -> None:
        try:
            if not self._closed and self._recording is not None:
                _disconnect_recording(self._recording)
        except Exception:
            pass
        finally:
            self._closed = True
            self._cleanup_temporary()

    def _cleanup_temporary(self) -> None:
        try:
            self._temporary_path.unlink(missing_ok=True)
        except OSError:
            pass

    def finish(self) -> CanonicalArtifact:
        if self._closed:
            raise RuntimeError("canonical writer is already closed")
        try:
            _disconnect_recording(self._recording)
        except Exception:
            self._closed = True
            self._cleanup_temporary()
            return _rejected(self.identity, self.frame_count, "RRD_DISCONNECT_FAILED")
        self._closed = True

        try:
            verified = _verify_rrd(self._temporary_path)
        except Exception:
            verified = False
        if not verified:
            self._cleanup_temporary()
            return _rejected(self.identity, self.frame_count, "RRD_VERIFY_FAILED")

        try:
            sha256 = _sha256(self._temporary_path)
        except Exception:
            self._cleanup_temporary()
            return _rejected(self.identity, self.frame_count, "RRD_HASH_FAILED")
        try:
            os.replace(self._temporary_path, self.path)
        except Exception:
            self._cleanup_temporary()
            return _rejected(self.identity, self.frame_count, "RRD_PUBLISH_FAILED")
        return CanonicalArtifact(
            identity=self.identity.canonical,
            status="ready",
            rrd_path=self.path,
            sha256=sha256,
            frame_count=self.frame_count,
            reason_codes=(),
        )


def _validate_feature_names(dataset: Any, config: ChallengeConfig) -> None:
    try:
        action_names = tuple(dataset.features["action"]["names"])
        state_names = tuple(dataset.features["observation.state"]["names"])
    except (KeyError, TypeError, AttributeError) as error:
        raise _CanonicalReject("FEATURE_SCHEMA_MISMATCH", "missing action/state feature names") from error

    def normalized(names: tuple[Any, ...]) -> tuple[str | None, ...]:
        return tuple(name[:-4] if isinstance(name, str) and name.endswith(".pos") else None for name in names)

    if normalized(action_names) != config.joint_names or normalized(state_names) != config.joint_names:
        raise _CanonicalReject(
            "FEATURE_SCHEMA_MISMATCH",
            "action/state feature order does not match the authenticated profile",
        )


def _row_matches_source_lock(row: InventoryRow, config: ChallengeConfig) -> bool:
    matches = [source for source in config.sources if source.repo_id == row.identity.repo_id]
    if len(matches) != 1:
        return False
    source = matches[0]
    return (
        row.role == "success"
        and source.role == "success"
        and row.identity.revision == source.revision
        and row.task_key == source.task_key
    )


def materialize_success(
    row: InventoryRow,
    config: ChallengeConfig,
    output: Path,
) -> CanonicalArtifact:
    """Convert one authoritative success row into a verified canonical RRD."""

    if not _row_matches_source_lock(row, config):
        return _rejected(row.identity, 0, "SOURCE_LOCK_MISMATCH")
    if (
        row.episode_index is None
        or isinstance(row.episode_index, bool)
        or not isinstance(row.episode_index, Integral)
    ):
        return _rejected(row.identity, 0, "MISSING_EPISODE_INDEX")
    if row.identity.source_key != str(row.episode_index):
        return _rejected(row.identity, 0, "SOURCE_IDENTITY_MISMATCH")

    try:
        dataset_class = _lerobot_dataset_class()
        dataset = dataset_class(
            repo_id=row.identity.repo_id,
            root=_episode_cache_root(row.identity),
            episodes=[row.episode_index],
            revision=row.identity.revision,
            force_cache_sync=True,
            download_videos=True,
            video_backend="pyav",
        )
    except Exception:
        return _rejected(row.identity, 0, "SOURCE_READ_FAILED")

    try:
        if getattr(dataset, "fps", None) != config.fps:
            raise _CanonicalReject("FPS_MISMATCH", "dataset FPS does not match the source lock")
        _validate_feature_names(dataset, config)
        if len(dataset) != row.frame_count:
            raise _CanonicalReject("FRAME_COUNT_MISMATCH", "decoded frame count does not match inventory")
    except _CanonicalReject as error:
        return _rejected(row.identity, 0, error.reason_code)
    except Exception:
        return _rejected(row.identity, 0, "SOURCE_READ_FAILED")

    expected_task = _TASKS[row.task_key]
    try:
        writer = CanonicalEpisodeWriter(Path(output), row.identity, expected_task, config.fps)
    except Exception:
        return _rejected(row.identity, 0, "WRITER_INIT_FAILED")
    try:
        for local_row in range(len(dataset)):
            sample = dataset[local_row]
            try:
                episode_index = _integral_scalar(sample["episode_index"])
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise _CanonicalReject("EPISODE_INDEX_MISMATCH", "invalid sample episode_index") from error
            if episode_index != row.episode_index:
                raise _CanonicalReject("EPISODE_INDEX_MISMATCH", "sample episode_index differs from inventory")

            try:
                frame = _integral_scalar(sample["frame_index"])
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise _CanonicalReject("FRAME_INDEX_NONCONTIGUOUS", "invalid sample frame_index") from error
            if frame != local_row:
                raise _CanonicalReject("FRAME_INDEX_NONCONTIGUOUS", "sample frame_index is not contiguous")

            try:
                timestamp_s = float(_scalar(sample["timestamp"]))
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise _CanonicalReject("TIMESTAMP_MISMATCH", "invalid sample timestamp") from error
            expected_timestamp = frame / config.fps
            if not math.isfinite(timestamp_s) or abs(timestamp_s - expected_timestamp) > _TIMESTAMP_TOLERANCE_S:
                raise _CanonicalReject("TIMESTAMP_MISMATCH", "sample timestamp differs from frame/FPS")
            if sample.get("task") != expected_task:
                raise _CanonicalReject("TASK_MISMATCH", "sample task differs from inventory task")

            try:
                action = _vector7(sample["action"])
            except (KeyError, TypeError, ValueError) as error:
                raise _CanonicalReject("INVALID_ACTION", "sample action is not a finite seven-vector") from error
            try:
                state = _vector7(sample["observation.state"])
            except (KeyError, TypeError, ValueError) as error:
                raise _CanonicalReject("INVALID_STATE", "sample state is not a finite seven-vector") from error
            try:
                front_rgb = _rgb_u8(sample["observation.images.front"])
            except (KeyError, TypeError, ValueError) as error:
                raise _CanonicalReject("INVALID_FRONT_IMAGE", "front image is not finite RGB") from error
            try:
                side_rgb = _rgb_u8(sample["observation.images.side"])
            except (KeyError, TypeError, ValueError) as error:
                raise _CanonicalReject("INVALID_SIDE_IMAGE", "side image is not finite RGB") from error

            writer.add(frame, timestamp_s, action, state, front_rgb, side_rgb)
        return writer.finish()
    except _CanonicalReject as error:
        processed = writer.frame_count
        writer.abort()
        return _rejected(row.identity, processed, error.reason_code)
    except Exception:
        processed = writer.frame_count
        writer.abort()
        return _rejected(row.identity, processed, "MATERIALIZATION_FAILED")
