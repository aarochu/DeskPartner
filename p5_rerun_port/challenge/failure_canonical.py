"""Transform authenticated native failure RRDs into canonical episode RRDs."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import math
from numbers import Integral
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

from .canonical import CanonicalArtifact, CanonicalEpisodeWriter
from .config import ChallengeConfig
from .models import EpisodeIdentity, InventoryRow
from p5_rerun_port.rerun_query import open_dataset_server


_SCALAR = "rerun.components.Scalar"
_BLOB = "rerun.components.Blob"
_MEDIA = "rerun.components.MediaType"
_TEXT = "rerun.components.Text"
_ATTEMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_TASKS = {
    "single_can": "Pick up one can and place it in the taped sorting zone",
    "two_can": "Pull one of the two cans into the blue-taped recycling zone.",
}


class IdentityError(ValueError):
    """A native RRD cannot be bound to exactly one inventory identity."""


class _Reject(ValueError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class _Rows:
    segment_id: str
    frames: tuple[int, ...]
    times_s: tuple[float, ...]
    actions: tuple[np.ndarray, ...]
    states: tuple[np.ndarray, ...]


def _rejected(identity: EpisodeIdentity, reason: str, frame_count: int = 0) -> CanonicalArtifact:
    return CanonicalArtifact(identity.canonical, "rejected", None, None, frame_count, (reason,))


def _authorized(row: InventoryRow, config: ChallengeConfig, source_rrd: Path) -> bool:
    matches = [source for source in config.sources if source.repo_id == row.identity.repo_id]
    if len(matches) != 1:
        return False
    source = matches[0]
    attempt_id = row.attempt_id
    if not (
        source.role == row.role == "failure"
        and source.revision == row.identity.revision
        and source.task_key == row.task_key
        and row.episode_index is None
        and isinstance(attempt_id, str)
        and _ATTEMPT_ID.fullmatch(attempt_id)
        and row.identity.source_key == attempt_id
    ):
        return False
    inventory_path = Path(row.source_path)
    allowed = {
        Path("failed_attempts") / attempt_id / "attempt.rrd",
        Path("failed_attempts") / attempt_id / "attempt.partial.rrd",
    }
    if inventory_path.is_absolute() or inventory_path not in allowed:
        return False
    try:
        actual = source_rrd.expanduser().resolve(strict=True)
    except OSError:
        return False
    return actual.is_file() and actual.parts[-len(inventory_path.parts) :] == inventory_path.parts


def _column(schema: Any, entity: str, component: str) -> str | None:
    names = tuple(str(name) for name in schema.column_names_for(
        entity_path=entity, component_type=component
    ))
    return names[0] if len(names) == 1 else None


def _table(
    dataset: Any,
    contents: list[str],
    index: str | None,
    columns: list[str],
) -> Any:
    query = dataset.filter_contents(contents).reader(index=index).select(*columns)
    if index is None:
        query = query.sort_by("rerun_segment_id")
    else:
        query = query.sort_by("rerun_segment_id", "attempt_frame", "attempt_time")
    return query.to_arrow_table()


def _cell(table: Any, column: str, row: int) -> Any:
    value = table[column][row]
    return value.as_py() if hasattr(value, "as_py") else value


def _one(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    while isinstance(value, (list, tuple)) or (
        isinstance(value, np.ndarray) and value.dtype == object
    ):
        if len(value) != 1:
            raise ValueError("component cell is not scalar width")
        value = value[0]
    return value


def _segment(value: Any) -> str:
    value = _one(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid segment")
    return value.strip()


def _frame(value: Any) -> int:
    value = _one(value)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("invalid frame")
    return int(value)


def _duration_ns(table: Any, row: int) -> int:
    scalar = table["attempt_time"][row]
    raw = getattr(scalar, "value", None)
    if isinstance(raw, Integral):
        return int(raw)
    value = scalar.as_py() if hasattr(scalar, "as_py") else scalar
    if hasattr(value, "total_seconds"):
        return round(value.total_seconds() * 1_000_000_000)
    raise ValueError("attempt_time is not duration[ns]")


def _scalar(value: Any) -> float:
    value = _one(value)
    if isinstance(value, bool):
        raise ValueError("boolean scalar")
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.number):
        raise ValueError("component is not one scalar")
    number = float(array.item())
    if not math.isfinite(number):
        raise ValueError("non-finite scalar")
    return number


def _timeline(
    table: Any,
    segment_id: str,
    frame_count: int,
    fps: int,
    frame_reason: str,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    try:
        segments = tuple(_segment(_cell(table, "rerun_segment_id", i)) for i in range(len(table)))
        frames = tuple(_frame(_cell(table, "attempt_frame", i)) for i in range(len(table)))
        times_ns = tuple(_duration_ns(table, i) for i in range(len(table)))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise _Reject(frame_reason, "invalid Query timeline") from error
    if not segments:
        raise _Reject(frame_reason, "Query timeline contains no rows")
    if set(segments) != {segment_id}:
        raise IdentityError("queried rows do not map to exactly one segment")
    if len(frames) != len(set(frames)):
        raise _Reject("FRAME_INDEX_DUPLICATE", "duplicate attempt_frame")
    if frames != tuple(range(len(frames))):
        raise _Reject(frame_reason, "attempt_frame is not contiguous from zero")
    if len(frames) != frame_count:
        raise _Reject("FRAME_COUNT_MISMATCH", "Query row count differs from inventory")
    for frame, timestamp_ns in zip(frames, times_ns, strict=True):
        expected = round(frame * 1_000_000_000 / fps)
        if abs(timestamp_ns - expected) > 10_000:
            raise _Reject("TIMESTAMP_MISMATCH", "attempt_time differs from frame/FPS")
    return frames, tuple(value / 1_000_000_000 for value in times_ns)


def _static_identity(dataset: Any, schema: Any, row: InventoryRow, segment_id: str) -> None:
    column = _column(schema, "/attempt/metadata", _TEXT)
    if column is None:
        raise _Reject("STATIC_IDENTITY_MISSING", "missing unique static metadata text")
    table = _table(dataset, ["/attempt/metadata"], None, ["rerun_segment_id", column])
    if len(table) != 1:
        raise _Reject("STATIC_IDENTITY_MISSING", "metadata must have one static row")
    try:
        if _segment(_cell(table, "rerun_segment_id", 0)) != segment_id:
            raise IdentityError("static metadata maps to another segment")
        document = json.loads(_one(_cell(table, column, 0)))
        attempt_id = document["attempt_id"]
    except IdentityError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise _Reject("STATIC_IDENTITY_MISSING", "metadata has no usable attempt_id") from error
    if attempt_id != row.attempt_id:
        raise _Reject("STATIC_IDENTITY_MISMATCH", "static attempt_id differs from inventory")


def _scalar_columns(
    schema: Any, config: ChallengeConfig
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    actions: list[str] = []
    states: list[str] = []
    for joint in config.joint_names:
        column = _column(schema, f"/action/{joint}/pos", _SCALAR)
        if column is None:
            raise _Reject("MISSING_ACTION_JOINT", f"missing unique action scalar for {joint}")
        actions.append(column)
    for joint in config.joint_names:
        column = _column(schema, f"/observation/{joint}/pos", _SCALAR)
        if column is None:
            raise _Reject("MISSING_STATE_JOINT", f"missing unique state scalar for {joint}")
        states.append(column)
    return tuple(actions), tuple(states)


def _dynamic_rows(
    dataset: Any,
    config: ChallengeConfig,
    row: InventoryRow,
    segment_id: str,
    action_columns: tuple[str, ...],
    state_columns: tuple[str, ...],
) -> _Rows:
    entities = [
        *(f"/action/{joint}/pos" for joint in config.joint_names),
        *(f"/observation/{joint}/pos" for joint in config.joint_names),
    ]
    table = _table(
        dataset,
        entities,
        "attempt_frame",
        ["rerun_segment_id", "attempt_frame", "attempt_time", *action_columns, *state_columns],
    )
    frames, times_s = _timeline(
        table, segment_id, row.frame_count, config.fps, "FRAME_INDEX_NONCONTIGUOUS"
    )
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for index in range(len(table)):
        try:
            actions.append(np.asarray(
                [_scalar(_cell(table, column, index)) for column in action_columns],
                dtype=np.float64,
            ))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise _Reject("INVALID_ACTION", "action row is not seven finite scalars") from error
        try:
            states.append(np.asarray(
                [_scalar(_cell(table, column, index)) for column in state_columns],
                dtype=np.float64,
            ))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise _Reject("INVALID_STATE", "state row is not seven finite scalars") from error
    return _Rows(segment_id, frames, times_s, tuple(actions), tuple(states))


def _camera_columns(schema: Any, entity: str, name: str) -> tuple[str, str]:
    blob = _column(schema, entity, _BLOB)
    media = _column(schema, entity, _MEDIA)
    if blob is None or media is None:
        raise _Reject(f"MISSING_{name.upper()}_CAMERA", f"missing {name} Blob/MediaType")
    return blob, media


def _media_type(value: Any) -> str:
    value = _one(value)
    if not isinstance(value, str):
        raise ValueError("media type is not text")
    return value


def _blob(value: Any) -> bytes:
    if hasattr(value, "as_py"):
        value = value.as_py()
    # Blob is list<list<uint8>>: one component instance containing its bytes.
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, np.ndarray) and value.dtype == np.uint8:
        return value.tobytes()
    if isinstance(value, list) and all(
        isinstance(item, int) and 0 <= item <= 255 for item in value
    ):
        return bytes(value)
    raise ValueError("camera blob is not bytes")


def _camera_rows(
    dataset: Any,
    config: ChallengeConfig,
    row: InventoryRow,
    dynamic: _Rows,
    entity: str,
    name: str,
    blob_column: str,
    media_column: str,
) -> tuple[np.ndarray, ...]:
    missing = f"MISSING_{name.upper()}_CAMERA"
    invalid = f"INVALID_{name.upper()}_CAMERA"

    # Coverage intentionally omits the Blob column so JPEG payloads are not materialized.
    coverage = _table(
        dataset,
        [entity],
        "attempt_frame",
        ["rerun_segment_id", "attempt_frame", "attempt_time", media_column],
    )
    coverage_frames, _ = _timeline(
        coverage, dynamic.segment_id, row.frame_count, config.fps, missing
    )
    try:
        media_types = tuple(
            _media_type(_cell(coverage, media_column, i))
            for i in range(len(coverage))
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _Reject(missing, f"{name} media coverage is incomplete") from error
    if coverage_frames != dynamic.frames or any(value != "image/jpeg" for value in media_types):
        raise _Reject(invalid, f"{name} is not complete JPEG coverage")

    # The transformation materializes only this camera's Query-returned JPEG bytes.
    table = _table(
        dataset,
        [entity],
        "attempt_frame",
        ["rerun_segment_id", "attempt_frame", "attempt_time", media_column, blob_column],
    )
    frames, _ = _timeline(table, dynamic.segment_id, row.frame_count, config.fps, missing)
    if frames != dynamic.frames:
        raise _Reject(missing, f"{name} frames do not align to actions")
    images: list[np.ndarray] = []
    for index in range(len(table)):
        try:
            if _media_type(_cell(table, media_column, index)) != "image/jpeg":
                raise ValueError("camera media type is not image/jpeg")
            encoded = _blob(_cell(table, blob_column, index))
            with Image.open(io.BytesIO(encoded)) as image:
                if image.format != "JPEG":
                    raise ValueError("camera bytes are not JPEG")
                images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise _Reject(invalid, f"{name} contains an invalid JPEG") from error
    return tuple(images)


def _transform(
    row: InventoryRow,
    config: ChallengeConfig,
    source_rrd: Path,
    output: Path,
) -> CanonicalArtifact:
    with open_dataset_server(
        f"native-{row.identity.source_key}", rrd_paths=[source_rrd]
    ) as dataset:
        segment_ids = tuple(str(segment) for segment in dataset.segment_ids())
        if len(segment_ids) != 1:
            raise IdentityError(
                f"native failure RRD must contain exactly one segment, found {len(segment_ids)}"
            )
        segment_id = segment_ids[0]
        schema = dataset.schema()
        _static_identity(dataset, schema, row, segment_id)
        action_columns, state_columns = _scalar_columns(schema, config)
        dynamic = _dynamic_rows(
            dataset, config, row, segment_id, action_columns, state_columns
        )
        front_columns = _camera_columns(schema, "/observation/front", "front")
        side_columns = _camera_columns(schema, "/observation/side", "side")
        front = _camera_rows(
            dataset, config, row, dynamic, "/observation/front", "front", *front_columns
        )
        side = _camera_rows(
            dataset, config, row, dynamic, "/observation/side", "side", *side_columns
        )

    try:
        writer = CanonicalEpisodeWriter(output, row.identity, _TASKS[row.task_key], config.fps)
    except Exception:
        return _rejected(row.identity, "WRITER_INIT_FAILED")
    try:
        for index, frame in enumerate(dynamic.frames):
            writer.add(
                frame,
                dynamic.times_s[index],
                dynamic.actions[index],
                dynamic.states[index],
                front[index],
                side[index],
            )
        return writer.finish()
    except Exception:
        processed = writer.frame_count
        writer.abort()
        return _rejected(row.identity, "MATERIALIZATION_FAILED", processed)


def materialize_failure(
    row: InventoryRow,
    config: ChallengeConfig,
    source_rrd: Path,
    output: Path,
) -> CanonicalArtifact:
    """Query one authenticated native failure RRD into a label-blind canonical RRD."""

    source_rrd = Path(source_rrd)
    if not _authorized(row, config, source_rrd):
        return _rejected(row.identity, "SOURCE_LOCK_MISMATCH")
    try:
        return _transform(row, config, source_rrd, Path(output))
    except IdentityError:
        raise
    except _Reject as error:
        return _rejected(row.identity, error.reason)
    except Exception:
        return _rejected(row.identity, "SOURCE_QUERY_FAILED")
