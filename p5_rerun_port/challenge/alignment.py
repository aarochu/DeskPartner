"""Segment-bound Rerun Query extraction and integer-nanosecond latest-at alignment."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from .config import ChallengeConfig


_SCALAR = "rerun.components.Scalar"
_TEXT = "rerun.components.Text"
_MEDIA = "rerun.components.MediaType"
_BLOB = "rerun.components.Blob"
_TASK_TEXT = {
    "single_can": "Pick up one can and place it in the taped sorting zone",
    "two_can": "Pull one of the two cans into the blue-taped recycling zone.",
}
_CAMERA_ENTITIES = {"front": "/camera/cam0", "side": "/camera/cam1"}


class AlignmentError(ValueError):
    """The Query result cannot be bound to one trustworthy episode."""

    def __init__(self, reason_codes: tuple[str, ...], message: str):
        super().__init__(message)
        self.reason_codes = reason_codes


@dataclass(frozen=True)
class AlignedEpisode:
    identity: str
    segment_id: str
    task_key: str
    expected_sample_count: int
    joint_names: tuple[str, ...]
    frame: np.ndarray
    action_time_ns: np.ndarray
    action_present: np.ndarray
    action: np.ndarray
    state_time_ns: np.ndarray
    state_age_ns: np.ndarray
    state_present: np.ndarray
    state: np.ndarray
    camera_time_ns: dict[str, np.ndarray]
    camera_age_ns: dict[str, np.ndarray]
    camera_present: dict[str, np.ndarray]
    # Exact DataFusion projection audit; camera blobs must never appear here.
    query_columns: tuple[str, ...] = ()
    action_dimension_valid: np.ndarray | None = None
    state_dimension_valid: np.ndarray | None = None

    @property
    def action_time_s(self) -> np.ndarray:
        return self.action_time_ns.astype(np.float64) / 1_000_000_000

    @property
    def state_time_s(self) -> np.ndarray:
        return np.where(self.state_present, self.state_time_ns / 1_000_000_000, np.nan)

    @property
    def state_age_s(self) -> np.ndarray:
        return np.where(self.state_present, self.state_age_ns / 1_000_000_000, np.nan)

    @property
    def camera_time_s(self) -> dict[str, np.ndarray]:
        return {
            key: np.where(self.camera_present[key], value / 1_000_000_000, np.nan)
            for key, value in self.camera_time_ns.items()
        }

    @property
    def camera_age_s(self) -> dict[str, np.ndarray]:
        return {
            key: np.where(self.camera_present[key], value / 1_000_000_000, np.nan)
            for key, value in self.camera_age_ns.items()
        }


@dataclass(frozen=True)
class _Stream:
    segment: np.ndarray
    time_ns: np.ndarray
    frame: np.ndarray | None
    values: tuple[Any, ...]
    projected: tuple[str, ...]


def _one(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    while isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1:
        value = value[0]
        if hasattr(value, "as_py"):
            value = value.as_py()
    return value


def _text(value: Any) -> str:
    value = _one(value)
    if not isinstance(value, str) or not value:
        raise ValueError("static text is blank or non-text")
    return value


def _duration_ns(value: Any) -> int:
    raw = getattr(value, "value", None)
    if isinstance(raw, Integral):
        return int(raw)
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, Integral):
        return int(value)
    if hasattr(value, "total_seconds"):
        return round(value.total_seconds() * 1_000_000_000)
    raise ValueError("timeline value is not Arrow duration[ns]")


def _column(schema: Any, entity: str, component: str) -> str:
    names = tuple(
        str(name)
        for name in schema.column_names_for(entity_path=entity, component_type=component)
    )
    if len(names) != 1:
        raise ValueError(f"expected one {component} column for {entity}, found {len(names)}")
    return names[0]


def _filtered(dataset: Any, segment_id: str) -> Any:
    # Keep this at every schema/read boundary. A retained view is not trusted to
    # remain segment-filtered across SDK operations.
    return dataset.filter_segments([segment_id])


def _static_value(dataset: Any, segment_id: str, entity: str, column: str) -> str:
    view = _filtered(dataset, segment_id).filter_contents([entity])
    table = (
        view.reader(index=None)
        .select("rerun_segment_id", column)
        .sort_by("rerun_segment_id")
        .to_arrow_table()
    )
    if len(table) != 1:
        raise ValueError(f"{entity} must have exactly one static row")
    if _text(table["rerun_segment_id"][0]) != segment_id:
        raise ValueError(f"{entity} belongs to another segment")
    return _text(table[column][0])


def _stream(
    dataset: Any,
    segment_id: str,
    entity: str,
    component_column: str,
    *,
    with_frame: bool,
) -> _Stream:
    projected = (
        ("rerun_segment_id", "time", "frame", component_column)
        if with_frame
        else ("rerun_segment_id", "time", component_column)
    )
    reader = _filtered(dataset, segment_id).filter_contents([entity]).reader(
        index="time", fill_latest_at=False
    )
    order = ("rerun_segment_id", "time", "frame") if with_frame else (
        "rerun_segment_id", "time"
    )
    table = reader.select(*projected).sort_by(*order).to_arrow_table()
    segments = np.asarray([_text(table["rerun_segment_id"][i]) for i in range(len(table))])
    times = np.asarray([_duration_ns(table["time"][i]) for i in range(len(table))], dtype=np.int64)
    frames = None
    if with_frame:
        raw_frames = [_one(table["frame"][i]) for i in range(len(table))]
        if any(isinstance(item, bool) or not isinstance(item, Integral) for item in raw_frames):
            raise ValueError(f"{entity} contains a non-integral frame")
        frames = np.asarray(raw_frames, dtype=np.int64)
    values = tuple(table[component_column][i] for i in range(len(table)))
    if set(segments.tolist()) not in ({segment_id}, set()):
        raise ValueError(f"{entity} contains another segment")
    return _Stream(segments, times, frames, values, projected)


def _vector7(value: Any) -> tuple[np.ndarray, bool, bool]:
    """Return vector, component presence, and exact-width status separately."""
    if hasattr(value, "as_py"):
        value = value.as_py()
    if value is None:
        return np.full(7, np.nan), False, True
    # Rerun component cells normally have one instance wrapping seven scalars.
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.full(7, np.nan), True, False
    if vector.size != 7:
        return np.full(7, np.nan), True, False
    return vector, True, True


def _media_present(value: Any) -> bool:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return isinstance(value, str) and bool(value)


def _latest_at(
    action_time_ns: np.ndarray,
    observation_time_ns: np.ndarray,
    max_age_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve latest observations at/before each action with inclusive age."""
    action_time_ns = np.asarray(action_time_ns, dtype=np.int64)
    observation_time_ns = np.asarray(observation_time_ns, dtype=np.int64)
    if not len(observation_time_ns):
        missing = np.full(len(action_time_ns), -1, dtype=np.int64)
        return missing.copy(), missing.copy(), missing.copy(), np.zeros(len(action_time_ns), dtype=bool)
    raw = np.searchsorted(observation_time_ns, action_time_ns, side="right") - 1
    present = raw >= 0
    safe = np.where(present, raw, 0)
    matched = np.where(present, observation_time_ns[safe], -1).astype(np.int64)
    age = np.where(present, action_time_ns - matched, -1).astype(np.int64)
    present &= age <= max_age_ns
    indexes = np.where(present, raw, -1).astype(np.int64)
    return indexes, np.where(present, matched, -1), np.where(present, age, -1), present


def _require_unique_observation_times(
    observation_time_ns: np.ndarray, reason_code: str, stream_name: str
) -> None:
    """Reject ambiguous latest-at sources instead of depending on row order."""
    times = np.asarray(observation_time_ns, dtype=np.int64)
    if len(times) != len(np.unique(times)):
        raise AlignmentError((reason_code,), f"duplicate {stream_name} timestamps")


def _reason_for_exception(reason: str, message: str, error: Exception) -> AlignmentError:
    return AlignmentError((reason,), f"{message}: {error}")


def extract_aligned_episode(
    dataset: Any,
    *,
    identity: str,
    task_key: str,
    expected_sample_count: int,
    config: ChallengeConfig,
) -> AlignedEpisode:
    """Extract one canonical segment without camera blob materialization."""
    segments = tuple(str(segment) for segment in dataset.segment_ids())
    if len(segments) != 1:
        raise AlignmentError(
            ("SEGMENT_COUNT",), f"expected exactly one segment, found {len(segments)}"
        )
    segment_id = segments[0]
    try:
        schema = _filtered(dataset, segment_id).schema()
    except Exception as error:
        raise _reason_for_exception("JOINT_SCHEMA_MISMATCH", "canonical schema unreadable", error)
    try:
        source_column = _column(schema, "/episode/source", _TEXT)
    except Exception as error:
        raise _reason_for_exception("IDENTITY_MISMATCH", "source schema mismatch", error)
    try:
        task_column = _column(schema, "/episode/task", _TEXT)
    except Exception as error:
        raise _reason_for_exception("TASK_MISMATCH", "task schema mismatch", error)
    try:
        joints_column = _column(schema, "/episode/joint_names", _TEXT)
    except Exception as error:
        raise _reason_for_exception("JOINT_SCHEMA_MISMATCH", "joint schema mismatch", error)
    try:
        action_column = _column(schema, "/follower/goal", _SCALAR)
    except Exception as error:
        raise _reason_for_exception("ACTION_MISSING", "action schema mismatch", error)
    try:
        state_column = _column(schema, "/follower/position", _SCALAR)
    except Exception as error:
        raise _reason_for_exception("STATE_MISSING_OR_STALE", "state schema mismatch", error)

    try:
        actual_identity = _static_value(dataset, segment_id, "/episode/source", source_column)
    except Exception as error:
        raise _reason_for_exception("IDENTITY_MISMATCH", "source provenance unreadable", error)
    if actual_identity != identity:
        raise AlignmentError(("IDENTITY_MISMATCH",), "static source differs from inventory")
    try:
        actual_task = _static_value(dataset, segment_id, "/episode/task", task_column)
    except Exception as error:
        raise _reason_for_exception("TASK_MISMATCH", "task provenance unreadable", error)
    if task_key not in _TASK_TEXT or actual_task != _TASK_TEXT[task_key]:
        raise AlignmentError(("TASK_MISMATCH",), "static task differs from inventory")
    try:
        actual_joints = tuple(
            _static_value(dataset, segment_id, "/episode/joint_names", joints_column).split(",")
        )
    except Exception as error:
        raise _reason_for_exception("JOINT_SCHEMA_MISMATCH", "joint provenance unreadable", error)
    if actual_joints != config.joint_names:
        raise AlignmentError(("JOINT_SCHEMA_MISMATCH",), "joint order differs from profile")

    try:
        action_rows = _stream(
            dataset, segment_id, "/follower/goal", action_column, with_frame=True
        )
    except Exception as error:
        raise _reason_for_exception("ACTION_MISSING", "action query failed", error)
    assert action_rows.frame is not None
    if len(action_rows.time_ns) != len(np.unique(action_rows.time_ns)):
        raise AlignmentError(("ACTION_TIME_NONMONOTONIC",), "duplicate action timestamps")
    if len(action_rows.frame) != len(np.unique(action_rows.frame)):
        raise AlignmentError(("FRAME_INDEX_NONCONTIGUOUS",), "duplicate action frames")
    if len(action_rows.time_ns) > 1 and np.any(np.diff(action_rows.time_ns) <= 0):
        raise AlignmentError(("ACTION_TIME_NONMONOTONIC",), "action time is not increasing")
    action = np.full((len(action_rows.values), 7), np.nan, dtype=np.float64)
    action_present = np.zeros(len(action), dtype=bool)
    action_dimension = np.ones(len(action), dtype=bool)
    for index, cell in enumerate(action_rows.values):
        vector, present, dimension = _vector7(cell)
        action[index] = vector
        action_present[index] = present
        action_dimension[index] = dimension

    try:
        state_rows = _stream(
            dataset, segment_id, "/follower/position", state_column, with_frame=False
        )
    except Exception as error:
        raise _reason_for_exception("STATE_MISSING_OR_STALE", "state query failed", error)
    _require_unique_observation_times(
        state_rows.time_ns, "STATE_MISSING_OR_STALE", "state"
    )
    state_vectors = []
    state_source_dimension = []
    state_valid = []
    for cell in state_rows.values:
        vector, present, dimension = _vector7(cell)
        state_vectors.append(vector)
        state_source_dimension.append(dimension)
        state_valid.append(present and dimension)
    state_source = (
        np.asarray(state_vectors, dtype=np.float64)
        if state_vectors
        else np.empty((0, 7), dtype=np.float64)
    )
    indexes, state_time, state_age, state_present = _latest_at(
        action_rows.time_ns, state_rows.time_ns, config.quality.max_state_age_ns
    )
    state = np.full((len(action), 7), np.nan, dtype=np.float64)
    state_dimension = np.ones(len(action), dtype=bool)
    for target, source in enumerate(indexes):
        if source >= 0 and state_valid[source]:
            state[target] = state_source[source]
        else:
            if source >= 0:
                state_dimension[target] = state_source_dimension[source]
            state_present[target] = False
            state_time[target] = -1
            state_age[target] = -1

    camera_time: dict[str, np.ndarray] = {}
    camera_age: dict[str, np.ndarray] = {}
    camera_present: dict[str, np.ndarray] = {}
    projected = list(action_rows.projected) + list(state_rows.projected)
    for key in config.camera_keys:
        entity = _CAMERA_ENTITIES[key]
        try:
            # Blob co-occurrence is a schema invariant established by canonicalization;
            # resolve it but deliberately never include it in the reader projection.
            _column(schema, entity, _BLOB)
            media_column = _column(schema, entity, _MEDIA)
            rows = _stream(dataset, segment_id, entity, media_column, with_frame=False)
        except Exception as error:
            reason = f"CAMERA_{key.upper()}_MISSING_OR_STALE"
            raise _reason_for_exception(reason, f"{key} camera query failed", error)
        reason = f"CAMERA_{key.upper()}_MISSING_OR_STALE"
        _require_unique_observation_times(rows.time_ns, reason, f"{key} camera")
        indexes, matched, ages, present = _latest_at(
            action_rows.time_ns, rows.time_ns, config.quality.max_camera_age_ns
        )
        for target, source in enumerate(indexes):
            if source >= 0 and not _media_present(rows.values[source]):
                present[target] = False
                matched[target] = -1
                ages[target] = -1
        camera_time[key] = matched
        camera_age[key] = ages
        camera_present[key] = present
        projected.extend(rows.projected)

    return AlignedEpisode(
        identity=identity,
        segment_id=segment_id,
        task_key=task_key,
        expected_sample_count=expected_sample_count,
        joint_names=config.joint_names,
        frame=action_rows.frame,
        action_time_ns=action_rows.time_ns,
        action_present=action_present,
        action=action,
        state_time_ns=state_time,
        state_age_ns=state_age,
        state_present=state_present,
        state=state,
        camera_time_ns=camera_time,
        camera_age_ns=camera_age,
        camera_present=camera_present,
        query_columns=tuple(dict.fromkeys(projected)),
        action_dimension_valid=action_dimension,
        state_dimension_valid=state_dimension,
    )


# Concise public alias for callers that describe this operation as alignment.
align_dataset = extract_aligned_episode
