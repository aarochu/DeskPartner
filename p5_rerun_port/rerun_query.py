"""Rerun Query / Catalog dataframe helpers for reBot recordings.

Uses ``rr.server.Server`` + dataset ``reader()`` (DataFusion) — the same Query API
surface as the SO-101 refine step — over local ``recordings/<dataset>/*.rrd``.

Does not open serial ports or the operator GUI. Run only after recording stops.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from p5_rerun_port.catalog import EpisodeRecord, query_episodes, sanitize_name
from p5_rerun_port.constants import DEFAULT_RECORDINGS_DIR, FOLLOWER, JOINT_NAMES

os.environ.setdefault("RERUN_INSECURE_SKIP_HOST_CHECK", "1")


def _import_rr():
    try:
        import rerun as rr
    except ImportError as exc:
        raise SystemExit(
            "FAIL: rerun-sdk required for Query API. pip install 'rerun-sdk>=0.28'"
        ) from exc
    return rr


def dataset_rrd_paths(recordings_dir: Path, dataset: str) -> list[Path]:
    """Return canonical episode recordings registered by a metadata sidecar."""
    directory = recordings_dir / sanitize_name(dataset)
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.glob("*.rrd")
        if not path.stem.endswith("_replay")
        and path.with_suffix(".meta.json").is_file()
    )


def rrd_paths_for_records(records: list[EpisodeRecord]) -> list[Path]:
    """Resolve authoritative recording paths in catalog selection order."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for record in records:
        if not record.rrd_path:
            raise SystemExit(
                f"FAIL: recording for {record.dataset}/{record.episode} has no rrd_path"
            )
        path = Path(record.rrd_path).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(
                f"FAIL: recording for {record.dataset}/{record.episode} does not exist: {path}"
            )
        if path in seen:
            raise SystemExit(f"FAIL: duplicate recording path selected: {path}")
        seen.add(path)
        paths.append(path)
    return paths


@contextmanager
def open_dataset_server(
    dataset: str,
    *,
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
    rrd_paths: list[Path] | None = None,
) -> Iterator[Any]:
    """Spin up an in-process Rerun catalog server for one dataset of .rrd files."""
    rr = _import_rr()
    paths = rrd_paths if rrd_paths is not None else dataset_rrd_paths(recordings_dir, dataset)
    if not paths:
        raise SystemExit(
            f"FAIL: no .rrd files for dataset {dataset!r} under {recordings_dir / sanitize_name(dataset)}"
        )
    name = sanitize_name(dataset)
    with rr.server.Server(datasets={name: [str(p.resolve()) for p in paths]}) as server:
        client = server.client()
        yield client.get_dataset(name)


def _normalize_index_name(raw: Any) -> str:
    """Turn SDK index objects like ``Index(timeline:time)`` into ``time``."""
    text = str(raw)
    if "timeline:" in text:
        # Index(timeline:time) or similar
        inner = text.split("timeline:", 1)[1]
        return inner.rstrip(")").strip()
    return text


def list_schema(dataset_entry: Any) -> dict[str, list[str]]:
    schema = dataset_entry.schema()
    indexes: list[str] = []
    components: list[str] = []
    try:
        indexes = [_normalize_index_name(c) for c in schema.index_columns()]
    except Exception:
        pass
    try:
        components = [str(c) for c in schema.component_columns()]
    except Exception:
        pass
    entity_paths: list[str] = []
    try:
        entity_paths = [str(p) for p in schema.entity_paths()]
    except Exception:
        # Fall back: parse from component column names when available
        for col_name in components:
            if ":" in col_name:
                entity_paths.append(col_name.split(":", 1)[0])
        entity_paths = sorted(set(entity_paths))
    return {"indexes": indexes, "components": components, "entities": entity_paths}


def pick_timeline(dataset_entry: Any, preferred: str | None = None) -> str | None:
    indexes = list_schema(dataset_entry)["indexes"]
    if preferred:
        pref = _normalize_index_name(preferred)
        if pref in indexes:
            return pref
    for name in ("time", "real_time", "log_time", "frame_nr", "timeline"):
        if name in indexes:
            return name
    return indexes[0] if indexes else None


def reader_to_pandas(dataset_entry: Any, *, index: str | None, contents: list[str] | None = None):
    view = dataset_entry
    if contents:
        view = view.filter_contents(contents)
    df_api = view.reader(index=index)
    if hasattr(df_api, "to_pandas"):
        return df_api.to_pandas()
    if hasattr(df_api, "to_arrow_table"):
        import pandas as pd

        return df_api.to_arrow_table().to_pandas()
    raise SystemExit("FAIL: dataframe reader has no to_pandas/to_arrow_table")


def find_scalar_columns(df, entity_substr: str) -> list[str]:
    cols = []
    for name in df.columns:
        s = str(name)
        if entity_substr in s and ("Scalar" in s or "scalars" in s.lower()):
            cols.append(s)
    if cols:
        return cols
    # Broader match
    for name in df.columns:
        s = str(name)
        if entity_substr in s and s not in ("rerun_segment_id", "log_time", "log_tick"):
            cols.append(s)
    return cols


def stack_scalar_column(series) -> np.ndarray | None:
    values = []
    for item in series:
        if item is None:
            continue
        try:
            if hasattr(item, "tolist"):
                arr = np.asarray(item.tolist(), dtype=np.float64)
            else:
                arr = np.asarray(item, dtype=np.float64)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            values.append(arr.reshape(-1))
        except Exception:
            continue
    if not values:
        return None
    width = max(v.shape[0] for v in values)
    out = np.full((len(values), width), np.nan, dtype=np.float64)
    for i, v in enumerate(values):
        out[i, : v.shape[0]] = v
    return out


@dataclass
class CompareResult:
    episode: str
    n_rows: int
    mean_abs_error: np.ndarray
    max_abs_error: np.ndarray
    rms_error: float
    joint_names: tuple[str, ...] = JOINT_NAMES

    def summary_lines(self) -> list[str]:
        lines = [
            f"episode={self.episode}  rows={self.n_rows}  rms={self.rms_error:.4f}",
            "joint                 mean|err|   max|err|",
        ]
        for i, name in enumerate(self.joint_names):
            if i >= len(self.mean_abs_error):
                break
            lines.append(
                f"{name:20s} {self.mean_abs_error[i]:10.4f} {self.max_abs_error[i]:10.4f}"
            )
        return lines


@dataclass(frozen=True)
class EpisodeSegmentIdentity:
    """Catalog episode identity bound to the recording segment it must query."""

    episode: str
    segment_id: str


def episode_segment_identity(record: EpisodeRecord) -> EpisodeSegmentIdentity:
    """Build the segment identity emitted by ``takes.begin_recording`` for a record."""
    episode = sanitize_name(record.episode)
    rrd_stem = Path(record.rrd_path).stem
    if not record.episode.strip() or episode != rrd_stem:
        raise SystemExit(
            "FAIL: catalog episode does not match its recording path: "
            f"{record.episode!r} != {rrd_stem!r}"
        )
    return EpisodeSegmentIdentity(
        episode=episode,
        segment_id=f"{sanitize_name(record.dataset)}-{rrd_stem}",
    )


@dataclass(frozen=True)
class AlignedVectorRows:
    """Goal/state vectors paired from the same Query API dataframe rows."""

    segment_ids: tuple[str, ...]
    goal: np.ndarray
    state: np.ndarray


def _valid_segment_id(value: Any) -> str | None:
    """Return a non-blank Rerun segment identifier, never a stringified null."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _finite_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.ndim == 0:
        vector = vector.reshape(1)
    else:
        vector = vector.reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        return None
    return vector


def aligned_vector_rows(df, *, goal_column: str, state_column: str) -> AlignedVectorRows:
    """Keep only finite goal/state vectors present together in one segment row."""
    if "rerun_segment_id" not in df.columns:
        raise SystemExit("FAIL: Query API dataframe has no rerun_segment_id column")

    segments: list[str] = []
    goals: list[np.ndarray] = []
    states: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    rows = df[["rerun_segment_id", goal_column, state_column]].itertuples(
        index=False, name=None
    )
    for segment_id, goal_value, state_value in rows:
        segment = _valid_segment_id(segment_id)
        if segment is None:
            continue
        goal = _finite_vector(goal_value)
        state = _finite_vector(state_value)
        if goal is None or state is None or goal.shape != state.shape:
            continue
        if expected_shape is None:
            expected_shape = goal.shape
        if goal.shape != expected_shape:
            continue
        segments.append(segment)
        goals.append(goal)
        states.append(state)

    width = expected_shape[0] if expected_shape else 0
    return AlignedVectorRows(
        segment_ids=tuple(segments),
        goal=np.asarray(goals, dtype=np.float64).reshape(len(goals), width),
        state=np.asarray(states, dtype=np.float64).reshape(len(states), width),
    )


def compare_goal_vs_position(
    dataset_entry: Any,
    *,
    episode: str | None = None,
    timeline: str | None = None,
    verified_identity: EpisodeSegmentIdentity | None = None,
) -> CompareResult:
    """Align follower/goal vs follower/position via Query API and score tracking error."""
    index = pick_timeline(dataset_entry, timeline)
    contents = [
        f"{FOLLOWER}/position",
        f"/{FOLLOWER}/position",
        f"{FOLLOWER}/goal",
        f"/{FOLLOWER}/goal",
    ]
    view = dataset_entry.filter_contents(contents)
    df = reader_to_pandas(view, index=index, contents=None)

    pos_cols = find_scalar_columns(df, f"{FOLLOWER}/position") or find_scalar_columns(df, f"/{FOLLOWER}/position")
    goal_cols = find_scalar_columns(df, f"{FOLLOWER}/goal") or find_scalar_columns(df, f"/{FOLLOWER}/goal")
    if not pos_cols or not goal_cols:
        schema = list_schema(dataset_entry)
        raise SystemExit(
            "FAIL: could not find follower/position and follower/goal scalar columns via Query API. "
            f"entities={schema['entities'][:20]} components_sample={schema['components'][:12]} "
            f"df_columns={list(df.columns)[:20]}"
        )

    aligned = aligned_vector_rows(
        df, goal_column=goal_cols[0], state_column=pos_cols[0]
    )
    if len(aligned.goal) == 0:
        raise SystemExit("FAIL: no overlapping non-null goal/position rows")

    err = np.abs(aligned.goal - aligned.state)
    mean_abs = np.nanmean(err, axis=0)
    max_abs = np.nanmax(err, axis=0)
    rms = float(np.sqrt(np.nanmean(err**2)))
    expected_segment_id = (
        _valid_segment_id(verified_identity.segment_id)
        if verified_identity is not None
        else None
    )
    observed_segment_ids = [_valid_segment_id(value) for value in df["rerun_segment_id"]]
    requested_episode_is_proven = (
        episode is not None
        and verified_identity is not None
        and episode == verified_identity.episode
        and expected_segment_id is not None
        and all(segment_id is not None for segment_id in observed_segment_ids)
        and set(observed_segment_ids) == {expected_segment_id}
    )
    return CompareResult(
        episode=episode if requested_episode_is_proven else "all",
        n_rows=len(aligned.goal),
        mean_abs_error=mean_abs,
        max_abs_error=max_abs,
        rms_error=rms,
    )


def episodes_for_query(
    *,
    dataset: str | None,
    tag: str | None,
    episode: str | None,
    recordings_dir: Path,
    catalog_path: Path,
) -> list[EpisodeRecord]:
    return query_episodes(
        dataset=dataset,
        tag=tag,
        episode=episode,
        catalog_path=catalog_path,
        recordings_dir=recordings_dir,
    )


def schema_report(
    dataset: str,
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
    *,
    rrd_paths: list[Path] | None = None,
) -> str:
    with open_dataset_server(
        dataset, recordings_dir=recordings_dir, rrd_paths=rrd_paths
    ) as ds:
        schema = list_schema(ds)
        timeline = pick_timeline(ds)
        lines = [
            f"dataset={sanitize_name(dataset)}",
            f"timeline={timeline}",
            f"indexes={schema['indexes']}",
            f"entities ({len(schema['entities'])}):",
        ]
        for ent in schema["entities"][:40]:
            lines.append(f"  - {ent}")
        if len(schema["entities"]) > 40:
            lines.append(f"  … +{len(schema['entities']) - 40} more")
        lines.append(f"component columns: {len(schema['components'])}")
        return "\n".join(lines)
