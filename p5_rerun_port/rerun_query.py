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
    directory = recordings_dir / sanitize_name(dataset)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.rrd"))


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


def compare_goal_vs_position(
    dataset_entry: Any,
    *,
    episode: str | None = None,
    timeline: str | None = None,
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

    pos = stack_scalar_column(df[pos_cols[0]])
    goal = stack_scalar_column(df[goal_cols[0]])
    if pos is None or goal is None:
        raise SystemExit("FAIL: empty position/goal series from Query API")

    n = min(len(pos), len(goal))
    pos, goal = pos[:n], goal[:n]
    # Drop rows where either is all-nan
    mask = ~(np.isnan(pos).all(axis=1) | np.isnan(goal).all(axis=1))
    pos, goal = pos[mask], goal[mask]
    if len(pos) == 0:
        raise SystemExit("FAIL: no overlapping non-null goal/position rows")

    err = np.abs(goal - pos)
    mean_abs = np.nanmean(err, axis=0)
    max_abs = np.nanmax(err, axis=0)
    rms = float(np.sqrt(np.nanmean(err**2)))
    return CompareResult(
        episode=episode or "all",
        n_rows=len(pos),
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


def schema_report(dataset: str, recordings_dir: Path = DEFAULT_RECORDINGS_DIR) -> str:
    with open_dataset_server(dataset, recordings_dir=recordings_dir) as ds:
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
