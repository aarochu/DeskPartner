"""Local episode catalog (filesystem stand-in for so100-server catalog).

Each episode gets a row in ``recordings/catalog.json`` plus a sidecar
``<episode>.meta.json`` next to the ``.rrd``. Query/export/replay use this so the
loop works without the SO gRPC catalog server.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from p5_rerun_port.constants import DEFAULT_CATALOG, DEFAULT_RECORDINGS_DIR, SEGMENT_TAGS


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    return cleaned or time.strftime("%Y%m%d-%H%M%S")


@dataclass
class EpisodeRecord:
    dataset: str
    episode: str
    task: str = ""
    tag: str = SEGMENT_TAGS[0]
    rrd_path: str = ""
    traj_path: str = ""
    meta_path: str = ""
    fps: float = 30.0
    n_frames: int = 0
    duration_s: float = 0.0
    robot_type: str = "seeed_b601_dm_follower"
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def episode_dir(recordings_dir: Path, dataset: str) -> Path:
    return recordings_dir / sanitize_name(dataset)


def episode_rrd_path(recordings_dir: Path, dataset: str, episode: str) -> Path:
    directory = episode_dir(recordings_dir, dataset)
    stem = sanitize_name(episode)
    path = directory / f"{stem}.rrd"
    counter = 2
    while path.exists():
        path = directory / f"{stem}-{counter}.rrd"
        counter += 1
    return path


def next_episode_name(recordings_dir: Path, dataset: str) -> str:
    directory = episode_dir(recordings_dir, dataset)
    highest = 0
    if directory.is_dir():
        for path in directory.glob("episode_*.rrd"):
            match = re.match(r"episode_(\d+)", path.stem)
            if match:
                highest = max(highest, int(match.group(1)))
    return f"episode_{highest + 1:02d}"


def load_catalog(catalog_path: Path = DEFAULT_CATALOG) -> list[EpisodeRecord]:
    if not catalog_path.exists():
        return []
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    episodes = raw.get("episodes", raw if isinstance(raw, list) else [])
    return [EpisodeRecord.from_dict(e) for e in episodes]


def save_catalog(episodes: list[EpisodeRecord], catalog_path: Path = DEFAULT_CATALOG) -> None:
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episodes": [e.to_dict() for e in episodes],
    }
    catalog_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upsert_episode(record: EpisodeRecord, catalog_path: Path = DEFAULT_CATALOG) -> None:
    episodes = load_catalog(catalog_path)
    key = (record.dataset, record.episode)
    episodes = [e for e in episodes if (e.dataset, e.episode) != key]
    episodes.append(record)
    episodes.sort(key=lambda e: (e.dataset, e.episode))
    save_catalog(episodes, catalog_path)

    if record.meta_path:
        meta = Path(record.meta_path)
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")


def query_episodes(
    *,
    dataset: str | None = None,
    tag: str | None = None,
    episode: str | None = None,
    catalog_path: Path = DEFAULT_CATALOG,
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
) -> list[EpisodeRecord]:
    """Return catalog rows; also pick up orphan ``*.meta.json`` under recordings/."""
    by_key: dict[tuple[str, str], EpisodeRecord] = {
        (e.dataset, e.episode): e for e in load_catalog(catalog_path)
    }
    if recordings_dir.is_dir():
        for meta in recordings_dir.glob("*/*.meta.json"):
            try:
                rec = EpisodeRecord.from_dict(json.loads(meta.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError):
                continue
            by_key.setdefault((rec.dataset, rec.episode), rec)

    rows = list(by_key.values())
    if dataset:
        rows = [e for e in rows if e.dataset == sanitize_name(dataset) or e.dataset == dataset]
    if tag:
        rows = [e for e in rows if e.tag == tag]
    if episode:
        rows = [e for e in rows if e.episode == episode]
    return sorted(rows, key=lambda e: (e.dataset, e.episode))
