from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from p5_rerun_port.challenge.config import ChallengeConfig
from p5_rerun_port.challenge.hub import HuggingFaceHubReader
from p5_rerun_port.challenge.inventory import InventoryError, build_inventory, write_source_lock


CONFIG_PATH = Path("config/rerun_query_challenge.yaml")
TASKS = {
    "single_can": "Pick up one can and place it in the taped sorting zone",
    "two_can": "Pull one of the two cans into the blue-taped recycling zone.",
}


class FakeHub:
    def __init__(self, config: ChallengeConfig) -> None:
        self.shas = {source.repo_id: source.revision for source in config.sources}
        self.jsons: dict[tuple[str, str], dict] = {}
        self.parquets: dict[tuple[str, str], pd.DataFrame] = {}
        self.downloads: dict[tuple[str, str], Path] = {}
        for source in config.sources:
            if source.role == "success":
                self._add_success(source.repo_id, source.task_key, source.expected_items, source.expected_frames or 0)
            else:
                self._add_failure(source.repo_id, source.task_key, source.expected_items)

    def _add_success(self, repo: str, task_key: str, count: int, frames: int) -> None:
        lengths = [861, 1086] if task_key == "single_can" else [648]
        lengths.extend([1] * (count - len(lengths)))
        lengths[-1] += frames - sum(lengths)
        rows = [
            {
                "episode_index": index,
                "tasks": [TASKS[task_key]],
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": index,
            }
            for index, length in enumerate(lengths)
        ]
        # Deliberately make one metadata file contain multiple episode rows.
        paths = ["meta/episodes/chunk-000/file-000.parquet"]
        self.parquets[(repo, paths[0])] = pd.DataFrame(rows[:2])
        if len(rows) > 2:
            paths.append("meta/episodes/chunk-000/file-001.parquet")
            self.parquets[(repo, paths[1])] = pd.DataFrame(rows[2:])
        self.jsons[(repo, "meta/info.json")] = {
            "codebase_version": "v3.0",
            "robot_type": "seeed_b601_dm_follower",
            "fps": 30,
            "total_episodes": count,
            "total_frames": frames,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "features": {
                "action": {"shape": [7], "names": [f"{name}.pos" for name in JOINTS]},
                "observation.state": {"shape": [7], "names": [f"{name}.pos" for name in JOINTS]},
                "observation.images.front": {"dtype": "video"},
                "observation.images.side": {"dtype": "video"},
            },
        }
        self.jsons[(repo, "SHARE_MANIFEST.json")] = {
            "files": [{"path": path} for path in paths],
            "source_attempts": [
                {
                    "episode_index": index,
                    "attempt_id": f"attempt-{index}",
                    "started_at_utc": f"2026-07-19T01:{index:02d}:00.000Z",
                }
                for index in range(count)
            ],
        }

    def _add_failure(self, repo: str, task_key: str, count: int) -> None:
        attempts = []
        for index in range(count):
            attempt_id = f"20260719T{index:06d}.000000Z-{task_key}-{index}"
            attempts.append(
                {
                    "archive_complete": index != 0,
                    "attempt_id": attempt_id,
                    "disposition": "aborted",
                    "samples": 100 + index,
                    "started_at": f"2026-07-19T02:{index:02d}:00.000Z",
                    "finished_at": f"2026-07-19T02:{index:02d}:05.000Z",
                    "task": TASKS[task_key],
                    "training_included": False,
                }
            )
            filename = "attempt.partial.rrd" if index == 0 else "attempt.rrd"
            path = f"failed_attempts/{attempt_id}/{filename}"
            self.downloads[(repo, path)] = Path("/fake") / repo / path
        self.jsons[(repo, "FAILURE_INDEX.json")] = {
            "training_eligible": False,
            "attempt_count": count,
            "attempts": attempts,
        }

    def dataset_sha(self, repo_id: str, revision: str) -> str:
        return self.shas[repo_id]

    def read_json(self, repo_id: str, revision: str, path: str) -> dict:
        return deepcopy(self.jsons[(repo_id, path)])

    def read_parquet(self, repo_id: str, revision: str, path: str) -> pd.DataFrame:
        return self.parquets[(repo_id, path)].copy(deep=True)

    def download(self, repo_id: str, revision: str, path: str) -> Path:
        try:
            return self.downloads[(repo_id, path)]
        except KeyError as error:
            raise FileNotFoundError(path) from error


JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
    "wrist_yaw", "wrist_roll", "gripper",
)


@pytest.fixture
def config() -> ChallengeConfig:
    return ChallengeConfig.load(CONFIG_PATH)


@pytest.fixture
def fake_hub(config: ChallengeConfig) -> FakeHub:
    return FakeHub(config)


def test_builds_all_102_revision_locked_rows(config: ChallengeConfig, fake_hub: FakeHub) -> None:
    inventory = build_inventory(config, fake_hub)

    assert len(inventory) == 102
    assert sum(row.role == "success" for row in inventory) == 77
    assert sum(row.role == "failure" for row in inventory) == 25
    assert len({row.identity.canonical for row in inventory}) == 102
    assert [row.frame_count for row in inventory if row.role == "success"][:2] == [861, 1086]
    partial = next(row for row in inventory if row.source_path.endswith("attempt.partial.rrd"))
    assert partial.source_path.startswith(f"failed_attempts/{partial.attempt_id}/")


def test_rejects_sha_mismatch(config: ChallengeConfig, fake_hub: FakeHub) -> None:
    fake_hub.shas[config.sources[0].repo_id] = "0" * 40
    with pytest.raises(InventoryError, match="resolved SHA"):
        build_inventory(config, fake_hub)


def test_rejects_duplicate_episode_index(config: ChallengeConfig, fake_hub: FakeHub) -> None:
    repo = config.sources[0].repo_id
    frame = fake_hub.parquets[(repo, "meta/episodes/chunk-000/file-000.parquet")]
    frame.loc[1, "episode_index"] = 0
    with pytest.raises(InventoryError, match="duplicate episode_index"):
        build_inventory(config, fake_hub)


def test_rejects_missing_or_ambiguous_failure_rrd(config: ChallengeConfig, fake_hub: FakeHub) -> None:
    source = next(source for source in config.sources if source.role == "failure")
    attempt = fake_hub.jsons[(source.repo_id, "FAILURE_INDEX.json")]["attempts"][0]
    partial = f"failed_attempts/{attempt['attempt_id']}/attempt.partial.rrd"
    del fake_hub.downloads[(source.repo_id, partial)]
    with pytest.raises(InventoryError, match="exactly one authoritative RRD"):
        build_inventory(config, fake_hub)

    fake_hub.downloads[(source.repo_id, partial)] = Path("/fake") / partial
    final = f"failed_attempts/{attempt['attempt_id']}/attempt.rrd"
    fake_hub.downloads[(source.repo_id, final)] = Path("/fake") / final
    with pytest.raises(InventoryError, match="exactly one authoritative RRD"):
        build_inventory(config, fake_hub)


def test_rejects_count_mismatch(config: ChallengeConfig, fake_hub: FakeHub) -> None:
    repo = config.sources[1].repo_id
    fake_hub.jsons[(repo, "meta/info.json")]["total_episodes"] -= 1
    with pytest.raises(InventoryError, match="total_episodes"):
        build_inventory(config, fake_hub)


def test_rejects_blank_capture_time(config: ChallengeConfig, fake_hub: FakeHub) -> None:
    repo = config.sources[0].repo_id
    fake_hub.jsons[(repo, "SHARE_MANIFEST.json")]["source_attempts"][0]["started_at_utc"] = " "
    with pytest.raises(InventoryError, match="started_at_utc"):
        build_inventory(config, fake_hub)


def test_rejects_source_path_escape(config: ChallengeConfig, fake_hub: FakeHub) -> None:
    repo = config.sources[0].repo_id
    fake_hub.jsons[(repo, "meta/info.json")]["data_path"] = "../secrets/file-{file_index}.parquet"
    with pytest.raises(InventoryError, match="source path"):
        build_inventory(config, fake_hub)


def test_validates_pos_suffix_order_and_training_flags(config: ChallengeConfig, fake_hub: FakeHub) -> None:
    success_repo = config.sources[0].repo_id
    fake_hub.jsons[(success_repo, "meta/info.json")]["features"]["action"]["names"][0] = "shoulder_lift.pos"
    with pytest.raises(InventoryError, match="joint order"):
        build_inventory(config, fake_hub)

    fake_hub = FakeHub(config)
    failure_repo = next(source.repo_id for source in config.sources if source.role == "failure")
    fake_hub.jsons[(failure_repo, "FAILURE_INDEX.json")]["attempts"][0]["training_included"] = True
    with pytest.raises(InventoryError, match="training_included"):
        build_inventory(config, fake_hub)


def test_writes_deterministic_source_lock(config: ChallengeConfig, fake_hub: FakeHub, tmp_path: Path) -> None:
    inventory = build_inventory(config, fake_hub)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    payload = write_source_lock(first, config, inventory, config_path=CONFIG_PATH)
    write_source_lock(second, config, tuple(reversed(inventory)), config_path=CONFIG_PATH)

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text()) == payload
    assert len(payload["config_sha256"]) == 64
    assert len(payload["inventory_payload_digest"]) == 64
    assert payload["expected_counts"] == {"failure": 25, "success": 77, "total": 102}
    assert payload["observed_counts"] == payload["expected_counts"]
    assert [source["resolved_sha"] for source in payload["sources"]] == [
        source.revision for source in config.sources
    ]


def test_production_adapter_forwards_the_locked_revision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str, str]] = []

    class Api:
        def dataset_info(self, repo_id: str, *, revision: str):
            calls.append(("info", repo_id, revision))
            return type("Info", (), {"sha": revision})()

    downloaded = tmp_path / "value.json"
    downloaded.write_text("{}", encoding="utf-8")

    def fake_download(*, repo_id: str, filename: str, repo_type: str, revision: str) -> str:
        assert repo_type == "dataset"
        calls.append((filename, repo_id, revision))
        return str(downloaded)

    monkeypatch.setattr("p5_rerun_port.challenge.hub.hf_hub_download", fake_download)
    reader = HuggingFaceHubReader(Api())  # type: ignore[arg-type]
    revision = "a" * 40

    assert reader.dataset_sha("owner/repo", revision) == revision
    assert reader.read_json("owner/repo", revision, "meta/info.json") == {}
    assert calls == [
        ("info", "owner/repo", revision),
        ("meta/info.json", "owner/repo", revision),
    ]
