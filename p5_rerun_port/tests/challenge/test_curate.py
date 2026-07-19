from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from p5_rerun_port.challenge import curate
from p5_rerun_port.challenge.artifacts import build_selection_manifest
from p5_rerun_port.challenge.config import ChallengeConfig
from p5_rerun_port.challenge.curate import CurateError, build_derivative, validate_derivative_fresh
from p5_rerun_port.challenge.models import EpisodeIdentity, InventoryRow
from p5_rerun_port.challenge.quality import EpisodeVerdict, ThresholdSnapshot


CONFIG = Path("config/rerun_query_challenge.yaml")


@pytest.fixture
def config() -> ChallengeConfig:
    return ChallengeConfig.load(CONFIG)


def _item(source, episode_index: int, frame_count: int = 2) -> dict:
    identity = f"{source.repo_id}@{source.revision}:{episode_index}"
    return {
        "identity": identity,
        "role": source.role,
        "task_key": source.task_key,
        "source": {
            "repo_id": source.repo_id,
            "revision": source.revision,
            "source_key": str(episode_index),
            "path": "data/chunk-000/file-000.parquet",
        },
        "source_path": "data/chunk-000/file-000.parquet",
        "episode_index": episode_index,
        "attempt_id": None,
        "frame_count": frame_count,
        "verdict": "PASS",
        "reason_codes": [],
        "selected": True,
    }


def _manifest(config: ChallengeConfig) -> dict:
    sources = [source for source in config.sources if source.role == "success"]
    items = [_item(source, 0) for source in sources]
    manifest = {
        "schema_version": 1,
        "generated_at": "2026-07-19T00:00:00Z",
        "source_lock": {
            "schema_version": 1,
            "config_sha256": "config",
            "query_code_commit": "code",
            "destination_repo": config.destination_repo,
            "joint_names": list(config.joint_names),
            "camera_keys": list(config.camera_keys),
            "sources": [
                {
                    "repo_id": source.repo_id,
                    "resolved_sha": source.revision,
                    "role": source.role,
                }
                for source in config.sources
            ],
        },
        "threshold_snapshot_digest": "thresholds",
        "destination_repo": config.destination_repo,
        "data_schema": {
            "action": {"dtype": "float32", "shape": [7], "names": list(config.joint_names)},
            "state": {"dtype": "float32", "shape": [7], "names": list(config.joint_names)},
            "cameras": list(config.camera_keys),
        },
        "source_counts": {"total": 2, "success": 2, "failure": 0},
        "selected_identities": [item["identity"] for item in items],
        "selected_episode_count": 2,
        "selected_frame_count": 4,
        "excluded_items": [],
        "items": items,
    }
    from p5_rerun_port.challenge.artifacts import selection_payload_digest

    manifest["selection_payload_digest"] = selection_payload_digest(manifest)
    return manifest


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "selection-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_artifact_manifest_is_accepted_by_curator_contract(
    tmp_path: Path, config: ChallengeConfig
) -> None:
    sources = [source for source in config.sources if source.role == "success"]
    rows = tuple(
        InventoryRow(
            EpisodeIdentity(source.repo_id, source.revision, "0"),
            "success",
            source.task_key,
            "data/chunk-000/file-000.parquet",
            0,
            None,
            2,
            "2026-07-19T00:00:00Z",
        )
        for source in sources
    )
    snapshot = ThresholdSnapshot((), tuple(row.identity.canonical for row in rows), (), "thresholds")
    verdicts = tuple(
        EpisodeVerdict(row.identity.canonical, "PASS", (), f"metrics-{index}", "thresholds", f"verdict-{index}")
        for index, row in enumerate(rows)
    )
    source_lock = _manifest(config)["source_lock"]
    manifest = build_selection_manifest(rows, verdicts, snapshot, source_lock, "2026-07-19T00:00:00Z")

    loaded, selections = curate._load_manifest(_write_manifest(tmp_path, manifest), config)

    assert loaded["selection_payload_digest"] == manifest["selection_payload_digest"]
    assert [selection.identity for selection in selections] == sorted(manifest["selected_identities"])


def _make_source(root: Path, repo_id: str, config: ChallengeConfig, seed: int) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = [f"{name}.pos" for name in config.joint_names]
    features = {
        "action": {"dtype": "float32", "shape": (7,), "names": names},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": names},
        "observation.images.front": {
            "dtype": "video", "shape": (8, 10, 3), "names": ["height", "width", "channels"]
        },
        "observation.images.side": {
            "dtype": "video", "shape": (8, 10, 3), "names": ["height", "width", "channels"]
        },
    }
    dataset = LeRobotDataset.create(
        repo_id, 30, features, root=root, robot_type=config.robot_type,
        use_videos=True, video_backend="pyav", vcodec="h264",
    )
    for frame in range(2):
        image = np.full((8, 10, 3), seed + frame, dtype=np.uint8)
        dataset.add_frame({
            "action": np.arange(7, dtype=np.float32) + frame,
            "observation.state": np.arange(7, dtype=np.float32) + frame + 0.25,
            "observation.images.front": image,
            "observation.images.side": image[:, ::-1].copy(),
            "task": "source task",
        })
    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()


def test_builds_two_source_derivative_and_fresh_loads_every_frame(
    tmp_path: Path, config: ChallengeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(config)
    selected_sources = [source for source in config.sources if source.role == "success"]
    roots = {}
    for index, source in enumerate(selected_sources):
        root = tmp_path / f"source-{index}"
        _make_source(root, source.repo_id, config, 20 + index * 20)
        roots[source.repo_id] = root

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    def open_local(selection):
        return LeRobotDataset(
            selection.repo_id,
            root=roots[selection.repo_id],
            episodes=[selection.episode_index],
            video_backend="pyav",
        )

    monkeypatch.setattr(curate, "_open_source", open_local)
    destination = build_derivative(_write_manifest(tmp_path, manifest), config, tmp_path / "out")
    digest = manifest["selection_payload_digest"]
    report = validate_derivative_fresh(destination, config.destination_repo, digest)

    assert report["ok"] is True
    assert report["episode_indexes"] == [0, 1]
    assert report["frames"] == 4
    assert report["source_identities"] == sorted(manifest["selected_identities"])
    assert report["tasks"] == [
        "Pick up one can and place it in the taped sorting zone",
        "Pull one of the two cans into the blue-taped recycling zone.",
    ]
    assert report["camera_keys"] == ["front", "side"]
    assert set(report["decoded_shapes"]) == {"front", "side"}
    embedded = json.loads((destination / "meta/rerun_query_selection_manifest.json").read_text())
    assert embedded["selection_payload_digest"] == digest


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest["items"][0].update(role="failure"),
        lambda manifest: manifest["items"][0].update(verdict="REVIEW"),
        lambda manifest: manifest["selected_identities"].append(manifest["selected_identities"][0]),
        lambda manifest: manifest["items"][0]["source"].update(revision="0" * 40),
        lambda manifest: manifest["data_schema"]["action"].update(names=["gripper"] * 7),
    ],
)
def test_manifest_tampering_fails_before_any_source_is_opened(
    tmp_path: Path, config: ChallengeConfig, monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    manifest = deepcopy(_manifest(config))
    mutation(manifest)
    monkeypatch.setattr(curate, "_open_source", lambda selection: pytest.fail("opened source"))
    with pytest.raises(CurateError):
        build_derivative(_write_manifest(tmp_path, manifest), config, tmp_path / "out")


@pytest.mark.parametrize("problem", ["fps", "joints", "video", "missing_episode"])
def test_source_contract_failures_are_closed(
    tmp_path: Path, config: ChallengeConfig, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    manifest = _manifest(config)
    names = [f"{name}.pos" for name in config.joint_names]
    features = {
        "action": {"dtype": "float32", "shape": (7,), "names": names},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": names},
        "observation.images.front": {"dtype": "video", "shape": (8, 10, 3), "names": ["height", "width", "channels"]},
        "observation.images.side": {"dtype": "video", "shape": (8, 10, 3), "names": ["height", "width", "channels"]},
    }
    dataset = SimpleNamespace(fps=30, features=features)
    if problem == "fps":
        dataset.fps = 29
    elif problem == "joints":
        dataset.features["action"]["names"] = list(reversed(names))
    elif problem == "video":
        del dataset.features["observation.images.side"]
    else:
        dataset = type(
            "MissingEpisode",
            (),
            {"fps": 30, "features": features, "__len__": lambda self: 0},
        )()
    monkeypatch.setattr(curate, "_open_source", lambda selection: dataset)
    with pytest.raises((CurateError, TypeError)):
        build_derivative(_write_manifest(tmp_path, manifest), config, tmp_path / "out")


def test_existing_nonempty_destination_is_never_overwritten(
    tmp_path: Path, config: ChallengeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "out" / config.destination_repo
    destination.mkdir(parents=True)
    marker = destination / "owned.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(curate, "_open_source", lambda selection: pytest.fail("opened source"))

    with pytest.raises(CurateError, match="nonempty"):
        build_derivative(_write_manifest(tmp_path, _manifest(config)), config, tmp_path / "out")
    assert marker.read_text(encoding="utf-8") == "keep"
