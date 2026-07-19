from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest

from p5_rerun_port.challenge import canonical
from p5_rerun_port.challenge.canonical import materialize_success
from p5_rerun_port.challenge.config import ChallengeConfig
from p5_rerun_port.challenge.models import EpisodeIdentity, InventoryRow
from p5_rerun_port.rerun_query import list_schema, open_dataset_server, reader_to_pandas


CONFIG = Path("config/rerun_query_challenge.yaml")
TASK = "Pick up one can and place it in the taped sorting zone"


class FakeTensor:
    """Small torch-shaped test double covering the real detach/cpu/numpy API."""

    def __init__(self, value: Any):
        self._value = np.asarray(value)

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._value

    def item(self) -> Any:
        return self._value.item()


class FakeLeRobotDataset:
    calls: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    action_names: tuple[str, ...] = ()
    state_names: tuple[str, ...] = ()
    dataset_fps = 30

    def __init__(self, **kwargs: Any):
        type(self).calls.append(kwargs)
        self.features = {
            "action": {"names": list(type(self).action_names)},
            "observation.state": {"names": list(type(self).state_names)},
        }
        self.fps = type(self).dataset_fps

    def __len__(self) -> int:
        return len(type(self).samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return type(self).samples[index]


@pytest.fixture
def config() -> ChallengeConfig:
    return ChallengeConfig.load(CONFIG)


@pytest.fixture
def row(config: ChallengeConfig) -> InventoryRow:
    source = next(source for source in config.sources if source.role == "success")
    return InventoryRow(
        identity=EpisodeIdentity(source.repo_id, source.revision, "3"),
        role="success",
        task_key="single_can",
        source_path="data/chunk-000/file-003.parquet",
        episode_index=3,
        attempt_id=None,
        frame_count=3,
        captured_at="2026-07-19T01:03:00.000Z",
    )


@pytest.fixture(autouse=True)
def fake_dataset(monkeypatch: pytest.MonkeyPatch, config: ChallengeConfig) -> None:
    names = tuple(f"{name}.pos" for name in config.joint_names)
    FakeLeRobotDataset.calls = []
    FakeLeRobotDataset.action_names = names
    FakeLeRobotDataset.state_names = names
    FakeLeRobotDataset.dataset_fps = 30
    FakeLeRobotDataset.samples = [_sample(frame) for frame in range(3)]
    monkeypatch.setattr(canonical, "_lerobot_dataset_class", lambda: FakeLeRobotDataset)


def _sample(frame: int) -> dict[str, Any]:
    front = np.empty((3, 2, 4), dtype=np.float32)
    front[0, :, :] = frame / 10
    front[1, :, :] = 0.5
    front[2, :, :] = 1.0
    side = np.full((6, 5, 3), 20 + frame, dtype=np.uint8)
    return {
        "action": FakeTensor(np.arange(7, dtype=np.float32) + frame),
        "episode_index": FakeTensor(3),
        "frame_index": FakeTensor(frame),
        "index": FakeTensor(1000 + frame),
        "observation.images.front": FakeTensor(front),
        "observation.images.side": side,
        "observation.state": np.arange(7, dtype=np.float32) + frame + 0.25,
        "task": TASK,
        "timestamp": FakeTensor(np.float32(frame / 30)),
    }


def _component_column(frame, marker: str) -> str:
    matches = [str(name) for name in frame.columns if marker in str(name)]
    assert len(matches) == 1, frame.columns
    return matches[0]


def _text_value(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple, np.ndarray)):
        assert len(value) == 1
        value = value[0]
    return str(value)


def _bytes_value(value: Any) -> bytes:
    if hasattr(value, "as_py"):
        value = value.as_py()
    while isinstance(value, (list, tuple)) or (
        isinstance(value, np.ndarray) and value.dtype == object
    ):
        assert len(value) == 1
        value = value[0]
    if isinstance(value, np.ndarray):
        return value.astype(np.uint8, copy=False).tobytes()
    return bytes(value)


def test_materializes_three_query_readable_frames_with_static_provenance(
    config: ChallengeConfig, row: InventoryRow, tmp_path: Path
) -> None:
    output = tmp_path / "episode-3.rrd"

    artifact = materialize_success(row, config, output)

    assert artifact.identity == row.identity.canonical
    assert artifact.status == "ready"
    assert artifact.rrd_path == output
    assert artifact.frame_count == 3
    assert artifact.reason_codes == ()
    assert artifact.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert FakeLeRobotDataset.calls == [{
        "repo_id": row.identity.repo_id,
        "root": canonical._episode_cache_root(row.identity),
        "episodes": [3],
        "revision": row.identity.revision,
        "force_cache_sync": True,
        "download_videos": True,
        "video_backend": "pyav",
    }]

    with open_dataset_server("three-frame-canonical", rrd_paths=[output]) as dataset:
        schema = list_schema(dataset)
        assert {
            "/follower/goal", "/follower/position", "/camera/cam0", "/camera/cam1",
            "/episode/source", "/episode/repo_id", "/episode/revision",
            "/episode/source_index", "/episode/task", "/episode/fps",
            "/episode/joint_names",
        }.issubset(set(schema["entities"]))

        dynamic = {}
        for entity in ("/follower/goal", "/follower/position", "/camera/cam0", "/camera/cam1"):
            frame = reader_to_pandas(dataset, index="frame", contents=[entity])
            assert frame["frame"].tolist() == [0, 1, 2]
            assert frame["time"].dt.total_seconds().to_numpy() == pytest.approx([0, 1 / 30, 2 / 30])
            assert len(set(frame["rerun_segment_id"])) == 1
            dynamic[entity] = frame

        goal_col = _component_column(dynamic["/follower/goal"], "Scalars:scalars")
        position_col = _component_column(dynamic["/follower/position"], "Scalars:scalars")
        assert np.asarray(dynamic["/follower/goal"][goal_col].iloc[2]).reshape(-1) == pytest.approx(
            np.arange(7) + 2
        )
        assert np.asarray(dynamic["/follower/position"][position_col].iloc[0]).reshape(-1) == pytest.approx(
            np.arange(7) + 0.25
        )

        for entity, expected_size in (("/camera/cam0", (4, 2)), ("/camera/cam1", (5, 6))):
            frame = dynamic[entity]
            media_col = _component_column(frame, "EncodedImage:media_type")
            blob_col = _component_column(frame, "EncodedImage:blob")
            assert [_text_value(value) for value in frame[media_col]] == ["image/jpeg"] * 3
            image = Image.open(__import__("io").BytesIO(_bytes_value(frame[blob_col].iloc[0])))
            assert image.mode == "RGB"
            assert image.size == expected_size

        expected_static = {
            "/episode/source": row.identity.canonical,
            "/episode/repo_id": row.identity.repo_id,
            "/episode/revision": row.identity.revision,
            "/episode/source_index": "3",
            "/episode/task": TASK,
            "/episode/fps": "30",
            "/episode/joint_names": ",".join(config.joint_names),
        }
        for entity, expected in expected_static.items():
            frame = reader_to_pandas(dataset, index=None, contents=[entity])
            column = _component_column(frame, "TextDocument:text")
            assert len(frame) == 1
            assert _text_value(frame[column].iloc[0]) == expected


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("action", np.arange(6), "INVALID_ACTION"),
        ("action", [0, 1, 2, 3, 4, 5, np.nan], "INVALID_ACTION"),
        ("observation.state", [0, 1, 2, 3, 4, 5, np.inf], "INVALID_STATE"),
    ],
)
def test_rejects_non_finite_or_wrong_width_vectors_without_publishing(
    field: str,
    value: Any,
    reason: str,
    config: ChallengeConfig,
    row: InventoryRow,
    tmp_path: Path,
) -> None:
    FakeLeRobotDataset.samples[1][field] = value
    output = tmp_path / "invalid-vector.rrd"

    artifact = materialize_success(row, config, output)

    assert artifact.status == "rejected"
    assert artifact.rrd_path is None
    assert artifact.sha256 is None
    assert artifact.frame_count == 1
    assert artifact.reason_codes == (reason,)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("mutation", "reason", "processed"),
    [
        (lambda samples: samples[0].__setitem__("episode_index", 4), "EPISODE_INDEX_MISMATCH", 0),
        (lambda samples: samples[1].__setitem__("frame_index", 7), "FRAME_INDEX_NONCONTIGUOUS", 1),
        (lambda samples: samples[2].__setitem__("timestamp", 9.0), "TIMESTAMP_MISMATCH", 2),
    ],
)
def test_rejects_mismatched_episode_frame_or_timestamp(
    mutation,
    reason: str,
    processed: int,
    config: ChallengeConfig,
    row: InventoryRow,
    tmp_path: Path,
) -> None:
    mutation(FakeLeRobotDataset.samples)

    artifact = materialize_success(row, config, tmp_path / "bad-index.rrd")

    assert artifact.status == "rejected"
    assert artifact.frame_count == processed
    assert artifact.reason_codes == (reason,)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("observation.images.front", np.zeros((2, 4, 1), dtype=np.float32), "INVALID_FRONT_IMAGE"),
        ("observation.images.side", np.full((6, 5, 3), np.nan), "INVALID_SIDE_IMAGE"),
        ("observation.images.side", np.zeros((6, 5, 4), dtype=np.uint8), "INVALID_SIDE_IMAGE"),
    ],
)
def test_rejects_malformed_images(
    field: str,
    value: Any,
    reason: str,
    config: ChallengeConfig,
    row: InventoryRow,
    tmp_path: Path,
) -> None:
    FakeLeRobotDataset.samples[0][field] = value

    artifact = materialize_success(row, config, tmp_path / "bad-image.rrd")

    assert artifact.status == "rejected"
    assert artifact.frame_count == 0
    assert artifact.reason_codes == (reason,)
    assert list(tmp_path.iterdir()) == []


def test_rejects_feature_order_and_declared_frame_count(
    config: ChallengeConfig, row: InventoryRow, tmp_path: Path
) -> None:
    FakeLeRobotDataset.action_names = tuple(reversed(FakeLeRobotDataset.action_names))
    output = tmp_path / "wrong-order.rrd"
    artifact = materialize_success(row, config, output)
    assert artifact.reason_codes == ("FEATURE_SCHEMA_MISMATCH",)
    assert not output.exists()

    FakeLeRobotDataset.action_names = tuple(f"{name}.pos" for name in config.joint_names)
    short_row = InventoryRow(**{**row.__dict__, "frame_count": 4})
    artifact = materialize_success(short_row, config, output)
    assert artifact.reason_codes == ("FRAME_COUNT_MISMATCH",)
    assert not output.exists()


def test_verify_failure_preserves_existing_destination_and_removes_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
    config: ChallengeConfig,
    row: InventoryRow,
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing.rrd"
    original = b"previous verified artifact"
    output.write_bytes(original)
    monkeypatch.setattr(canonical, "_verify_rrd", lambda path: False)

    artifact = materialize_success(row, config, output)

    assert artifact.status == "rejected"
    assert artifact.rrd_path is None
    assert artifact.sha256 is None
    assert artifact.frame_count == 3
    assert artifact.reason_codes == ("RRD_VERIFY_FAILED",)
    assert output.read_bytes() == original
    assert list(tmp_path.iterdir()) == [output]
