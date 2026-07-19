from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from p5_rerun_port.challenge.alignment import (
    AlignmentError,
    _latest_at,
    _require_unique_observation_times,
    extract_aligned_episode,
)
from p5_rerun_port.challenge.config import ChallengeConfig
from p5_rerun_port.rerun_query import open_dataset_server


CONFIG = Path("config/rerun_query_challenge.yaml")
IDENTITY = (
    "Cornerf/rebot-can-sort-stage1-v1-smoke@"
    "74d1f300786d58b4f6f55e1798cbb1a1a48f5409:0"
)
TASK = "Pick up one can and place it in the taped sorting zone"


def test_latest_at_uses_integer_ns_inclusive_ages_and_never_uses_future() -> None:
    action = np.asarray([100_000_000, 100_000_001, 200_000_000], dtype=np.int64)
    observed = np.asarray([0, 300_000_000], dtype=np.int64)

    indexes, matched, ages, present = _latest_at(action, observed, 100_000_000)

    assert indexes.tolist() == [0, -1, -1]
    assert matched.tolist() == [0, -1, -1]
    assert ages.tolist() == [100_000_000, -1, -1]
    assert present.tolist() == [True, False, False]


def test_latest_at_camera_boundary_is_exact() -> None:
    _, _, ages, present = _latest_at(
        np.asarray([66_666_667, 66_666_668]),
        np.asarray([0]),
        66_666_667,
    )
    assert ages.tolist() == [66_666_667, -1]
    assert present.tolist() == [True, False]


@pytest.mark.parametrize(
    ("reason", "name"),
    [
        ("STATE_MISSING_OR_STALE", "state"),
        ("CAMERA_FRONT_MISSING_OR_STALE", "front camera"),
    ],
)
def test_duplicate_observation_timestamps_reject_before_latest_at(
    reason: str, name: str
) -> None:
    with pytest.raises(AlignmentError, match=f"duplicate {name} timestamps") as raised:
        _require_unique_observation_times(
            np.asarray([0, 33_333_333, 33_333_333], dtype=np.int64), reason, name
        )
    assert raised.value.reason_codes == (reason,)


def _disconnect(recording: Any) -> None:
    try:
        recording.flush(blocking=True)
    except TypeError:
        recording.flush()
    recording.disconnect()


def _write_sparse_rrd(
    path: Path, config: ChallengeConfig, recording_id: str = "task6-segment"
) -> None:
    import rerun as rr

    recording = rr.RecordingStream("task6-test", recording_id=recording_id)
    recording.set_sinks(rr.FileSink(str(path)))
    recording.log("/episode/source", rr.TextDocument(IDENTITY), static=True)
    recording.log("/episode/task", rr.TextDocument(TASK), static=True)
    recording.log(
        "/episode/joint_names", rr.TextDocument(",".join(config.joint_names)), static=True
    )

    # Deliberately log rows out of timestamp order. The extractor must explicitly sort.
    for frame in (2, 0, 1):
        recording.set_time("frame", sequence=frame)
        recording.set_time("time", duration=frame / 30)
        recording.log("/follower/goal", rr.Scalars(np.arange(7) + frame))
    for frame, offset_ns in ((2, 0), (0, 0), (1, -100_000_000)):
        recording.set_time("frame", sequence=frame)
        recording.set_time("time", duration=(frame * 1_000_000_000 // 30 + offset_ns) / 1e9)
        recording.log("/follower/position", rr.Scalars(np.arange(7) + frame + 1))
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    for entity in ("/camera/cam0", "/camera/cam1"):
        for frame in (2, 0):
            recording.set_time("frame", sequence=frame)
            recording.set_time("time", duration=frame / 30)
            recording.log(entity, rr.Image(image).compress(jpeg_quality=80))
    _disconnect(recording)


def test_real_rerun_query_reads_static_duration_rows_and_blob_free_camera_projection(
    tmp_path: Path,
) -> None:
    config = ChallengeConfig.load(CONFIG)
    path = tmp_path / "sparse.rrd"
    _write_sparse_rrd(path, config)

    with open_dataset_server("task6-sparse", rrd_paths=[path]) as dataset:
        episode = extract_aligned_episode(
            dataset,
            identity=IDENTITY,
            task_key="single_can",
            expected_sample_count=3,
            config=config,
        )

    assert episode.segment_id == "task6-segment"
    assert episode.identity == IDENTITY
    assert episode.joint_names == config.joint_names
    assert episode.frame.tolist() == [0, 1, 2]
    assert episode.action_time_ns.tolist() == [0, 33_333_333, 66_666_667]
    assert episode.state_present.tolist() == [True, True, True]
    assert episode.state_time_ns.tolist() == [0, 0, 66_666_666]
    assert episode.state_age_ns.tolist() == [0, 33_333_333, 1]
    assert episode.camera_present["front"].tolist() == [True, True, True]
    assert episode.camera_time_ns["front"].tolist() == [0, 0, 66_666_667]
    assert episode.camera_age_ns["front"].tolist() == [0, 33_333_333, 0]
    # The extraction audit captures the actual DataFusion projection, not the full schema.
    assert all("blob" not in name.lower() for name in episode.query_columns)
    assert any("media_type" in name for name in episode.query_columns)


def test_extractor_requires_exactly_one_segment(tmp_path: Path) -> None:
    config = ChallengeConfig.load(CONFIG)
    first = tmp_path / "a.rrd"
    second = tmp_path / "b.rrd"
    _write_sparse_rrd(first, config, "task6-segment-a")
    _write_sparse_rrd(second, config, "task6-segment-b")
    with open_dataset_server("two-segments", rrd_paths=[first, second]) as dataset:
        with pytest.raises(AlignmentError, match="exactly one segment") as raised:
            extract_aligned_episode(
                dataset,
                identity=IDENTITY,
                task_key="single_can",
                expected_sample_count=3,
                config=config,
            )
    assert raised.value.reason_codes == ("SEGMENT_COUNT",)
