from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
from typing import Any, Iterator

import numpy as np
from PIL import Image
import pytest
import rerun as rr

from p5_rerun_port.challenge import failure_canonical
from p5_rerun_port.challenge.config import ChallengeConfig
from p5_rerun_port.challenge.failure_canonical import IdentityError, materialize_failure
from p5_rerun_port.challenge.models import EpisodeIdentity, InventoryRow
from p5_rerun_port.rerun_query import list_schema, open_dataset_server, reader_to_pandas


CONFIG = Path("config/rerun_query_challenge.yaml")


@pytest.fixture
def config() -> ChallengeConfig:
    return ChallengeConfig.load(CONFIG)


@pytest.fixture
def row(config: ChallengeConfig) -> InventoryRow:
    source = next(source for source in config.sources if source.role == "failure")
    attempt_id = "20260719T020311.738279Z-966741693b"
    return InventoryRow(
        identity=EpisodeIdentity(source.repo_id, source.revision, attempt_id),
        role="failure",
        task_key=source.task_key,
        source_path=f"failed_attempts/{attempt_id}/attempt.rrd",
        episode_index=None,
        attempt_id=attempt_id,
        frame_count=3,
        captured_at="2026-07-19T02:03:11.738279Z",
    )


def _jpeg_bytes(rgb: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (4, 3), rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _source_path(tmp_path: Path, row: InventoryRow) -> Path:
    path = tmp_path.joinpath(*Path(row.source_path).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_native_rrd(
    path: Path,
    row: InventoryRow,
    config: ChallengeConfig,
    *,
    frames: tuple[int, ...] = (2, 0, 1),
    missing_joint: str | None = None,
    missing_camera: str | None = None,
    metadata_attempt_id: str | None = None,
    omit_metadata: bool = False,
    result_only: bool = False,
    segment_id: str = "native-segment",
) -> None:
    recording = rr.RecordingStream("native-failure-fixture", recording_id=segment_id)
    recording.set_sinks(rr.FileSink(str(path)))
    metadata = {
        "attempt_id": row.attempt_id if metadata_attempt_id is None else metadata_attempt_id,
        "disposition": "failed",
        "failure_label": "dropped_object",
        "training_included": False,
        "note": "must not enter canonical output",
    }
    if not omit_metadata:
        recording.log("/attempt/metadata", rr.TextDocument(json.dumps(metadata)), static=True)
    recording.log(
        "/attempt/result",
        rr.TextDocument(json.dumps({"outcome": "failure", "label": "dropped_object"})),
        static=True,
    )
    if not result_only:
        front = _jpeg_bytes((220, 20, 10))
        side = _jpeg_bytes((10, 30, 220))
        for frame in frames:
            recording.set_time("attempt_frame", sequence=frame)
            recording.set_time("attempt_time", duration=frame / config.fps)
            for joint_index, joint in enumerate(config.joint_names):
                if missing_joint == f"action/{joint}":
                    continue
                recording.log(
                    f"/action/{joint}/pos",
                    rr.Scalars([frame + joint_index / 10]),
                )
                if missing_joint == f"observation/{joint}":
                    continue
                recording.log(
                    f"/observation/{joint}/pos",
                    rr.Scalars([frame + joint_index / 10 + 0.25]),
                )
            if missing_camera != "front":
                recording.log(
                    "/observation/front",
                    rr.EncodedImage(contents=front, media_type="image/jpeg"),
                )
            if missing_camera != "side":
                recording.log(
                    "/observation/side",
                    rr.EncodedImage(contents=side, media_type="image/jpeg"),
                )
    recording.disconnect()


def _component_column(frame: Any, marker: str) -> str:
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


def test_canonicalizes_real_native_rrd_through_query_and_excludes_labels(
    config: ChallengeConfig, row: InventoryRow, tmp_path: Path
) -> None:
    source = _source_path(tmp_path, row)
    output = tmp_path / "canonical.rrd"
    _write_native_rrd(source, row, config)

    artifact = materialize_failure(row, config, source, output)

    assert artifact.status == "ready", artifact
    assert artifact.identity == row.identity.canonical
    assert artifact.rrd_path == output
    assert artifact.frame_count == 3
    assert artifact.reason_codes == ()
    with open_dataset_server("canonical-native", rrd_paths=[output]) as dataset:
        schema = list_schema(dataset)
        entities = set(schema["entities"])
        assert {"/follower/goal", "/follower/position", "/camera/cam0", "/camera/cam1"} <= entities
        assert not any(
            token in entity
            for entity in entities
            for token in ("attempt", "result", "disposition", "failure", "training", "note")
        )

        goal = reader_to_pandas(dataset, index="frame", contents=["/follower/goal"])
        state = reader_to_pandas(dataset, index="frame", contents=["/follower/position"])
        assert goal["frame"].tolist() == [0, 1, 2]
        assert state["frame"].tolist() == [0, 1, 2]
        assert goal["time"].dt.total_seconds().to_numpy() == pytest.approx([0, 1 / 30, 2 / 30])
        goal_col = _component_column(goal, "Scalars:scalars")
        state_col = _component_column(state, "Scalars:scalars")
        assert np.asarray(goal[goal_col].iloc[2]).reshape(-1) == pytest.approx(
            [2 + index / 10 for index in range(7)]
        )
        assert np.asarray(state[state_col].iloc[0]).reshape(-1) == pytest.approx(
            [index / 10 + 0.25 for index in range(7)]
        )

        for entity in ("/camera/cam0", "/camera/cam1"):
            camera = reader_to_pandas(dataset, index="frame", contents=[entity])
            assert camera["frame"].tolist() == [0, 1, 2]
            assert [
                _text_value(value)
                for value in camera[_component_column(camera, "media_type")]
            ] == ["image/jpeg"] * 3
            blob = _bytes_value(camera[_component_column(camera, "EncodedImage:blob")].iloc[0])
            assert Image.open(io.BytesIO(blob)).size == (4, 3)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing_action", "MISSING_ACTION_JOINT"),
        ("missing_state", "MISSING_STATE_JOINT"),
        ("missing_front", "MISSING_FRONT_CAMERA"),
        ("missing_side", "MISSING_SIDE_CAMERA"),
        ("frame_gap", "FRAME_INDEX_NONCONTIGUOUS"),
        ("duplicate_frame", "FRAME_COUNT_MISMATCH"),
        ("result_only", "MISSING_ACTION_JOINT"),
        ("missing_metadata", "STATIC_IDENTITY_MISSING"),
        ("metadata_mismatch", "STATIC_IDENTITY_MISMATCH"),
    ],
)
def test_rejects_structurally_untrustworthy_native_rrds_with_stable_reasons(
    mutation: str,
    reason: str,
    config: ChallengeConfig,
    row: InventoryRow,
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path, row)
    kwargs: dict[str, Any] = {}
    if mutation == "missing_action":
        kwargs["missing_joint"] = f"action/{config.joint_names[2]}"
    elif mutation == "missing_state":
        kwargs["missing_joint"] = f"observation/{config.joint_names[4]}"
    elif mutation == "missing_side":
        kwargs["missing_camera"] = "side"
    elif mutation == "missing_front":
        kwargs["missing_camera"] = "front"
    elif mutation == "frame_gap":
        kwargs["frames"] = (0, 2, 3)
    elif mutation == "duplicate_frame":
        # Query latest-at collapses same-index writes; inventory count still fails closed.
        kwargs["frames"] = (0, 1, 1)
    elif mutation == "result_only":
        kwargs["result_only"] = True
    elif mutation == "missing_metadata":
        kwargs["omit_metadata"] = True
    else:
        kwargs["metadata_attempt_id"] = "different-attempt"
    _write_native_rrd(source, row, config, **kwargs)

    artifact = materialize_failure(row, config, source, tmp_path / "rejected.rrd")

    assert artifact.status == "rejected"
    assert artifact.rrd_path is None
    assert artifact.sha256 is None
    assert artifact.reason_codes == (reason,)
    assert not (tmp_path / "rejected.rrd").exists()


@pytest.mark.parametrize(
    "mismatch",
    ["repo", "revision", "role", "task", "attempt", "remote_path", "actual_path"],
)
def test_authenticates_exact_failure_source_and_actual_pinned_path_before_query(
    mismatch: str,
    monkeypatch: pytest.MonkeyPatch,
    config: ChallengeConfig,
    row: InventoryRow,
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path, row)
    _write_native_rrd(source, row, config)
    if mismatch == "repo":
        row = replace(row, identity=replace(row.identity, repo_id="Cornerf/unconfigured"))
    elif mismatch == "revision":
        row = replace(row, identity=replace(row.identity, revision="0" * 40))
    elif mismatch == "role":
        row = replace(row, role="success")
    elif mismatch == "task":
        row = replace(row, task_key="two_can")
    elif mismatch == "attempt":
        row = replace(row, attempt_id="different-attempt")
    elif mismatch == "remote_path":
        row = replace(row, source_path=f"failed_attempts/{row.attempt_id}/replay.rrd")
    else:
        source = tmp_path / "attempt.rrd"
        _write_native_rrd(source, row, config)

    @contextmanager
    def should_not_open(*args: Any, **kwargs: Any) -> Iterator[Any]:
        raise AssertionError("Query server opened before source authentication")
        yield

    monkeypatch.setattr(failure_canonical, "open_dataset_server", should_not_open)

    artifact = materialize_failure(row, config, source, tmp_path / "untrusted.rrd")

    assert artifact.status == "rejected"
    assert artifact.reason_codes == ("SOURCE_LOCK_MISMATCH",)


def test_raises_identity_error_for_real_rrd_with_two_segments(
    config: ChallengeConfig, row: InventoryRow, tmp_path: Path
) -> None:
    first = tmp_path / "first.rrd"
    second = tmp_path / "second.rrd"
    source = _source_path(tmp_path, row)
    _write_native_rrd(first, row, config, segment_id="segment-one")
    _write_native_rrd(second, row, config, segment_id="segment-two")
    subprocess.run(
        [
            str(Path(__import__("sys").executable).with_name("rerun")),
            "rrd",
            "merge",
            "--output",
            str(source),
            str(first),
            str(second),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with pytest.raises(IdentityError, match="exactly one segment"):
        materialize_failure(row, config, source, tmp_path / "ambiguous.rrd")


def test_accepts_authoritative_partial_filename_for_integrity_evaluation(
    config: ChallengeConfig, row: InventoryRow, tmp_path: Path
) -> None:
    row = replace(
        row,
        source_path=f"failed_attempts/{row.attempt_id}/attempt.partial.rrd",
    )
    source = _source_path(tmp_path, row)
    _write_native_rrd(source, row, config)

    artifact = materialize_failure(row, config, source, tmp_path / "partial-canonical.rrd")

    assert artifact.status == "ready", artifact


def test_query_api_failure_rejects_without_output(
    monkeypatch: pytest.MonkeyPatch,
    config: ChallengeConfig,
    row: InventoryRow,
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path, row)
    _write_native_rrd(source, row, config)

    @contextmanager
    def broken_query(*args: Any, **kwargs: Any) -> Iterator[Any]:
        raise RuntimeError("catalog unavailable")
        yield

    monkeypatch.setattr(failure_canonical, "open_dataset_server", broken_query)

    artifact = materialize_failure(row, config, source, tmp_path / "broken.rrd")

    assert artifact.status == "rejected"
    assert artifact.reason_codes == ("SOURCE_QUERY_FAILED",)
    assert not (tmp_path / "broken.rrd").exists()
