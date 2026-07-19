from pathlib import Path

import pandas as pd
import pytest

from p5_rerun_port.catalog import EpisodeRecord
from p5_rerun_port.rerun_query import (
    EpisodeSegmentIdentity,
    aligned_vector_rows,
    compare_goal_vs_position,
    dataset_rrd_paths,
    episode_segment_identity,
    rrd_paths_for_records,
)


def test_dataset_paths_exclude_replays_and_unregistered_rrds(tmp_path: Path) -> None:
    root = tmp_path / "recordings" / "cans"
    root.mkdir(parents=True)
    for name in ("episode_01.rrd", "episode_01_replay.rrd", "orphan.rrd"):
        (root / name).write_bytes(b"rrd")
    (root / "episode_01.meta.json").write_text("{}")
    assert dataset_rrd_paths(tmp_path / "recordings", "cans") == [root / "episode_01.rrd"]


def test_rrd_paths_for_records_uses_only_selected_episode(tmp_path: Path) -> None:
    selected = tmp_path / "episode_02.rrd"
    selected.write_bytes(b"rrd")
    record = EpisodeRecord(dataset="cans", episode="episode_02", rrd_path=str(selected))
    assert rrd_paths_for_records([record]) == [selected.resolve()]


def test_episode_segment_identity_rejects_stale_catalog_path(tmp_path: Path) -> None:
    record = EpisodeRecord(
        dataset="cans",
        episode="episode_02",
        rrd_path=str(tmp_path / "episode_99.rrd"),
    )

    with pytest.raises(SystemExit, match="does not match"):
        episode_segment_identity(record)


def test_alignment_never_pairs_rows_from_different_segments() -> None:
    df = pd.DataFrame({
        "rerun_segment_id": ["a", "b"],
        "time": [0.0, 0.0],
        "/follower/goal:Scalars:scalars": [[1.0] * 7, [9.0] * 7],
        "/follower/position:Scalars:scalars": [[0.0] * 7, None],
    })
    result = aligned_vector_rows(df, goal_column=df.columns[2], state_column=df.columns[3])
    assert result.segment_ids == ("a",)
    assert result.goal.shape == result.state.shape == (1, 7)


def test_compare_does_not_label_requested_episode_across_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import p5_rerun_port.rerun_query as rerun_query

    df = pd.DataFrame({
        "rerun_segment_id": ["a", "b"],
        "/follower/goal:Scalars:scalars": [[1.0] * 7, [2.0] * 7],
        "/follower/position:Scalars:scalars": [[0.0] * 7, [1.0] * 7],
    })
    monkeypatch.setattr(rerun_query, "pick_timeline", lambda *args: "time")
    monkeypatch.setattr(rerun_query, "reader_to_pandas", lambda *args, **kwargs: df)

    class Dataset:
        def filter_contents(self, contents):
            return self

    result = compare_goal_vs_position(Dataset(), episode="episode_02")

    assert result.episode == "all"
    assert result.n_rows == 2


def test_compare_does_not_label_requested_episode_for_one_mismatched_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import p5_rerun_port.rerun_query as rerun_query

    df = pd.DataFrame({
        "rerun_segment_id": ["cans-episode_99"],
        "/follower/goal:Scalars:scalars": [[1.0] * 7],
        "/follower/position:Scalars:scalars": [[0.0] * 7],
    })
    monkeypatch.setattr(rerun_query, "pick_timeline", lambda *args: "time")
    monkeypatch.setattr(rerun_query, "reader_to_pandas", lambda *args, **kwargs: df)

    class Dataset:
        def filter_contents(self, contents):
            return self

    result = compare_goal_vs_position(
        Dataset(),
        episode="episode_02",
        verified_identity=EpisodeSegmentIdentity(
            episode="episode_02", segment_id="cans-episode_02"
        ),
    )

    assert result.episode == "all"
    assert result.n_rows == 1


@pytest.mark.parametrize("invalid_segment", [float("nan"), pd.NA, "", "   "])
def test_alignment_rejects_invalid_segment_identifiers(invalid_segment: object) -> None:
    df = pd.DataFrame({
        "rerun_segment_id": ["cans-episode_02", invalid_segment],
        "/follower/goal:Scalars:scalars": [[1.0] * 7, [2.0] * 7],
        "/follower/position:Scalars:scalars": [[0.0] * 7, [1.0] * 7],
    })

    result = aligned_vector_rows(df, goal_column=df.columns[1], state_column=df.columns[2])

    assert result.segment_ids == ("cans-episode_02",)
    assert result.goal.shape == result.state.shape == (1, 7)


def test_invalid_segment_identifier_prevents_episode_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import p5_rerun_port.rerun_query as rerun_query

    df = pd.DataFrame({
        "rerun_segment_id": ["cans-episode_02", pd.NA],
        "/follower/goal:Scalars:scalars": [[1.0] * 7, [2.0] * 7],
        "/follower/position:Scalars:scalars": [[0.0] * 7, [1.0] * 7],
    })
    monkeypatch.setattr(rerun_query, "pick_timeline", lambda *args: "time")
    monkeypatch.setattr(rerun_query, "reader_to_pandas", lambda *args, **kwargs: df)

    class Dataset:
        def filter_contents(self, contents):
            return self

    result = compare_goal_vs_position(
        Dataset(),
        episode="episode_02",
        verified_identity=EpisodeSegmentIdentity(
            episode="episode_02", segment_id="cans-episode_02"
        ),
    )

    assert result.episode == "all"


def test_cli_schema_uses_only_paths_from_selected_catalog_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from p5_rerun_port import query_api_cli

    selected = tmp_path / "episode_02.rrd"
    selected.write_bytes(b"rrd")
    selected_record = EpisodeRecord(
        dataset="cans", episode="episode_02", tag="Good episode", rrd_path=str(selected)
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        query_api_cli,
        "episodes_for_query",
        lambda **kwargs: [selected_record],
    )
    monkeypatch.setattr(query_api_cli, "_print_table", lambda rows: None)
    monkeypatch.setattr(
        query_api_cli,
        "schema_report",
        lambda dataset, recordings_dir, *, rrd_paths: captured.update(paths=rrd_paths) or "schema",
    )

    assert query_api_cli.main(["--dataset", "cans", "--tag", "Good episode", "--schema"]) == 0
    assert captured["paths"] == [selected.resolve()]


def test_cli_exits_before_query_server_when_filter_selects_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from p5_rerun_port import query_api_cli

    monkeypatch.setattr(query_api_cli, "episodes_for_query", lambda **kwargs: [])
    monkeypatch.setattr(
        query_api_cli,
        "schema_report",
        lambda *args, **kwargs: pytest.fail("Query API server must not start"),
    )

    with pytest.raises(SystemExit, match="no catalog episodes"):
        query_api_cli.main(["--dataset", "cans", "--episode", "episode_02", "--schema"])
