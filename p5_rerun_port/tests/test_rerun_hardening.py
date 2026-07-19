from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from p5_rerun_port.config import resolve_urdf_path
from p5_rerun_port.constants import FOLLOWER, JOINT_NAMES, REPO_ROOT
from p5_rerun_port.rerun_query import (
    compare_goal_vs_position,
    list_schema,
    open_dataset_server,
)
from p5_rerun_port.urdf_log import log_urdf


def test_default_urdf_resolves_to_vendored_model_with_meshes() -> None:
    path = resolve_urdf_path()

    assert path == (
        REPO_ROOT
        / "rebot_setup/vendor/reBotArm_control_py/urdf"
        / "reBot-DevArm_fixend_description/urdf/reBot-DevArm_fixend.urdf"
    )
    assert path.is_file()
    mesh_dir = path.parent.parent / "meshes"
    assert mesh_dir.is_dir()
    assert {p.name for p in mesh_dir.glob("*.STL")} >= {
        "base_link.STL",
        "link1.STL",
        "link6.STL",
        "end_link.STL",
    }


def test_vendored_urdf_logs_real_mesh_entities(tmp_path: Path) -> None:
    import rerun as rr

    rrd_path = tmp_path / "urdf.rrd"
    rec = rr.RecordingStream("deskpartner-urdf-test")
    rec.set_sinks(rr.FileSink(str(rrd_path)))

    used = log_urdf(rec)
    rec.disconnect()

    assert used == resolve_urdf_path()
    assert rrd_path.stat().st_size > 1_000_000
    with open_dataset_server("urdf-test", rrd_paths=[rrd_path]) as dataset:
        entities = list_schema(dataset)["entities"]

    assert f"/{FOLLOWER}/urdf/base_link/visual_0" in entities
    assert any(entity.endswith("/link6/visual_0") for entity in entities)
    assert f"/{FOLLOWER}/urdf/source" in entities


def test_urdf_mesh_failure_falls_back_to_static_source_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot name='test'/>", encoding="utf-8")
    calls: list[tuple[str, object, bool]] = []

    class BrokenLogger:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("missing mesh")

    class TextDocument:
        def __init__(self, text: str) -> None:
            self.text = text

    rec = SimpleNamespace(
        log=lambda entity, value, static=False: calls.append((entity, value, static))
    )
    monkeypatch.setitem(sys.modules, "rerun", SimpleNamespace(TextDocument=TextDocument))
    monkeypatch.setitem(
        sys.modules, "rerun_loader_urdf", SimpleNamespace(URDFLogger=BrokenLogger)
    )

    assert log_urdf(rec, urdf_path) == urdf_path
    assert len(calls) == 1
    entity, document, static = calls[0]
    assert entity == f"{FOLLOWER}/urdf"
    assert document.text == f"URDF path: {urdf_path}"
    assert static is True


def test_query_api_compares_small_real_recording(tmp_path: Path) -> None:
    import rerun as rr

    rrd_path = tmp_path / "tracking.rrd"
    rec = rr.RecordingStream("deskpartner-query-test")
    rec.set_sinks(rr.FileSink(str(rrd_path)))
    offsets = np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.float64)
    for frame in range(3):
        rec.set_time("time", sequence=frame)
        position = np.full(len(JOINT_NAMES), frame, dtype=np.float64)
        rec.log(f"{FOLLOWER}/position", rr.Scalars(position))
        rec.log(f"{FOLLOWER}/goal", rr.Scalars(position + offsets))
    rec.disconnect()

    with open_dataset_server("tracking", rrd_paths=[rrd_path]) as dataset:
        schema = list_schema(dataset)
        result = compare_goal_vs_position(dataset, episode="episode_smoke")

    assert f"/{FOLLOWER}/position" in schema["entities"]
    assert f"/{FOLLOWER}/goal" in schema["entities"]
    assert result.episode == "episode_smoke"
    assert result.n_rows == 3
    np.testing.assert_allclose(result.mean_abs_error, offsets)
    np.testing.assert_allclose(result.max_abs_error, offsets)
    assert result.rms_error == pytest.approx(float(np.sqrt(np.mean(offsets**2))))


def test_hackathon_blueprint_prioritizes_cameras_robot_and_tracking() -> None:
    import rerun.blueprint as rrb

    from p5_rerun_port.viewer_blueprint import build_hackathon_blueprint

    blueprint = build_hackathon_blueprint()
    assert isinstance(blueprint, rrb.Blueprint)
    root = blueprint.root_container
    views = []

    def collect(part) -> None:
        if hasattr(part, "class_identifier"):
            views.append(part)
        for child in getattr(part, "contents", ()):
            if not isinstance(child, str):
                collect(child)

    collect(root)
    names = {str(view.name) for view in views}
    assert names >= {"Front camera", "Side camera", "reBot URDF", "Goal vs position"}
    origins = {str(view.origin) for view in views}
    assert origins >= {"/camera/cam0", "/camera/cam1", "/follower/urdf", "/follower"}


def test_no_viewer_recording_never_creates_grpc_sink_or_blueprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import p5_rerun_port.takes as takes
    import p5_rerun_port.viewer_blueprint as viewer_blueprint

    events: list[tuple] = []

    class Recording:
        def spawn(self, **kwargs) -> None:
            events.append(("spawn", kwargs))

        def set_sinks(self, *sinks, **kwargs) -> None:
            events.append(("set_sinks", sinks, kwargs))

        def log(self, *_args, **_kwargs) -> None:
            pass

    fake_rr = SimpleNamespace(
        RecordingStream=lambda *_args, **_kwargs: Recording(),
        FileSink=lambda path: ("file", path),
        GrpcSink=lambda: pytest.fail("headless mode created a GrpcSink"),
        TextDocument=lambda text: text,
    )
    monkeypatch.setattr(takes, "_import_rerun", lambda: fake_rr)
    monkeypatch.setattr(
        viewer_blueprint,
        "build_hackathon_blueprint",
        lambda: pytest.fail("headless mode built a Viewer blueprint"),
    )

    takes.begin_recording(
        tmp_path / "headless.rrd",
        episode="episode_01",
        dataset="cans",
        task="pick can",
        spawn_viewer=False,
        save_only=True,
    )

    assert events == [
        (
            "set_sinks",
            (("file", str(tmp_path / "headless.rrd")),),
            {"default_blueprint": None},
        )
    ]


def test_interactive_recording_activates_hackathon_blueprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import p5_rerun_port.takes as takes
    import p5_rerun_port.viewer_blueprint as viewer_blueprint

    events: list[tuple] = []
    marker = object()

    class Recording:
        def spawn(self, **kwargs) -> None:
            events.append(("spawn", kwargs))

        def set_sinks(self, *sinks, **kwargs) -> None:
            events.append(("set_sinks", sinks, kwargs))

        def send_blueprint(self, blueprint, **kwargs) -> None:
            events.append(("send_blueprint", blueprint, kwargs))

        def log(self, *_args, **_kwargs) -> None:
            pass

    fake_rr = SimpleNamespace(
        RecordingStream=lambda *_args, **_kwargs: Recording(),
        FileSink=lambda path: ("file", path),
        GrpcSink=lambda: ("grpc",),
        TextDocument=lambda text: text,
    )
    monkeypatch.setattr(takes, "_import_rerun", lambda: fake_rr)
    monkeypatch.setattr(viewer_blueprint, "build_hackathon_blueprint", lambda: marker)

    takes.begin_recording(
        tmp_path / "viewer.rrd",
        episode="episode_01",
        dataset="cans",
        task="pick can",
        spawn_viewer=True,
        save_only=False,
    )

    assert events[0] == (
        "spawn",
        {
            "connect": False,
            "hide_welcome_screen": True,
            "default_blueprint": marker,
        },
    )
    assert events[1][0] == "set_sinks"
    assert events[1][2] == {"default_blueprint": marker}
    assert events[2] == (
        "send_blueprint",
        marker,
        {"make_active": True, "make_default": True},
    )
