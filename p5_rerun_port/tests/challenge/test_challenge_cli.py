from __future__ import annotations

from pathlib import Path
import sys

from p5_rerun_port import query_challenge_cli


def test_help_lists_workflow_commands_and_challenge_verbs(capsys) -> None:
    assert query_challenge_cli.main(["--help"]) == 0
    output = capsys.readouterr().out.lower()
    for command in ("inventory", "materialize", "audit", "evaluate", "prepare", "run"):
        assert command in output
    for verb in ("inspect", "align", "filter", "compare", "transform", "evaluate", "prepare"):
        assert verb in output


def test_run_orders_stages_and_prints_machine_readable_outputs(tmp_path: Path, capsys) -> None:
    calls: list[str] = []

    def stage(name: str):
        def run(context, state):
            calls.append(name)
            assert context.config_path == Path("config.yaml").resolve()
            assert context.artifacts_root == tmp_path.resolve()
            return state.with_updates(
                run_id="abc123",
                report_html=tmp_path / "abc123" / "report.html",
                selection_manifest=tmp_path / "abc123" / "selection-manifest.json",
                derivative_root=tmp_path / "derivative",
            )

        return run

    stages = {name: stage(name) for name in query_challenge_cli.STAGE_ORDER}
    assert query_challenge_cli.main(
        ["run", "--config", "config.yaml", "--artifacts-root", str(tmp_path)],
        stage_functions=stages,
    ) == 0

    assert calls == list(query_challenge_cli.STAGE_ORDER)
    output = capsys.readouterr().out
    assert "RUN_ID=abc123" in output
    assert f"REPORT_HTML={(tmp_path / 'abc123' / 'report.html').resolve()}" in output
    assert f"SELECTION_MANIFEST={(tmp_path / 'abc123' / 'selection-manifest.json').resolve()}" in output
    assert f"DERIVATIVE_ROOT={(tmp_path / 'derivative').resolve()}" in output


def test_run_stops_after_first_stage_error(tmp_path: Path, capsys) -> None:
    calls: list[str] = []

    def ok(context, state):
        calls.append("inventory")
        return state

    def fail(context, state):
        calls.append("materialize")
        raise RuntimeError("fixture failure")

    def unexpected(context, state):
        calls.append("unexpected")
        return state

    stages = {
        "inventory": ok,
        "materialize": fail,
        "audit": unexpected,
        "evaluate": unexpected,
        "prepare": unexpected,
    }
    assert query_challenge_cli.main(
        ["run", "--config", "config.yaml", "--artifacts-root", str(tmp_path)],
        stage_functions=stages,
    ) == 1
    assert calls == ["inventory", "materialize"]
    assert "fixture failure" in capsys.readouterr().err


def test_cli_imports_no_hardware_modules_and_prepare_has_no_upload_flag(capsys) -> None:
    assert "p5_rerun_port.hardware" not in sys.modules


def test_default_inventory_stage_uses_real_workflow_boundary(tmp_path: Path, monkeypatch, capsys) -> None:
    from p5_rerun_port.challenge import workflow

    config_path = tmp_path / "config.yaml"
    config_path.write_text("fixture", encoding="utf-8")
    inventory = (type("Row", (), {"identity": type("Identity", (), {"canonical": "source@sha:0"})()})(),)

    class Config:
        @classmethod
        def load(cls, path):
            assert path == config_path.resolve()
            return cls()

    class Hub:
        pass

    def build(config, hub):
        assert isinstance(config, Config)
        assert isinstance(hub, Hub)
        return inventory

    def write(path, config, rows, *, config_path):
        assert rows is inventory
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return {"inventory_payload_digest": "locked"}

    monkeypatch.setattr(workflow, "_inventory_dependencies", lambda: (Config, Hub, build, write))
    monkeypatch.setattr(workflow, "_write_cache", lambda context, cache: None)
    assert query_challenge_cli.main(
        ["inventory", "--config", str(config_path), "--artifacts-root", str(tmp_path / "artifacts")]
    ) == 0
    assert "RUN_ID=" in capsys.readouterr().out
    assert query_challenge_cli.main(["prepare", "--help"]) == 0
    output = capsys.readouterr().out.lower()
    assert "upload" not in output
    assert "publish" not in output
    assert "p5_rerun_port.hardware" not in sys.modules
