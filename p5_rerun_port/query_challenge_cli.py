"""Competition CLI for the revision-locked Rerun Query quality gate.

The module deliberately imports only the Python standard library at import
time.  Data, Rerun, LeRobot, and Hugging Face dependencies are loaded by the
stage implementation that needs them; robot hardware modules are never used.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import importlib
from pathlib import Path
import sys
from typing import Any


STAGE_ORDER = ("inventory", "materialize", "audit", "evaluate", "prepare")


@dataclass(frozen=True)
class WorkflowContext:
    config_path: Path
    artifacts_root: Path


@dataclass(frozen=True)
class WorkflowState:
    run_id: str | None = None
    report_html: Path | None = None
    selection_manifest: Path | None = None
    derivative_root: Path | None = None
    payload: Any = None

    def with_updates(self, **changes: Any) -> "WorkflowState":
        return replace(self, **changes)


StageFunction = Callable[[WorkflowContext, WorkflowState], WorkflowState]


def _lazy_stage(name: str) -> StageFunction:
    """Resolve the data-heavy workflow implementation only when invoked."""

    def invoke(context: WorkflowContext, state: WorkflowState) -> WorkflowState:
        module = importlib.import_module("p5_rerun_port.challenge.workflow")
        function = getattr(module, f"run_{name}_stage")
        result = function(context, state)
        if not isinstance(result, WorkflowState):
            raise TypeError(f"{name} stage must return WorkflowState")
        return result

    return invoke


DEFAULT_STAGE_FUNCTIONS: Mapping[str, StageFunction] = {
    name: _lazy_stage(name) for name in STAGE_ORDER
}


def _parser() -> argparse.ArgumentParser:
    description = (
        "Revision-locked reBot data quality gate using the Rerun Query API to "
        "Inspect, Align, Filter, Compare, Transform, Evaluate, and Prepare training data."
    )
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="command", required=True)
    help_text = {
        "inventory": "Inspect and lock all 102 source items",
        "materialize": "Transform pinned sources into canonical RRDs",
        "audit": "Align, compare, and filter Query API evidence",
        "evaluate": "Evaluate frozen verdicts after label reveal",
        "prepare": "Prepare the local-only manifest, report, and derivative",
        "run": "Run inventory through local preparation in order",
    }
    for command in (*STAGE_ORDER, "run"):
        child = subparsers.add_parser(command, help=help_text[command], description=help_text[command])
        child.add_argument(
            "--config",
            type=Path,
            default=Path("config/rerun_query_challenge.yaml"),
            help="Checked-in challenge source lock",
        )
        child.add_argument(
            "--artifacts-root",
            type=Path,
            default=Path("artifacts/rerun-query"),
            help="Local immutable run directory root",
        )
    return parser


def _commands_for(command: str) -> Sequence[str]:
    if command == "run":
        return STAGE_ORDER
    return (command,)


def _print_outputs(state: WorkflowState) -> None:
    if state.run_id is not None:
        print(f"RUN_ID={state.run_id}")
    if state.report_html is not None:
        print(f"REPORT_HTML={state.report_html.resolve()}")
    if state.selection_manifest is not None:
        print(f"SELECTION_MANIFEST={state.selection_manifest.resolve()}")
    if state.derivative_root is not None:
        print(f"DERIVATIVE_ROOT={state.derivative_root.resolve()}")


def main(
    argv: list[str] | None = None,
    *,
    stage_functions: Mapping[str, StageFunction] | None = None,
) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        if error.code == 0:
            return 0
        raise

    context = WorkflowContext(args.config.resolve(), args.artifacts_root.resolve())
    stages = DEFAULT_STAGE_FUNCTIONS if stage_functions is None else stage_functions
    state = WorkflowState()
    try:
        for name in _commands_for(args.command):
            state = stages[name](context, state)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    _print_outputs(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
