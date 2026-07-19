"""Gated command-line orchestration for offline and physical policy rollout.

This module intentionally imports only the Python standard library. Optional
policy, NumPy, LeRobot, robot, SDK, and terminal helpers are imported only by
the subcommand handler that needs them.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import IO, Any


LIVE_OPERATOR_PHRASE = "I HAVE AN E-STOP OPERATOR"
EMPTY_WORKSPACE_PHRASE = "WORKSPACE IS EMPTY"
MIN_SPEED_SCALE = 0.10
MAX_SPEED_SCALE = 0.20
MAX_CLI_CYCLES = 20
MANUAL_RESET_PHRASE = "I RESET THE CAN AND CLEARED THE WORKSPACE"
EXPECTED_IMAGE_ORDER = (
    "observation.images.front",
    "observation.images.side",
)
EXPECTED_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)
EXPECTED_COORDINATE_FRAME = (
    "follower_degrees_after_direction_limits_and_step_cap"
)
EXPECTED_CONTROL_MODE = "absolute joint pose"
PROFILE_RUNTIME_ENTRIES = (
    "follower",
    "leader",
    "follower_driver_contract",
    "follower_base_implementation",
    "follower_dm_implementation",
    "leader_driver_contract",
    "leader_implementation",
)
CAMERA_METADATA_KEYS = (
    "excluded_screen_index",
    "excluded_screen_name",
    "minimum_measured_fps",
)


HELP_EPILOG = """
PHYSICAL SAFETY WARNING: software stop is not a physical e-stop. A dedicated
operator must hold the physical e-stop/power cut throughout every live cycle.

Required staged rollout:

# Gate A: offline only
python -m p3_vlm_orchestrator.policy_rollout.cli offline --checkpoint CHECKPOINT --dataset DATASET --episodes 2

# Gate B: hardware observation + prediction only, no send
python -m p3_vlm_orchestrator.policy_rollout.cli shadow --checkpoint CHECKPOINT --cycles 20

# Gate C: empty workspace, one cycle, 10% speed
python -m p3_vlm_orchestrator.policy_rollout.cli live --checkpoint CHECKPOINT --cycles 1 --speed-scale 0.10 --live

# Gate D: only after Gate C has no faults/clamps, up to five cycles
python -m p3_vlm_orchestrator.policy_rollout.cli live --checkpoint CHECKPOINT --cycles 5 --speed-scale 0.10 --live

Explicit held-out episode evaluation (Gate B-D behavior is unchanged unless
--episode is present): s = success, f = failure, q/x/Esc = stop. Each attempt
ends at the 30.0-second cap or 300 confirmed live actions. Stop and the physical
e-stop always take priority. --retry-on-failure permits at most one retry only
after an explicit manual reset acknowledgement; it never retries a safety
fault, timeout, or stop.
"""


@dataclass
class CliDependencies:
    """Injectable boundaries used by tests and resolved lazily in production."""

    stdout: IO[str] | None = None
    stderr: IO[str] | None = None
    input_fn: Callable[[str], str] | None = None
    utc_now: Callable[[], datetime] | None = None
    monotonic_clock: Callable[[], float] | None = None
    checkpoint_loader: Callable[[Path], object] | None = None
    offline_evaluator: Callable[..., object] | None = None
    policy_factory: Callable[[object, str], object] | None = None
    dummy_policy_factory: Callable[[], object] | None = None
    safety_factory: Callable[..., object] | None = None
    guard_factory: Callable[..., object] | None = None
    robot_factory: Callable[..., object] | None = None
    runner_factory: Callable[..., object] | None = None
    keyboard_stop_factory: Callable[[], object] | None = None
    serial_port_is_free: Callable[[str], bool] | None = None
    repo_root: Path | None = None

    def output(self) -> IO[str]:
        return self.stdout if self.stdout is not None else sys.stdout

    def errors(self) -> IO[str]:
        return self.stderr if self.stderr is not None else sys.stderr

    def read_input(self) -> Callable[[str], str]:
        return self.input_fn if self.input_fn is not None else input

    def now_utc(self) -> datetime:
        now = self.utc_now() if self.utc_now is not None else datetime.now(timezone.utc)
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("UTC clock must return a timezone-aware datetime")
        return now.astimezone(timezone.utc)

    def monotonic(self) -> Callable[[], float]:
        return self.monotonic_clock or time.monotonic

    def repository_root(self) -> Path:
        return self.repo_root if self.repo_root is not None else _repo_root()


@dataclass(frozen=True)
class _TerminalSummary:
    cycles_completed: int
    actions_attempted: int
    actions_confirmed: int
    terminal_reason: str
    primary_fault_reason: str | None
    cleanup_fault_reason: str | None
    audit_fault_reason: str | None
    attempt: int = 1
    elapsed_seconds: float = 0.0
    clamp_count: int = 0


class PreflightRobotAdapter:
    """Connect once, validate one discarded preflight, then delegate fresh reads."""

    def __init__(
        self,
        *,
        robot: object,
        expected_task: str,
        profile_snapshot: Mapping[str, Any],
    ) -> None:
        self.robot = robot
        self.expected_task = expected_task
        self.profile_snapshot = profile_snapshot
        self._connect_attempted = False
        self._connected = False
        self._disconnected = False
        self.cleanup_fault_reason: str | None = None

    def connect(self) -> None:
        if self._connect_attempted:
            raise RuntimeError("Preflight robot adapter cannot connect twice")
        self._connect_attempted = True
        try:
            self.robot.connect()
            self._connected = True
            observation = self.robot.observe()
            _validate_preflight_observation(
                observation,
                expected_task=self.expected_task,
                profile_snapshot=self.profile_snapshot,
            )
        except Exception:
            try:
                self._disconnect_once()
            except Exception:
                pass
            raise

    def observe(self) -> object:
        if not self._connected or self._disconnected:
            raise RuntimeError("Preflight robot adapter is not connected")
        return self.robot.observe()

    def send_action(self, action_deg: object) -> object:
        if not self._connected or self._disconnected:
            raise RuntimeError("Preflight robot adapter is not connected")
        return self.robot.send_action(action_deg)

    def disconnect(self) -> None:
        self._disconnect_once()

    def _disconnect_once(self) -> None:
        if self._disconnected or not self._connect_attempted:
            return
        self._disconnected = True
        self._connected = False
        try:
            self.robot.disconnect()
        except Exception as exc:
            self.cleanup_fault_reason = _exception_text(exc)
            raise


def canonical_profile_digest(profile: object) -> str:
    """Return the same canonical SHA-256 used by checkpoint sidecars."""

    payload = json.dumps(
        profile,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def default_serial_port_is_free(
    port: str,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> bool:
    """Use ``lsof <port>`` and fail closed on owners or checker errors."""

    if not isinstance(port, str) or not port:
        return False
    try:
        result = runner(
            ["lsof", port],
            capture_output=True,
            text=True,
            check=False,
        )
        returncode = getattr(result, "returncode", None)
        stdout = getattr(result, "stdout", "")
        stderr = getattr(result, "stderr", "")
    except Exception:
        return False
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        return False
    if stdout.strip() or stderr.strip():
        return False
    # lsof returns one when it found no matching open file. Every other status
    # is either an owner (zero) or a checker failure (greater than one).
    return returncode == 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cycles(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cycles must be an integer") from exc
    if not 1 <= result <= MAX_CLI_CYCLES:
        raise argparse.ArgumentTypeError(
            f"cycles must be within [1, {MAX_CLI_CYCLES}]"
        )
    return result


def _episodes(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("episodes must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("episodes must be positive")
    return result


def _add_hardware_arguments(parser: argparse.ArgumentParser) -> None:
    root = _repo_root()
    runtime_default = Path(
        os.environ.get(
            "RUNTIME_ROOT",
            root / "rebot_setup" / "vendor" / "rebot_lerobot",
        )
    )
    parser.add_argument("--cycles", type=_cycles)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speed-scale", type=float, default=MIN_SPEED_SCALE)
    parser.add_argument("--runtime-root", type=Path, default=runtime_default)
    parser.add_argument("--arm-config", type=Path, default=root / "config" / "arm.yaml")
    parser.add_argument(
        "--workspace-config",
        type=Path,
        default=root / "config" / "workspace.yaml",
    )
    parser.add_argument(
        "--workspace-calibration",
        type=Path,
        default=root / "data" / "calibration" / "calibration.json",
    )
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument(
        "--episode",
        action="store_true",
        help="Run one 30-second, human-verdict held-out evaluation attempt.",
    )
    parser.add_argument(
        "--retry-on-failure",
        action="store_true",
        help="After manual reset acknowledgement, retry operator failure once.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safety-gated ReBot learned-policy rollout.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Validate and describe a checkpoint without loading policy weights.",
    )
    inspect_parser.add_argument("--checkpoint", required=True, type=Path)

    offline_parser = subparsers.add_parser(
        "offline",
        help="Gate A: evaluate a checkpoint over finalized recorded episodes.",
    )
    offline_parser.add_argument("--checkpoint", required=True, type=Path)
    offline_parser.add_argument("--dataset", required=True, type=Path)
    offline_parser.add_argument("--episodes", type=_episodes, default=2)
    offline_parser.add_argument("--device", default="cpu")

    shadow_parser = subparsers.add_parser(
        "shadow",
        help="Gate B: observe and predict without sending an action.",
    )
    shadow_source = shadow_parser.add_mutually_exclusive_group(required=True)
    shadow_source.add_argument("--checkpoint", type=Path)
    shadow_source.add_argument("--dummy-hold", action="store_true")
    shadow_parser.add_argument("--profile", type=Path)
    _add_hardware_arguments(shadow_parser)
    shadow_parser.set_defaults(cycles=20)

    live_parser = subparsers.add_parser(
        "live",
        help="Gates C-D: bounded physical action after every safety gate passes.",
    )
    live_parser.add_argument("--checkpoint", required=True, type=Path)
    live_parser.add_argument(
        "--live",
        dest="live_gate",
        action="store_true",
        help="Required acknowledgement that this subcommand may send motion.",
    )
    _add_hardware_arguments(live_parser)
    live_parser.set_defaults(cycles=1)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: CliDependencies | None = None,
) -> int:
    deps = dependencies or CliDependencies()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            return _run_inspect(args, deps)
        if args.command == "offline":
            return _run_offline(args, deps)
        if args.command == "shadow":
            return _run_hardware(args, deps, mode="shadow")
        if args.command == "live":
            return _run_hardware(args, deps, mode="live")
        raise ValueError(f"Unsupported rollout command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {_exception_text(exc)}", file=deps.errors())
        return 2


def _checkpoint_loader(deps: CliDependencies) -> Callable[[Path], object]:
    if deps.checkpoint_loader is not None:
        return deps.checkpoint_loader
    from rebot_operator_kit.rollout.checkpoint import CheckpointBundle

    return CheckpointBundle.load


def _run_inspect(args: argparse.Namespace, deps: CliDependencies) -> int:
    bundle = _checkpoint_loader(deps)(args.checkpoint)
    config = _load_json_object(Path(bundle.path) / "config.json", "policy config")
    policy_type = config.get("type", config.get("policy_type", "unknown"))
    processor_artifacts = _processor_artifact_report(Path(bundle.path))
    coordinates = bundle.profile_snapshot.get("coordinate_contract", {})
    print(f"checkpoint={bundle.path}", file=deps.output())
    print(f"locked_task={bundle.task}", file=deps.output())
    print(f"policy_type={policy_type}", file=deps.output())
    print(
        "policy_config="
        + json.dumps(config, sort_keys=True, separators=(",", ":")),
        file=deps.output(),
    )
    print(
        "action_contract="
        f"dimension={bundle.action_dimension} chunk_size={bundle.chunk_size} "
        f"action_steps={bundle.action_steps} "
        f"frame={coordinates.get('frame')} control_mode={coordinates.get('control_mode')}",
        file=deps.output(),
    )
    joints = coordinates.get("joints", [])
    joint_names = [
        joint.get("name") for joint in joints if isinstance(joint, Mapping)
    ]
    print("joint_order=" + ",".join(joint_names), file=deps.output())
    print("image_order=" + ",".join(bundle.image_order), file=deps.output())
    print(
        "processor_artifacts=" + ",".join(processor_artifacts),
        file=deps.output(),
    )
    print(f"profile_digest={bundle.profile_digest}", file=deps.output())
    return 0


def _run_offline(args: argparse.Namespace, deps: CliDependencies) -> int:
    bundle = _checkpoint_loader(deps)(args.checkpoint)
    if deps.offline_evaluator is not None:
        evaluator = deps.offline_evaluator
    else:
        from p3_vlm_orchestrator.policy_rollout.offline import evaluate_checkpoint

        evaluator = evaluate_checkpoint
    results = evaluator(
        bundle,
        args.dataset,
        episodes=args.episodes,
        device=args.device,
        output=deps.output(),
    )
    if not results:
        raise RuntimeError("No samples were found in the selected dataset episodes")
    return 0


def _run_hardware(
    args: argparse.Namespace,
    deps: CliDependencies,
    *,
    mode: str,
) -> int:
    if mode == "live" and not args.live_gate:
        raise ValueError("live rollout requires the explicit --live flag")
    if args.retry_on_failure and not args.episode:
        raise ValueError("--retry-on-failure requires explicit --episode mode")
    speed_scale = _checked_speed_scale(args.speed_scale)
    cycles = args.cycles
    if not 1 <= cycles <= MAX_CLI_CYCLES:
        raise ValueError(f"cycles must be within [1, {MAX_CLI_CYCLES}]")
    log_path = _resolve_log_path(
        args.log_path,
        deps.now_utc(),
        repo_root=deps.repository_root(),
    )

    bundle: object | None = None
    if mode == "shadow" and args.dummy_hold:
        if args.profile is None:
            raise ValueError("shadow --dummy-hold requires --profile PATH")
        profile_snapshot = _load_standalone_profile(args.profile)
        profile_digest = canonical_profile_digest(profile_snapshot)
        profile_authentication = "standalone-untrusted"
        task = profile_snapshot["collection_defaults"]["task"]
        policy = _dummy_policy_factory(deps)()
    else:
        bundle = _checkpoint_loader(deps)(args.checkpoint)
        profile_snapshot = bundle.profile_snapshot
        profile_digest = bundle.profile_digest
        profile_authentication = "checkpoint-sidecar-verified"
        task = bundle.task
        processor_artifacts = _processor_artifact_report(Path(bundle.path))
        print(
            "processor_artifacts=" + ",".join(processor_artifacts),
            file=deps.output(),
        )
        policy = _policy_factory(deps)(bundle, args.device)

    if args.episode:
        return _run_episode_hardware(
            args,
            deps,
            mode=mode,
            speed_scale=speed_scale,
            log_path=log_path,
            profile_snapshot=profile_snapshot,
            profile_digest=profile_digest,
            profile_authentication=profile_authentication,
            task=task,
            policy=policy,
        )

    safety = _safety_factory(deps)(profile_snapshot, mode=mode)
    guard = _guard_factory(deps)(
        arm_config_path=args.arm_config,
        workspace_config_path=args.workspace_config,
        calibration_path=args.workspace_calibration,
        current_utc=deps.now_utc,
    )
    robot = _robot_factory(deps)(
        profile_snapshot=profile_snapshot,
        runtime_root=args.runtime_root,
        speed_scale=speed_scale,
        monotonic_clock=deps.monotonic(),
    )

    follower_port = _selected_follower_port(robot)
    serial_checker = deps.serial_port_is_free or default_serial_port_is_free
    try:
        serial_is_free = serial_checker(follower_port)
    except Exception as exc:
        raise RuntimeError(f"Follower serial ownership check failed: {exc}") from exc
    if serial_is_free is not True:
        raise RuntimeError("Follower serial port is owned or could not be checked")

    if mode == "live":
        _require_live_phrases(deps.read_input())

    _write_rollout_metadata(
        log_path,
        now=deps.now_utc(),
        mode=mode,
        task=task,
        profile_digest=profile_digest,
        profile_authentication=profile_authentication,
    )
    print(f"jsonl_path={log_path}", file=deps.output())
    if mode == "shadow" and args.dummy_hold:
        print(
            f"standalone_profile_digest={profile_digest} (schema-validated, unauthenticated)",
            file=deps.output(),
        )

    preflight_robot = PreflightRobotAdapter(
        robot=robot,
        expected_task=task,
        profile_snapshot=profile_snapshot,
    )
    keyboard = _keyboard_stop_factory(deps)()
    summary: object
    try:
        with keyboard:
            try:
                runner = _runner_factory(deps)(
                    policy=policy,
                    robot=preflight_robot,
                    safety=safety,
                    mode=mode,
                    log_path=log_path,
                    monotonic_clock=deps.monotonic(),
                    stop_requested=keyboard.event,
                    action_guard=guard,
                )
            except Exception as exc:
                summary = _fault_summary(
                    primary=f"runner construction failed: {_exception_text(exc)}",
                    cleanup=preflight_robot.cleanup_fault_reason,
                )
            else:
                try:
                    summary = runner.run(cycles)
                except Exception as exc:
                    summary = _fault_summary(
                        primary=f"runner execution failed: {_exception_text(exc)}",
                        cleanup=preflight_robot.cleanup_fault_reason,
                    )
    except Exception as exc:
        summary = _fault_summary(
            primary=f"keyboard stop setup failed: {_exception_text(exc)}",
            cleanup=preflight_robot.cleanup_fault_reason,
        )

    summary = _with_cleanup_fault(summary, preflight_robot.cleanup_fault_reason)

    _print_summary(summary, deps.output())
    has_fault = any(
        getattr(summary, field, None)
        for field in (
            "primary_fault_reason",
            "cleanup_fault_reason",
            "audit_fault_reason",
        )
    )
    return 1 if has_fault or summary.terminal_reason == "fault" else 0


def _run_episode_hardware(
    args: argparse.Namespace,
    deps: CliDependencies,
    *,
    mode: str,
    speed_scale: float,
    log_path: Path,
    profile_snapshot: Mapping[str, Any],
    profile_digest: str,
    profile_authentication: str,
    task: str,
    policy: object,
) -> int:
    """Run one episode and, only after manual reset, one fresh retry."""

    attempt = 1
    metadata_written = False
    while True:
        safety = _safety_factory(deps)(profile_snapshot, mode=mode)
        guard = _guard_factory(deps)(
            arm_config_path=args.arm_config,
            workspace_config_path=args.workspace_config,
            calibration_path=args.workspace_calibration,
            current_utc=deps.now_utc,
        )
        robot = _robot_factory(deps)(
            profile_snapshot=profile_snapshot,
            runtime_root=args.runtime_root,
            speed_scale=speed_scale,
            monotonic_clock=deps.monotonic(),
        )

        follower_port = _selected_follower_port(robot)
        serial_checker = deps.serial_port_is_free or default_serial_port_is_free
        try:
            serial_is_free = serial_checker(follower_port)
        except Exception as exc:
            raise RuntimeError(
                f"Follower serial ownership check failed: {exc}"
            ) from exc
        if serial_is_free is not True:
            raise RuntimeError("Follower serial port is owned or could not be checked")

        if mode == "live":
            _require_live_phrases(deps.read_input())

        if not metadata_written:
            _write_rollout_metadata(
                log_path,
                now=deps.now_utc(),
                mode=mode,
                task=task,
                profile_digest=profile_digest,
                profile_authentication=profile_authentication,
            )
            print(f"jsonl_path={log_path}", file=deps.output())
            if mode == "shadow" and args.dummy_hold:
                print(
                    f"standalone_profile_digest={profile_digest} "
                    "(schema-validated, unauthenticated)",
                    file=deps.output(),
                )
            metadata_written = True

        preflight_robot = PreflightRobotAdapter(
            robot=robot,
            expected_task=task,
            profile_snapshot=profile_snapshot,
        )
        keyboard = _keyboard_stop_factory(deps)()
        summary: object
        try:
            with keyboard:
                verdict_source = getattr(keyboard, "verdict", None)
                if not callable(verdict_source):
                    raise RuntimeError(
                        "Episode keyboard control did not expose nonblocking verdicts"
                    )
                try:
                    runner = _runner_factory(deps)(
                        policy=policy,
                        robot=preflight_robot,
                        safety=safety,
                        mode=mode,
                        log_path=log_path,
                        monotonic_clock=deps.monotonic(),
                        stop_requested=keyboard.event,
                        action_guard=guard,
                        operator_verdict=verdict_source,
                    )
                except Exception as exc:
                    summary = _fault_summary(
                        primary=(
                            "runner construction failed: " + _exception_text(exc)
                        ),
                        cleanup=preflight_robot.cleanup_fault_reason,
                        episode=True,
                        attempt=attempt,
                    )
                else:
                    try:
                        summary = runner.run_episode(attempt=attempt)
                    except Exception as exc:
                        summary = _fault_summary(
                            primary=(
                                "runner execution failed: " + _exception_text(exc)
                            ),
                            cleanup=preflight_robot.cleanup_fault_reason,
                            episode=True,
                            attempt=attempt,
                        )
        except Exception as exc:
            summary = _fault_summary(
                primary="keyboard stop setup failed: " + _exception_text(exc),
                cleanup=preflight_robot.cleanup_fault_reason,
                episode=True,
                attempt=attempt,
            )

        summary = _with_cleanup_fault(
            summary,
            preflight_robot.cleanup_fault_reason,
            episode=True,
        )
        _print_summary(summary, deps.output())
        has_fault = any(
            getattr(summary, field, None)
            for field in (
                "primary_fault_reason",
                "cleanup_fault_reason",
                "audit_fault_reason",
            )
        )
        if has_fault or summary.terminal_reason == "safety_fault":
            return 1
        if (
            summary.terminal_reason != "operator_failure"
            or not args.retry_on_failure
            or attempt >= 2
        ):
            return 0

        reset = deps.read_input()(
            "After manual reset with no automatic motion, type exactly "
            f'"{MANUAL_RESET_PHRASE}": '
        )
        if reset != MANUAL_RESET_PHRASE:
            print(
                "retry_skipped=manual_reset_not_acknowledged",
                file=deps.output(),
            )
            return 0
        attempt = 2


def _policy_factory(deps: CliDependencies) -> Callable[[object, str], object]:
    if deps.policy_factory is not None:
        return deps.policy_factory
    from p3_vlm_orchestrator.policy_rollout.lerobot_policy import (
        LeRobotPolicyAdapter,
    )

    return LeRobotPolicyAdapter.from_checkpoint


def _dummy_policy_factory(deps: CliDependencies) -> Callable[[], object]:
    if deps.dummy_policy_factory is not None:
        return deps.dummy_policy_factory
    from p3_vlm_orchestrator.policy_rollout.dummy_policy import HoldPositionPolicy

    return HoldPositionPolicy


def _safety_factory(deps: CliDependencies) -> Callable[..., object]:
    if deps.safety_factory is not None:
        return deps.safety_factory
    from rebot_operator_kit.rollout.safety import SafetyGovernor

    return SafetyGovernor.from_profile


def _guard_factory(deps: CliDependencies) -> Callable[..., object]:
    if deps.guard_factory is not None:
        return deps.guard_factory
    from p3_vlm_orchestrator.policy_rollout.workspace_guard import (
        CalibratedWorkspaceGuard,
    )

    return CalibratedWorkspaceGuard.from_files


def _robot_factory(deps: CliDependencies) -> Callable[..., object]:
    if deps.robot_factory is not None:
        return deps.robot_factory
    from p3_vlm_orchestrator.policy_rollout.rebot_robot import ReBotPolicyRobot

    return ReBotPolicyRobot.from_profile


def _runner_factory(deps: CliDependencies) -> Callable[..., object]:
    if deps.runner_factory is not None:
        return deps.runner_factory
    from p3_vlm_orchestrator.policy_rollout.runner import RolloutRunner

    return RolloutRunner


def _keyboard_stop_factory(deps: CliDependencies) -> Callable[[], object]:
    if deps.keyboard_stop_factory is not None:
        return deps.keyboard_stop_factory
    from p3_vlm_orchestrator.policy_rollout.keyboard_stop import KeyboardStop

    return KeyboardStop


def _checked_speed_scale(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("speed_scale must be within [0.10, 0.20]")
    try:
        speed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("speed_scale must be within [0.10, 0.20]") from exc
    if not math.isfinite(speed) or not MIN_SPEED_SCALE <= speed <= MAX_SPEED_SCALE:
        raise ValueError("speed_scale must be within [0.10, 0.20]")
    return speed


def _selected_follower_port(robot: object) -> str:
    for attribute in ("follower_port", "selected_follower_port"):
        value = getattr(robot, attribute, None)
        if isinstance(value, str) and value:
            return value
    follower = getattr(robot, "follower", None)
    config = getattr(follower, "config", None)
    value = getattr(config, "port", None)
    if isinstance(value, str) and value:
        return value
    raise RuntimeError("Authenticated follower adapter did not expose its selected port")


def _require_live_phrases(input_fn: Callable[[str], str]) -> None:
    operator = input_fn(f'Type exactly "{LIVE_OPERATOR_PHRASE}": ')
    if operator != LIVE_OPERATOR_PHRASE:
        raise RuntimeError("Physical e-stop operator confirmation was not received")
    empty = input_fn(f'Type exactly "{EMPTY_WORKSPACE_PHRASE}": ')
    if empty != EMPTY_WORKSPACE_PHRASE:
        raise RuntimeError("Empty-workspace confirmation was not received")


def _validate_preflight_observation(
    observation: object,
    *,
    expected_task: str,
    profile_snapshot: Mapping[str, Any],
) -> None:
    import numpy as np

    if getattr(observation, "task", None) != expected_task:
        raise ValueError("Preflight observation task does not match the locked task")
    try:
        state = np.asarray(getattr(observation, "state_deg"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Preflight state must contain seven finite values") from exc
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("Preflight state must contain seven finite values")

    cameras = _required_mapping(profile_snapshot, "camera_defaults", "profile")
    front_contract = _required_mapping(cameras, "front", "camera contract")
    side_contract = _required_mapping(cameras, "side", "camera contract")
    _validate_preflight_image(
        getattr(observation, "front", None),
        label="front",
        expected_height=front_contract.get("height"),
        expected_width=front_contract.get("width"),
    )
    _validate_preflight_image(
        getattr(observation, "side", None),
        label="side",
        expected_height=side_contract.get("height"),
        expected_width=side_contract.get("width"),
    )


def _validate_preflight_image(
    image: object,
    *,
    label: str,
    expected_height: object,
    expected_width: object,
) -> None:
    import numpy as np

    if (
        isinstance(expected_height, bool)
        or not isinstance(expected_height, int)
        or isinstance(expected_width, bool)
        or not isinstance(expected_width, int)
    ):
        raise ValueError(f"Profile {label} camera dimensions are invalid")
    array = np.asarray(image)
    expected_shape = (expected_height, expected_width, 3)
    if array.shape != expected_shape:
        raise ValueError(
            f"Preflight {label} image must be HWC RGB {expected_shape}; "
            f"received {array.shape}"
        )


def _load_standalone_profile(path: Path) -> dict[str, Any]:
    profile = _load_json_object(path, "standalone training profile")
    _validate_standalone_profile(profile)
    return profile


def _validate_standalone_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("schema_version") != 1:
        raise ValueError("Standalone training profile schema_version must be 1")
    profile_id = profile.get("profile_id")
    if not isinstance(profile_id, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_-]{2,95}", profile_id
    ):
        raise ValueError("Standalone training profile_id is invalid")
    version = profile.get("profile_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Standalone training profile_version must be positive")

    collection = _required_mapping(profile, "collection_defaults", "profile")
    task = collection.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Standalone training profile task must be nonempty")
    if collection.get("motor_velocity") != 2000.0:
        raise ValueError("Standalone training profile motor_velocity must equal 2000.0")
    fps = _positive_integer(collection.get("fps"), "collection FPS")
    gripper_force = _finite_number(
        collection.get("gripper_force"), "collection gripper force"
    )
    if not 0.0 <= gripper_force <= 1.0:
        raise ValueError("Standalone collection gripper force must be in [0, 1]")

    coordinates = _required_mapping(profile, "coordinate_contract", "profile")
    if coordinates.get("frame") != EXPECTED_COORDINATE_FRAME:
        raise ValueError("Standalone training profile coordinate frame is invalid")
    if coordinates.get("control_mode") != EXPECTED_CONTROL_MODE:
        raise ValueError("Standalone training profile control mode is invalid")
    if coordinates.get("action_dimension") != 7:
        raise ValueError("Standalone training profile action dimension must be 7")
    joints = coordinates.get("joints")
    if not isinstance(joints, list) or len(joints) != 7:
        raise ValueError("Standalone training profile must define exactly seven joints")
    for index, (joint, expected_name) in enumerate(
        zip(joints, EXPECTED_JOINT_NAMES, strict=True)
    ):
        if not isinstance(joint, Mapping):
            raise ValueError(f"Standalone training profile joint {index} is invalid")
        if joint.get("name") != expected_name or joint.get("feature") != f"{expected_name}.pos":
            raise ValueError("Standalone training profile joint order/features are invalid")
        _finite_nonzero(joint.get("leader_to_follower_scale"), f"joint {expected_name} scale")
        limits = joint.get("soft_limit_degrees")
        if not isinstance(limits, list) or len(limits) != 2:
            raise ValueError(f"Standalone training profile joint {expected_name} limits are invalid")
        low = _finite_number(limits[0], f"joint {expected_name} lower limit")
        high = _finite_number(limits[1], f"joint {expected_name} upper limit")
        if low >= high:
            raise ValueError(f"Standalone training profile joint {expected_name} limits are invalid")

    training = _required_mapping(profile, "training_defaults", "profile")
    if training.get("action_dimension") != 7:
        raise ValueError("Standalone training policy action dimension must be 7")
    _positive_integer(training.get("chunk_size"), "training chunk size")
    _positive_integer(training.get("n_action_steps"), "training action steps")
    if training.get("image_order") != list(EXPECTED_IMAGE_ORDER):
        raise ValueError("Standalone training profile image order must be front then side")
    if (
        training.get("state_normalization") != "quantile"
        or training.get("action_normalization") != "quantile"
        or training.get("normalize_gripper") is not True
    ):
        raise ValueError(
            "Standalone training profile must lock quantile normalization and gripper normalization"
        )

    cameras = _required_mapping(profile, "camera_defaults", "profile")
    extra = set(cameras) - {"front", "side"} - set(CAMERA_METADATA_KEYS)
    if extra:
        raise ValueError(f"Standalone camera profile has extra entries: {sorted(extra)}")
    expected_cameras = {
        "front": ("observation.images.front", 640, 480),
        "side": ("observation.images.side", 1280, 720),
    }
    indices: list[int] = []
    for name, (recording_key, width, height) in expected_cameras.items():
        camera = _required_mapping(cameras, name, "camera contract")
        if camera.get("recording_key") != recording_key:
            raise ValueError(f"Standalone {name} camera recording key is invalid")
        if camera.get("width") != width or camera.get("height") != height:
            raise ValueError(
                f"Standalone {name} camera must be exactly {width}x{height}"
            )
        if camera.get("fps") != fps:
            raise ValueError("Standalone camera FPS must match collection FPS")
        indices.append(_nonnegative_integer(camera.get("index"), f"{name} camera index"))
    if indices[0] == indices[1]:
        raise ValueError("Standalone front and side camera indices must be distinct")
    excluded_index = cameras.get("excluded_screen_index")
    if excluded_index is not None:
        excluded_index = _nonnegative_integer(
            excluded_index, "excluded screen camera index"
        )
        if excluded_index in indices:
            raise ValueError("Standalone excluded screen camera must not be recorded")
    minimum_fps = cameras.get("minimum_measured_fps")
    if minimum_fps is not None:
        minimum_fps = _positive_integer(minimum_fps, "minimum measured camera FPS")
        if minimum_fps > fps:
            raise ValueError("Standalone minimum measured camera FPS exceeds capture FPS")

    calibration = _required_mapping(profile, "calibration", "profile")
    for name in PROFILE_RUNTIME_ENTRIES:
        entry = _required_mapping(calibration, name, "profile calibration")
        relative = entry.get("runtime_relative_path")
        digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(f"Standalone profile calibration {name} is invalid")
    follower = _required_mapping(calibration, "follower", "profile calibration")
    if follower.get("type") != "seeed_b601_dm_follower" or follower.get("id") != "follower1":
        raise ValueError("Standalone follower type/ID contract is invalid")
    leader = _required_mapping(calibration, "leader", "profile calibration")
    if leader.get("type") != "rebot_arm_102_leader" or leader.get("id") != "rebot_arm_102_leader":
        raise ValueError("Standalone leader type/ID contract is invalid")

    hardware = _required_mapping(profile, "hardware_identity", "profile")
    for identity_name in ("follower_usb", "leader_usb"):
        usb_identity = _required_mapping(
            hardware, identity_name, "hardware identity"
        )
        for key in ("vid", "pid"):
            value = usb_identity.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 < value <= 0xFFFF
            ):
                raise ValueError(f"Standalone {identity_name} {key} is invalid")


def _required_mapping(
    parent: Mapping[str, Any], key: str, label: str
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label.capitalize()} {key} must be an object")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_nonzero(value: object, label: str) -> float:
    result = _finite_number(value, label)
    if result == 0.0:
        raise ValueError(f"{label} must be nonzero")
    return result


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label.capitalize()} must be an object")
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Nonfinite JSON constant is not permitted: {value}")


def _processor_artifact_report(checkpoint: Path) -> tuple[str, ...]:
    root = checkpoint.resolve()
    report: list[str] = []
    for filename in ("preprocessor_config.json", "postprocessor_config.json"):
        document = _load_json_object(checkpoint / filename, "saved processor config")
        report.append(filename)
        steps = document.get("steps")
        if not isinstance(steps, list):
            raise ValueError(f"Saved processor config {filename} steps must be a list")
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise ValueError(f"Saved processor config {filename} step {index} is invalid")
            state_file = step.get("state_file")
            if state_file is None:
                continue
            if not isinstance(state_file, str) or not state_file:
                raise ValueError(f"Saved processor config {filename} state_file is invalid")
            state_path = (checkpoint / state_file).resolve()
            try:
                state_path.relative_to(root)
            except ValueError as exc:
                raise ValueError("Saved processor state file leaves the checkpoint") from exc
            if not state_path.is_file():
                raise ValueError(f"Saved processor state file is missing: {state_file}")
            report.append(state_file)
    return tuple(report)


def _resolve_log_path(
    requested: Path | None,
    now: datetime,
    *,
    repo_root: Path,
) -> Path:
    """Resolve an audit path while confining it to the rollout log root."""

    repository = Path(os.path.abspath(Path(repo_root).expanduser()))
    allowed = repository / "runs" / "policy"
    if requested is None:
        timestamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_target = allowed / f"{timestamp}.jsonl"
    else:
        supplied = Path(requested).expanduser()
        if ".." in supplied.parts:
            raise ValueError(
                "Explicit log path must not traverse outside repo runs/policy/"
            )
        raw_target = (
            supplied
            if supplied.is_absolute()
            else Path(os.path.abspath(supplied))
        )

    raw_target = Path(os.path.abspath(raw_target))
    try:
        relative = raw_target.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(
            "Explicit log path must be under repo runs/policy/; protected data, "
            "models/checkpoints, config/calibration, env, and credential paths are forbidden"
        ) from exc
    if not relative.parts or raw_target.suffix != ".jsonl":
        raise ValueError("Rollout log path under runs/policy/ must name a .jsonl file")
    protected_names = (".env", "credential", "secret", "calibration", "checkpoint")
    if any(
        any(marker in part.lower() for marker in protected_names)
        for part in relative.parts
    ):
        raise ValueError(
            "Rollout log path under runs/policy/ cannot name protected env, "
            "credential, calibration, or checkpoint files"
        )

    for directory in (repository / "runs", allowed):
        if directory.is_symlink():
            raise ValueError("Rollout log path under runs/policy/ must not use symlinks")
    candidate = allowed
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("Rollout log path under runs/policy/ must not use symlinks")
    resolved_allowed = allowed.resolve(strict=False)
    resolved_target = raw_target.resolve(strict=False)
    try:
        resolved_target.relative_to(resolved_allowed)
    except ValueError as exc:
        raise ValueError(
            "Resolved rollout log path must remain under repo runs/policy/"
        ) from exc
    if resolved_target.exists() and not resolved_target.is_file():
        raise ValueError("Rollout log path under runs/policy/ must be a regular file")
    return resolved_target


def _write_rollout_metadata(
    path: Path,
    *,
    now: datetime,
    mode: str,
    task: str,
    profile_digest: str,
    profile_authentication: str,
) -> None:
    row = {
        "event": "rollout_metadata",
        "timestamp_utc": now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": mode,
        "task": task,
        "profile_digest": profile_digest,
        "profile_authentication": profile_authentication,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
    except Exception as exc:
        raise RuntimeError(f"Rollout metadata log could not be written: {exc}") from exc


def _fault_summary(
    *,
    primary: str,
    cleanup: str | None = None,
    audit: str | None = None,
    episode: bool = False,
    attempt: int = 1,
) -> _TerminalSummary:
    return _TerminalSummary(
        cycles_completed=0,
        actions_attempted=0,
        actions_confirmed=0,
        terminal_reason="safety_fault" if episode else "fault",
        primary_fault_reason=primary,
        cleanup_fault_reason=cleanup,
        audit_fault_reason=audit,
        attempt=attempt,
    )


def _with_cleanup_fault(
    summary: object,
    cleanup: str | None,
    *,
    episode: bool = False,
) -> object:
    if cleanup is None or getattr(summary, "cleanup_fault_reason", None) is not None:
        return summary
    return _TerminalSummary(
        cycles_completed=getattr(summary, "cycles_completed"),
        actions_attempted=getattr(summary, "actions_attempted"),
        actions_confirmed=getattr(summary, "actions_confirmed"),
        terminal_reason="safety_fault" if episode else "fault",
        primary_fault_reason=getattr(summary, "primary_fault_reason", None),
        cleanup_fault_reason=cleanup,
        audit_fault_reason=getattr(summary, "audit_fault_reason", None),
        attempt=getattr(summary, "attempt", 1),
        elapsed_seconds=getattr(summary, "elapsed_seconds", 0.0),
        clamp_count=getattr(summary, "clamp_count", 0),
    )


def _print_summary(summary: object, output: IO[str]) -> None:
    print(
        "rollout_summary "
        f"cycles={getattr(summary, 'cycles_completed')} "
        f"actions_attempted={getattr(summary, 'actions_attempted')} "
        f"actions_confirmed={getattr(summary, 'actions_confirmed')} "
        f"attempt={getattr(summary, 'attempt', 1)} "
        f"elapsed_seconds={float(getattr(summary, 'elapsed_seconds', 0.0)):.3f} "
        f"clamp_count={getattr(summary, 'clamp_count', 0)} "
        f"terminal_reason={getattr(summary, 'terminal_reason')} "
        f"primary_fault={_fault_text(getattr(summary, 'primary_fault_reason', None))} "
        f"cleanup_fault={_fault_text(getattr(summary, 'cleanup_fault_reason', None))} "
        f"audit_fault={_fault_text(getattr(summary, 'audit_fault_reason', None))}",
        file=output,
    )


def _fault_text(value: object) -> str:
    return "none" if value is None else str(value)


def _exception_text(exc: Exception) -> str:
    try:
        return str(exc)
    except Exception:
        return type(exc).__name__


if __name__ == "__main__":
    raise SystemExit(main())
