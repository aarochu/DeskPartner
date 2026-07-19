#!/usr/bin/env python
"""Test script for reBot Arm 102 leader arm using motorbridge-smart-servo SDK.

Covers all servo operations used by the leader arm integration:
  - ping:             verify each servo is online
  - unlock:           free-move mode (equivalent to Stop_On_Control_Mode "unlocked")
  - reset_multi_turn: reset per-servo multi-turn angle counter (equivalent to ResetLoop)
  - read_angle:       read current positions with reliability filter
  - set_origin_point: set current position as zero (used during calibrate)

Emergency stop behaviour: ServoBusError during read (e.g. main power cut while
USB is still live) is caught, EMERGENCY STOP warnings are logged, then re-raised
so the script exits with a non-zero status.

Usage:
    python examples/test_leader.py --port /dev/ttyUSB0 --action connect
    python examples/test_leader.py --port /dev/ttyUSB0 --action read --duration 10
    python examples/test_leader.py --port /dev/ttyUSB0 --action calibrate
    python examples/test_leader.py --port /dev/ttyUSB0 --action benchmark --duration 5
"""

import argparse
import logging
import time

from motorbridge_smart_servo import FashionStarServo, ServoMonitor, ServoBusError

from lerobot_teleoperator_rebot_arm_102 import RebotArm102Leader, RebotArm102LeaderConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

JOINT_IDS: dict[str, int] = {
    "shoulder_pan":  0,
    "shoulder_lift": 1,
    "elbow_flex":    2,
    "wrist_flex":    3,
    "wrist_yaw":     4,
    "wrist_roll":    5,
    "gripper":       6,
}

MEDIUM_TIMEOUT_SEC = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test reBot Arm 102 leader arm with motorbridge-smart-servo SDK."
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument(
        "--action",
        choices=["connect", "read", "calibrate", "benchmark"],
        default="connect",
        help=(
            "connect:   ping + unlock + reset_multi_turn + one angle read; "
            "read:      continuous angle read loop; "
            "calibrate: unlock + set_origin_point; "
            "benchmark: measure raw read and get_action loop speed"
        ),
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Action duration in seconds (used by --action read and --action benchmark)",
    )
    parser.add_argument(
        "--interval", type=float, default=0.2,
        help="Polling interval in seconds (only for --action read)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _emergency_stop_and_raise(exc: ServoBusError) -> None:
    """Log EMERGENCY STOP messages then re-raise to terminate the script."""
    logger.error(f"Servo read failed: {exc}")
    logger.warning("[EMERGENCY STOP] Please hold the follower arm and cut off the main power to the arms.")
    logger.warning("[EMERGENCY STOP] Break the teleoperation session and check the USB connection of the leader arm.")
    raise exc


def ping_all(bus: FashionStarServo) -> None:
    logger.info("Pinging all servos...")
    for motor_name, motor_id in JOINT_IDS.items():
        online = bus.ping(motor_id)
        status = "online" if online else "OFFLINE"
        logger.info(f"  {status:7s}  {motor_name} (id={motor_id})")
        if not online:
            raise RuntimeError(f"Servo not found: {motor_name} (id={motor_id})")
    logger.info("All servos online.")


def unlock_all(bus: FashionStarServo) -> None:
    """Disable torque on all servos (free-move / manual control mode)."""
    logger.info("Unlocking all servos (free-move)...")
    for motor_name, motor_id in JOINT_IDS.items():
        bus.unlock(motor_id)
        time.sleep(MEDIUM_TIMEOUT_SEC)
        logger.info(f"  Unlocked {motor_name} (id={motor_id})")


def reset_multi_turn_all(bus: FashionStarServo) -> None:
    """Reset multi-turn angle counter for each servo.

    The old SDK used broadcast id 0xFF (=255). The new library enforces
    id <= 253, so we reset each servo individually instead.
    """
    logger.info("Resetting multi-turn angle counters...")
    for motor_name, motor_id in JOINT_IDS.items():
        bus.reset_multi_turn(motor_id)
        logger.info(f"  Reset {motor_name} (id={motor_id})")


def read_angles_once(bus: FashionStarServo) -> dict[str, float]:
    """Read angles for all servos in a single sync_monitor command (~24ms for 7 servos).

    Uses angle_deg directly (already reliability-filtered by the library).
    Raises ServoBusError if any servo exceeds the consecutive loss threshold
    (e.g. main power cut while USB is still live).
    """
    result: dict[int, ServoMonitor | None] = bus.sync_monitor(list(JOINT_IDS.values()))
    id_to_name = {v: k for k, v in JOINT_IDS.items()}
    positions: dict[str, float] = {}
    for motor_id, m in result.items():
        motor_name = id_to_name[motor_id]
        if m is None:
            raise RuntimeError(f"Servo {motor_name} (id={motor_id}) has never responded.")
        positions[motor_name] = m.angle_deg
    return positions


def print_positions(positions: dict[str, float]) -> None:
    parts = [f"{name}={pos:+8.2f}°" for name, pos in positions.items()]
    print("  ".join(parts), flush=True)


def benchmark_callable(label: str, fn, duration: float) -> None:
    print(f"[benchmark] starting {label} for {duration:.2f}s", flush=True)
    count = 0
    total_s = 0.0
    min_s = float("inf")
    max_s = 0.0
    deadline = time.monotonic() + duration

    try:
        while time.monotonic() < deadline:
            t0 = time.perf_counter()
            fn()
            dt_s = time.perf_counter() - t0
            count += 1
            total_s += dt_s
            min_s = min(min_s, dt_s)
            max_s = max(max_s, dt_s)
    except ServoBusError as exc:
        _emergency_stop_and_raise(exc)

    if count == 0:
        print(f"[benchmark] {label}: no samples collected", flush=True)
        return

    avg_s = total_s / count
    print(
        f"[benchmark] {label}: {count} samples in {duration:.2f}s | "
        f"avg={avg_s * 1e3:.2f} ms | min={min_s * 1e3:.2f} ms | "
        f"max={max_s * 1e3:.2f} ms | hz={1.0 / avg_s:.1f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def action_connect(bus: FashionStarServo) -> None:
    """Ping all servos, unlock, reset multi-turn, then do one angle read."""
    ping_all(bus)
    unlock_all(bus)
    reset_multi_turn_all(bus)

    logger.info("Reading initial angles...")
    try:
        print_positions(read_angles_once(bus))
    except ServoBusError as exc:
        _emergency_stop_and_raise(exc)

    logger.info("Connection test passed.")


def action_read(bus: FashionStarServo, duration: float, interval: float) -> None:
    """Ping + unlock + reset, then read angles in a loop for `duration` seconds.

    ServoBusError (e.g. from main power loss) triggers EMERGENCY STOP and exits.
    """
    ping_all(bus)
    unlock_all(bus)
    reset_multi_turn_all(bus)

    logger.info(f"Reading angles for {duration:.1f}s (interval={interval:.2f}s). Ctrl+C to stop.")
    deadline = time.monotonic() + duration
    try:
        while time.monotonic() < deadline:
            t0 = time.perf_counter()
            try:
                positions = read_angles_once(bus)
            except ServoBusError as exc:
                _emergency_stop_and_raise(exc)
            dt_ms = (time.perf_counter() - t0) * 1e3
            print_positions(positions)
            logger.debug(f"sync_monitor took {dt_ms:.1f} ms")
            sleep = interval - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")


def action_calibrate(bus: FashionStarServo) -> None:
    """Unlock all servos, prompt user to set zero pose, then set origin point per servo."""
    ping_all(bus)

    input(
        "\nCalibration: Set Zero Position\n"
        "Please manually move the reBot Arm 102 to its zero pose and close the gripper.\n"
        "Press ENTER when ready..."
    )

    logger.info("Unlocking and setting origin point for each servo...")
    for motor_name, motor_id in JOINT_IDS.items():
        bus.unlock(motor_id)
        time.sleep(MEDIUM_TIMEOUT_SEC)
        bus.set_origin_point(motor_id)
        time.sleep(MEDIUM_TIMEOUT_SEC)
        logger.info(f"  Origin set for {motor_name} (id={motor_id})")

    logger.info("Resetting multi-turn angle counters after origin set...")
    reset_multi_turn_all(bus)

    logger.info("Reading angles after origin set (all should be near 0°)...")
    try:
        print_positions(read_angles_once(bus))
    except ServoBusError as exc:
        _emergency_stop_and_raise(exc)

    logger.info(
        "Calibration test done. "
        "(Calibration data is NOT persisted here — use the LeRobot integration for that.)"
    )


def action_benchmark(port: str, baudrate: int, duration: float) -> None:
    print(f"[benchmark] creating leader on {port} @ {baudrate}", flush=True)
    leader = RebotArm102Leader(
        RebotArm102LeaderConfig(
            id="benchmark_leader",
            port=port,
            baudrate=baudrate,
        )
    )
    print("[benchmark] connecting leader", flush=True)
    leader.connect(calibrate=False)
    print("[benchmark] leader connected", flush=True)
    try:
        benchmark_callable("leader._read_raw_positions", leader._read_raw_positions, duration)
        benchmark_callable("leader.get_action", leader.get_action, duration)
    finally:
        print("[benchmark] disconnecting leader", flush=True)
        leader.disconnect()
        print("[benchmark] done", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    print(f"[test_leader] action={args.action} port={args.port} baudrate={args.baudrate}", flush=True)
    if args.action == "benchmark":
        action_benchmark(args.port, args.baudrate, args.duration)
        return

    logger.info(f"Opening bus on {args.port} at {args.baudrate} baud...")
    with FashionStarServo(args.port, baudrate=args.baudrate) as bus:
        logger.info("Bus opened.")
        if args.action == "connect":
            action_connect(bus)
        elif args.action == "read":
            action_read(bus, args.duration, args.interval)
        elif args.action == "calibrate":
            action_calibrate(bus)


if __name__ == "__main__":
    main()
