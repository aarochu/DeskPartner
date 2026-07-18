#!/usr/bin/env python

import argparse
import os
import time
from pathlib import Path

from lerobot_teleoperator_rebot_arm_102 import (
    RebotArm102Leader,
    RebotArm102LeaderConfig,
)
from lerobot_robot_seeed_b601 import (
    SeeedB601DMFollower,
    SeeedB601DMFollowerConfig,
    SeeedB601RSFollower,
    SeeedB601RSFollowerConfig,
)


class PassiveSeeedB601DMFollower(SeeedB601DMFollower):
    """Read-only DM follower variant for manual comparison/debugging."""

    def connect(self, calibrate: bool = False) -> None:
        super().connect(calibrate=calibrate)
        self.disable_torque()


class PassiveSeeedB601RSFollower(SeeedB601RSFollower):
    """Read-only RS follower variant for manual comparison/debugging."""

    def connect(self, calibrate: bool = False) -> None:
        super().connect(calibrate=calibrate)
        self.disable_torque()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read reBot Arm 102 leader and B601 follower positions side by side."
    )
    parser.add_argument(
        "--leader-port", required=True, help="reBot Arm 102 serial port, e.g. /dev/ttyUSB0"
    )
    parser.add_argument("--leader-id", default="rebot_arm_102_leader")
    parser.add_argument("--leader-baudrate", type=int, default=1000000)
    parser.add_argument(
        "--follower-port",
        required=True,
        help="B601 CAN port, e.g. can0 or /dev/ttyACM0",
    )
    parser.add_argument("--follower-id", default="b601_follower")
    parser.add_argument("--follower-type", choices=["dm", "rs"], default="dm")
    parser.add_argument("--follower-can-adapter", default="socketcan", help="damiao, socketcan")
    parser.add_argument("--follower-dm-serial-baud", type=int, default=921600)
    parser.add_argument(
        "--interval", type=float, default=0.2, help="Polling interval in seconds"
    )
    return parser.parse_args()


def make_follower(args: argparse.Namespace):
    if args.follower_type == "dm":
        config = SeeedB601DMFollowerConfig(
            id=args.follower_id,
            port=args.follower_port,
            can_adapter=args.follower_can_adapter,
            dm_serial_baud=args.follower_dm_serial_baud,
            cameras={},
        )
        return PassiveSeeedB601DMFollower(config)

    config = SeeedB601RSFollowerConfig(
        id=args.follower_id,
        port=args.follower_port,
        can_adapter=args.follower_can_adapter,
        dm_serial_baud=args.follower_dm_serial_baud,
        cameras={},
    )
    return PassiveSeeedB601RSFollower(config)


def main() -> None:
    args = parse_args()

    leader = RebotArm102Leader(
        RebotArm102LeaderConfig(
            id=args.leader_id,
            port=args.leader_port,
            baudrate=args.leader_baudrate,
        )
    )
    follower = make_follower(args)

    leader.connect(calibrate=False)
    follower.connect(calibrate=False)

    try:
        if not leader.is_calibrated:
            raise RuntimeError(
                "No reBot Arm 102 calibration file found. Run examples/calibrate.py first."
            )

        ranges = leader.config.joint_ranges
        follower_directions = follower.config.joint_directions

        print("Reading leader/follower positions side by side. Press Ctrl+C to stop.")
        while True:
            raw_positions = leader._read_raw_positions()
            leader_action = leader.get_action()
            follower_obs = follower.get_observation()

            os.system("clear")
            print(
                "Reading leader/follower positions side by side. Press Ctrl+C to stop.\n"
            )
            print(
                f"{'joint':<16} {'raw':>8} {'range':>13} {'leader':>8} "
                f"{'f.dir':>6} {'mapped':>8} {'follower':>9} {'delta':>8}"
            )
            print(
                f"{'-' * 16} {'-' * 8} {'-' * 13} {'-' * 8} "
                f"{'-' * 6} {'-' * 8} {'-' * 9} {'-' * 8}"
            )

            for joint in leader.motor_names:
                raw = raw_positions[joint]
                r_min, r_max = ranges[joint]
                leader_pos = leader_action[f"{joint}.pos"]
                follower_direction = follower_directions.get(joint, 1.0)
                mapped = leader_pos * follower_direction
                follower_pos = follower_obs[f"{joint}.pos"]
                delta = follower_pos - mapped

                range_str = f"[{r_min},{r_max}]"
                print(
                    f"{joint:<16} {raw:8.2f} {range_str:>13} {leader_pos:8.2f} "
                    f"{follower_direction:6.1f} {mapped:8.2f} {follower_pos:9.2f} {delta:8.2f}"
                )

            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        follower.disconnect()
        leader.disconnect()


if __name__ == "__main__":
    main()
