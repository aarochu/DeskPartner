#!/usr/bin/env python

import argparse
import time

from fashionstar_uart_sdk.uart_pocket_handler import PortHandler


DEFAULT_JOINT_IDS = {
    "shoulder_pan": 0,
    "shoulder_lift": 1,
    "elbow_flex": 2,
    "wrist_flex": 3,
    "wrist_yaw": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read raw reBot Arm 102 servo angles without LeRobot calibration."
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1000000)
    parser.add_argument(
        "--interval", type=float, default=0.2, help="Polling interval in seconds"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    porthandler = PortHandler(args.port, args.baudrate)
    porthandler.openPort()

    # Initialize servos: ping, unlock, and reset multi-turn counter.
    # Without this the SDK returns None for all monitor fields.
    for servo_id in DEFAULT_JOINT_IDS.values():
        if not porthandler.ping(servo_id):
            print(f"WARNING: servo id={servo_id} did not respond to ping")
        porthandler.write["Stop_On_Control_Mode"](servo_id, "unlocked", 900)
        time.sleep(0.01)
    porthandler.ResetLoop(0xFF)

    try:
        print("Reading raw servo angles. Press Ctrl+C to stop.")
        while True:
            monitor_data = porthandler.sync_read["Monitor"](DEFAULT_JOINT_IDS)
            values = []
            for joint_name in DEFAULT_JOINT_IDS:
                state = monitor_data.get(joint_name)
                if state is None or state.current_position is None:
                    values.append(f"{joint_name}=<missing>")
                else:
                    values.append(f"{joint_name}={float(state.current_position):8.2f}")
            print("  ".join(values))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        porthandler.closePort()


if __name__ == "__main__":
    main()
