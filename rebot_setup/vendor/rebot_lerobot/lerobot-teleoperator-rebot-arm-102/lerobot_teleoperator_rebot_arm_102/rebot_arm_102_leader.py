import logging
import time
from typing import Any, Tuple

from motorbridge_smart_servo import FashionStarServo, ServoMonitor, ServoBusError
from lerobot.motors import MotorCalibration
from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_rebot_arm_102_leader import RebotArm102LeaderConfig

logger = logging.getLogger(__name__)

MEDIUM_TIMEOUT_SEC = 0.01


class RebotArm102Leader(Teleoperator):
    """
    LeRobot teleoperator integration for the reBot Arm 102 leader arm.

    This implementation keeps the lifecycle and calibration flow close to other
    LeRobot leader teleoperators. The servo SDK is only used for device access.
    """

    config_class = RebotArm102LeaderConfig
    name = "rebot_arm_102_leader"

    def __init__(self, config: RebotArm102LeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus: FashionStarServo | None = None
        self.motor_names = list(self.config.joint_ids.keys())
        self._last_raw_positions: dict[str, float] = {}
        self._validate_config()

    def _validate_config(self) -> None:
        required_keys = set(self.config.joint_ids)
        for field_name in ("joint_ranges",):
            keys = set(getattr(self.config, field_name))
            if keys != required_keys:
                raise ValueError(
                    f"{field_name} keys must match joint_ids keys. "
                    f"Expected {sorted(required_keys)}, got {sorted(keys)}."
                )
        for motor_name, joint_range in self.config.joint_ranges.items():
            if len(joint_range) != 2:
                raise ValueError(f"joint_ranges[{motor_name!r}] must contain exactly [min, max].")
            if joint_range[0] > joint_range[1]:
                raise ValueError(f"joint_ranges[{motor_name!r}] must satisfy min <= max.")

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Connecting arm on {self.config.port}...")
        bus = FashionStarServo(self.config.port, baudrate=self.config.baudrate)

        try:
            for motor_name, motor_id in self.config.joint_ids.items():
                if not bus.ping(motor_id):
                    raise RuntimeError(f"Servo not found for {motor_name} (id={motor_id}).")
                self._last_raw_positions[motor_name] = 0.0

            self.bus = bus

            if not self.is_calibrated and calibrate:
                logger.info(
                    "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
                )
                self.calibrate()

            self.configure()
        except Exception:
            bus.close()
            self.bus = None
            raise

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration) and set(self.calibration) == set(self.motor_names)

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                return
        
        logger.info(f"\nRunning calibration for {self}")
        input(
            "\nCalibration: Set Zero Position\n"
            "Please manually move the reBot Arm 102 to its zero pose and close the gripper.\n"
            "Press ENTER when ready..."
        )

        logger.info("Setting range: -90° to +90° by default for all joints")
        self.calibration = {}
        for motor_name, motor_id in self.config.joint_ids.items():
            self.bus.unlock(motor_id)
            time.sleep(MEDIUM_TIMEOUT_SEC)
            self.bus.set_origin_point(motor_id)
            self.calibration[motor_name] = MotorCalibration(
                id=self.config.joint_ids[motor_name],
                drive_mode=0,
                homing_offset=0,
                range_min=-90,
                range_max=90,
            )

        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        for motor_id in self.config.joint_ids.values():
            self.bus.unlock(motor_id)
            time.sleep(MEDIUM_TIMEOUT_SEC)

        # Reset multi-turn counter per servo (old SDK used broadcast 0xFF which is out of range).
        for motor_id in self.config.joint_ids.values():
            self.bus.reset_multi_turn(motor_id)

    def _read_raw_positions(self) -> dict[str, float]:
        result: dict[int, ServoMonitor | None] = self.bus.sync_monitor(
            list(self.config.joint_ids.values())
        )
        id_to_name = {v: k for k, v in self.config.joint_ids.items()}
        raw_positions: dict[str, float] = {}
        for motor_id, m in result.items():
            motor_name = id_to_name[motor_id]
            if m is None:
                raise RuntimeError(f"Servo {motor_name} (id={motor_id}) has never responded.")
            raw_positions[motor_name] = m.angle_deg
        return raw_positions

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))
    
    @staticmethod
    def _round_to_valid_range(value: float, min_value: float, max_value: float) -> Tuple[float, int]:
        """Unwrap a multi-turn angle back into the ±180° window centred on (min+max)/2.

        The servo may report an angle that has accumulated extra full rotations
        (value = true_angle + N*360).  We do not know the sign of N, so we use a
        bidirectional search: starting from k=0, simultaneously try value+k*360 and
        value-k*360 until the first candidate that falls inside [center-180, center+180].
        """
        center = (min_value + max_value) / 2.0
        low = center - 180.0
        high = center + 180.0
        for k in range(4096):
            candidate_plus = value + k * 360.0
            if low <= candidate_plus <= high:
                return candidate_plus, k
            candidate_minus = value - k * 360.0
            if low <= candidate_minus <= high:
                return candidate_minus, k
        # Fallback: direct modular arithmetic (should never be reached)
        return value - round((value - center) / 360.0) * 360.0, 4096

    def get_action(self) -> RobotAction:
        start = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        raw_positions: dict[str, Any] = {}
        try:
            raw_positions = self._read_raw_positions()
            logger.debug(f"Raw positions: {', '.join(f'{m}: {p:.1f}°' for m, p in raw_positions.items())}")
            self._last_raw_positions = raw_positions
        except Exception as e:
            logger.error(f"Failed to read raw positions: {e}")
            logger.warning("[EMERGENCY STOP] Please hold the follower arm and cut off the main power to the arms.")
            logger.warning("[EMERGENCY STOP] Break the teleoperation session and check the USB connection or power of the leader arm.")
            raw_positions = self._last_raw_positions
            
        action_dict: dict[str, Any] = {}
        for motor_name in self.motor_names:
            range_min, range_max = self.config.joint_ranges[motor_name]
            unwrapped, k = self._round_to_valid_range(raw_positions[motor_name],float(range_min),float(range_max))
            position = unwrapped
            if k > 0:
                logger.debug(
                    f"Servo {motor_name} (id={self.config.joint_ids[motor_name]}) has wrapped {k} * 360°. "
                    f"Unwrapped pos: {unwrapped:.1f}° (raw: {raw_positions[motor_name]:.1f}°)"
                )
            action_dict[f"{motor_name}.pos"] = self._clamp(position,float(range_min),float(range_max),)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("Feedback is not implemented for the reBot Arm 102 leader.")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.close()
        self.bus = None
        logger.info(f"{self} disconnected.")
