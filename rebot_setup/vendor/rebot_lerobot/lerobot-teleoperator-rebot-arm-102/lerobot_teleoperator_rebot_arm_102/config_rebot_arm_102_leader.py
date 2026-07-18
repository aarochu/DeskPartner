from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rebot_arm_102_leader")
@dataclass
class RebotArm102LeaderConfig(TeleoperatorConfig):
    """Configuration for the reBot Arm 102 leader arm."""

    port: str
    baudrate: int = 1_000_000
    joint_ids: dict[str, int] = field(
        default_factory=lambda: {
            "shoulder_pan": 0,
            "shoulder_lift": 1,
            "elbow_flex": 2,
            "wrist_flex": 3,
            "wrist_yaw": 4,
            "wrist_roll": 5,
            "gripper": 6,
        }
    )

    joint_ranges: dict[str, list[int]] = field(
        default_factory=lambda: {
            "shoulder_pan":  (-150.0, 150.0),
            "shoulder_lift": (-1.0, 170.0),
            "elbow_flex":    (-200.0, 1.0),
            "wrist_flex":    (-80.0, 90.0),
            "wrist_yaw":     (-90.0, 90.0),
            "wrist_roll":    (-90.0, 90.0),
            "gripper":       (-0.0, 270.0),
        }
    )
