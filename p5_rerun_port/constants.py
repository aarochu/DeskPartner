"""Shared constants for the reBot ↔ Rerun ↔ LeRobot v3 port."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

APP_ID = "deskpartner-rebot"

# Keep SO-compatible entity roles so query/export/replay stay close to so100-hackathon.
FOLLOWER = "follower"
LEADER = "leader"

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)

# LeRobot camera keys (config/recording.yaml) → Rerun cam indices (SO-style).
CAMERA_TO_RERUN = {
    "front": "camera/cam0",
    "side": "camera/cam1",
}
RERUN_TO_LEROBOT_CAM = {v: k for k, v in CAMERA_TO_RERUN.items()}

ROBOT_TYPE = "seeed_b601_dm_follower"
TELEOP_TYPE = "rebot_arm_102_leader"

SEGMENT_TAGS = ("Good episode", "Bad episode", "Needs review")

DEFAULT_RECORDINGS_DIR = REPO_ROOT / "recordings"
DEFAULT_DATASETS_DIR = REPO_ROOT / "datasets"
DEFAULT_CATALOG = DEFAULT_RECORDINGS_DIR / "catalog.json"

# Prefer SDK URDF; fall back to env override.
DEFAULT_URDF_CANDIDATES = (
    REPO_ROOT
    / "rebot_setup"
    / "vendor"
    / "reBotArm_control_py"
    / "urdf"
    / "reBot-DevArm_fixend_description"
    / "urdf"
    / "reBot-DevArm_fixend.urdf",
    Path.home() / "reBotArm_control_py" / "urdf" / "00-arm-rs_asm-v3" / "urdf" / "00-arm-rs_asm-v3.urdf",
    Path(r"C:\Users\aaron\reBotArm_control_py\urdf\00-arm-rs_asm-v3\urdf\00-arm-rs_asm-v3.urdf"),
)
