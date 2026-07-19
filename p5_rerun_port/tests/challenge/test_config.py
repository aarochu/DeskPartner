from pathlib import Path

import pytest

from p5_rerun_port.challenge.config import ChallengeConfig, ConfigError


CONFIG = Path("config/rerun_query_challenge.yaml")


def test_checked_in_config_locks_real_sources_and_robot_contract() -> None:
    config = ChallengeConfig.load(CONFIG)
    assert [(s.repo_id, s.revision, s.role, s.expected_items) for s in config.sources] == [
        ("Cornerf/rebot-can-sort-stage1-v1-smoke", "74d1f300786d58b4f6f55e1798cbb1a1a48f5409", "success", 52),
        ("Cornerf/rebot-two-can-recycle-v2-smoke", "778d0bf5de1096a80b1cf355073e369faa1409da", "success", 25),
        ("Cornerf/rebot-can-sort-stage1-v1-failed", "4952b618a23f8f2e5b09f736cea0a490c62e57b4", "failure", 19),
        ("Cornerf/rebot-two-can-recycle-v2-failed", "2d9ea53cf8f4835fcfc1656b23d56308696b3e5b", "failure", 6),
    ]
    assert config.joint_names == (
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
        "wrist_yaw", "wrist_roll", "gripper",
    )
    assert config.camera_keys == ("front", "side")
    assert config.destination_repo == "Cornerf/rebot-cansort-rerun-curated"


def test_config_rejects_duplicate_repo_or_non_sha_revision(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nsources: []\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        ChallengeConfig.load(path)
