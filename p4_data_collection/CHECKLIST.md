# P4 episode checklist

- [ ] Ports: follower `/dev/ttyACM0`, leader `/dev/ttyUSB0` (or updated in `config/arm.yaml`)
- [ ] `sudo chmod 666` on serial devices
- [ ] Leader calibrated; follower at zero with gripper closed before first connect
- [ ] Overhead + 45° cam indices confirmed (`lerobot-find-cameras opencv`)
- [ ] Both cams **physically locked**; keys will be `front` + `side` for all episodes
- [ ] Exposure locked (same as Track A)
- [ ] Teleop feels smooth (joint_directions tuned if needed)
- [ ] Throwaway episode recorded Friday night
- [ ] `verify_single_arm_dataset.py` **PASS** (no phantom second-arm channels)
- [ ] Saturday: 50+ episodes, one task string, consistent resets, cameras unmoved
- [ ] Dataset path + `repo_id` written here for P5: `________________`
