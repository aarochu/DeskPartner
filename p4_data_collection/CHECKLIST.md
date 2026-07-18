# P4 episode checklist

- [ ] Ports: follower `/dev/ttyACM0`, leader `/dev/ttyUSB0` (sync `config/recording.yaml` + `config/arm.yaml`)
- [ ] `sudo chmod 666` on serial devices
- [ ] Leader calibrated; follower at zero with gripper closed before first connect
- [ ] Logitech overhead index 0 + Innomaker wrist index 1 confirmed (`lerobot-find-cameras opencv`)
- [ ] Both cams **physically locked**; LeRobot keys `front` + `side` for all episodes
- [ ] Exposure locked (same booth lighting as Track A)
- [ ] `python -m p4_data_collection.check_camera_lock --save-ref` after lock
- [ ] Teleop feels smooth (`joint_directions` tuned if needed)
- [ ] Throwaway episode recorded Friday night
- [ ] `python -m p4_data_collection.verify_episode_format` **PASS** (no phantom second-arm channels)
- [ ] Saturday: 50+ episodes, task = crumpled paper → **trash**, consistent resets, cameras unmoved
- [ ] `check_camera_lock` PASS at start of every session
- [ ] Dataset path + `repo_id` for P5: `________________`
