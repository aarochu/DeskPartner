# P4 — Data Collection (Track B, bonus)

**Owns:** teleop practice until smooth, 50+ crumpled-paper pick episodes, consistent staging/resets, LeRobot dataset hygiene for P5.

**Quality of demos is the single biggest determinant of whether Track B works.**

## Constraints

- **Single arm** (B601 follower + 102 leader). Station is normally bimanual — recordings must contain **ONLY** the active arm's state and actions. No phantom second-arm channels, or the single-arm MolmoAct 2 fine-tune config chokes.
- **Cameras:** overhead (`front`) + 45° (`side`). Lock both before episode 1; MolmoAct 2 trains on whatever views are in the dataset. Prefer views where the gripper does not block the object at grasp.
- Arm block: **Saturday midday**. Get one throwaway episode Friday night and **verify schema** that night.

## Friday night

1. Calibrate leader (`lerobot-calibrate` — see root README).
2. `scripts/teleop.sh` until motions feel smooth.
3. Record **one** throwaway episode with `scripts/record_episodes.sh` (`NUM_EPISODES=1`).
4. Run `scripts/verify_single_arm_dataset.py` on that episode — must pass before Saturday.

## Saturday midday block

- ≥50 episodes, same task: crumpled paper → trash (one task string, keep it).
- Identical camera keys, lighting, staging, grasp style every episode.
- Clean resets between episodes.
- Hand organized dataset path + `repo_id` to P5 by end of block.

## Don'ts (from deck + MolmoAct 2)

- Don't move cameras mid-dataset.
- Don't mix inconsistent behaviors.
- Don't use USB hubs for cameras.
- Don't record bimanual / dual-follower schemas "just in case."
- Don't keep episodes where the gripper fully occludes the object in both views at grasp.

Rule of thumb: you should be able to do the task by only watching the camera feeds.
