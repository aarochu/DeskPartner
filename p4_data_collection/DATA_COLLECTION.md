# Data collection checklist (P4 → P5)

**Target:** ≥50 clean episodes before handoff.  
**Task (ground truth):** `Pick crumpled paper and drop in trash`  
**Cams:** `front` (overhead) + `side` (45°) — locked before episode 1, never moved.  
**Arm:** single follower only. No bimanual / second-arm channels.

---

## Before the first episode

1. [ ] Ports: follower `/dev/ttyACM0`, leader `/dev/ttyUSB0` (see `config/recording.yaml`)
2. [ ] `sudo chmod 666` serial devices
3. [ ] Leader calibrated; follower zero + gripper closed
4. [ ] Both cameras physically locked; gripper does **not** fully occlude paper in both views at grasp
5. [ ] `python -m p4_data_collection.check_camera_lock --save-ref`
6. [ ] Teleop feels smooth (`./p4_data_collection/scripts/teleop.sh`)
7. [ ] One **throwaway** episode + `python -m p4_data_collection.verify_episode_format` → **PASS**

## Every session start

```bash
python -m p4_data_collection.check_camera_lock    # must PASS
python -m p4_data_collection.batch_record --num 50
```

Or single episode:

```bash
python -m p4_data_collection.record_episode --num 1
python -m p4_data_collection.verify_episode_format
```

---

## Staging consistency

| Rule | Spec |
|------|------|
| Object | One crumpled paper ball |
| Start region | Same zone patch; allow ±5 cm after first ~20 eps |
| Destination | Taught **trash** container (same every episode) |
| Background | Do not change mid-dataset |
| Lighting | Locked exposure; re-lock + new camera ref if booth lights change |

## Reset procedure (between episodes)

1. Open gripper, clear any miss / drop.
2. Arm back to consistent ready / home.
3. Re-crumple spare paper from the spare box (same size/feel).
4. Place paper in the staging mark.
5. Press Enter in `batch_record` only when ready — **do not rush**.

## Clean grasping technique (everyone must match)

1. Approach from above (top-down), no side swipes.
2. Center the gripper on the ball before closing.
3. Close fully, lift straight up, transit high, release over trash center.
4. No nudging with fingers mid-grasp; no “fix” retries inside one episode — re-record the episode.
5. Keep both cameras seeing the paper until lift.

**Rule of thumb:** you should be able to redo the task by watching only the `front` + `side` feeds.

## Fail-loud rules

- Verify after **every** episode (especially #1).
- A FAIL from `verify_episode_format` means: fix config / re-record — do **not** keep the ep.
- Phantom second-arm channels → stop the whole batch and fix robot.type / recording config.
- Camera lock FAIL → stop; remount / re-lock / `--save-ref` only if intentional and you accept restarting the dataset.

## Handoff to P5

- [ ] ≥50 PASS episodes
- [ ] Same `repo_id` + dataset path written in `CHECKLIST.md`
- [ ] `verify_episode_format` PASS on final dataset root
- [ ] Tell P5 the camera keys (`front`, `side`) and task string exactly
