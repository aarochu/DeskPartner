# P4 — Data Collection (Track B, bonus)

**Owns (per GROUND_TRUTH):** teleop practice, 50+ crumpled-paper episodes, locked dual-cam hygiene, **single-arm-only** LeRobot state/actions (no phantom second arm).

**Never blocks Track A.** Arm block: Saturday midday. Friday night: one throwaway + format verify.

Quality of demos is the single biggest determinant of whether Track B works.

---

## Constraints (GT §3 / §10)

- Single arm: `seeed_b601_dm_follower` + `rebot_arm_102_leader`  
- Cams locked before episode 1: LeRobot keys **`front`** (overhead) + **`side`** (45°)  
- Prefer views where gripper does not block the object at grasp (MolmoAct 2 weak spot)  
- Record resolution in `config/recording.yaml` is **640×480** for both (training views). Physical overhead mount is shared with Track A VLM (which may grab higher res separately).

## Quick start

```bash
# once cams are physically locked for the dataset
python -m p4_data_collection.check_camera_lock --save-ref

# Friday night gate: throwaway + fail-loud verify
python -m p4_data_collection.record_episode --num 1
python -m p4_data_collection.verify_episode_format
# alias: python p4_data_collection/scripts/verify_single_arm_dataset.py

# every later session
python -m p4_data_collection.check_camera_lock          # must PASS
python -m p4_data_collection.batch_record --num 50      # Enter between resets
```

Teleop practice: `./p4_data_collection/scripts/teleop.sh`

## Config / docs

| Path | Role |
|------|------|
| [`config/recording.yaml`](../config/recording.yaml) | ports, cams, task, expect schema |
| [`DATA_COLLECTION.md`](./DATA_COLLECTION.md) | staging, reset, grasp technique, 50+ target |
| [`CHECKLIST.md`](./CHECKLIST.md) | tick boxes + repo_id for P5 |

**Default task (GT hero object):** `Pick crumpled paper and drop in trash`  
Override: `--task "..."` (do not change mid-dataset).

## Handoff to P5

Clean LeRobot dataset by Saturday midday + `verify_episode_format` PASS + same camera keys forever.

## Layout

See [`STRUCTURE.md`](./STRUCTURE.md).
