# P1 structure

```
p1_arm_motion/
├── README.md
├── STRUCTURE.md
├── arm_client.py              # DryRun + RebotArmEndPose wrapper (meters)
├── pick_and_drop.py           # named steps + StepError
├── teach_home.py
├── teach_destination.py
├── reach_check.py             # Friday exit: zone corners reachable
├── verify_click_to_reach.py   # P2 exit helper: move to (x,y) mm
├── home.py
├── gravity_comp.py
└── teach_replay.py
```

## SDK calls (do not reimplement)

| Need | API |
|------|-----|
| Connect | `RebotArm` + `RebotArmEndPose.start()` |
| Cartesian | `move_to_ik` / `move_to_traj(..., duration=)` |
| Gripper | `open_gripper` / `close_gripper` |
| FK tip | `joint_to_pose(q)` |
| Shutdown | `ctrl.end()` |

## Live vs dry-run

Default is dry-run (`DESKPARTNER_DRY_RUN=1`). Hardware:

```bash
DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.pick_and_drop --live ...
```
