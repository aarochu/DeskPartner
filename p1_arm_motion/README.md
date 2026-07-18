# P1 — Arm & Motion (Track A, must ship)

**Owns (per GROUND_TRUTH):** zero cal first, home pose, reach envelope, pick/drop primitive, grasp heights, destination teaching, gravity-comp + teach-by-grabbing.

**SDK:** wraps `reBotArm_control_py` `RebotArmEndPose` (`move_to_traj` / `move_to_ik` / gripper). No custom IK.

**Handoff in (from P3):** `arm_xy_mm` + item type + destination name → pick happens  
**Handoff out:** destinations + home filled in config; reliable `pick_and_drop`

Default is dry-run. Hardware: `DESKPARTNER_DRY_RUN=0` and `--live`.

---

## Friday night order (aligns with GT §5)

```bash
# 0) Zero BEFORE anything else (from ~/reBotArm_control_py)
cd ~/reBotArm_control_py && uv run python example/2_zero_and_read.py

# 1) Reach envelope BEFORE taping zone (GT: zone sized to reach)
cd ~/DeskPartner   # repo root
DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.reach_check \
  --corners 0,0 400,0 400,300 0,300 --live

# 2) After P2 click-to-verify ≤5mm — teach home + destinations (outside camera zone)
DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.teach_home --live
DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.teach_destination trash --live
DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.teach_destination pen_cup --live
DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.teach_destination tray --live

# 3) One hardcoded crumpled-paper pick-and-drop (GT Friday item 5)
DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.pick_and_drop \
  --x-mm 180 --y-mm 40 --item paper --dest trash --live
```

## Click-to-reach (P2 exit helper)

```bash
DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.verify_click_to_reach 180 40 --live
```

## Motion contract (GT §2.4)

Top-down only: hover → slow descend → grasp → lift → transit → release → **return home**.  
Photos are only taken from home (P3 enforces). No side grasps.

`pick_and_drop` raises `StepError` with `.step` in  
`hover | descend | grasp | lift | transit | release | return_home` (for closed-loop retries).

## Config

| File | Contents |
|------|----------|
| `config/arm.yaml` | home, `speed_multiplier` (default 0.5), grasp heights, transit Z |
| `config/destinations.yaml` | taught `xyz_m` / joints for trash, pen_cup, tray |

## Party tricks (GT second act)

```bash
cd ~/reBotArm_control_py
uv run python example/9_gravity_compensation.py
# teach-by-grabbing: python -m p1_arm_motion.teach_replay  (stub — wire when ahead)
```

Hand on e-stop. Keep `speed_multiplier` low near people.

## Layout

See [`STRUCTURE.md`](./STRUCTURE.md).
