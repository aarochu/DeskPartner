# P1 — Arm & Motion (Track A, must ship)

**Owns:** zero calibration first, home pose, reach envelope, pick/drop primitive, grasp heights, destination teaching, gravity-comp + teach-by-grabbing.

**Handoff in:** arm coordinates (mm) + category from P3  
**Handoff out:** pick/drop executed; destinations filled in `config/destinations.yaml`

## Friday night order

1. `uv run python example/2_zero_and_read.py` in `~/reBotArm_control_py` — before anything else.
2. Define out-of-frame **home** → write joints into `config/arm.yaml`.
3. Jog tip to all four future zone corners; confirm reach. Zone size follows reach.
4. Build hardcoded crumpled-paper pick-and-drop (exit for sleep with P2 cal).
5. Teach trash / pen_cup / tray by jogging; save to `config/destinations.yaml`.

## Saturday

- Harden `pick_drop.py` + per-type grasp heights.
- Own party tricks: `9_gravity_compensation.py` path + `teach_replay.py`.
- Hand on e-stop; keep speed limits low until consistent.

## Key SDK scripts (deck)

| Script | Role |
|--------|------|
| `2_zero_and_read.py` | Zero + monitor |
| `7_arm_ik_control.py` | Cartesian jog |
| `8_arm_traj_control.py` | Smooth trajectories |
| `9_gravity_compensation.py` | Weightless act 2 |

## Safety

- Speed/accel caps from `config/arm.yaml`.
- Never leave arm powered unattended during bring-up.
- Foam tape on fingers before object tuning.
