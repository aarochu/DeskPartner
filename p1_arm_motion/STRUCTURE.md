# P1 structure

```
p1_arm_motion/
├── README.md
├── STRUCTURE.md
├── __init__.py
├── arm_client.py          # thin wrapper over reBot SDK / serial
├── home.py                # go_home / is_at_home
├── pick_drop.py           # top-down pick + drop primitive (Skill A)
├── teach_destinations.py  # jog & save destinations.yaml
├── teach_replay.py        # record joint traj while gravity-comp, replay
├── gravity_comp.py        # launcher / notes around SDK example 9
└── tests/
    └── test_pick_drop_dry.py
```

## Primitive contract (P3 → P1)

```python
pick_and_drop(
    target_xy_mm=(x, y),
    grasp_height_mm=15,
    destination="trash",  # resolves via destinations.yaml
)
```

Sequence: hover → slow descend → close → lift → transit height → move above dest → open → retreat → caller returns home for photo.
