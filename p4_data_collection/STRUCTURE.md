# P4 structure

```
p4_data_collection/
├── README.md
├── STRUCTURE.md
├── CHECKLIST.md
├── scripts/
│   ├── teleop.sh
│   ├── record_episodes.sh
│   └── verify_single_arm_dataset.py   # Friday throwaway gate
└── notes/
    └── staging.md
```

Dataset lands under `~/.cache/huggingface/lerobot/` (LeRobot default) or a path you set — tell P5 the `repo_id` and local path.
