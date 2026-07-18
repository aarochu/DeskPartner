# P4 structure

```
p4_data_collection/
├── DATA_COLLECTION.md
├── README.md
├── CHECKLIST.md
├── config_loader.py
├── record_episode.py          # lerobot-record from config/recording.yaml
├── verify_episode_format.py   # fail-loud after EVERY episode
├── batch_record.py            # N eps + manual ready + verify
├── check_camera_lock.py       # preflight vs episode-1 ref
└── scripts/
    ├── record_episode.sh
    ├── teleop.sh
    └── verify_single_arm_dataset.py   # thin alias → verify_episode_format
```

Config: `config/recording.yaml` (ports, cams, task, expect schema).
