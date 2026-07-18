# P2 structure

```
p2_vision_calibration/
├── README.md
├── STRUCTURE.md
├── RECALIBRATION.md       # 10-minute procedure
├── __init__.py
├── camera.py              # capture with locked settings
├── aruco_homography.py    # pixel → plane mm
├── plane_to_arm.py        # plane mm → arm mm (4-point LS)
├── click_to_verify.py     # exit criterion UI
├── centroid_refine.py     # refine inside VLM bbox
├── cv_fallback.py         # empty-desk diff detector
└── calibrate_all.py       # orchestrates Friday night cal
```

## Contract API

```python
from p2_vision_calibration.plane_to_arm import Calibration

cal = Calibration.load("data/calibration")
x_mm, y_mm = cal.pixel_to_arm(u, v)
```
