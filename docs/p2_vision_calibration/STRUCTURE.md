# P2 structure

```
p2_vision_calibration/
├── RECALIBRATION.md                 # 10-minute procedure
├── calibration_io.py                # shared calibration.json
├── detect_aruco_corners.py
├── pixel_to_plane.py                # pixel_to_mm + fit H
├── plane_to_arm_registration.py     # plane_to_arm + interactive fit
├── pixel_to_arm.py                  # CalibrationChain (P3 contract)
├── click_to_verify.py               # ≤5mm exit criterion
├── centroid_refine.py
├── empty_desk_reference.py
├── cv_fallback.py
└── calibrate_all.py
```

## Shared truth

`data/calibration/calibration.json` — ArUco centers, exposure lock, H, plane→arm, empty-desk path.

## P3 contract

```python
from p2_vision_calibration.pixel_to_arm import CalibrationChain
cal = CalibrationChain()
arm_x_mm, arm_y_mm = cal.pixel_to_arm_coords(px, py)
```
