# P2 — Vision & Calibration (Track A, must ship)

**Owns (per GROUND_TRUTH):** camera mount + exposure lock, ArUco homography, click-to-verify, plane→arm registration (joint with P1), empty-desk reference, CV fallback, centroid refine, 10-min recal procedure.

**Shared truth:** `data/calibration/calibration.json`

**Handoff out to P3:** `pixel → arm_mm` via `CalibrationChain.pixel_to_arm_coords`

---

## Friday night (GT exit = tip ≤ 5 mm)

Nothing else builds until this passes.

```bash
python -m p2_vision_calibration.calibrate_all          # prints full checklist

python -m p2_vision_calibration.detect_aruco_corners --camera 0 --show
python -m p2_vision_calibration.pixel_to_plane --fit
DESKPARTNER_DRY_RUN=0 python -m p2_vision_calibration.plane_to_arm_registration --live
DESKPARTNER_DRY_RUN=0 python -m p2_vision_calibration.click_to_verify --live   # MUST pass
python -m p2_vision_calibration.empty_desk_reference --camera 0
```

- Markers: 50 mm ArUco 4×4, ids **0–3**, zone ~40×30 cm (after P1 reach-check).
- Overhead color cam (GT: innomaker / not OV9281 mono). Direct USB, no hub. Lock exposure/WB.
- Plane corners default mm in `calibration.json`: `(0,0) (400,0) (400,300) (0,300)` — override with `--plane` if zone size differs.

## P3 contract

```python
from p2_vision_calibration.pixel_to_arm import CalibrationChain
from p2_vision_calibration.centroid_refine import refine_centroid

cal = CalibrationChain()
u, v = refine_centroid(frame, bbox_xyxy)
arm_x_mm, arm_y_mm = cal.pixel_to_arm_coords(u, v)
```

## Recalibration (GT risk: bump)

Follow [`RECALIBRATION.md`](./RECALIBRATION.md) — target ≤10 minutes. Rehearse once.

## CV fallback (GT §2.2)

`cv_fallback.py` + empty-desk image. Orchestrator switches with `DESKPARTNER_PERCEPTION=cv`.

## Layout

See [`STRUCTURE.md`](./STRUCTURE.md).
