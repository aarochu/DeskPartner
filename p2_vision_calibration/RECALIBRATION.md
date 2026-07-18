# 10-minute recalibration (never seen this code before)

**Goal:** click a pixel → arm tip within **5 mm**. Stop everything else until that passes.

Work from the DeskPartner repo root on the Ubuntu robot laptop:

```bash
cd ~/DeskPartner   # or your path
source .venv/bin/activate   # if you use one
```

## 0. Stop and make safe (30 s)

1. Stop the cleaner / orchestrator.
2. Hand on e-stop. Move arm clear if needed.
3. Check: C-clamps tight? Camera mount still square? Four ArUco markers (ids 0–3) still at zone corners?

## 1. Detect markers + lock exposure (2 min)

Clear people from the frame. Lighting should match the demo booth.

```bash
python -m p2_vision_calibration.detect_aruco_corners --camera 0 --show
```

- Must see all four ids `0,1,2,3`. If not: fix tape/lighting, rerun.
- Writes `data/calibration/calibration.json` (pixel centers + exposure lock).

## 2. Fit pixel → plane homography (30 s)

Zone size default is 400×300 mm. Change `--plane` only if you re-taped a different size.

```bash
python -m p2_vision_calibration.pixel_to_plane --fit
```

## 3. Plane → arm registration with P1 (4 min)

For each of the four markers: jog the **arm tip** to the marker center, press Enter.

```bash
DESKPARTNER_DRY_RUN=0 python -m p2_vision_calibration.plane_to_arm_registration --live
```

(If the arm client isn’t hooked up yet, use `--manual` and type the tip x,y in mm from the SDK FK readout.)

## 4. Empty-desk reference (30 s)

Clear the zone completely:

```bash
python -m p2_vision_calibration.empty_desk_reference --camera 0
```

## 5. Click-to-verify exit criterion (2 min)

```bash
DESKPARTNER_DRY_RUN=0 python -m p2_vision_calibration.click_to_verify --camera 0 --live
```

- Left-click 3 random spots in the zone.
- Each must land within **5 mm** (printed as PASS/FAIL).
- **If any FAIL → do not resume the demo.** Re-check mounts and repeat from step 1.

Optional single-point arm-only test (P1):

```bash
DESKPARTNER_DRY_RUN=0 python -m p1_arm_motion.verify_click_to_reach 180 40 --live
```

## 6. Resume

All outputs live in one file:

`data/calibration/calibration.json`

P3 loads this via `p2_vision_calibration.pixel_to_arm.CalibrationChain`.

---

### Quick command cheat sheet

| Step | Command |
|------|---------|
| Markers | `python -m p2_vision_calibration.detect_aruco_corners --camera 0 --show` |
| Homography | `python -m p2_vision_calibration.pixel_to_plane --fit` |
| Plane→arm | `DESKPARTNER_DRY_RUN=0 python -m p2_vision_calibration.plane_to_arm_registration --live` |
| Empty desk | `python -m p2_vision_calibration.empty_desk_reference --camera 0` |
| Verify ≤5mm | `DESKPARTNER_DRY_RUN=0 python -m p2_vision_calibration.click_to_verify --live` |
