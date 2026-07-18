# P2 — Vision & Calibration (Track A, must ship)

**Owns:** camera mount + exposure lock, ArUco homography, click-to-verify, plane→arm registration (joint with P1 Friday night), empty-desk reference, CV fallback, centroid refinement, 10-minute recal procedure.

**Handoff out to P3:** `pixel_to_arm(u, v) → (x_mm, y_mm)` and refined centroids inside VLM boxes.

## Friday night exit

**Click a pixel on screen → arm tip within 5 mm.** Nothing else proceeds until this passes.

## Saturday

- Empty-desk reference after lighting locked.
- CV fallback detector + color rules in `config/cv_color_rules.yaml`.
- Rehearse [`RECALIBRATION.md`](./RECALIBRATION.md) once (someone will bump the camera).

## Hardware notes

- Color camera only (VLM needs color). Direct USB, no hubs.
- Lock exposure/WB the moment booth lighting is final.
- Markers: 50 mm ArUco 4×4, ids 0–3 at zone corners.
