# 10-minute recalibration (camera or arm bumped)

Target: back under 5 mm click-to-tip error in ≤ 10 minutes.

1. **Don't panic.** Stop the orchestrator. Arm to home. E-stop if needed.
2. **Check physical:** C-clamps tight? Camera mount still square? Markers still at corners?
3. **Re-lock exposure** if booth lights changed; re-shoot empty-desk reference.
4. **Re-detect ArUco** — run `python -m p2_vision_calibration.aruco_homography --save`.
5. **Re-register plane→arm** with P1: jog tip to each of 4 marker centers, record, fit.
6. **Click-to-verify** — must pass ≤ 5 mm on at least 3 random points.
7. Resume demo. Log the incident in `runs/` notes.

If markers moved: re-tape before step 4. Zone size does not change unless reach changed.
