# ReBot Teleop GUI

This is the local manual-control panel for the connected ReBot leader and
follower. It runs entirely on this Mac at `http://127.0.0.1:8765/` and starts
the verified LeRobot teleoperation process directly.

## Open it

Double-click `07_teleop_gui.command` in the parent folder. Keep the terminal
window open while using the panel. If the server is already running, the
launcher simply reopens it.

Use **Collect & train** in the top bar for the two-camera preflight,
high-rate/30-FPS demonstration recorder, dataset validation, and native 7D
MolmoAct2 training recipe. The full procedure is in the parent folder's
`TRAINING_GUIDE.md`.

The collection page automatically loads the pair-specific
`config/training_profile.json`. It verifies both calibration files and all five
runtime files that implement the leader/follower coordinate contract before
recording; the collector explicitly selects those calibration directories with
interactive recalibration disabled. Dataset resume, validation, and command
generation require authenticated profile and collection-contract digests.

During collection, the episode timer never auto-accepts a take. Use **Finish &
keep episode** for a clean success, or select a reason and use **Mark failed &
re-record**. Successes stay unlabeled and enter LeRobot; failures are labeled
and excluded. Every attempt is independently archived with two MP4s, a
synchronized `.rrd`, and an editable JSON sidecar under
`training-runs/attempts/<dataset>/<attempt-id>/`. The training page can replay
the `.rrd` in Rerun and can review every finished attempt. If a take was kept
by mistake, **Mark failed & exclude from LeRobot** removes that physical
episode from the active success-only dataset, reindexes and fresh-load verifies
the remaining episodes, and preserves the original dataset revision plus both
videos and the Rerun archive. Aborted and system-error attempts can also receive
a review reason without changing their recorded system outcome.

The follower's current seven-joint pose and the passive leader's corresponding
pose are captured once, when collection connects. After every kept or failed
attempt—including the final kept attempt—the follower automatically returns to
that session-home pose at a capped 120 deg/s. The operator returns the passive
leader and task objects manually. Recording cannot resume until both arms stay
within the captured pose tolerances and the configured reset timer completes.
Stop or abort disconnects without initiating extra return motion.

## Manual operation

1. Pick a preset or type custom values.
2. To set one motor velocity for the whole arm, enter it under **Set every
   motor** and click **Apply to all**.
3. Use the **Baseline speed multiplier** for proportional changes. Quick
   buttons provide 0.5×, 1×, 1.5×, 2×, 3×, and 4×; the number field accepts
   0.1×–10×. Every multiplier is recalculated from the saved baseline, so
   repeated clicks never compound.
4. For asymmetric limits, edit the seven joints individually.
5. Click **Start teleop**. The state changes from `STARTING` to `RUNNING` when
   the backend receives the first measured loop-rate sample.
6. Click **Stop teleop** or press Space while you are not typing in a field.
   Stop sends `SIGINT` to the teleop process group and waits for the follower
   and leader disconnect routines.

Settings remain editable during a run, but they take effect on the next Start.
The control panel never changes a running process behind your back.

## What each number controls

| Control | Meaning | GUI range |
| --- | --- | ---: |
| Command rate | How often the leader is read and a follower command is sent | 1–240 Hz |
| Motor velocity | Per-joint velocity passed to the follower driver, in joint order J1–J7 | 0.1–2000 °/s |
| Step limit | Maximum change accepted in one control loop | 0.01–45 °/cycle |
| Gripper force | Driver force/torque ratio for the gripper | 0–1 |
| Run duration | Automatic stop time; `0` means continuous | 0–86400 s |

The effective commanded rate is bounded by both the motor velocity and the
step limit:

`effective cap = min(motor velocity, command Hz × step limit)`

The multiplier scales this effective cap linearly. The manual GUI now starts
with 4x applied to the Hand-tracking 2000 baseline, matching the verified
240 Hz / 2000 deg/s / 33.6 deg/cycle operator setting. The motor ceiling means
the effective joint cap remains 2000 deg/s and the GUI correctly marks it
`LIMITED`.

| Hand-tracking setting | Command rate | Motor velocity | Step limit |
| --- | ---: | ---: | ---: |
| 1× baseline | 240 Hz | 2000 °/s | 8.4 °/cycle |
| 4× default | 240 Hz | 2000 °/s | 33.6 °/cycle |

When a requested multiplier reaches the configured Hz, velocity, or step
ceiling, the GUI clamps the generated draft and shows the actual achieved
multiplier with a `LIMITED` label. **Set current as baseline** snapshots the
current Hz, step limit, and all seven velocities as the new 1× reference.

The motor and leader serial baud rates are hardware protocol settings, not
motion-speed settings, so they remain fixed at the values used by the working
ReBot setup.

## Included presets

| Preset | Rate | Motor velocity | Step limit |
| --- | ---: | ---: | ---: |
| Low | 10 Hz | 5 °/s | 0.5 °/cycle |
| Balanced | 15 Hz | 15 °/s | 1.0 °/cycle |
| Responsive | 30 Hz | 150 °/s | 5.0 °/cycle |
| Fast 500 | 60 Hz | 500 °/s | 8.3 °/cycle |
| Low-latency 500 | 120 Hz | 500 °/s | 4.2 °/cycle |
| Hand-tracking 2000 baseline | 240 Hz | 2000 °/s | 8.4 °/cycle |

Hand-tracking 2000 is the multiplier baseline; 4x is applied by default, so the
actual starting step limit is 33.6 deg/cycle and matches the collection
profile. Any edited number turns the draft into a Custom profile and is saved
in this browser.

## Status and logs

- **Actual Hz** is measured by the live LeRobot loop, not copied from the
  requested value.
- The device cards show the detected physical serial ports and calibration
  state.
- While running, `Serial ownership: Teleop` means this GUI's child process owns
  both ports as expected.
- The log shows the exact effective settings, connection messages, driver
  warnings, clamps, exceptions, and disconnect result.
- The backend allows only local same-origin control requests and does not accept
  commands, paths, ports, or executable names from the web page.

Support the follower when stopping because the configured disconnect routine
releases follower torque.
