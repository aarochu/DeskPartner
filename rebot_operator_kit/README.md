# ReBot operator start guide

This is the reproducible operator workspace for the connected DeskPartner
ReBot system. Source is tracked in Git; `.gitignore` keeps calibration runtime
state, camera frames, datasets, attempt videos, logs, and models out of Git.

On a new Mac, run `../rebot_setup/setup.sh`, copy the two calibration JSON files
for this physical arm pair into the resulting
`../rebot_setup/vendor/rebot_lerobot/lerobot-home/calibration/` tree, then
double-click `07_teleop_gui.command`. Set `RUNTIME_ROOT` only when using a
runtime stored somewhere else.

For the current data-collection and MolmoAct2 workflow, use
[`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) and open
`http://127.0.0.1:8765/training` from the local GUI.

## Confirmed on this Mac

| Item | Current mapping | Status |
| --- | --- | --- |
| B601-DM follower | `/dev/cu.usbmodem00000000050C1`, USB `2e88:4603` | Seven motors found |
| reBot Arm 102 leader | `/dev/cu.usbserial-10`, USB `1a86:7523` | Servo IDs 0-6 online |
| Logitech overhead camera | OpenCV index `0`, dataset key `front` | Working; sees the full taped table zone |
| Innomaker wrist camera | OpenCV index `1`, dataset key `side` | Working; mounted on the ReBot claw for close-up grasp/release observations |
| Built-in Mac webcam | OpenCV index `3` | Explicitly excluded and never recorded |
| Calibration profile | `config/training_profile.json` for this exact physical pair | Fingerprinted and auto-loaded; do not recalibrate |

The only training cameras are the fixed overhead `Logitech BRIO` and the
claw-mounted `Innomaker-U20CAM-1080p-S1`. Camera indices can change after
reconnecting, so rerun the camera check after any USB topology change and
confirm the product/view mapping before recording.

## Teleop control panel (recommended)

Double-click `07_teleop_gui.command` to open the local ReBot Teleop GUI at
`http://127.0.0.1:8765/`. It provides manual Start and Stop buttons, measured
loop Hz, global and per-joint velocity fields, max-step control, gripper force,
optional duration, a non-compounding baseline multiplier, presets, device
state, and streaming logs. Settings can be edited while a run is active and
apply on the next Start.

The default preset is the verified hand-tracking setup: 2000 deg/s on all seven
joints, 240 Hz, and 8.4 deg/cycle. Choose any preset or type custom numbers;
**Apply to all** copies
the global motor velocity to all seven joints. The detailed control reference
is in `teleop_gui/README.md`.

## Equipment layout

### Follower arm

- Clamp the B601 base rigidly to the table with a metal C-clamp. The clamp and
  table edge must not enter the arm's joint sweep.
- Use only the follower's 24 V supply. Route the power cable behind the base,
  add strain relief, and keep its connector outside the camera/tripod area.
- Keep the hardware e-stop or power cut within one arm's reach of the operator.
- Leave at least one meter of clear space around the moving links. Remove cans,
  tools, loose cables, laptop power bricks, and people from that sweep volume.
- Support the follower whenever teleoperation stops: the current driver
  releases torque on disconnect and the arm can fall under gravity.

### Leader arm

- Use only the leader's 12 V supply. Never connect the follower's 24 V supply.
- Put the leader beside the operator, outside both camera views and outside the
  follower sweep. Secure the UC-01 and its first 3-pin bus cable so hand motion
  cannot pull the connector.
- Start each session with leader and follower in corresponding stable poses.
  The existing calibration belongs to this exact arm pair; do not run a new
  calibration merely because the computer or USB port changed.

### Overhead camera: `front`, index 0

- Use the Logitech camera as the fixed top-down task view unless a reconnect
  test proves the USB-to-index mapping changed.
- Mount it on a rigid boom or super-clamp 60-75 cm above the task surface, with
  the optical axis perpendicular to the table. Do not hand-hold it.
- Center the full manipulation zone, not the robot base. The image must include
  the pickup area, complete drop area, object, gripper, and gripper path at all
  times. A small strip of the base may remain at the edge.
- Put tape marks around the stand feet and on the boom height. Photograph the
  mount after it is final. Any mount movement creates a new dataset version.
- Remove people and unrelated objects from the image before episode 1.

### Wrist camera: `side`, index 1

- Use the Innomaker camera mounted rigidly on the ReBot claw. The dataset key
  remains `side` for schema compatibility, but the physical role is a moving
  wrist/gripper view, not a stationary side camera.
- Keep the gripper fingers, nearby object, and contact point visible. Verify the
  jaws do not block the grasp point in the usual approach pose.
- Secure the camera so it cannot rotate relative to the claw. Add strain relief
  plus enough cable service loop for the complete wrist and arm range without
  pulling, snagging, or entering the gripper.
- Clean the lens and confirm the image orientation before episode 1. Do not
  rotate or remount the camera within a dataset version.

### USB and lighting

- If both external cameras share a USB 2.0 hub, that is a bandwidth
  risk. Prefer separate direct ports or a powered USB 3.x hub. Keep the two arm
  serial adapters on separate USB branches when possible.
- The current gate is Logitech overhead `640x480` plus Innomaker wrist
  `1280x720`, MJPG, at 30 fps.
  If either view cannot sustain at least 27 fps or drops frames, first separate the USB paths.
  If that is impossible, run both cameras and the dataset at 15 fps; never mix
  30 fps and 15 fps views in one dataset.
- Use stable diffuse lighting from both sides of the work area. Avoid windows
  behind the object, flickering lamps, hard shadows, and automatic lighting
  changes. Lock exposure and white balance after placement if the camera tool
  allows it.

## Operating order

1. Double-click `07_teleop_gui.command` and open **Collect & train**.
2. Confirm **Default training profile** says **VERIFIED & LOCKED**. The page
   automatically loads the pair's task, motion, camera, 7D coordinate, and
   MolmoAct2 normalization defaults; use **Restore verified defaults** after
   any temporary edits.
3. Stop manual teleop so the collector has exclusive access to both serial
   ports.
4. Confirm the Logitech overhead mount and Innomaker claw mount are locked,
   arrange the actual task, and click
   **Run 6-second camera check**. Inspect both saved images; recording remains
   blocked until the automated gate and the human framing check both pass.
5. Record the first 10 clean Stage-1 can episodes from the training page. This uses
   the high-rate collector that saves the follower-space action actually sent
   to the motors; do not use the older `04_record_smoke.command` for new data.
6. Select the dataset and click **Validate selected dataset**. The generated
   training command stays disabled until the current dataset passes.
7. Start the throwaway conversion/training run as soon as the first 10 episodes
   pass, then resume the same fixed-camera Stage-1 dataset and checkpoint it at
   50, 100, and 150 clean successful episodes. Train models in parallel on an
   NVIDIA GPU host using the command generated by the GUI; do not attempt
   MolmoAct2 training on this Mac.

## Demonstration technique

- Use one plain-language task string for the entire dataset: `Pick up one can
  and place it in the taped sorting zone`.
- Pick exactly one can per episode, even if extra cans are visible. After every
  take, scatter the target can to a new random reachable position; cover
  corners, edges, and center rather than repeating a comfortable placement.
- Before starting collection, place both arms in the stable pose that should be
  this session's home. The collector captures their current poses on connect.
- After every kept or failed episode, the follower automatically returns to its
  captured pose at a capped 120 deg/s. Return the passive leader by hand and
  reset the can while keeping the taped zone fixed; the next attempt is blocked
  and the reset timer has completed. Reset motion is not recorded.
- Move smoothly and decisively. Delete jerky-but-successful takes as well as
  misses. Avoid repeated corrections, collisions, hesitations, and covering
  the object with your hand or body.
- End only after the object is visibly released inside the destination, then
  restore the scene during the automatic-return/reset period.
- For a failed, collided, obstructed, or heavily corrected take, choose a
  failure reason and click **Mark failed & re-record**. Failed takes are
  archived for review but never added to the training dataset; successful
  takes remain unlabeled.
- Every take has separate `overhead.mp4`, `wrist.mp4`, `attempt.rrd`, and
  `metadata.json` files under `training-runs/attempts/<dataset>/<attempt-id>/`.
  Use **Review every attempt** in the GUI to replay or relabel failures.
- The first 10 episodes are the pipeline smoke checkpoint. Hand them to the
  training owner immediately, then continue clean collection and retrain at 50,
  100, and 150 episodes. Only after 150 should random extra cans appear in frame;
  the demonstrated action must still pick exactly one can.
- Keep camera names (`front`, `side`), physical cameras, mounts, resolution,
  frame rate, lighting, task text, joint mapping, and arm IDs unchanged across
  recording, training, and evaluation.

## Training boundary

Offline training does not recalibrate or connect to either arm. The collector
loads the pinned leader/follower zero references explicitly, records exact
follower-coordinate state/action values, and writes
`meta/rebot_training_profile.json` into the dataset. Validation and training
command generation require that sidecar, the completed run manifests, both
calibration hashes, and all pinned coordinate-driver implementation hashes
match the active profile. Resume additionally requires the original task,
cameras, clocks, and motor settings. If the arm is physically recalibrated or
the joint mapping changes, create a new reviewed profile and dataset version;
never resume an older profile.

MolmoAct2 learns from both cameras, the seven-joint follower state, the task
sentence, and follower-coordinate actions jointly; it is not trained once per
camera. Ten episodes are only the conversion/schema smoke checkpoint. The
Stage-1 target is 150 clean episodes, with comparison checkpoints at 50 and
100, using the GUI's native 7D MolmoAct2 recipe on an NVIDIA GPU.

Mission/New Theory's currently available hosted SO-101 policy requires 6D
state/action and has no `wrist_yaw`. Raw ReBot collection therefore remains 7D,
and hosted upload remains disabled until Mission provides a native 7D base or
the team explicitly approves a fixed-`wrist_yaw` 6D adapter.

Never drive the real arm immediately from a new checkpoint. First inspect
dataset frames, validate policy outputs offline, run shadow mode, then test an
empty workspace at 10-20 percent speed with an independent e-stop operator.

The full reproducibility runbook is in
[`../REBOT_ARM_COMMANDS.md`](../REBOT_ARM_COMMANDS.md).
