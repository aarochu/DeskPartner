# ReBot data collection and MolmoAct2 training guide

This is the current operating procedure for the connected **ReBot B601-DM arm**
(the robot that moves and will be trained), **reBot Arm 102 teleop arm** (the
human-operated demonstration input), and two external cameras. The code also
uses the technical names `follower` and `leader`, respectively. It uses the isolated
`ReBot_Operator_Kit`; datasets, camera frames, run manifests, validation
reports, and model outputs stay outside the DeskPartner Git checkout.

## 1. Current status

| Item | Current status |
| --- | --- |
| DeskPartner repository | `main` is clean and synchronized with `origin/main` at `79d9163` |
| Leader/follower | Detected and calibrated; collection does not start until both serial ports are free |
| Collection software | Ready: high-rate control loop plus independent 30 FPS dataset sampling |
| Logitech overhead camera, index 0 | Working; fixed top-down view of the complete task zone |
| Innomaker wrist camera, index 1 | Working; mounted on the ReBot claw for grasp/contact detail |
| Built-in Mac webcam, index 3 | Excluded and never used for recording |
| Dataset | None recorded yet |
| Trained model | None yet |
| Native MolmoAct2 | Supported on an NVIDIA GPU with the official current LeRobot policy and seven ReBot actions |
| Mission/New Theory hosted route | Account/client works, but the available SO-101 contract is 6D; raw 7D ReBot upload is intentionally disabled |

Open the workspace at:

`http://127.0.0.1:8765/training`

If it is not running, double-click `07_teleop_gui.command` and choose
**Collect & train** in the top bar.

### Locked default profile

The single source of truth is:

`ReBot_Operator_Kit/config/training_profile.json`

The training page loads it automatically and shows **VERIFIED & LOCKED** only
when all seven fingerprints match: both calibration files plus the follower
configuration/base/DM implementations and leader configuration/implementation
that define the coordinate conversion. **Restore verified defaults** resets the
form to its task, camera, collection-clock, motion, and normalization values.

The two calibration JSON files are motor-zero references, not standalone
training normalization. Direction/scale, seven-joint order, follower soft
limits, camera semantics, and MolmoAct2 quantile/gripper normalization are also
part of the versioned profile. Offline training never opens either serial port,
recalibrates an arm, or reapplies the leader transform. It trains on the exact
follower-coordinate values saved in the dataset.

Every new dataset carries `meta/rebot_training_profile.json`, and every run
manifest and validation report carries the same authenticated profile and
collection-contract digests. Resume also requires an exact match for task,
cameras, image sizes, clocks, step cap, velocity, and gripper force. Malformed,
incomplete, or failed run manifests block validation and training. After any
physical recalibration, motor-side zero change, joint-map change, or reuse with
another arm pair, create a new reviewed profile and dataset version; never
silently resume an older one.

## 2. The exact dataset contract

Every stored sample contains:

| Key | Shape/meaning |
| --- | --- |
| `observation.images.front` | Logitech overhead RGB video from camera 0, `3 × 480 × 640` after decode |
| `observation.images.side` | Innomaker claw/wrist RGB video from camera 1, `3 × 720 × 1280` after decode |
| `observation.state` | Seven follower joint positions in degrees |
| `action` | The exact seven-joint follower target actually sent after direction mapping, joint limits, and the per-tick step cap |
| `task` | One unchanged imperative instruction for the dataset |

The fixed joint order is:

1. `shoulder_pan.pos`
2. `shoulder_lift.pos`
3. `elbow_flex.pos`
4. `wrist_flex.pos`
5. `wrist_yaw.pos`
6. `wrist_roll.pos`
7. `gripper.pos`

The collector uses two independent clocks:

- **240 Hz control:** reads the leader and sends the follower target quickly
  enough to track normal hand motion.
- **30 FPS data:** stores synchronized images, follower state, and the action
  from that control tick. Training FPS remains stable even if the motor loop
  runs faster.

Camera sampling is nonblocking and has a bounded freshness policy. Frames up
to 250 ms old are normally fresh. To tolerate brief macOS scheduling or USB
jitter, at most two consecutive samples may use a frame up to 500 ms old; a
third consecutive over-250 ms sample, any frame over 500 ms old, a stopped read
thread, or a disconnected camera stops the take and excludes it from training.
Every attempt records per-camera freshness counts and maximum observed age in
its metadata. This tolerance never waits for a camera and therefore cannot
reduce the motor-loop rate to camera FPS.

Do not use DeskPartner's old P4 batch recorder for new data. It does not append
episodes correctly and would save the wrong action coordinate frame for this
ReBot runtime.

## 3. Camera and equipment placement

### Logitech overhead camera — index 0 (`front`)

1. Use a rigid boom, super-clamp, or tripod; never hand-hold the camera.
2. Put the lens 60–75 cm above the work surface and point it almost
   perpendicular to the table.
3. Frame the complete path: starting gripper pose, the whole reachable can
   scatter area, the taped sorting zone, and the path between them.
4. Crop out the laptop, leader arm, loose adapters, tools, and cable
   piles visible in the current saved frame.
5. Tape the mount feet and boom height. If the mount moves, create a new
   dataset version.

### Innomaker wrist/claw camera — index 1 (`side`)

1. Keep it rigidly attached to the ReBot claw. The internal dataset key remains
   `side`, but this is a moving wrist view rather than a fixed side tripod.
2. Frame both gripper fingers, the nearby object, and the contact point in the
   normal approach and release poses.
3. Lock the camera's orientation relative to the claw before episode 1. A mount
   shift requires a new dataset version.
4. Add cable strain relief and a service loop that allows the full arm and wrist
   range without pulling the camera, snagging, or crossing the jaws.
5. Use a direct USB port or powered USB 3.x hub if the simultaneous check falls
   below 27 FPS. Never substitute the built-in Mac webcam at index 3.

### Both cameras

- Keep mounts outside the follower's sweep and add USB cable strain relief.
- Use stable diffuse lighting; avoid a bright window behind the task, flicker,
  hard shadows, or lighting changes midway through a dataset.
- Current locked request: Logitech overhead `640×480` and Innomaker wrist
  `1280×720`, both at 30 FPS MJPG.
  Different input sizes are valid because MolmoAct2 preprocesses each image,
  but the size for each named camera must not change within the dataset.
- The automated gate requires simultaneous capture at at least 27 FPS and
  rejects a dark, covered, wrong-size, or unstable stream.
- Passing the numerical gate is not enough. Inspect both new snapshots and
  reject bad framing, clutter, glare, or occlusion yourself.

### Arms and workspace

- Secure the follower and leader bases. Put the leader outside both camera
  views.
- Match leader and follower to corresponding stable poses before Start.
- Keep cables, tools, people, and camera stands out of the follower sweep.
- Keep the follower supported when ending a session: the current disconnect
  routine releases follower torque.

## 4. First task and assumptions

The confirmed Stage-1 task is:

`Pick up one can and place it in the taped sorting zone`

Use that exact sentence for every episode and later for inference. Each episode
contains exactly one can pick and one placement in the fixed taped zone. Extra
cans enter the scene only after the first 150 clean episodes, and even then the
demonstrated action still picks exactly one can.

One hosted-training decision remains:

1. For Mission-hosted training, can Mission provision a native seven-action
   ReBot base? If not, may `wrist_yaw` be held at a fixed pose and removed only
   in a separate reviewed 6D adapter dataset?

Neither answer blocks raw seven-action collection. Do not transform or discard
the raw seventh joint during recording.

## 5. Record the first 10-episode Stage-1 checkpoint

1. Stop manual teleop. On the training page, **Serial ownership** must read
   **Free**.
2. Place one can in a random reachable location and keep the taped sorting zone
   fixed. For the first 150 episodes, remove extra cans and every unrelated
   object.
3. Click **Run 6-second camera check**.
4. Wait for the job to finish. Confirm both camera cards show the requested
   dimensions, at least 27 FPS, usable brightness, and useful new images.
5. Leave the task instruction unchanged.
6. Use dataset name `rebot-can-sort-stage1-v1-smoke`.
7. Use these initial settings:
   - Episodes: `10`
   - Episode time: `1000 s` upper guard; finish each take manually
   - Reset time: `20 s`
   - Control: `240 Hz`
   - Dataset: `30 FPS`
   - Motor velocity: `2000 °/s` on all seven joints
   - Step cap: `33.6 °/tick`
   - Gripper force: `0.05`
8. Put both arms in the exact stable pose you want to use as the home position
   for this session, and confirm the follower is clear to move. The collector
   captures both current poses when it connects after you click Start.
9. Check the readiness confirmation and click **Start collection** once.
10. Wait until both arms connect. Then move the leader smoothly and decisively.
11. After one smooth pick and a clean, visible release in the taped zone, click
    **Finish & keep episode**. The browser immediately confirms that Rerun and
    LeRobot saving has started; do not press Stop while saving. Wait until the
    next attempt is ready.
12. After every kept or failed take, the follower automatically returns to the
    captured session-home pose at a capped 120 deg/s. Manually put the passive
    leader back in its own captured pose, then scatter the can to a new random
    reachable position. Cover corners, edges, and center across the dataset.
    The next attempt stays blocked until both arms are aligned and the reset
    timer has completed; reset frames are not saved. The collector waits without
    an alignment timeout, so taking longer to return the passive leader never
    disconnects the session. Use **Stop & finalize** to end intentionally.
13. For a miss, collision, occlusion, dropped object, wrong destination, large
    correction, awkward hesitation, or any jerky-but-successful motion, choose
    a failure reason and click **Mark failed & re-record**. The take is excluded
    from training, but its videos and Rerun recording are preserved.
14. The 1000-second episode time is only an upper guard. Finish each take
    manually as soon as the task is complete. At the limit, recording
    pauses motion and recording, then waits until you
    explicitly keep, mark failed, or stop the attempt. It never auto-keeps and
    cannot fill the disk while waiting for a browser decision.
15. Use **Review and label every finished attempt** to open the separate
    overhead/wrist MP4 files, replay the synchronized take in Rerun, or correct
    a failure label. Successful takes deliberately have no label. If later
    review shows that a take kept as a success actually failed or was only a
    test, choose a reason and click **Mark failed & exclude from LeRobot**. The
    GUI removes that episode from the active success-only LeRobot dataset,
    reindexes and fresh-load verifies any remaining episodes, invalidates the
    previous validation report, and keeps a recoverable copy of the prior
    dataset revision. The attempt's MP4s, Rerun recording, and review history
    are never removed.
16. After 10 clean saved episodes, click **Stop & finalize** only if a new
    episode has already begun; the partial take is archived as `aborted` and
    excluded while all saved episodes remain. Stop/abort disconnects without
    initiating an automatic return; support the follower when stopping.
17. Select the smoke dataset and click **Validate selected dataset** with a
    minimum of `10` episodes. Continue only after the log prints `PASS`, then
    hand this checkpoint to the conversion/training owner immediately.

The browser controls send signals directly to the collector. They do not use
global keyboard shortcuts or require macOS Accessibility permission.

Every physical attempt has an independent review directory under
`training-runs/attempts/<dataset>/<attempt-id>/` containing `overhead.mp4`,
`wrist.mp4`, `attempt.rrd`, and `metadata.json`. Only clean, explicitly kept
attempts become LeRobot episodes. Only manually failed attempts receive a
failure label; stopped or collector-error attempts remain system outcomes.

## 6. Grow the Stage-1 dataset

After the first 10 episodes pass conversion and a throwaway train starts,
resume the same fixed-camera dataset. Do not move the cameras, taped zone, or
lighting, and do not change the task sentence.

Stage-1 target: **150 clean successful episodes**.

| Checkpoint | Purpose |
| ---: | --- |
| 10 | Prove LeRobot conversion and start the first throwaway train immediately |
| 50 | First comparison checkpoint; keep collecting while multiple models train |
| 100 | Retrain and compare again; verify both camera views and joint logs have not drifted |
| 150 | Complete the clean single-can Stage-1 dataset and evaluate the best checkpoint |
| After 150 | Add random extra cans in frame while still picking exactly one can per episode |

Use the whole reachable workspace: scatter the can to a new random location
after every episode and balance corners, edges, and center. Spot-check both
camera views and joint logs in Rerun about every 25 episodes. If performance is
weak, add targeted clean demonstrations rather than inserting failures into the
policy-training dataset.

Episode acceptance rule: keep only a complete success with smooth intent,
visible grasp, controlled transport, and visible release inside the taped zone.
Natural path variation is useful; fumbling and recovery after a failed grasp
are not useful in this first clean imitation dataset.

Use **Resume** only when the task text, named physical cameras, mounts,
resolution, FPS, joint mapping, and collection settings are unchanged. The GUI
never overwrites a populated dataset without explicit Resume.

## 7. What validation checks

The validation button loads a distributed sample of up to 257 frames and
requires all of the following:

- at least the requested episode count and at least one second per episode;
- exactly `front`, `side`, one 7D state, and one 7D action contract;
- exact ReBot joint names and order;
- finite state/action values within ReBot joint limits;
- follower action/state difference no larger than the recorded step cap plus
  `0.5°`, which catches leader-coordinate labels;
- consistent nonempty task text;
- constant per-camera decoded shape;
- non-black, non-covered sampled camera frames;
- `q01` and `q99` state/action statistics required for MolmoAct2 quantile
  normalization, with seven finite ordered values in each statistic;
- a dataset-side profile snapshot whose digest matches every collection
  manifest and the currently verified pair calibration/driver profile.

A validation report is written to:

`ReBot_Operator_Kit/training-runs/<dataset>--validation.json`

The report is invalidated automatically if more frames or episodes are later
added. The training command button only enables for the current validated
dataset.

## 8. Native seven-action MolmoAct2 training

This is the recommended route. Plain **Molmo 2** is a video/vision-language
model; **MolmoAct2** is the action policy. The official current LeRobot policy
accepts action dimension 7 and pads it to the released model's maximum action
dimension of 32.

### GPU requirement

Use Linux with an NVIDIA GPU. The official measurements report about 16.5 GiB
for action-expert-only training at batch 8 and 18.3 GiB at batch 16 on an H100,
before extra runtime headroom. A 24 GB or larger GPU is a practical starting
point; reduce `--batch_size=16` to `8` if the actual host runs out of memory.

### Prepare the GPU host once

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv sync --locked --extra molmoact2
source .venv/bin/activate
mkdir -p rebot-training/data rebot-training/models
```

The official current install requires Python 3.12 and the MolmoAct2 optional
dependencies. The `uv sync` command creates the isolated environment.

### Copy the validated dataset

Run from this Mac, replacing the GPU login and path:

```bash
./08_share_dataset.command rsync \
  <gpu-user>@<gpu-host>:<lerobot-directory>/rebot-training/data/rebot-can-sort-stage1-v1-smoke
```

Copy the whole dataset directory, including `meta`, `data`, and `videos`.
Never copy only the MP4 files. The command validates the dataset and generates
a timestamped `SHARE_MANIFEST.json` with SHA-256 checksums before transferring
only changed files. It fails closed unless the collector is idle/finalized and
the GUI's validation report still matches the current episode/frame counts,
then shares from a stable copy-on-write snapshot so recording data is never
read or mutated during collection. The receiver runs
`09_receive_dataset.command verify` on the received directory.

For repeatable team-wide sharing through a private Hugging Face dataset:

```bash
./08_share_dataset.command hub \
  <owner-or-org>/rebot-can-sort-stage1-v1-smoke
```

The first run requires `./10_hf_oauth_login.command`. It installs a current
Hugging Face CLI in ignored local state and uses browser/device OAuth without
changing the pinned robotics runtime or asking you to paste a token. Later
runs reuse already uploaded content. Give teammates the immutable revision
printed by the command and the handoff in
`TEAMMATE_DATA_RECEIVER_PROMPT.md`; do not tell them to train from an
unspecified moving `main` revision.

The Hub repository slug must match the dataset's locked local slug. The
append-only smoke stream therefore stays `rebot-can-sort-stage1-v1-smoke`.
Publish `rebot-can-sort-stage1-v1` only after building the separate reviewed
kept-only export namespace described in `AGENTS.md`.

### Generate and run the exact command

1. In the local GUI, select the validated Stage-1 dataset checkpoint.
2. Click **Generate command for selected dataset**.
3. Confirm the command card shows the same profile ID and digest prefix as
   **Default training profile**, then copy the command.
4. On the GPU host, enter the LeRobot checkout, activate `.venv`, and run the
   copied command.

The generated first run uses:

- original checkpoint `allenai/MolmoAct2`;
- ordered `front`, then `side` image keys;
- continuous 10-step action chunks;
- native 7D absolute follower joint poses;
- action-expert-only fine-tuning for 5,000 steps;
- quantile normalization for state/action and
  `policy.normalize_gripper=true` for ReBot's degree-scale gripper;
- 10% held-out episodes for offline loss checks;
- checkpoints every 1,000 steps.

Outputs are written under:

`rebot-training/models/rebot-can-sort-stage1-v1-smoke-molmoact2-rebot`

Do not launch training until local validation passes. Do not run this model
directly on the physical arm after loss decreases: first inspect the held-out
loss and saved processor/config, run offline inference on recorded frames,
then build and verify a ReBot rollout adapter that restores the saved
normalization and clamps every predicted joint to the same limits.

## 9. Mission Robotics / New Theory route

`http://127.0.0.1:3000` remains useful for the hackathon lessons, but its
current Collect controls target SO-100/101 Feetech hardware and must not own the
ReBot serial ports.

The live hosted `so101` MolmoAct2 contract currently expects:

- state `[6]`;
- action chunk `[30, 6]`;
- cameras named `top` and `side`;
- joints: pan, lift, elbow, wrist flex, wrist roll, gripper.

ReBot has the additional `wrist_yaw`, so sending the raw dataset to Newt would
fail the shape gate and would not define how to command the seventh joint.
The GUI therefore shows this route as blocked. If Mission provisions a native
ReBot model entry, use its live registry contract. If the team approves a 6D
route, create a separate derived dataset that fixes `wrist_yaw`, drops only
that dimension, renames `front` to `top`, and records the exact reversible
transform. Never alter the immutable 7D raw data.

## 10. Local paths and reproducibility

| Purpose | Path |
| --- | --- |
| GUI | `ReBot_Operator_Kit/teleop_gui` |
| Default calibration/training profile | `ReBot_Operator_Kit/config/training_profile.json` |
| High-rate collector | `ReBot_Operator_Kit/teleop_gui/controlled_record.py` |
| Camera reports/images | `ReBot_Operator_Kit/camera-check` |
| Raw datasets | `ReBot_Operator_Kit/data` |
| Run manifests and validation reports | `ReBot_Operator_Kit/training-runs` |
| Returned model files | `ReBot_Operator_Kit/models` |
| DeskPartner checkout | `work/DeskPartner` — code only; no raw data or credentials |

Each collection start writes a manifest containing the full profile snapshot
and digest, all seven runtime fingerprints, an authenticated collection
contract, discovered serial ports, and a lifecycle result. The dataset also
contains its own authenticated profile/contract sidecar so it remains
self-describing after copying to a GPU host. The manifest, complete dataset
directory, this guide, profile file, and generated GPU command are the minimum
package needed to reproduce a run on another machine.
