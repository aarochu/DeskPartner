# Hackathon submission and judge demo

Use this as the short presentation layer. Detailed setup and safety procedures
remain in `docs/Rerun_bounty_progress.md`, `docs/p5_rerun_port/README.md`, and
`p3_vlm_orchestrator/PERSON4_RUNBOOK.md`.

## 60-second pitch

> We ported the Rerun SO-101 learning loop to reBot, a different seven-joint
> robot with Damiao motors, two named cameras, and an extra wrist-yaw degree of
> freedom. One episode flows from joint, goal, camera, and URDF logging into a
> local Rerun catalog; the Query API compares commanded and observed motion so
> we can reject bad takes; selected episodes export in a reBot LeRobot v3
> schema; and the same trajectory can be replayed or handed to our guarded
> learned-policy runner. The important part is the loop, not a URDF screenshot:
> collect, inspect, curate, export, and close the loop. Our checked-in evidence
> proves that entire path with synthetic hardware. [Only after a successful
> rehearsal: We also ran the same path on the physical reBot.] The result is a
> reproducible non-SO-101 integration with auditable safety and data contracts.

Do not say the bracketed live sentence unless the live row in the evidence
table below has been filled with a recording, command log, and outcome.

## Prize-track mapping

| Track | Submission position | Evidence and remaining gate |
|---|---|---|
| Rerun non-SO-101 end-to-end port ($1k) | **Primary** | `p5_rerun_port` covers log, record, catalog/query, LeRobot export, and replay for a 7-DOF reBot. Synthetic path is documented green; physical end-to-end remains a required venue proof. |
| Rerun Query API ($2k) | **Secondary** | `query_api_cli` uses the Rerun server/reader path for schema, entity reads, and goal-versus-position comparison. The tracked example is `docs/p5_rerun_port/examples/cans_query_report.md`; current proof is synthetic. |
| Interesting Rerun Viewer ($2k) | **Additional submission** | Interactive recordings activate a purpose-built Blueprint with synchronized front/side cameras, the 3D reBot URDF, goal-versus-position traces, and the time panel in one operator view. Rehearse the real Viewer before claiming live use. |

## Evidence boundary

| Capability | Synthetic evidence currently supported | Live evidence required before claiming it |
|---|---|---|
| reBot telemetry, cameras, goals, and URDF in Rerun | `log_rebot --fake --teleop`; tracked progress marks dry-run done | Saved live `.rrd` showing the physical seven-joint follower, both real cameras, and URDF |
| Episode recording and catalog | `record_episode --fake`; dry-run `.rrd`, trajectory sidecar, frames, and catalog path documented | One clean physical can-to-zone episode with matching `.rrd`, metadata, trajectory, and camera frames |
| Query API curation | Schema/entity/goal-vs-position commands and tracked example report | Run the same query against the physical episode and show the episode ID on screen |
| Central Viewer workflow | The checked-in Blueprint and real synthetic `.rrd` put both cameras, the URDF, and tracking error in one layout | Open the physical episode with the Blueprint active and use it during collection/curation, not only as a final screenshot |
| LeRobot export | Fallback export is documented green with reBot type, seven joints, `front`, then `side` | Load and validate an export made from the physical episode; record exact output path/revision |
| Replay / close loop | `replay_episode --fake` is documented green | Physical replay only after a supervised rehearsal, clear workspace, e-stop operator, and conservative speed |
| Learned autonomous pick | Guarded runner, offline/shadow/live gates, and tests exist | A real checkpoint must pass inspect, offline, shadow, and empty-workspace live gates before filming a can pick |
| Training quality or success rate | Training/evaluation tooling exists | A real checkpoint plus held-out trial report; do not infer success from training loss or code tests |

## Judge demo sequence

### 1. State the evidence level (5 seconds)

Say either “This is the reproducible synthetic pipeline” or “This is the live
pipeline validated in rehearsal.” Never switch labels mid-demo.

### 2. Show one complete synthetic episode (25 seconds)

Run from the repository root. A timestamped dataset avoids reusing stale demo
artifacts:

```bash
export DEMO_DATASET="hackathon-demo-$(date -u +%Y%m%dT%H%M%SZ)"

python -m p5_rerun_port.record_episode \
  --fake --dataset "$DEMO_DATASET" \
  --task "Pick up one can and place it in the taped sorting zone" \
  --tag "Good episode" --seconds 5 --no-viewer

rerun "recordings/$DEMO_DATASET/episode_01.rrd"
```

In Rerun, point out `follower/position`, `follower/goal`, the seven-joint arm,
and both camera streams. If any entity is absent, stop and use the last
rehearsed artifact without calling the new run successful.

### 3. Query and curate (15 seconds)

```bash
python -m p5_rerun_port.query_api_cli --dataset "$DEMO_DATASET" --schema
python -m p5_rerun_port.query_api_cli \
  --dataset "$DEMO_DATASET" --compare goal-vs-position
python -m p5_rerun_port.query_dataset \
  --dataset "$DEMO_DATASET" --tag "Good episode"
```

Explain that goal-versus-position error exposes lag, dropped samples, or bad
takes before they enter training.

### 4. Export and close the synthetic loop (10 seconds)

```bash
python -m p5_rerun_port.export_lerobot \
  --dataset "$DEMO_DATASET" --tag "Good episode" --fallback
python -m p5_rerun_port.replay_episode \
  --dataset "$DEMO_DATASET" --episode episode_01 \
  --fake --speed 0.5 --no-viewer
```

Call this a schema/export and fake-replay proof. Do not call it physical replay.

### 5. Optional live closeout

Only use this after the full preflight below and one successful private
rehearsal. Keep a dedicated operator on the physical e-stop/power cut.

```bash
./rebot_operator_kit/01_check_hardware.command
./rebot_operator_kit/02_dual_camera_check.command

python -m p5_rerun_port.log_rebot --teleop --seconds 20
python -m p5_rerun_port.record_episode \
  --dataset cans-live-demo \
  --task "Pick up one can and place it in the taped sorting zone" \
  --tag "Good episode"
python -m p5_rerun_port.query_api_cli \
  --dataset cans-live-demo --compare goal-vs-position
python -m p5_rerun_port.export_lerobot \
  --dataset cans-live-demo --tag "Good episode"
```

Physical replay is a separate safety decision. Do not improvise it during the
judge session. If it was approved and rehearsed, use the documented command:

```bash
python -m p5_rerun_port.replay_episode \
  --dataset cans-live-demo --episode episode_01 --speed 0.5
```

For a learned checkpoint, follow all four gates in
`p3_vlm_orchestrator/PERSON4_RUNBOOK.md`; never jump directly to a can pick.

## Preflight checklist

- [ ] The presenter can identify every artifact as synthetic or live.
- [ ] `git status --short` has no unexplained source changes; runtime recordings remain ignored.
- [ ] `python -m p5_rerun_port.record_episode --help` and `python -m p5_rerun_port.query_api_cli --help` open successfully.
- [ ] The timestamped synthetic sequence above has been rehearsed from a fresh dataset name.
- [ ] Rerun opens the saved `.rrd`; both cameras, `follower/position`, and `follower/goal` are visible.
- [ ] Query API schema and goal-versus-position comparison return for the same episode.
- [ ] Export contains seven ordered joints and the `front`, then `side` image contract.
- [ ] Demo screen recording, terminal font size, and backup artifact are ready.
- [ ] Live only: hardware discovery and simultaneous camera checks pass immediately before the demo.
- [ ] Live only: fixed cameras, lighting, taped zone, cables, and arm bases have not moved.
- [ ] Live only: workspace is clear, follower is supported, and one person owns the physical e-stop/power cut.
- [ ] Live only: no other process owns follower or leader serial ports.
- [ ] Live only: one private full-path rehearsal produced a saved `.rrd` and command log.
- [ ] Learned policy only: checkpoint identity, processor files, calibration/profile digest, offline, shadow, and empty-workspace live gates all pass.
- [ ] Off-site copy of the chosen `.rrd`, export, checkpoint (if used), and demo video exists.

## Failure fallback

If live hardware, cameras, Rerun, or policy gating fails, stop motion and show
the last verified synthetic `.rrd` plus the tracked query report. State the
failure plainly. A reproducible dry-run with a clear live gap is stronger than
an unsafe or mislabeled “live” claim.
