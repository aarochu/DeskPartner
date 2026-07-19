# Hackathon submission and judge demo

## 60-second pitch

> Rerun is not just showing our robot. Its Query API inspects and aligns 102
> revision-locked reBot demonstrations, compares commanded and observed
> motion, transforms native failure recordings, and issues deterministic
> `PASS`, `REVIEW`, or `REJECT` verdicts. We freeze thresholds using only the
> first 80 percent of operator-approved successes, then reveal labels for 16
> held-out successes and all 25 failures. Only operator-approved successes
> with a Query `PASS` enter a checksummed LeRobot derivative that is fresh-load
> validated before publication. We report misses honestly: telemetry is a
> data-quality detector, not proof that the robot semantically completed a
> grasp.

## Evidence on screen

The four sources are pinned to exact Hugging Face revisions:

| Source | SHA | Count |
|---|---|---:|
| `Cornerf/rebot-can-sort-stage1-v1-smoke` | `74d1f300786d58b4f6f55e1798cbb1a1a48f5409` | 52 successes |
| `Cornerf/rebot-two-can-recycle-v2-smoke` | `778d0bf5de1096a80b1cf355073e369faa1409da` | 25 successes |
| `Cornerf/rebot-can-sort-stage1-v1-failed` | `4952b618a23f8f2e5b09f736cea0a490c62e57b4` | 19 failures |
| `Cornerf/rebot-two-can-recycle-v2-failed` | `2d9ea53cf8f4835fcfc1656b23d56308696b3e5b` | 6 failures |

Total: 102 unique items, comprising 77 successes and 25 failures. Show the
measured report, not an expected accuracy. Name exact false positives and false
negatives and retain per-label misses.

## Rehearsed command

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python \
  -m p5_rerun_port.query_challenge_cli run \
  --config config/rerun_query_challenge.yaml \
  --artifacts-root artifacts/rerun-query
```

A verified run ends with:

```text
RUN_ID=<digest>
REPORT_HTML=<absolute path>
SELECTION_MANIFEST=<absolute path>
DERIVATIVE_ROOT=<absolute path>
```

If those lines or checksum/fresh-load validation are absent, use the last
verified report and do not claim a new run succeeded. The command is local-only
and cannot upload.

## 90-second judge script

### 0–15 seconds — locked inventory

Show `source-lock.json` and the four SHAs. Say: “These are 102 real labeled
demonstrations: 77 successes and 25 failures. The original repositories are
read-only.”

### 15–35 seconds — Inspect, Align, Transform

Open the report drill-down for one successful and one failed episode. Point to
the segment identity, action timestamp, matched state/camera timestamps, and
ages. Explain that native failure RRDs were transformed through Query API rows
into the same seven-joint, two-camera canonical schema.

### 35–55 seconds — Filter and Compare

Filter the review queue by task and reason. Compare per-joint action/state
error, zero-lag versus best-lag RMS, discontinuity, and camera coverage. Point
to deterministic reason codes rather than an opaque composite score.

### 55–75 seconds — Evaluate

Show the frozen 41 + 20 calibration split, then the label-blind evaluation of
11 + 5 held-out successes and 25 failures. Read the measured confusion matrix,
precision, recall, F1, false-rejection rate, and at least one miss. Say:
“`REVIEW` and `REJECT` predict questionable training data; labels were revealed
only after verdict digests were frozen.”

### 75–90 seconds — Prepare

Open `selection-manifest.json`, its `selection_payload_digest`, and the fresh
load validation. Show that every selected item is both an approved success and
`PASS`, while every failure and `REVIEW` item is excluded. Name the intended
destination `Cornerf/rebot-cansort-rerun-curated`; do not claim it was published
until an exact remote revision has been refetched and verified.

Close with: “Rerun decides which demonstrations are structurally sound,
measures what it can detect, shows what it cannot, and prepares the exact data
our policy trains on.”

## Challenge coverage checklist

- [ ] **Inspect:** schema, segments, timelines, joints, and cameras are visible.
- [ ] **Align:** source/matched timestamps and bounded ages are visible.
- [ ] **Filter:** task, verdict, reason, revision, and label filters work.
- [ ] **Compare:** traces, lag, tasks, and success/failure distributions appear.
- [ ] **Transform:** a native failed RRD has canonical provenance.
- [ ] **Evaluate:** measured held-out results and exact misses are shown.
- [ ] **Prepare:** checksummed manifest and derivative validation are shown.

## Truthful evidence boundary

- A `PASS` means the episode passed authenticated structural, telemetry,
  timing, camera, and calibrated anomaly checks.
- A `PASS` does not prove the can was grasped or placed correctly. Dropped
  objects and other semantic failures may require operator/video evidence.
- `REVIEW` is excluded from the first derivative; it is not silently treated as
  a success.
- Failed-source data is evaluation evidence only and can never enter training.
- Do not quote a target accuracy. Quote the measured report, including misses.
- Publication is staged and requires explicit approval after local validation.

## Backup and smoke test

Keep the last verified `report.html`, `report.md`, `selection-manifest.json`,
`checksums.json`, and source/run digests ready. If live querying fails, show
that backup without implying a new execution.

The legacy synthetic command is only a fast compatibility smoke test:

```bash
python -m p5_rerun_port.query_api_cli \
  --dataset query-smoke --compare goal-vs-position
```

Never substitute that single synthetic comparison for the real 102-item,
revision-locked, label-blind evaluation.
