# Task 6 report — aligned Query rows and explainable metrics

## Status

Implemented segment-bound Rerun 0.34 Query extraction, integer-nanosecond
latest-at alignment, authenticated robot/profile constants, and label-blind
episode metrics. No source archive, sidecar label, video upload, hardware, or
serial path is used.

## RED

Command written and run before the implementation modules existed:

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q \
  p5_rerun_port/tests/challenge/test_alignment.py \
  p5_rerun_port/tests/challenge/test_metrics.py
```

Observed: collection stopped with two `ModuleNotFoundError` errors for
`p5_rerun_port.challenge.alignment`.

## GREEN

Focused Task 6 command:

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q \
  p5_rerun_port/tests/challenge/test_alignment.py \
  p5_rerun_port/tests/challenge/test_metrics.py
```

Result: `12 passed`.

All challenge regressions:

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q \
  p5_rerun_port/tests/challenge
```

Result: `122 passed`.

All p5 regressions:

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q p5_rerun_port/tests
```

Result: `129 passed`.

Additional checks: `compileall` passed for `p5_rerun_port/challenge`, and
`git diff --check` is clean.

## Query and alignment evidence

- Requires exactly one `dataset.segment_ids()` value and applies
  `filter_segments([segment_id])` before every schema/read boundary.
- Resolves all component columns with `schema.column_names_for`; scalar vectors
  use singular `rerun.components.Scalar`.
- Reads source, task, and joint provenance through `reader(index=None)` and
  proves them against the locked inventory/config contract.
- Reads action, state, front MediaType, and side MediaType separately with
  `reader(index="time", fill_latest_at=False)` and explicit DataFusion sorting.
- Resolves but never projects either camera Blob column. The real Rerun 0.34
  integration test audits the actual projection and sees MediaType with no
  blob.
- Converts Arrow `duration[ns]` cells to integer nanoseconds only after sorting.
- Uses action rows as anchors and `searchsorted(..., side="right") - 1` for
  same-segment state/camera latest-at matching. Equality passes at
  `100,000,000 ns` state age and `66,666,667 ns` camera age; future-only and
  one-nanosecond-stale observations do not match.
- Retains segment, identity, task, locked joints, expected count, frames,
  action time/presence, matched state/camera time, age, presence, and invalid
  vector diagnostics.

## Profile and metric evidence

- Authenticates `rebot_operator_kit/config/training_profile.json` with file SHA
  `82363ef023de870db5b22173837fc95fef2f8adb426bb1d65ed46eb96ace31b6`
  and canonical semantic SHA
  `9370a9fc5d90df7ffb0174c0a035fcf043911c04bbef6820c9bf36ab04131208`.
- Derives the exact seven action limits/features from that authenticated
  profile. State bounds alone receive the separate inclusive `0.25 deg`
  tolerance.
- The cache-optional pinned Parquet regression ran locally over 77 episodes and
  51,207 frames: zero action-limit rejects, 54 episode false rejects with exact
  state limits, and zero state rejects with the authenticated tolerance.
- Computes 65 scalar/status metrics and 33 per-joint vectors: tracking error,
  common-window normalized lag, raw/normalized first through third
  differences, discontinuity, duration/ranges, stationary/saturation,
  gripper hysteresis/transitions/travel, and state/camera coverage/age/gaps.
- Lag `+k` compares `action[t]` with `state[t+k]`, searches `[-15,+15]`, uses
  one common anchor window, normalizes by `[290,170,200,170,180,180,270]`,
  rejects motion range below `0.01`, and breaks rounded score ties by
  `(rms, abs(k), k)`.
- Every scalar/status and per-joint metric has an explicit unit entry. Hard
  reasons are emitted in the fixed probe order; motion metrics remain soft.
- `AlignedEpisode` and `EpisodeMetrics` expose no role, disposition,
  failure-label, or training-eligibility fields.
- A rejected `CanonicalArtifact` returns one zero-sample metric row with its
  existing reasons and never opens an RRD.

## Self-review and remaining concern

- Confirmed no label/sidecar fields, source writes, uploads, hardware, or serial
  references exist in the Task 6 implementation.
- Confirmed configured magic constants are authenticated/validated rather than
  repeated inside metric logic.
- Confirmed discontinuity, lag, stationary, saturation, gripper, and derivative
  observations do not become hard reasons before Task 7 calibration.
- The metadata-only pinned scan cannot prove decoded camera coverage for all 77
  successes. Full camera acceptance remains dependent on Task 4 materialization
  of each canonical episode, exactly as recorded in the feasibility probe.

## Review fixes

The Task 6 review identified an edge-window comparability defect and missing
coverage/count outputs. The corrected lag implementation now computes exposed
`zero_lag_mae_deg`, `zero_lag_rms_deg`, and normalized zero-lag RMS on the
same fixed action anchors used by every lag candidate. The selected lag still
uses the authenticated `+k` convention, span-normalized RMS, and deterministic
rounded-score tie order. A regression places large errors only in the 15-frame
edges and proves they neither inflate the common-window zero score nor create a
false lag improvement.

State alignment now exports present count, missing count, coverage fraction,
and missing-run gap count in action-anchor order. Both accepted and rejected
rows export `sample_count` with unit `count`; the complete metric schema test
proves every scalar/status and per-joint key has a unit.

Direct duplicate-source regressions prove repeated state timestamps reject as
`STATE_MISSING_OR_STALE` and repeated camera timestamps reject as
`CAMERA_FRONT_MISSING_OR_STALE` before `searchsorted` can depend on ambiguous
row order.

Review-fix verification:

```text
focused alignment + metrics: 16 passed
all challenge tests:          126 passed
all p5 tests:                 133 passed
compileall:                   passed
git diff --check:             clean
```
