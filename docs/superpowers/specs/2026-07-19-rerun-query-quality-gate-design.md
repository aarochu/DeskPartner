# Rerun Query-to-Train Quality Gate Design

**Date:** 2026-07-19

**Prize target:** $2,000 — Best example using the Rerun Query API

**Project:** DeskPartner / reBot can sorting

**Status:** Approved design

## Objective

Build a reproducible robot-data quality gate in which Rerun Query API results
decide which operator-approved demonstrations are ready for training. The
submission must visibly use queries to inspect, align, filter, compare,
transform, evaluate, and prepare real robot data. Its final proof is a
checksummed selection manifest, a held-out evaluation over labeled successes
and failures, and a separately published LeRobot derivative dataset whose
contents can be traced back to exact Hugging Face revisions and Query API
verdicts.

The original Hugging Face repositories remain immutable. Failed attempts are
never training eligible. The system does not claim that joint telemetry alone
can recognize every semantic failure, such as a visually dropped can.

## Source evidence

The run locks these four public repositories to exact commits:

| Role | Hugging Face repository | Commit | Episodes/attempts | Frames |
|---|---|---|---:|---:|
| Successful single-can | `Cornerf/rebot-can-sort-stage1-v1-smoke` | `74d1f300786d58b4f6f55e1798cbb1a1a48f5409` | 52 | 36,729 |
| Successful two-can | `Cornerf/rebot-two-can-recycle-v2-smoke` | `778d0bf5de1096a80b1cf355073e369faa1409da` | 25 | 14,478 |
| Failed single-can | `Cornerf/rebot-can-sort-stage1-v1-failed` | `4952b618a23f8f2e5b09f736cea0a490c62e57b4` | 19 | labeled by attempt metadata |
| Failed two-can | `Cornerf/rebot-two-can-recycle-v2-failed` | `2d9ea53cf8f4835fcfc1656b23d56308696b3e5b` | 6 | labeled by attempt metadata |

The two successful datasets contain 77 LeRobot v3 episodes and 51,207 frames
at 30 FPS. Each sample has seven named action values, seven named observed
joint values, and `front` and `side` video streams. The failure archives contain
25 native `.rrd` recordings with matching metadata and operator outcomes:

- 17 aborted attempts;
- 5 collector errors;
- 3 explicit failures;
- labels including `dropped_object`, `hardware_or_control_problem`, and
  `test_or_setup`;
- 23 complete archives and 2 recoverable incomplete archives.

The merged success repository,
`Cornerf/rebot-cansort-recycle-merged@9446607fe0815f810a0c8ad92a5d4cdf4c32ece3`,
is a useful parity reference but is not an input authority. The two original
success repositories retain clearer per-task episode provenance.

## Current implementation audit

DeskPartner already proves that Rerun 0.34 can open local `.rrd` files through
`rr.server.Server`, obtain a `DatasetEntry`, call `filter_contents`, use
`dataset.reader()`, and materialize a DataFusion-backed dataframe. It also has
a synthetic goal-versus-position comparison and a Viewer Blueprint.

That implementation is not yet a reliable competition-quality curator:

1. `dataset_rrd_paths()` includes every `*.rrd`, including replay recordings.
2. `--episode` and `--tag` limit the printed local catalog but do not limit the
   `.rrd` files queried by `rr.server.Server`.
3. `compare_goal_vs_position()` removes nulls from each stream independently,
   truncates to the shorter array, and then compares by row order rather than
   segment and timestamp.
4. The requested episode name is copied into the result label without proving
   the queried rows came from that episode.
5. LeRobot export consumes local catalog tags, not a Query API selection
   manifest.
6. Checked-in Query API reports are synthetic rather than results over the 102
   real labeled demonstrations.

The new design replaces those weak boundaries rather than layering more report
formatting over them.

## Architecture

### 1. Source lock and inventory

A checked-in challenge configuration names the four repositories, their exact
commits, expected task, expected robot type, seven-joint order, camera keys,
episode counts, and destination derivative repository. The first command
fetches metadata only and refuses to proceed if a remote commit or schema no
longer matches the lock.

The inventory assigns every source item a globally unique identity:

```text
<source-repo>@<commit>:<episode-index-or-attempt-id>
```

Duplicate identities, missing episode indexes, mismatched counts, and unknown
tasks abort the run before any scoring or publication.

### 2. Canonical Rerun evidence

Every episode is represented by one canonical Rerun recording. Raw source data
is read-only.

For successful LeRobot episodes, the importer writes:

- `/follower/goal` — seven-element commanded action vector;
- `/follower/position` — seven-element observed state vector;
- `/camera/cam0` — `front` video stream/frame references;
- `/camera/cam1` — `side` video stream/frame references;
- `/episode/task` and `/episode/source` — static provenance;
- `frame` — integer sample timeline;
- `time` — seconds from episode start.

For native failed recordings, a Query API canonicalizer reads their existing
per-joint action/observation entities and camera streams, aligns them on the
recording timeline, and writes the same canonical schema to a separate cache.
This makes the transformation itself a documented Query API use. Original
failed `.rrd` files remain unchanged and independently verifiable.

Each canonical recording is named by a sanitized source identity. Replay
recordings, files without authoritative episode metadata, and duplicate
segments are excluded at inventory time rather than discovered after metrics
have been mixed.

Operator outcome and failure labels are kept in an evaluation-label table.
They may be displayed after scoring, but the metric scorer cannot read them.

### 3. Query engine and alignment

The query engine opens only the canonical files named by the locked inventory.
It groups rows by `rerun_segment_id` and asserts a one-to-one mapping between
segment and source identity.

Action timestamps define output rows. State and camera observations resolve by
latest-at semantics within the same segment. A resolved state older than 100
milliseconds and a camera reference more than two 30 FPS frame periods away
are missing observations, not silently carried values. Results preserve the
source timestamp, matched timestamp, and age so alignment quality is auditable.

All per-episode metrics accept an aligned table plus the immutable robot
profile. They do not receive source outcome labels.

### 4. Episode metrics

#### Hard integrity metrics

These conditions produce `REJECT`:

- action or state is missing, non-finite, or not exactly seven-dimensional;
- joint names or ordering differ from the authenticated reBot profile;
- timestamps are non-monotonic or frame indexes are non-contiguous;
- duplicate segment/episode identity exists;
- action or state exceeds authenticated profile limits;
- either required camera stream is absent;
- camera/action/state coverage or declared episode length is inconsistent;
- source revision or source checksum differs from the inventory lock.

The two known incomplete failed archives should exercise these gates and remain
useful evaluation evidence.

#### Motion-quality metrics

Episodes that pass integrity receive these metrics:

- per-joint and overall action-versus-state mean absolute error, RMS error, and
  maximum error;
- best action/state lag in frames over a bounded `[-15, +15]` frame search and
  zero-lag versus lag-corrected RMS;
- first-, second-, and third-difference action/state statistics, including
  normalized 95th-percentile jerk and maximum discontinuity;
- episode duration, motion range, stationary fraction, and limit-saturation
  fraction;
- gripper close/open transition count and gripper travel;
- front/side camera coverage, timestamp age, and gap counts.

Metric values are stored with units and per-joint detail. No opaque composite
score is allowed without the component values and reason codes that produced
it.

### 5. Threshold calibration and verdicts

Thresholds are learned without failed labels:

1. Order successful episodes by capture time within each task.
2. Use the first 80 percent of successes per task as calibration data: 41
   single-can episodes and 20 two-can episodes.
3. Reserve the final 20 percent: 11 single-can and 5 two-can episodes.
4. For upper-tail anomaly metrics, use the larger of the calibration 99th
   percentile and `median + 5 * MAD`.
5. For lower-tail metrics, use the smaller of the 1st percentile and
   `median - 5 * MAD`, bounded by physical invariants such as zero.
6. Store every derived threshold, calibration identity, and source commit in a
   threshold snapshot.

Verdicts are deterministic:

- `PASS`: all hard gates pass and no calibrated motion/camera anomaly fires;
- `REVIEW`: hard gates pass but one or more calibrated anomalies fire;
- `REJECT`: any hard gate fails.

For classifier-style evaluation, `REVIEW` and `REJECT` are both predictions of
an unsafe or questionable training take. The original operator outcome is
revealed only after all verdicts have been written and checksummed.

### 6. Evaluation

Evaluation uses the 16 held-out successes plus all 25 failed attempts. It
reports:

- confusion matrix;
- precision, recall, F1, and false-rejection rate;
- per-task breakdown;
- per-outcome and per-failure-label recall;
- exact false-positive and false-negative episode identities;
- comparison distributions for successful versus failed motion metrics;
- coverage showing which failures were structurally detectable and which were
  semantic failures visible only in operator/video evidence.

The primary quality claim is not a predetermined accuracy number. The report
publishes the measured result, including misses. This prevents telemetry-only
heuristics from being represented as semantic grasp recognition.

### 7. Selection manifest and derivative dataset

The final training selection requires both:

1. an authoritative operator-approved success source; and
2. a Query API verdict of `PASS`.

No failed source can enter the manifest, even if its metric verdict happens to
be `PASS`. `REVIEW` episodes are excluded from the first derivative release;
they remain in a ranked review queue for a later human-approved revision.

The selection manifest contains:

- schema version and creation timestamps;
- source repository commits;
- threshold snapshot digest;
- query code commit;
- included and excluded source identities;
- verdict and reason codes for every source item;
- action/state/camera schema;
- episode/frame totals;
- a deterministic `selection_payload_digest` computed over source locks,
  thresholds, verdicts, reasons, and selected identities but excluding
  wall-clock generation metadata;
- SHA-256 digest of each machine-readable artifact.

The curator builds `Cornerf/rebot-cansort-rerun-curated` without changing the
original datasets. Before upload, a fresh LeRobot process must load the whole
derivative and verify episode indexes, frame counts, 30 FPS timestamps,
seven-dimensional finite vectors, task mappings, both video keys, video decode,
and source provenance. After upload, the exact remote revision is fetched and
the same counts and manifest digest are checked again.

## Challenge-word coverage

| Challenge verb | Concrete Query API behavior | Judge evidence |
|---|---|---|
| Inspect | Discover segments, timelines, entities, component columns, dimensions, and cameras | Schema/episode inventory over real recordings |
| Align | Resolve action, state, and camera streams within each segment using timestamp-aware latest-at rules | Good/failed aligned-table drill-down with source and matched timestamps |
| Filter | Select by task, metric verdict, reason code, source revision, and post-score operator label | Ranked review queue and query filters |
| Compare | Compare action/state, zero-lag/lag-corrected traces, success/failure distributions, and single/two-can tasks | Per-joint plots and distribution report |
| Transform | Convert native failed RRD schemas into the canonical schema and aligned metric tables | Source-to-canonical provenance and transformation report |
| Evaluate | Score held-out successes and all failures after thresholds are frozen | Confusion matrix and per-label metrics |
| Prepare | Produce the checksummed selection manifest and validated derivative LeRobot dataset | Hugging Face derivative revision used for training |

## Component boundaries

The implementation uses focused modules:

- source configuration and Hugging Face revision locking;
- LeRobot-success-to-Rerun materialization;
- native-failure Rerun canonicalization through Query API;
- segment-aware Query API extraction and temporal alignment;
- pure metric calculation;
- threshold calibration, verdicts, and held-out evaluation;
- report and manifest generation;
- manifest-driven LeRobot curation and remote verification;
- one challenge CLI that orchestrates the pipeline.

The existing `query_api_cli.py` remains a compatibility and low-level
inspection surface. The competition workflow receives a separate subcommand
CLI so existing dataset inspection commands do not silently change behavior.

## Outputs

Each run writes an immutable run directory keyed by source and code digests:

```text
artifacts/rerun-query/<run-id>/
  source-lock.json
  inventory.parquet
  canonical-rrd/
  aligned-metrics.parquet
  thresholds.json
  verdicts.parquet
  evaluation.json
  selection-manifest.json
  review-queue.csv
  report.html
  report.md
  checksums.json
```

Large `.rrd`, video, and dataset artifacts remain outside normal Git tracking.
Git tracks the configuration, code, tests, a small deterministic example
report, the design/implementation documentation, and the final published Hugging
Face revision.

## Error handling and safety

- Querying and curation never open serial ports or send robot actions.
- Source revision, schema, segment identity, or Query API failures abort the
  run; no sidecar-only fallback may claim Query API success.
- A corrupt individual episode receives an explicit `REJECT` row when its
  identity is still trustworthy. If identity itself is ambiguous, the batch
  aborts.
- Publication is staged. No remote dataset write occurs until local fresh-load
  validation passes.
- Re-running identical sources and code must reproduce the same threshold,
  verdict, and `selection_payload_digest`; full report files may additionally
  record wall-clock generation metadata.
- Existing source repositories and failure archives are never modified or
  deleted.

## Verification strategy

### Unit tests

- segment isolation and episode identity;
- replay-file exclusion;
- episode/tag filtering that changes queried inputs, not only printed metadata;
- latest-at alignment with missing and delayed rows;
- timestamp gaps, duplicate frames, NaN, infinity, wrong dimensions, and
  profile-limit violations;
- deterministic metric values, thresholds, verdicts, and reason ordering;
- label-blind scoring API;
- manifest prohibition on failed or review sources.

### Integration tests

- tiny two-episode LeRobot fixture to canonical RRD;
- native per-joint failed-RRD fixture to canonical RRD through Query API;
- real Rerun 0.34 `Server` plus `dataset.reader()` extraction;
- front/side camera and seven-joint contract;
- mixed dataset proving no replay or cross-segment contamination;
- deterministic HTML, Markdown, JSON, Parquet, and checksum artifacts;
- manifest-driven derivative fresh load.

### Real-data acceptance

- lock and inventory all four public source commits;
- account for all 77 successes and 25 failures exactly once;
- materialize/query every eligible real recording;
- freeze thresholds from the 61 calibration successes;
- evaluate the 16 held-out successes and 25 failures;
- spot-check the highest- and lowest-ranked real episodes in Rerun Viewer;
- build the derivative dataset from `PASS` successes only;
- fresh-load and decode both cameras;
- upload only after explicit user confirmation, then refetch and verify remote
  parity.

## Judge demonstration

The 90-second demonstration uses one rehearsed command and one precomputed
backup report:

1. **0–15 seconds:** show the four pinned Hugging Face sources and the inventory
   of 102 real labeled demonstrations.
2. **15–35 seconds:** query one successful and one failed recording, displaying
   segment-safe action/state/camera alignment.
3. **35–55 seconds:** filter and compare their metrics and explain deterministic
   reason codes.
4. **55–75 seconds:** reveal operator labels and show held-out precision,
   recall, false rejections, and per-label misses.
5. **75–90 seconds:** open the checksummed selection manifest and the exact
   `Cornerf/rebot-cansort-rerun-curated` revision prepared for training.

The presenter says: “Rerun is not just showing our robot. Its Query API decides
which demonstrations are structurally sound, measures what it can detect,
shows what it cannot, and produces the exact dataset our policy trains on.”

If live querying fails, the presenter uses the last verified report and exact
source/run digests without claiming a new run succeeded.

## Acceptance criteria

The design is complete when all of the following are true:

1. All 102 source items have unique, revision-locked identities.
2. Every reported metric comes from Rerun Query API rows over a named segment.
3. No replay recording or unrelated segment can affect an episode result.
4. The held-out evaluation is label-blind until verdicts are finalized.
5. Every verdict has deterministic reason codes and inspectable component
   metrics.
6. The selection manifest contains no failure or review item.
7. The derivative dataset fresh-loads with the required robot, joint, task,
   timestamp, and video contracts.
8. The published remote revision matches the local manifest and counts.
9. A judge can reproduce the report from locked source revisions with one
   documented command.
10. Documentation and the pitch distinguish data-quality detection from
    semantic task-success recognition.
