# ReBot Data Conversion and Sharing Instructions

## Scope

These instructions apply to every file and subdirectory under
`rebot_operator_kit/`. They govern recording discovery, Rerun recovery,
LeRobot conversion, dataset validation, naming, timestamps, packaging, and
team sharing.

The priority is a trustworthy, immediately usable dataset with minimal
rework. Never trade provenance, episode quality, or recoverability for a fast
upload.

## Read this before changing anything

1. Read this file completely.
2. Read `README.md`, `TRAINING_GUIDE.md`, and the relevant current code before
   acting. Do not rely only on old chat summaries.
3. Run `git status --short --branch`. Other agents may be repairing the live
   recorder. Never overwrite or stage their unrelated changes.
4. Inspect the live source and destination paths. Do not assume that the newest
   chat message identifies the newest dataset.
5. Determine the actual current time from the machine using both commands:

   ```bash
   date -u '+%Y-%m-%dT%H:%M:%SZ'
   TZ=America/Los_Angeles date '+%Y-%m-%dT%H:%M:%S%z %Z'
   ```

6. Record the exact source paths, source timestamps, destination, intended
   visibility, and validation criteria before converting or uploading.

## Confirmed project context

- Source repository: `https://github.com/aarochu/DeskPartner`.
- Tracked operator source: `rebot_operator_kit/`.
- Local runtime data normally lives outside Git tracking under the Operator
  Kit's `data/`, `training-runs/`, `models/`, `.state/`, and camera-output
  directories.
- Stage 1 task: pick one can and place it in the taped sorting zone.
- Collect a small smoke dataset early, validate the full conversion/training
  path, then grow toward 150 clean single-can episodes.
- One can is picked per episode. Move its starting position around the full
  reachable workspace. Extra visible cans are introduced only after the first
  150 clean single-can episodes.
- Camera contract:
  - `observation.images.front`: camera 0, Logitech overhead, 640x480.
  - `observation.images.side`: camera 1, Innomaker wrist/claw, 1280x720.
  - camera 3 is not a training camera and must remain excluded.
- Dataset rate: 30 FPS.
- Robot/action contract: seven follower-space position values in this order:
  `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_yaw`,
  `wrist_roll`, `gripper`.
- The exact sent follower-space action is the learning target. Do not substitute
  raw leader commands.
- LeRobot owns `timestamp`, `frame_index`, `episode_index`, `index`, and
  `task_index`. Conversion code must not invent conflicting values.

These are confirmed defaults, not permission to ignore live metadata. If a
new reviewed profile changes the contract, start a new dataset version and
document the change.

## Source-of-truth hierarchy

Use evidence in this order:

1. A freshly loadable LeRobot v3 dataset plus its `meta/info.json`, episode
   Parquet, task Parquet, statistics, videos, and
   `meta/rebot_training_profile.json`.
2. Per-attempt `metadata.json` beside `attempt.rrd`, `overhead.mp4`, and
   `wrist.mp4`.
3. Run manifest and collection-contract digest.
4. Rerun timelines and logged entities inside `attempt.rrd`.
5. Filesystem timestamps.
6. Chat descriptions or filenames without embedded timestamps.

Never call a folder a valid dataset merely because it exists. Zero-frame
folders, partial writers, unfinished video encodes, and stale smoke outputs are
not valid datasets.

## Date and time rules

Time mistakes make incremental sharing unsafe. Apply all of these rules:

- Canonical machine-readable timestamps are UTC RFC 3339 with a trailing `Z`,
  for example `2026-07-18T23:11:07.524Z`.
- Also store a human-facing local timestamp in the IANA timezone
  `America/Los_Angeles`, including its numeric UTC offset. Never hard-code
  `PST` or `PDT`; daylight-saving rules must come from the timezone database.
- Prefer `metadata.json.started_at` and `finished_at`. Attempt IDs such as
  `20260718T231107.524013Z-dcabef6810` also encode UTC and may be used after
  parsing and validation.
- Do not infer capture order from chat order, directory listing order, or a
  file's modification time when authoritative metadata exists.
- Do not use ambiguous local-only names such as `7-18-final`, `latest`, or
  `new-data`.
- Every export manifest must include at least:
  - `captured_at_utc` or an episode-level UTC range;
  - `captured_at_local` with offset;
  - `timezone: America/Los_Angeles`;
  - `exported_at_utc`;
  - `exported_at_local` with offset;
  - source attempt IDs;
  - exporter version or Git commit;
  - source and output checksums.
- Determine "new since last share" from committed attempt IDs/checksums in the
  previous share manifest, not from `mtime` alone.

## Naming convention

Use lowercase ASCII slugs with hyphens. A dataset repository ID is
`<owner-or-org>/<dataset-slug>`.

Preferred stage-1 progression:

- `rebot-can-sort-stage1-v1-smoke` is the local, append-only collection stream:
  validate and start a throwaway train at 10 kept episodes, then resume the
  unchanged contract through the 50/100/150 checkpoints.
- `rebot-can-sort-stage1-v1` is the separate reviewed kept-only sharing/export
  namespace. Build it without changing the raw local collection, and publish
  immutable revisions for the approved 10/50/100/150 checkpoints.
- Increment the final version only when the semantic contract changes, such as
  camera identity/pose, resolution, task wording, robot calibration/profile,
  joint order, action semantics, or FPS.

Do not create `final-final`, `latest2`, or date-only dataset names. Dates belong
in manifests and release tags, not in the semantic dataset identity.

## Episode quality gate

The LeRobot training dataset is success-only.

- Only an explicit operator disposition of `kept` may enter training.
- Jerky-but-successful, aborted, failed, interrupted, `collector_error`, and
  uncertain-save attempts remain in the raw attempt archive and are excluded
  from training.
- Never silently relabel an aborted or collector-error attempt as a successful
  demonstration.
- A recovered attempt must be labeled `recovered_unreviewed` until a human
  reviews both camera views and motion quality.
- Preserve failure labels and notes. Raw failed attempts are useful for audit,
  debugging, and possible future negative-data work, but not for the current
  success-only policy.
- If the data-versus-metadata state is uncertain, fail closed and block upload
  to the training namespace.

Historical local can-recycling attempts included an aborted recording and
collector-error recordings. Their existence proves that frames may be
recoverable; it does not make them clean training episodes.

## Conversion contract

The required LeRobot v3 layout is:

```text
<dataset-root>/
├── data/chunk-000/file-000.parquet
├── meta/info.json
├── meta/stats.json
├── meta/tasks.parquet
├── meta/episodes/chunk-000/file-000.parquet
├── meta/rebot_training_profile.json
├── videos/observation.images.front/chunk-000/file-000.mp4
└── videos/observation.images.side/chunk-000/file-000.mp4
```

Additional provenance files may be added without replacing LeRobot metadata,
for example `SHARE_MANIFEST.json` and `DATA_QUALITY.md`.

For Rerun recovery:

1. Verify the `.rrd` using the pinned Rerun CLI before extraction.
2. Query the `attempt_frame` timeline. It must be contiguous from zero.
3. Extract action values from `/action/<joint>/pos` in the locked seven-joint
   order.
4. Extract observation position values from
   `/observation/<joint>/pos` in the same order.
5. Verify the Rerun sample count equals `metadata.json.samples`.
6. Verify each camera artifact reports exactly the same frame count and FPS.
7. Preserve the original `.rrd`, videos, and metadata unchanged.
8. Write a separate output root. Never convert in place.

## Required validation before sharing

An export is not complete until all checks pass:

1. Load it in a fresh process with the repository's pinned LeRobot runtime.
2. Confirm `codebase_version` is v3-compatible.
3. Confirm total episodes and frames match the sum of episode metadata.
4. Confirm every episode has contiguous frame indices and timestamps equal to
   `frame_index / 30` within floating-point tolerance.
5. Confirm state and action are finite float32 vectors of shape `[7]` in the
   locked joint order.
6. Confirm both video keys exist, decode, match the declared dimensions and
   FPS, and have the exact episode frame count.
7. Confirm the task text and task indices resolve correctly.
8. Confirm collection-profile and collection-contract digests match.
9. Run `validate_dataset.py` and the relevant repository tests.
10. Generate SHA-256 checksums for every shared artifact and verify them after
    packaging or upload.
11. Spot-check both camera views and joint traces for every smoke episode; for
    larger sets, check all suspect episodes plus a documented sample at least
    every 25 episodes.

Report exact failed checks. Never say "validated" when only file existence or
video presence was checked.

## Efficient team sharing

Keep source code and bulk data responsibilities separate:

- GitHub `aarochu/DeskPartner` is the source, documentation, validation, and
  automation repository.
- Bulk LeRobot Parquet/video datasets should normally live in a versioned
  dataset store such as a private Hugging Face dataset repository. If the team
  explicitly requires GitHub, configure Git LFS before adding videos, Rerun
  files, archives, or other large binaries. GitHub rejects ordinary files over
  100 MB.
- Never commit runtime caches, calibration secrets, `.state`, temporary PNG
  frame directories, incomplete encodes, models, or unreviewed raw attempts to
  the normal source tree.
- A teammate-facing share must include a dataset card, exact repo/revision,
  episode/frame counts, task, camera contract, robot/action contract, quality
  policy, capture/export timestamp ranges, checksums, validation command, and
  known limitations.
- Prefer incremental append/upload based on attempt IDs and checksums. Do not
  re-upload unchanged artifacts or rewrite history just to call a dataset
  "latest".
- Publish immutable revisions or tags. A moving default branch may point to the
  newest approved revision, but every training run must record the exact commit
  or dataset revision used.

## Safe update workflow

For every new recording batch:

1. Inventory raw attempts and compare IDs/checksums with the last
   `SHARE_MANIFEST.json`.
2. Filter to explicit `kept` attempts; quarantine all uncertain records.
3. Append with the existing LeRobot semantic contract. If the contract differs,
   create a new dataset version instead of mixing episodes.
4. Finalize and reopen after every saved episode. Pressing a global Stop button
   must not be required to make earlier kept episodes durable.
5. Validate the complete destination, not only the newly added files.
6. Create/update the manifest using the current UTC and local time commands.
7. Review the exact Git/Git-LFS or dataset-hub diff before upload.
8. Push only intended files. Preserve other agents' changes.
9. Fetch the remote revision or manifest after upload and recheck counts and
   checksums.
10. Report the share URL, immutable revision, counts, time range, validation
    result, exclusions, and next action.

## Current-work and concurrency rule

The recorder durability/save flow may be under active repair. Before building
or publishing a dataset, inspect current code, current tests, live manifests,
and `git status`. Do not base a new exporter on a stale copy from another task.
Do not delete or replace partial recovery artifacts created by another agent;
move your own incomplete output to an explicitly named scratch/quarantine path
and document it.

## Hard prohibitions

- Do not modify or delete raw recordings during conversion.
- Do not include an episode in training based only on apparent visual success.
- Do not merge datasets with different semantic contracts.
- Do not guess dates, timezones, frame counts, camera identities, or joint order.
- Do not use modification time as the only incremental-upload boundary.
- Do not publish large data through ordinary Git without checking Git LFS and
  remote limits.
- Do not push secrets, tokens, calibration-private material, caches, or local
  machine state.
- Do not call a partial, zero-frame, unreviewed, or fresh-load-failing export
  complete.
