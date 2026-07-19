# Rerun Query API quality gate

This is the competition workflow for the **$2,000 Best example using the
Rerun Query API** prize. It is a post-recording, read-only pipeline: it never
opens a serial port or sends a robot action.

The quality gate uses Rerun Query API results to decide which
operator-approved demonstrations are structurally ready for training. It does
not claim that joint telemetry can recognize every semantic failure. A dropped
can may look mechanically ordinary and remain detectable only from the
operator label or video.

## Locked real evidence

| Role | Hugging Face source | Immutable revision | Items |
|---|---|---|---:|
| Successful single-can | `Cornerf/rebot-can-sort-stage1-v1-smoke` | `74d1f300786d58b4f6f55e1798cbb1a1a48f5409` | 52 |
| Successful two-can | `Cornerf/rebot-two-can-recycle-v2-smoke` | `778d0bf5de1096a80b1cf355073e369faa1409da` | 25 |
| Failed single-can | `Cornerf/rebot-can-sort-stage1-v1-failed` | `4952b618a23f8f2e5b09f736cea0a490c62e57b4` | 19 |
| Failed two-can | `Cornerf/rebot-two-can-recycle-v2-failed` | `2d9ea53cf8f4835fcfc1656b23d56308696b3e5b` | 6 |

The locked inventory is 102 unique source identities: 77 operator-approved
successes (51,207 frames) and 25 failed attempts. Every success has seven
ordered action values, seven ordered observed-joint values, and `front` and
`side` cameras at 30 FPS. The source repositories remain immutable.

## What the Query API does

| Challenge verb | Evidence produced |
|---|---|
| **Inspect** | Discover recording segments, timelines, entities, dimensions, joints, and cameras. |
| **Align** | Resolve state and camera observations latest-at each action timestamp, within the same segment and bounded age. |
| **Filter** | Select by task, source revision, verdict, reason code, and post-score operator label. |
| **Compare** | Compare action/state error, zero-lag and lag-corrected traces, tasks, and success/failure distributions. |
| **Transform** | Canonicalize native failure RRDs through Query API rows into the same schema as successful episodes. |
| **Evaluate** | Freeze thresholds and verdicts before revealing labels, then score 16 held-out successes and all 25 failures. |
| **Prepare** | Write a checksummed selection manifest and locally fresh-load a derivative LeRobot dataset. |

Rerun is therefore not only a viewer. `rr.server.Server`, segment-filtered
datasets, `dataset.reader()`, and DataFusion-backed tables are the authority for
the metrics and verdicts.

## Reproduce the local workflow

From the repository root, install the pinned environment and run:

```bash
./rebot_setup/setup.sh
uv pip install --python rebot_setup/vendor/rebot_lerobot/.venv/bin/python -r requirements.txt

./rebot_setup/vendor/rebot_lerobot/.venv/bin/python \
  -m p5_rerun_port.query_challenge_cli run \
  --config config/rerun_query_challenge.yaml \
  --artifacts-root artifacts/rerun-query
```

The workflow stages are also independently addressable:

```text
inventory → materialize → audit → evaluate → prepare
```

`prepare` is local-only. The CLI intentionally has no upload or publish flag.
Publishing `Cornerf/rebot-cansort-rerun-curated` is a separate, explicitly
approved operation after local validation.

A completed run prints machine-readable final lines:

```text
RUN_ID=<digest>
REPORT_HTML=<absolute path>
SELECTION_MANIFEST=<absolute path>
DERIVATIVE_ROOT=<absolute path>
```

Do not call a new run successful unless all four lines are present and the
referenced artifacts pass their checksums and fresh-load validation.

## Immutable run output

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

The run must account for all 102 identities exactly once. Thresholds use the
first 80% of successes by capture time per task (41 single-can and 20 two-can)
without failure labels. The final evaluation contains 11 + 5 held-out
successes and all 25 failures. `REVIEW` and `REJECT` both predict questionable
training data.

The derivative selection requires both an operator-approved success source and
a Query verdict of `PASS`. Failures and `REVIEW` episodes are never included.
The derivative must fresh-load in a separate process with contiguous episode
indexes, finite `(7,)` action/state vectors, 30 FPS timestamps, both decoded
cameras, task mapping, source provenance, and the manifest digest. No upload is
authorized by this command.

## Determinism and reuse

Each stage may reuse prior output only when its input payload digest matches.
The selection digest excludes wall-clock report metadata and covers source
locks, thresholds, verdicts, reason codes, and selected identities. Identical
sources and code must reproduce the same threshold, verdict, and selection
payload digests.

## Smoke test: legacy synthetic comparison

The older single-recording command remains useful for a fast local smoke test,
but it is not the 102-item competition evaluation and must not be presented as
real-data evidence:

```bash
python -m p5_rerun_port.record_episode \
  --fake --dataset query-smoke --task "Synthetic smoke test" \
  --tag "Good episode" --seconds 5 --no-viewer

python -m p5_rerun_port.query_api_cli \
  --dataset query-smoke --compare goal-vs-position \
  --report docs/p5_rerun_port/examples/hackathon_smoke_query_report.md
```

That compatibility CLI inspects one local catalog. It does not provide source
revision locking, held-out evaluation, a selection manifest, or derivative
fresh-load validation.

## References

- [Approved design](../superpowers/specs/2026-07-19-rerun-query-quality-gate-design.md)
- [Rerun dataframe queries](https://rerun.io/docs/concepts/query-and-transform/dataframe-queries)
- [Get data out of Rerun](https://rerun.io/docs/howto/query-and-transform/get-data-out)
