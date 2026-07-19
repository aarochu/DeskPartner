# Rerun Query-to-Train Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a Rerun Query API quality gate that evaluates 77 successful and 25 failed real reBot demonstrations, produces auditable pass/review/reject verdicts, and prepares a separately validated LeRobot derivative dataset containing Query-approved successes only.

**Architecture:** Lock four Hugging Face source revisions, materialize one canonical Rerun recording per source episode, and query each segment through `rr.server.Server` plus `dataset.reader()`. Pure metric and quality modules consume timestamp-aligned Query API rows; downstream artifact and curation modules consume only checksummed verdict manifests. Existing sources stay read-only, failure labels remain unavailable to scoring, and publication occurs only after a local fresh-load gate and explicit user confirmation.

**Tech Stack:** Python 3.11, Rerun SDK 0.34 with DataFusion, LeRobot 0.4.4, Hugging Face Hub, PyArrow, Pandas, NumPy, OpenCV/PyAV, PyYAML, pytest/unittest.

## Global Constraints

- Execute in a dedicated worktree on branch `codex/rerun-query-quality-gate`; do not edit the dirty `main` checkout.
- Read `docs/superpowers/specs/2026-07-19-rerun-query-quality-gate-design.md` before every task.
- Pin the four source commits exactly as written in the design; a changed remote SHA aborts the run.
- Preserve the seven-joint order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_yaw`, `wrist_roll`, `gripper`.
- Preserve camera keys `front` then `side`, mapped to `/camera/cam0` and `/camera/cam1`.
- Rerun Query API data is authoritative for metrics. Sidecars may supply identity and post-score labels but may not substitute for query rows.
- Score functions cannot accept disposition, failure label, or training eligibility as input.
- Failed sources and `REVIEW` sources are never included in the first derivative dataset.
- No task opens serial ports or sends robot actions.
- Do not install `rerun-lerobot`; it requires Python 3.12/LeRobot 0.6 and converts in the opposite direction.
- Large downloads, canonical `.rrd`s, videos, reports, and derivative datasets stay ignored under `artifacts/`, `recordings/`, or `datasets/`.
- Before any Hugging Face upload, show the user the repository ID, source counts, selected counts, manifest digest, and exact files, then obtain explicit confirmation.

## File Map

| Path | Responsibility |
|---|---|
| `config/rerun_query_challenge.yaml` | Immutable challenge source and quality contract |
| `p5_rerun_port/challenge/models.py` | Shared frozen data contracts |
| `p5_rerun_port/challenge/config.py` | YAML loading and validation |
| `p5_rerun_port/challenge/hub.py` | Hugging Face read/download adapter |
| `p5_rerun_port/challenge/inventory.py` | Revision-locked source inventory |
| `p5_rerun_port/challenge/canonical.py` | Canonical RRD writer and LeRobot success materializer |
| `p5_rerun_port/challenge/failure_canonical.py` | Native failed-RRD Query API transformation |
| `p5_rerun_port/challenge/alignment.py` | Segment-safe Query API extraction and latest-at alignment |
| `p5_rerun_port/challenge/metrics.py` | Pure integrity and motion metric calculation |
| `p5_rerun_port/challenge/quality.py` | Threshold calibration and verdict generation |
| `p5_rerun_port/challenge/evaluate.py` | Label reveal and held-out evaluation |
| `p5_rerun_port/challenge/artifacts.py` | Deterministic Parquet/JSON/checksum artifacts |
| `p5_rerun_port/challenge/report.py` | Self-contained HTML/Markdown judge report |
| `p5_rerun_port/challenge/curate.py` | Manifest-driven derivative LeRobot builder/validator |
| `p5_rerun_port/query_challenge_cli.py` | One competition workflow CLI |
| `p5_rerun_port/tests/challenge/` | Focused unit and integration tests |

---

### Task 1: Freeze the source and quality contract

**Files:**
- Create: `config/rerun_query_challenge.yaml`
- Create: `p5_rerun_port/challenge/__init__.py`
- Create: `p5_rerun_port/challenge/models.py`
- Create: `p5_rerun_port/challenge/config.py`
- Modify: `requirements.txt:7-14`
- Test: `p5_rerun_port/tests/challenge/test_config.py`

**Interfaces:**
- Consumes: checked-in YAML.
- Produces: `ChallengeConfig.load(path: Path) -> ChallengeConfig`, `SourceSpec`, `EpisodeIdentity`, `InventoryRow`, and shared enums.

- [ ] **Step 1: Write the failing configuration tests**

```python
from pathlib import Path

import pytest

from p5_rerun_port.challenge.config import ChallengeConfig, ConfigError


CONFIG = Path("config/rerun_query_challenge.yaml")


def test_checked_in_config_locks_real_sources_and_robot_contract() -> None:
    config = ChallengeConfig.load(CONFIG)
    assert [(s.repo_id, s.revision, s.role, s.expected_items) for s in config.sources] == [
        ("Cornerf/rebot-can-sort-stage1-v1-smoke", "74d1f300786d58b4f6f55e1798cbb1a1a48f5409", "success", 52),
        ("Cornerf/rebot-two-can-recycle-v2-smoke", "778d0bf5de1096a80b1cf355073e369faa1409da", "success", 25),
        ("Cornerf/rebot-can-sort-stage1-v1-failed", "4952b618a23f8f2e5b09f736cea0a490c62e57b4", "failure", 19),
        ("Cornerf/rebot-two-can-recycle-v2-failed", "2d9ea53cf8f4835fcfc1656b23d56308696b3e5b", "failure", 6),
    ]
    assert config.joint_names == (
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
        "wrist_yaw", "wrist_roll", "gripper",
    )
    assert config.camera_keys == ("front", "side")
    assert config.destination_repo == "Cornerf/rebot-cansort-rerun-curated"


def test_config_rejects_duplicate_repo_or_non_sha_revision(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nsources: []\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        ChallengeConfig.load(path)
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

Run: `./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q p5_rerun_port/tests/challenge/test_config.py`

Expected: FAIL with `ModuleNotFoundError: p5_rerun_port.challenge`.

- [ ] **Step 3: Add explicit dependencies**

Add these requirements without changing the Rerun floor:

```text
rerun-sdk[datafusion]>=0.34,<0.35
huggingface-hub>=0.34,<1.0
pyarrow>=14
```

Keep `pandas>=2.0`, NumPy, OpenCV, PyYAML, and `rerun-loader-urdf`.

- [ ] **Step 4: Write the exact YAML contract**

```yaml
schema_version: 1
sources:
  - repo_id: Cornerf/rebot-can-sort-stage1-v1-smoke
    revision: 74d1f300786d58b4f6f55e1798cbb1a1a48f5409
    role: success
    task_key: single_can
    expected_items: 52
    expected_frames: 36729
  - repo_id: Cornerf/rebot-two-can-recycle-v2-smoke
    revision: 778d0bf5de1096a80b1cf355073e369faa1409da
    role: success
    task_key: two_can
    expected_items: 25
    expected_frames: 14478
  - repo_id: Cornerf/rebot-can-sort-stage1-v1-failed
    revision: 4952b618a23f8f2e5b09f736cea0a490c62e57b4
    role: failure
    task_key: single_can
    expected_items: 19
  - repo_id: Cornerf/rebot-two-can-recycle-v2-failed
    revision: 2d9ea53cf8f4835fcfc1656b23d56308696b3e5b
    role: failure
    task_key: two_can
    expected_items: 6
robot_type: seeed_b601_dm_follower
fps: 30
joint_names: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper]
camera_keys: [front, side]
destination_repo: Cornerf/rebot-cansort-rerun-curated
quality:
  max_state_age_ms: 100.0
  max_camera_age_ms: 66.666667
  lag_search_frames: 15
  calibration_fraction: 0.8
  upper_quantile: 0.99
  lower_quantile: 0.01
  mad_multiplier: 5.0
```

- [ ] **Step 5: Implement immutable contracts and strict validation**

Use frozen dataclasses and literal role/verdict types. `ChallengeConfig.load`
must reject non-40-character lowercase hexadecimal revisions, duplicate repos,
counts below one, FPS other than 30, joint/camera ordering changes, and a blank
destination. `EpisodeIdentity.canonical` must return
`<repo>@<revision>:<source_key>` and be the only identity string constructor.

```python
@dataclass(frozen=True)
class EpisodeIdentity:
    repo_id: str
    revision: str
    source_key: str

    @property
    def canonical(self) -> str:
        return f"{self.repo_id}@{self.revision}:{self.source_key}"


@dataclass(frozen=True)
class InventoryRow:
    identity: EpisodeIdentity
    role: Literal["success", "failure"]
    task_key: Literal["single_can", "two_can"]
    source_path: str
    episode_index: int | None
    attempt_id: str | None
    frame_count: int
    captured_at: str


@dataclass(frozen=True)
class SourceSpec:
    repo_id: str
    revision: str
    role: Literal["success", "failure"]
    task_key: Literal["single_can", "two_can"]
    expected_items: int
    expected_frames: int | None


@dataclass(frozen=True)
class QualityConfig:
    max_state_age_ms: float
    max_camera_age_ms: float
    lag_search_frames: int
    calibration_fraction: float
    upper_quantile: float
    lower_quantile: float
    mad_multiplier: float


@dataclass(frozen=True)
class ChallengeConfig:
    sources: tuple[SourceSpec, ...]
    robot_type: str
    fps: int
    joint_names: tuple[str, ...]
    camera_keys: tuple[str, ...]
    destination_repo: str
    quality: QualityConfig

    @classmethod
    def load(cls, path: Path) -> "ChallengeConfig": ...
```

- [ ] **Step 6: Run focused tests and commit**

Run: `./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q p5_rerun_port/tests/challenge/test_config.py`

Expected: PASS.

Commit:

```bash
git add config/rerun_query_challenge.yaml requirements.txt p5_rerun_port/challenge p5_rerun_port/tests/challenge/test_config.py
git commit -m "feat: lock Rerun Query challenge sources"
```

---

### Task 2: Repair the existing episode Query API boundary

**Files:**
- Modify: `p5_rerun_port/rerun_query.py:35-236`
- Modify: `p5_rerun_port/query_api_cli.py:32-165`
- Test: `p5_rerun_port/tests/challenge/test_query_selection.py`

**Interfaces:**
- Consumes: `EpisodeRecord` rows and Query API dataframes.
- Produces: `rrd_paths_for_records(records) -> list[Path]` and segment-safe `compare_goal_vs_position`.

- [ ] **Step 1: Write regression tests for the three proven bugs**

```python
def test_dataset_paths_exclude_replays_and_unregistered_rrds(tmp_path: Path) -> None:
    root = tmp_path / "recordings" / "cans"
    root.mkdir(parents=True)
    for name in ("episode_01.rrd", "episode_01_replay.rrd", "orphan.rrd"):
        (root / name).write_bytes(b"rrd")
    (root / "episode_01.meta.json").write_text("{}")
    assert dataset_rrd_paths(tmp_path / "recordings", "cans") == [root / "episode_01.rrd"]


def test_rrd_paths_for_records_uses_only_selected_episode(tmp_path: Path) -> None:
    selected = tmp_path / "episode_02.rrd"
    selected.write_bytes(b"rrd")
    record = EpisodeRecord(dataset="cans", episode="episode_02", rrd_path=str(selected))
    assert rrd_paths_for_records([record]) == [selected.resolve()]


def test_alignment_never_pairs_rows_from_different_segments() -> None:
    df = pd.DataFrame({
        "rerun_segment_id": ["a", "b"],
        "time": [0.0, 0.0],
        "/follower/goal:Scalars:scalars": [[1.0] * 7, [9.0] * 7],
        "/follower/position:Scalars:scalars": [[0.0] * 7, None],
    })
    result = aligned_vector_rows(df, goal_column=df.columns[2], state_column=df.columns[3])
    assert result.segment_ids == ("a",)
    assert result.goal.shape == result.state.shape == (1, 7)
```

- [ ] **Step 2: Run tests and confirm current behavior fails**

Run: `./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q p5_rerun_port/tests/challenge/test_query_selection.py`

Expected: replay/orphan inclusion and missing APIs fail.

- [ ] **Step 3: Implement selection and same-row alignment**

`dataset_rrd_paths` must accept only an `.rrd` with a matching `.meta.json` and
must reject stems ending `_replay`. `rrd_paths_for_records` must resolve every
path, require it to exist, reject duplicate paths, and return the selected
order. `aligned_vector_rows` must iterate dataframe rows once and include a row
only when both vectors are present, finite, and from that row's segment.

Update `query_api_cli.main` to derive rows first, call `rrd_paths_for_records`,
and pass those paths to every `open_dataset_server` invocation. If `--episode`
or `--tag` selects zero rows, exit before starting the server. The requested
episode may label a result only after exactly one matching segment is observed.

- [ ] **Step 4: Run old and new Query API tests**

Run:

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q \
  p5_rerun_port/tests/challenge/test_query_selection.py \
  p5_rerun_port/tests/test_rerun_hardening.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add p5_rerun_port/rerun_query.py p5_rerun_port/query_api_cli.py p5_rerun_port/tests/challenge/test_query_selection.py
git commit -m "fix: isolate Rerun Query episodes and segments"
```

---

### Task 3: Build the revision-locked Hugging Face inventory

**Files:**
- Create: `p5_rerun_port/challenge/hub.py`
- Create: `p5_rerun_port/challenge/inventory.py`
- Test: `p5_rerun_port/tests/challenge/test_inventory.py`

**Interfaces:**
- Consumes: `ChallengeConfig`, Hugging Face metadata/parquet/JSON.
- Produces: `build_inventory(config, hub) -> tuple[InventoryRow, ...]` and `write_source_lock(...)`.

- [ ] **Step 1: Write fake-Hub inventory tests**

Create a `FakeHub` implementing `dataset_sha`, `read_json`, `read_parquet`, and
`download`. Feed it 52 + 25 success rows and 19 + 6 failure index rows. Assert:

```python
inventory = build_inventory(config, fake_hub)
assert len(inventory) == 102
assert sum(row.role == "success" for row in inventory) == 77
assert sum(row.role == "failure" for row in inventory) == 25
assert len({row.identity.canonical for row in inventory}) == 102
assert [row.frame_count for row in inventory if row.role == "success"][:2] == [861, 1086]
```

Also test SHA mismatch, duplicate episode index, missing failed `.rrd`, count
mismatch, blank capture time, and a source path escaping its repository prefix.

- [ ] **Step 2: Run the test and confirm the inventory module is missing**

Run: `./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q p5_rerun_port/tests/challenge/test_inventory.py`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement the Hub protocol and production adapter**

```python
class HubReader(Protocol):
    def dataset_sha(self, repo_id: str, revision: str) -> str: ...
    def read_json(self, repo_id: str, revision: str, path: str) -> dict: ...
    def read_parquet(self, repo_id: str, revision: str, path: str) -> pd.DataFrame: ...
    def download(self, repo_id: str, revision: str, path: str) -> Path: ...
```

The production adapter uses `HfApi.dataset_info(..., revision=revision)` and
`hf_hub_download(repo_type="dataset", revision=revision)`. It never uses a
moving `main` revision after configuration is loaded.

- [ ] **Step 4: Implement success and failure inventory readers**

Success rows come from every `meta/episodes/chunk-*/file-*.parquet`; validate
`meta/info.json`, contiguous episode indexes, task, length, and total frames.
Failure rows come from `FAILURE_INDEX.json`; validate `training_eligible` is
false, `attempt_count`, `attempt_id`, `disposition`, `samples`, timestamps, and
the existence of `failed_attempts/<id>/attempt.rrd`.

Write `source-lock.json` using sorted keys and include the config SHA-256,
source SHAs, expected/observed counts, and inventory payload digest.

- [ ] **Step 5: Run tests and commit**

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q p5_rerun_port/tests/challenge/test_inventory.py
git add p5_rerun_port/challenge/hub.py p5_rerun_port/challenge/inventory.py p5_rerun_port/tests/challenge/test_inventory.py
git commit -m "feat: inventory labeled Hugging Face robot data"
```

---

### Task 4: Materialize successful LeRobot episodes as canonical RRDs

**Files:**
- Create: `p5_rerun_port/challenge/canonical.py`
- Test: `p5_rerun_port/tests/challenge/test_success_materialization.py`

**Interfaces:**
- Consumes: success `InventoryRow`, LeRobot dataset sample stream.
- Produces: `materialize_success(row, config, output) -> CanonicalArtifact`.

- [ ] **Step 1: Write a three-frame materialization integration test**

Use a fake episode reader returning three samples with `torch` or NumPy action,
state, and two images. Materialize a real `.rrd`, open it with
`open_dataset_server`, and assert canonical entities, three aligned rows,
source identity, and output SHA-256.

- [ ] **Step 2: Run and confirm failure**

Run: `./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q p5_rerun_port/tests/challenge/test_success_materialization.py`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement one canonical writer**

```python
class CanonicalEpisodeWriter:
    def __init__(self, path: Path, identity: EpisodeIdentity, task: str, fps: int): ...
    def add(self, frame: int, timestamp_s: float, action: np.ndarray,
            state: np.ndarray, front_rgb: np.ndarray, side_rgb: np.ndarray) -> None: ...
    def finish(self) -> CanonicalArtifact: ...
```

Define the returned type in the same module:

```python
@dataclass(frozen=True)
class CanonicalArtifact:
    identity: str
    status: Literal["ready", "rejected"]
    rrd_path: Path | None
    sha256: str | None
    frame_count: int
    reason_codes: tuple[str, ...]
```

At each frame, set `frame` sequence and `time` duration timelines, validate
finite `(7,)` vectors, and log `/follower/goal`, `/follower/position`,
`/camera/cam0`, and `/camera/cam1`. Use `rr.Image(rgb).compress(jpeg_quality=85)`
for the materialized cache. Log source identity, repo, revision, source index,
task, FPS, and joint names as static documents. Write through a temporary path,
disconnect, verify the `.rrd` with Rerun CLI, then atomically rename.

- [ ] **Step 4: Implement the real LeRobot reader**

Construct:

```python
LeRobotDataset(
    repo_id=row.identity.repo_id,
    episodes=[row.episode_index],
    revision=row.identity.revision,
    force_cache_sync=True,
    download_videos=True,
    video_backend="pyav",
)
```

Normalize CHW/HWC and float/uint8 images to RGB `uint8`. Require local
`episode_index` and `frame_index` to match the inventory and generate timestamp
as the sample timestamp after checking `frame_index / 30` tolerance.

- [ ] **Step 5: Test, then commit**

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q p5_rerun_port/tests/challenge/test_success_materialization.py
git add p5_rerun_port/challenge/canonical.py p5_rerun_port/tests/challenge/test_success_materialization.py
git commit -m "feat: materialize LeRobot successes in Rerun"
```

---

### Task 5: Canonicalize failed native RRDs through Query API

**Files:**
- Create: `p5_rerun_port/challenge/failure_canonical.py`
- Test: `p5_rerun_port/tests/challenge/test_failure_canonicalization.py`

**Interfaces:**
- Consumes: native attempt `.rrd` plus trusted `InventoryRow` identity.
- Produces: the same `CanonicalArtifact` as Task 4.

- [ ] **Step 1: Write a real RRD fixture test**

Create a native attempt recording with timelines `attempt_frame` and
`attempt_time`, seven `/action/<joint>/pos` scalars, seven
`/observation/<joint>/pos` scalars, and `/observation/front` plus
`/observation/side` images. Query-canonicalize it and assert the canonical RRD
contains three `(7,)` action/state rows and both canonical camera entities.

- [ ] **Step 2: Add failure cases**

Test missing joint, duplicate segment, absent side camera, non-contiguous frame,
and an RRD containing only `attempt/result`. Each trustworthy identity must
produce a `CanonicalArtifact` with `materialization_status="rejected"` and
ordered reason codes; ambiguous multi-segment identity raises `IdentityError`.

- [ ] **Step 3: Implement discovery and transformation**

Open exactly one source RRD with `open_dataset_server(..., rrd_paths=[path])`.
Discover the attempt timeline and per-joint component columns from schema, read
them through `dataset.reader()`, and build vectors in the locked joint order.
Camera rows use the Query API dataframe; no `.traj.npz` or video sidecar may
stand in for missing Query data. Pass valid rows to `CanonicalEpisodeWriter`.

- [ ] **Step 4: Run tests and commit**

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q p5_rerun_port/tests/challenge/test_failure_canonicalization.py
git add p5_rerun_port/challenge/failure_canonical.py p5_rerun_port/tests/challenge/test_failure_canonicalization.py
git commit -m "feat: transform failed RRDs through Query API"
```

---

### Task 6: Align Query API rows and compute explainable metrics

**Files:**
- Create: `p5_rerun_port/challenge/alignment.py`
- Create: `p5_rerun_port/challenge/metrics.py`
- Test: `p5_rerun_port/tests/challenge/test_alignment.py`
- Test: `p5_rerun_port/tests/challenge/test_metrics.py`

**Interfaces:**
- Consumes: one canonical `DatasetEntry` segment.
- Produces: `AlignedEpisode` and `EpisodeMetrics`; neither type contains outcome labels.

- [ ] **Step 1: Write latest-at alignment tests**

Use sparse action/state/camera fixtures. Assert action timestamps define rows,
state age over `0.100` seconds becomes missing, camera age over `2/30` seconds
becomes missing, segments never mix, and source/matched timestamps are retained.

- [ ] **Step 2: Write exact metric tests**

For constant seven-joint offsets `[1,2,3,4,5,6,7]`, assert per-joint MAE/max,
overall RMS `sqrt(mean(offset**2))`, zero jerk, contiguous frames, and no hard
reasons. Shift state by two frames and assert `best_lag_frames == 2`. Inject one
NaN, one missing camera row, and one discontinuous frame to assert deterministic
hard reason order.

- [ ] **Step 3: Implement alignment contracts**

```python
@dataclass(frozen=True)
class AlignedEpisode:
    identity: str
    task_key: str
    frame: np.ndarray
    action_time_s: np.ndarray
    state_time_s: np.ndarray
    camera_time_s: dict[str, np.ndarray]
    action: np.ndarray
    state: np.ndarray
    camera_present: dict[str, np.ndarray]


@dataclass(frozen=True)
class EpisodeMetrics:
    identity: str
    task_key: str
    sample_count: int
    values: dict[str, float]
    per_joint: dict[str, tuple[float, ...]]
    metric_units: dict[str, str]
    hard_reasons: tuple[str, ...]
```

Require exactly one segment and use latest-at indexes computed with
`np.searchsorted`. Never forward-fill across a segment boundary or beyond the
configured age.

- [ ] **Step 4: Implement pure metrics**

Use NumPy only. Compute action-state error, lag search over `[-15,+15]`,
first/second/third differences divided by sample period, duration, per-joint
range, stationary fraction, saturation fraction, gripper transitions/travel,
camera coverage/age/gaps, and ordered hard reasons. Store units next to every
metric name in `EpisodeMetrics.metric_units`.

`measure_artifact()` must also accept a materialization-rejected
`CanonicalArtifact` and return an `EpisodeMetrics` row with `sample_count=0`
and its ordered hard reasons without attempting to open a missing canonical
RRD. This guarantees one metric/verdict row for every inventory identity.

- [ ] **Step 5: Run tests and commit**

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q \
  p5_rerun_port/tests/challenge/test_alignment.py \
  p5_rerun_port/tests/challenge/test_metrics.py
git add p5_rerun_port/challenge/alignment.py p5_rerun_port/challenge/metrics.py p5_rerun_port/tests/challenge/test_alignment.py p5_rerun_port/tests/challenge/test_metrics.py
git commit -m "feat: measure aligned Rerun episode quality"
```

---

### Task 7: Calibrate thresholds and evaluate without label leakage

**Files:**
- Create: `p5_rerun_port/challenge/quality.py`
- Create: `p5_rerun_port/challenge/evaluate.py`
- Test: `p5_rerun_port/tests/challenge/test_quality.py`
- Test: `p5_rerun_port/tests/challenge/test_evaluate.py`

**Interfaces:**
- Consumes: success `EpisodeMetrics` for calibration; later joins frozen verdicts to labels.
- Produces: `ThresholdSnapshot`, `EpisodeVerdict`, and `EvaluationReport`.

Use these public contracts:

```python
@dataclass(frozen=True)
class MetricThreshold:
    metric: str
    units: str
    direction: Literal["upper", "lower"]
    value: float


@dataclass(frozen=True)
class ThresholdSnapshot:
    calibration_identities: tuple[str, ...]
    held_out_success_identities: tuple[str, ...]
    thresholds: tuple[MetricThreshold, ...]
    payload_digest: str


@dataclass(frozen=True)
class EpisodeVerdict:
    identity: str
    verdict: Literal["PASS", "REVIEW", "REJECT"]
    reason_codes: tuple[str, ...]
    metrics_digest: str


@dataclass(frozen=True)
class EvaluationReport:
    evaluated_identities: tuple[str, ...]
    confusion: dict[str, int]
    precision: float
    recall: float
    f1: float
    false_rejection_rate: float
    by_task: dict[str, dict[str, float]]
    by_failure_label: dict[str, dict[str, float]]
    false_positive_identities: tuple[str, ...]
    false_negative_identities: tuple[str, ...]


def calibrate_thresholds(metrics: tuple[EpisodeMetrics, ...], config: ChallengeConfig) -> ThresholdSnapshot: ...
def score_episode(metrics: EpisodeMetrics, thresholds: ThresholdSnapshot) -> EpisodeVerdict: ...
def evaluate_verdicts(verdicts: tuple[EpisodeVerdict, ...], labels: dict[str, str]) -> EvaluationReport: ...
```

- [ ] **Step 1: Write calibration/verdict tests**

Build dated success metrics for 52 single-can and 25 two-can episodes. Assert
the first 41/20 identities are calibration and the last 11/5 are held out.
Assert upper thresholds use `max(q99, median + 5*MAD)`, lower thresholds use
`min(q01, median - 5*MAD)` bounded at zero, hard reasons yield `REJECT`, soft
anomalies yield `REVIEW`, and clean rows yield `PASS`.

- [ ] **Step 2: Prove scoring cannot receive labels**

Use `inspect.signature(score_episode)` to assert parameters are exactly
`metrics, thresholds`. `EpisodeMetrics` must have no `role`, `disposition`,
`failure_label`, or `training_eligible` field.

- [ ] **Step 3: Write evaluation tests**

Join frozen verdicts with a separate label map and assert confusion matrix,
precision, recall, F1, false-rejection rate, per-task counts, per-label recall,
and exact false-positive/false-negative identity lists. Define predicted
questionable as `REVIEW` or `REJECT` and actual questionable as any failure
source.

- [ ] **Step 4: Implement deterministic snapshots and evaluation**

Sort identities before every aggregation. Serialize thresholds with metric
name, units, direction, calibration identities, quantiles, median, MAD, and
final threshold. Evaluation reveals labels only after verdict payload digests
are finalized.

- [ ] **Step 5: Run tests and commit**

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q \
  p5_rerun_port/tests/challenge/test_quality.py \
  p5_rerun_port/tests/challenge/test_evaluate.py
git add p5_rerun_port/challenge/quality.py p5_rerun_port/challenge/evaluate.py p5_rerun_port/tests/challenge/test_quality.py p5_rerun_port/tests/challenge/test_evaluate.py
git commit -m "feat: evaluate Query verdicts without label leakage"
```

---

### Task 8: Generate deterministic artifacts and judge report

**Files:**
- Create: `p5_rerun_port/challenge/artifacts.py`
- Create: `p5_rerun_port/challenge/report.py`
- Test: `p5_rerun_port/tests/challenge/test_artifacts.py`
- Test: `p5_rerun_port/tests/challenge/test_report.py`

**Interfaces:**
- Consumes: inventory, metrics, thresholds, verdicts, evaluation.
- Produces: run directory, `selection-manifest.json`, checksums, HTML/Markdown report.

Public functions:

```python
@dataclass(frozen=True)
class RunPayloads:
    source_lock: dict
    inventory: tuple[InventoryRow, ...]
    metrics: tuple[EpisodeMetrics, ...]
    thresholds: ThresholdSnapshot
    verdicts: tuple[EpisodeVerdict, ...]
    evaluation: EvaluationReport
    selection_manifest: dict


def build_selection_manifest(
    inventory: tuple[InventoryRow, ...],
    verdicts: tuple[EpisodeVerdict, ...],
    thresholds: ThresholdSnapshot,
    source_lock: dict,
    generated_at: str,
) -> dict: ...

def selection_payload_digest(manifest: dict) -> str: ...
def write_run_artifacts(run_root: Path, payloads: RunPayloads) -> Path: ...
def render_report(payloads: RunPayloads) -> tuple[str, str]: ...
```

- [ ] **Step 1: Write digest and manifest exclusion tests**

Assert two manifests with different `generated_at` values have identical
`selection_payload_digest`, reordered inputs serialize identically, attempts to
place a failure or `REVIEW` identity in the selected list raise `ManifestError`,
and every excluded source still appears with verdict plus ordered reasons.

- [ ] **Step 2: Write report contract tests**

Assert the self-contained HTML and Markdown include all seven challenge verbs,
102 total items, 77 successes, 25 failures, four source links with SHAs,
evaluation metrics, false positives/negatives, review queue, derivative repo,
and the telemetry-versus-semantic-failure limitation. Assert the HTML contains
no external script or stylesheet URLs.

- [ ] **Step 3: Implement atomic artifact writers**

Write JSON with sorted keys and compact canonical separators, Parquet through
PyArrow, and CSV with explicit columns. Build in a temporary directory, compute
SHA-256 for every machine-readable artifact, then atomically rename to a run ID
derived from config, source-lock, and code digests.

- [ ] **Step 4: Implement report generation**

Render tables and inline SVG bar/box plots from escaped values. Include a
good-versus-failed drill-down, per-joint tracking table, confusion matrix,
reason-code counts, and selected/excluded identities. Never hide misses.

- [ ] **Step 5: Run tests and commit**

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q \
  p5_rerun_port/tests/challenge/test_artifacts.py \
  p5_rerun_port/tests/challenge/test_report.py
git add p5_rerun_port/challenge/artifacts.py p5_rerun_port/challenge/report.py p5_rerun_port/tests/challenge/test_artifacts.py p5_rerun_port/tests/challenge/test_report.py
git commit -m "feat: report Rerun Query quality evidence"
```

---

### Task 9: Build and fresh-load the derivative LeRobot dataset

**Files:**
- Create: `p5_rerun_port/challenge/curate.py`
- Test: `p5_rerun_port/tests/challenge/test_curate.py`

**Interfaces:**
- Consumes: validated selection manifest and pinned success datasets.
- Produces: local `Cornerf/rebot-cansort-rerun-curated` tree and validation report.

Public functions:

```python
def build_derivative(
    manifest_path: Path,
    config: ChallengeConfig,
    output_root: Path,
) -> Path: ...

def validate_derivative_fresh(
    dataset_root: Path,
    repo_id: str,
    expected_manifest_digest: str,
) -> dict: ...
```

- [ ] **Step 1: Write a two-source tiny-dataset test**

Create two temporary LeRobot v3 datasets with one seven-joint/two-camera episode
each. Select one `PASS` episode from each, build the derivative, fresh-load it,
and assert contiguous new episode indexes, total frames, task mapping, source
identity metadata, both cameras, and decoded sample shapes.

- [ ] **Step 2: Write fail-closed manifest tests**

Reject a failure role, `REVIEW`, duplicate identity, missing source episode,
changed revision, non-30 FPS source, wrong joint order, missing video, and an
existing nonempty destination.

- [ ] **Step 3: Implement manifest-driven copying**

Open each pinned source with `LeRobotDataset(..., revision=sha,
episodes=[index], video_backend="pyav")`. Create one destination with
`LeRobotDataset.create(..., fps=30, robot_type="seeed_b601_dm_follower",
use_videos=True)`, add frames in sorted manifest order, and save each episode.
Write `meta/rerun_query_selection_manifest.json` before final validation.

- [ ] **Step 4: Implement fresh-process validation**

Spawn the pinned Python executable to load the destination without inherited
dataset objects. Verify counts, contiguous indexes, timestamp tolerance,
finite `(7,)` action/state, exact names, tasks, both camera keys, and decoded
RGB shapes. Return a JSON validation report and nonzero exit on any mismatch.

- [ ] **Step 5: Run tests and commit**

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q p5_rerun_port/tests/challenge/test_curate.py
git add p5_rerun_port/challenge/curate.py p5_rerun_port/tests/challenge/test_curate.py
git commit -m "feat: prepare Query-approved LeRobot dataset"
```

---

### Task 10: Add the competition CLI and reproducible demo documentation

**Files:**
- Create: `p5_rerun_port/query_challenge_cli.py`
- Modify: `docs/p5_rerun_port/QUERY_API.md:1-183`
- Modify: `docs/HACKATHON_SUBMISSION_DEMO.md:1-151`
- Modify: `docs/Rerun_bounty_progress.md:1-45`
- Test: `p5_rerun_port/tests/challenge/test_challenge_cli.py`

**Interfaces:**
- Consumes: challenge config and artifact root.
- Produces: `inventory`, `materialize`, `audit`, `evaluate`, `prepare`, and `run` commands.

- [ ] **Step 1: Write CLI boundary tests**

Assert `--help` lists every command and challenge verb. Inject fake stage
functions and prove `run` executes inventory → materialize → audit → evaluate →
prepare in order, stops after the first error, prints the run ID/manifest/report,
and never imports hardware modules. `prepare` must be local-only; no upload flag
exists.

- [ ] **Step 2: Implement lazy CLI orchestration**

The rehearsed command is:

```bash
python -m p5_rerun_port.query_challenge_cli run \
  --config config/rerun_query_challenge.yaml \
  --artifacts-root artifacts/rerun-query
```

Each subcommand reuses prior artifacts only when their input payload digests
match. Print machine-readable final lines:

```text
RUN_ID=<digest>
REPORT_HTML=<absolute path>
SELECTION_MANIFEST=<absolute path>
DERIVATIVE_ROOT=<absolute path>
```

- [ ] **Step 3: Rewrite the prize documentation around real evidence**

Document the four source repos/SHAs, 102-item inventory, label-blind evaluation,
all seven challenge verbs, exact run command, outputs, derivative validation,
90-second judge script, and truthful semantic-failure limitation. Move the old
synthetic single-comparison command into a clearly labeled smoke-test section.

- [ ] **Step 4: Run tests and commit**

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q p5_rerun_port/tests/challenge/test_challenge_cli.py
git add p5_rerun_port/query_challenge_cli.py p5_rerun_port/tests/challenge/test_challenge_cli.py docs/p5_rerun_port/QUERY_API.md docs/HACKATHON_SUBMISSION_DEMO.md docs/Rerun_bounty_progress.md
git commit -m "feat: add Rerun Query challenge workflow"
```

---

### Task 11: Run the 102-item acceptance and stage publication

**Files:**
- Runtime only: `artifacts/rerun-query/<run-id>/`
- Runtime only: `datasets/Cornerf/rebot-cansort-rerun-curated/`
- Modify after measured run: `docs/p5_rerun_port/examples/real_query_quality_report.md`
- Modify after measured run: `docs/Rerun_bounty_progress.md`

**Interfaces:**
- Consumes: exact public source revisions and completed CLI.
- Produces: measured report, local derivative, proposed upload manifest.

- [ ] **Step 1: Build the pinned runtime**

```bash
./rebot_setup/setup.sh
uv pip install --python rebot_setup/vendor/rebot_lerobot/.venv/bin/python -r requirements.txt
```

Verify:

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -c \
  "import rerun, lerobot, pyarrow; print(rerun.__version__, lerobot.__version__, pyarrow.__version__)"
```

Expected: Rerun `0.34.x`, LeRobot `0.4.4`, and an installed PyArrow version.

- [ ] **Step 2: Run inventory before large downloads**

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m p5_rerun_port.query_challenge_cli inventory \
  --config config/rerun_query_challenge.yaml \
  --artifacts-root artifacts/rerun-query
```

Expected: exactly 102 unique identities, 77 success, 25 failure, and all four
remote SHAs equal the lock.

- [ ] **Step 3: Run the complete local workflow**

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m p5_rerun_port.query_challenge_cli run \
  --config config/rerun_query_challenge.yaml \
  --artifacts-root artifacts/rerun-query
```

Expected: 61 calibration successes, 16 held-out successes, 25 held-out
failures, one verdict per source identity, a derivative containing only `PASS`
successes, and a fresh-load validation report with zero errors.

- [ ] **Step 4: Inspect the measured output without changing thresholds**

Open the HTML report, then open the highest-ranked pass, highest-ranked review,
one hard reject, one dropped-object failure, and one false negative in Rerun
Viewer. Record the measured confusion matrix and limitations exactly; do not
retune after seeing held-out labels.

- [ ] **Step 5: Commit the small measured example**

Copy only the Markdown summary and its source/run digests into
`docs/p5_rerun_port/examples/real_query_quality_report.md`. Update progress with
measured counts and the local manifest digest. Do not commit RRDs, videos,
Parquet caches, or the derivative dataset.

```bash
git add docs/p5_rerun_port/examples/real_query_quality_report.md docs/Rerun_bounty_progress.md
git commit -m "docs: record real Query API quality results"
```

- [ ] **Step 6: Pause for explicit publication confirmation**

Show the user:

- destination `Cornerf/rebot-cansort-rerun-curated`;
- exact selected/excluded/review counts;
- source and code commits;
- `selection_payload_digest`;
- local fresh-load/video validation result;
- upload file inventory and total bytes.

Do not continue without explicit approval.

- [ ] **Step 7: Publish and verify remote parity after approval**

Use `HfApi.create_repo(repo_type="dataset", exist_ok=True)` and
`HfApi.upload_folder(...)`. Record the returned commit SHA, fetch
`meta/rerun_query_selection_manifest.json` from that exact SHA, and assert its
payload digest/counts equal local values. Update the measured report and
progress document with the immutable dataset URL and commit, then commit those
two documentation changes.

---

### Task 12: Final regression gate, PR, and merge

**Files:**
- Verify all changed source/docs/tests.
- No new implementation files beyond Tasks 1-11.

**Interfaces:**
- Consumes: completed branch and published evidence.
- Produces: mergeable PR and remote-main containment proof.

- [ ] **Step 1: Run focused challenge tests**

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q p5_rerun_port/tests/challenge
```

Expected: PASS.

- [ ] **Step 2: Run existing Rerun and Person 4 regressions**

```bash
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m pytest -q \
  p5_rerun_port/tests rebot_operator_kit/tests/test_rerun_library.py
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m unittest discover -s p3_vlm_orchestrator/tests
./rebot_setup/vendor/rebot_lerobot/.venv/bin/python -m unittest \
  rebot_operator_kit.tests.test_rollout_contract \
  rebot_operator_kit.tests.test_rollout_safety
```

Expected: all pass; no automated test performs physical motion.

- [ ] **Step 3: Run source and artifact hygiene checks**

```bash
git diff --check origin/main...HEAD
git status --short
git ls-files | rg '\.(rrd|mp4|parquet|safetensors)$' && exit 1 || true
```

Expected: no whitespace errors, only intended source/docs/tests tracked, and no
large runtime artifact committed.

- [ ] **Step 4: Review against the design acceptance criteria**

Walk all ten acceptance criteria in
`docs/superpowers/specs/2026-07-19-rerun-query-quality-gate-design.md` and cite a
test, artifact, command output, or immutable remote revision for each. Block the
PR if any criterion lacks evidence.

- [ ] **Step 5: Push, create PR, and merge only when clean**

```bash
git push --set-upstream origin codex/rerun-query-quality-gate
gh pr create --base main --head codex/rerun-query-quality-gate \
  --title "Build Rerun Query-to-Train quality gate" \
  --body "Uses Rerun Query API to align, evaluate, and prepare 102 real reBot demonstrations. Includes label-blind held-out evaluation, checksummed selection manifest, derivative-dataset fresh-load proof, and truthful semantic-failure limits."
gh pr view --json mergeable,mergeStateStatus,statusCheckRollup,url
```

Merge only when GitHub reports `MERGEABLE` and a clean merge state. Fetch
`origin/main`, assert the PR head is an ancestor, then fast-forward the original
main checkout while preserving unrelated untracked agent state.

```bash
gh pr merge --merge --delete-branch
git fetch origin --prune
git merge-base --is-ancestor HEAD origin/main
git -C /Users/medikonda/dev/DeskPartner pull --ff-only origin main
```
