from __future__ import annotations

from dataclasses import replace
import json

import pytest

from p5_rerun_port.challenge.artifacts import (
    ManifestError,
    RunPayloads,
    build_selection_manifest,
    selection_payload_digest,
    write_run_artifacts,
)
from p5_rerun_port.challenge.evaluate import EvaluationReport
from p5_rerun_port.challenge.metrics import EpisodeMetrics
from p5_rerun_port.challenge.models import EpisodeIdentity, InventoryRow
from p5_rerun_port.challenge.quality import EpisodeVerdict, ThresholdSnapshot


def _inventory() -> tuple[InventoryRow, ...]:
    return (
        InventoryRow(EpisodeIdentity("org/good", "a" * 40, "0"), "success", "single_can", "data", 0, None, 10, "2026-01-01T00:00:00Z"),
        InventoryRow(EpisodeIdentity("org/good", "a" * 40, "1"), "success", "single_can", "data", 1, None, 20, "2026-01-02T00:00:00Z"),
        InventoryRow(EpisodeIdentity("org/bad", "b" * 40, "attempt"), "failure", "two_can", "attempt.rrd", None, "attempt", 5, "2026-01-03T00:00:00Z"),
    )


def _snapshot() -> ThresholdSnapshot:
    return ThresholdSnapshot(("cal",), ("hold",), (), "threshold-digest")


def _verdicts(rows: tuple[InventoryRow, ...]) -> tuple[EpisodeVerdict, ...]:
    values = (("PASS", ()), ("REVIEW", ("ANOMALY_TRACKING_RMS_DEG",)), ("PASS", ()))
    return tuple(
        EpisodeVerdict(row.identity.canonical, verdict, reasons, f"metrics-{index}", "threshold-digest", f"verdict-{index}")
        for index, (row, (verdict, reasons)) in enumerate(zip(rows, values, strict=True))
    )


def _source_lock() -> dict:
    return {
        "schema_version": 1,
        "config_sha256": "config-digest",
        "query_code_commit": "code-digest",
        "destination_repo": "Cornerf/rebot-cansort-rerun-curated",
        "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5", "j6"],
        "camera_keys": ["front", "side"],
        "sources": [
            {"repo_id": "org/good", "resolved_sha": "a" * 40, "role": "success"},
            {"repo_id": "org/bad", "resolved_sha": "b" * 40, "role": "failure"},
        ],
    }


def test_manifest_digest_ignores_generation_time_and_input_order() -> None:
    rows = _inventory()
    verdicts = _verdicts(rows)
    first = build_selection_manifest(rows, verdicts, _snapshot(), _source_lock(), "2026-01-01T00:00:00Z")
    second = build_selection_manifest(tuple(reversed(rows)), tuple(reversed(verdicts)), _snapshot(), _source_lock(), "2026-07-19T00:00:00Z")

    assert first["selection_payload_digest"] == second["selection_payload_digest"]
    assert selection_payload_digest(first) == selection_payload_digest(second)
    assert first["selected_identities"] == [rows[0].identity.canonical]
    assert [item["identity"] for item in first["excluded_items"]] == sorted(
        (rows[1].identity.canonical, rows[2].identity.canonical)
    )
    assert all("verdict" in item and "reason_codes" in item for item in first["excluded_items"])


@pytest.mark.parametrize("index", [1, 2])
def test_manifest_digest_rejects_tampered_review_or_failure_selection(index: int) -> None:
    rows = _inventory()
    manifest = build_selection_manifest(rows, _verdicts(rows), _snapshot(), _source_lock(), "now")
    manifest["selected_identities"].append(rows[index].identity.canonical)
    with pytest.raises(ManifestError):
        selection_payload_digest(manifest)


def test_write_run_artifacts_is_complete_and_checksummed(tmp_path) -> None:
    rows = _inventory()
    verdicts = _verdicts(rows)
    manifest = build_selection_manifest(rows, verdicts, _snapshot(), _source_lock(), "now")
    metrics = tuple(
        EpisodeMetrics(row.identity.canonical, f"segment-{index}", row.task_key, "1", row.frame_count, {"tracking_rms_deg": float(index)}, {"tracking_mae_per_joint_deg": (1.0,) * 7}, {"tracking_rms_deg": "deg"}, ())
        for index, row in enumerate(rows)
    )
    evaluation = EvaluationReport(tuple(row.identity.canonical for row in rows), {"tp": 1, "fp": 1, "tn": 0, "fn": 1}, .5, .5, .5, 1.0, {}, {}, {}, (rows[1].identity.canonical,), (rows[2].identity.canonical,))
    payloads = RunPayloads(_source_lock(), rows, metrics, _snapshot(), verdicts, evaluation, manifest)

    run_dir = write_run_artifacts(tmp_path, payloads)

    expected = {"source-lock.json", "inventory.parquet", "aligned-metrics.parquet", "thresholds.json", "verdicts.parquet", "evaluation.json", "selection-manifest.json", "review-queue.csv", "report.html", "report.md", "checksums.json"}
    assert {path.name for path in run_dir.iterdir()} == expected
    checksums = json.loads((run_dir / "checksums.json").read_text())
    assert set(checksums["sha256"]) == expected - {"checksums.json"}
    assert run_dir.name == checksums["run_id"]
