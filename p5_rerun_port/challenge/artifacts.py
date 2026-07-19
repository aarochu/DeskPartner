"""Deterministic, fail-closed artifacts for the Rerun Query quality gate."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .evaluate import EvaluationReport
from .metrics import EpisodeMetrics
from .models import InventoryRow
from .quality import EpisodeVerdict, ThresholdSnapshot


class ManifestError(ValueError):
    """The selection payload is incomplete, inconsistent, or unsafe."""


class ArtifactError(RuntimeError):
    """A complete immutable run directory could not be written."""


@dataclass(frozen=True)
class RunPayloads:
    source_lock: dict
    inventory: tuple[InventoryRow, ...]
    metrics: tuple[EpisodeMetrics, ...]
    thresholds: ThresholdSnapshot
    verdicts: tuple[EpisodeVerdict, ...]
    evaluation: EvaluationReport
    selection_manifest: dict


def _canonical(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical(asdict(value))
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+Infinity" if value > 0 else "-Infinity"
        return 0.0 if value == 0 else value
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unique(items: tuple[Any, ...], identity_of: Any, name: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        identity = identity_of(item)
        if not isinstance(identity, str) or not identity.strip():
            raise ManifestError(f"{name} identity must be non-blank")
        if identity in result:
            raise ManifestError(f"{name} contains duplicate identity {identity}")
        result[identity] = item
    return result


def _source_lock_copy(source_lock: dict) -> dict:
    copied = _canonical(source_lock)
    if not isinstance(copied, dict):
        raise ManifestError("source_lock must be an object")
    sources = copied.get("sources")
    if isinstance(sources, list):
        copied["sources"] = sorted(
            sources,
            key=lambda source: (
                str(source.get("repo_id", "")) if isinstance(source, dict) else "",
                str(source.get("resolved_sha", "")) if isinstance(source, dict) else "",
            ),
        )
    return copied


def build_selection_manifest(
    inventory: tuple[InventoryRow, ...],
    verdicts: tuple[EpisodeVerdict, ...],
    thresholds: ThresholdSnapshot,
    source_lock: dict,
    generated_at: str,
) -> dict:
    """Build the only training selection: operator-approved success and Query PASS."""

    rows = _unique(inventory, lambda row: row.identity.canonical, "inventory")
    scored = _unique(verdicts, lambda verdict: verdict.identity, "verdicts")
    if set(rows) != set(scored):
        raise ManifestError("verdict identities must exactly cover inventory identities")
    if not isinstance(generated_at, str) or not generated_at.strip():
        raise ManifestError("generated_at must be a non-blank string")
    if not isinstance(thresholds.payload_digest, str) or not thresholds.payload_digest.strip():
        raise ManifestError("threshold snapshot digest must be non-blank")

    items: list[dict[str, Any]] = []
    selected: list[str] = []
    excluded: list[dict[str, Any]] = []
    for identity in sorted(rows):
        row = rows[identity]
        verdict = scored[identity]
        if verdict.verdict not in ("PASS", "REVIEW", "REJECT"):
            raise ManifestError(f"{identity} has invalid verdict {verdict.verdict!r}")
        if verdict.threshold_digest != thresholds.payload_digest:
            raise ManifestError(f"{identity} verdict threshold digest does not match snapshot")
        reasons = list(verdict.reason_codes)
        is_selected = row.role == "success" and verdict.verdict == "PASS"
        item = {
            "identity": identity,
            "role": row.role,
            "task_key": row.task_key,
            "source": {
                "repo_id": row.identity.repo_id,
                "revision": row.identity.revision,
                "source_key": row.identity.source_key,
                "path": row.source_path,
            },
            "source_path": row.source_path,
            "episode_index": row.episode_index,
            "attempt_id": row.attempt_id,
            "frame_count": row.frame_count,
            "verdict": verdict.verdict,
            "reason_codes": reasons,
            "metrics_digest": verdict.metrics_digest,
            "verdict_digest": verdict.verdict_digest,
            "selected": is_selected,
        }
        items.append(item)
        if is_selected:
            selected.append(identity)
        else:
            excluded.append(
                {
                    "identity": identity,
                    "role": row.role,
                    "task_key": row.task_key,
                    "verdict": verdict.verdict,
                    "reason_codes": reasons,
                }
            )

    lock = _source_lock_copy(source_lock)
    counts = Counter(row.role for row in rows.values())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": generated_at,
        "source_lock": lock,
        "threshold_snapshot_digest": thresholds.payload_digest,
        "destination_repo": lock.get(
            "destination_repo", "Cornerf/rebot-cansort-rerun-curated"
        ),
        "data_schema": {
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": lock.get("joint_names", []),
            },
            "state": {
                "dtype": "float32",
                "shape": [7],
                "names": lock.get("joint_names", []),
            },
            "cameras": lock.get("camera_keys", ["front", "side"]),
        },
        "source_counts": {
            "total": len(items),
            "success": counts["success"],
            "failure": counts["failure"],
        },
        "selected_identities": selected,
        "selected_episode_count": len(selected),
        "selected_frame_count": sum(rows[identity].frame_count for identity in selected),
        "excluded_items": excluded,
        "items": items,
    }
    manifest["selection_payload_digest"] = selection_payload_digest(manifest)
    return manifest


def _validated_manifest_payload(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        raise ManifestError("manifest must be an object")
    raw_items = manifest.get("items")
    raw_selected = manifest.get("selected_identities")
    raw_excluded = manifest.get("excluded_items")
    if not isinstance(raw_items, list) or not isinstance(raw_selected, list) or not isinstance(raw_excluded, list):
        raise ManifestError("manifest items, selected_identities, and excluded_items must be lists")
    items: dict[str, dict] = {}
    for item in raw_items:
        if not isinstance(item, dict) or not isinstance(item.get("identity"), str):
            raise ManifestError("every manifest item requires an identity")
        identity = item["identity"]
        if identity in items:
            raise ManifestError(f"manifest contains duplicate item {identity}")
        items[identity] = item
    if any(not isinstance(identity, str) for identity in raw_selected):
        raise ManifestError("selected identities must be strings")
    if len(raw_selected) != len(set(raw_selected)):
        raise ManifestError("selected identities must be unique")
    selected = set(raw_selected)
    expected_selected = {
        identity
        for identity, item in items.items()
        if item.get("role") == "success" and item.get("verdict") == "PASS"
    }
    if selected != expected_selected:
        raise ManifestError("selection must contain exactly success-source PASS identities")
    for identity in selected:
        item = items.get(identity)
        if item is None or item.get("role") != "success" or item.get("verdict") != "PASS":
            raise ManifestError(f"selected identity {identity} is not an approved PASS success")
        if item.get("selected") is not True:
            raise ManifestError(f"selected identity {identity} item is inconsistent")
    for identity, item in items.items():
        if bool(item.get("selected")) != (identity in selected):
            raise ManifestError(f"manifest item {identity} selected flag is inconsistent")
    excluded_by_identity: dict[str, dict] = {}
    for item in raw_excluded:
        if not isinstance(item, dict) or not isinstance(item.get("identity"), str):
            raise ManifestError("every excluded item requires an identity")
        identity = item["identity"]
        if identity in excluded_by_identity:
            raise ManifestError(f"excluded items contain duplicate identity {identity}")
        excluded_by_identity[identity] = item
    if set(excluded_by_identity) != set(items) - selected:
        raise ManifestError("excluded items must cover every non-selected source")
    for identity, excluded_item in excluded_by_identity.items():
        source_item = items[identity]
        if excluded_item.get("verdict") != source_item.get("verdict") or excluded_item.get("reason_codes") != source_item.get("reason_codes"):
            raise ManifestError(f"excluded item {identity} verdict or reasons are inconsistent")
    if manifest.get("selected_episode_count") != len(selected):
        raise ManifestError("selected episode count is inconsistent")
    if manifest.get("selected_frame_count") != sum(
        int(items[identity].get("frame_count", 0)) for identity in selected
    ):
        raise ManifestError("selected frame count is inconsistent")
    source_counts = manifest.get("source_counts")
    if not isinstance(source_counts, dict) or source_counts.get("total") != len(items):
        raise ManifestError("source counts are inconsistent")

    payload = dict(manifest)
    payload.pop("generated_at", None)
    payload.pop("selection_payload_digest", None)
    payload["selected_identities"] = sorted(raw_selected)
    payload["items"] = [items[identity] for identity in sorted(items)]
    payload["excluded_items"] = [excluded_by_identity[identity] for identity in sorted(excluded_by_identity)]
    if isinstance(payload.get("source_lock"), dict):
        payload["source_lock"] = _source_lock_copy(payload["source_lock"])
    return payload


def selection_payload_digest(manifest: dict) -> str:
    """Hash the semantic selection, excluding wall-clock generation metadata."""

    return _sha256_bytes(_json_bytes(_validated_manifest_payload(manifest)))


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _inventory_records(rows: tuple[InventoryRow, ...]) -> list[dict[str, Any]]:
    records = []
    for row in sorted(rows, key=lambda value: value.identity.canonical):
        records.append(
            {
                "identity": row.identity.canonical,
                "repo_id": row.identity.repo_id,
                "revision": row.identity.revision,
                "source_key": row.identity.source_key,
                "role": row.role,
                "task_key": row.task_key,
                "source_path": row.source_path,
                "episode_index": row.episode_index,
                "attempt_id": row.attempt_id,
                "frame_count": row.frame_count,
                "captured_at": row.captured_at,
            }
        )
    return records


def _metrics_records(rows: tuple[EpisodeMetrics, ...]) -> list[dict[str, Any]]:
    return [
        {
            "identity": row.identity,
            "segment_id": row.segment_id,
            "task_key": row.task_key,
            "metric_schema_version": row.metric_schema_version,
            "sample_count": row.sample_count,
            "values_json": _json_bytes(row.values).decode("utf-8"),
            "per_joint_json": _json_bytes(row.per_joint).decode("utf-8"),
            "metric_units_json": _json_bytes(row.metric_units).decode("utf-8"),
            "hard_reasons": list(row.hard_reasons),
        }
        for row in sorted(rows, key=lambda value: value.identity)
    ]


def _verdict_records(rows: tuple[EpisodeVerdict, ...]) -> list[dict[str, Any]]:
    return [
        {
            "identity": row.identity,
            "verdict": row.verdict,
            "reason_codes": list(row.reason_codes),
            "metrics_digest": row.metrics_digest,
            "threshold_digest": row.threshold_digest,
            "verdict_digest": row.verdict_digest,
        }
        for row in sorted(rows, key=lambda value: value.identity)
    ]


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, compression="zstd", version="2.6", write_statistics=True)


def _write_review_queue(path: Path, payloads: RunPayloads) -> None:
    from .report import review_queue_rows

    fields = (
        "rank", "identity", "role", "task_key", "anomaly_count",
        "worst_metric", "worst_value", "reason_codes",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for rank, identity, role, task_key, anomaly_count, metric, value, reasons in review_queue_rows(payloads):
            writer.writerow(
                {
                    "rank": rank,
                    "identity": identity,
                    "role": role,
                    "task_key": task_key,
                    "anomaly_count": anomaly_count,
                    "worst_metric": metric,
                    "worst_value": value,
                    "reason_codes": reasons,
                }
            )


def _run_id(payloads: RunPayloads) -> str:
    value = {
        "source_lock": _source_lock_copy(payloads.source_lock),
        "threshold_digest": payloads.thresholds.payload_digest,
        "selection_digest": selection_payload_digest(payloads.selection_manifest),
        "query_code_commit": payloads.source_lock.get("query_code_commit", "unknown"),
    }
    return _sha256_bytes(_json_bytes(value))


def verify_run_artifacts(run_dir: Path, expected_manifest_digest: str | None = None) -> dict:
    """Fail closed unless every immutable run artifact matches its checksum."""

    run_dir = Path(run_dir)
    try:
        checksums = json.loads((run_dir / "checksums.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"run checksums are unreadable: {run_dir}") from error
    if checksums.get("run_id") != run_dir.name or not isinstance(checksums.get("sha256"), dict):
        raise ArtifactError(f"run checksum identity is invalid: {run_dir}")
    for name, expected in checksums["sha256"].items():
        path = run_dir / name
        if not path.is_file() or _sha256_bytes(path.read_bytes()) != expected:
            raise ArtifactError(f"run artifact checksum mismatch: {path}")
    try:
        manifest = json.loads((run_dir / "selection-manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError("selection manifest is unreadable") from error
    actual_digest = selection_payload_digest(manifest)
    if actual_digest != manifest.get("selection_payload_digest"):
        raise ArtifactError("selection manifest payload digest is invalid")
    if expected_manifest_digest is not None and actual_digest != expected_manifest_digest:
        raise ArtifactError("existing run belongs to a different selection manifest")
    return checksums


def write_run_artifacts(run_root: Path, payloads: RunPayloads) -> Path:
    """Build a complete run beside its destination, then rename it atomically."""

    from .report import render_report

    if selection_payload_digest(payloads.selection_manifest) != payloads.selection_manifest.get("selection_payload_digest"):
        raise ManifestError("selection manifest digest does not match its payload")
    inventory_ids = {row.identity.canonical for row in payloads.inventory}
    if inventory_ids != {row.identity for row in payloads.metrics}:
        raise ArtifactError("metrics must exactly cover inventory identities")
    if inventory_ids != {row.identity for row in payloads.verdicts}:
        raise ArtifactError("verdicts must exactly cover inventory identities")

    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(payloads)
    destination = run_root / run_id
    if destination.exists():
        verify_run_artifacts(
            destination, payloads.selection_manifest["selection_payload_digest"]
        )
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=run_root))
    try:
        _write_json(temporary / "source-lock.json", payloads.source_lock)
        _write_parquet(temporary / "inventory.parquet", _inventory_records(payloads.inventory))
        _write_parquet(temporary / "aligned-metrics.parquet", _metrics_records(payloads.metrics))
        _write_json(temporary / "thresholds.json", payloads.thresholds)
        _write_parquet(temporary / "verdicts.parquet", _verdict_records(payloads.verdicts))
        _write_json(temporary / "evaluation.json", payloads.evaluation)
        _write_json(temporary / "selection-manifest.json", payloads.selection_manifest)
        _write_review_queue(temporary / "review-queue.csv", payloads)
        html_report, markdown_report = render_report(payloads)
        (temporary / "report.html").write_text(html_report, encoding="utf-8")
        (temporary / "report.md").write_text(markdown_report, encoding="utf-8")

        checksums = {
            path.name: _sha256_bytes(path.read_bytes())
            for path in sorted(temporary.iterdir(), key=lambda value: value.name)
            if path.is_file()
        }
        _write_json(
            temporary / "checksums.json",
            {"schema_version": 1, "run_id": run_id, "sha256": checksums},
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
