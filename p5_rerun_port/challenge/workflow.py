"""Concrete, resumable stage orchestration for the challenge CLI.

Imports stay inside stage functions so ``query_challenge_cli --help`` remains
hardware-free and does not require the data stack to be initialized.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
from typing import Any


def _code_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    cli = root.parent / "query_challenge_cli.py"
    if cli.is_file():
        digest.update(cli.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(cli.read_bytes())
    return digest.hexdigest()


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(context: Any) -> Path:
    key = _digest(
        {
            "config": hashlib.sha256(context.config_path.read_bytes()).hexdigest(),
            "code": _code_sha256(),
        }
    )[:16]
    return context.artifacts_root / ".workflow" / key / "state.pkl"


def _read_cache(context: Any) -> dict[str, Any]:
    path = _cache_path(context)
    expected = {
        "config_sha256": hashlib.sha256(context.config_path.read_bytes()).hexdigest(),
        "query_code_commit": _code_sha256(),
    }
    if not path.exists():
        return expected
    with path.open("rb") as stream:
        value = pickle.load(stream)  # local cache under the caller-selected artifacts root
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        return expected
    return value


def _write_cache(context: Any, cache: dict[str, Any]) -> None:
    path = _cache_path(context)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(cache, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _stage_input(cache: dict[str, Any], prerequisite: str | None) -> str:
    return _digest(
        {
            "config_sha256": cache["config_sha256"],
            "query_code_commit": cache["query_code_commit"],
            "prerequisite": None if prerequisite is None else cache.get(f"{prerequisite}_digest"),
        }
    )


def _inventory_dependencies():
    from .config import ChallengeConfig
    from .hub import HuggingFaceHubReader
    from .inventory import build_inventory, write_source_lock

    return ChallengeConfig, HuggingFaceHubReader, build_inventory, write_source_lock


def run_inventory_stage(context: Any, state: Any) -> Any:
    cache = _read_cache(context)
    input_digest = _stage_input(cache, None)
    if cache.get("inventory_input_digest") != input_digest:
        ChallengeConfig, Hub, build_inventory, write_source_lock = _inventory_dependencies()
        config = ChallengeConfig.load(context.config_path)
        inventory = build_inventory(config, Hub())
        staging = _cache_path(context).parent
        source_lock = write_source_lock(
            staging / "source-lock.json", config, inventory, config_path=context.config_path
        )
        source_lock["query_code_commit"] = cache["query_code_commit"]
        (staging / "source-lock.json").write_text(
            json.dumps(source_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        cache.update(
            config=config,
            inventory=inventory,
            source_lock=source_lock,
            inventory_input_digest=input_digest,
            inventory_digest=_digest(
                {"source_lock": source_lock, "identities": [row.identity.canonical for row in inventory]}
            ),
        )
        _write_cache(context, cache)
    return state.with_updates(run_id=cache["inventory_digest"][:16], payload=cache)


def _safe_name(identity: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", identity) + ".rrd"


def run_materialize_stage(context: Any, state: Any) -> Any:
    cache = state.payload if isinstance(state.payload, dict) else _read_cache(context)
    if "inventory_digest" not in cache:
        raise RuntimeError("materialize requires a matching inventory stage")
    input_digest = _stage_input(cache, "inventory")
    if cache.get("materialize_input_digest") != input_digest:
        from .canonical import materialize_success
        from .failure_canonical import materialize_failure
        from .hub import HuggingFaceHubReader

        hub = HuggingFaceHubReader()
        output_root = _cache_path(context).parent / "canonical-rrd"
        output_root.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for row in cache["inventory"]:
            output = output_root / _safe_name(row.identity.canonical)
            if row.role == "success":
                artifact = materialize_success(row, cache["config"], output)
            else:
                source = hub.download(row.identity.repo_id, row.identity.revision, row.source_path)
                artifact = materialize_failure(row, cache["config"], source, output)
            artifacts.append(artifact)
        cache.update(
            canonical_artifacts=tuple(artifacts),
            materialize_input_digest=input_digest,
            materialize_digest=_digest([asdict(item) for item in artifacts]),
        )
        _write_cache(context, cache)
    return state.with_updates(run_id=cache["materialize_digest"][:16], payload=cache)


def run_audit_stage(context: Any, state: Any) -> Any:
    cache = state.payload if isinstance(state.payload, dict) else _read_cache(context)
    if "materialize_digest" not in cache:
        raise RuntimeError("audit requires matching canonical materialization artifacts")
    input_digest = _stage_input(cache, "materialize")
    if cache.get("audit_input_digest") != input_digest:
        from .metrics import measure_artifact

        metrics = tuple(measure_artifact(item, cache["config"]) for item in cache["canonical_artifacts"])
        cache.update(
            metrics=metrics,
            audit_input_digest=input_digest,
            audit_digest=_digest([asdict(item) for item in metrics]),
        )
        _write_cache(context, cache)
    return state.with_updates(run_id=cache["audit_digest"][:16], payload=cache)


def run_evaluate_stage(context: Any, state: Any) -> Any:
    cache = state.payload if isinstance(state.payload, dict) else _read_cache(context)
    if "audit_digest" not in cache:
        raise RuntimeError("evaluate requires matching aligned metrics")
    input_digest = _stage_input(cache, "audit")
    if cache.get("evaluate_input_digest") != input_digest:
        from .evaluate import build_evaluation_labels, evaluate_verdicts
        from .hub import HuggingFaceHubReader
        from .quality import CalibrationCapture, calibrate_thresholds, score_episode

        successes = tuple(row for row in cache["inventory"] if row.role == "success")
        success_ids = {row.identity.canonical for row in successes}
        success_metrics = tuple(item for item in cache["metrics"] if item.identity in success_ids)
        captures = tuple(
            CalibrationCapture(row.identity.canonical, row.task_key, row.captured_at)
            for row in successes
        )
        thresholds = calibrate_thresholds(success_metrics, captures, cache["config"])
        all_verdicts = tuple(score_episode(item, thresholds) for item in cache["metrics"])
        evaluated_ids = set(thresholds.held_out_success_identities) | {
            row.identity.canonical for row in cache["inventory"] if row.role == "failure"
        }
        evaluation_verdicts = tuple(item for item in all_verdicts if item.identity in evaluated_ids)
        labels = build_evaluation_labels(cache["inventory"], HuggingFaceHubReader())
        evaluation = evaluate_verdicts(thresholds, evaluation_verdicts, labels)
        cache.update(
            thresholds=thresholds,
            verdicts=all_verdicts,
            evaluation=evaluation,
            evaluate_input_digest=input_digest,
            evaluate_digest=_digest(
                {"thresholds": asdict(thresholds), "verdicts": [asdict(v) for v in all_verdicts], "evaluation": asdict(evaluation)}
            ),
        )
        _write_cache(context, cache)
    return state.with_updates(run_id=cache["evaluate_digest"][:16], payload=cache)


def run_prepare_stage(context: Any, state: Any) -> Any:
    cache = state.payload if isinstance(state.payload, dict) else _read_cache(context)
    if "evaluate_digest" not in cache:
        raise RuntimeError("prepare requires matching frozen verdicts and evaluation")
    input_digest = _stage_input(cache, "evaluate")
    if cache.get("prepare_input_digest") != input_digest:
        from .artifacts import RunPayloads, build_selection_manifest, write_run_artifacts
        from .curate import build_derivative, validate_derivative_fresh

        manifest = build_selection_manifest(
            cache["inventory"],
            cache["verdicts"],
            cache["thresholds"],
            cache["source_lock"],
            datetime.now(timezone.utc).isoformat(),
        )
        payloads = RunPayloads(
            cache["source_lock"], cache["inventory"], cache["metrics"], cache["thresholds"],
            cache["verdicts"], cache["evaluation"], manifest,
        )
        run_dir = write_run_artifacts(context.artifacts_root, payloads).resolve()
        manifest_path = run_dir / "selection-manifest.json"
        derivative_root = (context.artifacts_root / "_derivatives").resolve()
        dataset_root = build_derivative(manifest_path, cache["config"], derivative_root)
        expected_digest = manifest["selection_payload_digest"]
        validation = validate_derivative_fresh(dataset_root, cache["config"].destination_repo, expected_digest)
        validation_path = Path(dataset_root).parent / "derivative-validation.json"
        validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
        cache.update(
            prepare_input_digest=input_digest,
            prepare_digest=_digest({"manifest": expected_digest, "validation": validation}),
            run_dir=run_dir,
            report_html=run_dir / "report.html",
            selection_manifest=manifest_path,
            derivative_root=Path(dataset_root),
        )
        _write_cache(context, cache)
    else:
        from .artifacts import verify_run_artifacts
        from .curate import validate_derivative_fresh

        manifest = json.loads(Path(cache["selection_manifest"]).read_text(encoding="utf-8"))
        verify_run_artifacts(
            Path(cache["run_dir"]), manifest.get("selection_payload_digest")
        )
        validate_derivative_fresh(
            Path(cache["derivative_root"]),
            cache["config"].destination_repo,
            manifest["selection_payload_digest"],
        )
    return state.with_updates(
        run_id=Path(cache["run_dir"]).name,
        report_html=Path(cache["report_html"]),
        selection_manifest=Path(cache["selection_manifest"]),
        derivative_root=Path(cache["derivative_root"]),
        payload=cache,
    )
