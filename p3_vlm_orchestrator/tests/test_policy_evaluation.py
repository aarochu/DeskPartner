from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from p3_vlm_orchestrator.policy_rollout import evaluation
except ImportError:
    evaluation = None  # type: ignore[assignment]

from p3_vlm_orchestrator.policy_rollout.cli import CliDependencies, build_parser, main


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
SUMMARY_FIELDS = (
    "checkpoint",
    "trials",
    "grasp_successes",
    "placement_successes",
    "safety_faults",
    "clamps",
    "mean_completion_s",
    "overall_success_rate",
)


def make_trial(
    index: int,
    *,
    checkpoint: str = "/models/checkpoint-a",
    digest: str = DIGEST_A,
    grasp_success: bool = True,
    placement_success: bool = True,
    terminal_reason: str = "operator_success",
    safety_faults: int = 0,
    clamps: int = 0,
    completion_s: float | None = None,
) -> dict[str, object]:
    return {
        "checkpoint": checkpoint,
        "checkpoint_digest": digest,
        "placement_id": f"held-out-{index:02d}",
        "attempts_used": 1,
        "grasp_success": grasp_success,
        "placement_success": placement_success,
        "terminal_reason": terminal_reason,
        "safety_faults": safety_faults,
        "clamps": clamps,
        "completion_s": float(index if completion_s is None else completion_s),
        "source_jsonl_paths": [f"/audit/held-out-{index:02d}.jsonl"],
    }


def make_manifest(
    count: int = 10,
    *,
    checkpoint: str = "/models/checkpoint-a",
    digest: str = DIGEST_A,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "checkpoint": checkpoint,
        "checkpoint_digest": digest,
        "trials": [
            make_trial(index, checkpoint=checkpoint, digest=digest)
            for index in range(1, count + 1)
        ],
    }


class PolicyEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def require_module(self):
        self.assertIsNotNone(evaluation, "evaluation module is missing")
        return evaluation

    def write_manifest(
        self,
        manifest: dict[str, object],
        name: str = "trials.json",
        *,
        materialize_audits: bool = True,
    ) -> Path:
        if materialize_audits:
            audit_root = self.root / "audit" / Path(name).stem
            audit_root.mkdir(parents=True, exist_ok=True)
            for trial_index, trial in enumerate(
                manifest.get("trials", []),  # type: ignore[union-attr]
                start=1,
            ):
                if not isinstance(trial, dict):
                    continue
                sources = trial.get("source_jsonl_paths")
                if (
                    not isinstance(sources, list)
                    or not sources
                    or not all(
                        isinstance(source, str) and source.startswith("/audit/")
                        for source in sources
                    )
                ):
                    continue
                audit_path = audit_root / (
                    f"{trial_index:02d}-{trial.get('placement_id', 'invalid')}.jsonl"
                )
                attempts = trial.get("attempts_used", 1)
                attempts = attempts if isinstance(attempts, int) and not isinstance(attempts, bool) else 1
                completion = trial.get("completion_s", 0.0)
                completion = (
                    float(completion)
                    if isinstance(completion, (int, float))
                    and not isinstance(completion, bool)
                    else 0.0
                )
                clamps = trial.get("clamps", 0)
                clamps = clamps if isinstance(clamps, int) and not isinstance(clamps, bool) else 0
                rows: list[dict[str, object]] = [{"event": "rollout_metadata"}]
                for attempt in range(1, max(1, attempts) + 1):
                    final = attempt == attempts
                    rows.append(
                        {
                            "event": "terminal",
                            "attempt": attempt,
                            "terminal_reason": (
                                trial.get("terminal_reason")
                                if final
                                else "operator_failure"
                            ),
                            "clamp_count": clamps if final else 0,
                            "elapsed_seconds": completion if final else 0.0,
                        }
                    )
                audit_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                trial["source_jsonl_paths"] = [str(audit_path.resolve())]
        path = self.root / name
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_accepts_exactly_ten_or_fifteen_distinct_final_trials(self) -> None:
        module = self.require_module()
        for count in (10, 15):
            with self.subTest(count=count):
                manifest = module.load_trial_manifest(
                    self.write_manifest(make_manifest(count), f"trials-{count}.json")
                )
                self.assertEqual(len(manifest.trials), count)

    def test_rejects_nine_or_sixteen_trials_and_duplicate_placements(self) -> None:
        module = self.require_module()
        for count in (9, 16):
            with self.subTest(count=count):
                path = self.write_manifest(make_manifest(count), f"bad-{count}.json")
                with self.assertRaisesRegex(ValueError, "10 to 15"):
                    module.load_trial_manifest(path)
        duplicate = make_manifest()
        duplicate["trials"][1]["placement_id"] = "held-out-01"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "distinct placement"):
            module.load_trial_manifest(self.write_manifest(duplicate, "duplicate.json"))

    def test_rejects_mixed_checkpoint_identity_or_digest(self) -> None:
        module = self.require_module()
        for field, value in (("checkpoint", "/models/other"), ("checkpoint_digest", DIGEST_B)):
            with self.subTest(field=field):
                manifest = make_manifest()
                manifest["trials"][3][field] = value  # type: ignore[index]
                with self.assertRaisesRegex(ValueError, "mixed checkpoint"):
                    module.load_trial_manifest(
                        self.write_manifest(manifest, f"mixed-{field}.json")
                    )

    def test_rejects_nonfinite_negative_or_boolean_counts_and_bad_attempts(self) -> None:
        module = self.require_module()
        cases = (
            ("completion_s", -0.1),
            ("completion_s", float("nan")),
            ("completion_s", "1.0"),
            ("safety_faults", -1),
            ("clamps", True),
            ("attempts_used", 0),
            ("attempts_used", 3),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                manifest = make_manifest()
                manifest["trials"][0][field] = value  # type: ignore[index]
                path = self.write_manifest(manifest, f"bad-{field}-{value!s}.json")
                with self.assertRaises(ValueError):
                    module.load_trial_manifest(path)

    def test_rejects_unknown_terminal_reason_and_inconsistent_success_labels(self) -> None:
        module = self.require_module()
        cases = (
            {"terminal_reason": "unknown"},
            {"grasp_success": False, "placement_success": True},
            {
                "terminal_reason": "safety_fault",
                "safety_faults": 1,
                "grasp_success": True,
                "placement_success": True,
            },
            {"terminal_reason": "safety_fault", "safety_faults": 0,
             "grasp_success": False, "placement_success": False},
            {"terminal_reason": "operator_success", "placement_success": False},
        )
        for index, changes in enumerate(cases):
            with self.subTest(changes=changes):
                manifest = make_manifest()
                manifest["trials"][0].update(changes)  # type: ignore[index]
                with self.assertRaises(ValueError):
                    module.load_trial_manifest(
                        self.write_manifest(manifest, f"inconsistent-{index}.json")
                    )

    def test_rejects_missing_extra_or_invalid_audit_fields(self) -> None:
        module = self.require_module()
        manifests: list[dict[str, object]] = []
        missing = make_manifest()
        del missing["trials"][0]["source_jsonl_paths"]  # type: ignore[index]
        manifests.append(missing)
        extra = make_manifest()
        extra["trials"][0]["training_loss"] = 0.1  # type: ignore[index]
        manifests.append(extra)
        invalid_source = make_manifest()
        invalid_source["trials"][0]["source_jsonl_paths"] = []  # type: ignore[index]
        manifests.append(invalid_source)
        for index, manifest in enumerate(manifests):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "schema|source JSONL"):
                    module.load_trial_manifest(
                        self.write_manifest(manifest, f"schema-{index}.json")
                    )

    def test_shared_retry_jsonl_is_complete_and_input_files_stay_unchanged(self) -> None:
        module = self.require_module()
        manifest = make_manifest()
        manifest["trials"][0].update(  # type: ignore[index]
            attempts_used=2,
            completion_s=3.25,
            clamps=2,
        )
        manifest_path = self.write_manifest(manifest, "shared-retry.json")
        audit_path = Path(manifest["trials"][0]["source_jsonl_paths"][0])  # type: ignore[index]
        before = (manifest_path.read_bytes(), audit_path.read_bytes())

        loaded = module.load_trial_manifest(manifest_path)

        self.assertEqual(loaded.source_path, manifest_path)
        self.assertEqual(len(loaded.trials[0].source_jsonl_paths), 1)
        self.assertEqual(
            (manifest_path.read_bytes(), audit_path.read_bytes()),
            before,
        )

    def test_rejects_source_jsonl_reused_across_placements(self) -> None:
        module = self.require_module()
        manifest = make_manifest()
        manifest["trials"][1]["completion_s"] = 1.0  # type: ignore[index]
        path = self.write_manifest(manifest, "reused-source.json")
        shared = manifest["trials"][0]["source_jsonl_paths"]  # type: ignore[index]
        manifest["trials"][1]["source_jsonl_paths"] = shared  # type: ignore[index]
        path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "reused across placements"):
            module.load_trial_manifest(path)

    def test_rejects_nonabsolute_missing_symlink_or_malformed_audit_path(self) -> None:
        module = self.require_module()
        cases: list[tuple[str, str]] = []
        cases.append(("relative", "relative.jsonl"))
        cases.append(("missing", str((self.root / "missing.jsonl").resolve())))
        target = self.root / "target.jsonl"
        target.write_text("{}\n", encoding="utf-8")
        symlink = self.root / "linked.jsonl"
        symlink.symlink_to(target)
        cases.append(("symlink", str(symlink)))
        malformed = self.root / "malformed.jsonl"
        malformed.write_text("not-json\n", encoding="utf-8")
        cases.append(("malformed", str(malformed)))
        for name, source in cases:
            with self.subTest(name=name):
                manifest = make_manifest()
                path = self.write_manifest(manifest, f"audit-{name}.json")
                manifest["trials"][0]["source_jsonl_paths"] = [source]  # type: ignore[index]
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError,
                    "absolute|regular non-symlink|JSONL",
                ):
                    module.load_trial_manifest(path)

    def test_rejects_incomplete_duplicate_or_mismatched_terminal_attempts(self) -> None:
        module = self.require_module()
        cases = (
            (
                "missing",
                [{"event": "terminal", "attempt": 1, "terminal_reason": "operator_failure", "clamp_count": 0, "elapsed_seconds": 0.0}],
            ),
            (
                "duplicate",
                [
                    {"event": "terminal", "attempt": 1, "terminal_reason": "operator_failure", "clamp_count": 0, "elapsed_seconds": 0.0},
                    {"event": "terminal", "attempt": 1, "terminal_reason": "operator_success", "clamp_count": 0, "elapsed_seconds": 3.0},
                ],
            ),
            (
                "final-reason",
                [
                    {"event": "terminal", "attempt": 1, "terminal_reason": "operator_failure", "clamp_count": 0, "elapsed_seconds": 0.0},
                    {"event": "terminal", "attempt": 2, "terminal_reason": "operator_failure", "clamp_count": 0, "elapsed_seconds": 3.0},
                ],
            ),
        )
        for name, rows in cases:
            with self.subTest(name=name):
                manifest = make_manifest()
                manifest["trials"][0].update(attempts_used=2, completion_s=3.0)  # type: ignore[index]
                path = self.write_manifest(manifest, f"terminal-{name}.json")
                audit = Path(manifest["trials"][0]["source_jsonl_paths"][0])  # type: ignore[index]
                audit.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "attempt|terminal reason"):
                    module.load_trial_manifest(path)

    def test_rejects_terminal_clamp_safety_or_completion_aggregate_mismatch(self) -> None:
        module = self.require_module()
        cases = (
            ("clamps", {"clamps": 2}, {"clamp_count": 1}),
            ("completion", {"completion_s": 3.0}, {"elapsed_seconds": 2.0}),
            (
                "safety",
                {
                    "terminal_reason": "safety_fault",
                    "grasp_success": False,
                    "placement_success": False,
                    "safety_faults": 2,
                },
                {},
            ),
        )
        for name, trial_changes, row_changes in cases:
            with self.subTest(name=name):
                manifest = make_manifest()
                manifest["trials"][0].update(trial_changes)  # type: ignore[index]
                path = self.write_manifest(manifest, f"aggregate-{name}.json")
                audit = Path(manifest["trials"][0]["source_jsonl_paths"][0])  # type: ignore[index]
                rows = [json.loads(line) for line in audit.read_text().splitlines()]
                terminal = next(row for row in rows if row.get("event") == "terminal")
                terminal.update(row_changes)
                audit.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "clamp|safety|completion",
                ):
                    module.load_trial_manifest(path)

    def test_accepts_terminal_fallback_as_a_safety_fault_audit_row(self) -> None:
        module = self.require_module()
        manifest = make_manifest()
        manifest["trials"][0].update(  # type: ignore[index]
            terminal_reason="safety_fault",
            grasp_success=False,
            placement_success=False,
            safety_faults=1,
        )
        path = self.write_manifest(manifest, "terminal-fallback.json")
        audit = Path(manifest["trials"][0]["source_jsonl_paths"][0])  # type: ignore[index]
        rows = [json.loads(line) for line in audit.read_text().splitlines()]
        terminal = next(row for row in rows if row.get("event") == "terminal")
        terminal["event"] = "terminal_fallback"
        audit.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        loaded = module.load_trial_manifest(path)

        self.assertEqual(loaded.trials[0].terminal_reason, "safety_fault")

    def test_exact_aggregation_uses_all_final_trial_durations(self) -> None:
        module = self.require_module()
        manifest = make_manifest()
        trials = manifest["trials"]  # type: ignore[assignment]
        trials[0].update(  # type: ignore[index]
            grasp_success=False,
            placement_success=False,
            terminal_reason="operator_failure",
            completion_s=1.0,
            clamps=2,
        )
        trials[1].update(  # type: ignore[index]
            grasp_success=True,
            placement_success=False,
            terminal_reason="timeout",
            completion_s=2.0,
            clamps=1,
        )
        trials[2].update(  # type: ignore[index]
            grasp_success=False,
            placement_success=False,
            terminal_reason="safety_fault",
            safety_faults=1,
            completion_s=3.0,
        )
        for index, trial in enumerate(trials[3:], start=4):  # type: ignore[index]
            trial["completion_s"] = float(index)

        loaded = module.load_trial_manifest(self.write_manifest(manifest))
        summary = module.summarize(loaded)

        self.assertEqual(tuple(summary), SUMMARY_FIELDS)
        self.assertEqual(
            summary,
            {
                "checkpoint": "/models/checkpoint-a",
                "trials": 10,
                "grasp_successes": 8,
                "placement_successes": 7,
                "safety_faults": 1,
                "clamps": 3,
                "mean_completion_s": 5.5,
                "overall_success_rate": 0.7,
            },
        )

    def test_comparison_requires_same_placements_and_ranks_without_training_loss(self) -> None:
        module = self.require_module()
        first = make_manifest(checkpoint="/models/a", digest=DIGEST_A)
        second = make_manifest(checkpoint="/models/b", digest=DIGEST_B)
        third = make_manifest(checkpoint="/models/c", digest="c" * 64)
        for trial in first["trials"][:2]:  # type: ignore[index]
            trial.update(grasp_success=False, placement_success=False,
                         terminal_reason="operator_failure")
        for trial in second["trials"][:2]:  # type: ignore[index]
            trial.update(grasp_success=False, placement_success=False,
                         terminal_reason="operator_failure", clamps=1)
        for trial in third["trials"][:3]:  # type: ignore[index]
            trial.update(grasp_success=False, placement_success=False,
                         terminal_reason="operator_failure")
        ranked = module.compare(
            [
                module.load_trial_manifest(self.write_manifest(first, "a.json")),
                module.load_trial_manifest(self.write_manifest(second, "b.json")),
                module.load_trial_manifest(self.write_manifest(third, "c.json")),
            ]
        )
        self.assertEqual([row["checkpoint"] for row in ranked], ["/models/a", "/models/b", "/models/c"])

        changed = make_manifest(checkpoint="/models/d", digest="d" * 64)
        changed["trials"][0]["placement_id"] = "different"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "same held-out placement"):
            module.compare(
                [
                    module.load_trial_manifest(self.write_manifest(first, "a2.json")),
                    module.load_trial_manifest(self.write_manifest(changed, "d.json")),
                ]
            )

    def test_exact_ties_are_deterministic_by_checkpoint_identity(self) -> None:
        module = self.require_module()
        manifests = []
        for name, digest in (("z", DIGEST_B), ("a", DIGEST_A)):
            path = self.write_manifest(
                make_manifest(checkpoint=f"/models/{name}", digest=digest),
                f"{name}.json",
            )
            manifests.append(module.load_trial_manifest(path))
        ranked = module.compare(manifests)
        self.assertEqual([row["checkpoint"] for row in ranked], ["/models/a", "/models/z"])

    def test_comparison_uses_unrounded_mean_time_for_ranking(self) -> None:
        module = self.require_module()
        fast = make_manifest(checkpoint="/models/z-fast", digest=DIGEST_A)
        slow = make_manifest(checkpoint="/models/a-slow", digest=DIGEST_B)
        for trial in fast["trials"]:  # type: ignore[index]
            trial["completion_s"] = 1.0000001
        for trial in slow["trials"]:  # type: ignore[index]
            trial["completion_s"] = 1.0000002
        loaded_fast = module.load_trial_manifest(
            self.write_manifest(fast, "raw-fast.json")
        )
        loaded_slow = module.load_trial_manifest(
            self.write_manifest(slow, "raw-slow.json")
        )
        self.assertEqual(
            module.summarize(loaded_fast)["mean_completion_s"],
            module.summarize(loaded_slow)["mean_completion_s"],
        )

        ranked = module.compare([loaded_slow, loaded_fast])

        self.assertEqual(
            [row["checkpoint"] for row in ranked],
            ["/models/z-fast", "/models/a-slow"],
        )

    def test_comparison_rejects_duplicate_digest_at_different_paths(self) -> None:
        module = self.require_module()
        first = module.load_trial_manifest(
            self.write_manifest(
                make_manifest(checkpoint="/models/a", digest=DIGEST_A),
                "digest-a.json",
            )
        )
        second = module.load_trial_manifest(
            self.write_manifest(
                make_manifest(checkpoint="/models/b", digest=DIGEST_A),
                "digest-b.json",
            )
        )

        with self.assertRaisesRegex(ValueError, "digest"):
            module.compare([first, second])

    def test_ranking_breaks_success_ties_by_safety_then_clamps_then_time(self) -> None:
        module = self.require_module()
        specs = (
            ("fast", "d" * 64, "operator_failure", 0, 0, 1.0),
            ("slow", "e" * 64, "operator_failure", 0, 0, 2.0),
            ("clamped", "f" * 64, "operator_failure", 0, 1, 0.1),
            ("faulted", "1" * 64, "safety_fault", 1, 0, 0.1),
        )
        manifests = []
        for name, digest, reason, faults, clamps, duration in specs:
            manifest = make_manifest(checkpoint=f"/models/{name}", digest=digest)
            for trial in manifest["trials"]:  # type: ignore[index]
                trial["completion_s"] = duration
            manifest["trials"][0].update(  # type: ignore[index]
                grasp_success=False,
                placement_success=False,
                terminal_reason=reason,
                safety_faults=faults,
                clamps=clamps,
            )
            manifests.append(
                module.load_trial_manifest(
                    self.write_manifest(manifest, f"ranking-{name}.json")
                )
            )

        ranked = module.compare(list(reversed(manifests)))

        self.assertEqual(
            [row["checkpoint"] for row in ranked],
            ["/models/fast", "/models/slow", "/models/clamped", "/models/faulted"],
        )

    def test_report_writer_is_deterministic_confined_and_read_only(self) -> None:
        module = self.require_module()
        manifest_path = self.write_manifest(make_manifest())
        before = manifest_path.read_bytes()
        loaded = module.load_trial_manifest(manifest_path)

        first = module.write_reports([loaded], output_name="checkpoint-a", repo_root=self.root)
        json_bytes = first.json_path.read_bytes()
        csv_bytes = first.csv_path.read_bytes()
        first.json_path.unlink()
        first.csv_path.unlink()
        second = module.write_reports([loaded], output_name="checkpoint-a", repo_root=self.root)

        self.assertEqual(json_bytes, second.json_path.read_bytes())
        self.assertEqual(csv_bytes, second.csv_path.read_bytes())
        self.assertEqual(manifest_path.read_bytes(), before)
        self.assertEqual(second.json_path.parent, (self.root / "runs/policy/reports").resolve())
        self.assertEqual(csv_bytes.decode().splitlines()[0], ",".join(SUMMARY_FIELDS))

    def test_report_writer_rejects_traversal_existing_outputs_and_symlink_escape(self) -> None:
        module = self.require_module()
        loaded = module.load_trial_manifest(self.write_manifest(make_manifest()))
        for name in ("../escape", "checkpoint.json", "credentials", "calibration"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    module.write_reports([loaded], output_name=name, repo_root=self.root)

        module.write_reports([loaded], output_name="immutable", repo_root=self.root)
        with self.assertRaisesRegex(ValueError, "already exists"):
            module.write_reports([loaded], output_name="immutable", repo_root=self.root)

        other = self.root / "other"
        other.mkdir()
        reports = self.root / "runs" / "policy" / "reports"
        for child in reports.iterdir():
            child.unlink()
        reports.rmdir()
        reports.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            module.write_reports([loaded], output_name="escape", repo_root=self.root)

    def test_cli_report_and_compare_help_and_output_are_lazy(self) -> None:
        module = self.require_module()
        del module
        help_text = build_parser().format_help()
        self.assertIn("report", help_text)
        self.assertIn("compare", help_text)
        first = self.write_manifest(make_manifest(checkpoint="/models/a", digest=DIGEST_A), "a.json")
        second = self.write_manifest(make_manifest(checkpoint="/models/b", digest=DIGEST_B), "b.json")
        stdout = StringIO()
        stderr = StringIO()
        status = main(
            ["compare", "--manifest", str(first), str(second), "--output-name", "held-out"],
            dependencies=CliDependencies(stdout=stdout, stderr=stderr, repo_root=self.root),
        )
        self.assertEqual(status, 0, stderr.getvalue())
        self.assertIn("report_json=", stdout.getvalue())
        self.assertIn("report_csv=", stdout.getvalue())
        self.assertIn("winner=/models/a", stdout.getvalue())

    def test_fresh_process_report_imports_no_hardware_or_model_stack(self) -> None:
        self.require_module()
        manifest = self.write_manifest(make_manifest())
        repo_root = Path(__file__).resolve().parents[2]
        script = f"""
import sys
from pathlib import Path
from p3_vlm_orchestrator.policy_rollout.cli import CliDependencies, main
status = main(
    ['report', '--manifest', {str(manifest)!r}, '--output-name', 'fresh'],
    dependencies=CliDependencies(repo_root=Path({str(self.root)!r})),
)
assert status == 0, status
forbidden = []
for name in sys.modules:
    lower = name.lower()
    if (name == 'serial' or name.startswith('serial.') or name == 'cv2'
        or name.startswith('cv2.') or name == 'torch' or name.startswith('torch.')
        or name == 'lerobot' or name.startswith('lerobot.')
        or 'rebot_robot' in lower or lower.startswith('rebotarm_control_py')):
        forbidden.append(name)
assert not forbidden, forbidden
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=repo_root,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_person4_runbook_documents_exact_safe_workflow(self) -> None:
        runbook = Path(__file__).resolve().parents[1] / "PERSON4_RUNBOOK.md"
        self.assertTrue(runbook.is_file(), "Person 4 runbook is missing")
        text = runbook.read_text(encoding="utf-8")
        required = (
            "Pick up one can and place it in the taped sorting zone",
            "shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper",
            "front`, then `side",
            "Gate A",
            "Gate B",
            "Gate C",
            "Gate D",
            "10-15",
            "--episode --retry-on-failure",
            "physical e-stop",
            "10-20%",
            "plane_to_arm",
            "no automatic home",
            "releases torque",
            "runs/policy/reports/",
            "absolute, existing, regular, non-symlink JSONL files",
            "List the shared JSONL path once",
            "MolmoAct2",
            "No automated test performs physical motion",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
