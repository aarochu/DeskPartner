from __future__ import annotations

from p5_rerun_port.challenge.artifacts import RunPayloads, build_selection_manifest
from p5_rerun_port.challenge.evaluate import EvaluationReport
from p5_rerun_port.challenge.metrics import EpisodeMetrics
from p5_rerun_port.challenge.models import EpisodeIdentity, InventoryRow
from p5_rerun_port.challenge.quality import EpisodeVerdict, MetricThreshold, ThresholdSnapshot
from p5_rerun_port.challenge.report import render_report


def test_report_is_self_contained_complete_and_truthful() -> None:
    repositories = (
        ("Cornerf/one", "1" * 40, "success", "single_can", 52),
        ("Cornerf/two", "2" * 40, "success", "two_can", 25),
        ("Cornerf/failed-one", "3" * 40, "failure", "single_can", 19),
        ("Cornerf/failed-two", "4" * 40, "failure", "two_can", 6),
    )
    rows = []
    for repo, sha, role, task, count in repositories:
        for index in range(count):
            key = str(index) if role == "success" else f"attempt-{index}"
            rows.append(InventoryRow(EpisodeIdentity(repo, sha, key), role, task, "source", index if role == "success" else None, key if role == "failure" else None, 10, "2026-01-01T00:00:00Z"))
    inventory = tuple(rows)
    threshold = MetricThreshold("tracking_rms_deg", "deg", "upper", 0, 1, .5, .1, 1, None, "linear", 61, 1)
    snapshot = ThresholdSnapshot(tuple(row.identity.canonical for row in inventory[:61]), tuple(row.identity.canonical for row in inventory[61:77]), (threshold,), "threshold")
    verdicts = []
    for i, row in enumerate(inventory):
        if i == 0:
            verdict, reasons = "REVIEW", ("ANOMALY_TRACKING_RMS_DEG",)
        elif i == 1:
            verdict, reasons = "REVIEW", ("ANOMALY_TRACKING_RMS_DEG", "ANOMALY_STATE_P95_AGE_MS")
        else:
            verdict, reasons = ("PASS", ()) if row.role == "success" else ("REJECT", ("ACTION_MISSING",))
        verdicts.append(EpisodeVerdict(row.identity.canonical, verdict, reasons, f"m-{i}", "threshold", f"v-{i}"))
    verdicts = tuple(verdicts)
    metrics = tuple(
        EpisodeMetrics(
            row.identity.canonical,
            f"s-{i}",
            row.task_key,
            "1",
            10,
            {
                "tracking_rms_deg": float(i + 1),
                "state_p95_age_ms": float(i % 4),
                "camera_front_p95_age_ms": float(i % 3),
                "camera_front_coverage": 1.0,
                "camera_side_coverage": .9,
                "state_coverage_fraction": .95,
            },
            {"tracking_mae_deg": tuple(float(j + i) for j in range(7))},
            {
                "tracking_rms_deg": "deg",
                "tracking_mae_deg": "deg",
                "state_p95_age_ms": "ms",
                "camera_front_p95_age_ms": "ms",
                "camera_front_coverage": "ratio",
                "camera_side_coverage": "ratio",
                "state_coverage_fraction": "ratio",
            },
            (),
        )
        for i, row in enumerate(inventory)
    )
    evaluation = EvaluationReport(tuple(row.identity.canonical for row in inventory[61:]), {"tp": 24, "fp": 1, "tn": 15, "fn": 1}, .96, .96, .96, .0625, {"single_can": {"recall": .95}, "two_can": {"recall": 1.0}}, {"failed": {"recall": .9}}, {"dropped_object": {"recall": .5}}, (inventory[61].identity.canonical,), (inventory[-1].identity.canonical,))
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]
    source_lock = {"sources": [{"repo_id": repo, "resolved_sha": sha, "role": role} for repo, sha, role, _, _ in repositories], "destination_repo": "Cornerf/rebot-cansort-rerun-curated", "joint_names": joint_names, "camera_keys": ["front", "side"]}
    manifest = build_selection_manifest(inventory, verdicts, snapshot, source_lock, "2026-07-19T00:00:00Z")

    html, markdown = render_report(RunPayloads(source_lock, inventory, metrics, snapshot, verdicts, evaluation, manifest))

    combined = html + markdown
    for verb in ("Inspect", "Align", "Filter", "Compare", "Transform", "Evaluate", "Prepare"):
        assert verb in combined
    for text in ("102", "77", "25", "precision", "recall", "false positive", "false negative", "review queue", "Cornerf/rebot-cansort-rerun-curated", "semantic"):
        assert text.lower() in combined.lower()
    for repo, sha, *_ in repositories:
        assert repo in combined and sha in combined
    assert "<script" not in html.lower()
    assert "stylesheet" not in html.lower()
    assert "url(" not in html.lower()
    for joint in joint_names:
        assert joint in combined
    assert "Success-versus-failure metric distributions" in combined
    assert "State P95 age ms" in combined
    assert "Front P95 age ms" in combined
    assert "Rerun segment" in combined
    assert "normalized frozen-threshold excursion" in html
    assert html.index(inventory[1].identity.canonical, html.index("<h2>Review queue")) < html.index(
        inventory[0].identity.canonical, html.index("<h2>Review queue")
    )
