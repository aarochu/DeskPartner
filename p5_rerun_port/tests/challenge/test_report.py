from __future__ import annotations

from p5_rerun_port.challenge.artifacts import RunPayloads, build_selection_manifest
from p5_rerun_port.challenge.evaluate import EvaluationReport
from p5_rerun_port.challenge.metrics import EpisodeMetrics
from p5_rerun_port.challenge.models import EpisodeIdentity, InventoryRow
from p5_rerun_port.challenge.quality import EpisodeVerdict, ThresholdSnapshot
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
    snapshot = ThresholdSnapshot(tuple(row.identity.canonical for row in inventory[:61]), tuple(row.identity.canonical for row in inventory[61:77]), (), "threshold")
    verdicts = tuple(EpisodeVerdict(row.identity.canonical, "PASS" if row.role == "success" else "REJECT", () if row.role == "success" else ("ACTION_MISSING",), f"m-{i}", "threshold", f"v-{i}") for i, row in enumerate(inventory))
    metrics = tuple(EpisodeMetrics(row.identity.canonical, f"s-{i}", row.task_key, "1", 10, {"tracking_rms_deg": float(i), "camera_front_coverage": 1.0}, {"tracking_mae_per_joint_deg": tuple(float(j) for j in range(7))}, {"tracking_rms_deg": "deg"}, ()) for i, row in enumerate(inventory))
    evaluation = EvaluationReport(tuple(row.identity.canonical for row in inventory[61:]), {"tp": 24, "fp": 1, "tn": 15, "fn": 1}, .96, .96, .96, .0625, {"single_can": {"recall": .95}, "two_can": {"recall": 1.0}}, {"failed": {"recall": .9}}, {"dropped_object": {"recall": .5}}, (inventory[61].identity.canonical,), (inventory[-1].identity.canonical,))
    source_lock = {"sources": [{"repo_id": repo, "resolved_sha": sha, "role": role} for repo, sha, role, _, _ in repositories], "destination_repo": "Cornerf/rebot-cansort-rerun-curated"}
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
