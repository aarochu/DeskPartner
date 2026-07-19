"""Self-contained Markdown and HTML judge evidence for the Query quality gate."""

from __future__ import annotations

from collections import Counter
from html import escape
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .artifacts import RunPayloads


_CHALLENGE_VERBS = ("Inspect", "Align", "Filter", "Compare", "Transform", "Evaluate", "Prepare")
_LIMITATION = (
    "Telemetry detects structural and motion-quality problems; it cannot prove every semantic "
    "task failure, such as a visually dropped object. Operator and video evidence remain authoritative."
)


def _number(value: float) -> str:
    return f"{value:.4f}"


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def _html_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    head = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _source_rows(payloads: RunPayloads) -> list[tuple[str, str, str]]:
    rows = []
    sources = payloads.source_lock.get("sources", [])
    if isinstance(sources, list):
        for source in sorted(sources, key=lambda item: str(item.get("repo_id", ""))):
            repo = str(source.get("repo_id", ""))
            revision = str(source.get("resolved_sha", source.get("configured_revision", "")))
            role = str(source.get("role", ""))
            rows.append((repo, revision, role))
    return rows


def _drill_down(payloads: RunPayloads) -> list[tuple[str, str, str, str]]:
    inventory = {row.identity.canonical: row for row in payloads.inventory}
    metrics = {row.identity: row for row in payloads.metrics}
    chosen = []
    for role in ("success", "failure"):
        identities = sorted(identity for identity, row in inventory.items() if row.role == role)
        if not identities:
            continue
        identity = identities[0]
        metric = metrics.get(identity)
        rms = metric.values.get("tracking_rms_deg") if metric else None
        hard = ", ".join(metric.hard_reasons) if metric and metric.hard_reasons else "none"
        chosen.append((role, identity, str(rms), hard))
    return chosen


def _joint_rows(payloads: RunPayloads) -> list[tuple[int, str]]:
    for metric in sorted(payloads.metrics, key=lambda row: row.identity):
        for name in ("tracking_mae_per_joint_deg", "tracking_mae_per_joint"):
            values = metric.per_joint.get(name)
            if values:
                return [(index, _number(float(value))) for index, value in enumerate(values)]
    return []


def _reason_rows(payloads: RunPayloads) -> list[tuple[str, int]]:
    counts = Counter(reason for verdict in payloads.verdicts for reason in verdict.reason_codes)
    return sorted(counts.items())


def _svg_confusion(confusion: dict[str, int]) -> str:
    values = [(key.upper(), int(confusion.get(key, 0))) for key in ("tp", "fp", "tn", "fn")]
    maximum = max((value for _, value in values), default=1) or 1
    bars = []
    for index, (label, value) in enumerate(values):
        height = int(90 * value / maximum)
        x = 20 + index * 70
        bars.append(f'<rect x="{x}" y="{105-height}" width="42" height="{height}" fill="#4b66d1"/><text x="{x+21}" y="120" text-anchor="middle">{label}</text><text x="{x+21}" y="{99-height}" text-anchor="middle">{value}</text>')
    return '<svg viewBox="0 0 300 130" role="img" aria-label="confusion matrix bars">' + "".join(bars) + "</svg>"


def render_report(payloads: RunPayloads) -> tuple[str, str]:
    """Render deterministic, transparent evidence with no external assets."""

    counts = Counter(row.role for row in payloads.inventory)
    verdict_counts = Counter(verdict.verdict for verdict in payloads.verdicts)
    total = len(payloads.inventory)
    sources = _source_rows(payloads)
    evaluation = payloads.evaluation
    derivative = str(payloads.selection_manifest.get("destination_repo", "Cornerf/rebot-cansort-rerun-curated"))
    review_identities = sorted(verdict.identity for verdict in payloads.verdicts if verdict.verdict == "REVIEW")
    challenge_rows = [
        ("Inspect", "Discover locked segments, entities, dimensions, and cameras."),
        ("Align", "Resolve action, state, and camera timestamps inside one segment."),
        ("Filter", "Select by task, verdict, reason, revision, and revealed label."),
        ("Compare", "Contrast tracking, lag, camera, success, and failure distributions."),
        ("Transform", "Canonicalize native failed recordings through Query API rows."),
        ("Evaluate", "Score held-out successes and failures after thresholds freeze."),
        ("Prepare", "Checksum the PASS-success manifest used for derivative training data."),
    ]
    source_md = [
        (f"[{repo}](https://huggingface.co/datasets/{repo}/tree/{revision})", revision, role)
        for repo, revision, role in sources
    ]
    metrics_rows = [
        ("precision", _number(evaluation.precision)),
        ("recall", _number(evaluation.recall)),
        ("F1", _number(evaluation.f1)),
        ("false rejection rate", _number(evaluation.false_rejection_rate)),
    ]
    false_positive = list(evaluation.false_positive_identities) or ["none"]
    false_negative = list(evaluation.false_negative_identities) or ["none"]
    drill = _drill_down(payloads)
    joints = _joint_rows(payloads)
    reasons = _reason_rows(payloads)
    selected = payloads.selection_manifest.get("selected_identities", [])
    excluded = payloads.selection_manifest.get("excluded_items", [])

    markdown = f"""# Rerun Query-to-Train Quality Gate

## Inventory and locked sources

This run inspects **{total} total items**: **{counts['success']} successes** and **{counts['failure']} failures**. Thresholds use {len(payloads.thresholds.calibration_identities)} calibration successes; evaluation contains {len(evaluation.evaluated_identities)} held-out items.

{_markdown_table(("Source", "SHA", "Role"), source_md)}

## Query API challenge coverage

{_markdown_table(("Verb", "Evidence"), challenge_rows)}

## Held-out evaluation

{_markdown_table(("Metric", "Value"), metrics_rows)}

Confusion matrix: TP={evaluation.confusion.get('tp', 0)}, FP={evaluation.confusion.get('fp', 0)}, TN={evaluation.confusion.get('tn', 0)}, FN={evaluation.confusion.get('fn', 0)}.

False positive identities:
{chr(10).join(f'- `{identity}`' for identity in false_positive)}

False negative identities:
{chr(10).join(f'- `{identity}`' for identity in false_negative)}

Per-task evaluation: `{json.dumps(evaluation.by_task, sort_keys=True)}`

Per-failure-label evaluation: `{json.dumps(evaluation.by_failure_label, sort_keys=True)}`

## Good-versus-failed Query drill-down

{_markdown_table(("Source role", "Identity", "Tracking RMS", "Hard reasons"), drill)}

## Per-joint tracking

{_markdown_table(("Joint index", "Tracking MAE"), joints) if joints else 'No per-joint row was available.'}

## Reason-code counts

{_markdown_table(("Reason", "Count"), reasons) if reasons else 'No reason codes fired.'}

## Review queue

Verdicts: PASS={verdict_counts['PASS']}, REVIEW={verdict_counts['REVIEW']}, REJECT={verdict_counts['REJECT']}.
{chr(10).join(f'- `{identity}`' for identity in review_identities) if review_identities else '- empty'}

## Training preparation

Derivative repository: `{derivative}`

Selection payload digest: `{payloads.selection_manifest.get('selection_payload_digest', '')}`

Selected identities ({len(selected)}):
{chr(10).join(f'- `{identity}`' for identity in selected) if selected else '- none'}

Excluded identities ({len(excluded)}):
{chr(10).join(f'- `{item.get("identity", "")}` — {item.get("verdict", "")} — {", ".join(item.get("reason_codes", [])) or "no anomaly reason; excluded by source authority"}' for item in excluded) if excluded else '- none'}

## Honest limitation

{_LIMITATION}
"""

    source_html_rows = [
        (f'<a href="https://huggingface.co/datasets/{escape(repo)}/tree/{escape(revision)}">{escape(repo)}</a>', revision, role)
        for repo, revision, role in sources
    ]
    # Source links are the only pre-rendered cells; build this table explicitly.
    source_html = "<table><thead><tr><th>Source</th><th>SHA</th><th>Role</th></tr></thead><tbody>" + "".join(
        f"<tr><td>{link}</td><td>{escape(revision)}</td><td>{escape(role)}</td></tr>"
        for link, revision, role in source_html_rows
    ) + "</tbody></table>"
    list_html = lambda values: "<ul>" + "".join(f"<li><code>{escape(str(value))}</code></li>" for value in values) + "</ul>"
    excluded_html = list_html(
        f"{item.get('identity', '')} — {item.get('verdict', '')} — {', '.join(item.get('reason_codes', [])) or 'excluded by source authority'}"
        for item in excluded
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rerun Query Quality Gate</title><style>body{{font:15px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#172033}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #c9d0dc;padding:.45rem;text-align:left;vertical-align:top}}th{{background:#eef1f7}}code{{overflow-wrap:anywhere}}svg{{max-width:460px;background:#f7f8fb;border:1px solid #d7dce6}}.notice{{border-left:5px solid #c25616;padding:.75rem;background:#fff5e9}}</style></head><body>
<h1>Rerun Query-to-Train Quality Gate</h1>
<h2>Inventory and locked sources</h2><p>This run inspects <strong>{total} total items</strong>: <strong>{counts['success']} successes</strong> and <strong>{counts['failure']} failures</strong>.</p>{source_html}
<h2>Query API challenge coverage</h2>{_html_table(("Verb", "Evidence"), challenge_rows)}
<h2>Held-out evaluation</h2>{_html_table(("Metric", "Value"), metrics_rows)}<p>Confusion matrix: TP={evaluation.confusion.get('tp', 0)}, FP={evaluation.confusion.get('fp', 0)}, TN={evaluation.confusion.get('tn', 0)}, FN={evaluation.confusion.get('fn', 0)}.</p>{_svg_confusion(evaluation.confusion)}
<h3>False positive identities</h3>{list_html(false_positive)}<h3>False negative identities</h3>{list_html(false_negative)}
<h3>Per-task and failure-label evaluation</h3><pre>{escape(json.dumps({'by_task': evaluation.by_task, 'by_failure_label': evaluation.by_failure_label}, sort_keys=True, indent=2))}</pre>
<h2>Good-versus-failed Query drill-down</h2>{_html_table(("Source role", "Identity", "Tracking RMS", "Hard reasons"), drill)}
<h2>Per-joint tracking</h2>{_html_table(("Joint index", "Tracking MAE"), joints) if joints else '<p>No per-joint row was available.</p>'}
<h2>Reason-code counts</h2>{_html_table(("Reason", "Count"), reasons) if reasons else '<p>No reason codes fired.</p>'}
<h2>Review queue</h2><p>PASS={verdict_counts['PASS']}, REVIEW={verdict_counts['REVIEW']}, REJECT={verdict_counts['REJECT']}.</p>{list_html(review_identities or ['empty'])}
<h2>Training preparation</h2><p>Derivative repository: <code>{escape(derivative)}</code></p><p>Selection payload digest: <code>{escape(str(payloads.selection_manifest.get('selection_payload_digest', '')))}</code></p><h3>Selected identities ({len(selected)})</h3>{list_html(selected or ['none'])}<h3>Excluded identities ({len(excluded)})</h3>{excluded_html}
<h2>Honest limitation</h2><p class="notice">{escape(_LIMITATION)}</p>
</body></html>"""
    return html, markdown
