"""Self-contained Markdown and HTML judge evidence for the Query quality gate."""

from __future__ import annotations

from collections import Counter
from html import escape
import json
import math
import statistics
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


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _summary(values: list[float]) -> tuple[int, str, str, str, str]:
    ordered = sorted(values)
    if not ordered:
        return (0, "unavailable", "unavailable", "unavailable", "unavailable")
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return (
        len(ordered),
        _number(min(ordered)),
        _number(statistics.median(ordered)),
        _number(ordered[p95_index]),
        _number(max(ordered)),
    )


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


def _drill_down(payloads: RunPayloads) -> list[tuple[str, str, str, str, str]]:
    inventory = {row.identity.canonical: row for row in payloads.inventory}
    metrics = {row.identity: row for row in payloads.metrics}
    chosen = []
    for role in ("success", "failure"):
        identities = sorted(identity for identity, row in inventory.items() if row.role == role)
        if not identities:
            continue
        identity = max(
            identities,
            key=lambda value: (
                _finite_number(metrics[value].values.get("tracking_rms_deg"))
                if value in metrics and _finite_number(metrics[value].values.get("tracking_rms_deg")) is not None
                else -math.inf,
                value,
            ),
        )
        metric = metrics.get(identity)
        rms = metric.values.get("tracking_rms_deg") if metric else None
        hard = ", ".join(metric.hard_reasons) if metric and metric.hard_reasons else "none"
        chosen.append((role, identity, metric.segment_id if metric else "unavailable", str(rms), hard))
    return chosen


def _joint_rows(payloads: RunPayloads) -> list[tuple[str, str, str, str]]:
    inventory = {row.identity.canonical: row for row in payloads.inventory}
    names = payloads.source_lock.get("joint_names", [])
    if not isinstance(names, list) or len(names) != 7:
        names = [f"joint_{index}" for index in range(7)]
    per_role: dict[str, list[tuple[float, ...]]] = {"success": [], "failure": []}
    unit = "deg"
    for metric in payloads.metrics:
        values = metric.per_joint.get("tracking_mae_deg")
        if values is None:
            values = metric.per_joint.get("tracking_mae_per_joint_deg")
        if values is None or len(values) != 7:
            continue
        numeric = tuple(_finite_number(value) for value in values)
        if any(value is None for value in numeric):
            continue
        role = inventory.get(metric.identity).role if metric.identity in inventory else None
        if role in per_role:
            per_role[role].append(tuple(float(value) for value in numeric if value is not None))
        unit = metric.metric_units.get("tracking_mae_deg", unit)
    rows = []
    for index, name in enumerate(names):
        role_values = {
            role: [values[index] for values in vectors]
            for role, vectors in per_role.items()
        }
        rows.append(
            (
                str(name),
                _number(statistics.median(role_values["success"])) if role_values["success"] else "unavailable",
                _number(statistics.median(role_values["failure"])) if role_values["failure"] else "unavailable",
                unit,
            )
        )
    return rows if any(row[1] != "unavailable" or row[2] != "unavailable" for row in rows) else []


def _distribution_rows(payloads: RunPayloads) -> list[tuple[str, str, int, str, str, str, str, str]]:
    inventory = {row.identity.canonical: row for row in payloads.inventory}
    candidates = (
        "tracking_rms_deg",
        "lag_corrected_rms_deg",
        "action_jerk_p95_normalized_s3",
        "state_jerk_p95_normalized_s3",
        "state_coverage_fraction",
        "camera_front_coverage",
        "camera_side_coverage",
        "state_p95_age_ms",
        "camera_front_p95_age_ms",
        "camera_side_p95_age_ms",
    )
    rows = []
    for metric_name in candidates:
        by_role = {"success": [], "failure": []}
        unit = "unavailable"
        for metric in payloads.metrics:
            role = inventory.get(metric.identity).role if metric.identity in inventory else None
            number = _finite_number(metric.values.get(metric_name))
            if role in by_role and number is not None:
                by_role[role].append(number)
                unit = metric.metric_units.get(metric_name, unit)
        if not by_role["success"] and not by_role["failure"]:
            continue
        for role in ("success", "failure"):
            count, minimum, median, p95, maximum = _summary(by_role[role])
            rows.append((metric_name, role, count, minimum, median, p95, maximum, unit))
    return rows


def _alignment_rows(payloads: RunPayloads) -> list[tuple[str, ...]]:
    inventory = {row.identity.canonical: row for row in payloads.inventory}
    rows = []
    for metric in sorted(payloads.metrics, key=lambda value: value.identity):
        values = metric.values
        evidence = [
            values.get("state_p95_age_ms"),
            values.get("camera_front_p95_age_ms"),
            values.get("camera_side_p95_age_ms"),
            values.get("state_coverage_fraction"),
            values.get("camera_front_coverage"),
            values.get("camera_side_coverage"),
        ]
        if not any(_finite_number(value) is not None for value in evidence):
            continue
        display = lambda value: _number(value) if (value := _finite_number(value)) is not None else "unavailable"
        role = inventory.get(metric.identity).role if metric.identity in inventory else "unknown"
        rows.append((metric.identity, role, metric.segment_id, *(display(value) for value in evidence)))
    return rows


def review_queue_rows(payloads: RunPayloads) -> list[tuple[int, str, str, str, int, str, str, str]]:
    """Return REVIEW episodes ranked by frozen-threshold exceedance, then identity."""

    inventory = {row.identity.canonical: row for row in payloads.inventory}
    metrics = {row.identity: row for row in payloads.metrics}
    thresholds = {threshold.metric: threshold for threshold in payloads.thresholds.thresholds}
    ranked = []
    for verdict in payloads.verdicts:
        if verdict.verdict != "REVIEW":
            continue
        metric = metrics.get(verdict.identity)
        excursions: list[tuple[float, str, float, float]] = []
        for reason in verdict.reason_codes:
            if not reason.startswith("ANOMALY_") or metric is None:
                continue
            name = reason.removeprefix("ANOMALY_").lower()
            threshold = thresholds.get(name)
            value = _finite_number(metric.values.get(name))
            if threshold is None or value is None:
                continue
            scale = max(abs(threshold.value), 1e-12)
            distance = (value - threshold.value) if threshold.direction == "upper" else (threshold.value - value)
            excursions.append((max(0.0, distance / scale), name, value, threshold.value))
        worst = max(excursions, default=(0.0, "unavailable", math.nan, math.nan))
        row = inventory[verdict.identity]
        ranked.append((len(verdict.reason_codes), worst, verdict.identity, row, verdict))
    ranked.sort(key=lambda item: (-item[0], -item[1][0], item[2]))
    rows = []
    for rank, (_, worst, identity, row, verdict) in enumerate(ranked, start=1):
        rows.append(
            (
                rank,
                identity,
                row.role,
                row.task_key,
                len(verdict.reason_codes),
                worst[1],
                _number(worst[2]) if math.isfinite(worst[2]) else "unavailable",
                "|".join(verdict.reason_codes),
            )
        )
    return rows


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
    distributions = _distribution_rows(payloads)
    alignment = _alignment_rows(payloads)
    review_rows = review_queue_rows(payloads)
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

{_markdown_table(("Source role", "Identity", "Rerun segment", "Tracking RMS", "Hard reasons"), drill)}

## Success-versus-failure metric distributions

{_markdown_table(("Metric", "Role", "N", "Min", "Median", "P95", "Max", "Units"), distributions) if distributions else 'No finite comparison distribution was available.'}

## Query alignment and camera evidence

Rows below are emitted only when the corresponding Query-derived aggregate is present; `unavailable` is never inferred as zero.

{_markdown_table(("Identity", "Role", "Rerun segment", "State P95 age ms", "Front P95 age ms", "Side P95 age ms", "State coverage", "Front coverage", "Side coverage"), alignment) if alignment else 'No alignment-age or camera-coverage aggregates were available.'}

## Per-joint tracking

{_markdown_table(("Joint", "Success median MAE", "Failure median MAE", "Units"), joints) if joints else 'No finite per-joint comparison was available.'}

## Reason-code counts

{_markdown_table(("Reason", "Count"), reasons) if reasons else 'No reason codes fired.'}

## Review queue

Verdicts: PASS={verdict_counts['PASS']}, REVIEW={verdict_counts['REVIEW']}, REJECT={verdict_counts['REJECT']}.
{_markdown_table(("Rank", "Identity", "Role", "Task", "Anomalies", "Worst metric", "Value", "Reasons"), review_rows) if review_rows else '- empty'}

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
<h2>Good-versus-failed Query drill-down</h2>{_html_table(("Source role", "Identity", "Rerun segment", "Tracking RMS", "Hard reasons"), drill)}
<h2>Success-versus-failure metric distributions</h2>{_html_table(("Metric", "Role", "N", "Min", "Median", "P95", "Max", "Units"), distributions) if distributions else '<p>No finite comparison distribution was available.</p>'}
<h2>Query alignment and camera evidence</h2><p>Rows are shown only for Query-derived aggregates that exist; unavailable values are not inferred as zero.</p>{_html_table(("Identity", "Role", "Rerun segment", "State P95 age ms", "Front P95 age ms", "Side P95 age ms", "State coverage", "Front coverage", "Side coverage"), alignment) if alignment else '<p>No alignment-age or camera-coverage aggregates were available.</p>'}
<h2>Per-joint tracking</h2>{_html_table(("Joint", "Success median MAE", "Failure median MAE", "Units"), joints) if joints else '<p>No finite per-joint comparison was available.</p>'}
<h2>Reason-code counts</h2>{_html_table(("Reason", "Count"), reasons) if reasons else '<p>No reason codes fired.</p>'}
<h2>Review queue</h2><p>PASS={verdict_counts['PASS']}, REVIEW={verdict_counts['REVIEW']}, REJECT={verdict_counts['REJECT']}. Ranking is deterministic: anomaly count, normalized frozen-threshold excursion, then identity.</p>{_html_table(("Rank", "Identity", "Role", "Task", "Anomalies", "Worst metric", "Value", "Reasons"), review_rows) if review_rows else '<p>empty</p>'}
<h2>Training preparation</h2><p>Derivative repository: <code>{escape(derivative)}</code></p><p>Selection payload digest: <code>{escape(str(payloads.selection_manifest.get('selection_payload_digest', '')))}</code></p><h3>Selected identities ({len(selected)})</h3>{list_html(selected or ['none'])}<h3>Excluded identities ({len(excluded)})</h3>{excluded_html}
<h2>Honest limitation</h2><p class="notice">{escape(_LIMITATION)}</p>
</body></html>"""
    return html, markdown
