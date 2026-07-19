"""Shared payload type kept separate to avoid report/artifact import cycles."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .evaluate import EvaluationReport
from .metrics import EpisodeMetrics
from .models import InventoryRow
from .quality import EpisodeVerdict, ThresholdSnapshot

@dataclass(frozen=True)
class RunPayloads:
    source_lock: dict
    inventory: tuple[InventoryRow, ...]
    metrics: tuple[EpisodeMetrics, ...]
    thresholds: ThresholdSnapshot
    verdicts: tuple[EpisodeVerdict, ...]
    evaluation: EvaluationReport
    selection_manifest: dict
