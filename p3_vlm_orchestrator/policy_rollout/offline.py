"""Offline-only checkpoint evaluation over finalized LeRobot datasets."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, TextIO

import numpy as np

from p3_vlm_orchestrator.policy_rollout.lerobot_policy import LeRobotPolicyAdapter
from rebot_operator_kit.rollout.checkpoint import CheckpointBundle
from rebot_operator_kit.rollout.contracts import RolloutObservation


@dataclass(frozen=True)
class OfflinePrediction:
    episode_index: int
    sample_index: int
    shape: tuple[int, ...]
    minimum: float
    maximum: float
    latency_s: float


def _cpu_numpy(value: Any) -> np.ndarray:
    converted = value
    for method_name in ("detach", "cpu"):
        method = getattr(converted, method_name, None)
        if callable(method):
            converted = method()
    to_numpy = getattr(converted, "numpy", None)
    if callable(to_numpy):
        converted = to_numpy()
    return np.array(converted, copy=True)


def _scalar(value: Any, *, label: str) -> float:
    array = _cpu_numpy(value)
    if array.size != 1:
        raise ValueError(f"dataset {label} must be scalar; received {array.shape}")
    try:
        return float(array.reshape(()).item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dataset {label} must be numeric") from exc


def _camera_array(value: Any, *, key: str) -> np.ndarray:
    array = _cpu_numpy(value)
    if array.ndim != 3:
        raise ValueError(f"dataset {key} must be a three-dimensional image")

    channels = (1, 3, 4)
    if array.shape[-1] in channels and array.shape[0] not in channels:
        image = array
    elif array.shape[0] in channels:
        image = np.moveaxis(array, 0, -1)
    elif array.shape[-1] in channels:
        image = array
    else:
        raise ValueError(f"dataset {key} must be CHW or HWC")

    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError(f"dataset {key} contains nonfinite pixels")
        minimum = float(image.min())
        maximum = float(image.max())
        if minimum >= 0.0 and maximum <= 1.0:
            image = np.rint(image * 255.0)
        elif minimum >= 0.0 and maximum <= 255.0:
            image = np.rint(image)
        else:
            raise ValueError(f"dataset {key} pixels must be in [0, 1] or [0, 255]")
    elif np.issubdtype(image.dtype, np.integer):
        if image.size and (int(image.min()) < 0 or int(image.max()) > 255):
            raise ValueError(f"dataset {key} pixels must be in [0, 255]")
    else:
        raise ValueError(f"dataset {key} must contain numeric pixels")
    return np.array(image, dtype=np.uint8, copy=True)


def _state_array(value: Any) -> np.ndarray:
    try:
        state = np.array(_cpu_numpy(value), dtype=np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("dataset observation.state must be numeric") from exc
    if state.shape != (7,):
        raise ValueError(
            "dataset observation.state must have shape (7,); "
            f"received {state.shape}"
        )
    if not np.isfinite(state).all():
        raise ValueError("dataset observation.state contains nonfinite values")
    return state


def _load_lerobot_dataset(root: Path) -> Any:
    """Load only a complete local dataset; LeRobot remains an optional import."""

    info_path = root / "meta" / "info.json"
    if not root.is_dir() or not info_path.is_file():
        raise ValueError(f"finalized LeRobot dataset root is missing: {root}")
    required_paths = (
        root / "meta" / "stats.json",
        root / "meta" / "tasks.parquet",
        root / "meta" / "episodes",
        root / "data",
    )
    missing = [str(path.relative_to(root)) for path in required_paths if not path.exists()]
    if missing:
        raise ValueError(
            "finalized LeRobot dataset is incomplete; missing " + ", ".join(missing)
        )
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"finalized LeRobot dataset info is unreadable: {exc}") from exc
    if not isinstance(info, dict):
        raise ValueError("finalized LeRobot dataset info must be an object")
    if not isinstance(info.get("total_episodes"), int) or info["total_episodes"] < 1:
        raise ValueError("finalized LeRobot dataset contains no episodes")
    if not isinstance(info.get("total_frames"), int) or info["total_frames"] < 1:
        raise ValueError("finalized LeRobot dataset contains no frames")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise RuntimeError(
            "Offline evaluation requires the same LeRobot environment used to "
            f"record the dataset. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    repo_id = info.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id.strip():
        repo_id = f"local/{root.name}"
    return LeRobotDataset(repo_id=repo_id, root=root)


def evaluate_checkpoint(
    bundle: CheckpointBundle,
    dataset_root: Path,
    *,
    episodes: int,
    device: str = "cpu",
    dataset_loader: Callable[[Path], Any] | None = None,
    adapter_factory: Callable[[CheckpointBundle, str], Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    output: TextIO | None = None,
) -> list[OfflinePrediction]:
    """Evaluate every sample belonging to exactly N distinct episodes."""

    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    root = Path(dataset_root).expanduser().resolve()
    load_dataset = dataset_loader or _load_lerobot_dataset
    make_adapter = adapter_factory or LeRobotPolicyAdapter.from_checkpoint
    stream = output or sys.stdout

    dataset = load_dataset(root)
    adapter = make_adapter(bundle, device)
    selected_episodes: list[int] = []
    selected_set: set[int] = set()
    results: list[OfflinePrediction] = []

    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        if not isinstance(sample, dict):
            raise ValueError(f"dataset sample {sample_index} must be a mapping")
        if "episode_index" not in sample:
            raise ValueError(f"dataset sample {sample_index} has no episode_index")
        episode_index = int(
            _scalar(sample["episode_index"], label="episode_index")
        )
        if episode_index not in selected_set:
            if len(selected_episodes) >= episodes:
                continue
            selected_episodes.append(episode_index)
            selected_set.add(episode_index)

        task = sample.get("task")
        if task != bundle.task:
            raise ValueError(
                "dataset task mismatch at "
                f"episode {episode_index}, sample {sample_index}: "
                f"expected {bundle.task!r}, received {task!r}"
            )
        for key in (
            "observation.images.front",
            "observation.images.side",
            "observation.state",
        ):
            if key not in sample:
                raise ValueError(f"dataset sample {sample_index} is missing {key}")

        captured_s = (
            _scalar(sample["timestamp"], label="timestamp")
            if "timestamp" in sample
            else 0.0
        )
        observation = RolloutObservation(
            front=_camera_array(
                sample["observation.images.front"],
                key="observation.images.front",
            ),
            side=_camera_array(
                sample["observation.images.side"],
                key="observation.images.side",
            ),
            state_deg=_state_array(sample["observation.state"]),
            task=bundle.task,
            captured_monotonic_s=captured_s,
        )

        started = clock()
        prediction = np.asarray(adapter.predict(observation), dtype=np.float64)
        finished = clock()
        if prediction.ndim != 2 or prediction.shape[0] < 1 or prediction.shape[1] != 7:
            raise ValueError(
                "offline policy prediction must have shape [steps, 7]; "
                f"received {prediction.shape}"
            )
        if not np.isfinite(prediction).all():
            raise ValueError("offline policy prediction contains nonfinite values")
        result = OfflinePrediction(
            episode_index=episode_index,
            sample_index=sample_index,
            shape=tuple(prediction.shape),
            minimum=float(prediction.min()),
            maximum=float(prediction.max()),
            latency_s=max(0.0, finished - started),
        )
        results.append(result)
        print(
            f"episode={result.episode_index} sample={result.sample_index} "
            f"shape={result.shape} min={result.minimum:.6f} "
            f"max={result.maximum:.6f} latency_ms={result.latency_s * 1000:.3f}",
            file=stream,
        )

    if len(selected_episodes) != episodes:
        raise ValueError(
            f"requested {episodes} distinct episodes but found "
            f"{len(selected_episodes)} in the dataset"
        )

    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a saved LeRobot checkpoint over held-out recorded episodes."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    bundle = CheckpointBundle.load(args.checkpoint)
    results = evaluate_checkpoint(
        bundle,
        args.dataset,
        episodes=args.episodes,
        device=args.device,
    )
    if not results:
        raise RuntimeError("No samples were found in the selected dataset episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
