#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def shape_of(value):
    return tuple(getattr(value, "shape", np.asarray(value).shape))


def array_of(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def canonical_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def load_profile(path: Path):
    try:
        profile = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"training profile cannot be read: {exc}") from exc
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise SystemExit("unsupported training profile schema")
    coordinates = profile.get("coordinate_contract")
    if not isinstance(coordinates, dict) or coordinates.get("action_dimension") != 7:
        raise SystemExit("training profile must define a native seven-action contract")
    joints = coordinates.get("joints")
    if not isinstance(joints, list) or len(joints) != 7:
        raise SystemExit("training profile must define exactly seven joints")
    names = [joint.get("feature") for joint in joints if isinstance(joint, dict)]
    limits = [joint.get("soft_limit_degrees") for joint in joints if isinstance(joint, dict)]
    if len(names) != 7 or len(limits) != 7 or any(
        not isinstance(limit, list) or len(limit) != 2 for limit in limits
    ):
        raise SystemExit("training profile joint features or limits are invalid")
    return profile, canonical_digest(profile), names, np.asarray(limits, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--minimum-episodes", type=int, default=5)
    parser.add_argument("--max-action-state-delta", type=float, default=None)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "training_profile.json",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    profile, profile_digest, expected_names, limits = load_profile(args.profile)
    if args.max_action_state_delta is None:
        try:
            args.max_action_state_delta = (
                float(profile["collection_defaults"]["max_step"]) + 0.5
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                "training profile collection_defaults.max_step is invalid"
            ) from exc
    sidecar_path = args.root / "meta" / "rebot_training_profile.json"
    if not sidecar_path.is_file():
        raise SystemExit("dataset is missing meta/rebot_training_profile.json")
    try:
        sidecar = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("dataset training-profile sidecar is unreadable") from exc
    if not isinstance(sidecar, dict):
        raise SystemExit("dataset training-profile sidecar root is invalid")
    profile_snapshot = sidecar.get("profile_snapshot")
    collection_contract = sidecar.get("collection_contract")
    collection_digest = sidecar.get("collection_contract_digest")
    if not isinstance(profile_snapshot, dict) or not isinstance(collection_contract, dict):
        raise SystemExit("dataset profile or collection-contract snapshot is missing")
    if (
        sidecar.get("training_profile_digest") != profile_digest
        or canonical_digest(profile_snapshot) != sidecar.get("training_profile_digest")
        or profile_snapshot != profile
    ):
        raise SystemExit("dataset was collected under a different calibration/training profile")
    if canonical_digest(collection_contract) != collection_digest:
        raise SystemExit("dataset collection-contract snapshot has been modified")
    if (
        sidecar.get("training_profile_id") != profile_snapshot.get("profile_id")
        or sidecar.get("training_profile_version") != profile_snapshot.get("profile_version")
    ):
        raise SystemExit("dataset profile identity does not match its snapshot")

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    if collection_contract.get("repo_id") != args.repo_id:
        raise SystemExit("dataset repo_id does not match the locked collection contract")
    if collection_contract.get("dataset") != args.root.name:
        raise SystemExit("dataset folder name does not match the locked collection contract")
    if collection_contract.get("fps") != dataset.fps:
        raise SystemExit("dataset FPS does not match the locked collection contract")
    keys = set(dataset.features)
    required = {
        "observation.images.front",
        "observation.images.side",
        "observation.state",
        "action",
    }
    missing = sorted(required - keys)
    if missing:
        raise SystemExit(f"missing required features: {missing}")

    forbidden = [
        key
        for key in keys
        if any(token in key.lower() for token in ("left_arm", "right_arm", "bimanual"))
    ]
    if forbidden:
        raise SystemExit(f"unexpected multi-arm features: {forbidden}")

    if dataset.num_episodes < args.minimum_episodes:
        raise SystemExit(
            f"expected at least {args.minimum_episodes} episodes, found {dataset.num_episodes}"
        )
    if dataset.num_frames <= 0:
        raise SystemExit("dataset has no frames")

    episode_lengths = [int(episode.get("length", 0)) for episode in (dataset.meta.episodes or [])]
    if len(episode_lengths) != dataset.num_episodes or any(
        length < dataset.fps for length in episode_lengths
    ):
        raise SystemExit(
            "every finalized episode must contain at least one second of frames; "
            f"lengths={episode_lengths}"
        )

    stats = dataset.meta.stats if isinstance(dataset.meta.stats, dict) else {}
    missing_quantiles = {
        key: sorted({"q01", "q99"} - set(stats.get(key, {})))
        for key in ("observation.state", "action")
        if not isinstance(stats.get(key), dict)
        or not {"q01", "q99"}.issubset(stats[key])
    }
    if missing_quantiles:
        raise SystemExit(
            "MolmoAct2 quantile statistics are missing: "
            f"{missing_quantiles}. Re-export or locally augment the dataset before training."
        )
    for key in ("observation.state", "action"):
        q01 = array_of(stats[key]["q01"]).astype(np.float32).reshape(-1)
        q99 = array_of(stats[key]["q99"]).astype(np.float32).reshape(-1)
        if q01.shape != (7,) or q99.shape != (7,):
            raise SystemExit(
                f"{key} quantiles must each contain seven values, got {q01.shape}/{q99.shape}"
            )
        if not np.all(np.isfinite(q01)) or not np.all(np.isfinite(q99)):
            raise SystemExit(f"{key} quantiles contain non-finite values")
        if np.any(q01 > q99):
            raise SystemExit(f"{key} q01 exceeds q99 for at least one joint")

    action_feature = dataset.features.get("action", {})
    state_feature = dataset.features.get("observation.state", {})
    if action_feature.get("names") != expected_names:
        raise SystemExit(f"unexpected action joint order: {action_feature.get('names')}")
    if state_feature.get("names") != expected_names:
        raise SystemExit(f"unexpected state joint order: {state_feature.get('names')}")

    sample_count = min(dataset.num_frames, 257)
    sample_indices = sorted(
        {int(round(value)) for value in np.linspace(0, dataset.num_frames - 1, sample_count)}
    )
    maximum_delta = 0.0
    task_values: set[str] = set()
    expected_camera_shapes: dict[str, tuple[int, ...]] = {}
    minimum_camera_brightness = {"front": float("inf"), "side": float("inf")}
    minimum_camera_contrast = {"front": float("inf"), "side": float("inf")}
    for index in sample_indices:
        sample = dataset[index]
        front_shape = shape_of(sample["observation.images.front"])
        side_shape = shape_of(sample["observation.images.side"])
        state_shape = shape_of(sample["observation.state"])
        action_shape = shape_of(sample["action"])
        for label, image_shape in (("front", front_shape), ("side", side_shape)):
            if len(image_shape) != 3 or image_shape[0] != 3 or min(image_shape[1:]) <= 0:
                raise SystemExit(f"frame {index}: invalid {label} camera shape {image_shape}")
            if label in expected_camera_shapes and image_shape != expected_camera_shapes[label]:
                raise SystemExit(
                    f"frame {index}: {label} camera shape changed from "
                    f"{expected_camera_shapes[label]} to {image_shape}"
                )
            expected_camera_shapes[label] = image_shape
            image = array_of(sample[f"observation.images.{label}"]).astype(np.float32)
            if image.size and float(np.nanmax(image)) <= 1.5:
                image *= 255.0
            brightness = float(np.nanmean(image))
            contrast = float(np.nanstd(image))
            minimum_camera_brightness[label] = min(minimum_camera_brightness[label], brightness)
            minimum_camera_contrast[label] = min(minimum_camera_contrast[label], contrast)
            if not np.isfinite(brightness) or not np.isfinite(contrast):
                raise SystemExit(f"frame {index}: {label} camera contains non-finite pixels")
            if brightness < 4.0 or contrast < 1.5:
                raise SystemExit(
                    f"frame {index}: {label} camera is dark/covered "
                    f"(brightness={brightness:.2f}, contrast={contrast:.2f})"
                )
        if state_shape[-1] != 7 or action_shape[-1] != 7:
            raise SystemExit(
                f"frame {index}: expected seven joints, state={state_shape}, action={action_shape}"
            )
        state = array_of(sample["observation.state"]).astype(np.float32)
        action = array_of(sample["action"]).astype(np.float32)
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
            raise SystemExit(f"frame {index}: state/action contains a non-finite value")
        if np.any(action < limits[:, 0] - 0.1) or np.any(action > limits[:, 1] + 0.1):
            raise SystemExit(f"frame {index}: follower-space action exceeds ReBot joint limits: {action}")
        delta = float(np.max(np.abs(action - state)))
        maximum_delta = max(maximum_delta, delta)
        if delta > args.max_action_state_delta:
            raise SystemExit(
                f"frame {index}: action/state delta {delta:.3f}° exceeds "
                f"{args.max_action_state_delta:.3f}°; actions may be in leader coordinates"
            )
        task = str(sample.get("task", "")).strip()
        if not task:
            raise SystemExit(f"frame {index}: task instruction is empty")
        task_values.add(task)

    if len(task_values) != 1:
        raise SystemExit(f"dataset mixes task instructions: {sorted(task_values)}")
    if next(iter(task_values)) != collection_contract.get("task"):
        raise SystemExit("dataset task does not match the locked collection contract")

    report = {
        "passed": True,
        "repo_id": args.repo_id,
        "root": str(args.root),
        "episodes": dataset.num_episodes,
        "frames": dataset.num_frames,
        "fps": dataset.fps,
        "camera_keys": list(dataset.meta.camera_keys),
        "camera_shapes": {
            "front": list(front_shape),
            "side": list(side_shape),
        },
        "state_shape": list(state_shape),
        "action_shape": list(action_shape),
        "joint_names": expected_names,
        "task": next(iter(task_values)),
        "sampled_frames": len(sample_indices),
        "episode_lengths": episode_lengths,
        "quantile_stats": True,
        "minimum_sampled_camera_brightness": minimum_camera_brightness,
        "minimum_sampled_camera_contrast": minimum_camera_contrast,
        "maximum_sampled_action_state_delta_deg": maximum_delta,
        "action_coordinates": "follower degrees after direction, joint-limit, and per-tick clipping",
        "training_profile_id": profile["profile_id"],
        "training_profile_version": profile["profile_version"],
        "training_profile_digest": profile_digest,
        "collection_contract_digest": collection_digest,
        "collection_contract": collection_contract,
        "training_profile_sidecar": str(sidecar_path),
        "mission_newt_compatible": False,
        "mission_newt_blocker": "ReBot state/action are 7D; current New Theory SO-101 contract is 6D",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    print("PASS")
    print(f"episodes={dataset.num_episodes}")
    print(f"frames={dataset.num_frames}")
    print(f"fps={dataset.fps}")
    print(f"camera_keys={dataset.meta.camera_keys}")
    print(f"task={next(iter(task_values))}")
    print(f"sampled_frames={len(sample_indices)}")
    print(f"max_action_state_delta_deg={maximum_delta:.3f}")
    if args.report:
        print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
