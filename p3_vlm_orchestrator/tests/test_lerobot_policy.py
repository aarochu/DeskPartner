from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import importlib
import importlib.metadata
from pathlib import Path
from io import StringIO
import json
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from p3_vlm_orchestrator.policy_rollout.lerobot_policy import (
    LeRobotCompatibilityError,
    LeRobotPolicyAdapter,
    _LeRobotAPI,
    _import_lerobot_api,
)
from p3_vlm_orchestrator.policy_rollout.offline import evaluate_checkpoint
from rebot_operator_kit.rollout.checkpoint import CheckpointBundle


TASK = "Pick up one can and place it in the taped sorting zone"


def checkpoint_bundle(path: Path) -> CheckpointBundle:
    return CheckpointBundle(
        path=path,
        task=TASK,
        action_dimension=7,
        chunk_size=10,
        action_steps=10,
        image_order=(
            "observation.images.front",
            "observation.images.side",
        ),
        profile_digest="test-digest",
        profile_snapshot={},
    )


class ImportBoundaryTest(unittest.TestCase):
    def test_uses_lerobot_registrar_only_for_third_party_policy_plugins(self) -> None:
        imported_plugins: list[str] = []

        def package(name: str) -> ModuleType:
            module = ModuleType(name)
            module.__path__ = []  # type: ignore[attr-defined]
            return module

        torch = ModuleType("torch")
        torch.inference_mode = lambda: None  # type: ignore[attr-defined]
        configs_policies = ModuleType("lerobot.configs.policies")
        configs_policies.PreTrainedConfig = type(  # type: ignore[attr-defined]
            "PreTrainedConfig",
            (),
            {"from_pretrained": classmethod(lambda cls, path, **kwargs: None)},
        )
        policies_factory = ModuleType("lerobot.policies.factory")
        policies_factory.get_policy_class = lambda name: object  # type: ignore[attr-defined]
        policies_factory.make_pre_post_processors = lambda **kwargs: (  # type: ignore[attr-defined]
            object(),
            object(),
        )
        policies_utils = ModuleType("lerobot.policies.utils")
        policies_utils.prepare_observation_for_inference = (  # type: ignore[attr-defined]
            lambda observation, device, task: observation
        )
        import_utils = ModuleType("lerobot.utils.import_utils")

        def register_third_party_plugins() -> None:
            for distribution in importlib.metadata.distributions():
                name = distribution.metadata.get("Name")
                if isinstance(name, str) and name.startswith(
                    (
                        "lerobot_robot_",
                        "lerobot_camera_",
                        "lerobot_teleoperator_",
                        "lerobot_policy_",
                    )
                ):
                    importlib.import_module(name)

        import_utils.register_third_party_plugins = (  # type: ignore[attr-defined]
            register_third_party_plugins
        )
        modules = {
            "torch": torch,
            "lerobot": package("lerobot"),
            "lerobot.configs": package("lerobot.configs"),
            "lerobot.configs.policies": configs_policies,
            "lerobot.policies": package("lerobot.policies"),
            "lerobot.policies.factory": policies_factory,
            "lerobot.policies.utils": policies_utils,
            "lerobot.utils": package("lerobot.utils"),
            "lerobot.utils.import_utils": import_utils,
        }
        distributions = [
            SimpleNamespace(metadata={"Name": "lerobot_robot_hardware"}),
            SimpleNamespace(metadata={"Name": "lerobot_teleoperator_hardware"}),
            SimpleNamespace(metadata={"Name": "lerobot_policy_custom"}),
        ]

        with (
            patch.dict(sys.modules, modules),
            patch.object(
                importlib.metadata,
                "distributions",
                return_value=distributions,
            ),
            patch.object(importlib.metadata, "version", return_value="99.0"),
            patch.object(
                importlib,
                "import_module",
                side_effect=lambda name: imported_plugins.append(name)
                or ModuleType(name),
            ),
        ):
            _import_lerobot_api()

        self.assertEqual(imported_plugins, ["lerobot_policy_custom"])


class FakeConfig:
    def __init__(self) -> None:
        self.type = "registered_test_policy"
        self.device = "training-device"


class FakePolicy:
    def __init__(self) -> None:
        self.to_calls: list[str] = []
        self.eval_calls = 0
        self.reset_calls = 0
        self.prediction: object | None = None
        self.predict_calls: list[object] = []

    def to(self, device: str) -> FakePolicy:
        self.to_calls.append(device)
        return self

    def eval(self) -> FakePolicy:
        self.eval_calls += 1
        return self

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_action_chunk(self, observation: object) -> object:
        self.predict_calls.append(observation)
        return self.prediction


class FakePolicyClass:
    calls: list[tuple[str, FakeConfig, bool]] = []
    policy = FakePolicy()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        config: FakeConfig,
        local_files_only: bool,
    ) -> FakePolicy:
        cls.calls.append((checkpoint, config, local_files_only))
        return cls.policy


class FakeProcessor:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.reset_error: Exception | None = None
        self.calls: list[object] = []
        self.transform = lambda value: value

    def reset(self) -> None:
        self.reset_calls += 1
        if self.reset_error is not None:
            raise self.reset_error

    def __call__(self, value: object) -> object:
        self.calls.append(value)
        return self.transform(value)


class FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.detach_calls = 0
        self.cpu_calls = 0

    def detach(self) -> FakeTensor:
        self.detach_calls += 1
        return self

    def cpu(self) -> FakeTensor:
        self.cpu_calls += 1
        return self

    def numpy(self) -> np.ndarray:
        return self.array

    def item(self) -> object:
        return self.array.item()


class FakeBackendFixture:
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.checkpoint = Path(temporary.name).resolve()
        self.config = FakeConfig()
        self.preprocessor = FakeProcessor()
        self.postprocessor = FakeProcessor()
        self.config_calls: list[str] = []
        self.config_local_only_calls: list[bool] = []
        self.policy_type_calls: list[str] = []
        self.processor_calls: list[dict[str, object]] = []
        self.prepare_calls: list[tuple[dict[str, np.ndarray], str, str]] = []
        self.mutate_during_prepare = False
        self.prepared_task_override: str | None = None
        self.inference_entries = 0
        FakePolicyClass.calls = []
        FakePolicyClass.policy = FakePolicy()

        def config_from_pretrained(
            path: str, *, local_files_only: bool
        ) -> FakeConfig:
            self.config_calls.append(path)
            self.config_local_only_calls.append(local_files_only)
            return self.config

        def get_policy_class(policy_type: str) -> type[FakePolicyClass]:
            self.policy_type_calls.append(policy_type)
            return FakePolicyClass

        def make_pre_post_processors(**kwargs: object) -> tuple[FakeProcessor, FakeProcessor]:
            self.processor_calls.append(kwargs)
            return self.preprocessor, self.postprocessor

        def prepare_observation(
            observation: dict[str, np.ndarray], device: str, task: str
        ) -> dict[str, object]:
            self.prepare_calls.append((dict(observation), device, task))
            if self.mutate_during_prepare:
                for value in observation.values():
                    value[...] = 99
            observation["task"] = (
                task
                if self.prepared_task_override is None
                else self.prepared_task_override
            )
            observation["robot_type"] = ""
            return observation

        @contextmanager
        def inference_mode():
            self.inference_entries += 1
            yield

        self.api = _LeRobotAPI(
            config_from_pretrained=config_from_pretrained,
            get_policy_class=get_policy_class,
            make_pre_post_processors=make_pre_post_processors,
            prepare_observation=prepare_observation,
            inference_mode=inference_mode,
            runtime_description="fake LeRobot 99 on Python 99",
        )


class CheckpointLoadingTest(FakeBackendFixture, unittest.TestCase):
    def test_loads_config_weights_and_saved_processors_from_one_directory(self) -> None:
        with patch(
            "p3_vlm_orchestrator.policy_rollout.lerobot_policy._import_lerobot_api",
            return_value=self.api,
        ):
            adapter = LeRobotPolicyAdapter.from_checkpoint(
                checkpoint_bundle(self.checkpoint), "cpu"
            )

        checkpoint = str(self.checkpoint)
        self.assertEqual(self.config_calls, [checkpoint])
        self.assertEqual(self.config_local_only_calls, [True])
        self.assertEqual(self.policy_type_calls, ["registered_test_policy"])
        self.assertEqual(FakePolicyClass.calls, [(checkpoint, self.config, True)])
        self.assertEqual(len(self.processor_calls), 1)
        self.assertIs(self.processor_calls[0]["policy_cfg"], self.config)
        self.assertEqual(self.processor_calls[0]["pretrained_path"], checkpoint)
        self.assertEqual(
            self.processor_calls[0]["preprocessor_config_filename"],
            "preprocessor_config.json",
        )
        self.assertEqual(
            self.processor_calls[0]["postprocessor_config_filename"],
            "postprocessor_config.json",
        )
        self.assertIs(adapter.policy, FakePolicyClass.policy)

    def test_sets_requested_device_and_resets_loaded_inference_components(self) -> None:
        with patch(
            "p3_vlm_orchestrator.policy_rollout.lerobot_policy._import_lerobot_api",
            return_value=self.api,
        ):
            LeRobotPolicyAdapter.from_checkpoint(
                checkpoint_bundle(self.checkpoint), "mps"
            )

        self.assertEqual(self.config.device, "mps")
        self.assertEqual(FakePolicyClass.policy.to_calls, ["mps"])
        self.assertEqual(FakePolicyClass.policy.eval_calls, 1)
        self.assertEqual(FakePolicyClass.policy.reset_calls, 1)
        self.assertEqual(self.preprocessor.reset_calls, 1)
        self.assertEqual(self.postprocessor.reset_calls, 1)

    def test_names_the_unresolved_registered_policy_in_compatibility_error(self) -> None:
        self.config.type = "molmoact2"

        def unavailable(policy_type: str) -> type:
            raise ValueError(f"Policy type {policy_type!r} is not installed")

        api = replace(self.api, get_policy_class=unavailable)
        with patch(
            "p3_vlm_orchestrator.policy_rollout.lerobot_policy._import_lerobot_api",
            return_value=api,
        ):
            with self.assertRaisesRegex(
                LeRobotCompatibilityError,
                r"registered policy 'molmoact2' resolution.*fake LeRobot 99",
            ):
                LeRobotPolicyAdapter.from_checkpoint(
                    checkpoint_bundle(self.checkpoint), "cpu"
                )

    def test_names_both_saved_processor_files_in_schema_error(self) -> None:
        def incompatible_processors(**kwargs: object) -> tuple[object, object]:
            raise KeyError("unknown processor step from newer checkpoint")

        api = replace(
            self.api,
            make_pre_post_processors=incompatible_processors,
        )
        with patch(
            "p3_vlm_orchestrator.policy_rollout.lerobot_policy._import_lerobot_api",
            return_value=api,
        ):
            with self.assertRaisesRegex(
                LeRobotCompatibilityError,
                r"saved processor schema loading .*preprocessor_config.json.*postprocessor_config.json",
            ):
                LeRobotPolicyAdapter.from_checkpoint(
                    checkpoint_bundle(self.checkpoint), "cpu"
                )

    def test_missing_saved_processor_state_fails_before_factory_loading(self) -> None:
        (self.checkpoint / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "name": "policy_preprocessor",
                    "steps": [
                        {
                            "registry_name": "normalizer_processor",
                            "config": {"device": "cuda"},
                            "state_file": "missing-normalizer.safetensors",
                        }
                    ],
                }
            )
        )
        (self.checkpoint / "postprocessor_config.json").write_text(
            json.dumps({"name": "policy_postprocessor", "steps": []})
        )

        with patch(
            "p3_vlm_orchestrator.policy_rollout.lerobot_policy._import_lerobot_api",
            return_value=self.api,
        ):
            with self.assertRaisesRegex(
                LeRobotCompatibilityError,
                r"saved processor state file is missing.*missing-normalizer.safetensors",
            ):
                LeRobotPolicyAdapter.from_checkpoint(
                    checkpoint_bundle(self.checkpoint), "cpu"
                )

        self.assertEqual(self.processor_calls, [])


class AdapterInferenceTest(FakeBackendFixture, unittest.TestCase):
    def make_adapter(self) -> LeRobotPolicyAdapter:
        with patch(
            "p3_vlm_orchestrator.policy_rollout.lerobot_policy._import_lerobot_api",
            return_value=self.api,
        ):
            return LeRobotPolicyAdapter.from_checkpoint(
                checkpoint_bundle(self.checkpoint), "cpu"
            )

    def observation(self) -> object:
        from rebot_operator_kit.rollout.contracts import RolloutObservation

        return RolloutObservation(
            front=np.full((3, 4, 3), 11, dtype=np.uint8),
            side=np.full((2, 5, 3), 22, dtype=np.uint8),
            state_deg=np.arange(7, dtype=np.float32),
            task="caller task must not replace locked task",
            captured_monotonic_s=123.0,
        )

    def test_reset_clears_policy_and_saved_processor_state_and_fails_closed(self) -> None:
        adapter = self.make_adapter()

        adapter.reset()

        self.assertEqual(FakePolicyClass.policy.reset_calls, 2)
        self.assertEqual(self.preprocessor.reset_calls, 2)
        self.assertEqual(self.postprocessor.reset_calls, 2)

        self.preprocessor.reset_error = RuntimeError("state reset exploded")
        with self.assertRaisesRegex(RuntimeError, "preprocessor reset failed"):
            adapter.reset()

    def test_maps_raw_observation_predicts_and_postprocesses_a_chunk(self) -> None:
        adapter = self.make_adapter()
        observation = self.observation()
        preprocessed = {"complete": "preprocessed batch"}
        raw_prediction = FakeTensor(np.full((1, 2, 7), -0.25, dtype=np.float32))
        postprocessed_array = np.arange(14, dtype=np.float32).reshape(1, 2, 7)
        postprocessed = FakeTensor(postprocessed_array)
        self.preprocessor.transform = lambda value: preprocessed
        FakePolicyClass.policy.prediction = raw_prediction
        self.postprocessor.transform = lambda value: postprocessed

        result = adapter.predict(observation)

        self.assertEqual(len(self.prepare_calls), 1)
        raw, device, task = self.prepare_calls[0]
        self.assertEqual(
            list(raw),
            [
                "observation.images.front",
                "observation.images.side",
                "observation.state",
            ],
        )
        self.assertIsNot(raw["observation.images.front"], observation.front)
        self.assertIsNot(raw["observation.images.side"], observation.side)
        self.assertIsNot(raw["observation.state"], observation.state_deg)
        self.assertEqual(device, "cpu")
        self.assertEqual(task, TASK)
        prepared = self.preprocessor.calls[0]
        self.assertEqual(
            list(prepared),
            [
                "observation.images.front",
                "observation.images.side",
                "observation.state",
                "task",
            ],
        )
        self.assertEqual(prepared["task"], TASK)
        self.assertEqual(FakePolicyClass.policy.predict_calls, [preprocessed])
        self.assertEqual(self.postprocessor.calls, [raw_prediction])
        self.assertEqual(self.inference_entries, 1)
        self.assertEqual(result.shape, (2, 7))
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_equal(result, np.arange(14).reshape(2, 7))
        postprocessed_array[:] = 999
        self.assertFalse(np.any(result == 999))
        self.assertEqual(postprocessed.detach_calls, 1)
        self.assertEqual(postprocessed.cpu_calls, 1)

    def test_copies_images_and_coerces_copied_state_before_mutating_prepare(self) -> None:
        adapter = self.make_adapter()
        observation = self.observation()
        object.__setattr__(
            observation,
            "state_deg",
            np.arange(7, dtype=np.float64),
        )
        original_front = observation.front.copy()
        original_side = observation.side.copy()
        original_state = observation.state_deg.copy()
        self.mutate_during_prepare = True
        FakePolicyClass.policy.prediction = FakeTensor(
            np.zeros((1, 2, 7), dtype=np.float32)
        )

        result = adapter.predict(observation)

        raw, _, _ = self.prepare_calls[0]
        self.assertIsNot(raw["observation.images.front"], observation.front)
        self.assertIsNot(raw["observation.images.side"], observation.side)
        self.assertIsNot(raw["observation.state"], observation.state_deg)
        self.assertEqual(raw["observation.state"].dtype, np.float32)
        np.testing.assert_array_equal(observation.front, original_front)
        np.testing.assert_array_equal(observation.side, original_side)
        np.testing.assert_array_equal(observation.state_deg, original_state)
        self.assertEqual(observation.state_deg.dtype, np.float64)
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(result.shape, (2, 7))

    def test_rejects_when_preparation_does_not_preserve_locked_task(self) -> None:
        adapter = self.make_adapter()
        self.prepared_task_override = "a different prepared task"
        FakePolicyClass.policy.prediction = FakeTensor(
            np.zeros((1, 2, 7), dtype=np.float32)
        )

        with self.assertRaisesRegex(ValueError, "prepared task.*checkpoint task"):
            adapter.predict(self.observation())

        self.assertEqual(self.preprocessor.calls, [])

    def test_rejects_a_non_seven_element_state_before_preprocessing(self) -> None:
        adapter = self.make_adapter()
        observation = self.observation()
        object.__setattr__(observation, "state_deg", np.zeros(6))

        with self.assertRaisesRegex(ValueError, r"observation state.*\(7,\)"):
            adapter.predict(observation)

        self.assertEqual(self.prepare_calls, [])

    def test_rejects_malformed_or_nonfinite_postprocessed_chunks(self) -> None:
        adapter = self.make_adapter()
        FakePolicyClass.policy.prediction = FakeTensor(np.zeros((1, 2, 7)))
        cases = (
            (np.zeros(7), "shape"),
            (np.zeros((1, 0, 7)), "at least one"),
            (np.zeros((1, 2, 6)), "shape"),
            (np.zeros((2, 2, 7)), "batch"),
            (np.full((1, 2, 7), np.nan), "nonfinite"),
        )

        for output, message in cases:
            with self.subTest(shape=output.shape, message=message):
                self.postprocessor.transform = lambda value, output=output: FakeTensor(
                    output
                )
                with self.assertRaisesRegex(ValueError, message):
                    adapter.predict(self.observation())


class FakeDataset:
    def __init__(self, samples: list[dict[str, object]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.samples[index]


class FakeOfflineAdapter:
    def __init__(self) -> None:
        self.observations: list[object] = []
        self.reset_calls = 0
        self.state = 0

    def reset(self) -> None:
        self.reset_calls += 1
        self.state = 0

    def predict(self, observation: object) -> np.ndarray:
        self.observations.append(observation)
        self.state += 1
        return np.full((2, 7), self.state, dtype=np.float64)


def dataset_sample(
    episode_index: int,
    *,
    value: float,
    task: str = TASK,
) -> dict[str, object]:
    return {
        "episode_index": FakeTensor(np.array(episode_index)),
        "timestamp": FakeTensor(np.array(value / 10.0)),
        "observation.images.front": FakeTensor(
            np.full((3, 2, 5), value / 255.0, dtype=np.float32)
        ),
        "observation.images.side": FakeTensor(
            np.full((3, 4, 6), (value + 1) / 255.0, dtype=np.float32)
        ),
        "observation.state": FakeTensor(
            np.arange(7, dtype=np.float32) + value
        ),
        "task": task,
    }


class OfflineEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.dataset_root = Path(temporary.name).resolve()
        self.bundle = checkpoint_bundle(self.dataset_root / "checkpoint")
        self.adapter = FakeOfflineAdapter()

    def test_evaluates_every_sample_from_exactly_requested_distinct_episodes(self) -> None:
        dataset = FakeDataset(
            [
                dataset_sample(4, value=10),
                dataset_sample(4, value=20),
                dataset_sample(9, value=30),
                dataset_sample(12, value=40),
            ]
        )
        loader_calls: list[Path] = []
        factory_calls: list[tuple[CheckpointBundle, str]] = []
        clock_values = iter([1.0, 1.1, 2.0, 2.2, 3.0, 3.3])
        output = StringIO()

        results = evaluate_checkpoint(
            self.bundle,
            self.dataset_root,
            episodes=2,
            device="cpu",
            dataset_loader=lambda root: loader_calls.append(root) or dataset,
            adapter_factory=lambda bundle, device: factory_calls.append(
                (bundle, device)
            )
            or self.adapter,
            clock=lambda: next(clock_values),
            output=output,
        )

        self.assertEqual(loader_calls, [self.dataset_root])
        self.assertEqual(factory_calls, [(self.bundle, "cpu")])
        self.assertEqual([result.episode_index for result in results], [4, 4, 9])
        self.assertEqual([result.sample_index for result in results], [0, 1, 2])
        self.assertEqual([result.shape for result in results], [(2, 7)] * 3)
        self.assertEqual([result.minimum for result in results], [1.0, 2.0, 1.0])
        self.assertEqual([result.maximum for result in results], [1.0, 2.0, 1.0])
        self.assertEqual(self.adapter.reset_calls, 2)
        np.testing.assert_allclose(
            [result.latency_s for result in results], [0.1, 0.2, 0.3]
        )
        self.assertEqual(len(self.adapter.observations), 3)
        first = self.adapter.observations[0]
        self.assertEqual(first.front.shape, (2, 5, 3))
        self.assertEqual(first.front.dtype, np.uint8)
        self.assertTrue(np.all(first.front == 10))
        self.assertEqual(first.side.shape, (4, 6, 3))
        self.assertTrue(np.all(first.side == 11))
        np.testing.assert_array_equal(first.state_deg, np.arange(7) + 10)
        self.assertEqual(first.state_deg.dtype, np.float32)
        self.assertEqual(first.task, TASK)
        self.assertEqual(first.captured_monotonic_s, 1.0)
        printed = output.getvalue()
        self.assertIn("episode=4 sample=0 shape=(2, 7)", printed)
        self.assertIn("min=1.000000 max=1.000000 latency_ms=100.000", printed)

    def test_rejects_a_recorded_task_mismatch_before_prediction(self) -> None:
        dataset = FakeDataset(
            [dataset_sample(0, value=10, task="a different manipulation task")]
        )

        with self.assertRaisesRegex(ValueError, "dataset task mismatch"):
            evaluate_checkpoint(
                self.bundle,
                self.dataset_root,
                episodes=1,
                dataset_loader=lambda root: dataset,
                adapter_factory=lambda bundle, device: self.adapter,
                output=StringIO(),
            )

        self.assertEqual(self.adapter.observations, [])

    def test_rejects_when_dataset_has_fewer_distinct_episodes_than_requested(self) -> None:
        dataset = FakeDataset(
            [
                dataset_sample(4, value=10),
                dataset_sample(4, value=20),
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            r"requested 2 distinct episodes.*found 1",
        ):
            evaluate_checkpoint(
                self.bundle,
                self.dataset_root,
                episodes=2,
                dataset_loader=lambda root: dataset,
                adapter_factory=lambda bundle, device: self.adapter,
                output=StringIO(),
            )

    def test_requires_a_positive_episode_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "episodes must be a positive integer"):
            evaluate_checkpoint(
                self.bundle,
                self.dataset_root,
                episodes=0,
                dataset_loader=lambda root: FakeDataset([]),
                adapter_factory=lambda bundle, device: self.adapter,
                output=StringIO(),
            )


if __name__ == "__main__":
    unittest.main()
