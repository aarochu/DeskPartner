"""Revision-pinned, read-only access to Hugging Face datasets."""

from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


class HubReader(Protocol):
    """Small boundary used by inventory code and deterministic fakes."""

    def dataset_sha(self, repo_id: str, revision: str) -> str: ...

    def read_json(self, repo_id: str, revision: str, path: str) -> dict[str, Any]: ...

    def read_parquet(self, repo_id: str, revision: str, path: str) -> pd.DataFrame: ...

    def list_paths(self, repo_id: str, revision: str, prefix: str) -> tuple[str, ...]: ...

    def download(self, repo_id: str, revision: str, path: str) -> Path: ...


class HuggingFaceHubReader:
    """Production adapter that forwards the configured immutable revision."""

    def __init__(self, api: HfApi | None = None) -> None:
        self._api = api or HfApi()

    def dataset_sha(self, repo_id: str, revision: str) -> str:
        info = self._api.dataset_info(repo_id, revision=revision)
        if not info.sha:
            raise ValueError(f"dataset {repo_id}@{revision} did not resolve to a SHA")
        return info.sha

    def read_json(self, repo_id: str, revision: str, path: str) -> dict[str, Any]:
        import json

        local_path = self.download(repo_id, revision, path)
        value = json.loads(local_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{repo_id}@{revision}:{path} must contain a JSON object")
        return value

    def read_parquet(self, repo_id: str, revision: str, path: str) -> pd.DataFrame:
        return pd.read_parquet(self.download(repo_id, revision, path))

    def list_paths(self, repo_id: str, revision: str, prefix: str) -> tuple[str, ...]:
        entries = self._api.list_repo_tree(
            repo_id,
            path_in_repo=prefix,
            recursive=True,
            revision=revision,
            repo_type="dataset",
        )
        return tuple(sorted({entry.path for entry in entries}))

    def download(self, repo_id: str, revision: str, path: str) -> Path:
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=path,
                    repo_type="dataset",
                    revision=revision,
                )
            )
        except EntryNotFoundError as error:
            raise FileNotFoundError(f"{repo_id}@{revision}:{path}") from error
