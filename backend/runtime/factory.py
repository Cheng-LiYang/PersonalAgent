"""Shared construction of the index, Agent, and persisted model settings."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from backend.agent import KnowledgeAgent
from backend.config import ensure_data_dirs, load_config
from backend.indexing import IndexService
from backend.settings import SettingsStore


def create_runtime(
    config: dict[str, Any] | None = None,
    *,
    settings_data_dir: str | Path | None = None,
    restore_model: bool = True,
) -> tuple[dict[str, Any], IndexService, KnowledgeAgent]:
    """Create one runtime and consistently restore encrypted model settings."""

    active_config = deepcopy(config) if config is not None else load_config()
    ensure_data_dirs(active_config)
    indexer = IndexService(active_config)
    agent = KnowledgeAgent(active_config, indexer.store)
    if restore_model:
        settings_root = settings_data_dir or active_config["app"]["data_dir"]
        saved_model = SettingsStore(settings_root).load_model()
        if saved_model:
            agent.configure_model(saved_model, validate=False)
    return active_config, indexer, agent
