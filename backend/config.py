"""YAML configuration loading."""
from __future__ import annotations
import os
import sys
import shutil
from pathlib import Path
from typing import Any
import yaml

SOURCE_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else SOURCE_ROOT


def _default_config_path() -> Path:
    """Prefer an editable config beside the executable, then bundled defaults."""
    external = RUNTIME_ROOT / "config" / "config.yaml"
    if external.is_file():
        return external
    bundle_root = Path(getattr(sys, "_MEIPASS", SOURCE_ROOT))
    return bundle_root / "config" / "config.yaml"

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    configured = path or os.getenv("PKA_CONFIG")
    config_path = Path(configured) if configured else _default_config_path()
    if not config_path.is_absolute():
        config_path = RUNTIME_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if getattr(sys, "frozen", False):
        local_app_data = Path(os.getenv("LOCALAPPDATA", RUNTIME_ROOT)) / "PersonalKnowledgeAgent"
        config["app"]["data_dir"] = str(local_app_data / "data")
        config["app"]["knowledge_base"] = str(local_app_data / "KnowledgeBase")
    for key in ("data_dir", "knowledge_base"):
        value = Path(config["app"][key])
        config["app"][key] = str(value if value.is_absolute() else RUNTIME_ROOT / value)
    return config

def ensure_data_dirs(config: dict[str, Any]) -> None:
    data_dir = Path(config["app"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    Path(config["app"]["knowledge_base"]).mkdir(parents=True, exist_ok=True)
    if getattr(sys, "frozen", False):
        legacy_dir = RUNTIME_ROOT / "data"
        for name in ("vectors.sqlite3", "memory.sqlite3", "model_settings.json"):
            source, target = legacy_dir / name, data_dir / name
            if source.is_file() and not target.exists():
                shutil.copy2(source, target)
