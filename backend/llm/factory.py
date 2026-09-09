"""Language model factory with offline fallback."""
import logging
from typing import Any
from backend.llm.base import LanguageModel
from backend.llm.local import ExtractiveModel
from backend.llm.openai import OpenAIChatModel
LOGGER = logging.getLogger(__name__)

def create_llm(config: dict[str, Any]) -> LanguageModel:
    provider = config.get("provider", "extractive")
    try:
        if provider == "openai":
            return OpenAIChatModel(config.get("model") or config["openai_model"], api_key=config.get("api_key"))
        if provider in {"local", "compatible"}:
            return OpenAIChatModel(
                config.get("model") or config.get("local_model", "deepseek-chat"),
                config.get("base_url") or config.get("local_base_url"),
                config.get("api_key"),
            )
    except (ImportError, OSError, RuntimeError) as exc:
        LOGGER.warning("LLM fallback activated: %s", exc)
    return ExtractiveModel()
