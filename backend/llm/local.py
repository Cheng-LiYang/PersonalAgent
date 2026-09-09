"""Offline extractive answer provider."""
from backend.llm.base import LanguageModel

class ExtractiveModel(LanguageModel):
    """Explain that synthesis requires a configured generative model."""
    def generate(self, system: str, prompt: str) -> str:
        if "Return exactly" in prompt:
            return '{}'
        return (
            "已检索到相关资料，但当前尚未连接生成模型，无法对证据进行可靠归纳。\n\n"
            "请打开左侧“模型设置”，填写可用的模型名称、API Base URL 和 API Key 后重新提问。"
        )
