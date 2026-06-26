from .moonshot_client import MoonShotClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class SiliconDeepSeekClient(LLMApiClient):
    pass


class SiliconDeepSeekClient(MoonShotClient):
    def __init__(self, model: str = ""):
        base_url = "https://api.siliconflow.cn/v1"
        config = Config()

        api_key = config.get("siliconflow_key")
        super().__init__(api_key, base_url, max_tokens=8192)
        self.model = config.resolve_value(
            model,
            ("silicon_deep_seek_client_model",),
            "Pro/deepseek-ai/DeepSeek-V3",
        )
