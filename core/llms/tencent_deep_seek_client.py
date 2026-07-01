from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class TencentDeepSeekClient(LLMApiClient):
    pass



class TencentDeepSeekClient(OpenAIChatClient):
    def __init__(self, model: str = ""):
        base_url = "https://api.lkeap.cloud.tencent.com/v1"

        config = Config()

        api_key = config.get("tencent_api_key")
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("tencent_deep_seek_client_model",),
            "deepseek-v3",
        )
