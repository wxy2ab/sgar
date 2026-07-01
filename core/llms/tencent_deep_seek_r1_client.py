from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class TencentDeepSeekR1Client(LLMApiClient):
    pass




class TencentDeepSeekR1Client(OpenAIChatClient):
    def __init__(self, model: str = ""):
        base_url = "https://api.lkeap.cloud.tencent.com/v1"


        config = Config()

        api_key = config.get("tencent_api_key")
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("tencent_deep_seek_r1_client_model",),
            "deepseek-r1",
        )
