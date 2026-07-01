from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class VolcDeepSeekR1Client(LLMApiClient):
    pass


#deepseek-r1-250120
class VolcDeepSeekR1Client(OpenAIChatClient):
    def __init__(self, model: str = ""):
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
        config = Config()

        api_key = config.get("volcengine_api_key")
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("volc_deep_seek_r1_client_model",),
            "deepseek-r1-250528",
        )
