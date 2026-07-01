from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class VolcDeepSeekClient(LLMApiClient):
    pass



class VolcDeepSeekClient(OpenAIChatClient):
    def __init__(self, model: str = ""):
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
        config = Config()

        api_key = config.get("volcengine_api_key")
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("volc_deep_seek_client_model",),
            "deepseek-v3-250324",
        )
