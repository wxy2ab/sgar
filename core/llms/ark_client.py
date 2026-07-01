from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class ArkClient(LLMApiClient):
    pass

class ArkClient(OpenAIChatClient):
    def __init__(self, model: str = ""):
        base_url = "https://ark.cn-beijing.volces.com/api/coding/v3"
        config = Config()
        api_key = config.get("ark_coding_key")
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("ark_client_model",),
            "ark-code-latest",
        )
