from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class InfinityDeepSeekClient(LLMApiClient):
    pass




class InfinityDeepSeekClient(OpenAIChatClient):
    def __init__(self, model: str = ""):
        base_url = "https://cloud.infini-ai.com/maas/v1"

        config = Config()


        api_key = config.get("infinity_api_key")
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("infini_deep_seek_client_model",),
            "deepseek-v3",
        )
