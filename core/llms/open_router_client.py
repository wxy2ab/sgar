from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class OpenRouterClient(LLMApiClient):
    pass

class OpenRouterClient(OpenAIChatClient):
    DEFAULT_MODEL = "openai/gpt-5.5"

    def __init__(self, model: str = ""):
        base_url = "https://openrouter.ai/api/v1"
        config = Config()
        api_key = config.get("openrouter_api_key")
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("open_router_model",),
            self.DEFAULT_MODEL,
        )
