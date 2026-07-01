
from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient



class MiClient(LLMApiClient):
    pass


class MiClient(OpenAIChatClient):
    def __init__(self, model: str = "",thinking: bool = True):
        base_url = "https://api.xiaomimimo.com/v1/chat/completions"
        config = Config()

        api_key = config.get("mi_key")
        super().__init__(api_key, base_url, thinking)
        self.model = config.resolve_value(
            model,
            ("mi_client_model",),
            "mimo-v2-pro",
        )