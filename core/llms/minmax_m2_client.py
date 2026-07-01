
from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient



class MinMaxM2Client(LLMApiClient):
    pass


class MinMaxM2Client(OpenAIChatClient):
    def __init__(self, model: str = ""):
        base_url = "https://api.minimax.chat/v1"
        config = Config()

        api_key = config.get("minimax_api_key")
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("minmax_m2_client_model",),
            "MiniMax-M3",
        )