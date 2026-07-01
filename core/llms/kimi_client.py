
from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient



class KimiClient(LLMApiClient):
    pass


class KimiClient(OpenAIChatClient):
    def __init__(self, model: str = "", thinking: bool = True):
        base_url = "https://api.moonshot.cn/v1"
        config = Config()

        api_key = config.get("moonshot_api_key")
        super().__init__(api_key, base_url, enable_thinking=thinking)
        self.model = config.resolve_value(
            model,
            ("kimi_client_model",),
            "kimi-k2.7-code",
        )
