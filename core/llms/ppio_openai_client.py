
from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class PPioOpenAIClient(LLMApiClient):
    pass


class PPioOpenAIClient(OpenAIChatClient):
    DEFAULT_MODEL = "gpt-5.5"

    def __init__(self, model: str = ""):
        base_url = "https://api.ppinfra.com/openai"
        config = Config()

        api_key = config.get("ppio_api_key")
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("ppio_openai_model",),
            self.DEFAULT_MODEL,
        )
