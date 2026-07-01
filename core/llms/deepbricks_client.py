from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class DeepBricksClient(LLMApiClient):
    pass

class DeepBricksClient(OpenAIChatClient):
    DEFAULT_MODEL = "gpt-5.5"

    def __init__(self, model: str = ""):
        base_url = "https://api.deepbricks.ai/v1/"
        config = Config()
        api_key = config.get("deepbricks_api_key")
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("deepbricks_model",),
            self.DEFAULT_MODEL,
        )
