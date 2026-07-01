
from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class PPioClaudeOpusClient(LLMApiClient):
    pass


class PPioClaudeOpusClient(OpenAIChatClient):
    DEFAULT_MODEL = "pa/claude-sonnet-4-6"

    def __init__(self, model: str = ""):
        base_url = "https://api.ppinfra.com/openai"
        config = Config()

        api_key = config.get("ppio_api_key")
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("ppio_claude_sonnet_model", "ppio_claude_model"),
            self.DEFAULT_MODEL,
        )
