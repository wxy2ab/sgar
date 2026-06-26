
from .moonshot_client import MoonShotClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class PPioClaudeClient(LLMApiClient):
    pass


class PPioClaudeClient(MoonShotClient):
    DEFAULT_MODEL = "pa/claude-sonnet-4-6"

    def __init__(self, model: str = ""):
        base_url = "https://api.ppinfra.com/openai"
        config = Config()

        api_key = config.get("ppio_api_key")
        super().__init__(api_key, base_url, max_tokens=8192)
        self.model = config.resolve_value(
            model,
            ("ppio_claude_model",),
            self.DEFAULT_MODEL,
        )
