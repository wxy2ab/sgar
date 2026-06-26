
from .moonshot_client import MoonShotClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class PPioClaudeOpusClient(LLMApiClient):
    pass


class PPioClaudeOpusClient(MoonShotClient):
    DEFAULT_MODEL = "pa/claude-opus-4-8"

    def __init__(self, model: str = ""):
        base_url = "https://api.ppinfra.com/openai"
        config = Config()

        api_key = config.get("ppio_api_key")
        super().__init__(api_key, base_url, max_tokens=128000)
        self.model = config.resolve_value(
            model,
            ("ppio_cluade_opus_model",),
            self.DEFAULT_MODEL,
        )
