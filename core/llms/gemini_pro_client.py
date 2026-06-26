#"gemini-2.0-pro-exp-02-05"

from .gemini2_client import Gemini2Client
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient


class GeminiProClient(LLMApiClient):
    pass

class GeminiProClient(Gemini2Client):
    def __init__(self, model: str = ""):
        config = Config()
        self.model = config.resolve_value(
            model,
            ("gemini_pro_client_model",),
            "gemini-2.0-pro-exp-02-05",
        )
        super().__init__(self.model)
