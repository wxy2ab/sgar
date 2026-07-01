
from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class Zero1LLamaImproverClient(LLMApiClient):
    pass

class Zero1LLamaImproverClient(OpenAIChatClient):
    DEFAULT_MODEL = "yi-lightning"

    def __init__(self, model: str = ""):
        base_url = "https://api.lingyiwanwu.com/v1"
        config = Config()
        api_key = config.get("zero_one_api_key")
        super().__init__(api_key, base_url)
        self._model_list=["yi-large","yi-medium","yi-large-turbo","yi-lightning"]
        self.model = config.resolve_value(
            model,
            ("zero1_improver_model",),
            self.DEFAULT_MODEL,
        )
