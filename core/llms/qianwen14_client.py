from .qianwen_client import QianWenClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class QianWenCoder14Client(LLMApiClient):
    pass

class QianWenCoder14Client(QianWenClient):
    def __init__(self, model: str = ""):
        config = Config()
        api_key = config.get("dashscope_api_key")
        self.model = config.resolve_value(
            model,
            ("qianwen14_client_model",),
            "qwen3-14bt",
        )
        super().__init__(api_key, max_tokens=8192 ,model=self.model)
