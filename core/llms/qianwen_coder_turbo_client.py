from .qianwen_client import QianWenClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class QianWenCoderTurboClient(LLMApiClient):
    pass

class QianWenCoderTurboClient(QianWenClient):
    def __init__(self, model: str = ""):
        config = Config()
        api_key = config.get("dashscope_api_key")
        self.model = config.resolve_value(
            model,
            ("qianwen_coder_turbo_client_model",),
            "qwen3-coder-480b-a35b-instruct",
        )
        super().__init__(api_key, max_tokens=8192 ,model=self.model)
