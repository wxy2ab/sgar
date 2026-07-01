from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class QianWenPlusClient(LLMApiClient):
    pass

class QianWenPlusClient(OpenAIChatClient):
    def __init__(self, model: str = ""):
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        config = Config()
        api_key = config.get("dashscope_api_key") or ""
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("qianwen_plus_model",),
            "qwen3.6-plus",
        )
        
