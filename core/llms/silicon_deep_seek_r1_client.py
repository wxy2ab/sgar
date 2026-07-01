from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class SiliconDeepSeekR1Client(LLMApiClient):
    pass


class SiliconDeepSeekR1Client(OpenAIChatClient):
    def __init__(self, model: str = ""):
        base_url = "https://api.siliconflow.cn/v1"
        config = Config()


        api_key = config.get("siliconflow_key")
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("silicon_deep_seek_r1_client_model",),
            "Pro/deepseek-ai/DeepSeek-R1",
        )
