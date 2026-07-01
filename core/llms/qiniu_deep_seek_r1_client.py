from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class QiniuDeepSeekR1Client(LLMApiClient):
    pass




class QiniuDeepSeekR1Client(OpenAIChatClient):
    def __init__(self, model: str = ""):
        base_url = "https://api.qnaigc.com/v1/"

        config = Config()

        api_key = config.get("qiniu_api_key")
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("qiniu_deep_seek_r1_client_model",),
            "deepseek-r1-0528",
        )


