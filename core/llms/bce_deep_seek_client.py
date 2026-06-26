from .moonshot_client import MoonShotClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class BceDeepSeekClient(LLMApiClient):
    pass



class BceDeepSeekClient(MoonShotClient):
    def __init__(self, model: str = ""):
        base_url = "https://qianfan.baidubce.com/v2"
        config = Config()


        api_key = config.get("baidu_bce_api_key")
        super().__init__(api_key, base_url, max_tokens=8192)
        self.model = config.resolve_value(
            model,
            ("bce_deep_seek_client_model",),
            "deepseek-v3",
        )
