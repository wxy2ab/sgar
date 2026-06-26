from .moonshot_client import MoonShotClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class SenseDeepSeekClient(LLMApiClient):
    pass



class SenseDeepSeekClient(MoonShotClient):
    def __init__(self, model: str = ""):
        base_url = "https://api.sensenova.cn/compatible-mode/v1/"

        config = Config()

        api_key = config.get("sense_api_key")
        super().__init__(api_key, base_url, max_tokens=8196,presence_penalty=None,frequency_penalty=None,top_p=None)
        self.model = config.resolve_value(
            model,
            ("sense_deep_seek_client_model",),
            "DeepSeek-V3",
        )
