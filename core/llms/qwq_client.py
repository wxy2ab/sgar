from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class QwqClient(LLMApiClient):
    pass


class QwqClient(OpenAIChatClient):
    def __init__(self, model: str = ""):
        base_url = "https://api.siliconflow.cn/v1"
        config = Config()

        api_key = "sk-W0rpStc95T7JVYVwDYc29IyirjtpPPby6SozFMQr17m8KWeo"
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("qwq_client_model",),
            "free:QwQ-32B",
        )
