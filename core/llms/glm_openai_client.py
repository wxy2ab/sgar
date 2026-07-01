from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient


class GLMOpenAIClient(LLMApiClient):
    pass


class GLMOpenAIClient(OpenAIChatClient):
    def __init__(self, model: str = ""):
        base_url = "https://open.bigmodel.cn/api/coding/paas/v4"
        config = Config()

        api_key = config.get("glm_api_key")
        super().__init__(api_key, base_url)
        self.model = config.resolve_value(
            model,
            ("glm_openai_client_model",),
            "glm-5.2",
        )
