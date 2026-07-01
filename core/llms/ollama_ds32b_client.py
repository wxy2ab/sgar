from .openai_chat_client import OpenAIChatClient
from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class OllamaDS32bClient(LLMApiClient):
    pass



class OllamaDS32bClient(OpenAIChatClient):
    def __init__(self, model: str = ""):
        
        config = Config()

        ollama_url = config.get("ollama_url")
        base_url = ollama_url
        api_key='ollama'
        super().__init__(api_key, base_url, max_tokens=None)
        self.model = config.resolve_value(
            model,
            ("ollama_ds32b_client_model",),
            "deepseek-r1:32b",
        )
