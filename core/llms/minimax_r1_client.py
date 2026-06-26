from core.llms.mini_max_client import MiniMaxClient

from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient

class MiniMaxR1Client(LLMApiClient):
    pass



class MiniMaxR1Client(MiniMaxClient):
    def __init__(self, model: str = ""):
        config = Config()
        api_key = config.get("minimax_api_key")
        model = config.resolve_value(
            model,
            ("minimax_r1_client_model",),
            "DeepSeek-R1",
        )
        super().__init__(api_key, model)



