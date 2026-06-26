from core.llms.mini_max_client import MiniMaxClient


class MiniMaxTextClient(MiniMaxClient):
    def __init__(self, model: str = ""):
        # Delegate key resolution to MiniMaxClient (minimax_api_key / env).
        super().__init__("", model)



