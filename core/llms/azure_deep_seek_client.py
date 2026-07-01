from typing import Union, List, Any, Iterator, Dict
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from abc import ABC, abstractmethod

from ..utils.config_setting import Config
#pip install azure-ai-inference

from core.llms._llm_api_client import LLMApiClient

class AzureDeepSeekClient(LLMApiClient):
    """Azure DeepSeek LLM API客户端实现。"""
    
    def __init__(self):
        # 获取API密钥并初始化Azure客户端
        config = Config()
        # 与本文件其它客户端保持一致：缺失的 key 不在构造时报错，而是留到
        # 真正调用时让请求自然失败（AzureKeyCredential 接受空字符串）。
        api_key = config.get("AZURE_DEEPSEEK_API_KEY") or ""

        self.client = ChatCompletionsClient(
            endpoint='https://DeepSeek-R1-wxy2ab.eastus.models.ai.azure.com',
            credential=AzureKeyCredential(api_key)
        )
        # 用于保存聊天历史记录的变量（仅在text_chat中使用）
        # 初始化聊天历史
        self.history = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        self.max_tokens = 4096
        self.model_name = "DeepSeek-R1"




    def _to_azure_messages(self, message: Union[str, List[Union[str, Any]]]) -> List[Any]:
        if isinstance(message, str):
            return [UserMessage(content=message)]
        azure_messages = []
        for item in message:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                if role == "system":
                    azure_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    azure_messages.append(AssistantMessage(content=content))
                else:
                    azure_messages.append(UserMessage(content=content))
            else:
                azure_messages.append(UserMessage(content=item))
        return azure_messages

    def _iterate_stream(self, response) -> Iterator[str]:
        for update in response:
            if update.choices:
                content = update.choices[0].delta.content
                if content:
                    yield content

    def one_chat(self, message: Union[str, List[Union[str, Any]]], is_stream: bool = False) -> Union[str, Iterator[str]]:
        """执行单次聊天交互，不使用或存储聊天历史记录。"""
        request_messages = self._to_azure_messages(message)

        if is_stream:
            response = self.client.complete(
                stream=True,
                messages=request_messages,
                max_tokens=self.max_tokens,
                model=self.model_name
            )
            return self._iterate_stream(response)
        else:
            response = self.client.complete(
                messages=request_messages,
                max_tokens=self.max_tokens,
                model=self.model_name
            )
            return response.choices[0].message.content


    def text_chat(self, message: str, is_stream: bool = False) -> Union[str, Iterator[str]]:
        """处理文本消息并记录历史。"""
        # 将用户消息添加到聊天历史中
        self.history.append({"role": "user", "content": message})
        payload = {"messages": self.history, "max_tokens": self.max_tokens}
        

        if is_stream:
            # 返回流式响应（使用迭代器逐步返回）
            response = self.client.complete(payload, stream=True)
            return (choice.message['content'] for choice in response)

        # 普通响应：一次性返回完整的内容
        response = self.client.complete(payload)
        # 将模型的回应加入历史记录
        assistant_response = response.choices[0].message['content']
        self.history.append({"role": "assistant", "content": assistant_response})
        return assistant_response

    def tool_chat(self, user_message: str, tools: List[Dict[str, Any]], function_module: Any, is_stream: bool = False) -> Union[str, Iterator[str]]:
        # 处理可以访问外部工具的文本消息（目前不实现具体的工具调用）
        raise NotImplementedError("Tool chat is not implemented for AzureDeepSeekClient.")

    def tool_invoke(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        azure_messages = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                azure_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                azure_messages.append(AssistantMessage(content=content))
            else:
                azure_messages.append(UserMessage(content=content))

        response = self.client.complete({
            "messages": azure_messages,
            "max_tokens": self.max_tokens,
            "model": self.model_name
        })
        content = response.choices[0].message["content"] if response.choices else ""
        return {"content": content or "", "tool_calls": []}

    def audio_chat(self, message: str, audio_path: str) -> str:
        # Azure DeepSeek API不支持音频聊天
        raise NotImplementedError("Audio chat is not supported by AzureDeepSeekClient.")

    def video_chat(self, message: str, video_path: str) -> str:
        # Azure DeepSeek API不支持视频聊天
        raise NotImplementedError("Video chat is not supported by AzureDeepSeekClient.")

    def clear_chat(self):
        """清除聊天历史记录。"""
        self.history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """返回API使用情况统计信息（例如，token使用情况）。"""
        model_info = self.client.get_model_info()
        return {
            "model_name": model_info.model_name,
            "model_type": model_info.model_type,
            "model_provider_name": model_info.model_provider_name
        }
