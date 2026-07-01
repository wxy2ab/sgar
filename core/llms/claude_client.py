import inspect
from typing import List, Dict, Any, Optional, Union, Iterator
from anthropic import Anthropic
import json
import os
import base64
from PIL import Image
import io
from ..utils.retry import retry
from ._llm_api_client import LLMApiClient
from ..utils.config_setting import Config
from ..utils.handle_max_tokens import handle_max_tokens

class ClaudeClient(LLMApiClient):
    DEFAULT_MODEL = "claude-opus-4-8"
    supports_structured_output = True
    # Anthropic's Messages API declares max_tokens as a required positive int
    # (unlike OpenAI, which happily omits it). Falls back only when neither
    # the constructor nor the per-call max_tokens is set, so a caller who
    # never thought about it (e.g. LLMFactory.get_instance("ClaudeClient")
    # with no kwargs) gets a working request instead of a client-side crash.
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = 1,
                 top_p: float = 1.0,
                 top_k: int = 250,
                 max_tokens: Optional[int] = None,
                 stop_sequences: Optional[List[str]] = None,
                 thinking: Optional[bool] = None,
                 thinking_budget_tokens: int = 10000,
                 enable_prompt_caching: bool = False,
                 metadata: Optional[Dict[str, Any]] = None):
        config = Config()
        self.model = config.resolve_value(
            model,
            ("claude_model",),
            self.DEFAULT_MODEL,
        )
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences or []
        # Extended thinking: when enabled, Anthropic forbids sending
        # temperature/top_p/top_k at all (not just non-default values) and
        # requires max_tokens to exceed thinking_budget_tokens — see
        # _base_create_kwargs. budget_tokens is billed as output tokens.
        self.thinking = thinking
        self.thinking_budget_tokens = thinking_budget_tokens
        # Prompt caching (default OFF — additive, changes request shape):
        # wraps the system prompt and, for tool calls, the last tool
        # definition with cache_control so repeated turns / static tool
        # schemas aren't re-billed at full price.
        self.enable_prompt_caching = enable_prompt_caching
        self.metadata = metadata
        self.client = self._create_client(api_key)
        self.history = []
        self.system_message: str = ""
        self._response_format: Optional[Dict[str, Any]] = None
        self._last_thinking_content: Optional[str] = None
        self.stat = {
            "call_count": {"text_chat": 0, "image_chat": 0, "tool_chat": 0},
            'input_tokens': 0,
            'output_tokens': 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    def _create_client(self, api_key):
        config = Config()
        if api_key is None:
            api_key = config.get("claude_api_key")
        if not api_key:
            raise ValueError("API key not found. Please provide an API key or configure it in your settings.")
        return Anthropic(api_key=api_key)

    def set_system_message(self, system_message: str = "你是一个智能助手,擅长把复杂问题清晰明白通俗易懂地解答出来"):
        # Anthropic's Messages API does not accept role="system" inside
        # `messages` — the system prompt is a separate top-level `system`
        # param (see _build_system_param). Store it separately instead of
        # pushing it into self.history like the OpenAI-style clients do.
        self.system_message = system_message

    def set_response_format(self, fmt: Optional[Dict[str, Any]]) -> None:
        if fmt is not None and not isinstance(fmt, dict):
            raise TypeError("response_format must be a dict or None")
        self._response_format = fmt

    def _build_system_param(self, system_text: Optional[str] = None) -> Union[str, List[Dict[str, Any]], None]:
        text = system_text if system_text is not None else self.system_message
        if not text:
            return None
        if self.enable_prompt_caching:
            return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        return text

    def _build_thinking_param(self) -> Optional[Dict[str, Any]]:
        if not self.thinking:
            return None
        return {"type": "enabled", "budget_tokens": self.thinking_budget_tokens}

    def _base_create_kwargs(self, max_tokens: Optional[int] = None, system_override: Optional[str] = None) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens or self.DEFAULT_MAX_TOKENS,
        }
        system_param = self._build_system_param(system_override)
        if system_param is not None:
            kwargs["system"] = system_param
        if self.stop_sequences:
            kwargs["stop_sequences"] = self.stop_sequences
        if self.metadata:
            kwargs["metadata"] = self.metadata

        thinking_param = self._build_thinking_param()
        if thinking_param is not None:
            # Extended thinking disallows custom temperature/top_p/top_k.
            kwargs["thinking"] = thinking_param
        else:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            kwargs["top_k"] = self.top_k
        return kwargs

    def _split_system_from_messages(self, messages: List[Dict[str, Any]]) -> "tuple[Optional[str], List[Dict[str, Any]]]":
        """tool_invoke's contract (see base class) allows a full messages list
        that may include a role="system" entry. Anthropic rejects that role
        inside `messages`, so pull it out into the top-level `system` param.
        """
        system_parts = []
        remaining = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content")
                system_parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
            else:
                remaining.append(msg)
        combined = "\n\n".join(part for part in system_parts if part)
        return (combined or None), remaining

    def _json_tool_from_format(self, fmt: Dict[str, Any]) -> Dict[str, Any]:
        name = None
        input_schema = None
        if fmt.get("type") == "json_schema":
            schema_spec = fmt.get("json_schema")
            if isinstance(schema_spec, dict):
                name = schema_spec.get("name")
                input_schema = schema_spec.get("schema")
        return {
            "name": name or "emit_json",
            "description": "Return the requested data as this tool's input — this is the only way to answer.",
            "input_schema": input_schema or {"type": "object"},
        }

    def _extract_json_tool_output(self, response, tool_name: str) -> str:
        for block in response.content:
            if getattr(block, "type", "") == "tool_use" and getattr(block, "name", "") == tool_name:
                return json.dumps(block.input, ensure_ascii=False)
        # Model didn't use the forced tool (e.g. stopped early on max_tokens)
        # — fall back to whatever text it did produce so callers still get
        # something to parse instead of an empty string.
        return self._split_content_blocks(response.content)["text"]

    def _split_content_blocks(self, content_blocks) -> Dict[str, Any]:
        text_parts = []
        thinking_parts = []
        tool_uses = []
        for block in content_blocks:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif block_type == "tool_use":
                tool_uses.append(block)
        return {
            "text": "".join(text_parts),
            "thinking": "".join(thinking_parts),
            "tool_uses": tool_uses,
        }

    def _prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned_tools = [tool.copy() for tool in tools]
        for tool in cleaned_tools:
            tool.pop('output_schema', None)
        if self.enable_prompt_caching and cleaned_tools:
            cleaned_tools[-1] = dict(cleaned_tools[-1])
            cleaned_tools[-1]["cache_control"] = {"type": "ephemeral"}
        return cleaned_tools

    def _update_stats(self, source: Any):
        usage = source if isinstance(source, dict) else getattr(source, "usage", None)
        if not usage:
            return

        def _get(key: str) -> int:
            value = usage.get(key, 0) if isinstance(usage, dict) else getattr(usage, key, 0)
            return int(value or 0)

        input_tokens = _get("input_tokens")
        output_tokens = _get("output_tokens")
        self.stat['input_tokens'] += input_tokens
        self.stat['output_tokens'] += output_tokens
        self.stat['total_tokens'] += input_tokens + output_tokens
        self.stat['cache_creation_input_tokens'] += _get("cache_creation_input_tokens")
        self.stat['cache_read_input_tokens'] += _get("cache_read_input_tokens")

    @handle_max_tokens
    def text_chat(self, message: str, max_tokens: Optional[int] = None, is_stream: bool = False) -> Union[str, Iterator[str]]:
        if not self.system_message:
            self.set_system_message()
        if self._response_format is not None and is_stream:
            raise NotImplementedError("Structured JSON output is not supported together with streaming on ClaudeClient.")

        copy_history = self.history.copy()
        copy_history.append({"role": "user", "content": message})

        kwargs = self._base_create_kwargs(max_tokens)
        kwargs["messages"] = copy_history

        json_tool = None
        if self._response_format is not None:
            if self.thinking:
                # Anthropic requires tool_choice to be "auto" (or omitted)
                # whenever extended thinking is enabled — forcing a specific
                # tool is rejected by the API, so fail loudly instead of
                # sending a request that's guaranteed to 400.
                raise NotImplementedError(
                    "Structured JSON output via forced tool_choice is not supported together with "
                    "extended thinking on ClaudeClient (Anthropic requires tool_choice='auto' when "
                    "thinking is enabled)."
                )
            json_tool = self._json_tool_from_format(self._response_format)
            kwargs["tools"] = [json_tool]
            kwargs["tool_choice"] = {"type": "tool", "name": json_tool["name"]}

        self.stat["call_count"]["text_chat"] += 1
        if is_stream:
            return self._handle_stream_response(kwargs, message)
        else:
            response = self.client.messages.create(**kwargs)
            self._update_stats(response)
            if json_tool is not None:
                assistant_message = self._extract_json_tool_output(response, json_tool["name"])
            else:
                parsed = self._split_content_blocks(response.content)
                self._last_thinking_content = parsed["thinking"] or None
                assistant_message = parsed["text"]
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": assistant_message})
            return assistant_message

    def _handle_stream_response(self, kwargs, message=None):
        # Uses the SDK's stream() context manager + get_final_message()
        # rather than hand-parsing raw SSE events: get_final_message()
        # returns the SDK's own reconstructed content blocks (thinking
        # blocks with their required `signature` intact, etc.), which is
        # far less fragile than re-deriving that structure from deltas.
        full_response = ""
        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                if getattr(event, "type", "") == 'content_block_delta' and event.delta.type == 'text_delta':
                    text = event.delta.text
                    full_response += text
                    yield text
            final_message = stream.get_final_message()

        self._update_stats(final_message)
        parsed = self._split_content_blocks(final_message.content)
        if parsed["thinking"]:
            self._last_thinking_content = parsed["thinking"]
        # one_chat calls this with message=None and must not touch history
        # (its documented contract is "no history read/write").
        if message is not None:
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": full_response})

    def tool_chat(self, user_message: str, tools: List[Dict[str, Any]], function_module: Any, max_tokens: Optional[int] = None, is_stream: bool = False) -> Union[str, Iterator[str]]:
        if not self.system_message:
            self.set_system_message()
        self.history.append({"role": "user", "content": user_message})
        cleaned_tools = self._prepare_tools(tools)

        kwargs = self._base_create_kwargs(max_tokens)
        kwargs["messages"] = self.history
        kwargs["tools"] = cleaned_tools

        self.stat["call_count"]["tool_chat"] += 1

        if is_stream:
            return self._handle_tool_stream(kwargs, function_module)
        else:
            response = self.client.messages.create(**kwargs)
            self._update_stats(response)
            return self._handle_tool_response(response, function_module, max_tokens)

    def tool_invoke(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        system_from_messages, remaining_messages = self._split_system_from_messages(messages)
        cleaned_tools = self._prepare_tools(tools)

        kwargs = self._base_create_kwargs(system_override=system_from_messages)
        kwargs["messages"] = remaining_messages
        kwargs["tools"] = cleaned_tools
        kwargs["stream"] = False

        response = self.client.messages.create(**kwargs)
        self._update_stats(response)
        parsed = self._split_content_blocks(response.content)
        tool_calls = [
            {
                "id": getattr(block, "id", ""),
                "name": getattr(block, "name", ""),
                "input": getattr(block, "input", {}),
            }
            for block in parsed["tool_uses"]
        ]
        return self._normalize_tool_invoke_response(parsed["text"], tool_calls)

    def _handle_tool_stream(self, kwargs, function_module):
        # As in _handle_stream_response: rely on the SDK's get_final_message()
        # for the authoritative content-block list instead of hand-rebuilding
        # it from deltas. This matters even more here than in the plain-text
        # path, because a hand-rebuilt assistant turn that's replayed back
        # into history for tool continuation must be byte-for-byte valid
        # (in particular, a thinking block replayed without its original
        # `signature` — which streaming deltas don't expose piecemeal — is
        # rejected by the API when thinking is enabled).
        assistant_message = ""
        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                if getattr(event, "type", "") == 'content_block_delta' and event.delta.type == 'text_delta':
                    assistant_message += event.delta.text
                    yield event.delta.text
            final_message = stream.get_final_message()

        self._update_stats(final_message)
        parsed = self._split_content_blocks(final_message.content)
        if parsed["thinking"]:
            self._last_thinking_content = parsed["thinking"]
        tool_use_blocks = parsed["tool_uses"]

        if tool_use_blocks:
            self.history.append({"role": "assistant", "content": final_message.content})
            tool_results = []
            for block in tool_use_blocks:
                function_name = block.name
                function_args = block.input if isinstance(block.input, dict) else {}
                tool_result = self._execute_function(function_name, function_args, function_module)
                yield f"\n使用工具: {function_name}\n参数: {function_args}\n工具结果: {tool_result}\n"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(tool_result),
                })
            self.history.append({"role": "user", "content": tool_results})

            follow_up_kwargs = dict(kwargs)
            follow_up_kwargs["messages"] = self.history
            with self.client.messages.stream(**follow_up_kwargs) as follow_up_stream:
                for chunk in follow_up_stream:
                    if getattr(chunk, "type", "") == 'content_block_delta' and chunk.delta.type == 'text_delta':
                        yield chunk.delta.text
                final_follow_up = follow_up_stream.get_final_message()
            self._update_stats(final_follow_up)
            self.history.append({"role": "assistant", "content": final_follow_up.content})
        else:
            self.history.append({"role": "assistant", "content": assistant_message})

    def _parse_function_args(self, args_str: str, function_name: str, function_module: Any) -> Dict[str, Any]:
        if not args_str.strip():
            return {}

        try:
            args = json.loads(args_str)
            if isinstance(args, dict):
                return args
            else:
                return {"raw_input": args}
        except json.JSONDecodeError:
            return {"raw_input": args_str}

    def _execute_function(self, function_name: str, function_args: Dict[str, Any], function_module: Any) -> Any:
        if hasattr(function_module, function_name):
            function = getattr(function_module, function_name)
            try:
                sig = inspect.signature(function)
                if len(sig.parameters) == 0:
                    return function()
                elif len(sig.parameters) == 1 and "raw_input" in function_args:
                    return function(function_args["raw_input"])
                else:
                    return function(**function_args)
            except Exception as e:
                return f"Error executing {function_name}: {str(e)}"
        else:
            return f"Function {function_name} not found in the provided module."

    def _handle_tool_response(self, response, function_module, max_tokens):
        parsed = self._split_content_blocks(response.content)
        assistant_message = parsed["text"]
        tool_use_blocks = parsed["tool_uses"]

        self.history.append({"role": "assistant", "content": response.content})

        if tool_use_blocks:
            tool_results = []
            for block in tool_use_blocks:
                function_name = block.name
                function_args = block.input if isinstance(block.input, dict) else {}
                tool_result = self._execute_function(function_name, function_args, function_module)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(tool_result),
                })
            self.history.append({"role": "user", "content": tool_results})

            kwargs = self._base_create_kwargs(max_tokens)
            kwargs["messages"] = self.history
            kwargs["stream"] = False
            final_response = self.client.messages.create(**kwargs)
            self._update_stats(final_response)
            final_assistant_message = self._split_content_blocks(final_response.content)["text"]
            self.history.append({"role": "assistant", "content": final_assistant_message})
            return f"*首轮消息：*{assistant_message}\n*使用工具：*{[b.name for b in tool_use_blocks]}\n*最终结果：*{final_assistant_message}"
        else:
            return assistant_message

    def one_chat(self, message: Union[str, List[Union[str, Any]]], max_tokens: Optional[int] = None, is_stream: bool = False) -> Union[str, Iterator[str]]:
        if self._response_format is not None and is_stream:
            raise NotImplementedError("Structured JSON output is not supported together with streaming on ClaudeClient.")

        if isinstance(message, str):
            messages = [{"role": "user", "content": message}]
            system_override = None
        else:
            system_override, messages = self._split_system_from_messages(message)

        kwargs = self._base_create_kwargs(max_tokens, system_override=system_override)
        kwargs["messages"] = messages

        json_tool = None
        if self._response_format is not None:
            if self.thinking:
                # Anthropic requires tool_choice to be "auto" (or omitted)
                # whenever extended thinking is enabled — forcing a specific
                # tool is rejected by the API, so fail loudly instead of
                # sending a request that's guaranteed to 400.
                raise NotImplementedError(
                    "Structured JSON output via forced tool_choice is not supported together with "
                    "extended thinking on ClaudeClient (Anthropic requires tool_choice='auto' when "
                    "thinking is enabled)."
                )
            json_tool = self._json_tool_from_format(self._response_format)
            kwargs["tools"] = [json_tool]
            kwargs["tool_choice"] = {"type": "tool", "name": json_tool["name"]}

        self.stat["call_count"]["text_chat"] += 1
        if is_stream:
            return self._handle_stream_response(kwargs)
        else:
            response = self.client.messages.create(**kwargs)
            self._update_stats(response)
            if json_tool is not None:
                return self._extract_json_tool_output(response, json_tool["name"])
            parsed = self._split_content_blocks(response.content)
            self._last_thinking_content = parsed["thinking"] or None
            return parsed["text"]

    def image_chat(self, message: str, image_path: str, max_tokens: Optional[int] = None) -> str:
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img_format = img.format or "PNG"
            img.save(buffered, format=img_format)
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        image_message = {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": f"image/{img_format.lower()}", "data": base64_image}},
                {"type": "text", "text": message}
            ]
        }
        self.history.append(image_message)

        kwargs = self._base_create_kwargs(max_tokens)
        kwargs["messages"] = self.history
        kwargs["stream"] = False
        response = self.client.messages.create(**kwargs)

        assistant_message = self._split_content_blocks(response.content)["text"]
        self.history.append({"role": "assistant", "content": assistant_message})
        self._update_stats(response)
        self.stat["call_count"]["image_chat"] += 1
        return assistant_message

    def clear_chat(self) -> None:
        self.history.clear()

    def get_stats(self) -> Dict[str, Any]:
        return self.stat

    def audio_chat(self, message: str, audio_path: str) -> str:
        raise NotImplementedError("Audio chat is not supported in this version of Claude API client.")

    def video_chat(self, message: str, video_path: str) -> str:
        raise NotImplementedError("Video chat is not supported in this version of Claude API client.")
