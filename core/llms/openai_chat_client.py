from typing import Iterator, List, Dict, Any, Optional, Union
from openai import OpenAI
import json
import base64
import httpx
from ._llm_api_client import LLMApiClient
from ._empty_content_retry import (
    DEFAULT_EMPTY_CONTENT_RETRIES,
    is_blank,
    reissue_completion_on_empty,
)
from ..utils.config_setting import Config
from ..utils.handle_max_tokens import handle_max_tokens
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_fixed


class OpenAIChatClient(LLMApiClient):
    """Chat-Completions-based OpenAI client.

    The constructor mirrors :class:`~core.llms.moonshot_client.MoonShotClient`
    parameter-for-parameter (same names/order/defaults for the first ten
    args) so this class can later serve as a drop-in replacement / new base
    class for MoonShotClient and its OpenAI-compatible subclasses
    (VolcCodingClient, GLMOpenAIClient, PPioOpenAIClient, ...): existing
    ``super().__init__(api_key, base_url, max_tokens=..., enable_thinking=...)``
    -style calls keep working without a ``TypeError``. This is a
    call-signature guarantee, not a promise that every individual
    parameter's request-time semantics are byte-identical to whatever
    MoonShotClient does with it internally — e.g. here ``max_tokens`` is
    wired live into every request (the standard, expected behavior for a
    ``max_tokens`` constructor arg), which may or may not match
    MoonShotClient's current internal handling of that same argument.
    New parameters are appended after ``reasoning_effort``.
    """

    supports_structured_output = True
    MODEL_CONFIG_KEYS = ("openai_chat_model", "openai_model")
    DEFAULT_MODEL = "gpt-5.5"

    def __init__(self, api_key: str = "", base_url: str = "https://api.openai.com/v1",
                 max_tokens: Optional[int] = None, temperature: float = 1, top_p: Optional[float] = None,
                 presence_penalty: Optional[float] = 0, frequency_penalty: Optional[float] = 0,
                 stop: Optional[Union[str, List[str]]] = None,
                 enable_thinking: Optional[bool] = None, reasoning_effort: Optional[str] = "high",
                 model: Optional[str] = None, verbosity: Optional[str] = None,
                 parallel_tool_calls: Optional[bool] = None, seed: Optional[int] = None,
                 service_tier: Optional[str] = None, extra_body: Optional[Dict[str, Any]] = None,
                 context_window_tokens: Optional[int] = None,
                 ):
        config = Config()
        if api_key == "" and config.has_key("openai_api_key"):
            api_key = config.get("openai_api_key")

        http_client = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=200),
            timeout=httpx.Timeout(timeout=600.0, connect=60.0, read=600.0, write=120.0)
        )

        self.client = OpenAI(
            # The openai SDK itself raises openai.OpenAIError at construction
            # time when api_key is falsy and neither OPENAI_API_KEY nor
            # OPENAI_ADMIN_KEY is set in the environment. Every other client
            # in this codebase defers a missing/unconfigured key to the first
            # real API call instead of crashing at construction (e.g.
            # SimpleDeepSeekClient) — substituting a non-empty placeholder
            # here preserves that behavior; a genuinely missing key still
            # fails naturally as a 401 on first use.
            api_key=api_key or "not-configured",
            base_url=base_url,
            http_client=http_client,
            max_retries=5
        )
        self.chat_count = 0
        self.token_count = 0
        self.prompt_token_count = 0
        self.completion_token_count = 0
        # Cached-prompt / reasoning-token accounting surfaced by the latest
        # OpenAI usage payloads (usage.prompt_tokens_details.cached_tokens,
        # usage.completion_tokens_details.reasoning_tokens). Cache hits bill
        # at a steep discount and reasoning tokens are billed as output but
        # invisible in the response text, so both are worth tracking
        # separately — mirrors SimpleDeepSeekClient's accounting.
        self.cached_prompt_token_count = 0
        self.reasoning_token_count = 0
        self.truncated_count = 0
        self._last_finish_reason: Optional[str] = None
        self._last_reasoning_content: Optional[str] = None
        self.history = []
        self.model = config.resolve_value(
            model,
            self.MODEL_CONFIG_KEYS,
            self.DEFAULT_MODEL,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.stop = stop
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.parallel_tool_calls = parallel_tool_calls
        self.seed = seed
        self.service_tier = service_tier
        self.extra_body = dict(extra_body) if extra_body else {}
        self._response_format: Optional[Dict[str, Any]] = None
        # Total context window (tokens) — cc's query_loop reads this via
        # getattr(llm_client, "context_window_tokens", None) to drive
        # token-budget-aware conversation compaction instead of a fixed
        # message-count trim (see query_loop.py's _COMPACT_TRIGGER_RATIO).
        # That mechanism was written against SimpleDeepSeekClient (which sets
        # this unconditionally) but is client-agnostic by design; None here
        # (the default, matching every subclass that doesn't pass it) falls
        # back to the legacy message-count trim, so this is purely additive.
        self.context_window_tokens = context_window_tokens

    def set_system_message(self, system_message: str = "你是一个智能助手,擅长把复杂问题清晰明白通俗易懂地解答出来"):
        self.history = [{"role": "system", "content": system_message}]

    def set_response_format(self, fmt: Optional[Dict[str, Any]]) -> None:
        if fmt is not None and not isinstance(fmt, dict):
            raise TypeError("response_format must be a dict or None")
        self._response_format = fmt

    @handle_max_tokens
    def text_chat(self, message: str, is_stream: bool = False) -> Union[str, Iterator[str]]:
        if not self.history:
            self.set_system_message()
        self.history.append({"role": "user", "content": message})
        return self._create_chat_completion(self.history, is_stream)

    @sleep_and_retry
    @limits(calls=20, period=1)
    @retry(stop=stop_after_attempt(12), wait=wait_fixed(5))
    def one_chat(self, message: str, is_stream: bool = False) -> Union[str, Iterator[str]]:
        if not self.history:
            self.set_system_message()
        msg = [{"role": "user", "content": message}] if isinstance(message, str) else message
        return self._create_chat_completion(msg, is_stream)

    def tool_chat(self, user_message: str, tools: List[Dict[str, Any]], function_module: Any, is_stream: bool = False) -> Union[str, Iterator[str]]:
        if not self.history:
            self.set_system_message()
        self.history.append({"role": "user", "content": user_message})
        if is_stream:
            return self._unified_tool_stream(self.history, tools, function_module)
        else:
            response = self._create_chat_completion(self.history, is_stream, tools, raw_response=True)
            return self._process_tool_response(response, tools, function_module)

    def image_chat(self, message: str, image_path_or_url: str, is_stream: bool = False) -> Union[str, Iterator[str]]:
        if not self.history:
            self.set_system_message()

        if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
            image_content = {"type": "image_url", "image_url": {"url": image_path_or_url}}
        else:
            image_content = {"type": "image_url", "image_url": {"url": self._encode_image_to_base64(image_path_or_url)}}

        self.history.append({
            "role": "user",
            "content": [{"type": "text", "text": message}, image_content],
        })
        return self._create_chat_completion(self.history, is_stream)

    def audio_chat(self, message: str, audio_path: str, is_stream: bool = False) -> Union[str, Iterator[str]]:
        if not self.history:
            self.set_system_message()

        lower_path = audio_path.lower()
        audio_format = "mp3" if lower_path.endswith(".mp3") else "wav"
        with open(audio_path, "rb") as audio_file:
            audio_b64 = base64.b64encode(audio_file.read()).decode("utf-8")

        self.history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": message},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": audio_format}},
            ],
        })
        return self._create_chat_completion(self.history, is_stream)

    def video_chat(self, message: str, video_path: str) -> str:
        raise NotImplementedError("OpenAI Chat Completions API does not support video input.")

    def _encode_image_to_base64(self, image_path: str) -> str:
        ext = image_path.rsplit(".", 1)[-1].lower() if "." in image_path else "jpeg"
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "jpeg")
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/{mime};base64,{base64_image}"

    def _unified_tool_stream(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]], function_module: Any) -> Iterator[str]:
        try:
            response_stream = self._create_chat_completion(messages, True, tools, raw_response=True)
            full_response = ""
            tool_calls = []

            for chunk in response_stream:
                if isinstance(chunk, str):
                    content = chunk
                elif hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    content = delta.content if hasattr(delta, 'content') and delta.content is not None else None
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if tool_call.index >= len(tool_calls):
                                tool_calls.append({
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments or ""}
                                })
                            else:
                                tool_calls[tool_call.index]["function"]["arguments"] += tool_call.function.arguments or ""
                if content:
                    yield content
                    full_response += content

            if tool_calls:
                tool_outputs = self._execute_tool_calls(tool_calls, function_module)
                tool_results = []
                for tool_output in tool_outputs:
                    result = f"工具 {tool_output['tool_call_id']} 返回结果: {tool_output['content']}"
                    tool_results.append(result)
                    yield result + "\n"

                tool_result_message = "\n".join(tool_results)
                messages.append({"role": "assistant", "content": f"{full_response}\n\n工具调用结果:\n{tool_result_message}"})

                explanation_request = "请解释上述工具调用的结果，并提供一个简洁明了的回答。"
                messages.append({"role": "user", "content": explanation_request})

                explanation_stream = self._create_chat_completion(messages, True, tools, raw_response=True)
                for chunk in explanation_stream:
                    if isinstance(chunk, str):
                        yield chunk
                    elif hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        content = delta.content if hasattr(delta, 'content') and delta.content is not None else None
                        if content:
                            yield content
            elif full_response.strip():
                yield f"\n{full_response}\n"
            else:
                yield "\n无法生成回答。请尝试重新提问。\n"
        except Exception as e:
            yield f"发生错误: {str(e)}"

        self.history = [msg for msg in messages[-5:] if msg.get('content', '').strip()]

    def _create_chat_completion(self, messages: List[Dict[str, str]], is_stream: bool, tools: List[Dict[str, Any]] = None, raw_response: bool = False) -> Union[str, Iterator[str]]:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": is_stream,
            "timeout": 600,
        }
        if is_stream:
            kwargs["stream_options"] = {"include_usage": True}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        extra_body = dict(self.extra_body) if self.extra_body else {}
        if self.enable_thinking is not None:
            extra_body["enable_thinking"] = self.enable_thinking
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.verbosity is not None:
            kwargs["verbosity"] = self.verbosity

        if extra_body:
            kwargs["extra_body"] = extra_body

        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty
        if self.stop is not None:
            kwargs["stop"] = self.stop
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.service_tier is not None:
            kwargs["service_tier"] = self.service_tier
        if tools:
            kwargs["tools"] = tools
            if self.parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = self.parallel_tool_calls
        # Skip JSON mode when tools are present: OpenAI-compatible APIs treat
        # the tools + response_format={"type":"json_object"} combination as
        # ambiguous (tool_calls may not respect the JSON envelope; structured
        # outputs can conflict with tool schemas). Mirrors MoonShotClient.
        if self._response_format and not tools:
            kwargs["response_format"] = self._response_format

        completion = self.client.chat.completions.create(**kwargs)
        if is_stream:
            return completion if raw_response else self._process_stream(completion)
        else:
            if raw_response:
                return completion
            if not completion.choices:
                raise RuntimeError("LLM API returned empty choices")
            response = self._extract_completion_text(completion)
            # A reasoning-capable model can spend the whole turn on hidden
            # reasoning_content and return empty visible content with
            # finish_reason="stop" (no exception, so one_chat's @retry never
            # fires). Only the plain text path can be silently empty — the
            # tool path may legitimately carry empty content alongside
            # tool_calls, so it is left untouched. See _empty_content_retry.
            if tools or not is_blank(response):
                return response
            return reissue_completion_on_empty(
                create=lambda call_kwargs: self.client.chat.completions.create(**call_kwargs),
                extract=self._extract_completion_text,
                kwargs=kwargs,
                enable_thinking=self.enable_thinking,
                retries=self._EMPTY_CONTENT_RETRIES,
                client_name=type(self).__name__,
                last_finish_reason=self._last_finish_reason,
                last_reasoning=self._last_reasoning_content or "",
            )

    #: Re-issues after an empty visible content (last attempt disables
    #: thinking). Subclasses may override; VolcCodingClient relies on this.
    _EMPTY_CONTENT_RETRIES = DEFAULT_EMPTY_CONTENT_RETRIES

    def _extract_completion_text(self, completion) -> Optional[str]:
        """Pull visible content from a non-stream completion, recording
        finish_reason / truncation / reasoning / usage as a side effect.
        Returns None when the completion carries no choices (treated as empty
        by the retry loop)."""
        if not completion.choices:
            return None
        choice = completion.choices[0]
        self._last_finish_reason = getattr(choice, "finish_reason", None)
        if self._last_finish_reason == "length":
            self.truncated_count += 1
        self._last_reasoning_content = getattr(choice.message, "reasoning_content", None)
        self._update_stats(completion.usage)
        return choice.message.content

    @retry(stop=stop_after_attempt(4), wait=wait_fixed(5))
    def tool_invoke(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Unlike one_chat (above), this had no retry until this decorator was
        # added: a single transient httpx.RemoteProtocolError / APIConnectionError
        # (observed against the Volc coding endpoint — a long silent "thinking"
        # gap on this stream=False request can get idle-closed by an
        # intermediary proxy, the same root cause already diagnosed for
        # DeepSeek reasoning clients elsewhere in this project) used to burn an
        # entire ccx doc-mode investigator dimension's tool-call budget for one
        # bad connection. Mirrors the identical fix in MoonShotClient.tool_invoke.
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
            "timeout": 600,
            "tools": tools,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        extra_body = dict(self.extra_body) if self.extra_body else {}
        if self.enable_thinking is not None:
            extra_body["enable_thinking"] = self.enable_thinking
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.verbosity is not None:
            kwargs["verbosity"] = self.verbosity
        if extra_body:
            kwargs["extra_body"] = extra_body
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty
        if self.stop is not None:
            kwargs["stop"] = self.stop
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.service_tier is not None:
            kwargs["service_tier"] = self.service_tier
        if self.parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = self.parallel_tool_calls
        # No response_format on tool paths — see _create_chat_completion gate
        # for rationale.

        completion = self.client.chat.completions.create(**kwargs)
        self._update_stats(completion.usage)
        if not completion.choices:
            return self._normalize_tool_invoke_response("", None)
        message = completion.choices[0].message
        return self._normalize_tool_invoke_response(
            getattr(message, "content", "") or "",
            getattr(message, "tool_calls", None)
        )

    def _process_tool_response(self, response, tools: List[Dict[str, Any]], function_module: Any) -> str:
        if not response.choices:
            return ""
        assistant_output = response.choices[0].message
        self._update_stats(response.usage)

        if hasattr(assistant_output, 'tool_calls') and assistant_output.tool_calls:
            self.history.append({"role": "assistant", "content": assistant_output.content, "tool_calls": assistant_output.tool_calls})
            tool_outputs = self._execute_tool_calls(assistant_output.tool_calls, function_module)
            self.history.extend(tool_outputs)
            second_response = self._create_chat_completion(self.history, False, tools, raw_response=True)
            if not second_response.choices:
                final_output = ""
            else:
                self._update_stats(second_response.usage)
                final_output = second_response.choices[0].message.content or ""
        else:
            self.history.append({"role": "assistant", "content": assistant_output.content})
            final_output = assistant_output.content

        return final_output

    def _process_stream(self, stream) -> Iterator[str]:
        full_response = ""
        reasoning_response = ""
        usage_updated = False
        for chunk in stream:
            usage = getattr(chunk, 'usage', None)
            if usage is not None:
                self._update_stats(usage)
                usage_updated = True
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                reasoning_delta = getattr(delta, 'reasoning_content', None)
                if reasoning_delta:
                    reasoning_response += reasoning_delta
                if hasattr(delta, 'content') and delta.content:
                    full_response += delta.content
                    yield delta.content
        if reasoning_response:
            self._last_reasoning_content = reasoning_response
        if not usage_updated:
            self._update_stats(None)
        self.history.append({"role": "assistant", "content": full_response})

    def _execute_tool_calls(self, tool_calls, function_module: Any) -> List[Dict[str, str]]:
        tool_outputs = []
        for tool_call in tool_calls:
            tc_function = getattr(tool_call, "function", None) or (tool_call.get("function") if isinstance(tool_call, dict) else None)
            tool_name = getattr(tc_function, "name", None) or (tc_function.get("name", "") if isinstance(tc_function, dict) else "")
            raw_args = getattr(tc_function, "arguments", None) or (tc_function.get("arguments", "{}") if isinstance(tc_function, dict) else "{}")
            tc_id = getattr(tool_call, "id", None) or (tool_call.get("id", "") if isinstance(tool_call, dict) else "")
            try:
                tool_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
            except (json.JSONDecodeError, TypeError):
                tool_args = {}

            if hasattr(function_module, tool_name):
                tool_func = getattr(function_module, tool_name)
                try:
                    tool_output = tool_func(**tool_args)
                    tool_outputs.append({
                        "role": "tool",
                        "content": str(tool_output),
                        "tool_call_id": tc_id,
                    })
                except Exception as e:
                    tool_outputs.append({
                        "role": "tool",
                        "content": f"Error executing {tool_name}: {str(e)}",
                        "tool_call_id": tc_id,
                    })
            else:
                tool_outputs.append({
                    "role": "tool",
                    "content": f"Error: Function {tool_name} not found.",
                    "tool_call_id": tc_id,
                })

        return tool_outputs

    def _extract_usage_counts(self, usage: Any) -> Dict[str, int]:
        normalized_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        if usage is None:
            return normalized_usage

        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif not isinstance(usage, dict):
            usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key, 0) if isinstance(usage, dict) else 0
            try:
                normalized_usage[key] = int(value or 0)
            except (TypeError, ValueError):
                normalized_usage[key] = 0

        if normalized_usage["total_tokens"] == 0:
            normalized_usage["total_tokens"] = (
                normalized_usage["prompt_tokens"] +
                normalized_usage["completion_tokens"]
            )

        prompt_details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
        if isinstance(prompt_details, dict):
            try:
                normalized_usage["cached_tokens"] = int(prompt_details.get("cached_tokens", 0) or 0)
            except (TypeError, ValueError):
                pass

        completion_details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        if isinstance(completion_details, dict):
            try:
                normalized_usage["reasoning_tokens"] = int(completion_details.get("reasoning_tokens", 0) or 0)
            except (TypeError, ValueError):
                pass

        return normalized_usage

    def _update_stats(self, usage: Any):
        self.chat_count += 1
        usage_counts = self._extract_usage_counts(usage)
        self.prompt_token_count += usage_counts["prompt_tokens"]
        self.completion_token_count += usage_counts["completion_tokens"]
        self.token_count += usage_counts["total_tokens"]
        self.cached_prompt_token_count += usage_counts["cached_tokens"]
        self.reasoning_token_count += usage_counts["reasoning_tokens"]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chats": self.chat_count,
            "total_tokens": self.token_count,
            "prompt_tokens": self.prompt_token_count,
            "completion_tokens": self.completion_token_count,
            "cached_prompt_tokens": self.cached_prompt_token_count,
            "reasoning_tokens": self.reasoning_token_count,
            "truncated_count": self.truncated_count,
        }

    def clear_chat(self) -> None:
        self.history = []
