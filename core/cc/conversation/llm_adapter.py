from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import json
import re
import threading
from typing import Any, Callable

from ..errors import CCError


class LLMAdapterError(CCError):
    error_code = "QE1002"


@dataclass(slots=True)
class StandardizedLLMResponse:
    raw: Any
    payload: dict[str, Any]
    text: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    continue_requested: bool = False
    stop_reason: str | None = None


class LLMAdapter:
    def __init__(self, llm_client: Any, *, timeout_seconds: float | None = 240.0) -> None:
        self.llm_client = llm_client
        self.timeout_seconds = timeout_seconds

    async def invoke(self, *, system_prompt: str, user_text: str, tools: list[dict[str, Any]] | None = None) -> Any:
        if tools and hasattr(self.llm_client, "tool_invoke"):
            method = self.llm_client.tool_invoke
            return await self._invoke_with_timeout(
                lambda: self._invoke_tool_invoke(system_prompt=system_prompt, user_text=user_text, tools=tools),
                method=method,
            )
        elif hasattr(self.llm_client, "one_chat"):
            method = self.llm_client.one_chat
            return await self._invoke_with_timeout(
                lambda: self._invoke_chat_method(method, system_prompt=system_prompt, user_text=user_text),
                method=method,
            )
        elif hasattr(self.llm_client, "text_chat"):
            method = self.llm_client.text_chat
            return await self._invoke_with_timeout(
                lambda: self._invoke_chat_method(method, system_prompt=system_prompt, user_text=user_text),
                method=method,
            )
        elif callable(self.llm_client):
            return await self._invoke_with_timeout(
                lambda: self._invoke_callable(system_prompt=system_prompt, user_text=user_text),
                method=self.llm_client,
            )
        else:
            raise LLMAdapterError("Unsupported llm_client interface.", error_code="QE1002")

    def _invoke_chat_method(self, method: Any, *, system_prompt: str, user_text: str) -> Any:
        try:
            signature = inspect.signature(method)
            if "system_prompt" in signature.parameters:
                return method(user_text, system_prompt=system_prompt)
        except (TypeError, ValueError):
            pass
        if getattr(method, "__name__", "") == "one_chat":
            return method(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ]
            )
        if hasattr(self.llm_client, "set_system_message"):
            self.llm_client.set_system_message(system_prompt)
            return method(user_text)
        return method(self._merge_prompt(system_prompt=system_prompt, user_text=user_text))

    def _invoke_tool_invoke(self, *, system_prompt: str, user_text: str, tools: list[dict[str, Any]]) -> Any:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        return self.llm_client.tool_invoke(messages, tools)

    async def invoke_with_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Invoke LLM with a full conversation history instead of (system, user) pair."""
        if tools and hasattr(self.llm_client, "tool_invoke"):
            method = self.llm_client.tool_invoke
            return await self._invoke_with_timeout(
                lambda: method(messages, tools),
                method=method,
            )
        elif hasattr(self.llm_client, "one_chat"):
            method = self.llm_client.one_chat
            return await self._invoke_with_timeout(
                lambda: method(messages),
                method=method,
            )
        else:
            system_prompt = ""
            user_text = ""
            for msg in messages:
                role = msg.get("role", "")
                if role == "system" and not system_prompt:
                    system_prompt = str(msg.get("content", ""))
            for msg in reversed(messages):
                role = msg.get("role", "")
                if role == "user" and not user_text:
                    user_text = str(msg.get("content", ""))
                    break
            return await self.invoke(system_prompt=system_prompt, user_text=user_text, tools=tools)

    async def complete_with_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> StandardizedLLMResponse:
        """Complete with a full conversation history."""
        raw = await self.invoke_with_messages(messages=messages, tools=tools)
        return self.normalize(raw)

    def _invoke_callable(self, *, system_prompt: str, user_text: str) -> Any:
        try:
            signature = inspect.signature(self.llm_client)
            if "system_prompt" in signature.parameters:
                if "user_text" in signature.parameters:
                    return self.llm_client(user_text=user_text, system_prompt=system_prompt)
                return self.llm_client(user_text, system_prompt=system_prompt)
        except (TypeError, ValueError):
            pass
        return self.llm_client(self._merge_prompt(system_prompt=system_prompt, user_text=user_text))

    def _merge_prompt(self, *, system_prompt: str, user_text: str) -> str:
        return f"{system_prompt}\n\n{user_text}"

    async def _invoke_with_timeout(self, operation: Callable[[], Any], *, method: Any) -> Any:
        if self.timeout_seconds is None:
            return await self._execute_operation(operation, method=method)
        # A coroutine method can be cancelled cleanly, so ``wait_for`` enforces
        # the deadline directly. A *synchronous* method runs in a worker thread,
        # and ``asyncio.wait_for`` CANNOT cancel a running thread: on timeout it
        # chains to the executor future and blocks until the thread returns, so a
        # wedged reasoning client (dead connection that never raises) defeats the
        # timeout entirely and the whole call hangs forever. Run sync calls in a
        # daemon thread we can ABANDON at the deadline instead — mirrors
        # ``core/ccx/agents/governed_goal._call_llm``.
        if inspect.iscoroutinefunction(method):
            try:
                return await asyncio.wait_for(
                    self._execute_operation(operation, method=method),
                    timeout=float(self.timeout_seconds),
                )
            except asyncio.TimeoutError as exc:
                raise LLMAdapterError(
                    f"LLM request timed out after {self.timeout_seconds:g} seconds.",
                    error_code="QE1009",
                ) from exc
        return await self._execute_sync_with_deadline(
            operation, float(self.timeout_seconds)
        )

    async def _execute_operation(self, operation: Callable[[], Any], *, method: Any) -> Any:
        if inspect.iscoroutinefunction(method):
            result = operation()
        else:
            result = await asyncio.to_thread(operation)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _execute_sync_with_deadline(self, operation: Callable[[], Any], timeout: float) -> Any:
        """Run a blocking ``operation`` in a daemon thread bounded by ``timeout``.

        Unlike ``asyncio.wait_for(asyncio.to_thread(...))`` — which cannot cancel a
        running thread and so blocks indefinitely on a wedged reasoning client —
        this abandons the thread once the deadline passes and raises a timeout
        error so the caller (cc_query_loop / v5) can retry. The abandoned thread
        is a daemon, so a permanently-hung call never blocks process exit.
        """
        loop = asyncio.get_running_loop()
        done = asyncio.Event()
        box: dict[str, Any] = {}

        def _worker() -> None:
            try:
                box["result"] = operation()
            except BaseException as exc:  # noqa: BLE001 — propagate, don't swallow
                box["error"] = exc
            finally:
                try:
                    loop.call_soon_threadsafe(done.set)
                except RuntimeError:
                    pass  # loop already closed; this call was abandoned on timeout

        threading.Thread(target=_worker, name="cc-llm-call", daemon=True).start()
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise LLMAdapterError(
                f"LLM request timed out after {timeout:g} seconds.",
                error_code="QE1009",
            ) from exc
        if "error" in box:
            raise box["error"]
        result = box.get("result")
        if inspect.isawaitable(result):
            return await result
        return result

    async def complete(self, *, system_prompt: str, user_text: str, tools: list[dict[str, Any]] | None = None) -> StandardizedLLMResponse:
        raw = await self.invoke(system_prompt=system_prompt, user_text=user_text, tools=tools)
        return self.normalize(raw)

    async def complete_with_continue(
        self,
        *,
        system_prompt: str,
        user_text: str,
        max_auto_continue: int,
        continue_prompt_builder: Callable[[str, str], str] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[StandardizedLLMResponse, int]:
        response = await self.complete(system_prompt=system_prompt, user_text=user_text, tools=tools)
        content_parts = [response.content]
        continue_count = 0
        while response.continue_requested and not response.tool_calls and continue_count < max_auto_continue:
            if continue_prompt_builder is None:
                break
            continue_count += 1
            continuation_prompt = continue_prompt_builder(user_text, "".join(content_parts))
            response = await self.complete(system_prompt=system_prompt, user_text=continuation_prompt, tools=tools)
            content_parts.append(response.content)
        if len(content_parts) > 1:
            merged_payload = dict(response.payload)
            merged_payload["content"] = "".join(content_parts).strip()
            response = self.normalize(merged_payload)
        return response, continue_count

    def normalize(self, response: Any) -> StandardizedLLMResponse:
        payload = self._normalize_payload(response)
        content = str(payload.get("content", ""))
        tool_calls = payload.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []
        # Preserve the dict-only invariant for the rest of the pipeline.
        tool_calls = [c for c in tool_calls if isinstance(c, dict)]
        stop_reason = self._extract_stop_reason(payload)
        return StandardizedLLMResponse(
            raw=response,
            payload=payload,
            text=self.coerce_text(response),
            content=content,
            tool_calls=tool_calls,
            continue_requested=self._needs_continue(payload),
            stop_reason=stop_reason,
        )

    def coerce_text(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            if "content" in response:
                return str(response["content"])
            return json.dumps(response, ensure_ascii=False)
        return str(response)

    def load_json_object_text(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                stripped = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise LLMAdapterError("LLM response is not valid JSON.", error_code="QE1003") from exc
        if not isinstance(payload, dict):
            raise LLMAdapterError("LLM response must be a JSON object.", error_code="QE1003")
        return payload

    def _normalize_payload(self, response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return dict(response)
        text = self.coerce_text(response).strip()
        payload = self._extract_json_payload(text)
        if payload is not None:
            return payload
        payload = self._extract_xml_tool_payload(text)
        if payload is not None:
            return payload
        return {"content": text, "tool_calls": []}

    def _extract_json_payload(self, text: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        fenced_match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
        if not fenced_match:
            return None
        try:
            parsed = json.loads(fenced_match.group(1))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _extract_xml_tool_payload(self, text: str) -> dict[str, Any] | None:
        tool_calls: list[dict[str, Any]] = []
        content = text
        pattern = re.compile(r"<function=([A-Za-z0-9_.-]+)>\s*(.*?)\s*</function>", re.DOTALL)
        matches = list(pattern.finditer(text))

        for index, match in enumerate(matches):
            tool_name = match.group(1).strip()
            raw_body = match.group(2)
            arguments: dict[str, Any] = {}
            for param_match in re.finditer(
                r"<parameter=([A-Za-z0-9_.-]+)>\s*(.*?)\s*</parameter>",
                raw_body,
                re.DOTALL,
            ):
                arguments[param_match.group(1).strip()] = param_match.group(2).strip()
            if not tool_name:
                continue
            tool_calls.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_use_id": f"text_tool_{index}",
            })

        if matches:
            content = pattern.sub("", content)

        alt_pattern = re.compile(
            r"<tool_call\s+name\s*=\s*\"([A-Za-z0-9_.-]+)\"\s*>\s*(.*?)\s*</tool_call>",
            re.DOTALL,
        )
        alt_matches = list(alt_pattern.finditer(content))
        for index, match in enumerate(alt_matches):
            tool_name = match.group(1).strip()
            raw_body = match.group(2)
            arguments = self._parse_tool_call_body(raw_body)
            if not tool_name:
                continue
            tool_calls.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_use_id": f"text_tool_alt_{index}",
            })

        if alt_matches:
            content = alt_pattern.sub("", content)
            content = re.sub(r"</?tool_calls>", "", content)

        if not tool_calls:
            return None

        content = re.sub(r"</?tool_call>", "", content)
        content = content.strip()
        return {"content": content, "tool_calls": tool_calls}

    def _parse_tool_call_body(self, raw_body: str) -> dict[str, Any]:
        body = raw_body.strip()
        args_match = re.search(
            r"<arguments\s*(?:[^>]*)>\s*(.*?)\s*</arguments>",
            body,
            re.DOTALL,
        )
        if args_match:
            body = args_match.group(1).strip()
        param_pattern = re.compile(
            r"<parameter\s+name\s*=\s*\"([A-Za-z0-9_.-]+)\"\s*>\s*(.*?)\s*</parameter>",
            re.DOTALL,
        )
        param_matches = list(param_pattern.finditer(body))
        if param_matches:
            arguments: dict[str, Any] = {}
            for pm in param_matches:
                arguments[pm.group(1).strip()] = pm.group(2)
            return arguments
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return {}

    def _extract_stop_reason(self, payload: dict[str, Any]) -> str | None:
        if "stop_reason" in payload:
            return str(payload["stop_reason"])
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and "stop_reason" in metadata:
            return str(metadata["stop_reason"])
        return None

    def _needs_continue(self, payload: dict[str, Any]) -> bool:
        if bool(payload.get("continue")):
            return True
        stop_reason = self._extract_stop_reason(payload)
        return str(stop_reason or "").lower() == "max_output_tokens"
