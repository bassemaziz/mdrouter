from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

import httpx

from mdrouter.adapters.base import ProviderAdapter
from mdrouter.models import UpstreamProviderRequest

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
QUIRK_ANTHROPIC_EXPLICIT_CACHE = "anthropic_explicit_cache"

# ---------- OpenAI-shaped messages → Anthropic request body ----------


def _openai_content_part_to_anthropic(part: dict[str, Any]) -> dict[str, Any]:
    """Convert a single content part from OpenAI format to Anthropic format."""
    part_type = part.get("type", "")
    if part_type == "image_url":
        image_url = part.get("image_url", {})
        url = image_url.get("url", "")
        # data:image/png;base64,<data> or raw base64
        media_type = "image/jpeg"
        data = url
        if url.startswith("data:"):
            header, data = url.split(",", 1)
            if ";" in header:
                media_type = header.split(":")[1].split(";")[0]
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }
    if part_type == "text":
        return {"type": "text", "text": part.get("text", "")}
    # Unknown type — try to infer
    if "image_url" in part:
        return _openai_content_part_to_anthropic(
            {"type": "image_url", "image_url": part["image_url"]}
        )
    if "text" in part:
        return {"type": "text", "text": str(part["text"])}
    # Fallback
    return {"type": "text", "text": json.dumps(part)}


def _openai_tool_call_to_anthropic(tc: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI tool_call to an Anthropic tool_use content block."""
    func = tc.get("function", {})
    try:
        tool_input = json.loads(func.get("arguments", "{}"))
    except (json.JSONDecodeError, TypeError):
        tool_input = {}
    return {
        "type": "tool_use",
        "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
        "name": func.get("name", ""),
        "input": tool_input,
    }


def _openai_tool_result_to_anthropic(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI tool message to an Anthropic tool_result content block."""
    content = msg.get("content", "")
    if isinstance(content, str):
        content_text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        content_text = "\n".join(parts)
    else:
        content_text = str(content)
    return {
        "type": "tool_result",
        "tool_use_id": msg.get("tool_call_id", ""),
        "content": [{"type": "text", "text": content_text}],
    }


def _openai_messages_to_anthropic(
    messages: list[dict[str, Any]], options: dict[str, Any] | None
) -> tuple[
    list[dict[str, Any]], str | list[dict[str, Any]] | None, int, dict[str, Any]
]:
    """Convert OpenAI-shaped messages to Anthropic Messages API format.

    Returns (anthropic_messages, system, max_tokens, top_level_params).
    top_level_params contains: temperature, top_p, thinking, stop_sequences, metadata.
    """
    opts = dict(options or {})
    anthropic_messages: list[dict[str, Any]] = []
    system: str | list[dict[str, Any]] | None = None
    system_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(str(part.get("text", "")))
            continue

        if role == "tool":
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [_openai_tool_result_to_anthropic(msg)],
                }
            )
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            content_blocks: list[dict[str, Any]] = []
            # Text content
            text_content = msg.get("content", "")
            if isinstance(text_content, str) and text_content.strip():
                content_blocks.append({"type": "text", "text": text_content})
            # Tool calls
            if isinstance(tool_calls, list) and tool_calls:
                for tc in tool_calls:
                    content_blocks.append(_openai_tool_call_to_anthropic(tc))
            if not content_blocks:
                content_blocks = [{"type": "text", "text": ""}]
            anthropic_messages.append({"role": "assistant", "content": content_blocks})
            continue

        # role == "user" (or anything else)
        content = msg.get("content", "")
        if isinstance(content, str):
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": content}],
                }
            )
        elif isinstance(content, list):
            blocks = [_openai_content_part_to_anthropic(p) for p in content]
            anthropic_messages.append({"role": "user", "content": blocks})
        elif isinstance(content, dict):
            blocks = [_openai_content_part_to_anthropic(content)]
            anthropic_messages.append({"role": "user", "content": blocks})
        else:
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": str(content)}],
                }
            )

    if system_parts:
        system = "\n\n".join(system_parts)

    # Extract max_tokens (required by Anthropic)
    max_tokens = opts.pop("max_tokens", None)
    if isinstance(max_tokens, int) and max_tokens > 0:
        pass
    else:
        max_tokens = opts.pop("max_completion_tokens", None)
        if not (isinstance(max_tokens, int) and max_tokens > 0):
            max_tokens = None

    top_level: dict[str, Any] = {}
    for key in (
        "temperature",
        "top_p",
        "thinking",
        "reasoning_effort",
        "stop_sequences",
        "metadata",
        "stop",
    ):
        if key in opts:
            top_level[key] = opts.pop(key)

    return anthropic_messages, system, max_tokens or DEFAULT_MAX_TOKENS, top_level


# ---------- Anthropic response → OpenAI-shaped normalized response ----------


def _map_stop_reason(stop_reason: str | None) -> str:
    """Map Anthropic stop_reason to OpenAI finish_reason."""
    if not stop_reason:
        return "stop"
    mapping = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
    }
    return mapping.get(stop_reason, stop_reason)


def _normalize_anthropic_non_stream(
    *, model: str, body: dict[str, Any]
) -> dict[str, Any]:
    """Convert an Anthropic non-streaming Messages response to the normalized
    shape expected by the router: {model, created_at, message: {role, content,
    tool_calls?}, done: True, done_reason, usage?}.
    """
    from datetime import UTC, datetime

    content_blocks = body.get("content") or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in content_blocks:
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "tool_use":
            tool_input = block.get("input") or {}
            tool_calls.append(
                {
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:16]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(tool_input, ensure_ascii=True),
                    },
                }
            )
        # thinking blocks are skipped (not surfaced to OpenAI clients)

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else "",
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    stop_reason = _map_stop_reason(body.get("stop_reason"))

    usage = body.get("usage") or {}
    normalized_usage: dict[str, Any] | None = None
    if usage:
        normalized_usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0)
            + usage.get("output_tokens", 0),
        }

    payload: dict[str, Any] = {
        "model": model,
        "created_at": datetime.now(UTC).isoformat(),
        "message": message,
        "done": True,
        "done_reason": stop_reason,
    }
    if normalized_usage:
        payload["usage"] = normalized_usage
    return payload


# ---------- Anthropic streaming SSE → OpenAI-shaped normalized chunks ----------


def _normalize_anthropic_stream_event(
    event_type: str, data: dict[str, Any], state: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Process one Anthropic SSE event and yield zero or more OpenAI-shaped chunks.

    Args:
        event_type: The SSE event name (e.g. "message_start", "content_block_delta")
        data: The JSON-parsed data payload
        state: Mutable accumulator dict (caller persists across events)

    Returns:
        (list of normalized chunks, updated state)
    """
    from datetime import UTC, datetime

    chunks: list[dict[str, Any]] = []

    if state.get("done"):
        return chunks, state

    now = datetime.now(UTC).isoformat()
    model = state.get("model", "")

    if event_type == "message_start":
        msg = data.get("message") or {}
        state["message_id"] = msg.get("id", "")
        state["model"] = msg.get("model", model)
        state["current_text_block_index"] = None
        state["tool_use_blocks"] = state.get("tool_use_blocks", {})
        state["thinking_blocks"] = state.get("thinking_blocks", {})
        state["pending_tool_calls"] = state.get("pending_tool_calls", [])
        return chunks, state

    if event_type == "ping":
        return chunks, state

    if event_type == "content_block_start":
        idx = data.get("index")
        block = data.get("content_block") or {}
        block_type = block.get("type", "")
        if block_type == "text":
            state["current_text_block_index"] = idx
        elif block_type == "tool_use":
            state["tool_use_blocks"][idx] = {
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "input_json": "",
            }
        elif block_type == "thinking":
            state["thinking_blocks"][idx] = {
                "thinking": "",
                "signature": "",
            }
        return chunks, state

    if event_type == "content_block_delta":
        idx = data.get("index")
        delta = data.get("delta") or {}
        delta_type = delta.get("type", "")

        if delta_type == "text_delta":
            text = delta.get("text", "")
            if text:
                chunks.append(
                    {
                        "model": model,
                        "created_at": now,
                        "message": {"role": "assistant", "content": text},
                        "delta": {"content": text},
                        "done": False,
                    }
                )

        elif delta_type == "input_json_delta":
            partial = delta.get("partial_json", "")
            if idx in state["tool_use_blocks"]:
                state["tool_use_blocks"][idx]["input_json"] += partial

        elif delta_type == "thinking_delta":
            thinking_text = delta.get("thinking", "")
            if idx in state["thinking_blocks"]:
                state["thinking_blocks"][idx]["thinking"] += thinking_text
            if thinking_text:
                chunks.append(
                    {
                        "model": model,
                        "created_at": now,
                        "message": {"role": "assistant", "content": ""},
                        "delta": {"reasoning_content": thinking_text},
                        "done": False,
                    }
                )

        elif delta_type == "signature_delta":
            signature = delta.get("signature", "")
            if idx in state["thinking_blocks"]:
                state["thinking_blocks"][idx]["signature"] = signature

        return chunks, state

    if event_type == "content_block_stop":
        idx = data.get("index")
        if idx in state["tool_use_blocks"]:
            block_info = state["tool_use_blocks"][idx]
            try:
                tool_input = (
                    json.loads(block_info["input_json"])
                    if block_info["input_json"].strip()
                    else {}
                )
            except json.JSONDecodeError:
                tool_input = {}
            tool_call = {
                "id": block_info["id"],
                "type": "function",
                "function": {
                    "name": block_info["name"],
                    "arguments": json.dumps(tool_input, ensure_ascii=True),
                },
            }
            state.setdefault("pending_tool_calls", []).append(tool_call)
            chunks.append(
                {
                    "model": model,
                    "created_at": now,
                    "message": {"role": "assistant", "content": ""},
                    "delta": {"tool_calls": [tool_call]},
                    "done": False,
                }
            )
        return chunks, state

    if event_type == "message_delta":
        delta = data.get("delta") or {}
        stop_reason = _map_stop_reason(delta.get("stop_reason"))
        usage = data.get("usage") or {}
        normalized_usage: dict[str, Any] | None = None
        if usage:
            normalized_usage = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0)
                + usage.get("output_tokens", 0),
            }
        done_chunk: dict[str, Any] = {
            "model": model,
            "created_at": now,
            "message": {"role": "assistant", "content": ""},
            "delta": {},
            "done": True,
            "done_reason": stop_reason,
        }
        if normalized_usage:
            done_chunk["usage"] = normalized_usage
        chunks.append(done_chunk)
        state["done"] = True
        return chunks, state

    if event_type == "message_stop":
        if not state.get("done"):
            state["done"] = True
            chunks.append(
                {
                    "model": model,
                    "created_at": now,
                    "message": {"role": "assistant", "content": ""},
                    "delta": {},
                    "done": True,
                    "done_reason": "stop",
                }
            )
        return chunks, state

    return chunks, state


# ---------- OpenAI tools → Anthropic tools ----------


def _openai_tools_to_anthropic(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Convert OpenAI-format tools to Anthropic-format tools."""
    if not tools:
        return None
    result: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function") or {}
            result.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        else:
            # Pass through unknown tool types
            result.append(tool)
    return result


def _openai_tool_choice_to_anthropic(
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert OpenAI tool_choice to Anthropic format."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in {"none", "auto", "required", "any"}:
            # Anthropic uses slightly different names but most are compatible
            if tool_choice == "required":
                return {"type": "any"}
            if tool_choice == "auto":
                return {"type": "auto"}
            if tool_choice == "any":
                return {"type": "any"}
            return None  # "none" means no tool call
        return {"type": "tool", "name": tool_choice}
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            func_name = tool_choice.get("function", {}).get("name", "")
            return {"type": "tool", "name": func_name}
        return {"type": "tool", "name": str(tool_choice.get("name", ""))}
    return None


# ---------- Adapter class ----------


class AnthropicCompatibleAdapter(ProviderAdapter):
    """ProviderAdapter for Anthropic Messages API and compatible providers."""

    def __init__(
        self,
        *,
        base_url: str,
        headers: dict[str, str],
        timeout: float,
        quirks: set[str] | None = None,
        client: httpx.AsyncClient | None = None,
        model_extra: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers)
        self.headers.setdefault("anthropic-version", ANTHROPIC_VERSION)
        self.timeout = timeout
        self.quirks = set(quirks) if quirks is not None else set()
        self.model_extra = model_extra or {}
        self._client = client

    def _build_payload(
        self, request: UpstreamProviderRequest, *, stream: bool
    ) -> dict[str, Any]:
        options = dict(request.options or {})

        # Extract OpenAI-format tools and convert to Anthropic format
        tools = _openai_tools_to_anthropic(options.pop("tools", None))
        tool_choice = _openai_tool_choice_to_anthropic(options.pop("tool_choice", None))

        messages, system, max_tokens, top_level = _openai_messages_to_anthropic(
            request.messages, options
        )

        # max_tokens from options; fall back to model_extra or default
        if max_tokens is None or max_tokens <= 0:
            max_tokens = self.model_extra.get("max_output", DEFAULT_MAX_TOKENS)

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        # Enable Anthropic's automatic prompt caching when the quirk is set.
        if QUIRK_ANTHROPIC_EXPLICIT_CACHE in self.quirks:
            payload["cache_control"] = {"type": "ephemeral"}

        if system:
            payload["system"] = system

        if tools:
            payload["tools"] = tools
        if tool_choice and tool_choice.get("type") != "none":
            payload["tool_choice"] = tool_choice

        # Thinking / extended thinking
        thinking = top_level.get("thinking") or request.thinking
        reasoning_effort = top_level.get("reasoning_effort") or request.reasoning_effort
        if thinking is not None:
            if isinstance(thinking, dict):
                payload["thinking"] = thinking
            elif isinstance(thinking, str):
                payload["thinking"] = {"type": thinking}
        elif reasoning_effort is not None:
            # Map reasoning_effort to Anthropic thinking.budget_tokens
            budget_map = {
                "low": 1024,
                "medium": 4096,
                "high": 16000,
            }
            budget = budget_map.get(str(reasoning_effort).lower(), 4096)
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}

        for key in ("temperature", "top_p", "top_k"):
            val = top_level.get(key)
            if val is not None:
                payload[key] = val

        stop = top_level.get("stop") or top_level.get("stop_sequences")
        if stop:
            payload["stop_sequences"] = stop if isinstance(stop, list) else [stop]

        if isinstance(top_level.get("metadata"), dict):
            payload["metadata"] = top_level["metadata"]

        return payload

    async def chat_once(self, request: UpstreamProviderRequest) -> dict[str, Any]:
        payload = self._build_payload(request, stream=False)
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        should_close = self._client is None
        try:
            response = await client.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            # Convert to OpenAI-like wire format so router's normalize_chat_non_stream works
            normalized = _normalize_anthropic_non_stream(model=request.model, body=body)
            return _anthropic_normalized_to_openai_wire(normalized)
        finally:
            if should_close:
                await client.aclose()

    async def chat_stream(
        self, request: UpstreamProviderRequest
    ) -> AsyncIterator[dict[str, Any]]:
        payload = self._build_payload(request, stream=True)
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        should_close = self._client is None
        state: dict[str, Any] = {"model": request.model}

        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload,
            ) as response:
                if response.is_error:
                    await response.aread()
                    response.raise_for_status()

                current_event: str | None = None
                data_lines: list[str] = []

                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        if current_event and data_lines:
                            data_str = "".join(data_lines).strip()
                            if data_str:
                                try:
                                    parsed = json.loads(data_str)
                                    chunks, state = _normalize_anthropic_stream_event(
                                        current_event, parsed, state
                                    )
                                    for chunk in chunks:
                                        yield chunk
                                except json.JSONDecodeError:
                                    pass
                            data_lines = []
                        current_event = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:])
                    elif line.strip() == "" and data_lines:
                        # Empty line (SSE delimiter) — flush current event
                        if current_event:
                            data_str = "".join(data_lines).strip()
                            if data_str:
                                try:
                                    parsed = json.loads(data_str)
                                    chunks, state = _normalize_anthropic_stream_event(
                                        current_event, parsed, state
                                    )
                                    for chunk in chunks:
                                        yield chunk
                                except json.JSONDecodeError:
                                    pass
                        current_event = None
                        data_lines = []
                    else:
                        # Continuation line
                        pass

                # Flush any remaining event
                if current_event and data_lines:
                    data_str = "".join(data_lines).strip()
                    if data_str:
                        try:
                            parsed = json.loads(data_str)
                            chunks, state = _normalize_anthropic_stream_event(
                                current_event, parsed, state
                            )
                            for chunk in chunks:
                                yield chunk
                        except json.JSONDecodeError:
                            pass
        finally:
            if should_close:
                await client.aclose()


def _anthropic_normalized_to_openai_wire(normalized: dict[str, Any]) -> dict[str, Any]:
    """Convert the internal normalized shape back to an OpenAI-like wire response
    so the router's normalize_chat_non_stream() can process it correctly."""
    msg = normalized.get("message") or {}
    choices = [
        {
            "index": 0,
            "message": dict(msg),
            "finish_reason": normalized.get("done_reason", "stop"),
        }
    ]
    wire: dict[str, Any] = {
        "id": "chatcmpl-anthropic",
        "object": "chat.completion",
        "created": 0,
        "model": normalized.get("model", ""),
        "choices": choices,
    }
    usage = normalized.get("usage")
    if isinstance(usage, dict):
        wire["usage"] = usage
    return wire
