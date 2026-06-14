"""Tests for AnthropicCompatibleAdapter and conversion functions."""

from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from mdrouter.adapters.anthropic_compat import (
    AnthropicCompatibleAdapter,
    _openai_messages_to_anthropic,
    _openai_tools_to_anthropic,
    _normalize_anthropic_non_stream,
    DEFAULT_MAX_TOKENS,
)
from mdrouter.models import UpstreamProviderRequest


# ---------- _openai_messages_to_anthropic ----------


def test_simple_text_conversation():
    messages = [
        {"role": "user", "content": "hello"},
    ]
    ant_msgs, system, max_tokens, top = _openai_messages_to_anthropic(messages, None)
    assert len(ant_msgs) == 1
    assert ant_msgs[0]["role"] == "user"
    assert ant_msgs[0]["content"][0]["type"] == "text"
    assert ant_msgs[0]["content"][0]["text"] == "hello"
    assert system is None
    assert max_tokens == DEFAULT_MAX_TOKENS


def test_system_prompt_extraction():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ]
    ant_msgs, system, _, _ = _openai_messages_to_anthropic(messages, None)
    assert len(ant_msgs) == 1
    assert ant_msgs[0]["role"] == "user"
    assert system == "You are a helpful assistant."


def test_multiple_system_messages_concatenated():
    messages = [
        {"role": "system", "content": "First rule."},
        {"role": "system", "content": "Second rule."},
        {"role": "user", "content": "hello"},
    ]
    _, system, _, _ = _openai_messages_to_anthropic(messages, None)
    assert system == "First rule.\n\nSecond rule."


def test_max_tokens_from_options():
    messages = [{"role": "user", "content": "hello"}]
    _, _, max_tokens, _ = _openai_messages_to_anthropic(messages, {"max_tokens": 1024})
    assert max_tokens == 1024


def test_max_completion_tokens_fallback():
    messages = [{"role": "user", "content": "hello"}]
    _, _, max_tokens, _ = _openai_messages_to_anthropic(
        messages, {"max_completion_tokens": 512}
    )
    assert max_tokens == 512


def test_temperature_and_top_p_extraction():
    messages = [{"role": "user", "content": "hello"}]
    _, _, _, top = _openai_messages_to_anthropic(
        messages, {"temperature": 0.7, "top_p": 0.9}
    )
    assert top["temperature"] == 0.7
    assert top["top_p"] == 0.9


def test_image_content():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        }
    ]
    ant_msgs, _, _, _ = _openai_messages_to_anthropic(messages, None)
    assert len(ant_msgs) == 1
    content = ant_msgs[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["data"] == "AAAA"


def test_tool_call_to_anthropic():
    messages = [
        {
            "role": "assistant",
            "content": "Let me read a file.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "README.md"}',
                    },
                }
            ],
        }
    ]
    ant_msgs, _, _, _ = _openai_messages_to_anthropic(messages, None)
    assert len(ant_msgs) == 1
    content = ant_msgs[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Let me read a file."
    assert content[1]["type"] == "tool_use"
    assert content[1]["name"] == "read_file"
    assert content[1]["input"] == {"path": "README.md"}


def test_tool_result_conversion():
    messages = [
        {"role": "tool", "content": "file contents here", "tool_call_id": "call_1"},
    ]
    ant_msgs, _, _, _ = _openai_messages_to_anthropic(messages, None)
    assert len(ant_msgs) == 1
    assert ant_msgs[0]["role"] == "user"
    content = ant_msgs[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "tool_result"
    assert content[0]["tool_use_id"] == "call_1"


# ---------- _openai_tools_to_anthropic ----------


def test_tools_conversion():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    result = _openai_tools_to_anthropic(tools)
    assert len(result) == 1
    assert result[0]["name"] == "get_weather"
    assert result[0]["description"] == "Get weather for a location"
    assert result[0]["input_schema"]["type"] == "object"


# ---------- _normalize_anthropic_non_stream ----------


def test_normalize_simple_text_response():
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "content": [{"type": "text", "text": "Hello, world!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    result = _normalize_anthropic_non_stream(model="claude-opus-4-7", body=body)
    assert result["model"] == "claude-opus-4-7"
    assert result["done"] is True
    assert result["message"]["content"] == "Hello, world!"
    assert result["message"]["role"] == "assistant"
    assert result["done_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 5


def test_normalize_tool_use_response():
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "content": [
            {"type": "text", "text": "Let me check the weather."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"location": "San Francisco"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 20, "output_tokens": 30},
    }
    result = _normalize_anthropic_non_stream(model="claude-opus-4-7", body=body)
    assert result["done_reason"] == "tool_calls"
    msg = result["message"]
    assert msg["content"] == "Let me check the weather."
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["id"] == "toolu_1"
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {
        "location": "San Francisco"
    }


def test_normalize_max_tokens_stop_reason():
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "content": [{"type": "text", "text": "truncated..."}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    result = _normalize_anthropic_non_stream(model="claude-opus-4-7", body=body)
    assert result["done_reason"] == "length"


def test_normalize_empty_content():
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "content": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }
    result = _normalize_anthropic_non_stream(model="claude-opus-4-7", body=body)
    assert result["message"]["content"] == ""
    assert result["message"]["role"] == "assistant"


# ---------- Adapter chat_once ----------


@pytest.mark.asyncio
@respx.mock
async def test_adapter_chat_once():
    """Test that the adapter POSTs to the right URL with correct format."""

    captured: dict[str, object] = {}

    def handler(request):
        captured.update(json.loads(request.content.decode("utf-8")))
        return Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "Hello!"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
        )

    respx.post("http://upstream.test/v1/messages").mock(side_effect=handler)

    adapter = AnthropicCompatibleAdapter(
        base_url="http://upstream.test/v1",
        headers={},
        timeout=5,
    )
    request = UpstreamProviderRequest(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        options=None,
    )

    result = await adapter.chat_once(request)

    # Check request format
    assert captured.get("model") == "claude-opus-4-7"
    assert captured.get("max_tokens") == DEFAULT_MAX_TOKENS

    # Check response normalization
    assert result["choices"][0]["message"]["content"] == "Hello!"
    assert result["object"] == "chat.completion"


@pytest.mark.asyncio
@respx.mock
async def test_adapter_chat_once_with_tools():
    """Test that tools and tool calls are converted correctly."""

    captured: dict[str, object] = {}

    def handler(request):
        captured.update(json.loads(request.content.decode("utf-8")))
        return Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 15},
            },
        )

    respx.post("http://upstream.test/v1/messages").mock(side_effect=handler)

    adapter = AnthropicCompatibleAdapter(
        base_url="http://upstream.test/v1",
        headers={},
        timeout=5,
    )
    request = UpstreamProviderRequest(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "read README.md"}],
        stream=False,
        options={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                }
            ]
        },
    )

    result = await adapter.chat_once(request)

    # Check tools were converted in request
    req_tools = captured.get("tools") or []
    assert len(req_tools) == 1
    assert req_tools[0]["name"] == "read_file"

    # Check tool_use in response
    assert (
        result["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "read_file"
    )


# ---------- Adapter chat_stream ----------


@pytest.mark.asyncio
@respx.mock
async def test_adapter_chat_stream():
    """Test streaming SSE parsing."""

    sse_events = (
        "event: message_start\n"
        'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"usage":{"input_tokens":5,"output_tokens":1}}}\n'
        "\n"
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n'
        "\n"
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n'
        "\n"
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n'
        "\n"
        "event: content_block_stop\n"
        'data: {"type":"content_block_stop","index":0}\n'
        "\n"
        "event: message_delta\n"
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}\n'
        "\n"
        "event: message_stop\n"
        'data: {"type":"message_stop"}\n'
        "\n"
    )

    respx.post("http://upstream.test/v1/messages").mock(
        return_value=Response(200, text=sse_events)
    )

    adapter = AnthropicCompatibleAdapter(
        base_url="http://upstream.test/v1",
        headers={},
        timeout=5,
    )
    request = UpstreamProviderRequest(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        options=None,
    )

    chunks = []
    async for chunk in adapter.chat_stream(request):
        chunks.append(chunk)

    # We should get text content and a done chunk
    texts = [
        c.get("message", {}).get("content", "")
        for c in chunks
        if c.get("message", {}).get("content")
    ]
    assert "Hello world" in "".join(texts)
    assert any(c.get("done") for c in chunks)


@pytest.mark.asyncio
@respx.mock
async def test_adapter_chat_stream_with_tool_use():
    """Test streaming SSE parsing with tool_use."""

    sse_events = (
        "event: message_start\n"
        'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"usage":{"input_tokens":10,"output_tokens":1}}}\n'
        "\n"
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n'
        "\n"
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me check."}}\n'
        "\n"
        "event: content_block_stop\n"
        'data: {"type":"content_block_stop","index":0}\n'
        "\n"
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"read_file","input":{}}}\n'
        "\n"
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\": \\"README.md\\"}"}}\n'
        "\n"
        "event: content_block_stop\n"
        'data: {"type":"content_block_stop","index":1}\n'
        "\n"
        "event: message_delta\n"
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}\n'
        "\n"
        "event: message_stop\n"
        'data: {"type":"message_stop"}\n'
        "\n"
    )

    respx.post("http://upstream.test/v1/messages").mock(
        return_value=Response(200, text=sse_events)
    )

    adapter = AnthropicCompatibleAdapter(
        base_url="http://upstream.test/v1",
        headers={},
        timeout=5,
    )
    request = UpstreamProviderRequest(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "read README.md"}],
        stream=True,
        options=None,
    )

    chunks = []
    async for chunk in adapter.chat_stream(request):
        chunks.append(chunk)

    # Should have a tool_calls delta in at least one chunk
    tool_call_chunks = [c for c in chunks if (c.get("delta") or {}).get("tool_calls")]
    assert len(tool_call_chunks) >= 1
    tc = tool_call_chunks[0]["delta"]["tool_calls"][0]
    assert tc["function"]["name"] == "read_file"


@pytest.mark.asyncio
@respx.mock
async def test_adapter_handles_ping_events():
    """Test that ping events are silently ignored."""

    sse_events = (
        "event: message_start\n"
        'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"usage":{"input_tokens":5,"output_tokens":1}}}\n'
        "\n"
        "event: ping\n"
        'data: {"type":"ping"}\n'
        "\n"
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n'
        "\n"
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n'
        "\n"
        "event: ping\n"
        'data: {"type":"ping"}\n'
        "\n"
        "event: content_block_stop\n"
        'data: {"type":"content_block_stop","index":0}\n'
        "\n"
        "event: message_delta\n"
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}\n'
        "\n"
        "event: message_stop\n"
        'data: {"type":"message_stop"}\n'
        "\n"
    )

    respx.post("http://upstream.test/v1/messages").mock(
        return_value=Response(200, text=sse_events)
    )

    adapter = AnthropicCompatibleAdapter(
        base_url="http://upstream.test/v1",
        headers={},
        timeout=5,
    )
    request = UpstreamProviderRequest(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        options=None,
    )

    chunks = []
    async for chunk in adapter.chat_stream(request):
        chunks.append(chunk)

    # Should complete normally despite pings
    assert any(c.get("done") for c in chunks)


@pytest.mark.asyncio
@respx.mock
async def test_adapter_chat_once_sends_anthropic_version_header():
    """Verify the anthropic-version header is sent."""

    captured_headers: dict[str, str] = {}

    def handler(request):
        captured_headers.update(dict(request.headers))
        return Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    respx.post("http://upstream.test/v1/messages").mock(side_effect=handler)

    adapter = AnthropicCompatibleAdapter(
        base_url="http://upstream.test/v1",
        headers={},
        timeout=5,
    )
    request = UpstreamProviderRequest(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        options=None,
    )

    await adapter.chat_once(request)
    assert captured_headers.get("anthropic-version") == "2023-06-01"
