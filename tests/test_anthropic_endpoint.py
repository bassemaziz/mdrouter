"""Tests for the Anthropic-compatible /v1/messages endpoint."""

from __future__ import annotations

import json

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response

from mdrouter.main import create_app


@pytest.fixture(autouse=True)
def _isolate_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTER_CACHE_ENABLED", "false")
    monkeypatch.setenv("ROUTER_SEM_CACHE_ENABLED", "false")
    monkeypatch.setenv("ROUTER_CACHE_BACKEND", "memory")
    monkeypatch.setenv("ROUTER_ENABLED_PROVIDERS", "anthropic")


def _write_anthropic_config(tmp_path) -> str:
    """Write a config with an anthropic_compat provider."""
    config = {
        "server": {
            "host": "127.0.0.1",
            "port": 11434,
            "log_level": "info",
            "request_timeout": 30,
            "bind_localhost_only": True,
        },
        "providers": {
            "anthropic": {
                "type": "anthropic_compat",
                "base_url": "http://upstream.test/v1",
                "headers": {},
                "wire_format": "anthropic_messages",
                "timeout": 30,
            }
        },
        "models": {
            "anthropic/claude-sonnet-4-6": {
                "provider": "anthropic",
                "upstream_model": "claude-sonnet-4-6-20250715",
                "capabilities": ["chat", "stream", "tools", "thinking", "vision"],
                "context_length": 200000,
                "extra": {"max_output": 128000},
            }
        },
        "routing": {"strict_provider_prefix": True, "unknown_model_behavior": "error"},
    }
    config_path = tmp_path / "providers.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return str(config_path)


@respx.mock
def test_v1_messages_non_stream(tmp_path):
    """Test POST /v1/messages with non-streaming simple text response."""
    respx.post("http://upstream.test/v1/messages").mock(
        return_value=Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6-20250715",
                "content": [{"type": "text", "text": "Hello, Claude Code!"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
        )
    )

    app = create_app(_write_anthropic_config(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 256,
            "stream": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "anthropic/claude-sonnet-4-6"
    assert len(body["content"]) >= 1
    assert body["content"][0]["type"] == "text"
    assert "Hello" in body["content"][0]["text"]
    assert body["stop_reason"] == "end_turn"
    assert "usage" in body
    assert body["usage"]["input_tokens"] == 5
    assert body["usage"]["output_tokens"] == 3


@respx.mock
def test_v1_messages_non_stream_with_tools(tmp_path):
    """Test /v1/messages with tool use in response."""
    respx.post("http://upstream.test/v1/messages").mock(
        return_value=Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6-20250715",
                "content": [
                    {"type": "text", "text": "Let me read that file."},
                    {
                        "type": "tool_use",
                        "id": "toolu_abc123",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 15, "output_tokens": 20},
            },
        )
    )

    app = create_app(_write_anthropic_config(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "read the readme"}],
            "max_tokens": 256,
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ],
            "stream": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    content = body["content"]
    # Find the tool_use block
    tool_blocks = [b for b in content if b["type"] == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["name"] == "read_file"
    assert tool_blocks[0]["input"] == {"path": "README.md"}


@respx.mock
def test_v1_messages_stream(tmp_path):
    """Test POST /v1/messages with streaming."""
    sse = (
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
        return_value=Response(200, text=sse)
    )

    app = create_app(_write_anthropic_config(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 256,
            "stream": True,
        },
    )

    assert response.status_code == 200

    # Parse SSE events from response
    text = response.text
    events = []
    current_event = None
    current_data = ""
    for line in text.splitlines():
        if line.startswith("event: "):
            if current_event is not None and current_data:
                events.append((current_event, current_data))
            current_event = line[len("event: ") :]
            current_data = ""
        elif line.startswith("data: "):
            current_data = line[len("data: ") :]
        elif line == "" and current_event is not None:
            if current_data:
                events.append((current_event, current_data))
            current_event = None
            current_data = ""

    if current_event is not None and current_data:
        events.append((current_event, current_data))

    event_types = [e[0] for e in events]
    assert "message_start" in event_types
    assert "content_block_start" in event_types
    assert "content_block_delta" in event_types
    assert "content_block_stop" in event_types
    assert "message_delta" in event_types
    assert "message_stop" in event_types


@respx.mock
def test_v1_messages_stream_with_tool_use(tmp_path):
    """Test /v1/messages streaming with tool_use."""
    sse = (
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
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":25}}\n'
        "\n"
        "event: message_stop\n"
        'data: {"type":"message_stop"}\n'
        "\n"
    )

    respx.post("http://upstream.test/v1/messages").mock(
        return_value=Response(200, text=sse)
    )

    app = create_app(_write_anthropic_config(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "read file"}],
            "max_tokens": 256,
            "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    # Verify the response contains the expected SSE event types
    text = response.text
    assert "content_block_delta" in text
    assert "message_stop" in text


@respx.mock
def test_v1_messages_upstream_error(tmp_path):
    """Test error handling when upstream returns an error."""
    respx.post("http://upstream.test/v1/messages").mock(
        return_value=Response(401, json={"error": {"message": "Invalid API key"}})
    )

    app = create_app(_write_anthropic_config(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 256,
            "stream": False,
        },
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "Invalid API key" in detail or "Upstream HTTP 401" in detail


@respx.mock
def test_v1_messages_system_prompt(tmp_path):
    """Test that system prompt is passed through correctly."""
    captured: dict[str, object] = {}

    def handler(request):
        captured.update(json.loads(request.content.decode("utf-8")))
        return Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6-20250715",
                "content": [{"type": "text", "text": "I am helpful."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 3},
            },
        )

    respx.post("http://upstream.test/v1/messages").mock(side_effect=handler)

    app = create_app(_write_anthropic_config(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "system": "You are a brilliant coder.",
            "max_tokens": 256,
            "stream": False,
        },
    )

    assert response.status_code == 200
    # System prompt should be forwarded to upstream
    assert captured.get("system") == "You are a brilliant coder."
