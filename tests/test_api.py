from __future__ import annotations

import json

import respx
import pytest
from fastapi.testclient import TestClient
from httpx import Response

from mdrouter.main import create_app


@pytest.fixture(autouse=True)
def _isolate_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTER_CACHE_ENABLED", "false")
    monkeypatch.setenv("ROUTER_SEM_CACHE_ENABLED", "false")
    monkeypatch.setenv("ROUTER_CACHE_BACKEND", "memory")


def _write_config(tmp_path) -> str:
    config = {
        "server": {
            "host": "127.0.0.1",
            "port": 11434,
            "log_level": "info",
            "request_timeout": 30,
            "bind_localhost_only": True,
        },
        "providers": {
            "novita": {
                "type": "openai_compat",
                "base_url": "http://upstream.test/v1",
                "headers": {},
                "wire_format": "openai_chat",
                "timeout": 30,
            }
        },
        "models": {
            "novita/demo-model": {
                "provider": "novita",
                "upstream_model": "demo-upstream",
                "capabilities": ["chat", "stream"],
                "context_length": 131072,
                "extra": {},
            }
        },
        "routing": {"strict_provider_prefix": True, "unknown_model_behavior": "error"},
    }
    config_path = tmp_path / "providers.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return str(config_path)


def test_tags_endpoint(tmp_path):
    app = create_app(_write_config(tmp_path))
    client = TestClient(app)
    response = client.get("/api/tags")
    assert response.status_code == 200
    payload = response.json()
    assert payload["models"][0]["name"] == "novita/demo-model"
    assert payload["models"][0]["model_info"]["router.context_length"] == 131072


def test_show_endpoint_uses_model_context_length(tmp_path):
    app = create_app(_write_config(tmp_path))
    client = TestClient(app)
    response = client.post("/api/show", json={"model": "novita/demo-model"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_info"]["router.context_length"] == 131072


@respx.mock
def test_chat_non_stream(tmp_path):
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {"message": {"role": "assistant", "content": "Hello from upstream"}}
                ],
            },
        )
    )
    app = create_app(_write_config(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={
            "model": "novita/demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["message"]["content"] == "Hello from upstream"


@respx.mock
def test_chat_stream(tmp_path):
    sse = (
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        'data: {"choices":[{"delta":{"content":" world"}}]}\n'
        "data: [DONE]\n"
    )
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=Response(200, text=sse)
    )

    app = create_app(_write_config(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={
            "model": "novita/demo-model",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert response.status_code == 200

    lines = [json.loads(line) for line in response.text.splitlines()]
    assert lines[0]["message"]["content"] == "Hello"
    assert lines[1]["message"]["content"] == " world"
    assert lines[-1]["done"] is True


@respx.mock
def test_generate_non_stream(tmp_path):
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {"message": {"role": "assistant", "content": "Generated answer"}}
                ],
            },
        )
    )

    app = create_app(_write_config(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/api/generate",
        json={"model": "novita/demo-model", "prompt": "hello", "stream": False},
    )
    assert response.status_code == 200
    assert response.json()["response"] == "Generated answer"


@respx.mock
def test_v1_chat_completions_preserves_tools_and_tool_calls(tmp_path):
    captured_request: dict[str, object] = {}

    def upstream_handler(request):
        captured_request.update(
            request.content and json.loads(request.content.decode("utf-8"))
        )
        return Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    respx.post("http://upstream.test/v1/chat/completions").mock(
        side_effect=upstream_handler
    )

    app = create_app(_write_config(tmp_path))
    client = TestClient(app)

    payload = {
        "model": "novita/demo-model",
        "messages": [{"role": "user", "content": "list files"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert captured_request.get("tools") == payload["tools"]
    assert captured_request.get("tool_choice") == "auto"
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert (
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "read_file"
    )


@respx.mock
def test_v1_chat_completions_stream_preserves_tool_call_chunks(tmp_path):
    sse = (
        'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}]},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=Response(200, text=sse)
    )

    app = create_app(_write_config(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "novita/demo-model",
            "messages": [{"role": "user", "content": "list files"}],
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
            ],
            "stream": True,
        },
    )

    assert response.status_code == 200
    events: list[dict[str, object]] = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            continue
        events.append(json.loads(data))

    assert any(
        (event["choices"][0].get("delta") or {}).get("tool_calls") for event in events
    )
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
