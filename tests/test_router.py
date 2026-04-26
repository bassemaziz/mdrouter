from __future__ import annotations

from mdrouter.config import AppConfig
from mdrouter.router import ModelRouter


def _config() -> AppConfig:
    payload = {
        "server": {
            "host": "127.0.0.1",
            "port": 11435,
            "log_level": "info",
            "request_timeout": 30,
            "bind_localhost_only": True,
        },
        "providers": {
            "go": {
                "type": "openai_compat",
                "base_url": "http://upstream.test/v1",
                "headers": {},
                "wire_format": "openai_chat",
                "timeout": 30,
            },
            "novita": {
                "type": "openai_compat",
                "base_url": "http://upstream.test/v1",
                "headers": {},
                "wire_format": "openai_chat",
                "timeout": 30,
            },
        },
        "models": {
            "go/kimi-k2.6": {
                "provider": "go",
                "upstream_model": "kimi-k2.6",
                "capabilities": ["chat", "stream", "tools"],
                "context_length": 65536,
                "extra": {},
            },
            "novita/demo-model": {
                "provider": "novita",
                "upstream_model": "demo-upstream",
                "capabilities": ["chat", "stream", "tools"],
                "context_length": 65536,
                "extra": {},
            },
        },
        "routing": {
            "strict_provider_prefix": True,
            "unknown_model_behavior": "error",
        },
    }
    return AppConfig.model_validate(payload)


def test_go_provider_injects_reasoning_content_for_tool_call_history():
    router = ModelRouter(_config())
    messages = [
        {
            "role": "assistant",
            "content": "Let me inspect the project structure.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "list_dir", "arguments": "{}"},
                }
            ],
        }
    ]

    _, req, provider_name = router._to_provider_request(
        model_alias="go/kimi-k2.6",
        messages=messages,
        stream=True,
        options=None,
    )

    assert provider_name == "go"
    assert req.messages[0].get("reasoning_content") == messages[0]["content"]


def test_non_go_provider_does_not_inject_reasoning_content():
    router = ModelRouter(_config())
    messages = [
        {
            "role": "assistant",
            "content": "Let me inspect the project structure.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "list_dir", "arguments": "{}"},
                }
            ],
        }
    ]

    _, req, provider_name = router._to_provider_request(
        model_alias="novita/demo-model",
        messages=messages,
        stream=True,
        options=None,
    )

    assert provider_name == "novita"
    assert "reasoning_content" not in req.messages[0]
