from __future__ import annotations

import pytest
from fastapi import HTTPException

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
            "strict_like": {
                "type": "openai_compat",
                "base_url": "http://upstream.test/v1",
                "headers": {},
                "wire_format": "openai_chat",
                "timeout": 30,
                "quirks": ["require_reasoning_content_for_tool_calls"],
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
            "go/vision-model": {
                "provider": "go",
                "upstream_model": "vision-upstream",
                "capabilities": ["chat", "stream", "tools", "vision"],
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
            "strict_like/demo-model": {
                "provider": "strict_like",
                "upstream_model": "strict-upstream",
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


def test_configured_provider_quirk_injects_reasoning_content():
    router = ModelRouter(_config())
    messages = [
        {
            "role": "assistant",
            "content": "I will inspect files.",
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
        model_alias="strict_like/demo-model",
        messages=messages,
        stream=True,
        options=None,
    )

    assert provider_name == "strict_like"
    assert req.messages[0].get("reasoning_content") == messages[0]["content"]


def test_non_vision_model_rejects_image_content():
    router = ModelRouter(_config())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]

    with pytest.raises(HTTPException) as exc_info:
        router._to_provider_request(
            model_alias="go/kimi-k2.6",
            messages=messages,
            stream=True,
            options=None,
        )

    assert exc_info.value.status_code == 400
    assert "does not support vision" in str(exc_info.value)


def test_router_normalizes_dict_image_content_before_provider_request():
    router = ModelRouter(_config())
    messages = [
        {
            "role": "user",
            "content": {"image_url": {"url": "data:image/png;base64,AAAA"}},
        }
    ]

    _, req, _ = router._to_provider_request(
        model_alias="go/vision-model",
        messages=messages,
        stream=False,
        options=None,
    )

    content = req.messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
