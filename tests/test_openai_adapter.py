from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from mdrouter.adapters.openai_compat import OpenAICompatibleAdapter
from mdrouter.models import UpstreamProviderRequest


@pytest.mark.asyncio
@respx.mock
async def test_go_adapter_injects_reasoning_content_for_tool_calls() -> None:
    captured: dict[str, object] = {}

    def handler(request):
        captured.update(json.loads(request.content.decode("utf-8")))
        return Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    respx.post("http://upstream.test/zen/go/v1/chat/completions").mock(
        side_effect=handler
    )

    adapter = OpenAICompatibleAdapter(
        base_url="http://upstream.test/zen/go/v1",
        headers={},
        timeout=5,
    )
    request = UpstreamProviderRequest(
        model="kimi-k2.6",
        messages=[
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
        ],
        stream=False,
        options=None,
    )

    await adapter.chat_once(request)

    outgoing_messages = captured.get("messages") or []
    assert outgoing_messages[0]["reasoning_content"] == "I will inspect files."


@pytest.mark.asyncio
@respx.mock
async def test_non_go_adapter_does_not_inject_reasoning_content() -> None:
    captured: dict[str, object] = {}

    def handler(request):
        captured.update(json.loads(request.content.decode("utf-8")))
        return Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    respx.post("http://upstream.test/v1/chat/completions").mock(side_effect=handler)

    adapter = OpenAICompatibleAdapter(
        base_url="http://upstream.test/v1",
        headers={},
        timeout=5,
    )
    request = UpstreamProviderRequest(
        model="demo",
        messages=[
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
        ],
        stream=False,
        options=None,
    )

    await adapter.chat_once(request)

    outgoing_messages = captured.get("messages") or []
    assert "reasoning_content" not in outgoing_messages[0]
