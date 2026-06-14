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


@pytest.mark.asyncio
@respx.mock
async def test_go_adapter_infers_missing_multimodal_content_types() -> None:
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
        model="glm-5.1",
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": "what is in this image?"},
                    {"image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ],
        stream=False,
        options=None,
    )

    await adapter.chat_once(request)

    outgoing_messages = captured.get("messages") or []
    content = outgoing_messages[0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


@pytest.mark.asyncio
@respx.mock
async def test_go_adapter_wraps_single_multimodal_part_dict_content() -> None:
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
        model="glm-5.1",
        messages=[
            {
                "role": "user",
                "content": {"image_url": {"url": "data:image/png;base64,AAAA"}},
            }
        ],
        stream=False,
        options=None,
    )

    await adapter.chat_once(request)

    outgoing_messages = captured.get("messages") or []
    content = outgoing_messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"


@pytest.mark.asyncio
@respx.mock
async def test_non_go_adapter_wraps_single_multimodal_part_dict_content() -> None:
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
                "role": "user",
                "content": {"image_url": {"url": "data:image/png;base64,AAAA"}},
            }
        ],
        stream=False,
        options=None,
    )

    await adapter.chat_once(request)

    outgoing_messages = captured.get("messages") or []
    content = outgoing_messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"


@pytest.mark.asyncio
@respx.mock
async def test_options_cannot_override_core_payload_messages() -> None:
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
                "role": "user",
                "content": {"image_url": {"url": "data:image/png;base64,AAAA"}},
            }
        ],
        stream=False,
        options={
            "messages": [{"role": "user", "content": {"image_url": {"url": "bad"}}}],
            "stream": True,
            "model": "overridden-model",
        },
    )

    await adapter.chat_once(request)

    assert captured["model"] == "demo"
    assert captured["stream"] is False
    outgoing_messages = captured.get("messages") or []
    content = outgoing_messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == "data:image/png;base64,AAAA"


@pytest.mark.asyncio
@respx.mock
async def test_chat_stream_parses_data_prefix_without_space() -> None:
    stream_payload = (
        'data:{"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        'data:{"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        "data:[DONE]\n"
    )
    respx.post("http://upstream.test/v1/chat/completions").mock(
        return_value=Response(
            200,
            text=stream_payload,
            headers={"content-type": "text/event-stream"},
        )
    )

    adapter = OpenAICompatibleAdapter(
        base_url="http://upstream.test/v1",
        headers={},
        timeout=5,
    )
    request = UpstreamProviderRequest(
        model="demo",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        options=None,
    )

    chunks: list[dict[str, object]] = []
    async for chunk in adapter.chat_stream(request):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hi"
    assert chunks[1]["choices"][0]["finish_reason"] == "stop"


def test_prepare_messages_is_idempotent():
    """Normalization must produce identical output when re-applied — this is
    the foundation for upstream prefix-cache stability."""
    adapter = OpenAICompatibleAdapter(
        base_url="http://upstream.test/v1",
        headers={},
        timeout=5,
    )
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What's the weather?"},
    ]
    first = adapter._prepare_messages(messages)
    second = adapter._prepare_messages(messages)
    assert len(first) == len(second)
    for i, (a, b) in enumerate(zip(first, second)):
        assert a == b, f"Message {i} differs: {a!r} != {b!r}"


def test_prepare_messages_multi_turn_prefix_stable():
    """In a multi-turn conversation, message prefixes must be byte-identical
    between turn N-1 (full history) and turn N (history + new message)."""
    adapter = OpenAICompatibleAdapter(
        base_url="http://upstream.test/v1",
        headers={},
        timeout=5,
    )
    turn1 = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    turn2 = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What's the weather?"},
    ]
    t1_result = adapter._prepare_messages(turn1)
    t2_result = adapter._prepare_messages(turn2)
    # The prefix of turn2's result must equal turn1's full result
    assert (
        t2_result[: len(t1_result)] == t1_result
    ), "Prefix mismatch: turn2 prefix != turn1 result"


def test_stable_prefix_quirk_skips_prefix_normalization():
    """With QUIRK_STABLE_PREFIX, only the last message is normalized; earlier
    messages pass through verbatim as shallow copies."""
    adapter = OpenAICompatibleAdapter(
        base_url="http://upstream.test/v1",
        headers={},
        timeout=5,
        quirks={"stable_prefix"},
    )
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What's the weather?"},
    ]
    result = adapter._prepare_messages(messages)
    # First 3 messages should be shallow-copy-identical to input
    for i in range(len(messages) - 1):
        # Must be a different object (shallow copy) but equal content
        assert result[i] is not messages[i], f"Message {i} not copied"
        assert result[i] == messages[i], f"Message {i} mutated"


def test_stable_prefix_with_thinking_is_idempotent():
    """QUIRK_STABLE_PREFIX + thinking mode: assistant messages with reasoning_content
    already set must not be re-injected."""
    adapter = OpenAICompatibleAdapter(
        base_url="http://upstream.test/v1",
        headers={},
        timeout=5,
        quirks={"stable_prefix", "require_reasoning_content_for_thinking"},
    )
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Think about this."},
        {
            "role": "assistant",
            "content": "Let me think...",
            "reasoning_content": "Already injected",
        },
        {"role": "user", "content": "Continue."},
    ]
    result = adapter._prepare_messages(messages)
    # The assistant message already has reasoning_content — must not change
    assert result[2]["reasoning_content"] == "Already injected"
