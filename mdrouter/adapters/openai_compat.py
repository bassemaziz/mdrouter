from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from mdrouter.adapters.base import ProviderAdapter
from mdrouter.models import UpstreamProviderRequest


QUIRK_REQUIRE_REASONING_CONTENT_FOR_TOOL_CALLS = (
    "require_reasoning_content_for_tool_calls"
)
QUIRK_NORMALIZE_MULTIMODAL_CONTENT = "normalize_multimodal_content"


class OpenAICompatibleAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        base_url: str,
        headers: dict[str, str],
        timeout: float,
        quirks: set[str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self.timeout = timeout
        default_quirks: set[str] = set()
        if "/zen/go/" in self.base_url:
            default_quirks = {
                QUIRK_REQUIRE_REASONING_CONTENT_FOR_TOOL_CALLS,
                QUIRK_NORMALIZE_MULTIMODAL_CONTENT,
            }
        self.quirks = set(quirks) if quirks is not None else default_quirks
        self._client = client

    @staticmethod
    def _normalize_multimodal_content_part(part: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(part)
        part_type = normalized.get("type")
        if isinstance(part_type, str) and part_type.strip():
            return normalized

        # Some clients omit `type` and only send the payload key.
        if "image_url" in normalized:
            normalized["type"] = "image_url"
        elif "text" in normalized:
            normalized["type"] = "text"
        return normalized

    def _normalize_message_content(self, message: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(message)
        content = normalized.get("content")

        if isinstance(content, dict):
            normalized["content"] = [self._normalize_multimodal_content_part(content)]
            return normalized

        if isinstance(content, list):
            rebuilt: list[Any] = []
            for part in content:
                if isinstance(part, dict):
                    rebuilt.append(self._normalize_multimodal_content_part(part))
                else:
                    rebuilt.append(part)
            normalized["content"] = rebuilt

        return normalized

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        patched: list[dict[str, Any]] = []
        for msg in messages:
            # Keep payload OpenAI-compatible for all providers:
            # content must be either a string or an array of typed parts.
            clone = self._normalize_message_content(msg)

            if (
                QUIRK_REQUIRE_REASONING_CONTENT_FOR_TOOL_CALLS in self.quirks
                and clone.get("role") == "assistant"
                and isinstance(clone.get("tool_calls"), list)
            ):
                # Some strict validators require non-empty reasoning_content.
                reasoning = clone.get("reasoning_content", "")
                if not reasoning or not reasoning.strip():
                    content = clone.get("content")
                    if isinstance(content, str) and content.strip():
                        clone["reasoning_content"] = content
                    else:
                        clone["reasoning_content"] = "Calling tool."

                # Ensure content field exists (can be None or empty string)
                if "content" not in clone:
                    clone["content"] = None
            elif (
                clone.get("role") == "assistant"
                and isinstance(clone.get("tool_calls"), list)
            ):
                # For tool-call history, keep content key stable across providers.
                if "content" not in clone:
                    clone["content"] = None
            patched.append(clone)
        return patched

    def _build_payload(self, request: UpstreamProviderRequest, *, stream: bool) -> dict[str, Any]:
        prepared_messages = self._prepare_messages(request.messages)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": prepared_messages,
            "stream": stream,
        }
        if request.options:
            # Preserve canonical transport keys even if callers pass them in options.
            payload.update(
                {
                    key: value
                    for key, value in request.options.items()
                    if key not in {"model", "messages", "stream"}
                }
            )
        return payload

    async def chat_once(self, request: UpstreamProviderRequest) -> dict[str, Any]:
        payload = self._build_payload(request, stream=False)
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        should_close = self._client is None
        try:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        finally:
            if should_close:
                await client.aclose()

    async def chat_stream(
        self, request: UpstreamProviderRequest
    ) -> AsyncIterator[dict[str, Any]]:
        payload = self._build_payload(request, stream=True)

        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        should_close = self._client is None
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
            ) as response:
                if response.is_error:
                    await response.aread()
                    response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                    else:
                        data = line
                    if data.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue
        finally:
            if should_close:
                await client.aclose()
