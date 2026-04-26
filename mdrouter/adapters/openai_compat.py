from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from mdrouter.adapters.base import ProviderAdapter
from mdrouter.models import UpstreamProviderRequest


class OpenAICompatibleAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        base_url: str,
        headers: dict[str, str],
        timeout: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self.timeout = timeout
        self._client = client

    def _is_go_provider(self) -> bool:
        return "/zen/go/" in self.base_url

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Moonshot AI and other go-provider models require non-empty reasoning_content on assistant tool-call turns.
        if not self._is_go_provider():
            return messages

        patched: list[dict[str, Any]] = []
        for msg in messages:
            clone = dict(msg)
            if (
                clone.get("role") == "assistant"
                and isinstance(clone.get("tool_calls"), list)
            ):
                # Fix missing or empty reasoning_content
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
            patched.append(clone)
        return patched

    async def chat_once(self, request: UpstreamProviderRequest) -> dict[str, Any]:
        payload = {
            "model": request.model,
            "messages": self._prepare_messages(request.messages),
            "stream": False,
        }
        if request.options:
            payload.update(request.options)
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
        payload = {
            "model": request.model,
            "messages": self._prepare_messages(request.messages),
            "stream": True,
        }
        if request.options:
            payload.update(request.options)

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
