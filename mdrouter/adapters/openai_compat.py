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

    async def chat_once(self, request: UpstreamProviderRequest) -> dict[str, Any]:
        payload = {
            "model": request.model,
            "messages": request.messages,
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
            "messages": request.messages,
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
