from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from mdrouter.models import UpstreamProviderRequest


class ProviderAdapter(ABC):
    @abstractmethod
    async def chat_once(self, request: UpstreamProviderRequest) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def chat_stream(
        self, request: UpstreamProviderRequest
    ) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError
