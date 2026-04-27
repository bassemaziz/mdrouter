from __future__ import annotations

import hashlib
import os
import time
from datetime import UTC, datetime
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException

from mdrouter.adapters.base import ProviderAdapter
from mdrouter.adapters.openai_compat import OpenAICompatibleAdapter
from mdrouter.adapters.openai_compat import QUIRK_NORMALIZE_MULTIMODAL_CONTENT
from mdrouter.adapters.openai_compat import QUIRK_REQUIRE_REASONING_CONTENT_FOR_TOOL_CALLS
from mdrouter.config import AppConfig
from mdrouter.config import ModelConfig
from mdrouter.models import (
    OllamaTagModel,
    UpstreamProviderRequest,
)
from mdrouter.runtime import build_response_cache
from mdrouter.runtime import RuntimeSettings

import re as _re

_TOOL_CALL_PATTERNS = _re.compile(
    r"<tool[_\s/]"  # XML: <tool_call>, <tool >, </tool>
    r'|"tool"\s*:'  # JSON key: {"tool": ...}
    r'|"tool_calls"\s*:'  # OpenAI JSON: {"tool_calls": ...}
    r'|```(?:json)?\s*\{\s*"tool"\s*:',  # markdown code block with JSON tool call
    _re.IGNORECASE,
)


def _content_has_tool_call(content_lower: str) -> bool:
    return bool(_TOOL_CALL_PATTERNS.search(content_lower))


def _message_contains_image_input(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, dict):
        return "image_url" in content
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url" or "image_url" in part:
            return True
    return False


def _messages_require_vision(messages: list[dict[str, Any]]) -> bool:
    return any(_message_contains_image_input(message) for message in messages)


def _normalize_multimodal_content_part(part: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(part)
    part_type = normalized.get("type")
    if isinstance(part_type, str) and part_type.strip():
        return normalized
    if "image_url" in normalized:
        normalized["type"] = "image_url"
    elif "text" in normalized:
        normalized["type"] = "text"
    return normalized


def _normalize_message_content(message: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    content = normalized.get("content")
    if isinstance(content, dict):
        normalized["content"] = [_normalize_multimodal_content_part(content)]
        return normalized
    if isinstance(content, list):
        rebuilt: list[Any] = []
        for part in content:
            if isinstance(part, dict):
                rebuilt.append(_normalize_multimodal_content_part(part))
            else:
                rebuilt.append(part)
        normalized["content"] = rebuilt
    return normalized


def _normalize_messages_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_message_content(message) for message in messages]


def _inject_reasoning_content_for_tool_calls(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    patched: list[dict[str, Any]] = []
    for msg in messages:
        clone = dict(msg)
        if (
            clone.get("role") == "assistant"
            and isinstance(clone.get("tool_calls"), list)
            and "reasoning_content" not in clone
        ):
            content = clone.get("content")
            # Use actual content if available and non-empty, otherwise provide placeholder
            if isinstance(content, str) and content.strip():
                clone["reasoning_content"] = content
            else:
                # Moonshot AI requires non-empty reasoning_content when thinking is enabled
                clone["reasoning_content"] = "Calling tool."
        patched.append(clone)
    return patched


class ModelRouter:
    def __init__(
        self, config: AppConfig, runtime: RuntimeSettings | None = None
    ) -> None:
        self.config = config
        self.runtime = runtime or RuntimeSettings.from_env()
        self.response_cache = build_response_cache(self.runtime)
        self.adapters: dict[str, ProviderAdapter] = {}
        self.provider_quirks: dict[str, frozenset[str]] = {}
        self._build_adapters()

    @staticmethod
    def _default_provider_quirks(provider_name: str) -> set[str]:
        if provider_name == "go":
            return {
                QUIRK_REQUIRE_REASONING_CONTENT_FOR_TOOL_CALLS,
                QUIRK_NORMALIZE_MULTIMODAL_CONTENT,
            }
        return set()

    def _build_adapters(self) -> None:
        for provider_name, provider_cfg in self.config.providers.items():
            if provider_cfg.type != "openai_compat":
                raise ValueError(f"Unsupported provider type '{provider_cfg.type}'.")
            timeout = provider_cfg.timeout or self.config.server.request_timeout
            quirks = self._default_provider_quirks(provider_name).union(
                provider_cfg.quirks
            )
            self.provider_quirks[provider_name] = frozenset(quirks)
            self.adapters[provider_name] = OpenAICompatibleAdapter(
                base_url=provider_cfg.base_url,
                headers=provider_cfg.resolve_headers(allow_missing_api_key=True),
                timeout=timeout,
                quirks=quirks,
            )

    def lookup_model_config(self, model_name: str) -> tuple[str, ModelConfig]:
        direct = self.config.models.get(model_name)
        if direct:
            return model_name, direct

        if "/" not in model_name:
            suffix = f"/{model_name}"
            matches = [alias for alias in self.config.models if alias.endswith(suffix)]
            if len(matches) == 1:
                alias = matches[0]
                return alias, self.config.models[alias]
            if len(matches) > 1:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Ambiguous model '{model_name}'. Use provider-prefixed alias, "
                        f"for example '{matches[0]}'."
                    ),
                )

        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'.")

    def list_models(self) -> list[OllamaTagModel]:
        now = datetime.now(UTC)
        result: list[OllamaTagModel] = []
        for alias, model_cfg in self.config.models.items():
            digest = hashlib.sha256(alias.encode("utf-8")).hexdigest()
            caps = []
            if "vision" in model_cfg.capabilities:
                caps.append("vision")
            if "tools" in model_cfg.capabilities:
                caps.append("tools")
            result.append(
                OllamaTagModel(
                    name=alias,
                    model=alias,
                    modified_at=now,
                    size=0,
                    digest=f"sha256:{digest}",
                    capabilities=caps,
                    model_info={
                        "general.basename": model_cfg.upstream_model.split("/")[-1],
                        "general.architecture": "router",
                        "router.context_length": model_cfg.context_length,
                    },
                    supports={
                        "vision": "vision" in model_cfg.capabilities,
                        "tool_calls": "tools" in model_cfg.capabilities,
                    },
                    details={
                        "format": "router",
                        "family": model_cfg.provider,
                        "families": [model_cfg.provider],
                        "parameter_size": "unknown",
                        "quantization_level": "unknown",
                        "capabilities": caps,
                    },
                )
            )
        return result

    def _resolve_model(self, alias: str) -> tuple[ProviderAdapter, str, str]:
        _, model_cfg = self.lookup_model_config(alias)
        provider_cfg = self.config.providers[model_cfg.provider]
        if provider_cfg.api_key_env and not os.getenv(provider_cfg.api_key_env):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Provider '{model_cfg.provider}' requires environment variable "
                    f"'{provider_cfg.api_key_env}'."
                ),
            )
        adapter = self.adapters.get(model_cfg.provider)
        if not adapter:
            raise HTTPException(
                status_code=500,
                detail=f"Provider adapter '{model_cfg.provider}' is missing.",
            )
        return adapter, model_cfg.upstream_model, model_cfg.provider

    @staticmethod
    def _inject_alibaba_explicit_cache(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        patched: list[dict[str, Any]] = []
        injected = False
        for msg in messages:
            clone = dict(msg)
            if (
                not injected
                and clone.get("role") == "system"
                and isinstance(clone.get("content"), str)
                and clone.get("content", "").strip()
            ):
                clone["content"] = [
                    {
                        "type": "text",
                        "text": clone["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                injected = True
            patched.append(clone)
        return patched

    def _to_provider_request(
        self,
        model_alias: str,
        messages: list[dict[str, Any]],
        stream: bool,
        options: dict[str, Any] | None,
    ) -> tuple[ProviderAdapter, UpstreamProviderRequest, str]:
        mutable_messages = _normalize_messages_content(messages)
        _, model_cfg = self.lookup_model_config(model_alias)
        if "vision" not in model_cfg.capabilities and _messages_require_vision(
            mutable_messages
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_alias}' does not support vision content.",
            )

        adapter, upstream_model, provider_name = self._resolve_model(model_alias)
        mutable_options = dict(options or {})

        if (
            self.runtime.prompt_cache_key_enabled
            and "prompt_cache_key" not in mutable_options
        ):
            mutable_options["prompt_cache_key"] = f"router:{model_alias}"
        if (
            self.runtime.prompt_cache_retention
            and "prompt_cache_retention" not in mutable_options
        ):
            mutable_options["prompt_cache_retention"] = (
                self.runtime.prompt_cache_retention
            )
        provider_quirks = self.provider_quirks.get(provider_name, frozenset())
        if QUIRK_REQUIRE_REASONING_CONTENT_FOR_TOOL_CALLS in provider_quirks:
            mutable_messages = _inject_reasoning_content_for_tool_calls(
                mutable_messages
            )
        if provider_name == "alibaba" and self.runtime.alibaba_explicit_cache:
            mutable_messages = self._inject_alibaba_explicit_cache(mutable_messages)

        return (
            adapter,
            UpstreamProviderRequest(
                model=upstream_model,
                messages=mutable_messages,
                stream=stream,
                options=mutable_options or None,
            ),
            provider_name,
        )

    async def chat_once(
        self,
        *,
        model_alias: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        adapter, provider_req, provider_name = self._to_provider_request(
            model_alias=model_alias, messages=messages, stream=False, options=options
        )
        exact_key = self.response_cache.make_exact_key(
            model_alias=model_alias,
            provider=provider_name,
            messages=messages,
            options=options,
        )
        if self.runtime.cache_enabled:
            cached, cache_meta = await self.response_cache.lookup(
                exact_key=exact_key,
                model_alias=model_alias,
                provider=provider_name,
                messages=messages,
            )
            if cached is not None:
                return cached, {
                    "provider": provider_name,
                    "upstream_model": provider_req.model,
                    "cache_backend": self.response_cache.backend_name,
                    "cache_hit": cache_meta["cache_hit"],
                    "similarity": cache_meta["similarity"],
                    "semantic_eligible": cache_meta.get("semantic_eligible"),
                    "latency_ms": 0,
                    "usage": cached.get("usage"),
                }

        started = time.perf_counter()
        try:
            upstream = await adapter.chat_once(provider_req)
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504, detail="Upstream request timed out."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = (
                exc.response.text
                if exc.response is not None
                else "Upstream HTTP error."
            )
            raise HTTPException(status_code=502, detail=detail[:1000]) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        payload = normalize_chat_non_stream(model_alias=model_alias, upstream=upstream)
        latency_ms = int((time.perf_counter() - started) * 1000)
        upstream_choices = upstream.get("choices") or []
        upstream_has_tool_calls = bool(
            upstream_choices
            and (upstream_choices[0].get("message") or {}).get("tool_calls")
        )
        message_content = payload.get("message", {}).get("content", "")
        content_lower = (
            message_content.lower() if isinstance(message_content, str) else ""
        )
        if (
            self.runtime.cache_enabled
            and not upstream_has_tool_calls
            and not _content_has_tool_call(content_lower)
        ):
            await self.response_cache.store(
                exact_key=exact_key,
                model_alias=model_alias,
                provider=provider_name,
                messages=messages,
                response=payload,
            )
        return payload, {
            "provider": provider_name,
            "upstream_model": provider_req.model,
            "cache_backend": self.response_cache.backend_name,
            "cache_hit": "upstream",
            "similarity": 0.0,
            "semantic_eligible": False,
            "latency_ms": latency_ms,
            "usage": upstream.get("usage"),
        }

    async def chat_stream(
        self,
        *,
        model_alias: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        adapter, provider_req, provider_name = self._to_provider_request(
            model_alias=model_alias, messages=messages, stream=True, options=options
        )
        exact_key = self.response_cache.make_exact_key(
            model_alias=model_alias,
            provider=provider_name,
            messages=messages,
            options=options,
        )
        if self.runtime.cache_enabled:
            cached, _ = await self.response_cache.lookup(
                exact_key=exact_key,
                model_alias=model_alias,
                provider=provider_name,
                messages=messages,
            )
            if cached is not None:
                content = cached.get("message", {}).get("content", "")
                if content:
                    yield {
                        "model": model_alias,
                        "created_at": cached.get(
                            "created_at", datetime.now(UTC).isoformat()
                        ),
                        "message": {"role": "assistant", "content": content},
                        "done": False,
                    }
                yield {
                    "model": model_alias,
                    "created_at": cached.get(
                        "created_at", datetime.now(UTC).isoformat()
                    ),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": cached.get("done_reason", "stop"),
                }
                return

        content_sent = False
        collected: list[str] = []
        has_tool_calls = False
        saw_done = False
        try:
            async for upstream_chunk in adapter.chat_stream(provider_req):
                choices = upstream_chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("tool_calls"):
                        has_tool_calls = True
                chunk = normalize_chat_stream_chunk(
                    model_alias=model_alias, upstream=upstream_chunk
                )
                if chunk.get("done"):
                    saw_done = True
                content = chunk["message"]["content"]
                delta = chunk.get("delta") or {}
                has_emittable_delta = bool(
                    content or delta.get("tool_calls") or delta.get("role")
                )
                if content:
                    content_sent = True
                    collected.append(content)
                if has_emittable_delta or chunk.get("done"):
                    yield chunk
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504, detail="Upstream request timed out."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = "Upstream HTTP error."
            if exc.response is not None:
                status = exc.response.status_code
                detail = f"Upstream HTTP {status}."
                try:
                    raw = await exc.response.aread()
                except Exception:
                    raw = b""
                if raw:
                    text = raw.decode("utf-8", errors="replace").strip()
                    if text:
                        detail = text
            raise HTTPException(status_code=502, detail=detail[:1000]) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        else:
            full_content = "".join(collected)
            cacheable = (
                self.runtime.cache_enabled
                and collected
                and not has_tool_calls
                and not _content_has_tool_call(full_content.lower())
            )
            if cacheable:
                payload = {
                    "model": model_alias,
                    "created_at": datetime.now(UTC).isoformat(),
                    "message": {"role": "assistant", "content": full_content},
                    "done": True,
                    "done_reason": "stop",
                }
                await self.response_cache.store(
                    exact_key=exact_key,
                    model_alias=model_alias,
                    provider=provider_name,
                    messages=messages,
                    response=payload,
                )
            if not saw_done:
                yield done_chunk(model_alias=model_alias, had_content=content_sent)


def normalize_chat_non_stream(
    *, model_alias: str, upstream: dict[str, Any]
) -> dict[str, Any]:
    choices = upstream.get("choices") or []
    done_reason = "stop"
    if not choices:
        message: dict[str, Any] = {"role": "assistant", "content": ""}
    else:
        choice = choices[0]
        message = dict(choice.get("message") or {})
        if "role" not in message:
            message["role"] = "assistant"
        if message.get("content") is None:
            message["content"] = ""
        done_reason = choice.get("finish_reason") or "stop"

    payload = {
        "model": model_alias,
        "created_at": datetime.now(UTC).isoformat(),
        "message": message,
        "done": True,
        "done_reason": done_reason,
    }
    usage = upstream.get("usage")
    if isinstance(usage, dict):
        payload["usage"] = usage
    return payload


def normalize_chat_stream_chunk(
    *, model_alias: str, upstream: dict[str, Any]
) -> dict[str, Any]:
    choices = upstream.get("choices") or []
    delta: dict[str, Any] = {}
    finish_reason: str | None = None
    if choices:
        choice = choices[0]
        delta = dict(choice.get("delta") or {})
        finish_reason = choice.get("finish_reason")
    content = delta.get("content") or ""
    payload = {
        "model": model_alias,
        "created_at": datetime.now(UTC).isoformat(),
        "message": {"role": "assistant", "content": content},
        "delta": delta,
        "done": finish_reason is not None,
    }
    if finish_reason is not None:
        payload["done_reason"] = finish_reason
    return payload


def done_chunk(*, model_alias: str, had_content: bool) -> dict[str, Any]:
    return {
        "model": model_alias,
        "created_at": datetime.now(UTC).isoformat(),
        "message": {"role": "assistant", "content": "" if had_content else " "},
        "done": True,
        "done_reason": "stop",
    }
