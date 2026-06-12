from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from mdrouter.config import AppConfig
from mdrouter.models import (
    AnthropicChatRequest,
    AnthropicContentBlock,
    AnthropicMessage,
    OllamaChatRequest,
    OllamaGenerateRequest,
)
from mdrouter.runtime import RequestLogger
from mdrouter.runtime import RuntimeSettings
from mdrouter.router import ModelRouter

DEFAULT_CONFIG_PATH = Path("config/providers.json")
OLLAMA_COMPAT_VERSION = "0.12.6"


def load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(dotenv_path=Path(".env"), override=False)


class OllamaShowRequest(BaseModel):
    model: str


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    # DeepSeek thinking mode
    thinking: dict[str, Any] | None = None
    reasoning_effort: str | None = None


def create_app(config_path: str | Path = DEFAULT_CONFIG_PATH) -> FastAPI:
    load_env_file()
    config = AppConfig.from_file(config_path)
    runtime = RuntimeSettings.from_env()
    app = FastAPI(title="mdrouter", version="0.1.0")
    router = ModelRouter(config, runtime=runtime)
    request_logger = RequestLogger(runtime)

    def _upstream_cache_metrics(
        usage: dict[str, Any] | None,
    ) -> dict[str, int]:
        """Return unified upstream cache hit/miss token counts from any provider."""
        if not usage:
            return {
                "upstream_cache_hit_tokens": 0,
                "upstream_cache_miss_tokens": 0,
                "cached_tokens": 0,
            }
        # DeepSeek: top-level prompt_cache_hit_tokens / prompt_cache_miss_tokens
        hit = usage.get("prompt_cache_hit_tokens")
        miss = usage.get("prompt_cache_miss_tokens")
        if hit is not None or miss is not None:
            try:
                return {
                    "upstream_cache_hit_tokens": int(hit or 0),
                    "upstream_cache_miss_tokens": int(miss or 0),
                    "cached_tokens": int(hit or 0),
                }
            except (TypeError, ValueError):
                pass
        # Anthropic: top-level cache_read_input_tokens / cache_creation_input_tokens
        read = usage.get("cache_read_input_tokens")
        create = usage.get("cache_creation_input_tokens")
        if read is not None or create is not None:
            try:
                return {
                    "upstream_cache_hit_tokens": int(read or 0),
                    "upstream_cache_miss_tokens": int(create or 0),
                    "cached_tokens": int(read or 0),
                }
            except (TypeError, ValueError):
                pass
        # OpenAI-compat: nested usage.prompt_tokens_details.cached_tokens
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = details.get("cached_tokens", 0)
            try:
                hit_val = int(cached)
                prompt = usage.get("prompt_tokens", 0)
                try:
                    miss_val = max(0, int(prompt) - hit_val)
                except (TypeError, ValueError):
                    miss_val = 0
                return {
                    "upstream_cache_hit_tokens": hit_val,
                    "upstream_cache_miss_tokens": miss_val,
                    "cached_tokens": hit_val,
                }
            except (TypeError, ValueError):
                pass
        return {
            "upstream_cache_hit_tokens": 0,
            "upstream_cache_miss_tokens": 0,
            "cached_tokens": 0,
        }

    def visible_model_name(requested_model: str, meta: dict[str, Any] | None) -> str:
        if requested_model != "mdrouter/auto":
            return requested_model
        if not isinstance(meta, dict):
            return requested_model
        routed_alias = meta.get("routed_model_alias")
        if isinstance(routed_alias, str) and routed_alias.strip():
            return routed_alias.strip()
        upstream = meta.get("upstream_model")
        if isinstance(upstream, str) and upstream.strip():
            return upstream.strip()
        return requested_model

    def _content_input_chars(content: Any) -> int:
        if isinstance(content, str):
            return len(content)
        if isinstance(content, (list, dict)):
            return len(json.dumps(content, ensure_ascii=True, sort_keys=True))
        return len(str(content))

    def _iter_user_text(messages: list[dict[str, Any]]) -> str:
        chunks: list[str] = []
        for message in messages:
            if str(message.get("role")) != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
        return "\n".join(chunks)

    def _request_class_tag(messages: list[dict[str, Any]], options: dict[str, Any] | None) -> str:
        message_count = len(messages)
        input_chars = sum(_content_input_chars(msg.get("content", "")) for msg in messages)

        tool_def_count = 0
        if isinstance(options, dict):
            tools = options.get("tools")
            if isinstance(tools, list):
                tool_def_count = len(tools)
        history_tool_calls = 0
        for msg in messages:
            calls = msg.get("tool_calls")
            if isinstance(calls, list):
                history_tool_calls += len(calls)
        if tool_def_count > 0 or history_tool_calls > 0:
            return "tool_heavy"
        if message_count >= 16 or input_chars >= 12000:
            return "long_context"

        user_text = _iter_user_text(messages).lower()
        if re.search(r"\b(refactor|rewrite|migrate|re-architect|rearchitect)\b", user_text):
            return "heavy_refactor"
        return "default_coding"

    def _request_telemetry(messages: list[dict[str, Any]], options: dict[str, Any] | None) -> dict[str, Any]:
        message_count = len(messages)
        input_chars = sum(_content_input_chars(msg.get("content", "")) for msg in messages)
        tool_def_count = 0
        if isinstance(options, dict):
            tools = options.get("tools")
            if isinstance(tools, list):
                tool_def_count = len(tools)
        history_tool_calls = 0
        for msg in messages:
            calls = msg.get("tool_calls")
            if isinstance(calls, list):
                history_tool_calls += len(calls)
        return {
            "message_count": message_count,
            "input_chars": input_chars,
            "tool_call_count": history_tool_calls + tool_def_count,
            "request_class_tag": _request_class_tag(messages, options),
        }

    async def stream_chat_result(
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None,
        format_name: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        resolved_alias, request_class_tag = router._resolve_runtime_alias(  # noqa: SLF001
            model_alias=model,
            messages=messages,
            options=options,
        )
        adapter, provider_req, provider_name = router._to_provider_request(  # noqa: SLF001
            model_alias=model,
            messages=messages,
            stream=True,
            options=options,
            resolved_alias=resolved_alias,
        )
        exact_key = router.response_cache.make_exact_key(
            model_alias=resolved_alias,
            provider=provider_name,
            messages=messages,
            options=options,
        )
        cached = None
        cache_meta = {"cache_hit": "miss", "similarity": 0.0}
        if runtime.cache_enabled:
            cached, cache_meta = await router.response_cache.lookup(
                exact_key=exact_key,
                model_alias=resolved_alias,
                provider=provider_name,
                messages=messages,
            )
        return ([cached] if cached is not None else []), {
            "provider": provider_name,
            "upstream_model": provider_req.model,
            "routed_model_alias": resolved_alias,
            "request_class_tag": request_class_tag,
            "cache_backend": router.response_cache.backend_name,
            "cache_hit": cache_meta["cache_hit"],
            "similarity": cache_meta["similarity"],
            "semantic_eligible": cache_meta.get("semantic_eligible"),
            "exact_key": exact_key,
            "provider_req": provider_req,
            "format_name": format_name,
        }

    @app.middleware("http")
    async def access_log_middleware(req: Request, call_next):
        started = time.perf_counter()
        response = None
        try:
            response = await call_next(req)
            return response
        finally:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            request_logger.write(
                {
                    "event": "http_access",
                    "path": req.url.path,
                    "method": req.method,
                    "status": response.status_code if response is not None else None,
                    "latency_ms": elapsed_ms,
                    "client": req.client.host if req.client else None,
                }
            )

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "mdrouter",
            "cache_backend": router.response_cache.backend_name,
        }

    @app.get("/api/version")
    async def api_version() -> dict[str, str]:
        return {"version": OLLAMA_COMPAT_VERSION}

    @app.get("/api/tags")
    async def api_tags() -> dict[str, list[dict]]:
        return {"models": [m.model_dump() for m in router.list_models()]}

    @app.post("/api/chat")
    async def api_chat(payload: OllamaChatRequest, req: Request):
        messages = [m.model_dump() for m in payload.messages]
        started = time.perf_counter()
        if payload.stream:
            cached_entries, meta = await stream_chat_result(
                model=payload.model,
                messages=messages,
                options=payload.options,
                format_name="ollama",
            )

            async def stream() -> AsyncIterator[str]:
                if cached_entries:
                    cached = cached_entries[0]
                    content = cached.get("message", {}).get("content", "")
                    if content:
                        yield (
                            json.dumps(
                                {
                                    "model": payload.model,
                                    "created_at": cached.get("created_at"),
                                    "message": {
                                        "role": "assistant",
                                        "content": content,
                                    },
                                    "done": False,
                                }
                            )
                            + "\n"
                        )
                    yield (
                        json.dumps(
                            {
                                "model": payload.model,
                                "created_at": cached.get("created_at"),
                                "message": {"role": "assistant", "content": ""},
                                "done": True,
                                "done_reason": cached.get("done_reason", "stop"),
                            }
                        )
                        + "\n"
                    )
                    request_logger.write(
                        {
                            "path": "/api/chat",
                            "method": "POST",
                            "model": payload.model,
                            "model_alias": payload.model,
                            "stream": True,
                            "provider": meta.get("provider"),
                            "upstream_model": meta.get("upstream_model"),
                            "routed_model_alias": meta.get("routed_model_alias"),
                            "cache_backend": meta.get("cache_backend"),
                            "cache_hit": meta.get("cache_hit"),
                            "cache_hit_type": meta.get("cache_hit"),
                            "semantic_similarity": meta.get("similarity"),
                            "semantic_eligible": meta.get("semantic_eligible"),
                            "latency_ms": 0,
                            "client": req.client.host if req.client else None,
                            "status": 200,
                            "event": "stream_cache_hit",
                            **_request_telemetry(messages, payload.options),
                        }
                    )
                    return

                request_logger.write(
                    {
                        "path": "/api/chat",
                        "method": "POST",
                        "model": payload.model,
                        "model_alias": payload.model,
                        "stream": True,
                        "provider": meta.get("provider"),
                        "upstream_model": meta.get("upstream_model"),
                        "routed_model_alias": meta.get("routed_model_alias"),
                        "cache_backend": meta.get("cache_backend"),
                        "cache_hit": "miss",
                        "cache_hit_type": "miss",
                        "semantic_eligible": meta.get("semantic_eligible"),
                        "client": req.client.host if req.client else None,
                        "status": 200,
                        "event": "stream_cache_miss",
                        **_request_telemetry(messages, payload.options),
                    }
                )
                stream_collected: list[str] = []
                try:
                    async for chunk in router.chat_stream(
                        model_alias=payload.model,
                        messages=messages,
                        options=payload.options,
                    ):
                        if chunk["message"]["content"]:
                            stream_collected.append(chunk["message"]["content"])
                        yield json.dumps(chunk) + "\n"
                except HTTPException as exc:
                    yield (
                        json.dumps(
                            {
                                "model": payload.model,
                                "message": {"role": "assistant", "content": ""},
                                "done": True,
                                "error": {
                                    "status": exc.status_code,
                                    "message": str(exc.detail),
                                },
                            }
                        )
                        + "\n"
                    )
                request_logger.write(
                    {
                        "path": "/api/chat",
                        "method": "POST",
                        "model": payload.model,
                        "model_alias": payload.model,
                        "stream": True,
                        "provider": meta.get("provider"),
                        "upstream_model": meta.get("upstream_model"),
                        "routed_model_alias": meta.get("routed_model_alias"),
                        "client": req.client.host if req.client else None,
                        "status": 200,
                        "event": "stream_done",
                        "response_body": {"content": "".join(stream_collected)}
                        if runtime.log_response_body
                        else None,
                        **_request_telemetry(messages, payload.options),
                    }
                )

            request_logger.write(
                {
                    "path": "/api/chat",
                    "method": "POST",
                    "model": payload.model,
                    "model_alias": payload.model,
                    "stream": True,
                    "client": req.client.host if req.client else None,
                    "event": "request_start",
                    "request_body": payload.model_dump()
                    if runtime.log_request_body
                    else None,
                    **_request_telemetry(messages, payload.options),
                }
            )
            return StreamingResponse(stream(), media_type="application/x-ndjson")

        response_payload, meta = await router.chat_once(
            model_alias=payload.model, messages=messages, options=payload.options
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        usage = (
            response_payload.get("usage")
            if isinstance(response_payload, dict)
            else None
        )
        request_logger.write(
            {
                "path": "/api/chat",
                "method": "POST",
                "model": payload.model,
                "model_alias": payload.model,
                "stream": False,
                "provider": meta.get("provider"),
                "upstream_model": meta.get("upstream_model"),
                "routed_model_alias": meta.get("routed_model_alias"),
                "cache_backend": meta.get("cache_backend"),
                "cache_hit": meta.get("cache_hit"),
                "cache_hit_type": meta.get("cache_hit"),
                "semantic_similarity": meta.get("similarity"),
                "semantic_eligible": meta.get("semantic_eligible"),
                "latency_ms": meta.get("latency_ms", elapsed_ms),
                "client": req.client.host if req.client else None,
                "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "completion_tokens": (usage or {}).get("completion_tokens"),
                **_upstream_cache_metrics(usage),
                "status": 200,
                "response_body": response_payload
                if runtime.log_response_body
                else None,
                "request_body": payload.model_dump()
                if runtime.log_request_body
                else None,
                **_request_telemetry(messages, payload.options),
            }
        )
        return JSONResponse(response_payload)

    @app.post("/api/generate")
    async def api_generate(payload: OllamaGenerateRequest, req: Request):
        messages = []
        if payload.system:
            messages.append({"role": "system", "content": payload.system})
        messages.append({"role": "user", "content": payload.prompt})

        started = time.perf_counter()
        if payload.stream:
            cached_entries, meta = await stream_chat_result(
                model=payload.model,
                messages=messages,
                options=payload.options,
                format_name="ollama_generate",
            )

            async def stream() -> AsyncIterator[str]:
                if cached_entries:
                    cached = cached_entries[0]
                    response_text = cached.get("message", {}).get("content", "")
                    if response_text:
                        yield (
                            json.dumps(
                                {
                                    "model": payload.model,
                                    "created_at": cached.get("created_at"),
                                    "response": response_text,
                                    "done": False,
                                }
                            )
                            + "\n"
                        )
                    yield (
                        json.dumps(
                            {
                                "model": payload.model,
                                "created_at": cached.get("created_at"),
                                "response": "",
                                "done": True,
                                "done_reason": cached.get("done_reason", "stop"),
                            }
                        )
                        + "\n"
                    )
                    request_logger.write(
                        {
                            "path": "/api/generate",
                            "method": "POST",
                            "model": payload.model,
                            "model_alias": payload.model,
                            "stream": True,
                            "provider": meta.get("provider"),
                            "upstream_model": meta.get("upstream_model"),
                            "routed_model_alias": meta.get("routed_model_alias"),
                            "cache_backend": meta.get("cache_backend"),
                            "cache_hit": meta.get("cache_hit"),
                            "cache_hit_type": meta.get("cache_hit"),
                            "semantic_similarity": meta.get("similarity"),
                            "semantic_eligible": meta.get("semantic_eligible"),
                            "latency_ms": 0,
                            "client": req.client.host if req.client else None,
                            "status": 200,
                            "event": "stream_cache_hit",
                            **_request_telemetry(messages, payload.options),
                        }
                    )
                    return

                request_logger.write(
                    {
                        "path": "/api/generate",
                        "method": "POST",
                        "model": payload.model,
                        "model_alias": payload.model,
                        "stream": True,
                        "provider": meta.get("provider"),
                        "upstream_model": meta.get("upstream_model"),
                        "routed_model_alias": meta.get("routed_model_alias"),
                        "cache_backend": meta.get("cache_backend"),
                        "cache_hit": "miss",
                        "cache_hit_type": "miss",
                        "semantic_eligible": meta.get("semantic_eligible"),
                        "client": req.client.host if req.client else None,
                        "status": 200,
                        "event": "stream_cache_miss",
                        **_request_telemetry(messages, payload.options),
                    }
                )
                stream_collected: list[str] = []
                try:
                    async for chunk in router.chat_stream(
                        model_alias=payload.model,
                        messages=messages,
                        options=payload.options,
                    ):
                        output = {
                            "model": chunk["model"],
                            "created_at": chunk["created_at"],
                            "response": chunk["message"]["content"],
                            "done": chunk["done"],
                        }
                        if chunk["message"]["content"]:
                            stream_collected.append(chunk["message"]["content"])
                        if chunk["done"]:
                            output["done_reason"] = chunk.get("done_reason", "stop")
                        yield json.dumps(output) + "\n"
                except HTTPException as exc:
                    yield (
                        json.dumps(
                            {
                                "model": payload.model,
                                "response": "",
                                "done": True,
                                "done_reason": "error",
                                "error": {
                                    "status": exc.status_code,
                                    "message": str(exc.detail),
                                },
                            }
                        )
                        + "\n"
                    )
                request_logger.write(
                    {
                        "path": "/api/generate",
                        "method": "POST",
                        "model": payload.model,
                        "model_alias": payload.model,
                        "stream": True,
                        "provider": meta.get("provider"),
                        "upstream_model": meta.get("upstream_model"),
                        "routed_model_alias": meta.get("routed_model_alias"),
                        "client": req.client.host if req.client else None,
                        "status": 200,
                        "event": "stream_done",
                        "response_body": {"content": "".join(stream_collected)}
                        if runtime.log_response_body
                        else None,
                        **_request_telemetry(messages, payload.options),
                    }
                )

            request_logger.write(
                {
                    "path": "/api/generate",
                    "method": "POST",
                    "model": payload.model,
                    "model_alias": payload.model,
                    "stream": True,
                    "client": req.client.host if req.client else None,
                    "event": "request_start",
                    "request_body": payload.model_dump()
                    if runtime.log_request_body
                    else None,
                    **_request_telemetry(messages, payload.options),
                }
            )
            return StreamingResponse(stream(), media_type="application/x-ndjson")

        chat_payload, meta = await router.chat_once(
            model_alias=payload.model, messages=messages, options=payload.options
        )
        response_payload = {
            "model": chat_payload["model"],
            "created_at": chat_payload["created_at"],
            "response": chat_payload["message"]["content"],
            "done": True,
            "done_reason": chat_payload.get("done_reason", "stop"),
        }
        usage = chat_payload.get("usage") if isinstance(chat_payload, dict) else None
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        request_logger.write(
            {
                "path": "/api/generate",
                "method": "POST",
                "model": payload.model,
                "model_alias": payload.model,
                "stream": False,
                "provider": meta.get("provider"),
                "upstream_model": meta.get("upstream_model"),
                "routed_model_alias": meta.get("routed_model_alias"),
                "cache_backend": meta.get("cache_backend"),
                "cache_hit": meta.get("cache_hit"),
                "cache_hit_type": meta.get("cache_hit"),
                "semantic_similarity": meta.get("similarity"),
                "semantic_eligible": meta.get("semantic_eligible"),
                "latency_ms": meta.get("latency_ms", elapsed_ms),
                "client": req.client.host if req.client else None,
                "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "completion_tokens": (usage or {}).get("completion_tokens"),
                **_upstream_cache_metrics(usage),
                "status": 200,
                "response_body": response_payload
                if runtime.log_response_body
                else None,
                "request_body": payload.model_dump()
                if runtime.log_request_body
                else None,
                **_request_telemetry(messages, payload.options),
            }
        )
        return JSONResponse(response_payload)

    @app.post("/api/show")
    async def api_show(request: OllamaShowRequest) -> dict[str, Any]:
        if request.model == "mdrouter/auto":
            auto_context_length = router.auto_context_length()
            return {
                "template": "",
                "capabilities": ["vision", "tools"],
                "details": {"family": "router"},
                "model": "mdrouter/auto",
                "remote_model": "mdrouter/auto",
                "model_info": {
                    "general.basename": "MDAuto",
                    "general.architecture": "router",
                    "router.context_length": auto_context_length,
                    "llama.context_length": auto_context_length,
                },
            }

        try:
            resolved_alias, model_cfg = router.lookup_model_config(request.model)
        except HTTPException:
            return {
                "template": "",
                "capabilities": [],
                "details": {"family": "router"},
                "model_info": {
                    "general.basename": request.model,
                    "general.architecture": "router",
                    "router.context_length": 32768,
                    "llama.context_length": 32768,
                },
            }

        caps = []
        if "vision" in model_cfg.capabilities:
            caps.append("vision")
        if "tools" in model_cfg.capabilities:
            caps.append("tools")
        return {
            "template": "",
            "capabilities": caps,
            "details": {"family": model_cfg.provider},
            "model": resolved_alias,
            "remote_model": model_cfg.upstream_model,
            "model_info": {
                "general.basename": resolved_alias,
                "general.architecture": "router",
                "router.upstream_model": model_cfg.upstream_model,
                "router.context_length": model_cfg.context_length,
                "llama.context_length": model_cfg.context_length,
            },
        }

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(payload: OpenAIChatRequest, req: Request):
        # Extract thinking/reasoning_effort from payload (not options) so they
        # land at the top level of the upstream payload.
        options = payload.model_dump(
            exclude={"model", "messages", "stream", "thinking", "reasoning_effort"},
            exclude_none=True,
        )
        thinking = payload.thinking
        reasoning_effort = payload.reasoning_effort
        if thinking is not None:
            options["thinking"] = thinking
        if reasoning_effort is not None:
            options["reasoning_effort"] = reasoning_effort

        started = time.perf_counter()
        if payload.stream:
            cached_entries, meta = await stream_chat_result(
                model=payload.model,
                messages=payload.messages,
                options=options or None,
                format_name="openai_chat",
            )
            response_model_name = visible_model_name(payload.model, meta)

            async def stream() -> AsyncIterator[str]:
                if cached_entries:
                    cached = cached_entries[0]
                    content = cached.get("message", {}).get("content", "")
                    if content:
                        yield f"data: {json.dumps({'id': 'chatcmpl-router', 'object': 'chat.completion.chunk', 'created': 0, 'model': response_model_name, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"
                    yield f"data: {json.dumps({'id': 'chatcmpl-router', 'object': 'chat.completion.chunk', 'created': 0, 'model': response_model_name, 'choices': [{'index': 0, 'delta': {'content': ''}, 'finish_reason': cached.get('done_reason', 'stop')}]})}\n\n"
                    yield "data: [DONE]\n\n"
                    request_logger.write(
                        {
                            "path": "/v1/chat/completions",
                            "method": "POST",
                            "model": payload.model,
                            "model_alias": payload.model,
                            "stream": True,
                            "provider": meta.get("provider"),
                            "upstream_model": meta.get("upstream_model"),
                            "routed_model_alias": meta.get("routed_model_alias"),
                            "cache_backend": meta.get("cache_backend"),
                            "cache_hit": meta.get("cache_hit"),
                            "cache_hit_type": meta.get("cache_hit"),
                            "semantic_similarity": meta.get("similarity"),
                            "semantic_eligible": meta.get("semantic_eligible"),
                            "latency_ms": 0,
                            "client": req.client.host if req.client else None,
                            "status": 200,
                            "event": "stream_cache_hit",
                            **_request_telemetry(payload.messages, options or None),
                        }
                    )
                    return

                request_logger.write(
                    {
                        "path": "/v1/chat/completions",
                        "method": "POST",
                        "model": payload.model,
                        "model_alias": payload.model,
                        "stream": True,
                        "provider": meta.get("provider"),
                        "upstream_model": meta.get("upstream_model"),
                        "routed_model_alias": meta.get("routed_model_alias"),
                        "cache_backend": meta.get("cache_backend"),
                        "cache_hit": "miss",
                        "cache_hit_type": "miss",
                        "semantic_eligible": meta.get("semantic_eligible"),
                        "client": req.client.host if req.client else None,
                        "status": 200,
                        "event": "stream_cache_miss",
                        **_request_telemetry(payload.messages, options or None),
                    }
                )
                stream_collected: list[str] = []
                stream_usage: dict[str, Any] | None = None
                try:
                    async for chunk in router.chat_stream(
                        model_alias=payload.model,
                        messages=payload.messages,
                        options=options or None,
                    ):
                        if chunk["message"]["content"]:
                            stream_collected.append(chunk["message"]["content"])
                        if isinstance(chunk.get("usage"), dict):
                            stream_usage = chunk["usage"]
                        delta = chunk.get("delta") or {
                            "content": chunk["message"]["content"]
                        }
                        choice = {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": chunk.get("done_reason")
                            if chunk.get("done")
                            else None,
                        }
                        chunk_payload = {
                            "id": "chatcmpl-router",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": response_model_name,
                            "choices": [choice],
                        }
                        yield f"data: {json.dumps(chunk_payload)}\n\n"
                        # Emit a separate usage-only chunk (OpenAI spec: choices=[])
                        # after the finish_reason chunk so all clients see it.
                        if chunk.get("done") and isinstance(chunk.get("usage"), dict):
                            usage_payload = {
                                "id": "chatcmpl-router",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": response_model_name,
                                "choices": [],
                                "usage": chunk["usage"],
                            }
                            yield f"data: {json.dumps(usage_payload)}\n\n"
                except HTTPException as exc:
                    error_message = f"[upstream_error:{exc.status_code}] {exc.detail}"
                    chunk_payload = {
                        "id": "chatcmpl-router",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": response_model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": error_message,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_payload)}\n\n"
                yield "data: [DONE]\n\n"
                request_logger.write(
                    {
                        "path": "/v1/chat/completions",
                        "method": "POST",
                        "model": payload.model,
                        "model_alias": payload.model,
                        "stream": True,
                        "provider": meta.get("provider"),
                        "upstream_model": meta.get("upstream_model"),
                        "routed_model_alias": meta.get("routed_model_alias"),
                        "client": req.client.host if req.client else None,
                        "status": 200,
                        "event": "stream_done",
                        "prompt_tokens": (stream_usage or {}).get("prompt_tokens"),
                        "completion_tokens": (stream_usage or {}).get("completion_tokens"),
                        **_upstream_cache_metrics(stream_usage),
                        "response_body": {"content": "".join(stream_collected)}
                        if runtime.log_response_body
                        else None,
                        **_request_telemetry(payload.messages, options or None),
                    }
                )

            request_logger.write(
                {
                    "path": "/v1/chat/completions",
                    "method": "POST",
                    "model": payload.model,
                    "model_alias": payload.model,
                    "stream": True,
                    "client": req.client.host if req.client else None,
                    "event": "request_start",
                    "request_body": payload.model_dump()
                    if runtime.log_request_body
                    else None,
                    **_request_telemetry(payload.messages, options or None),
                }
            )
            return StreamingResponse(stream(), media_type="text/event-stream")

        chat_payload, meta = await router.chat_once(
            model_alias=payload.model,
            messages=payload.messages,
            options=options or None,
        )
        response_model_name = visible_model_name(payload.model, meta)
        usage = chat_payload.get("usage") if isinstance(chat_payload, dict) else None
        response = {
            "id": "chatcmpl-router",
            "object": "chat.completion",
            "created": 0,
            "model": response_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": chat_payload["message"],
                    "finish_reason": chat_payload.get("done_reason", "stop"),
                }
            ],
            "usage": usage
            or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        request_logger.write(
            {
                "path": "/v1/chat/completions",
                "method": "POST",
                "model": payload.model,
                "model_alias": payload.model,
                "stream": False,
                "provider": meta.get("provider"),
                "upstream_model": meta.get("upstream_model"),
                "routed_model_alias": meta.get("routed_model_alias"),
                "cache_backend": meta.get("cache_backend"),
                "cache_hit": meta.get("cache_hit"),
                "cache_hit_type": meta.get("cache_hit"),
                "semantic_similarity": meta.get("similarity"),
                "semantic_eligible": meta.get("semantic_eligible"),
                "latency_ms": meta.get("latency_ms", elapsed_ms),
                "client": req.client.host if req.client else None,
                "prompt_tokens": (response.get("usage") or {}).get("prompt_tokens"),
                "completion_tokens": (response.get("usage") or {}).get(
                    "completion_tokens"
                ),
                **_upstream_cache_metrics(response.get("usage")),
                "status": 200,
                "response_body": response if runtime.log_response_body else None,
                "request_body": payload.model_dump()
                if runtime.log_request_body
                else None,
                **_request_telemetry(payload.messages, options or None),
            }
        )
        return JSONResponse(response)

    # ---------- Anthropic ↔ internal conversion helpers ----------

    def _anthropic_messages_to_openai(
        messages: list[AnthropicMessage], system: str | list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        """Convert Anthropic messages to OpenAI-shaped internal format."""
        result: list[dict[str, Any]] = []

        # Inject system prompt as first message
        if isinstance(system, str) and system.strip():
            result.append({"role": "system", "content": system})
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text.strip():
                        result.append({"role": "system", "content": text})

        for msg in messages:
            role = msg.role
            content = msg.content

            if role == "assistant":
                openai_msg: dict[str, Any] = {"role": "assistant"}
                if isinstance(content, str):
                    openai_msg["content"] = content
                elif isinstance(content, list):
                    text_parts: list[str] = []
                    tool_calls: list[dict[str, Any]] = []
                    for block in content:
                        bt = block.type if isinstance(block, AnthropicContentBlock) else (block.get("type") if isinstance(block, dict) else "")
                        if bt == "text":
                            t = block.text if isinstance(block, AnthropicContentBlock) else (block.get("text") if isinstance(block, dict) else "")
                            if t:
                                text_parts.append(t)
                        elif bt == "tool_use":
                            blk_dict = block.model_dump() if isinstance(block, AnthropicContentBlock) else (block if isinstance(block, dict) else {})
                            tool_calls.append({
                                "id": blk_dict.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": blk_dict.get("name", ""),
                                    "arguments": json.dumps(blk_dict.get("input", {}), ensure_ascii=True),
                                },
                            })
                    openai_msg["content"] = "\n".join(text_parts) if text_parts else ""
                    if tool_calls:
                        openai_msg["tool_calls"] = tool_calls
                else:
                    openai_msg["content"] = str(content) if content else ""
                result.append(openai_msg)

            elif role == "user":
                if isinstance(content, list):
                    # Separate tool_result blocks from regular content blocks.
                    # Each tool_result becomes its own "tool" message so that
                    # OpenAI providers see assistant(tool_calls) → tool → tool → ...
                    text_parts: list[dict[str, Any]] = []
                    for block in content:
                        bt = block.type if isinstance(block, AnthropicContentBlock) else (block.get("type") if isinstance(block, dict) else "")
                        if bt == "tool_result":
                            blk_dict = block.model_dump() if isinstance(block, AnthropicContentBlock) else (block if isinstance(block, dict) else {})
                            tc = blk_dict.get("content", "")
                            if isinstance(tc, list):
                                tc_text = "\n".join(
                                    (c.get("text", "") if isinstance(c, dict) else str(c))
                                    for c in tc
                                )
                            else:
                                tc_text = str(tc)
                            # Emit any pending text blocks as a user message first
                            if text_parts:
                                result.append({"role": "user", "content": text_parts})
                                text_parts = []
                            result.append({
                                "role": "tool",
                                "content": tc_text,
                                "tool_call_id": blk_dict.get("tool_use_id", ""),
                            })
                        elif bt == "image":
                            src = block.source if isinstance(block, AnthropicContentBlock) else (block.get("source") if isinstance(block, dict) else {})
                            if isinstance(src, dict):
                                media_type = src.get("media_type", "image/jpeg")
                                data = src.get("data", "")
                                url = f"data:{media_type};base64,{data}"
                            else:
                                url = str(src) if src else ""
                            text_parts.append({"type": "image_url", "image_url": {"url": url}})
                        elif bt == "text":
                            t = block.text if isinstance(block, AnthropicContentBlock) else (block.get("text") if isinstance(block, dict) else "")
                            text_parts.append({"type": "text", "text": t})
                        else:
                            text_parts.append({"type": "text", "text": str(block)})
                    if text_parts:
                        result.append({"role": "user", "content": text_parts})
                elif isinstance(content, str):
                    result.append({"role": "user", "content": content})
                else:
                    result.append({"role": "user", "content": str(content) if content else ""})

        return result

    def _anthropic_tools_to_openai(
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Convert Anthropic tools to OpenAI format."""
        if not tools:
            return None
        result: list[dict[str, Any]] = []
        for tool in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return result

    def _normalized_to_anthropic_response(
        payload: dict[str, Any], *, model_name: str
    ) -> dict[str, Any]:
        """Convert router's normalized response to Anthropic Messages format."""
        import uuid

        message = payload.get("message") or {}
        content_text = message.get("content", "")
        tool_calls = message.get("tool_calls") or []

        # Build Anthropic content blocks
        content_blocks: list[dict[str, Any]] = []
        if isinstance(content_text, str) and content_text.strip():
            content_blocks.append({"type": "text", "text": content_text})

        for tc in tool_calls:
            func = tc.get("function", {})
            try:
                tool_input = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                tool_input = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": func.get("name", ""),
                "input": tool_input,
            })

        if not content_blocks:
            content_blocks = [{"type": "text", "text": ""}]

        # Map finish_reason
        done_reason = payload.get("done_reason", "stop")
        stop_reason_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }
        stop_reason = stop_reason_map.get(done_reason, done_reason)

        # Map usage
        usage = payload.get("usage") or {}
        anthropic_usage = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }

        return {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": model_name,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": anthropic_usage,
        }

    # ---------- POST /v1/messages (Anthropic-compatible) ----------

    @app.post("/v1/messages")
    async def v1_messages(payload: AnthropicChatRequest, req: Request):
        """Anthropic-compatible Messages endpoint for Claude Code etc."""
        # Convert Anthropic messages → internal OpenAI format
        openai_messages = _anthropic_messages_to_openai(
            payload.messages, payload.system
        )

        # Build options dict from Anthropic request
        options: dict[str, Any] = {}

        if payload.tools:
            openai_tools = _anthropic_tools_to_openai(
                [t.model_dump() if hasattr(t, "model_dump") else t for t in payload.tools]
            )
            if openai_tools:
                options["tools"] = openai_tools

        if payload.tool_choice is not None:
            options["tool_choice"] = payload.tool_choice

        if payload.temperature is not None:
            options["temperature"] = payload.temperature
        if payload.top_p is not None:
            options["top_p"] = payload.top_p
        if payload.max_tokens:
            options["max_tokens"] = payload.max_tokens
        if payload.thinking is not None:
            options["thinking"] = payload.thinking
        if payload.stop_sequences:
            options["stop"] = payload.stop_sequences

        started = time.perf_counter()

        if payload.stream:
            cached_entries, meta = await stream_chat_result(
                model=payload.model,
                messages=openai_messages,
                options=options or None,
                format_name="anthropic_messages",
            )
            response_model_name = visible_model_name(payload.model, meta)

            async def stream() -> AsyncIterator[str]:
                if cached_entries:
                    cached = cached_entries[0]
                    anthropic_resp = _normalized_to_anthropic_response(
                        cached, model_name=response_model_name
                    )
                    content_text = cached.get("message", {}).get("content", "")
                    if content_text:
                        msg_id = anthropic_resp["id"]
                        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': response_model_name, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content_text}})}\n\n"
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': anthropic_resp['stop_reason'], 'stop_sequence': None}, 'usage': anthropic_resp['usage']})}\n\n"
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

                    request_logger.write(
                        {
                            "path": "/v1/messages",
                            "method": "POST",
                            "model": payload.model,
                            "model_alias": payload.model,
                            "stream": True,
                            "provider": meta.get("provider"),
                            "upstream_model": meta.get("upstream_model"),
                            "routed_model_alias": meta.get("routed_model_alias"),
                            "cache_backend": meta.get("cache_backend"),
                            "cache_hit": meta.get("cache_hit"),
                            "cache_hit_type": meta.get("cache_hit"),
                            "semantic_similarity": meta.get("similarity"),
                            "semantic_eligible": meta.get("semantic_eligible"),
                            "latency_ms": 0,
                            "client": req.client.host if req.client else None,
                            "status": 200,
                            "event": "stream_cache_hit",
                            **_request_telemetry(openai_messages, options or None),
                        }
                    )
                    return

                request_logger.write(
                    {
                        "path": "/v1/messages",
                        "method": "POST",
                        "model": payload.model,
                        "model_alias": payload.model,
                        "stream": True,
                        "provider": meta.get("provider"),
                        "upstream_model": meta.get("upstream_model"),
                        "routed_model_alias": meta.get("routed_model_alias"),
                        "cache_backend": meta.get("cache_backend"),
                        "cache_hit": "miss",
                        "cache_hit_type": "miss",
                        "semantic_eligible": meta.get("semantic_eligible"),
                        "client": req.client.host if req.client else None,
                        "status": 200,
                        "event": "stream_cache_miss",
                        **_request_telemetry(openai_messages, options or None),
                    }
                )

                stream_collected: list[str] = []
                stream_usage: dict[str, Any] | None = None
                message_start_sent = False
                text_block_open = False
                text_block_index = 0
                tool_use_blocks: dict[int, dict[str, Any]] = {}
                next_tool_index = 1
                stop_reason = "end_turn"

                try:
                    async for chunk in router.chat_stream(
                        model_alias=payload.model,
                        messages=openai_messages,
                        options=options or None,
                    ):
                        delta = chunk.get("delta") or {}
                        content = chunk["message"]["content"]
                        tool_calls_delta = delta.get("tool_calls") or []

                        # message_start
                        if not message_start_sent:
                            message_start_sent = True
                            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': f'msg_{uuid.uuid4().hex[:24]}', 'type': 'message', 'role': 'assistant', 'content': [], 'model': response_model_name, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

                        # Text content
                        if content:
                            if not text_block_open:
                                text_block_open = True
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_block_index, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"
                            stream_collected.append(content)

                        # Tool calls — close text block first, then accumulate args
                        if tool_calls_delta and text_block_open:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_block_index})}\n\n"
                            text_block_open = False

                        for tc in tool_calls_delta:
                            tc_idx = tc.get("index", 0)
                            tc_id = tc.get("id") or ""
                            tc_name = tc.get("function", {}).get("name") or ""
                            tc_args = tc.get("function", {}).get("arguments") or ""

                            if tc_idx not in tool_use_blocks:
                                tool_use_blocks[tc_idx] = {
                                    "id": tc_id,
                                    "name": tc_name,
                                    "arguments": tc_args,
                                }
                                next_tool_index = max(next_tool_index, tc_idx + 1)
                            else:
                                if tc_id and not tool_use_blocks[tc_idx]["id"]:
                                    tool_use_blocks[tc_idx]["id"] = tc_id
                                if tc_name and not tool_use_blocks[tc_idx]["name"]:
                                    tool_use_blocks[tc_idx]["name"] = tc_name
                                tool_use_blocks[tc_idx]["arguments"] += tc_args


                        # Done
                        if chunk.get("done"):
                            if text_block_open:
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_block_index})}\n\n"
                                text_block_open = False
                            stop_reason_map = {
                                "stop": "end_turn",
                                "tool_calls": "tool_use",
                                "length": "max_tokens",
                            }
                            stop_reason = stop_reason_map.get(
                                chunk.get("done_reason", "stop"), "end_turn"
                            )
                            if isinstance(chunk.get("usage"), dict):
                                stream_usage = chunk["usage"]

                            # Emit accumulated tool_use blocks now that args are complete
                            if tool_use_blocks:
                                for tc_idx in sorted(tool_use_blocks.keys()):
                                    tb = tool_use_blocks[tc_idx]
                                    tc_index = text_block_index + 1 + tc_idx
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': tc_index, 'content_block': {'type': 'tool_use', 'id': tb['id'], 'name': tb['name'], 'input': {}}})}\n\n"
                                    # Send accumulated args as partial JSON
                                    try:
                                        parsed = json.loads(tb['arguments'])
                                        args_json = json.dumps(parsed, ensure_ascii=True)
                                    except (json.JSONDecodeError, TypeError):
                                        args_json = tb['arguments']
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': tc_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tc_index})}\n\n"

                except HTTPException as exc:
                    error_message = f"[upstream_error:{exc.status_code}] {exc.detail}"
                    if not message_start_sent:
                        message_start_sent = True
                        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': f'msg_{uuid.uuid4().hex[:24]}', 'type': 'message', 'role': 'assistant', 'content': [], 'model': response_model_name, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
                    if not text_block_open:
                        text_block_open = True
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_block_index, 'delta': {'type': 'text_delta', 'text': error_message}})}\n\n"
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_block_index})}\n\n"
                    stop_reason = "end_turn"

                anthropic_usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
                if isinstance(stream_usage, dict):
                    anthropic_usage = {
                        "input_tokens": stream_usage.get("prompt_tokens", 0),
                        "output_tokens": stream_usage.get("completion_tokens", 0),
                    }

                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': anthropic_usage})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

                request_logger.write(
                    {
                        "path": "/v1/messages",
                        "method": "POST",
                        "model": payload.model,
                        "model_alias": payload.model,
                        "stream": True,
                        "provider": meta.get("provider"),
                        "upstream_model": meta.get("upstream_model"),
                        "routed_model_alias": meta.get("routed_model_alias"),
                        "client": req.client.host if req.client else None,
                        "status": 200,
                        "event": "stream_done",
                        "prompt_tokens": (stream_usage or {}).get("prompt_tokens"),
                        "completion_tokens": (stream_usage or {}).get("completion_tokens"),
                        **_upstream_cache_metrics(stream_usage),
                        "response_body": {"content": "".join(stream_collected)}
                        if runtime.log_response_body
                        else None,
                        **_request_telemetry(openai_messages, options or None),
                    }
                )

            request_logger.write(
                {
                    "path": "/v1/messages",
                    "method": "POST",
                    "model": payload.model,
                    "model_alias": payload.model,
                    "stream": True,
                    "client": req.client.host if req.client else None,
                    "event": "request_start",
                    "request_body": payload.model_dump()
                    if runtime.log_request_body
                    else None,
                    **_request_telemetry(openai_messages, options or None),
                }
            )
            return StreamingResponse(stream(), media_type="text/event-stream")

        # Non-streaming
        chat_payload, meta = await router.chat_once(
            model_alias=payload.model,
            messages=openai_messages,
            options=options or None,
        )
        response_model_name = visible_model_name(payload.model, meta)
        anthropic_resp = _normalized_to_anthropic_response(
            chat_payload, model_name=response_model_name
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        usage = chat_payload.get("usage") or {}
        request_logger.write(
            {
                "path": "/v1/messages",
                "method": "POST",
                "model": payload.model,
                "model_alias": payload.model,
                "stream": False,
                "provider": meta.get("provider"),
                "upstream_model": meta.get("upstream_model"),
                "routed_model_alias": meta.get("routed_model_alias"),
                "cache_backend": meta.get("cache_backend"),
                "cache_hit": meta.get("cache_hit"),
                "cache_hit_type": meta.get("cache_hit"),
                "semantic_similarity": meta.get("similarity"),
                "semantic_eligible": meta.get("semantic_eligible"),
                "latency_ms": meta.get("latency_ms", elapsed_ms),
                "client": req.client.host if req.client else None,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                **_upstream_cache_metrics(usage),
                "status": 200,
                "response_body": anthropic_resp
                if runtime.log_response_body
                else None,
                "request_body": payload.model_dump()
                if runtime.log_request_body
                else None,
                **_request_telemetry(openai_messages, options or None),
            }
        )
        return JSONResponse(anthropic_resp)

    @app.get("/v1/models")
    async def v1_models() -> dict[str, Any]:
        """Anthropic-compatible list models endpoint for Claude Code validation."""
        model_list: list[dict[str, Any]] = []
        for alias, model_cfg in router.config.models.items():
            model_list.append({
                "id": model_cfg.upstream_model,
                "display_name": alias,
                "type": "model",
                "created_at": "2025-01-01T00:00:00Z",
            })
        return {"data": model_list, "has_more": False, "first_id": model_list[0]["id"] if model_list else None, "last_id": model_list[-1]["id"] if model_list else None}

    return app


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"status", "stats", "cachestatus"}:
        from mdrouter.ops import main as ops_main

        ops_main()
        return

    parser = argparse.ArgumentParser(description="Run Ollama-compatible router.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to providers.json",
    )
    args = parser.parse_args()

    load_env_file()
    config = AppConfig.from_file(args.config)
    app = create_app(args.config)

    import uvicorn

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
    )


if __name__ == "__main__":
    main()
