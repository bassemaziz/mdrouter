from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from mdrouter.config import AppConfig
from mdrouter.models import OllamaChatRequest, OllamaGenerateRequest
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


def create_app(config_path: str | Path = DEFAULT_CONFIG_PATH) -> FastAPI:
    load_env_file()
    config = AppConfig.from_file(config_path)
    runtime = RuntimeSettings.from_env()
    app = FastAPI(title="mdrouter", version="0.1.0")
    router = ModelRouter(config, runtime=runtime)
    request_logger = RequestLogger(runtime)

    def cached_tokens_from_usage(usage: dict[str, Any] | None) -> int:
        if not usage:
            return 0
        details = usage.get("prompt_tokens_details")
        if not isinstance(details, dict):
            return 0
        value = details.get("cached_tokens", 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def visible_model_name(requested_model: str, meta: dict[str, Any] | None) -> str:
        if requested_model != "mdrouter/auto":
            return requested_model
        if not isinstance(meta, dict):
            return requested_model
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
                "cached_tokens": cached_tokens_from_usage(usage),
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
                "cached_tokens": cached_tokens_from_usage(usage),
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

        basename = model_cfg.upstream_model.split("/")[-1]
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
                "general.basename": basename,
                "general.architecture": "router",
                "router.context_length": model_cfg.context_length,
                "llama.context_length": model_cfg.context_length,
            },
        }

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(payload: OpenAIChatRequest, req: Request):
        options = payload.model_dump(
            exclude={"model", "messages", "stream"},
            exclude_none=True,
        )

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
                        "cached_tokens": cached_tokens_from_usage(stream_usage),
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
                "cached_tokens": cached_tokens_from_usage(response.get("usage")),
                "status": 200,
                "response_body": response if runtime.log_response_body else None,
                "request_body": payload.model_dump()
                if runtime.log_request_body
                else None,
                **_request_telemetry(payload.messages, options or None),
            }
        )
        return JSONResponse(response)

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
