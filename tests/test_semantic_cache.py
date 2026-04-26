from __future__ import annotations

import pytest

from mdrouter.runtime import MemoryResponseCache, RuntimeSettings


def _settings() -> RuntimeSettings:
    return RuntimeSettings(
        log_enabled=False,
        log_file="logs/router_requests.jsonl",
        log_request_body=False,
        log_response_body=False,
        log_rotate_mb=50,
        log_backups=5,
        prompt_cache_key_enabled=True,
        prompt_cache_retention=None,
        alibaba_explicit_cache=False,
        cache_enabled=True,
        sem_cache_enabled=True,
        cache_ttl_sec=300,
        cache_max_entries=1000,
        sem_cache_threshold=0.35,
        cache_backend="memory",
        redis_url="redis://127.0.0.1:6379/0",
        redis_prefix="mdrouter_cache",
        sem_cache_max_turns=3,
        sem_cache_include_assistant=False,
        sem_cache_single_turn_only=True,
    )


@pytest.mark.asyncio
async def test_semantic_cache_hits_on_paraphrase():
    cache = MemoryResponseCache(_settings())
    original_messages = [
        {"role": "system", "content": "You are an Ollama router assistant."},
        {"role": "user", "content": "What are your capabilities?"},
    ]
    response = {"model": "novita/deepseek-r1", "message": {"content": "capabilities"}}
    key = cache.make_exact_key(
        model_alias="novita/deepseek-r1",
        provider="novita",
        messages=original_messages,
        options=None,
    )
    await cache.store(
        exact_key=key,
        model_alias="novita/deepseek-r1",
        provider="novita",
        messages=original_messages,
        response=response,
    )

    paraphrase = [
        {"role": "system", "content": "You are an Ollama router assistant."},
        {"role": "user", "content": "What capabilities do you have?"},
    ]
    hit, meta = await cache.lookup(
        exact_key=cache.make_exact_key(
            model_alias="novita/deepseek-r1",
            provider="novita",
            messages=paraphrase,
            options=None,
        ),
        model_alias="novita/deepseek-r1",
        provider="novita",
        messages=paraphrase,
    )

    assert hit is not None
    assert meta["cache_hit"] == "semantic"
    assert meta["similarity"] >= 0.35


@pytest.mark.asyncio
async def test_semantic_cache_ignores_system_prompt_noise():
    cache = MemoryResponseCache(_settings())
    original_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Always be concise.",
        },
        {"role": "user", "content": "How do I configure Redis for the router?"},
    ]
    response = {"model": "novita/deepseek-r1", "message": {"content": "redis setup"}}
    key = cache.make_exact_key(
        model_alias="novita/deepseek-r1",
        provider="novita",
        messages=original_messages,
        options=None,
    )
    await cache.store(
        exact_key=key,
        model_alias="novita/deepseek-r1",
        provider="novita",
        messages=original_messages,
        response=response,
    )

    changed_question = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Always be concise.",
        },
        {"role": "user", "content": "List the files in the repository."},
    ]
    hit, meta = await cache.lookup(
        exact_key=cache.make_exact_key(
            model_alias="novita/deepseek-r1",
            provider="novita",
            messages=changed_question,
            options=None,
        ),
        model_alias="novita/deepseek-r1",
        provider="novita",
        messages=changed_question,
    )

    assert hit is None
    assert meta["cache_hit"] == "miss"
    assert meta["semantic_eligible"] is True
