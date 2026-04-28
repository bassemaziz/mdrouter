from __future__ import annotations

import pytest

from mdrouter.runtime import MemoryResponseCache, RedisResponseCache, RuntimeSettings


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
        sem_cache_latest_user_threshold=0.3,
        cache_backend="memory",
        redis_url="redis://127.0.0.1:6385/0",
        redis_prefix="mdrouter_cache",
        sem_cache_max_turns=3,
        sem_cache_include_assistant=False,
        sem_cache_single_turn_only=True,
        cache_profile="balanced",
    )


def _production_like_settings() -> RuntimeSettings:
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
        cache_ttl_sec=3600,
        cache_max_entries=1000,
        sem_cache_threshold=0.9,
        sem_cache_latest_user_threshold=0.3,
        cache_backend="memory",
        redis_url="redis://127.0.0.1:6385/0",
        redis_prefix="mdrouter_cache",
        sem_cache_max_turns=6,
        sem_cache_include_assistant=False,
        sem_cache_single_turn_only=False,
        cache_profile="production",
    )


def _permissive_semantic_settings() -> RuntimeSettings:
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
        sem_cache_threshold=0.2,
        sem_cache_latest_user_threshold=0.0,
        cache_backend="memory",
        redis_url="redis://127.0.0.1:6385/0",
        redis_prefix="mdrouter_cache",
        sem_cache_max_turns=3,
        sem_cache_include_assistant=False,
        sem_cache_single_turn_only=True,
        cache_profile="balanced",
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


@pytest.mark.asyncio
async def test_semantic_cache_requires_latest_user_turn_similarity():
    cache = MemoryResponseCache(_production_like_settings())
    stored_messages = [
        {"role": "user", "content": "Summarize the repo structure."},
        {"role": "user", "content": "List key env vars used by mdrouter."},
        {"role": "user", "content": "Explain semantic cache threshold tradeoffs."},
        {"role": "user", "content": "What is the default Redis URL fallback?"},
        {"role": "user", "content": "Show me the last request log event."},
        {"role": "user", "content": "Write me a 100 token story."},
    ]
    response = {"model": "mdrouter/auto", "message": {"content": "story"}}
    await cache.store(
        exact_key=cache.make_exact_key(
            model_alias="mdrouter/auto",
            provider="go",
            messages=stored_messages,
            options=None,
        ),
        model_alias="mdrouter/auto",
        provider="go",
        messages=stored_messages,
        response=response,
    )

    changed_latest_turn = [
        {"role": "user", "content": "Summarize the repo structure."},
        {"role": "user", "content": "List key env vars used by mdrouter."},
        {"role": "user", "content": "Explain semantic cache threshold tradeoffs."},
        {"role": "user", "content": "What is the default Redis URL fallback?"},
        {"role": "user", "content": "Show me the last request log event."},
        {
            "role": "user",
            "content": "Open the Linux calculator app and tell me which model did it.",
        },
    ]
    hit, meta = await cache.lookup(
        exact_key=cache.make_exact_key(
            model_alias="mdrouter/auto",
            provider="go",
            messages=changed_latest_turn,
            options=None,
        ),
        model_alias="mdrouter/auto",
        provider="go",
        messages=changed_latest_turn,
    )

    assert hit is None
    assert meta["cache_hit"] == "miss"


@pytest.mark.asyncio
async def test_semantic_cache_allows_latest_user_turn_near_match():
    cache = MemoryResponseCache(_production_like_settings())
    stored_messages = [
        {"role": "user", "content": "Summarize the repo structure."},
        {"role": "user", "content": "List key env vars used by mdrouter."},
        {"role": "user", "content": "Explain semantic cache threshold tradeoffs."},
        {"role": "user", "content": "What is the default Redis URL fallback?"},
        {"role": "user", "content": "Show me the last request log event."},
        {"role": "user", "content": "Write me a 100 token story."},
    ]
    response = {"model": "mdrouter/auto", "message": {"content": "story"}}
    await cache.store(
        exact_key=cache.make_exact_key(
            model_alias="mdrouter/auto",
            provider="go",
            messages=stored_messages,
            options=None,
        ),
        model_alias="mdrouter/auto",
        provider="go",
        messages=stored_messages,
        response=response,
    )

    paraphrased_latest_turn = [
        {"role": "user", "content": "Summarize the repo structure."},
        {"role": "user", "content": "List key env vars used by mdrouter."},
        {"role": "user", "content": "Explain semantic cache threshold tradeoffs."},
        {"role": "user", "content": "What is the default Redis URL fallback?"},
        {"role": "user", "content": "Show me the last request log event."},
        {"role": "user", "content": "Write a 100 token story."},
    ]
    hit, meta = await cache.lookup(
        exact_key=cache.make_exact_key(
            model_alias="mdrouter/auto",
            provider="go",
            messages=paraphrased_latest_turn,
            options=None,
        ),
        model_alias="mdrouter/auto",
        provider="go",
        messages=paraphrased_latest_turn,
    )

    assert hit is not None
    assert meta["cache_hit"] == "semantic"


@pytest.mark.asyncio
async def test_semantic_cache_blocks_hello_to_calc_replay_in_production_profile():
    cache = MemoryResponseCache(_production_like_settings())
    stored_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Hello please open calc app"},
    ]
    response = {
        "model": "go/glm-5.1",
        "message": {"content": "I cannot open applications directly."},
    }
    await cache.store(
        exact_key=cache.make_exact_key(
            model_alias="go/glm-5.1",
            provider="go",
            messages=stored_messages,
            options=None,
        ),
        model_alias="go/glm-5.1",
        provider="go",
        messages=stored_messages,
        response=response,
    )

    changed_latest_turn = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Hello please open calc app"},
        {"role": "assistant", "content": "I cannot open applications directly."},
        {"role": "user", "content": "Open calc app"},
    ]
    hit, meta = await cache.lookup(
        exact_key=cache.make_exact_key(
            model_alias="go/glm-5.1",
            provider="go",
            messages=changed_latest_turn,
            options=None,
        ),
        model_alias="go/glm-5.1",
        provider="go",
        messages=changed_latest_turn,
    )

    assert hit is None
    assert meta["cache_hit"] == "miss"


@pytest.mark.asyncio
async def test_semantic_cache_does_not_reuse_unrelated_single_turn_request():
    cache = MemoryResponseCache(_permissive_semantic_settings())
    hello_messages = [{"role": "user", "content": "Hello"}]
    response = {
        "model": "mdrouter/auto",
        "message": {"content": "Hello! I can help with mdrouter."},
    }

    await cache.store(
        exact_key=cache.make_exact_key(
            model_alias="mdrouter/auto",
            provider="go",
            messages=hello_messages,
            options=None,
        ),
        model_alias="mdrouter/auto",
        provider="go",
        messages=hello_messages,
        response=response,
    )

    calc_request = [{"role": "user", "content": "Please open calc app"}]
    hit, meta = await cache.lookup(
        exact_key=cache.make_exact_key(
            model_alias="mdrouter/auto",
            provider="go",
            messages=calc_request,
            options=None,
        ),
        model_alias="mdrouter/auto",
        provider="go",
        messages=calc_request,
    )

    assert hit is None
    assert meta["cache_hit"] == "miss"


class _FailingRedisClient:
    async def get(self, _key: str):
        raise RuntimeError("redis unavailable")

    async def smembers(self, _key: str):
        raise RuntimeError("redis unavailable")

    async def set(self, *_args, **_kwargs):
        raise RuntimeError("redis unavailable")

    async def sadd(self, *_args, **_kwargs):
        raise RuntimeError("redis unavailable")

    async def expire(self, *_args, **_kwargs):
        raise RuntimeError("redis unavailable")


@pytest.mark.asyncio
async def test_redis_cache_lookup_degrades_to_miss_when_backend_unavailable():
    cache = RedisResponseCache(_settings())
    cache.client = _FailingRedisClient()

    messages = [{"role": "user", "content": "hello"}]
    hit, meta = await cache.lookup(
        exact_key=cache.make_exact_key(
            model_alias="mdrouter/auto",
            provider="novita",
            messages=messages,
            options=None,
        ),
        model_alias="mdrouter/auto",
        provider="novita",
        messages=messages,
    )

    assert hit is None
    assert meta["cache_hit"] == "miss"


@pytest.mark.asyncio
async def test_redis_cache_store_ignores_backend_failure():
    cache = RedisResponseCache(_settings())
    cache.client = _FailingRedisClient()

    messages = [{"role": "user", "content": "hello"}]
    exact_key = cache.make_exact_key(
        model_alias="mdrouter/auto",
        provider="novita",
        messages=messages,
        options=None,
    )

    await cache.store(
        exact_key=exact_key,
        model_alias="mdrouter/auto",
        provider="novita",
        messages=messages,
        response={"model": "mdrouter/auto", "message": {"content": "ok"}},
    )
