from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class RuntimeSettings:
    log_enabled: bool
    log_file: str
    log_request_body: bool
    log_response_body: bool
    log_rotate_mb: int
    log_backups: int
    prompt_cache_key_enabled: bool
    prompt_cache_retention: str | None
    alibaba_explicit_cache: bool
    cache_enabled: bool
    sem_cache_enabled: bool
    cache_ttl_sec: int
    cache_max_entries: int
    sem_cache_threshold: float
    cache_backend: str
    redis_url: str
    redis_prefix: str
    sem_cache_max_turns: int
    sem_cache_include_assistant: bool
    sem_cache_single_turn_only: bool

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        retention = os.getenv("ROUTER_PROMPT_CACHE_RETENTION")
        if retention:
            retention = retention.strip()
        else:
            retention = None
        return cls(
            log_enabled=_env_bool("ROUTER_LOG_ENABLED", False),
            log_file=os.getenv("ROUTER_LOG_FILE", "logs/router_requests.jsonl"),
            log_request_body=_env_bool("ROUTER_LOG_REQUEST_BODY", False),
            log_response_body=_env_bool("ROUTER_LOG_RESPONSE_BODY", False),
            log_rotate_mb=_env_int("ROUTER_LOG_ROTATE_MB", 50),
            log_backups=_env_int("ROUTER_LOG_BACKUPS", 5),
            prompt_cache_key_enabled=_env_bool("ROUTER_PROMPT_CACHE_KEY_ENABLED", True),
            prompt_cache_retention=retention,
            alibaba_explicit_cache=_env_bool("ROUTER_ALIBABA_EXPLICIT_CACHE", False),
            cache_enabled=_env_bool("ROUTER_CACHE_ENABLED", True),
            sem_cache_enabled=_env_bool("ROUTER_SEM_CACHE_ENABLED", True),
            cache_ttl_sec=_env_int("ROUTER_CACHE_TTL_SEC", 300),
            cache_max_entries=_env_int("ROUTER_CACHE_MAX_ENTRIES", 1000),
            sem_cache_threshold=_env_float("ROUTER_SEM_CACHE_THRESHOLD", 0.93),
            cache_backend=os.getenv("ROUTER_CACHE_BACKEND", "memory").strip().lower(),
            redis_url=os.getenv("ROUTER_REDIS_URL", "redis://127.0.0.1:6379/0"),
            redis_prefix=os.getenv("ROUTER_REDIS_PREFIX", "mdrouter_cache"),
            sem_cache_max_turns=_env_int("ROUTER_SEM_CACHE_MAX_TURNS", 3),
            sem_cache_include_assistant=_env_bool(
                "ROUTER_SEM_CACHE_INCLUDE_ASSISTANT", False
            ),
            sem_cache_single_turn_only=_env_bool(
                "ROUTER_SEM_CACHE_SINGLE_TURN_ONLY", True
            ),
        )


class RequestLogger:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.logger: logging.Logger | None = None
        if settings.log_enabled:
            self.logger = logging.getLogger("mdrouter.requests")
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
            if not self.logger.handlers:
                log_path = Path(settings.log_file).expanduser().resolve()
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handler = RotatingFileHandler(
                    log_path,
                    maxBytes=settings.log_rotate_mb * 1024 * 1024,
                    backupCount=settings.log_backups,
                    encoding="utf-8",
                )
                handler.setFormatter(logging.Formatter("%(message)s"))
                self.logger.addHandler(handler)

    def write(self, payload: dict[str, Any]) -> None:
        if not self.logger:
            return
        event = dict(payload)
        event["ts"] = datetime.now(UTC).isoformat()
        self.logger.info(json.dumps(event, ensure_ascii=True))


@dataclass
class CacheEntry:
    key: str
    model_alias: str
    provider: str
    query_text: str
    response: dict[str, Any]
    expires_at: datetime


class ResponseCacheBackend(ABC):
    backend_name: str = "unknown"
    semantic_index_version: str = "v2"

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, list):
            content_text = json.dumps(content, sort_keys=True, ensure_ascii=True)
        else:
            content_text = str(content)
        content_text = content_text.replace("\r\n", "\n").replace("\r", "\n")
        content_text = " ".join(content_text.split())
        return content_text.strip()

    @classmethod
    def semantic_text(
        cls, messages: list[dict[str, Any]], *, settings: RuntimeSettings
    ) -> str:
        if settings.sem_cache_single_turn_only:
            user_messages = [msg for msg in messages if str(msg.get("role")) == "user"]
            assistant_or_tool = [
                msg for msg in messages if str(msg.get("role")) in {"assistant", "tool"}
            ]
            if len(user_messages) != 1 or assistant_or_tool:
                return ""

        selected: list[str] = []
        turn_count = 0
        for msg in reversed(messages):
            role = str(msg.get("role", "user"))
            if role == "tool":
                continue
            if role == "system":
                continue
            if role == "assistant" and not settings.sem_cache_include_assistant:
                continue
            if role not in {"user", "assistant"}:
                continue
            text = cls._normalize_content(msg.get("content", ""))
            if not text:
                continue
            selected.append(f"{role}:{text}")
            if role == "user":
                turn_count += 1
            if turn_count >= max(1, settings.sem_cache_max_turns):
                break
        return "\n".join(reversed(selected)).strip()

    @staticmethod
    def normalize_text(
        messages: list[dict[str, Any]], *, settings: RuntimeSettings
    ) -> str:
        chunks: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            if role not in {"system", "user", "assistant"}:
                continue
            content_text = ResponseCacheBackend._normalize_content(
                msg.get("content", "")
            )
            if content_text:
                chunks.append(f"{role}:{content_text}")
        return "\n".join(chunks).strip()

    @staticmethod
    def make_exact_key(
        *,
        model_alias: str,
        provider: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None,
    ) -> str:
        normalized_options = dict(options or {})
        normalized_options.pop("prompt_cache_key", None)
        blob = {
            "model_alias": model_alias,
            "provider": provider,
            "messages": messages,
            "options": normalized_options,
        }
        raw = json.dumps(blob, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def semantic_score(a: str, b: str) -> float:
        if not a or not b:
            return 0.0

        def word_tokens(text: str) -> set[str]:
            return {token for token in text.lower().split() if len(token) > 2}

        def char_ngrams(text: str, n: int = 3) -> set[str]:
            compact = f" {text.lower()} "
            if len(compact) < n:
                return {compact}
            return {compact[i : i + n] for i in range(len(compact) - n + 1)}

        word_a = word_tokens(a)
        word_b = word_tokens(b)
        if word_a and word_b:
            word_score = len(word_a & word_b) / len(word_a | word_b)
        else:
            word_score = 0.0

        gram_a = char_ngrams(a)
        gram_b = char_ngrams(b)
        if gram_a and gram_b:
            gram_score = len(gram_a & gram_b) / len(gram_a | gram_b)
        else:
            gram_score = 0.0

        seq = SequenceMatcher(a=a, b=b).ratio()
        return (0.45 * word_score) + (0.35 * gram_score) + (0.20 * seq)

    @abstractmethod
    async def lookup(
        self,
        *,
        exact_key: str,
        model_alias: str,
        provider: str,
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def store(
        self,
        *,
        exact_key: str,
        model_alias: str,
        provider: str,
        messages: list[dict[str, Any]],
        response: dict[str, Any],
    ) -> None:
        raise NotImplementedError


class MemoryResponseCache(ResponseCacheBackend):
    backend_name = "memory"

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self._entries: list[CacheEntry] = []
        self._exact_index: dict[str, CacheEntry] = {}
        self._lock = Lock()

    def _cleanup_locked(self, now: datetime) -> None:
        alive = [entry for entry in self._entries if entry.expires_at > now]
        self._entries = alive
        self._exact_index = {entry.key: entry for entry in alive}
        if len(self._entries) > self.settings.cache_max_entries:
            self._entries.sort(key=lambda e: e.expires_at)
            self._entries = self._entries[-self.settings.cache_max_entries :]
            self._exact_index = {entry.key: entry for entry in self._entries}

    async def lookup(
        self,
        *,
        exact_key: str,
        model_alias: str,
        provider: str,
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        now = datetime.now(UTC)
        query_text = self.semantic_text(messages, settings=self.settings)
        semantic_eligible = bool(query_text)
        with self._lock:
            self._cleanup_locked(now)
            exact = self._exact_index.get(exact_key)
            if exact and exact.expires_at > now:
                return copy.deepcopy(exact.response), {
                    "cache_hit": "exact",
                    "similarity": 1.0,
                    "semantic_eligible": semantic_eligible,
                }

            if not self.settings.sem_cache_enabled or not semantic_eligible:
                return None, {
                    "cache_hit": "miss",
                    "similarity": 0.0,
                    "semantic_eligible": semantic_eligible,
                }

            best: CacheEntry | None = None
            best_score = 0.0
            for entry in self._entries:
                if entry.model_alias != model_alias or entry.provider != provider:
                    continue
                score = self.semantic_score(query_text, entry.query_text)
                if score > best_score:
                    best_score = score
                    best = entry
            if best and best_score >= self.settings.sem_cache_threshold:
                return copy.deepcopy(best.response), {
                    "cache_hit": "semantic",
                    "similarity": best_score,
                    "semantic_eligible": semantic_eligible,
                }

            return None, {
                "cache_hit": "miss",
                "similarity": best_score,
                "semantic_eligible": semantic_eligible,
            }

    async def store(
        self,
        *,
        exact_key: str,
        model_alias: str,
        provider: str,
        messages: list[dict[str, Any]],
        response: dict[str, Any],
    ) -> None:
        if not self.settings.cache_enabled:
            return
        now = datetime.now(UTC)
        semantic_text = self.semantic_text(messages, settings=self.settings)
        entry = CacheEntry(
            key=exact_key,
            model_alias=model_alias,
            provider=provider,
            query_text=semantic_text,
            response=copy.deepcopy(response),
            expires_at=now + timedelta(seconds=self.settings.cache_ttl_sec),
        )
        with self._lock:
            self._entries.append(entry)
            self._exact_index[exact_key] = entry
            self._cleanup_locked(now)


class RedisResponseCache(ResponseCacheBackend):
    backend_name = "redis"

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        from redis.asyncio import Redis

        self.client: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
        self.prefix = settings.redis_prefix

    def _exact_redis_key(self, exact_key: str) -> str:
        return f"{self.prefix}:exact:{exact_key}"

    def _model_index_key(self, provider: str, model_alias: str) -> str:
        return f"{self.prefix}:sem:{self.semantic_index_version}:idx:{provider}:{model_alias}"

    async def lookup(
        self,
        *,
        exact_key: str,
        model_alias: str,
        provider: str,
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        exact_key_name = self._exact_redis_key(exact_key)
        query_text = self.semantic_text(messages, settings=self.settings)
        semantic_eligible = bool(query_text)
        raw = await self.client.get(exact_key_name)
        if raw:
            try:
                payload = json.loads(raw)
                return payload["response"], {
                    "cache_hit": "exact",
                    "similarity": 1.0,
                    "semantic_eligible": semantic_eligible,
                }
            except Exception:
                pass

        if not self.settings.sem_cache_enabled or not semantic_eligible:
            return None, {
                "cache_hit": "miss",
                "similarity": 0.0,
                "semantic_eligible": semantic_eligible,
            }

        idx_key = self._model_index_key(provider, model_alias)
        members = await self.client.smembers(idx_key)
        if not members:
            return None, {
                "cache_hit": "miss",
                "similarity": 0.0,
                "semantic_eligible": semantic_eligible,
            }

        best_payload: dict[str, Any] | None = None
        best_score = 0.0
        for key_name in list(members)[: self.settings.cache_max_entries]:
            blob = await self.client.get(key_name)
            if not blob:
                continue
            try:
                item = json.loads(blob)
            except Exception:
                continue
            candidate_query = str(item.get("query_text", ""))
            score = self.semantic_score(query_text, candidate_query)
            if score > best_score:
                best_score = score
                best_payload = item

        if best_payload and best_score >= self.settings.sem_cache_threshold:
            return best_payload["response"], {
                "cache_hit": "semantic",
                "similarity": best_score,
                "semantic_eligible": semantic_eligible,
            }
        return None, {
            "cache_hit": "miss",
            "similarity": best_score,
            "semantic_eligible": semantic_eligible,
        }

    async def store(
        self,
        *,
        exact_key: str,
        model_alias: str,
        provider: str,
        messages: list[dict[str, Any]],
        response: dict[str, Any],
    ) -> None:
        if not self.settings.cache_enabled:
            return

        exact_key_name = self._exact_redis_key(exact_key)
        semantic_text = self.semantic_text(messages, settings=self.settings)
        payload = {
            "model_alias": model_alias,
            "provider": provider,
            "query_text": semantic_text,
            "response": response,
            "stored_at": time.time(),
        }
        await self.client.set(
            exact_key_name,
            json.dumps(payload, ensure_ascii=True),
            ex=self.settings.cache_ttl_sec,
        )
        if semantic_text:
            idx_key = self._model_index_key(provider, model_alias)
            await self.client.sadd(idx_key, exact_key_name)
            await self.client.expire(
                idx_key, max(self.settings.cache_ttl_sec * 2, 3600)
            )


def build_response_cache(settings: RuntimeSettings) -> ResponseCacheBackend:
    if settings.cache_backend == "redis":
        try:
            return RedisResponseCache(settings)
        except Exception:
            return MemoryResponseCache(settings)
    return MemoryResponseCache(settings)
