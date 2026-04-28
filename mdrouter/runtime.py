from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any


_runtime_logger = logging.getLogger("mdrouter.runtime")


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
    sem_cache_latest_user_threshold: float
    cache_backend: str
    redis_url: str
    redis_prefix: str
    sem_cache_max_turns: int
    sem_cache_include_assistant: bool
    sem_cache_single_turn_only: bool
    cache_profile: str

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        retention = os.getenv("ROUTER_PROMPT_CACHE_RETENTION")
        if retention:
            retention = retention.strip()
        else:
            retention = None

        cache_profile = os.getenv("ROUTER_CACHE_PROFILE", "balanced").strip().lower()
        if cache_profile in {"prod", "production", "coding", "coding_prod"}:
            default_cache_ttl_sec = 3600
            default_sem_cache_threshold = 0.9
            default_sem_cache_latest_user_threshold = 0.3
            default_sem_cache_max_turns = 6
            default_sem_cache_single_turn_only = False
            default_sem_cache_include_assistant = False
        else:
            default_cache_ttl_sec = 300
            default_sem_cache_threshold = 0.93
            default_sem_cache_latest_user_threshold = 0.3
            default_sem_cache_max_turns = 3
            default_sem_cache_single_turn_only = True
            default_sem_cache_include_assistant = False

        redis_url = os.getenv("ROUTER_REDIS_URL")
        if redis_url:
            redis_url = redis_url.strip()
        else:
            redis_host = os.getenv("ROUTER_REDIS_HOST", "127.0.0.1").strip()
            redis_port = _env_int("ROUTER_REDIS_PORT", 6385)
            redis_db = _env_int("ROUTER_REDIS_DB", 0)
            redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

        # Temporary safety switch: disable router-side response caching by default
        # until semantic reuse behavior is re-validated in production traffic.
        cache_force_off = _env_bool("ROUTER_CACHE_FORCE_OFF", True)
        cache_enabled = _env_bool("ROUTER_CACHE_ENABLED", True)
        sem_cache_enabled = _env_bool("ROUTER_SEM_CACHE_ENABLED", True)
        if cache_force_off:
            cache_enabled = False
            sem_cache_enabled = False

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
            cache_enabled=cache_enabled,
            sem_cache_enabled=sem_cache_enabled,
            cache_ttl_sec=_env_int("ROUTER_CACHE_TTL_SEC", default_cache_ttl_sec),
            cache_max_entries=_env_int("ROUTER_CACHE_MAX_ENTRIES", 1000),
            sem_cache_threshold=_env_float(
                "ROUTER_SEM_CACHE_THRESHOLD", default_sem_cache_threshold
            ),
            sem_cache_latest_user_threshold=_env_float(
                "ROUTER_SEM_CACHE_LATEST_USER_THRESHOLD",
                default_sem_cache_latest_user_threshold,
            ),
            cache_backend=os.getenv("ROUTER_CACHE_BACKEND", "memory").strip().lower(),
            redis_url=redis_url,
            redis_prefix=os.getenv("ROUTER_REDIS_PREFIX", "mdrouter_cache"),
            sem_cache_max_turns=_env_int(
                "ROUTER_SEM_CACHE_MAX_TURNS", default_sem_cache_max_turns
            ),
            sem_cache_include_assistant=_env_bool(
                "ROUTER_SEM_CACHE_INCLUDE_ASSISTANT",
                default_sem_cache_include_assistant,
            ),
            sem_cache_single_turn_only=_env_bool(
                "ROUTER_SEM_CACHE_SINGLE_TURN_ONLY",
                default_sem_cache_single_turn_only,
            ),
            cache_profile=cache_profile,
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
    latest_user_text: str
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

    @classmethod
    def latest_user_text(cls, messages: list[dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if str(msg.get("role", "user")) != "user":
                continue
            text = cls._normalize_content(msg.get("content", ""))
            if text:
                return text
        return ""

    @staticmethod
    def latest_user_from_semantic_query(query_text: str) -> str:
        if not query_text:
            return ""
        for line in reversed(query_text.splitlines()):
            line = line.strip()
            if line.startswith("user:"):
                return line[5:].strip()
        return ""

    @staticmethod
    def _effective_latest_user_threshold(settings: RuntimeSettings) -> float:
        # In stricter profiles, require stronger latest-user intent alignment to
        # avoid stale semantic replays in growing conversations.
        return max(
            settings.sem_cache_latest_user_threshold,
            settings.sem_cache_threshold * 0.75,
        )

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

        def semantic_base(text: str) -> str:
            # Drop role labels so structural prompt formatting doesn't inflate matches.
            lines: list[str] = []
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if line.startswith("user:"):
                    line = line[5:].strip()
                elif line.startswith("assistant:"):
                    line = line[10:].strip()
                elif line.startswith("system:"):
                    line = line[7:].strip()
                if line:
                    lines.append(line)
            return "\n".join(lines) if lines else text.strip()

        base_a = semantic_base(a)
        base_b = semantic_base(b)

        def word_tokens(text: str) -> set[str]:
            return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}

        def char_ngrams(text: str, n: int = 3) -> set[str]:
            compact = f" {text.lower()} "
            if len(compact) < n:
                return {compact}
            return {compact[i : i + n] for i in range(len(compact) - n + 1)}

        word_a = word_tokens(base_a)
        word_b = word_tokens(base_b)
        if word_a and word_b:
            word_score = len(word_a & word_b) / len(word_a | word_b)
        else:
            word_score = 0.0

        gram_a = char_ngrams(base_a)
        gram_b = char_ngrams(base_b)
        if gram_a and gram_b:
            gram_score = len(gram_a & gram_b) / len(gram_a | gram_b)
        else:
            gram_score = 0.0

        seq = SequenceMatcher(a=base_a, b=base_b).ratio()
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
        query_latest_user = self.latest_user_text(messages)
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
            latest_user_threshold = self._effective_latest_user_threshold(
                self.settings
            )
            for entry in self._entries:
                if entry.model_alias != model_alias or entry.provider != provider:
                    continue
                if query_latest_user and entry.latest_user_text:
                    latest_user_score = self.semantic_score(
                        query_latest_user, entry.latest_user_text
                    )
                    if latest_user_score < latest_user_threshold:
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
        latest_user_text = self.latest_user_text(messages)
        entry = CacheEntry(
            key=exact_key,
            model_alias=model_alias,
            provider=provider,
            query_text=semantic_text,
            latest_user_text=latest_user_text,
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
        query_latest_user = self.latest_user_text(messages)
        semantic_eligible = bool(query_text)
        try:
            raw = await self.client.get(exact_key_name)
        except Exception as exc:
            _runtime_logger.warning(
                "cache_lookup_degraded",
                extra={
                    "event": "cache_lookup_degraded",
                    "backend": self.backend_name,
                    "provider": provider,
                    "model_alias": model_alias,
                    "error": str(exc),
                },
            )
            return None, {
                "cache_hit": "miss",
                "similarity": 0.0,
                "semantic_eligible": semantic_eligible,
            }
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
        try:
            members = await self.client.smembers(idx_key)
        except Exception as exc:
            _runtime_logger.warning(
                "cache_semantic_lookup_degraded",
                extra={
                    "event": "cache_semantic_lookup_degraded",
                    "backend": self.backend_name,
                    "provider": provider,
                    "model_alias": model_alias,
                    "error": str(exc),
                },
            )
            return None, {
                "cache_hit": "miss",
                "similarity": 0.0,
                "semantic_eligible": semantic_eligible,
            }
        if not members:
            return None, {
                "cache_hit": "miss",
                "similarity": 0.0,
                "semantic_eligible": semantic_eligible,
            }

        best_payload: dict[str, Any] | None = None
        best_score = 0.0
        latest_user_threshold = self._effective_latest_user_threshold(self.settings)
        for key_name in list(members)[: self.settings.cache_max_entries]:
            try:
                blob = await self.client.get(key_name)
            except Exception:
                continue
            if not blob:
                continue
            try:
                item = json.loads(blob)
            except Exception:
                continue
            candidate_query = str(item.get("query_text", ""))
            candidate_latest_user = str(item.get("latest_user_text", "")).strip()
            if not candidate_latest_user:
                candidate_latest_user = self.latest_user_from_semantic_query(
                    candidate_query
                )
            if query_latest_user and candidate_latest_user:
                latest_user_score = self.semantic_score(
                    query_latest_user, candidate_latest_user
                )
                if latest_user_score < latest_user_threshold:
                    continue
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
        latest_user_text = self.latest_user_text(messages)
        payload = {
            "model_alias": model_alias,
            "provider": provider,
            "query_text": semantic_text,
            "latest_user_text": latest_user_text,
            "response": response,
            "stored_at": time.time(),
        }
        try:
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
        except Exception as exc:
            _runtime_logger.warning(
                "cache_store_degraded",
                extra={
                    "event": "cache_store_degraded",
                    "backend": self.backend_name,
                    "provider": provider,
                    "model_alias": model_alias,
                    "error": str(exc),
                },
            )


def build_response_cache(settings: RuntimeSettings) -> ResponseCacheBackend:
    if settings.cache_backend == "redis":
        try:
            return RedisResponseCache(settings)
        except Exception:
            return MemoryResponseCache(settings)
    return MemoryResponseCache(settings)
