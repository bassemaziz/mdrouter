from __future__ import annotations

from mdrouter.runtime import RuntimeSettings


def test_runtime_settings_builds_redis_url_from_host_port_db(monkeypatch):
    monkeypatch.delenv("ROUTER_REDIS_URL", raising=False)
    monkeypatch.setenv("ROUTER_REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("ROUTER_REDIS_PORT", "6385")
    monkeypatch.setenv("ROUTER_REDIS_DB", "4")

    settings = RuntimeSettings.from_env()

    assert settings.redis_url == "redis://127.0.0.1:6385/4"


def test_runtime_settings_production_profile_defaults(monkeypatch):
    monkeypatch.setenv("ROUTER_CACHE_PROFILE", "production")
    monkeypatch.delenv("ROUTER_CACHE_TTL_SEC", raising=False)
    monkeypatch.delenv("ROUTER_SEM_CACHE_THRESHOLD", raising=False)
    monkeypatch.delenv("ROUTER_SEM_CACHE_MAX_TURNS", raising=False)
    monkeypatch.delenv("ROUTER_SEM_CACHE_SINGLE_TURN_ONLY", raising=False)

    settings = RuntimeSettings.from_env()

    assert settings.cache_profile == "production"
    assert settings.cache_ttl_sec == 3600
    assert settings.sem_cache_threshold == 0.9
    assert settings.sem_cache_latest_user_threshold == 0.3
    assert settings.sem_cache_max_turns == 6
    assert settings.sem_cache_single_turn_only is False


def test_runtime_settings_explicit_ttl_overrides_profile(monkeypatch):
    monkeypatch.setenv("ROUTER_CACHE_PROFILE", "production")
    monkeypatch.setenv("ROUTER_CACHE_TTL_SEC", "180")

    settings = RuntimeSettings.from_env()

    assert settings.cache_ttl_sec == 180


def test_runtime_settings_force_off_disables_cache(monkeypatch):
    monkeypatch.setenv("ROUTER_CACHE_FORCE_OFF", "true")
    monkeypatch.setenv("ROUTER_CACHE_ENABLED", "true")
    monkeypatch.setenv("ROUTER_SEM_CACHE_ENABLED", "true")

    settings = RuntimeSettings.from_env()

    assert settings.cache_enabled is False
    assert settings.sem_cache_enabled is False


def test_runtime_settings_force_off_false_respects_flags(monkeypatch):
    monkeypatch.setenv("ROUTER_CACHE_FORCE_OFF", "false")
    monkeypatch.setenv("ROUTER_CACHE_ENABLED", "true")
    monkeypatch.setenv("ROUTER_SEM_CACHE_ENABLED", "true")

    settings = RuntimeSettings.from_env()

    assert settings.cache_enabled is True
    assert settings.sem_cache_enabled is True
