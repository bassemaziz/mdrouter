from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 11435
    log_level: str = "info"
    request_timeout: float = 90.0
    bind_localhost_only: bool = True


class ProviderConfig(BaseModel):
    type: str = "openai_compat"
    base_url: str
    api_key_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    wire_format: str = "openai_chat"
    timeout: float | None = None

    def resolve_headers(self, *, allow_missing_api_key: bool = True) -> dict[str, str]:
        result = dict(self.headers)
        if self.api_key_env:
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                if allow_missing_api_key:
                    return result
                raise ValueError(f"Missing environment variable '{self.api_key_env}'.")
            result.setdefault("Authorization", f"Bearer {api_key}")
        return result


class ModelConfig(BaseModel):
    provider: str
    upstream_model: str
    capabilities: list[str] = Field(default_factory=lambda: ["chat", "stream", "tools"])
    context_length: int = 32768
    extra: dict[str, Any] = Field(default_factory=dict)


class RoutingConfig(BaseModel):
    strict_provider_prefix: bool = True
    unknown_model_behavior: str = "error"


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    @model_validator(mode="after")
    def validate_references(self) -> "AppConfig":
        for alias, model_cfg in self.models.items():
            if "/" not in alias:
                raise ValueError(
                    f"Model alias '{alias}' must use provider-prefixed format 'provider/model'."
                )
            if model_cfg.provider not in self.providers:
                raise ValueError(
                    f"Model alias '{alias}' references missing provider '{model_cfg.provider}'."
                )
            if self.routing.strict_provider_prefix:
                expected_prefix = f"{model_cfg.provider}/"
                if not alias.startswith(expected_prefix):
                    raise ValueError(
                        f"Model alias '{alias}' must start with '{expected_prefix}'."
                    )
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        provider_files = payload.pop("provider_files", [])
        if provider_files is None:
            provider_files = []
        if not isinstance(provider_files, list):
            raise ValueError("'provider_files' must be a list of file paths.")

        enabled_providers = payload.pop("enabled_providers", None)
        enabled_set: set[str] | None = None
        if enabled_providers is not None:
            if not isinstance(enabled_providers, list):
                raise ValueError(
                    "'enabled_providers' must be a list of provider names."
                )
            enabled_set = {
                name
                for name in enabled_providers
                if isinstance(name, str) and name.strip()
            }

        merged_providers: dict[str, Any] = dict(payload.get("providers") or {})
        merged_models: dict[str, Any] = dict(payload.get("models") or {})

        config_dir = config_path.parent
        for rel_path in provider_files:
            if not isinstance(rel_path, str) or not rel_path.strip():
                raise ValueError(
                    "Each entry in 'provider_files' must be a non-empty string."
                )
            file_path = (config_dir / rel_path).expanduser().resolve()
            with file_path.open("r", encoding="utf-8") as f:
                section = json.load(f)

            file_providers = section.get("providers") or {}
            file_models = section.get("models") or {}
            if not isinstance(file_providers, dict):
                raise ValueError(f"File '{rel_path}' contains non-object 'providers'.")
            if not isinstance(file_models, dict):
                raise ValueError(f"File '{rel_path}' contains non-object 'models'.")

            duplicate_providers = set(merged_providers).intersection(file_providers)
            if duplicate_providers:
                name = sorted(duplicate_providers)[0]
                raise ValueError(
                    f"Duplicate provider '{name}' found while loading '{rel_path}'."
                )
            duplicate_models = set(merged_models).intersection(file_models)
            if duplicate_models:
                name = sorted(duplicate_models)[0]
                raise ValueError(
                    f"Duplicate model alias '{name}' found while loading '{rel_path}'."
                )

            merged_providers.update(file_providers)
            merged_models.update(file_models)

        if enabled_set is not None:
            merged_providers = {
                name: cfg
                for name, cfg in merged_providers.items()
                if name in enabled_set
            }
            merged_models = {
                alias: cfg
                for alias, cfg in merged_models.items()
                if isinstance(cfg, dict) and cfg.get("provider") in enabled_set
            }

        payload["providers"] = merged_providers
        payload["models"] = merged_models
        return cls.model_validate(payload)
