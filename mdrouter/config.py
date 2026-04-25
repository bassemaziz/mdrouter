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
        with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.model_validate(payload)
