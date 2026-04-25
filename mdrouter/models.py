from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class OllamaMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str = ""


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaMessage] = Field(default_factory=list)
    stream: bool = True
    options: dict[str, Any] | None = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = True
    system: str | None = None
    options: dict[str, Any] | None = None


class ModelDetails(BaseModel):
    format: str = "gguf"
    family: str = "router"
    families: list[str] = Field(default_factory=lambda: ["router"])
    parameter_size: str = "unknown"
    quantization_level: str = "unknown"
    capabilities: list[str] = Field(default_factory=list)


class OllamaTagModel(BaseModel):
    name: str
    model: str
    modified_at: datetime
    size: int = 0
    digest: str
    details: ModelDetails = Field(default_factory=ModelDetails)
    capabilities: list[str] = Field(default_factory=list)
    model_info: dict[str, Any] = Field(default_factory=dict)
    supports: dict[str, bool] = Field(default_factory=dict)


class UpstreamProviderRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    stream: bool
    options: dict[str, Any] | None = None
