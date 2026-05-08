from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MAX_TOKENS = 4096


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
    # Thinking/reasoning mode (DeepSeek, etc.)
    thinking: dict[str, Any] | None = None
    reasoning_effort: str | None = None


# ---------- Anthropic-compatible client models ----------


class AnthropicContentBlock(BaseModel):
    """A content block in an Anthropic Messages request/response."""
    model_config = ConfigDict(extra="allow")

    type: str  # text, image, tool_use, tool_result, thinking
    text: str | None = None
    source: dict[str, Any] | None = None  # for image
    id: str | None = None  # for tool_use
    name: str | None = None  # for tool_use
    input: dict[str, Any] | None = None  # for tool_use
    tool_use_id: str | None = None  # for tool_result
    content: str | list[dict[str, Any]] | None = None  # for tool_result (can be string for errors)
    thinking: str | None = None  # for thinking blocks
    signature: str | None = None  # for thinking signature
    is_error: bool | None = None  # for error blocks
    cache_control: dict[str, Any] | None = None  # for cache control (object, e.g. {"type":"ephemeral"})


class AnthropicMessage(BaseModel):
    """A message in an Anthropic Messages request."""
    model_config = ConfigDict(extra="allow")

    role: str  # user, assistant
    content: list[AnthropicContentBlock] | str  # list first to avoid union ordering issues


class AnthropicToolSpec(BaseModel):
    """An Anthropic tool definition."""
    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class AnthropicChatRequest(BaseModel):
    """Anthropic-compatible /v1/messages request body."""
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[AnthropicMessage]
    system: str | list[dict[str, Any]] | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    thinking: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    stop_sequences: list[str] | None = None
    tools: list[AnthropicToolSpec] | None = None
    tool_choice: dict[str, Any] | str | None = None
