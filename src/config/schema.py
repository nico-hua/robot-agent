"""Configuration models for the local agent application."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

ProviderType = Literal["ollama", "openai_compat"]


class ProviderConfig(BaseModel):
    """Provider settings loaded from project environment configuration."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: ProviderType
    api_key: str = ""
    api_base: str
    default_model: str = Field(
        validation_alias=AliasChoices("model", "default_model"),
    )
    default_max_tokens: int = Field(
        default=1024,
        gt=0,
        validation_alias=AliasChoices("max_tokens", "default_max_tokens"),
    )
    default_temperature: float = Field(
        default=0.7,
        ge=0,
        le=2,
        validation_alias=AliasChoices("temperature", "default_temperature"),
    )
    think: bool = False
    request_timeout_seconds: float = Field(default=60.0, gt=0)

    @field_validator("api_base", "default_model")
    @classmethod
    def _reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class ApiConfig(BaseModel):
    """Local HTTP API settings loaded from project environment configuration."""

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    request_timeout_seconds: float = Field(default=60.0, gt=0)

    @field_validator("host")
    @classmethod
    def _reject_blank_host(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("must not be blank")
        return normalized_value


class AgentConfig(BaseModel):
    """Root configuration used to assemble the local agent application."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderConfig
    api: ApiConfig = Field(default_factory=ApiConfig)
    workspace_path: Path
