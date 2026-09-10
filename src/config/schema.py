"""Configuration models for the project's LLM provider."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

ProviderType = Literal["openai_compat", "anthropic_compat"]


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

    @field_validator("api_base", "default_model")
    @classmethod
    def _reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value
