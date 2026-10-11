"""Core provider contract and protocol definitions for Open-AIOps.

Defines standard request/response data types and the async Provider protocol
conforming to architecture specifications (Section 3.3).
"""

from typing import Any, Protocol, runtime_checkable
from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Token usage metrics reported by a model provider.
    
    None indicates the provider did not report token counts.
    """

    input_tokens: int | None = Field(
        default=None, ge=0, description="Number of input/prompt tokens consumed"
    )
    output_tokens: int | None = Field(
        default=None, ge=0, description="Number of output/completion tokens produced"
    )

    @property
    def total_tokens(self) -> int | None:
        """Calculate total tokens if at least one token metric is reported."""
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


class ProviderRequest(BaseModel):
    """Standardized request structure passed to any provider adapter."""

    model: str = Field(..., min_length=1, description="Model identifier to invoke")
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation messages in role/content dictionary format",
    )
    max_output_tokens: int | None = Field(
        default=None, gt=0, description="Maximum completion tokens requested"
    )
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional provider-specific passthrough parameters",
    )


class ProviderResponse(BaseModel):
    """Standardized completion response returned by a provider adapter."""

    text: str = Field(..., description="Generated text completion")
    model: str = Field(..., min_length=1, description="Model identifier that served the response")
    usage: Usage | None = Field(
        default=None, description="Reported token consumption metrics"
    )
    latency: float | None = Field(
        default=None, ge=0.0, description="Total execution latency in seconds"
    )
    raw_response: dict[str, Any] | None = Field(
        default=None, description="Raw provider payload metadata"
    )


@runtime_checkable
class Provider(Protocol):
    """Abstract asynchronous protocol that all LLM provider adapters must satisfy."""

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Execute a text completion request asynchronously.

        Args:
            request: Standardized request payload.

        Returns:
            ProviderResponse containing output text and usage metrics.

        Raises:
            ProviderError or one of its canonical subclasses upon failure.
        """
        ...
