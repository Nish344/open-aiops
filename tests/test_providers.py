"""Unit tests for the provider contract, schemas, and error hierarchy (INT-10)."""

import asyncio
import pytest
from pydantic import ValidationError

from open_aiops.governance.fallbacks import AllProvidersFailed, call_with_failover
from open_aiops.governance.router import ModelRouter, ProviderConfig, RouteRequest
from open_aiops.providers.base import (
    Provider,
    ProviderRequest,
    ProviderResponse,
    Usage,
)
from open_aiops.providers.errors import (
    AuthError,
    BadRequest,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimitError,
)


# ---------------------------------------------------------------------------
# Schema and Data Model Tests
# ---------------------------------------------------------------------------


def test_usage_initialization_and_total_tokens():
    """Verify Usage calculation and handling of unrecorded token counts."""
    empty_usage = Usage()
    assert empty_usage.input_tokens is None
    assert empty_usage.output_tokens is None
    assert empty_usage.total_tokens is None

    partial_usage = Usage(input_tokens=150)
    assert partial_usage.input_tokens == 150
    assert partial_usage.output_tokens is None
    assert partial_usage.total_tokens == 150

    full_usage = Usage(input_tokens=120, output_tokens=80)
    assert full_usage.input_tokens == 120
    assert full_usage.output_tokens == 80
    assert full_usage.total_tokens == 200


def test_provider_request_validation():
    """Verify ProviderRequest field validation and default behaviors."""
    req = ProviderRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello world"}],
        max_output_tokens=500,
        temperature=0.7,
    )
    assert req.model == "gpt-4o"
    assert len(req.messages) == 1
    assert req.max_output_tokens == 500
    assert req.temperature == 0.7
    assert req.extra_params == {}

    # Empty model identifier should fail validation
    with pytest.raises(ValidationError):
        ProviderRequest(model="")


def test_provider_response_attributes():
    """Verify ProviderResponse serialization and attribute accessibility."""
    usage = Usage(input_tokens=50, output_tokens=25)
    resp = ProviderResponse(
        text="Sample output text",
        model="claude-3-5-sonnet",
        usage=usage,
        latency=0.342,
        raw_response={"finish_reason": "stop"},
    )
    assert resp.text == "Sample output text"
    assert resp.model == "claude-3-5-sonnet"
    assert resp.usage.total_tokens == 75
    assert resp.latency == 0.342
    assert resp.raw_response == {"finish_reason": "stop"}


def test_provider_protocol_structural_subtyping():
    """Verify runtime Protocol check for compliant and non-compliant provider classes."""

    class CompliantProvider:
        async def complete(self, request: ProviderRequest) -> ProviderResponse:
            return ProviderResponse(text="ok", model=request.model)

    class NonCompliantProvider:
        def other_method(self) -> None:
            pass

    compliant = CompliantProvider()
    non_compliant = NonCompliantProvider()

    assert isinstance(compliant, Provider)
    assert not isinstance(non_compliant, Provider)


# ---------------------------------------------------------------------------
# Canonical Error Hierarchy Tests
# ---------------------------------------------------------------------------


def test_base_provider_error_attributes():
    """Verify ProviderError base attributes and formatting."""
    err = ProviderError("General failure", provider="test-provider")
    assert err.message == "General failure"
    assert err.provider == "test-provider"
    assert err.retryable is False
    assert err.failover_eligible is False
    assert "provider=test-provider" in str(err)


def test_rate_limit_error_flags_and_retry_after():
    """Verify RateLimitError carries retryable and failover_eligible flags and retry_after delays."""
    err = RateLimitError(retry_after_s=15.0, provider="groq-fast")
    assert isinstance(err, ProviderError)
    assert err.retryable is True
    assert err.failover_eligible is True
    assert err.status_code == 429
    assert err.retry_after_s == 15.0
    assert err.retry_after == 15.0
    assert err.provider == "groq-fast"

    # Backward compatibility with retry_after argument name
    err_compat = RateLimitError(retry_after=5.5)
    assert err_compat.retry_after == 5.5
    assert err_compat.retry_after_s == 5.5


def test_provider_timeout_flags():
    """Verify ProviderTimeout is retryable and failover-eligible."""
    err = ProviderTimeout(provider="ollama-local")
    assert isinstance(err, ProviderError)
    assert err.retryable is True
    assert err.failover_eligible is True
    assert err.status_code == 408
    assert err.provider == "ollama-local"


def test_provider_unavailable_5xx_flags():
    """Verify ProviderUnavailable is retryable and failover-eligible across 5xx codes."""
    err = ProviderUnavailable(status_code=502, provider="primary-llm")
    assert isinstance(err, ProviderError)
    assert err.retryable is True
    assert err.failover_eligible is True
    assert err.status_code == 502


def test_auth_error_flags():
    """Verify AuthError is neither retryable nor failover-eligible."""
    err = AuthError(provider="private-ai", status_code=401)
    assert isinstance(err, ProviderError)
    assert err.retryable is False
    assert err.failover_eligible is False
    assert err.status_code == 401


def test_bad_request_flags():
    """Verify BadRequest is neither retryable nor failover-eligible."""
    err = BadRequest(provider="private-ai", status_code=422)
    assert isinstance(err, ProviderError)
    assert err.retryable is False
    assert err.failover_eligible is False
    assert err.status_code == 422


# ---------------------------------------------------------------------------
# Resilience & Governance Failover Integration Tests
# ---------------------------------------------------------------------------


def test_failover_integration_with_canonical_errors():
    """Verify that failover-eligible provider errors trigger automated failovers."""
    p_primary = ProviderConfig(
        name="primary",
        model="gpt-4o",
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.03,
        expected_latency_ms=200.0,
    )
    p_backup = ProviderConfig(
        name="backup",
        model="claude-3-haiku",
        cost_per_1k_input_tokens=0.02,
        cost_per_1k_output_tokens=0.04,
        expected_latency_ms=250.0,
    )
    router = ModelRouter([p_primary, p_backup])
    req = RouteRequest(input_tokens=100, max_output_tokens=100)

    # 1. RateLimitError triggers failover
    calls_1 = []

    async def caller_rate_limit(p: ProviderConfig):
        calls_1.append(p.name)
        if p.name == "primary":
            raise RateLimitError(retry_after_s=5.0)
        return "success-backup"

    res_1 = asyncio.run(call_with_failover(router, req, caller_rate_limit))
    assert res_1 == "success-backup"
    assert calls_1 == ["primary", "backup"]

    # 2. ProviderUnavailable triggers failover
    calls_2 = []

    async def caller_unavailable(p: ProviderConfig):
        calls_2.append(p.name)
        if p.name == "primary":
            raise ProviderUnavailable()
        return "success-backup"

    res_2 = asyncio.run(call_with_failover(router, req, caller_unavailable))
    assert res_2 == "success-backup"
    assert calls_2 == ["primary", "backup"]

    # 3. ProviderTimeout triggers failover
    calls_3 = []

    async def caller_timeout(p: ProviderConfig):
        calls_3.append(p.name)
        if p.name == "primary":
            raise ProviderTimeout()
        return "success-backup"

    res_3 = asyncio.run(call_with_failover(router, req, caller_timeout))
    assert res_3 == "success-backup"
    assert calls_3 == ["primary", "backup"]

    # 4. AuthError immediately halts and raises without failover
    calls_4 = []

    async def caller_auth(p: ProviderConfig):
        calls_4.append(p.name)
        if p.name == "primary":
            raise AuthError("Invalid API key")
        return "never-reached"

    with pytest.raises(AuthError):
        asyncio.run(call_with_failover(router, req, caller_auth))
    assert calls_4 == ["primary"]

    # 5. BadRequest immediately halts and raises without failover
    calls_5 = []

    async def caller_bad_request(p: ProviderConfig):
        calls_5.append(p.name)
        if p.name == "primary":
            raise BadRequest("Invalid prompt parameters")
        return "never-reached"

    with pytest.raises(BadRequest):
        asyncio.run(call_with_failover(router, req, caller_bad_request))
    assert calls_5 == ["primary"]
