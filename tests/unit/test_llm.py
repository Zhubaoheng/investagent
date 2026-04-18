"""Tests for poorcharlie.llm."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from poorcharlie.llm import LLMClient, _get_quota_gate, _quota_gates


def test_llm_client_default_model():
    client = LLMClient(client=MagicMock())
    assert client.model == "claude-sonnet-4-20250514"


def test_llm_client_custom_model():
    client = LLMClient(model="claude-haiku-4-5-20251001", client=MagicMock())
    assert client.model == "claude-haiku-4-5-20251001"


def test_llm_client_passes_base_url_and_api_key():
    """When no client is injected, base_url and api_key reach AsyncAnthropic."""
    with patch("poorcharlie.llm.anthropic.AsyncAnthropic") as mock_cls:
        with patch("poorcharlie.llm.httpx.AsyncClient"):
            LLMClient(
                model="MiniMax-M2.7",
                base_url="https://api.minimaxi.com/anthropic",
                api_key="test-key-123",
            )
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["base_url"] == "https://api.minimaxi.com/anthropic"
            assert call_kwargs["api_key"] == "test-key-123"
            assert "http_client" in call_kwargs


def test_llm_client_defaults_none_without_base_url():
    """Without base_url/api_key, AsyncAnthropic gets defaults."""
    with patch("poorcharlie.llm.anthropic.AsyncAnthropic") as mock_cls:
        LLMClient(model="claude-sonnet-4-20250514")
        mock_cls.assert_called_once_with()


def test_llm_client_injected_client_ignores_base_url():
    """When client is injected, base_url and api_key are not used."""
    mock_client = MagicMock()
    llm = LLMClient(
        model="MiniMax-M2.7",
        base_url="https://should.be.ignored",
        api_key="ignored-key",
        client=mock_client,
    )
    assert llm._client is mock_client


# ----------------------------------------------------------------------
# Quota gate: single-prober coordination under 2056 (MiniMax usage limit)
# ----------------------------------------------------------------------


@pytest.fixture
def reset_quota_gates():
    _quota_gates.clear()
    yield
    _quota_gates.clear()


def _make_2056_error() -> anthropic.APIStatusError:
    """Construct an APIStatusError matching MiniMax's 2056 quota shape."""
    response = MagicMock()
    response.status_code = 429
    return anthropic.APIStatusError(
        message="{'base_resp': {'status_code': 2056, 'status_msg': 'usage limit exceeded'}}",
        response=response,
        body=None,
    )


@pytest.mark.asyncio
async def test_quota_gate_serializes_concurrent_2056_callers(
    reset_quota_gates, monkeypatch,
):
    """When N concurrent calls all hit 2056, only the prober should sleep;
    others await the shared Event without making API calls or sleeping.
    """
    # Poll interval must be short for the test to finish quickly.
    monkeypatch.setenv("MINIMAX_QUOTA_POLL_SECONDS", "1")
    monkeypatch.setenv("MINIMAX_QUOTA_MAX_WAIT_SECONDS", "30")

    call_count = 0
    success_after = 3  # prober's N-th call succeeds

    async def fake_create(**_kwargs):
        nonlocal call_count
        call_count += 1
        # First two calls: 2056 for the prober; third: success.
        # Other callers should NOT reach here until quota is healthy.
        if call_count < success_after:
            raise _make_2056_error()
        # Build a minimal successful response.
        resp = MagicMock()
        resp.usage.input_tokens = 10
        resp.usage.output_tokens = 5
        resp.model = "MiniMax-M2.7"
        resp.stop_reason = "tool_use"
        return resp

    # Each coroutine triggers 2056 once on its OWN first real call; the
    # prober gets back more 2056s during probing; others wait on the gate.
    # After prober succeeds, everyone else retries — fake_create returns
    # success for them.
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=fake_create)

    llm = LLMClient(provider="minimax", client=mock_client)

    N_WAITERS = 20
    tools = [{"name": "t", "input_schema": {"type": "object", "properties": {}}}]
    results = await asyncio.gather(*[
        llm.create_message(system="s", messages=[{"role": "user", "content": "x"}], tools=tools)
        for _ in range(N_WAITERS)
    ])
    assert len(results) == N_WAITERS
    # Prober burns `success_after` calls; every other waiter makes exactly 1
    # call (succeeds on first retry after gate opens). So total should be
    # success_after + (N_WAITERS - 1).
    assert call_count == success_after + (N_WAITERS - 1), (
        f"Expected {success_after + (N_WAITERS - 1)} API calls "
        f"(no thundering herd), got {call_count}"
    )


@pytest.mark.asyncio
async def test_quota_gate_prober_release_on_max_wait(
    reset_quota_gates, monkeypatch,
):
    """If the prober exhausts max_wait, it must release the gate so other
    waiters can retry (not deadlock)."""
    monkeypatch.setenv("MINIMAX_QUOTA_POLL_SECONDS", "1")
    monkeypatch.setenv("MINIMAX_QUOTA_MAX_WAIT_SECONDS", "2")

    permanent_2056 = True

    async def fake_create(**_kwargs):
        if permanent_2056:
            raise _make_2056_error()
        resp = MagicMock()
        resp.usage.input_tokens = 1
        resp.usage.output_tokens = 1
        resp.model = "x"
        resp.stop_reason = "tool_use"
        return resp

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=fake_create)
    llm = LLMClient(provider="minimax", client=mock_client)

    tools = [{"name": "t", "input_schema": {"type": "object", "properties": {}}}]

    with pytest.raises(anthropic.APIStatusError):
        await llm.create_message(
            system="s", messages=[{"role": "user", "content": "x"}], tools=tools,
        )

    # After prober gives up, the gate must be released (healthy True again),
    # otherwise any subsequent caller deadlocks on the Event forever.
    gate = _get_quota_gate("minimax")
    assert gate.healthy.is_set()
    assert gate.prober_active is False
