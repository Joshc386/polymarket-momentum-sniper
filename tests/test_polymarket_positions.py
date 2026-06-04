"""Tests for PolymarketClient.get_positions() — position discovery.

The kill switch relies on get_positions() as its PRIMARY position-discovery
source ("account-level ground truth", ADR-0002 §3). A spike (2026-06-04)
proved the original implementation was a silent stub: it called
``ClobClient.get_positions()``, which does not exist in py-clob-client, so
the AttributeError was swallowed and it always returned ``[]``.

These tests pin the corrected behaviour: discovery hits Polymarket's public
Data API (``data-api.polymarket.com/positions?user=<funder>``), works WITHOUT
CLOB auth, and fails safe (returns ``[]`` on any error).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.polymarket_client import DATA_API_HOST, PolymarketClient


# --- Fixtures -------------------------------------------------------------

FUNDER = "0xabc0000000000000000000000000000000000def"
YES_TOKEN = "111111111111111111111111111111"
NO_TOKEN = "222222222222222222222222222222"
COND_ID = "0xcondition"


class _FakeConfig:
    """Minimal stand-in for core.config.Config."""

    polymarket_funder_address = FUNDER


def _client(funder: str = FUNDER) -> PolymarketClient:
    cfg = _FakeConfig()
    cfg.polymarket_funder_address = funder
    # Note: deliberately NOT calling connect() — discovery must not depend on
    # CLOB authentication. _authenticated stays False throughout.
    return PolymarketClient(cfg)


def _mock_async_client(*, json_data=None, get_exc=None, status_exc=None):
    """Build a MagicMock factory standing in for ``httpx.AsyncClient(...)``.

    Returns an object usable as ``async with httpx.AsyncClient() as http``.
    """
    resp = MagicMock()
    if status_exc is not None:
        resp.raise_for_status.side_effect = status_exc
    else:
        resp.raise_for_status.return_value = None
    resp.json.return_value = json_data

    http = MagicMock()
    get_mock = AsyncMock()
    if get_exc is not None:
        get_mock.side_effect = get_exc
    else:
        get_mock.return_value = resp
    http.get = get_mock

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=http)
    cm.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock(return_value=cm)
    factory._get_mock = get_mock  # exposed for assertions
    return factory


def _data_api_position(asset: str, size, cond: str = COND_ID) -> dict:
    """A realistic Data API /positions row (extra fields included on purpose)."""
    return {
        "proxyWallet": FUNDER,
        "asset": asset,
        "conditionId": cond,
        "size": size,
        "avgPrice": 0.52,
        "currentValue": 5.1,
        "title": "Bitcoin Up or Down",
        "outcome": "Up",
    }


# --- Tests ----------------------------------------------------------------

def test_parses_data_api_positions():
    factory = _mock_async_client(
        json_data=[
            _data_api_position(YES_TOKEN, 10.0),
            _data_api_position(NO_TOKEN, 3.5),
        ]
    )
    with patch("core.polymarket_client.httpx.AsyncClient", factory):
        out = asyncio.run(_client().get_positions())

    assert out == [
        {"token_id": YES_TOKEN, "size": 10.0, "condition_id": COND_ID},
        {"token_id": NO_TOKEN, "size": 3.5, "condition_id": COND_ID},
    ]
    # queried the Data API positions endpoint with the funder as ?user=
    args, kwargs = factory._get_mock.call_args
    assert args[0] == f"{DATA_API_HOST}/positions"
    assert kwargs["params"] == {"user": FUNDER}


def test_works_without_clob_auth():
    """Discovery must succeed even though connect()/auth never ran."""
    c = _client()
    assert c.is_authenticated is False
    factory = _mock_async_client(json_data=[_data_api_position(YES_TOKEN, 7.0)])
    with patch("core.polymarket_client.httpx.AsyncClient", factory):
        out = asyncio.run(c.get_positions())
    assert len(out) == 1 and out[0]["token_id"] == YES_TOKEN


def test_empty_positions_returns_empty_list():
    factory = _mock_async_client(json_data=[])
    with patch("core.polymarket_client.httpx.AsyncClient", factory):
        assert asyncio.run(_client().get_positions()) == []


def test_zero_size_positions_filtered_out():
    factory = _mock_async_client(
        json_data=[
            _data_api_position(YES_TOKEN, 0),       # closed/dust → excluded
            _data_api_position(NO_TOKEN, 4.0),
        ]
    )
    with patch("core.polymarket_client.httpx.AsyncClient", factory):
        out = asyncio.run(_client().get_positions())
    assert out == [{"token_id": NO_TOKEN, "size": 4.0, "condition_id": COND_ID}]


def test_no_funder_address_returns_empty_without_http():
    factory = _mock_async_client(json_data=[_data_api_position(YES_TOKEN, 9.0)])
    with patch("core.polymarket_client.httpx.AsyncClient", factory):
        out = asyncio.run(_client(funder="").get_positions())
    assert out == []
    factory.assert_not_called()  # never reached the network


def test_http_error_returns_empty():
    import httpx

    factory = _mock_async_client(get_exc=httpx.ConnectError("boom"))
    with patch("core.polymarket_client.httpx.AsyncClient", factory):
        assert asyncio.run(_client().get_positions()) == []


def test_http_status_error_returns_empty():
    import httpx

    factory = _mock_async_client(
        status_exc=httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
    )
    with patch("core.polymarket_client.httpx.AsyncClient", factory):
        assert asyncio.run(_client().get_positions()) == []


def test_malformed_response_returns_empty():
    # Data API returning a dict (e.g. an error envelope) instead of a list.
    factory = _mock_async_client(json_data={"error": "bad request"})
    with patch("core.polymarket_client.httpx.AsyncClient", factory):
        assert asyncio.run(_client().get_positions()) == []


def test_non_numeric_size_skipped():
    factory = _mock_async_client(
        json_data=[
            _data_api_position(YES_TOKEN, "not-a-number"),
            _data_api_position(NO_TOKEN, 2.0),
        ]
    )
    with patch("core.polymarket_client.httpx.AsyncClient", factory):
        out = asyncio.run(_client().get_positions())
    assert out == [{"token_id": NO_TOKEN, "size": 2.0, "condition_id": COND_ID}]
