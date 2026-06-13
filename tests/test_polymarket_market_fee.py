"""Tests for PolymarketClient.get_market_fee (CLOB V2, 2026-06-13).

V2 sets fees by the protocol at match time and exposes them per token via
get_fee_rate_bps(token_id) (basis points) and get_fee_exponent(token_id).
get_market_fee returns (rate_fraction, exponent) for the live PnL resolver;
the runner fetches it once per window. Returns None on failure so the caller
keeps its current/default fee rather than mis-pricing.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from core.polymarket_client import PolymarketClient


def _authed(fake):
    c = PolymarketClient(MagicMock())
    c.client = fake
    c._authenticated = True
    return c


def test_get_market_fee_returns_rate_fraction_and_exponent():
    fake = MagicMock()
    fake.get_fee_rate_bps.return_value = 1000   # 10% in bps
    fake.get_fee_exponent.return_value = 1.0
    c = _authed(fake)
    result = asyncio.run(c.get_market_fee("0xTOK"))
    assert result == pytest.approx((0.10, 1.0))
    fake.get_fee_rate_bps.assert_called_once_with("0xTOK")
    fake.get_fee_exponent.assert_called_once_with("0xTOK")


def test_get_market_fee_returns_none_on_error():
    fake = MagicMock()
    fake.get_fee_rate_bps.side_effect = RuntimeError("boom")
    c = _authed(fake)
    assert asyncio.run(c.get_market_fee("0xTOK")) is None


def test_get_market_fee_returns_none_when_unauthenticated():
    c = PolymarketClient(MagicMock())
    c._authenticated = False
    assert asyncio.run(c.get_market_fee("0xTOK")) is None
