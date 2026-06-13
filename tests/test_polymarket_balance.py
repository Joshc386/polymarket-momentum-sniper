"""Regression test for PolymarketClient.get_balance (2026-06-12).

The live bankroll sync (LiveExecutionEngine.sync_bankroll -> get_balance)
feeds wallet-proportional sizing. get_balance was calling
``get_balance_allowance(asset_type="COLLATERAL")`` with a bare kwarg, but
the real py-clob-client (verified live via tools/verify_live_auth.py,
2026-06-12) only honours the params-object form
``get_balance_allowance(params=BalanceAllowanceParams(asset_type=...))``.
The bare form raised -> caught -> returned 0.0 -> sync skipped (its
``if balance > 0`` guard) -> live sizing silently fell back to the phantom
$100 default instead of the real wallet. These tests pin the call contract
so it can't regress, and validate the 6-decimal USDC conversion.
"""

import asyncio
from unittest.mock import MagicMock, patch

from core.polymarket_client import PolymarketClient


class _FakeParams:
    """Stand-in for py_clob_client.clob_types.BalanceAllowanceParams."""

    def __init__(self, asset_type=None, **kw):
        self.asset_type = asset_type


def _authed_client(fake_clob):
    c = PolymarketClient(MagicMock())
    c.client = fake_clob
    c._authenticated = True
    return c


def test_get_balance_passes_params_object_and_scales_6dp():
    captured = {}

    def _gba(params=None):
        # Mimic the real client: the bare asset_type kwarg is not accepted.
        captured["params"] = params
        return {"balance": "25308984", "allowances": {}}

    fake = MagicMock()
    fake.get_balance_allowance.side_effect = _gba
    c = _authed_client(fake)

    with patch("core.polymarket_client.BalanceAllowanceParams", _FakeParams):
        bal = asyncio.run(c.get_balance())

    assert bal == 25.308984                       # 25308984 / 1e6
    assert isinstance(captured["params"], _FakeParams)
    assert captured["params"].asset_type == "COLLATERAL"


def test_get_balance_returns_zero_when_unauthenticated():
    c = PolymarketClient(MagicMock())
    c._authenticated = False
    assert asyncio.run(c.get_balance()) == 0.0
