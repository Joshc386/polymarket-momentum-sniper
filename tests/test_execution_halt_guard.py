"""Tests for the pre-order HALT guard on LiveExecutionEngine (kill switch §5).

Once the kill switch writes the sticky HALT flag, the bot must place NO new
orders — the kill switch owns cancellation/flattening, and a recovering bot
must not race it. The guard sits at the top of every real order path
(``execute_trade`` for entries; ``close_position_early`` for the bot's own
exit) and short-circuits before any ``place_order`` call.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.execution import LiveExecutionEngine


def _engine():
    db = MagicMock()
    poly = MagicMock()
    poly.place_order = AsyncMock()
    return LiveExecutionEngine(db=db, poly_client=poly), poly


def _entry_kwargs():
    return dict(
        # $5 at 0.50 = 10 shares — clears the 5-share exchange minimum so the
        # not-halted path reaches the order machinery (the guard under test).
        side="YES", price=0.5, size_usdc=5.0, market_id="m", market_slug="s",
        oracle_lag_signal=0.0, momentum_signal=0.0, liquidation_signal=0.0,
        combined_signal=0.0, estimated_prob_up=0.5, market_implied_prob=0.5,
        edge=0.05, time_remaining_secs=120.0, btc_price=70000.0,
        oracle_price=70000.0, oracle_open_price=70000.0,
        yes_token_id="0xYES", no_token_id="0xNO",
    )


def test_execute_trade_blocked_when_halted():
    eng, poly = _engine()
    eng._execute_gtc = AsyncMock()
    eng._execute_fok = AsyncMock()
    with patch("core.execution.halt_active", return_value=True):
        out = asyncio.run(eng.execute_trade(**_entry_kwargs()))
    assert out is None
    poly.place_order.assert_not_called()
    eng._execute_gtc.assert_not_called()
    eng._execute_fok.assert_not_called()


def test_execute_trade_proceeds_when_not_halted():
    eng, poly = _engine()
    # Isolate the guard from the fill machinery: stub the order path.
    eng._execute_gtc = AsyncMock(return_value=None)
    eng._execute_fok = AsyncMock(return_value=None)
    with patch("core.execution.halt_active", return_value=False):
        asyncio.run(eng.execute_trade(**_entry_kwargs()))
    # time_remaining 120s > 60s → GTC path attempted (guard did not block).
    eng._execute_gtc.assert_called_once()
    eng._execute_fok.assert_not_called()


def test_close_position_early_blocked_when_halted():
    eng, poly = _engine()
    sentinel = MagicMock()  # stand-in pending trade
    eng.pending_trade = sentinel
    eng.pending_token_id = "0xTOK"
    with patch("core.execution.halt_active", return_value=True):
        out = asyncio.run(eng.close_position_early(exit_price=0.6, reason="stop"))
    assert out is None
    # Deferred to the kill switch — the bot did not touch its position,
    # and placed no SELL.
    assert eng.pending_trade is sentinel
    poly.place_order.assert_not_called()
