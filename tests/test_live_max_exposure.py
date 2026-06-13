"""Independent max-exposure cap on the LIVE order path (CSO finding, 2026-06-13).

The per-trade size cap (4%*min(wallet,$200)) lives in strategy/sizing.py.
execute_trade must ALSO enforce a hard absolute ceiling so a sizing bug or bad
config can never risk more than max_trade_usdc on a single live trade —
defense-in-depth, independent of the sizing layer (CLAUDE.md Execution Agent:
"all orders must include maximum position size caps"). LIVE only.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.execution import LiveExecutionEngine
import core.execution_rounds as er


def _engine(tmp_path, monkeypatch, max_trade_usdc):
    monkeypatch.setattr(er, "ROUNDS_LOG_PATH", tmp_path / "rounds.log")
    poly = MagicMock()
    poly.place_order = AsyncMock()
    db = MagicMock()
    db.conn = None
    eng = LiveExecutionEngine(
        db=db, poly_client=poly, bot_id="bot_t", max_trade_usdc=max_trade_usdc
    )
    return eng, poly


def _kwargs(price, size_usdc):
    return dict(
        side="YES", price=price, size_usdc=size_usdc, market_id="m",
        market_slug="s", oracle_lag_signal=0.0, momentum_signal=0.0,
        liquidation_signal=0.0, combined_signal=0.0, estimated_prob_up=0.5,
        market_implied_prob=0.5, edge=0.05, time_remaining_secs=120.0,
        btc_price=70000.0, oracle_price=70000.0, oracle_open_price=70000.0,
        yes_token_id="0xYES", no_token_id="0xNO",
    )


def test_oversized_order_refused_before_placement(tmp_path, monkeypatch):
    eng, poly = _engine(tmp_path, monkeypatch, max_trade_usdc=10.0)
    eng._execute_gtc = AsyncMock()
    eng._execute_fok = AsyncMock()
    with patch("core.execution.halt_active", return_value=False):
        out = asyncio.run(eng.execute_trade(**_kwargs(0.50, 50.0)))  # $50 > $10 cap
    assert out is None
    eng._execute_gtc.assert_not_called()
    eng._execute_fok.assert_not_called()
    poly.place_order.assert_not_called()


def test_order_within_cap_proceeds(tmp_path, monkeypatch):
    eng, poly = _engine(tmp_path, monkeypatch, max_trade_usdc=10.0)
    eng._execute_gtc = AsyncMock(return_value=None)
    eng._execute_fok = AsyncMock(return_value=None)
    with patch("core.execution.halt_active", return_value=False):
        asyncio.run(eng.execute_trade(**_kwargs(0.50, 5.0)))  # $5 <= $10 cap
    eng._execute_gtc.assert_called_once()
