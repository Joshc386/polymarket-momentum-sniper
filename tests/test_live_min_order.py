"""Live exchange-minimum order guard (2026-06-12).

The 5-min BTC market reports min_order_size = 5 shares (observed live via
tools/live_smoke_test.py). The bot sizes in USDC -> shares = usdc / price,
so a sized order clears the minimum only when usdc >= 5 * price. On a ~$100
wallet the $1 floor produces sub-minimum orders that the CLOB rejects; in
the re-post loop a rejected post looks like a miss and spins. Decision
(user, 2026-06-12): SKIP such trades and log/count them — no over-betting,
sizing model stays honest. Applies to LIVE only; paper is unchanged.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.execution import LiveExecutionEngine
import core.execution_rounds as er


def _live_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "ROUNDS_LOG_PATH", tmp_path / "rounds.log")
    poly = MagicMock()
    poly.place_order = AsyncMock()
    db = MagicMock()
    db.conn = None
    return LiveExecutionEngine(db=db, poly_client=poly, bot_id="bot_t"), poly


def _kwargs(price, size_usdc):
    return dict(
        side="YES", price=price, size_usdc=size_usdc, market_id="m",
        market_slug="s", oracle_lag_signal=0.0, momentum_signal=0.0,
        liquidation_signal=0.0, combined_signal=0.0, estimated_prob_up=0.5,
        market_implied_prob=0.5, edge=0.05, time_remaining_secs=120.0,
        btc_price=70000.0, oracle_price=70000.0, oracle_open_price=70000.0,
        yes_token_id="0xYES", no_token_id="0xNO",
    )


def test_sub_minimum_order_is_skipped_not_posted(tmp_path, monkeypatch):
    eng, poly = _live_engine(tmp_path, monkeypatch)
    eng._execute_gtc = AsyncMock()
    eng._execute_fok = AsyncMock()
    # $1.00 at price 0.50 = 2 shares < 5 -> skip.
    with patch("core.execution.halt_active", return_value=False):
        out = asyncio.run(eng.execute_trade(**_kwargs(0.50, 1.00)))
    assert out is None
    poly.place_order.assert_not_called()
    eng._execute_gtc.assert_not_called()
    eng._execute_fok.assert_not_called()
    assert eng.below_min_skips == 1


def test_order_at_or_above_minimum_proceeds(tmp_path, monkeypatch):
    eng, poly = _live_engine(tmp_path, monkeypatch)
    eng._execute_gtc = AsyncMock(return_value=None)
    eng._execute_fok = AsyncMock(return_value=None)
    # $2.50 at price 0.50 = exactly 5 shares -> allowed.
    with patch("core.execution.halt_active", return_value=False):
        asyncio.run(eng.execute_trade(**_kwargs(0.50, 2.50)))
    eng._execute_gtc.assert_called_once()
    assert eng.below_min_skips == 0


def test_skip_is_recorded_in_telemetry(tmp_path, monkeypatch):
    eng, _ = _live_engine(tmp_path, monkeypatch)
    eng._execute_gtc = AsyncMock()
    eng._execute_fok = AsyncMock()
    with patch("core.execution.halt_active", return_value=False):
        asyncio.run(eng.execute_trade(**_kwargs(0.68, 1.00)))
    import json
    lines = (tmp_path / "rounds.log").read_text().strip().splitlines()
    rec = json.loads(lines[-1])
    assert rec["outcome"] == "below_min"
    assert rec["bot_id"] == "bot_t"
    assert rec["min_shares"] == 5
