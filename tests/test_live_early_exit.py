"""Tests for the REAL live early exit (ADR-0003 phase 2).

The live ``close_position_early`` was a simulation stub -- it marked the
trade closed without placing a SELL, which would have made the $90 BTC
distance stop fake on live capital. The real contract (grilled 2026-06-12):
best-effort flatten -- aggressive marketable SELL (cross the best bid one
tick), partials accepted, bounded price-stepping retries, backstopped by
resolution. The ledger records only actual fills at actual prices.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.execution import LiveExecutionEngine


def _engine_with_position(**kwargs):
    poly = MagicMock()
    poly.place_order = AsyncMock()
    poly.get_order_status = AsyncMock()
    poly.cancel_order = AsyncMock(return_value=True)
    poly.get_orderbook = MagicMock(return_value={
        "bids": [{"price": "0.55", "size": "50"}],
        "asks": [{"price": "0.57", "size": "50"}],
    })
    db = MagicMock()
    db.conn = None  # skip SQL in unit tests
    eng = LiveExecutionEngine(db=db, poly_client=poly, bot_id="bot_test", **kwargs)

    # Open a position the fast way: engine state as execute_trade leaves it.
    poly.place_order.return_value = {"orderID": "ord-entry"}
    poly.get_order_status.return_value = {
        "status": "MATCHED", "price": 0.50, "size_matched": 4.0,
    }
    with patch("core.execution.asyncio.sleep", new=AsyncMock()), \
         patch("core.execution.halt_active", return_value=False), \
         patch("core.execution._log_to_db", return_value=1):
        trade = asyncio.run(eng.execute_trade(
            side="YES", price=0.50, size_usdc=2.0, market_id="m",
            market_slug="s", oracle_lag_signal=0.0, momentum_signal=0.0,
            liquidation_signal=0.0, combined_signal=0.0,
            estimated_prob_up=0.5, market_implied_prob=0.5, edge=0.05,
            time_remaining_secs=120.0, btc_price=70000.0,
            oracle_price=70000.0, oracle_open_price=70000.0,
            yes_token_id="0xYES", no_token_id="0xNO",
        ))
    assert trade is not None and eng.pending_trade is trade
    bankroll_after_entry = eng.bankroll
    return eng, poly, trade, bankroll_after_entry


def test_full_exit_places_marketable_sell_and_books_actual_fill():
    """The stop's teeth: a real SELL crossing the bid by one tick; the
    ledger books the ACTUAL fill price, not the requested one."""
    eng, poly, trade, bankroll0 = _engine_with_position()
    # Exit SELL: filled in full at 0.54 (one tick under the 0.55 bid).
    poly.place_order.return_value = {"orderID": "ord-exit"}
    poly.get_order_status.return_value = {
        "status": "MATCHED", "price": 0.54, "size_matched": 4.0,
    }

    with patch("core.execution.asyncio.sleep", new=AsyncMock()), \
         patch("core.execution.halt_active", return_value=False):
        out = asyncio.run(
            eng.close_position_early(exit_price=0.55, reason="btc_stop_$92")
        )

    sell_call = poly.place_order.await_args_list[-1].kwargs
    assert sell_call["side"] == "SELL"
    assert sell_call["token_id"] == "0xYES"
    assert abs(sell_call["price"] - 0.54) < 1e-9   # bid 0.55 crossed one tick
    assert sell_call["size"] == 4.0

    assert out is not None
    assert out.resolution == "EARLY_EXIT:btc_stop_$92"
    # PnL from the ACTUAL 0.54 fill: (0.54-0.50)*4 - entry fee on 4 shares.
    from strategy.feature_snapshot import fee_per_share
    expected_pnl = (0.54 - 0.50) * 4.0 - fee_per_share(0.50) * 4.0
    assert abs(out.pnl - expected_pnl) < 1e-9
    assert eng.pending_trade is None
    # Bankroll: entry capital returned + pnl.
    assert abs(eng.bankroll - (bankroll0 + trade.size_usdc + out.pnl)) < 1e-9


def test_unfilled_exit_rides_to_resolution_with_no_ledger_change():
    """The resolution backstop: nothing sold -> the position stays pending,
    the ledger untouched, descending retry ladder exhausted."""
    eng, poly, trade, bankroll0 = _engine_with_position()
    poly.place_order.return_value = {"orderID": "ord-exit"}
    poly.get_order_status.return_value = {"status": "LIVE", "size_matched": 0}

    with patch("core.execution.asyncio.sleep", new=AsyncMock()), \
         patch("core.execution.halt_active", return_value=False):
        out = asyncio.run(
            eng.close_position_early(exit_price=0.55, reason="btc_stop_$90")
        )

    assert out is None
    assert eng.pending_trade is trade          # still pending
    assert eng.bankroll == bankroll0           # untouched
    # 3 attempts at descending prices: 0.54, 0.53, 0.52 (bid 0.55).
    sells = [c.kwargs for c in poly.place_order.await_args_list
             if c.kwargs.get("side") == "SELL"]
    assert [round(s["price"], 2) for s in sells] == [0.54, 0.53, 0.52]
    assert poly.cancel_order.await_count == 3  # every remainder cancelled


def test_partial_exit_splits_the_record(monkeypatch):
    """Sold 1.5 of 4: the sold portion books NOW at the actual price; the
    remaining 2.5 shares stay pending with a proportional cost basis."""
    eng, poly, trade, bankroll0 = _engine_with_position()
    poly.place_order.return_value = {"orderID": "ord-exit"}
    # Attempt 1 partially fills 1.5 and is cancelled; attempts 2-3 miss.
    statuses = iter([
        {"status": "LIVE", "size_matched": 0},          # pre-cancel check 1
        {"status": "CANCELED", "price": 0.54, "size_matched": 1.5},  # final 1
        {"status": "LIVE", "size_matched": 0},          # pre-cancel check 2
        {"status": "CANCELED", "size_matched": 0},      # final 2
        {"status": "LIVE", "size_matched": 0},          # pre-cancel check 3
        {"status": "CANCELED", "size_matched": 0},      # final 3
    ])
    poly.get_order_status = AsyncMock(side_effect=lambda _id: next(statuses))

    with patch("core.execution.asyncio.sleep", new=AsyncMock()), \
         patch("core.execution.halt_active", return_value=False), \
         patch("core.execution._log_to_db", return_value=99):
        out = asyncio.run(
            eng.close_position_early(exit_price=0.55, reason="btc_stop_$91")
        )

    from strategy.feature_snapshot import fee_per_share
    # Sold portion: booked at actual 0.54 on 1.5 shares.
    assert out is not None
    assert out.num_shares == 1.5
    assert out.resolution == "EARLY_EXIT:btc_stop_$91"
    expected_pnl = (0.54 - 0.50) * 1.5 - fee_per_share(0.50) * 1.5
    assert abs(out.pnl - expected_pnl) < 1e-9
    assert out.db_id == 99                     # its own ledger row
    # Remainder rides: pending trade reduced proportionally.
    assert eng.pending_trade is trade
    assert trade.num_shares == 2.5
    assert abs(trade.size_usdc - 0.50 * 2.5) < 1e-9
    # Bankroll credited only with the sold portion's capital + pnl.
    assert abs(eng.bankroll - (bankroll0 + out.size_usdc + out.pnl)) < 1e-9
