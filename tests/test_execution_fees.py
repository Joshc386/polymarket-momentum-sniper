"""Fee-accounting regression tests for the execution resolvers (2026-06-03).

Polymarket crypto markets charge a taker fee at ENTRY on every trade
(win or lose):

    fee = FEE_RATE * C * p * (1 - p)

where C = number of contracts (shares), p = entry (fill) price, and
FEE_RATE = 0.07. The fee is charged once, at entry, regardless of how the
position closes (window resolution OR early exit).

Before this fix the resolvers used a wrong 2%-of-gross-profit fee charged
only on winners (`core/execution.py`). These are fixed-input → known-output
regression tests so the correct money math can't silently drift. See
decisions_BTC J17.
"""

import pytest

from core.execution import (
    PaperExecutionEngine,
    LiveExecutionEngine,
    TradeRecord,
    _resolve_trade,
)
from strategy.feature_snapshot import FEE_RATE, fee_per_share
from logging_db.database import Database


def _make_trade(side: str, entry_price: float, num_shares: float) -> TradeRecord:
    """Build a minimal TradeRecord with cost = shares * entry_price."""
    return TradeRecord(
        market_id="m", market_slug="s", side=side,
        entry_price=entry_price, size_usdc=num_shares * entry_price,
        num_shares=num_shares,
        oracle_lag_signal=0.0, momentum_signal=0.0, liquidation_signal=0.0,
        combined_signal=0.0, estimated_prob_up=0.5, market_implied_prob=0.5,
        edge=0.0, time_remaining_secs=120.0, btc_price_at_entry=0.0,
        oracle_price_at_entry=0.0, oracle_price_at_open=0.0, is_paper=True,
    )


class TestFeeRate:
    def test_fee_rate_is_seven_percent(self) -> None:
        assert FEE_RATE == 0.07

    def test_fee_per_share_known_input(self) -> None:
        # 0.07 * 0.40 * 0.60 = 0.0168 per share
        assert fee_per_share(0.40) == pytest.approx(0.0168)

    def test_fee_per_share_max_at_midpoint(self) -> None:
        # peaks at p = 0.50: 0.07 * 0.25 = 0.0175
        assert fee_per_share(0.50) == pytest.approx(0.0175)


class TestResolveTradeFee:
    """Entry fee charged on every trade; worked $40 / 100-contract example."""

    def test_yes_win_subtracts_entry_fee(self) -> None:
        trade = _make_trade("YES", 0.40, 100)  # cost $40
        won = _resolve_trade(trade, "UP")
        # fee = 0.07*100*0.40*0.60 = 1.68; gross = 100 - 40 = 60
        assert won is True
        assert trade.pnl == pytest.approx(58.32)

    def test_yes_loss_still_charges_entry_fee(self) -> None:
        trade = _make_trade("YES", 0.40, 100)
        won = _resolve_trade(trade, "DOWN")
        # loss: -cost - fee = -40 - 1.68
        assert won is False
        assert trade.pnl == pytest.approx(-41.68)

    def test_no_win_subtracts_entry_fee(self) -> None:
        trade = _make_trade("NO", 0.40, 100)
        won = _resolve_trade(trade, "DOWN")
        assert won is True
        assert trade.pnl == pytest.approx(58.32)

    def test_no_loss_still_charges_entry_fee(self) -> None:
        trade = _make_trade("NO", 0.40, 100)
        won = _resolve_trade(trade, "UP")
        assert won is False
        assert trade.pnl == pytest.approx(-41.68)

    def test_winner_always_net_positive(self) -> None:
        # fee can never flip a correct-direction binary win negative
        trade = _make_trade("YES", 0.50, 100)
        _resolve_trade(trade, "UP")
        assert trade.pnl > 0


class TestEarlyExitFee:
    """Early-exit subtracts the same single entry fee (p = entry price)."""

    def _paper_engine(self) -> PaperExecutionEngine:
        eng = PaperExecutionEngine(db=Database(":memory:"))  # not connected
        return eng

    def test_paper_early_exit_charges_entry_fee(self) -> None:
        eng = self._paper_engine()
        trade = _make_trade("YES", 0.40, 100)
        trade.db_id = None
        eng.pending_trade = trade
        result = eng.close_position_early(exit_price=0.50, reason="t")
        # gross = (0.50-0.40)*100 = 10; fee = 1.68 -> pnl = 8.32
        assert result.pnl == pytest.approx(8.32)

    def test_live_early_exit_charges_entry_fee(self) -> None:
        eng = LiveExecutionEngine(db=Database(":memory:"), poly_client=None)
        trade = _make_trade("YES", 0.40, 100)
        trade.db_id = None
        eng.pending_trade = trade
        result = eng.close_position_early(exit_price=0.45, reason="t")
        # gross = (0.45-0.40)*100 = 5; fee = 1.68 -> pnl = 3.32
        assert result.pnl == pytest.approx(3.32)
