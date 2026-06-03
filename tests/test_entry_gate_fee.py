"""Entry-gate fee regression tests (2026-06-03).

The entry gate's `best_ev > 0` condition decides which trades fire. It must
use the real Polymarket taker fee `0.07*p*(1-p)` (charged at entry on every
trade), NOT the old 2%-of-profit-on-winners haircut. Using the lenient 2%
fee let marginal near-50/50 trades through that are unprofitable at real
fees — a "shock to the system" in live testing.

Correct per-share EV = (q - p) - fee_per_share(p), identical to
strategy.feature_snapshot.net_ev_per_share. See decisions_BTC J17.
"""

import pytest

from strategy.entry_logic import EntryLogic
from strategy.feature_snapshot import fee_per_share


def _evaluate(logic: EntryLogic, est_prob_up: float, yes_ask: float,
              no_ask: float, yes_bid: float, no_bid: float):
    return logic.evaluate(
        est_prob_up=est_prob_up,
        yes_best_ask=yes_ask, no_best_ask=no_ask,
        yes_best_bid=yes_bid, no_best_bid=no_bid,
        seconds_remaining=180.0, has_position=False,
    )


class TestGateUsesRealFee:
    def test_ev_yes_matches_net_ev_formula(self) -> None:
        logic = EntryLogic()
        # q=0.60, p(yes_ask)=0.55 -> ev = (0.60-0.55) - 0.07*0.55*0.45
        d = _evaluate(logic, 0.60, 0.55, 0.45, 0.53, 0.43)
        expected = (0.60 - 0.55) - fee_per_share(0.55)
        assert d.ev_yes == pytest.approx(expected)
        assert d.ev_yes == pytest.approx(0.032675)

    def test_ev_no_matches_net_ev_formula(self) -> None:
        logic = EntryLogic()
        # NO side: q_no = 1-0.40 = 0.60, p(no_ask)=0.55
        d = _evaluate(logic, 0.40, 0.45, 0.55, 0.43, 0.53)
        expected = ((1.0 - 0.40) - 0.55) - fee_per_share(0.55)
        assert d.ev_no == pytest.approx(expected)

    def test_thin_near_midpoint_trade_now_negative_ev(self) -> None:
        # q=0.53, p=0.52: gross 0.01, real fee 0.07*0.52*0.48=0.017472
        # New EV = 0.01 - 0.017472 < 0 (was +0.0049 under the old 2% fee).
        logic = EntryLogic()
        d = _evaluate(logic, 0.53, 0.52, 0.48, 0.51, 0.47)
        assert d.ev_yes < 0
        assert d.ev_yes == pytest.approx((0.53 - 0.52) - fee_per_share(0.52))

    def test_thin_midpoint_trade_blocked_from_entry(self) -> None:
        # Even with enough prob_edge, negative real-fee EV must block entry.
        logic = EntryLogic(min_edge=0.0, max_edge=0.0, min_confidence=0.0)
        d = _evaluate(logic, 0.53, 0.52, 0.48, 0.51, 0.47)
        assert d.should_enter is False
