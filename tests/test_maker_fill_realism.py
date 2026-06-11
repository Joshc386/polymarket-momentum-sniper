"""Regression tests for the maker-fill realism fill model (the go-live gate)."""
import numpy as np
import pytest

from backtest.maker_fill_realism import taker_fee, would_fill


def test_would_fill_true_when_side_drops_one_spread():
    # entry bid 0.50, spread 0.02 -> fills if path reaches <= 0.48
    assert would_fill(0.50, 0.02, np.array([0.49, 0.48, 0.47])) is True


def test_would_fill_false_when_drop_too_small():
    # never reaches 0.48
    assert would_fill(0.50, 0.02, np.array([0.495, 0.49, 0.485])) is False


def test_would_fill_boundary_exactly_one_spread():
    # 0.48 == 0.50 - 0.02 should fill (<=)
    assert would_fill(0.50, 0.02, np.array([0.48])) is True


def test_would_fill_empty_path_is_no_fill():
    assert would_fill(0.50, 0.02, np.array([])) is False


def test_would_fill_price_rising_never_fills():
    # adverse-selection intuition: if the side strengthens (rises), no maker fill
    assert would_fill(0.50, 0.02, np.array([0.51, 0.55, 0.60])) is False


def test_taker_fee_formula():
    assert taker_fee(0.50) == pytest.approx(0.072 * 0.25)
    assert taker_fee(0.40) == pytest.approx(0.072 * 0.40 * 0.60)


# --- capture-missed-fills chase model --------------------------------------- #
from backtest.capture_missed_fills import capture_pnl, chase_ask_at_timeout


def test_capture_pnl_win_and_loss():
    # win at chase ask 0.55: 1 - 0.55 - fee(0.55)
    assert capture_pnl(1, 0.55) == pytest.approx(1 - 0.55 - 0.072 * 0.55 * 0.45)
    # loss: -ask - fee
    assert capture_pnl(0, 0.55) == pytest.approx(-0.55 - 0.072 * 0.55 * 0.45)


def test_chase_ask_is_mid_plus_spread_capped():
    assert chase_ask_at_timeout(0.50, 0.02) == pytest.approx(0.52)
    assert chase_ask_at_timeout(0.985, 0.02) == 0.99  # cap


def test_capture_pnl_breakeven_intuition():
    # a 50/50 coin chased at 0.52 + fee must be negative EV
    ev = 0.5 * capture_pnl(1, 0.52) + 0.5 * capture_pnl(0, 0.52)
    assert ev < 0


# --- sequential re-post model ------------------------------------------------ #
from backtest.capture_missed_fills import repost_fill


def _mk(secs, pside):
    return np.array(secs, dtype=float), np.array(pside, dtype=float)


def test_repost_round0_fill_at_entry_bid():
    # price drops a full spread within first 10s -> fills at entry bid
    secs, ps = _mk([268, 265, 262], [0.49, 0.48, 0.47])
    filled, px = repost_fill(270, 0.50, 0.02, secs, ps)
    assert filled and px == 0.50


def test_repost_later_round_fills_at_new_touch():
    # price runs up (no round-0 fill), bot re-posts at the ~260s touch (0.56),
    # then price trades back through 0.54 within that round -> fill at the
    # NEW bid 0.56, not the original 0.50
    secs, ps = _mk([268, 263, 258, 255, 252], [0.53, 0.55, 0.56, 0.55, 0.54])
    filled, px = repost_fill(270, 0.50, 0.02, secs, ps)
    assert filled and px == pytest.approx(0.56)


def test_repost_never_fills_when_price_only_rises():
    secs, ps = _mk([265, 255, 245, 235], [0.52, 0.54, 0.56, 0.58])
    filled, _ = repost_fill(270, 0.50, 0.02, secs, ps)
    assert not filled


def test_repost_stops_at_entry_floor():
    # all ticks below the 60s floor -> nothing rests, no fill
    secs, ps = _mk([55, 50, 45], [0.30, 0.30, 0.30])
    filled, _ = repost_fill(65, 0.50, 0.02, secs, ps)
    assert not filled


def test_repost_gate_blocks_reposting():
    # price path where re-posting would eventually fill, but gate is False
    # after entry -> only round 0 rests -> no fill
    secs, ps = _mk([268, 263, 258, 252, 249], [0.53, 0.55, 0.56, 0.55, 0.54])
    gate = np.array([False] * 5)
    filled, _ = repost_fill(270, 0.50, 0.02, secs, ps, gate_ok=gate)
    assert not filled
