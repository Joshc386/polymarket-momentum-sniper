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
