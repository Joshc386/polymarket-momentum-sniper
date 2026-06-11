"""Tests for the J28 resolution-repair tool's PnL math (money-critical)."""
import pytest

from tools.repair_resolutions import expected_pnl


def test_win_pnl_post_j17_fee():
    # 10 shares at 0.50: win pays 10*(1-0.5) minus fee 0.07*0.5*0.5*10
    assert expected_pnl(won=True, entry=0.50, shares=10.0) == pytest.approx(
        10 * 0.5 - 0.07 * 0.5 * 0.5 * 10)


def test_loss_pnl_post_j17_fee():
    # loss costs the stake plus the entry fee
    assert expected_pnl(won=False, entry=0.50, shares=10.0) == pytest.approx(
        -10 * 0.5 - 0.07 * 0.5 * 0.5 * 10)


def test_flip_win_to_loss_changes_sign_and_magnitude():
    w = expected_pnl(won=True, entry=0.45, shares=11.0)
    l = expected_pnl(won=False, entry=0.45, shares=11.0)
    assert w > 0 > l
    # win+loss = shares*(1 - 2*entry) - 2*fee
    fee = 0.07 * 0.45 * 0.55 * 11.0
    assert w + l == pytest.approx(11.0 * (1 - 2 * 0.45) - 2 * fee)
