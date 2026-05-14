"""Tests for OracleLagSignal — the L1 signal after the 2026-05-14 fix.

The original L1 blended 60% (exchange - oracle_NOW) and 40% (exchange -
oracle_OPEN). The 60% lag component was removed because it was
mathematically measuring half the inter-exchange spread, injecting a
persistent ~+0.21 offset that biased Bot G and Bot K to 100% YES trades.

The fixed signal uses only the open component at full weight.
"""

import pytest

from signals.oracle_lag import OracleLagSignal


class TestBasicBehaviour:
    def test_returns_zero_when_exchange_price_invalid(self):
        sig = OracleLagSignal()
        assert sig.compute(0.0, 80000.0, 80000.0) == 0.0
        assert sig.compute(-1.0, 80000.0, 80000.0) == 0.0

    def test_returns_zero_when_no_window_open_price(self):
        # Without a window-open reference, signal is undefined.
        sig = OracleLagSignal()
        assert sig.compute(80100.0, 80050.0, 0.0) == 0.0
        assert sig.compute(80100.0, 80050.0, -1.0) == 0.0

    def test_positive_when_btc_above_open(self):
        # BTC moved up from open by 0.1% (max signal magnitude)
        sig = OracleLagSignal()
        result = sig.compute(80080.0, 80000.0, 80000.0)
        # (80080 - 80000) / 80000 = 0.001 = max_expected_lag
        assert result == pytest.approx(1.0, abs=0.001)

    def test_negative_when_btc_below_open(self):
        sig = OracleLagSignal()
        result = sig.compute(79920.0, 80000.0, 80000.0)
        assert result == pytest.approx(-1.0, abs=0.001)

    def test_zero_when_btc_at_open(self):
        sig = OracleLagSignal()
        assert sig.compute(80000.0, 80000.0, 80000.0) == 0.0

    def test_proportional_to_move(self):
        # Half the max move = half the signal
        sig = OracleLagSignal()
        result = sig.compute(80040.0, 80000.0, 80000.0)
        assert result == pytest.approx(0.5, abs=0.01)


class TestLagComponentRemoved:
    """Verify that oracle_NOW divergence no longer affects the signal."""

    def test_oracle_now_does_not_affect_signal(self):
        # Construct a case where oracle_NOW != exchange_price but
        # exchange == oracle_OPEN. Old code would have produced a nonzero
        # signal from the lag component. New code returns 0.
        sig = OracleLagSignal()
        result = sig.compute(
            exchange_price=80000.0,
            oracle_price=79984.0,  # $16 below exchange (the bug condition)
            oracle_open_price=80000.0,
        )
        assert result == 0.0

    def test_oracle_now_offset_does_not_inject_constant_bias(self):
        """The bug we're fixing: a persistent oracle_NOW vs exchange offset
        would have injected a constant +0.21-ish signal. Verify it doesn't.
        """
        sig = OracleLagSignal()
        # Simulate the May 2026 conditions: exchange consistently $16 above
        # oracle_NOW, but BTC at the window open price (no real movement).
        for exchange in [80000, 81000, 75000, 90000]:
            offset = 16.0
            result = sig.compute(
                exchange_price=exchange,
                oracle_price=exchange - offset,
                oracle_open_price=exchange,  # BTC at open = no signal
            )
            assert result == 0.0, (
                f"With exchange={exchange}, oracle_NOW offset, but no move "
                f"from open, signal should be 0 (got {result})"
            )


class TestDiagnosticFields:
    """The instance attrs used by SignalDiagnosticLogger."""

    def test_last_lag_component_always_zero(self):
        sig = OracleLagSignal()
        sig.compute(80100.0, 80050.0, 80000.0)
        # Lag component no longer contributes - always 0.
        assert sig.last_lag_component == 0.0

    def test_last_open_component_matches_signal(self):
        sig = OracleLagSignal()
        result = sig.compute(80040.0, 80000.0, 80000.0)
        assert sig.last_open_component == pytest.approx(result)

    def test_diagnostic_fields_reset_on_invalid_input(self):
        sig = OracleLagSignal()
        sig.compute(80040.0, 80000.0, 80000.0)  # set fields
        sig.compute(0.0, 80000.0, 80000.0)  # invalid -> should reset
        assert sig.last_lag_component == 0.0
        assert sig.last_open_component == 0.0


class TestClamping:
    def test_clamps_at_plus_one(self):
        sig = OracleLagSignal()
        # 1% move = 10x max_expected_lag, should clamp to +1
        result = sig.compute(80800.0, 80000.0, 80000.0)
        assert result == 1.0

    def test_clamps_at_minus_one(self):
        sig = OracleLagSignal()
        result = sig.compute(79200.0, 80000.0, 80000.0)
        assert result == -1.0


class TestCustomMaxLag:
    def test_custom_max_lag_changes_saturation(self):
        # max_expected_lag = 0.002 means a 0.2% move = full signal
        sig = OracleLagSignal(max_expected_lag=0.002)
        # 0.1% move = 0.5 signal (half of new max)
        result = sig.compute(80080.0, 80000.0, 80000.0)
        assert result == pytest.approx(0.5, abs=0.01)
