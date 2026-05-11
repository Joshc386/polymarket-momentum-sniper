"""Tests for L9b Absorption Detection Signal.

Verifies that:
- Bid absorption (heavy selling + resilient bids) produces bullish signal
- Ask absorption (heavy buying + resilient asks) produces bearish signal
- Balanced flow produces no signal
- Signal decays when absorption stops
- Volume and imbalance thresholds gate the signal correctly
- Reset clears all state
- Integration with combiner works correctly
"""

import pytest
from dataclasses import dataclass

from signals.absorption import AbsorptionSignal
from signals.combiner import SignalCombiner


@dataclass
class MockTradeFlow:
    """Minimal mock of CLOBTradeFlow for testing."""

    yes_buy_volume: float = 0.0
    yes_sell_volume: float = 0.0
    no_buy_volume: float = 0.0
    no_sell_volume: float = 0.0
    yes_trade_count: int = 0
    no_trade_count: int = 0
    total_volume: float = 0.0
    net_yes_flow: float = 0.0
    imbalance: float = 0.0


def _run_ticks(signal: AbsorptionSignal, flow: MockTradeFlow,
               bid_d: float, ask_d: float, no_bid_d: float, no_ask_d: float,
               n: int = 5) -> float:
    """Run n ticks with the same inputs and return the final signal."""
    val = 0.0
    for _ in range(n):
        val = signal.compute(flow, bid_d, ask_d, no_bid_d, no_ask_d)
    return val


# ── Bid absorption (bullish) ─────────────────────────────────────


class TestBidAbsorption:
    """Heavy selling pressure + resilient bids = bullish signal."""

    def test_sell_pressure_with_stable_bids_is_bullish(self) -> None:
        a = AbsorptionSignal(smoothing_window=3)
        flow = MockTradeFlow(yes_sell_volume=200, no_buy_volume=100, total_volume=300)
        val = _run_ticks(a, flow, bid_d=50, ask_d=0, no_bid_d=0, no_ask_d=20)
        assert val > 0

    def test_stronger_resilience_gives_stronger_signal(self) -> None:
        a1 = AbsorptionSignal(smoothing_window=3)
        flow = MockTradeFlow(yes_sell_volume=200, total_volume=200)
        v1 = _run_ticks(a1, flow, bid_d=10, ask_d=0, no_bid_d=0, no_ask_d=0)

        a2 = AbsorptionSignal(smoothing_window=3)
        v2 = _run_ticks(a2, flow, bid_d=200, ask_d=0, no_bid_d=0, no_ask_d=0)
        assert v2 >= v1

    def test_sell_pressure_with_collapsing_bids_no_absorption(self) -> None:
        """If bids are collapsing under sell pressure, that's not absorption."""
        a = AbsorptionSignal(smoothing_window=3, resilience_threshold=0.0)
        flow = MockTradeFlow(yes_sell_volume=200, total_volume=200)
        # Bids are decreasing (negative delta) — no resilience
        val = _run_ticks(a, flow, bid_d=-100, ask_d=0, no_bid_d=0, no_ask_d=0)
        # Signal should be zero or very small (no bid absorption)
        assert val <= 0.01


# ── Ask absorption (bearish) ─────────────────────────────────────


class TestAskAbsorption:
    """Heavy buying pressure + resilient asks = bearish signal."""

    def test_buy_pressure_with_stable_asks_is_bearish(self) -> None:
        a = AbsorptionSignal(smoothing_window=3)
        flow = MockTradeFlow(yes_buy_volume=200, no_sell_volume=100, total_volume=300)
        val = _run_ticks(a, flow, bid_d=0, ask_d=50, no_bid_d=20, no_ask_d=0)
        assert val < 0

    def test_buy_pressure_with_collapsing_asks_no_absorption(self) -> None:
        a = AbsorptionSignal(smoothing_window=3, resilience_threshold=0.0)
        flow = MockTradeFlow(yes_buy_volume=200, total_volume=200)
        val = _run_ticks(a, flow, bid_d=0, ask_d=-100, no_bid_d=0, no_ask_d=0)
        assert val >= -0.01


# ── Neutral conditions ───────────────────────────────────────────


class TestNeutralConditions:
    """No absorption detected in balanced or low-volume conditions."""

    def test_balanced_flow_gives_zero(self) -> None:
        a = AbsorptionSignal(smoothing_window=3)
        flow = MockTradeFlow(
            yes_buy_volume=100, yes_sell_volume=100, total_volume=200
        )
        val = _run_ticks(a, flow, bid_d=0, ask_d=0, no_bid_d=0, no_ask_d=0)
        assert abs(val) < 0.01

    def test_no_trade_flow_gives_zero(self) -> None:
        a = AbsorptionSignal()
        val = a.compute(None, 0, 0, 0, 0)
        assert val == 0.0

    def test_below_min_volume_gives_zero(self) -> None:
        a = AbsorptionSignal(min_pressure_volume=500, smoothing_window=3)
        flow = MockTradeFlow(yes_sell_volume=10, total_volume=10)
        val = _run_ticks(a, flow, bid_d=50, ask_d=0, no_bid_d=0, no_ask_d=0)
        assert abs(val) < 0.01

    def test_below_imbalance_threshold_gives_zero(self) -> None:
        """Flow is one-sided but not enough to exceed threshold."""
        a = AbsorptionSignal(
            pressure_imbalance_threshold=0.40, smoothing_window=3
        )
        # 55/45 split — below 0.40 threshold (would need 90/10)
        flow = MockTradeFlow(
            yes_sell_volume=55, yes_buy_volume=45, total_volume=100
        )
        val = _run_ticks(a, flow, bid_d=50, ask_d=0, no_bid_d=0, no_ask_d=0)
        assert abs(val) < 0.01


# ── Signal dynamics ──────────────────────────────────────────────


class TestSignalDynamics:
    """Decay, smoothing, and signal clamping behaviour."""

    def test_signal_decays_without_reinforcement(self) -> None:
        a = AbsorptionSignal(smoothing_window=3, decay_rate=0.5)
        flow = MockTradeFlow(yes_sell_volume=300, total_volume=300)
        _run_ticks(a, flow, bid_d=100, ask_d=0, no_bid_d=0, no_ask_d=50)
        peak = a.last_signal
        assert peak > 0

        # No flow for several ticks — should decay
        for _ in range(3):
            a.compute(None, 0, 0, 0, 0)
        assert abs(a.last_signal) < abs(peak)

    def test_signal_clamped_to_unit_range(self) -> None:
        a = AbsorptionSignal(smoothing_window=3, strength_scale=100.0)
        flow = MockTradeFlow(yes_sell_volume=1000, total_volume=1000)
        val = _run_ticks(a, flow, bid_d=500, ask_d=0, no_bid_d=0, no_ask_d=500)
        assert -1.0 <= val <= 1.0

    def test_needs_minimum_history(self) -> None:
        a = AbsorptionSignal(smoothing_window=5)
        flow = MockTradeFlow(yes_sell_volume=200, total_volume=200)
        # First 2 ticks: not enough history
        v1 = a.compute(flow, 50, 0, 0, 0)
        v2 = a.compute(flow, 50, 0, 0, 0)
        assert v1 == 0.0
        assert v2 == 0.0
        # Third tick: now has minimum (3)
        v3 = a.compute(flow, 50, 0, 0, 0)
        # May or may not fire depending on accumulated pressure
        # but at least it should not error


# ── Reset ────────────────────────────────────────────────────────


class TestReset:
    """Reset clears all internal state."""

    def test_reset_zeroes_signal(self) -> None:
        a = AbsorptionSignal(smoothing_window=3)
        flow = MockTradeFlow(yes_sell_volume=200, total_volume=200)
        _run_ticks(a, flow, bid_d=50, ask_d=0, no_bid_d=0, no_ask_d=0)
        assert a.last_signal != 0.0

        a.reset()
        assert a.last_signal == 0.0

    def test_reset_clears_history(self) -> None:
        a = AbsorptionSignal(smoothing_window=3)
        flow = MockTradeFlow(yes_sell_volume=200, total_volume=200)
        _run_ticks(a, flow, bid_d=50, ask_d=0, no_bid_d=0, no_ask_d=0)

        a.reset()
        assert len(a._pressure_history) == 0
        assert len(a._bid_resilience_history) == 0
        assert len(a._ask_resilience_history) == 0


# ── Combiner integration ────────────────────────────────────────


class TestCombinerIntegration:
    """Absorption signal integrates correctly with SignalCombiner."""

    def test_default_weight_is_zero(self) -> None:
        c = SignalCombiner()
        assert c.absorption_weight == 0.0

    def test_zero_weight_disables_absorption(self) -> None:
        c = SignalCombiner(absorption_weight=0.0)
        r1 = c.combine(
            oracle_lag_signal=0.5, momentum_signal=0.3, liquidation_signal=0.0,
            seconds_remaining=200, absorption_signal=1.0,
        )
        r2 = c.combine(
            oracle_lag_signal=0.5, momentum_signal=0.3, liquidation_signal=0.0,
            seconds_remaining=200, absorption_signal=0.0,
        )
        assert r1 == r2

    def test_nonzero_weight_adds_absorption(self) -> None:
        c = SignalCombiner(absorption_weight=0.15)
        r_with = c.combine(
            oracle_lag_signal=0.3, momentum_signal=0.2, liquidation_signal=0.0,
            seconds_remaining=200, absorption_signal=0.8,
        )
        r_without = c.combine(
            oracle_lag_signal=0.3, momentum_signal=0.2, liquidation_signal=0.0,
            seconds_remaining=200, absorption_signal=0.0,
        )
        # Positive absorption should increase the raw signal
        assert r_with[0] > r_without[0]

    def test_negative_absorption_reduces_signal(self) -> None:
        c = SignalCombiner(absorption_weight=0.15)
        r_with = c.combine(
            oracle_lag_signal=0.3, momentum_signal=0.2, liquidation_signal=0.0,
            seconds_remaining=200, absorption_signal=-0.8,
        )
        r_without = c.combine(
            oracle_lag_signal=0.3, momentum_signal=0.2, liquidation_signal=0.0,
            seconds_remaining=200, absorption_signal=0.0,
        )
        assert r_with[0] < r_without[0]


# ── Custom configuration ────────────────────────────────────────


class TestCustomConfig:
    """Custom parameters work as expected."""

    def test_high_strength_scale_amplifies(self) -> None:
        a_low = AbsorptionSignal(smoothing_window=3, strength_scale=1.0)
        a_high = AbsorptionSignal(smoothing_window=3, strength_scale=5.0)
        flow = MockTradeFlow(yes_sell_volume=150, total_volume=150)

        v_low = _run_ticks(a_low, flow, bid_d=30, ask_d=0, no_bid_d=0, no_ask_d=0)
        v_high = _run_ticks(a_high, flow, bid_d=30, ask_d=0, no_bid_d=0, no_ask_d=0)
        # Higher scale should produce larger signal (up to clamp)
        assert abs(v_high) >= abs(v_low)

    def test_zero_decay_latches_signal(self) -> None:
        a = AbsorptionSignal(smoothing_window=3, decay_rate=0.0)
        flow = MockTradeFlow(yes_sell_volume=200, total_volume=200)
        _run_ticks(a, flow, bid_d=50, ask_d=0, no_bid_d=0, no_ask_d=0)
        peak = a.last_signal

        # No flow — should NOT decay
        a.compute(None, 0, 0, 0, 0)
        assert a.last_signal == peak

    def test_full_decay_drops_instantly(self) -> None:
        a = AbsorptionSignal(smoothing_window=3, decay_rate=1.0)
        flow = MockTradeFlow(yes_sell_volume=200, total_volume=200)
        _run_ticks(a, flow, bid_d=50, ask_d=0, no_bid_d=0, no_ask_d=0)
        assert a.last_signal != 0.0

        a.compute(None, 0, 0, 0, 0)
        assert a.last_signal == pytest.approx(0.0, abs=0.001)
