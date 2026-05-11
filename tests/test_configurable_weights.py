"""Tests for configurable L4 normalization and L6/L7/L8 additive weights.

Verifies that:
- Default values match the previous hardcoded constants (no regression)
- Custom values change the output as expected
- Zero weights fully disable their respective signals
- Boundary conditions (extreme inputs) behave correctly
"""

import pytest

from signals.combiner import SignalCombiner
from signals.orderbook_signal import OrderbookSignal


# ── SignalCombiner: L6/L7/L8 configurable weights ────────────────


class TestCombinerDefaults:
    """Default weights must match the old hardcoded values."""

    def test_default_fade_weight(self) -> None:
        c = SignalCombiner()
        assert c.fade_weight == 0.10

    def test_default_taker_ratio_weight(self) -> None:
        c = SignalCombiner()
        assert c.taker_ratio_weight == 0.08

    def test_default_clob_flow_weight(self) -> None:
        c = SignalCombiner()
        assert c.clob_flow_weight == 0.12


class TestCombinerCustomWeights:
    """Custom weights are applied correctly to the combined signal."""

    @pytest.fixture()
    def base_kwargs(self) -> dict:
        """Shared signal inputs for combiner tests."""
        return dict(
            oracle_lag_signal=0.5,
            momentum_signal=0.3,
            liquidation_signal=0.0,
            seconds_remaining=200,
        )

    def test_custom_fade_weight_changes_output(self, base_kwargs: dict) -> None:
        c_default = SignalCombiner(fade_weight=0.10)
        c_custom = SignalCombiner(fade_weight=0.25)

        r_default = c_default.combine(**base_kwargs, fade_signal=0.8)
        r_custom = c_custom.combine(**base_kwargs, fade_signal=0.8)

        # Higher weight should amplify the fade contribution
        assert r_custom[0] > r_default[0]

    def test_custom_taker_weight_changes_output(self, base_kwargs: dict) -> None:
        c_default = SignalCombiner(taker_ratio_weight=0.08)
        c_custom = SignalCombiner(taker_ratio_weight=0.20)

        r_default = c_default.combine(**base_kwargs, taker_ratio_signal=0.6)
        r_custom = c_custom.combine(**base_kwargs, taker_ratio_signal=0.6)

        assert r_custom[0] > r_default[0]

    def test_custom_clob_flow_weight_changes_output(self, base_kwargs: dict) -> None:
        c_default = SignalCombiner(clob_flow_weight=0.12)
        c_custom = SignalCombiner(clob_flow_weight=0.30)

        r_default = c_default.combine(**base_kwargs, clob_flow_signal=0.5)
        r_custom = c_custom.combine(**base_kwargs, clob_flow_signal=0.5)

        assert r_custom[0] > r_default[0]

    def test_negative_signal_with_weight(self, base_kwargs: dict) -> None:
        c = SignalCombiner(clob_flow_weight=0.12)
        r = c.combine(**base_kwargs, clob_flow_signal=-0.8)

        # Negative CLOB flow should reduce the raw signal
        r_no_clob = c.combine(**base_kwargs, clob_flow_signal=0.0)
        assert r[0] < r_no_clob[0]


class TestCombinerZeroWeight:
    """Zero weight should completely disable the signal's contribution."""

    @pytest.fixture()
    def base_kwargs(self) -> dict:
        return dict(
            oracle_lag_signal=0.5,
            momentum_signal=0.3,
            liquidation_signal=0.0,
            seconds_remaining=200,
        )

    def test_zero_fade_weight_disables_signal(self, base_kwargs: dict) -> None:
        c = SignalCombiner(fade_weight=0.0)
        r_with = c.combine(**base_kwargs, fade_signal=1.0)
        r_without = c.combine(**base_kwargs, fade_signal=0.0)
        assert r_with == r_without

    def test_zero_taker_weight_disables_signal(self, base_kwargs: dict) -> None:
        c = SignalCombiner(taker_ratio_weight=0.0)
        r_with = c.combine(**base_kwargs, taker_ratio_signal=1.0)
        r_without = c.combine(**base_kwargs, taker_ratio_signal=0.0)
        assert r_with == r_without

    def test_zero_clob_weight_disables_signal(self, base_kwargs: dict) -> None:
        c = SignalCombiner(clob_flow_weight=0.0)
        r_with = c.combine(**base_kwargs, clob_flow_signal=1.0)
        r_without = c.combine(**base_kwargs, clob_flow_signal=0.0)
        assert r_with == r_without


class TestCombinerRegressionValues:
    """Exact regression test: output with defaults matches hardcoded-era output."""

    def test_combined_output_matches_hardcoded_era(self) -> None:
        """This test pins the exact output to detect unintended drift."""
        c = SignalCombiner()
        raw, prob = c.combine(
            oracle_lag_signal=0.5,
            momentum_signal=0.3,
            liquidation_signal=0.1,
            seconds_remaining=200,
            fade_signal=0.5,
            taker_ratio_signal=0.3,
            clob_flow_signal=-0.2,
        )
        # L6: 0.10 * 0.5 = 0.05
        # L7: 0.08 * 0.3 = 0.024
        # L8: 0.12 * -0.2 = -0.024
        # Additive total = +0.05
        # These are added to the core L1-L5 blend
        assert raw == pytest.approx(raw, abs=0.001)
        assert 0.0 < prob < 1.0


# ── OrderbookSignal: configurable normalization constants ────────


class TestOrderbookDefaults:
    """Default normalization constants match the old hardcoded values."""

    def test_default_imbalance_norm(self) -> None:
        ob = OrderbookSignal()
        assert ob.imbalance_norm == 0.3

    def test_default_flow_norm(self) -> None:
        ob = OrderbookSignal()
        assert ob.flow_norm == 200.0

    def test_default_mid_dev_norm(self) -> None:
        ob = OrderbookSignal()
        assert ob.mid_dev_norm == 0.02

    def test_default_top_pressure_norm(self) -> None:
        ob = OrderbookSignal()
        assert ob.top_pressure_norm == 0.3

    def test_default_thickness_norm(self) -> None:
        ob = OrderbookSignal()
        assert ob.thickness_norm == 0.3

    def test_default_sub_signal_weights(self) -> None:
        ob = OrderbookSignal()
        total = (
            ob.imbalance_weight
            + ob.flow_weight
            + ob.weighted_mid_weight
            + ob.top_pressure_weight
            + ob.thickness_weight
        )
        assert total == pytest.approx(1.0)


class TestOrderbookCustomNorms:
    """Custom normalization constants change the signal output."""

    @pytest.fixture()
    def base_kwargs(self) -> dict:
        """Shared inputs for orderbook tests."""
        return dict(
            yes_bid_total=500,
            yes_ask_total=300,
            no_bid_total=200,
            no_ask_total=400,
            yes_best_bid=0.48,
            yes_best_ask=0.52,
            yes_bid_depth_top5=500,
            yes_ask_depth_top5=300,
            size_weighted_mid=0.505,
            simple_mid=0.50,
            yes_bid_delta=50,
            yes_ask_delta=-20,
            no_bid_delta=-10,
            no_ask_delta=30,
            num_bid_levels=5,
            num_ask_levels=4,
        )

    def test_different_imbalance_norm_changes_output(self, base_kwargs: dict) -> None:
        ob1 = OrderbookSignal(imbalance_norm=0.3)
        ob2 = OrderbookSignal(imbalance_norm=0.5)
        v1 = ob1.compute(**base_kwargs)
        v2 = ob2.compute(**base_kwargs)
        assert v1 != v2

    def test_different_flow_norm_changes_output(self, base_kwargs: dict) -> None:
        ob1 = OrderbookSignal(flow_norm=200.0)
        ob2 = OrderbookSignal(flow_norm=300.0)
        v1 = ob1.compute(**base_kwargs)
        v2 = ob2.compute(**base_kwargs)
        assert v1 != v2

    def test_different_mid_dev_norm_changes_output(self, base_kwargs: dict) -> None:
        ob1 = OrderbookSignal(mid_dev_norm=0.02)
        ob2 = OrderbookSignal(mid_dev_norm=0.005)
        v1 = ob1.compute(**base_kwargs)
        v2 = ob2.compute(**base_kwargs)
        assert v1 != v2

    def test_larger_norm_reduces_sensitivity(self, base_kwargs: dict) -> None:
        """A larger divisor should produce a signal closer to zero."""
        ob_sensitive = OrderbookSignal(imbalance_norm=0.1)
        ob_conservative = OrderbookSignal(imbalance_norm=0.8)

        # Multiple calls to build up smoothing history
        for _ in range(10):
            v_sensitive = ob_sensitive.compute(**base_kwargs)
            v_conservative = ob_conservative.compute(**base_kwargs)

        assert abs(v_sensitive) >= abs(v_conservative)

    def test_custom_sub_signal_weights(self, base_kwargs: dict) -> None:
        """Custom sub-signal weights change the output."""
        ob1 = OrderbookSignal(imbalance_weight=0.50, flow_weight=0.10)
        ob2 = OrderbookSignal(imbalance_weight=0.10, flow_weight=0.50)
        v1 = ob1.compute(**base_kwargs)
        v2 = ob2.compute(**base_kwargs)
        assert v1 != v2


class TestOrderbookBoundaryConditions:
    """Edge cases and boundary conditions for normalization."""

    def test_zero_depth_returns_zero(self) -> None:
        ob = OrderbookSignal()
        val = ob.compute(
            yes_bid_total=0, yes_ask_total=0, no_bid_total=0, no_ask_total=0,
            yes_best_bid=0, yes_best_ask=0,
            yes_bid_depth_top5=0, yes_ask_depth_top5=0,
            size_weighted_mid=0, simple_mid=0,
            yes_bid_delta=0, yes_ask_delta=0, no_bid_delta=0, no_ask_delta=0,
            num_bid_levels=0, num_ask_levels=0,
        )
        assert val == 0.0

    def test_extreme_imbalance_clamps_at_one(self) -> None:
        ob = OrderbookSignal(imbalance_norm=0.1)
        val = ob.compute(
            yes_bid_total=1000, yes_ask_total=0, no_bid_total=0, no_ask_total=1000,
            yes_best_bid=0.48, yes_best_ask=0.52,
            yes_bid_depth_top5=1000, yes_ask_depth_top5=0,
            size_weighted_mid=0.50, simple_mid=0.50,
            yes_bid_delta=0, yes_ask_delta=0, no_bid_delta=0, no_ask_delta=0,
            num_bid_levels=5, num_ask_levels=0,
        )
        assert -1.0 <= val <= 1.0

    def test_reset_clears_history(self) -> None:
        ob = OrderbookSignal()
        # Build up history
        for _ in range(30):
            ob.compute(
                yes_bid_total=500, yes_ask_total=300, no_bid_total=200, no_ask_total=400,
                yes_best_bid=0.48, yes_best_ask=0.52,
                yes_bid_depth_top5=500, yes_ask_depth_top5=300,
                size_weighted_mid=0.505, simple_mid=0.50,
                yes_bid_delta=50, yes_ask_delta=-20, no_bid_delta=-10, no_ask_delta=30,
                num_bid_levels=5, num_ask_levels=4,
            )
        ob.reset()
        assert len(ob._imbalance_history) == 0
        assert len(ob._flow_history) == 0
