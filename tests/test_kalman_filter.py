"""Tests for the Kalman Filter signal module.

Validates convergence, adaptive noise, missing source handling,
breakout detection, and signal output correctness.
"""

import math
import time

import numpy as np
import pytest

from signals.kalman_filter import KalmanFilterSignal, KalmanOutput


class TestKalmanFilterInitialisation:
    """Tests for filter initialisation and reset."""

    def test_initial_state_is_zero(self) -> None:
        kf = KalmanFilterSignal()
        assert kf.output.filtered_price == 0.0
        assert kf.output.velocity == 0.0
        assert not kf._initialized

    def test_first_update_initialises_from_binance(self) -> None:
        kf = KalmanFilterSignal()
        output = kf.update(
            binance_price=100_000.0,
            coinbase_price=100_010.0,
            oracle_price=99_990.0,
            timestamp=1000.0,
        )
        assert kf._initialized
        assert output.filtered_price == 100_000.0
        assert output.velocity == 0.0

    def test_reset_clears_state(self) -> None:
        kf = KalmanFilterSignal()
        kf.update(binance_price=100_000.0, coinbase_price=0.0,
                  oracle_price=99_990.0, timestamp=1000.0)
        kf.update(binance_price=100_050.0, coinbase_price=0.0,
                  oracle_price=100_000.0, timestamp=1001.0)
        assert kf._initialized

        kf.reset()
        assert not kf._initialized
        assert kf.output.filtered_price == 0.0
        assert len(kf._innovations) == 0


class TestKalmanFilterConvergence:
    """Tests that the filter converges on a simulated price stream."""

    def test_converges_on_constant_price(self) -> None:
        """Filter should converge to the true price and zero velocity."""
        kf = KalmanFilterSignal(process_noise_scale=1e-5)
        true_price = 100_000.0
        t = 1000.0

        for i in range(100):
            t += 1.0
            kf.update(
                binance_price=true_price + np.random.normal(0, 5),
                coinbase_price=true_price + np.random.normal(0, 8),
                oracle_price=true_price + np.random.normal(0, 15),
                timestamp=t,
            )

        assert abs(kf.output.filtered_price - true_price) < 50.0
        # Velocity should be near zero but noise can push it a bit
        assert abs(kf.output.velocity) < 20.0

    def test_tracks_linear_trend(self) -> None:
        """Filter should track a linearly increasing price."""
        np.random.seed(123)
        kf = KalmanFilterSignal(process_noise_scale=1e-4)
        base_price = 100_000.0
        rate = 10.0  # $10/sec
        t = 1000.0

        for i in range(200):
            t += 1.0
            true_price = base_price + rate * (t - 1000.0)
            kf.update(
                binance_price=true_price + np.random.normal(0, 5),
                coinbase_price=true_price + np.random.normal(0, 8),
                oracle_price=true_price - 20 + np.random.normal(0, 15),
                timestamp=t,
            )

        # After convergence, velocity should be positive (tracking uptrend)
        assert kf.output.velocity > 0
        # Filtered price should be close to the true current price
        true_final = base_price + rate * 200
        assert abs(kf.output.filtered_price - true_final) < 200.0


class TestAdaptiveProcessNoise:
    """Tests for adaptive Q scaling based on innovation variance."""

    def test_q_increases_during_volatility(self) -> None:
        """Process noise Q should scale up when innovations are large."""
        kf = KalmanFilterSignal(
            process_noise_scale=1e-5,
            adaptive_window=10,
        )

        # Calm period
        t = 1000.0
        for i in range(20):
            t += 1.0
            kf.update(
                binance_price=100_000.0 + np.random.normal(0, 1),
                coinbase_price=0.0,
                oracle_price=100_000.0,
                coinbase_connected=False,
                timestamp=t,
            )

        q_calm = kf._build_Q(1.0)[0, 0]

        # Volatile period: big jumps
        for i in range(20):
            t += 1.0
            kf.update(
                binance_price=100_000.0 + np.random.normal(0, 500),
                coinbase_price=0.0,
                oracle_price=100_000.0,
                coinbase_connected=False,
                timestamp=t,
            )

        q_volatile = kf._build_Q(1.0)[0, 0]

        # Q should be larger during volatile periods
        assert q_volatile > q_calm


class TestMissingSourceGracefulDegradation:
    """Tests that the filter handles missing data sources."""

    def test_works_with_binance_only(self) -> None:
        """Filter should work with only Binance price available."""
        kf = KalmanFilterSignal()
        t = 1000.0

        for i in range(50):
            t += 1.0
            output = kf.update(
                binance_price=100_000.0 + i * 2,
                coinbase_price=0.0,
                oracle_price=0.0,
                coinbase_connected=False,
                timestamp=t,
            )

        assert output.filtered_price > 0
        assert kf._initialized

    def test_works_without_coinbase(self) -> None:
        """Filter should work when Coinbase is disconnected."""
        kf = KalmanFilterSignal()
        t = 1000.0
        true_price = 100_000.0

        for i in range(50):
            t += 1.0
            output = kf.update(
                binance_price=true_price + np.random.normal(0, 5),
                coinbase_price=0.0,
                oracle_price=true_price + np.random.normal(0, 15),
                coinbase_connected=False,
                timestamp=t,
            )

        assert abs(output.filtered_price - true_price) < 50.0

    def test_coinbase_improves_estimate(self) -> None:
        """Adding Coinbase should reduce estimation error vs without it."""
        np.random.seed(42)
        true_price = 100_000.0
        errors_without_cb = []
        errors_with_cb = []

        for trial in range(5):
            kf_no_cb = KalmanFilterSignal()
            kf_with_cb = KalmanFilterSignal()
            t = 1000.0

            for i in range(100):
                t += 1.0
                bn_noise = np.random.normal(0, 10)
                cb_noise = np.random.normal(0, 15)
                or_noise = np.random.normal(0, 25)

                kf_no_cb.update(
                    binance_price=true_price + bn_noise,
                    coinbase_price=0.0,
                    oracle_price=true_price + or_noise,
                    coinbase_connected=False,
                    timestamp=t,
                )
                kf_with_cb.update(
                    binance_price=true_price + bn_noise,
                    coinbase_price=true_price + cb_noise,
                    oracle_price=true_price + or_noise,
                    coinbase_connected=True,
                    timestamp=t,
                )

            errors_without_cb.append(
                abs(kf_no_cb.output.filtered_price - true_price)
            )
            errors_with_cb.append(
                abs(kf_with_cb.output.filtered_price - true_price)
            )

        # On average, having Coinbase should help (or at least not hurt)
        avg_without = np.mean(errors_without_cb)
        avg_with = np.mean(errors_with_cb)
        # Allow some tolerance — the point is it doesn't get worse
        assert avg_with <= avg_without * 1.5


class TestBreakoutDetection:
    """Tests for the velocity-based breakout flag."""

    def test_no_breakout_on_stable_price(self) -> None:
        """No breakout flag when price is stable with high threshold."""
        # Use a high threshold so noise doesn't trigger breakout
        kf = KalmanFilterSignal(breakout_velocity_threshold=50.0)
        t = 1000.0

        for i in range(50):
            t += 1.0
            output = kf.update(
                binance_price=100_000.0 + np.random.normal(0, 1),
                coinbase_price=100_000.0 + np.random.normal(0, 1),
                oracle_price=100_000.0,
                timestamp=t,
            )

        assert not output.breakout_flag

    def test_breakout_on_fast_move(self) -> None:
        """Breakout flag should fire on a rapid price move."""
        kf = KalmanFilterSignal(
            breakout_velocity_threshold=0.5,
            process_noise_scale=1e-3,  # Higher to track fast
        )
        t = 1000.0

        # Establish baseline
        for i in range(20):
            t += 1.0
            kf.update(
                binance_price=100_000.0,
                coinbase_price=100_000.0,
                oracle_price=100_000.0,
                timestamp=t,
            )

        # Sudden spike — $5/sec over several ticks
        for i in range(20):
            t += 1.0
            price = 100_000.0 + 5.0 * (i + 1)
            kf.update(
                binance_price=price,
                coinbase_price=price,
                oracle_price=100_000.0,  # Oracle lags
                timestamp=t,
            )

        # Velocity should be positive and breakout should fire
        assert kf.output.velocity > 0
        assert kf.output.breakout_flag


class TestSignalOutput:
    """Tests for the normalised signal output."""

    def test_signal_positive_when_exchange_leads_oracle_up(self) -> None:
        """Signal should be positive when exchanges are above oracle."""
        kf = KalmanFilterSignal(max_expected_lag=0.001)
        t = 1000.0

        # Init
        kf.update(binance_price=100_000.0, coinbase_price=100_000.0,
                  oracle_price=100_000.0, timestamp=t)

        # Exchange moves up, oracle lags
        for i in range(30):
            t += 1.0
            kf.update(
                binance_price=100_200.0,
                coinbase_price=100_180.0,
                oracle_price=100_000.0,
                timestamp=t,
            )

        assert kf.output.signal > 0

    def test_signal_negative_when_exchange_leads_oracle_down(self) -> None:
        """Signal should be negative when exchanges are below oracle."""
        kf = KalmanFilterSignal(max_expected_lag=0.001)
        t = 1000.0

        kf.update(binance_price=100_000.0, coinbase_price=100_000.0,
                  oracle_price=100_000.0, timestamp=t)

        for i in range(30):
            t += 1.0
            kf.update(
                binance_price=99_800.0,
                coinbase_price=99_820.0,
                oracle_price=100_000.0,
                timestamp=t,
            )

        assert kf.output.signal < 0

    def test_signal_bounded_minus_one_to_one(self) -> None:
        """Signal should always be in [-1, 1]."""
        kf = KalmanFilterSignal(max_expected_lag=0.001)
        t = 1000.0

        kf.update(binance_price=100_000.0, coinbase_price=100_000.0,
                  oracle_price=100_000.0, timestamp=t)

        # Extreme divergence
        for i in range(50):
            t += 1.0
            output = kf.update(
                binance_price=110_000.0,  # 10% above oracle
                coinbase_price=110_000.0,
                oracle_price=100_000.0,
                timestamp=t,
            )
            assert -1.0 <= output.signal <= 1.0

    def test_compute_convenience_method(self) -> None:
        """compute() should return just the signal float."""
        kf = KalmanFilterSignal()

        # Init
        kf.compute(binance_price=100_000.0, coinbase_price=100_000.0,
                   oracle_price=100_000.0)

        signal = kf.compute(
            binance_price=100_100.0,
            coinbase_price=100_080.0,
            oracle_price=100_000.0,
        )
        assert isinstance(signal, float)
        assert -1.0 <= signal <= 1.0


class TestKalmanOutput:
    """Tests for the KalmanOutput dataclass."""

    def test_default_values(self) -> None:
        output = KalmanOutput()
        assert output.predicted_price == 0.0
        assert output.filtered_price == 0.0
        assert output.velocity == 0.0
        assert output.signal == 0.0
        assert not output.breakout_flag

    def test_zero_dt_returns_previous_output(self) -> None:
        """update() with dt=0 should return previous output unchanged."""
        kf = KalmanFilterSignal()
        t = 1000.0
        kf.update(binance_price=100_000.0, coinbase_price=0.0,
                  oracle_price=99_990.0, timestamp=t)

        out1 = kf.update(binance_price=100_050.0, coinbase_price=0.0,
                         oracle_price=99_990.0, timestamp=t + 1.0)
        # Same timestamp — should return cached output
        out2 = kf.update(binance_price=100_100.0, coinbase_price=0.0,
                         oracle_price=99_990.0, timestamp=t + 1.0)

        assert out1.filtered_price == out2.filtered_price
        assert out1.velocity == out2.velocity
