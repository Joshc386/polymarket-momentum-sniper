"""Tests for the HMM regime detector.

Validates initialisation, forward algorithm, regime classification,
adaptive emissions, state probability output, and RegimeState
compatibility.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from signals.hmm_regime import (
    HmmRegimeDetector,
    HmmState,
    N_FEATURES,
    N_STATES,
    STATE_INDEX,
)
from strategy.regime_detector import Regime, RegimeState, REGIME_PARAMS


@dataclass
class FakeCandle:
    """Minimal candle for testing."""
    open: float
    high: float
    low: float
    close: float
    volume: float = 100.0


def make_trending_up_candles(n: int = 30, base: float = 100_000.0) -> list:
    """Generate candles with a clear uptrend."""
    candles = []
    price = base
    for i in range(n):
        o = price
        c = price + 20 + np.random.normal(0, 3)
        h = max(o, c) + abs(np.random.normal(0, 5))
        l = min(o, c) - abs(np.random.normal(0, 3))
        candles.append(FakeCandle(open=o, high=h, low=l, close=c))
        price = c
    return candles


def make_trending_down_candles(n: int = 30, base: float = 100_000.0) -> list:
    """Generate candles with a clear downtrend."""
    candles = []
    price = base
    for i in range(n):
        o = price
        c = price - 20 - np.random.normal(0, 3)
        h = max(o, c) + abs(np.random.normal(0, 3))
        l = min(o, c) - abs(np.random.normal(0, 5))
        candles.append(FakeCandle(open=o, high=h, low=l, close=c))
        price = c
    return candles


def make_ranging_candles(n: int = 30, base: float = 100_000.0) -> list:
    """Generate candles that oscillate around a mean."""
    candles = []
    for i in range(n):
        mid = base + np.sin(i * 0.5) * 10
        o = mid + np.random.normal(0, 3)
        c = mid + np.random.normal(0, 3)
        h = max(o, c) + abs(np.random.normal(0, 2))
        l = min(o, c) - abs(np.random.normal(0, 2))
        candles.append(FakeCandle(open=o, high=h, low=l, close=c))
    return candles


def make_high_vol_candles(n: int = 30, base: float = 100_000.0) -> list:
    """Generate candles with extreme volatility."""
    candles = []
    price = base
    for i in range(n):
        o = price
        c = price + np.random.normal(0, 200)
        h = max(o, c) + abs(np.random.normal(0, 100))
        l = min(o, c) - abs(np.random.normal(0, 100))
        candles.append(FakeCandle(open=o, high=h, low=l, close=c))
        price = c
    return candles


def make_low_vol_candles(n: int = 30, base: float = 100_000.0) -> list:
    """Generate candles with very low volatility."""
    candles = []
    for i in range(n):
        o = base + np.random.normal(0, 0.5)
        c = o + np.random.normal(0, 0.3)
        h = max(o, c) + abs(np.random.normal(0, 0.2))
        l = min(o, c) - abs(np.random.normal(0, 0.2))
        candles.append(FakeCandle(open=o, high=h, low=l, close=c))
    return candles


class TestHmmInitialisation:
    """Tests for HMM initialisation and state."""

    def test_initial_state_is_uniform(self) -> None:
        hmm = HmmRegimeDetector()
        probs = hmm.get_state_probabilities()
        for regime_name, prob in probs.items():
            assert abs(prob - 1.0 / N_STATES) < 1e-6

    def test_transition_matrix_rows_sum_to_one(self) -> None:
        hmm = HmmRegimeDetector()
        for i in range(N_STATES):
            assert abs(hmm._A[i].sum() - 1.0) < 1e-10

    def test_transition_matrix_stickiness(self) -> None:
        stickiness = 0.92
        hmm = HmmRegimeDetector(transition_stickiness=stickiness)
        for i in range(N_STATES):
            # Self-transition should be close to stickiness
            # (structural biases adjust it slightly)
            assert hmm._A[i, i] >= stickiness * 0.9

    def test_emission_params_shapes(self) -> None:
        hmm = HmmRegimeDetector()
        assert len(hmm._mu) == N_STATES
        assert len(hmm._sigma) == N_STATES
        for k in range(N_STATES):
            assert hmm._mu[k].shape == (N_FEATURES,)
            assert hmm._sigma[k].shape == (N_FEATURES, N_FEATURES)

    def test_reset_returns_to_uniform(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector()
        # Feed some data to change state
        candles = make_trending_up_candles(30)
        hmm.detect(candles)

        hmm.reset()
        probs = hmm.get_state_probabilities()
        for prob in probs.values():
            assert abs(prob - 1.0 / N_STATES) < 1e-6


class TestHmmRegimeDetection:
    """Tests for regime classification accuracy."""

    def test_returns_regime_state(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector()
        candles = make_trending_up_candles(30)
        result = hmm.detect(candles)
        assert isinstance(result, RegimeState)
        assert isinstance(result.regime, Regime)
        assert 0.0 <= result.confidence <= 1.0

    def test_insufficient_candles_returns_default(self) -> None:
        hmm = HmmRegimeDetector(min_candles=16)
        candles = make_trending_up_candles(5)
        result = hmm.detect(candles)
        # Should return default (ranging)
        assert result.regime == Regime.RANGING

    def test_detects_trending_up(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector()
        candles = make_trending_up_candles(50)
        # Feed candles incrementally to build belief
        for i in range(20, len(candles)):
            result = hmm.detect(candles[:i + 1])
        assert result.regime == Regime.TRENDING_UP

    def test_detects_trending_down(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector()
        candles = make_trending_down_candles(50)
        for i in range(20, len(candles)):
            result = hmm.detect(candles[:i + 1])
        assert result.regime == Regime.TRENDING_DOWN

    def test_detects_high_volatility(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector()
        candles = make_high_vol_candles(50)
        for i in range(20, len(candles)):
            result = hmm.detect(candles[:i + 1])
        assert result.regime == Regime.HIGH_VOLATILITY

    def test_detects_low_volatility(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector()
        candles = make_low_vol_candles(50)
        for i in range(20, len(candles)):
            result = hmm.detect(candles[:i + 1])
        assert result.regime == Regime.LOW_VOLATILITY


class TestHmmStateProbabilities:
    """Tests for the probability distribution output."""

    def test_probabilities_sum_to_one(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector()
        candles = make_trending_up_candles(30)
        hmm.detect(candles)
        probs = hmm.get_state_probabilities()
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_regimes_in_output(self) -> None:
        hmm = HmmRegimeDetector()
        candles = make_ranging_candles(30)
        hmm.detect(candles)
        probs = hmm.get_state_probabilities()
        expected_keys = {r.value for r in Regime}
        assert set(probs.keys()) == expected_keys

    def test_dominant_state_has_highest_probability(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector()
        candles = make_trending_up_candles(50)
        for i in range(20, len(candles)):
            hmm.detect(candles[:i + 1])
        probs = hmm.get_state_probabilities()
        # The detected regime should have the highest probability
        detected = hmm._last_regime.regime.value
        assert probs[detected] == max(probs.values())


class TestHmmGetParams:
    """Tests for regime parameter compatibility."""

    def test_returns_same_params_as_baseline(self) -> None:
        hmm = HmmRegimeDetector()
        for regime in Regime:
            params = hmm.get_params(regime)
            expected = REGIME_PARAMS[regime]
            assert params == expected

    def test_params_have_required_keys(self) -> None:
        hmm = HmmRegimeDetector()
        for regime in Regime:
            params = hmm.get_params(regime)
            assert "weight_adjustments" in params
            assert "edge_multiplier" in params
            assert "size_multiplier" in params


class TestHmmAdaptiveEmissions:
    """Tests for adaptive emission parameter updates."""

    def test_emissions_adapt_after_observations(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector(adaptation_rate=0.1)

        # Store initial emission means
        initial_mu = [m.copy() for m in hmm._mu]

        # Feed lots of trending up data (force adaptation)
        candles = make_trending_up_candles(100)
        for i in range(20, len(candles)):
            hmm.detect(candles[:i + 1])

        # At least one emission mean should have shifted
        changed = False
        for k in range(N_STATES):
            if not np.allclose(hmm._mu[k], initial_mu[k], atol=1e-8):
                changed = True
                break
        assert changed

    def test_no_adaptation_when_rate_zero(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector(adaptation_rate=0.0)

        initial_mu = [m.copy() for m in hmm._mu]

        candles = make_trending_up_candles(80)
        for i in range(20, len(candles)):
            hmm.detect(candles[:i + 1])

        for k in range(N_STATES):
            assert np.allclose(hmm._mu[k], initial_mu[k])


class TestHmmFeatureExtraction:
    """Tests for the feature extraction method."""

    def test_feature_vector_shape(self) -> None:
        hmm = HmmRegimeDetector(feature_lookback=5)
        candles = make_ranging_candles(10)
        features = hmm._extract_features(candles)
        assert features is not None
        assert features.shape == (N_FEATURES,)

    def test_returns_none_for_insufficient_candles(self) -> None:
        hmm = HmmRegimeDetector(feature_lookback=5)
        candles = make_ranging_candles(3)
        features = hmm._extract_features(candles)
        assert features is None

    def test_uptrend_has_positive_return(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector(feature_lookback=5)
        candles = make_trending_up_candles(10)
        features = hmm._extract_features(candles)
        assert features[0] > 0  # return should be positive

    def test_downtrend_has_negative_return(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector(feature_lookback=5)
        candles = make_trending_down_candles(10)
        features = hmm._extract_features(candles)
        assert features[0] < 0  # return should be negative

    def test_high_vol_has_higher_volatility_feature(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector(feature_lookback=5)
        low_vol_features = hmm._extract_features(make_low_vol_candles(10))
        high_vol_features = hmm._extract_features(make_high_vol_candles(10))
        # Volatility feature (index 1) should be higher for high vol
        assert high_vol_features[1] > low_vol_features[1]


class TestHmmRegimeStateCompatibility:
    """Verify RegimeState output is fully compatible with the baseline."""

    def test_label_property_works(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector()
        candles = make_ranging_candles(30)
        result = hmm.detect(candles)
        assert result.label == result.regime.value

    def test_str_method_works(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector()
        candles = make_ranging_candles(30)
        result = hmm.detect(candles)
        s = str(result)
        assert "conf=" in s
        assert "trend=" in s

    def test_all_metrics_in_range(self) -> None:
        np.random.seed(42)
        hmm = HmmRegimeDetector()
        candles = make_trending_up_candles(30)
        result = hmm.detect(candles)
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.trend_strength <= 1.0
        assert 0.0 <= result.volatility_pct <= 1.0
        assert 0.0 <= result.choppiness <= 1.0
