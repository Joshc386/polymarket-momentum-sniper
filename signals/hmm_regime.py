"""Hidden Markov Model regime detector — probabilistic market state inference.

Replaces the ATR-based RegimeDetector with a proper HMM that infers
hidden market regimes from observable features. Instead of hard thresholds
on ATR/ADX/choppiness, the HMM learns transition probabilities between
states and uses Gaussian emission models to determine the most likely
current regime.

Model:
    Hidden states: 5 regimes (trending_up, trending_down, ranging,
                              high_vol, low_vol)

    Observable features (computed from 1-min candles):
        - Return:          (close - open) / open  (signed, captures direction)
        - Volatility:      ATR / close  (normalised true range)
        - Directional:     (close - open) / (high - low)  (body ratio, captures conviction)

    Emission model: Multivariate Gaussian per state
        P(obs | state) = N(obs; mu_state, sigma_state)

    Transition model: 5x5 matrix A[i,j] = P(state_j | state_i)
        - High self-transition probability (regimes are sticky)
        - Learned adaptively from recent observations via Bayesian updates

    Inference: Forward algorithm (online filtering)
        alpha_t(j) = P(obs_1:t, state_t=j)
        Updated each tick as new candle data arrives.

Advantages over ATR-based detector:
    1. Probabilistic: outputs confidence per state, not just the winner
    2. Temporal: uses transition probabilities (regimes are sticky)
    3. Adaptive: emission parameters update from recent observations
    4. Smoother: less regime-flipping in ambiguous conditions
"""

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from strategy.regime_detector import Regime, RegimeState, REGIME_PARAMS

logger = logging.getLogger(__name__)

# Map HMM state indices to Regime enum
STATE_INDEX = {
    0: Regime.TRENDING_UP,
    1: Regime.TRENDING_DOWN,
    2: Regime.RANGING,
    3: Regime.HIGH_VOLATILITY,
    4: Regime.LOW_VOLATILITY,
}

REGIME_INDEX = {v: k for k, v in STATE_INDEX.items()}

N_STATES = 5
N_FEATURES = 3  # return, volatility, directional


@dataclass
class HmmState:
    """Internal state of the HMM filter."""

    alpha: np.ndarray = field(
        default_factory=lambda: np.ones(N_STATES) / N_STATES
    )
    """Forward probabilities (belief over hidden states)."""


class HmmRegimeDetector:
    """Hidden Markov Model for market regime detection.

    Uses online forward algorithm to maintain a belief distribution
    over 5 market regimes, updated each time new candle data arrives.

    Args:
        transition_stickiness: Self-transition probability (how sticky
            regimes are). Higher = slower to switch. Default 0.90.
        adaptation_rate: How fast emission parameters adapt to new
            observations. 0 = fixed, 1 = instant. Default 0.02.
        min_candles: Minimum candles required before producing a regime
            estimate. Default 16.
        feature_lookback: Number of candles to compute features from
            for a single observation. Default 5.
        emission_floor: Minimum emission probability to prevent
            numerical underflow. Default 1e-10.
    """

    def __init__(
        self,
        transition_stickiness: float = 0.90,
        adaptation_rate: float = 0.02,
        min_candles: int = 16,
        feature_lookback: int = 5,
        emission_floor: float = 1e-10,
    ) -> None:
        self._stickiness = transition_stickiness
        self._adaptation_rate = adaptation_rate
        self._min_candles = min_candles
        self._feature_lookback = feature_lookback
        self._emission_floor = emission_floor

        # Transition matrix: A[i,j] = P(state_j at t+1 | state_i at t)
        self._A = self._init_transition_matrix(transition_stickiness)

        # Emission parameters: mean and covariance per state
        # Features: [return, volatility, directional]
        self._mu, self._sigma = self._init_emission_params()

        # Inverse and determinant cache (updated when sigma changes)
        self._sigma_inv = [np.linalg.inv(s) for s in self._sigma]
        self._sigma_logdet = [np.linalg.slogdet(s)[1] for s in self._sigma]

        # Forward state
        self._state = HmmState()
        self._observation_count = 0

        # Feature history for adaptive updates
        self._feature_history: deque[np.ndarray] = deque(maxlen=200)

        # Last result cache
        self._last_regime = RegimeState(
            regime=Regime.RANGING,
            confidence=0.0,
            trend_strength=0.0,
            volatility_pct=0.5,
            choppiness=0.5,
        )

    def detect(self, candles) -> RegimeState:
        """Detect current market regime using HMM forward algorithm.

        Drop-in replacement for RegimeDetector.detect().

        Args:
            candles: List/deque of Candle objects (1-min).

        Returns:
            RegimeState with detected regime, confidence, and metrics.
        """
        candle_list = list(candles)
        if len(candle_list) < self._min_candles:
            return self._last_regime

        # Extract observable features from recent candles
        obs = self._extract_features(candle_list)
        if obs is None:
            return self._last_regime

        # ── Forward algorithm step ──
        # Predict: propagate through transition matrix
        predicted = self._A.T @ self._state.alpha

        # Update: multiply by emission probability
        emissions = self._compute_emissions(obs)
        updated = predicted * emissions

        # Normalise
        total = updated.sum()
        if total > 0:
            updated /= total
        else:
            # Numerical issue — reset to uniform
            updated = np.ones(N_STATES) / N_STATES

        self._state.alpha = updated
        self._observation_count += 1

        # ── Adapt emission parameters ──
        self._feature_history.append(obs)
        if self._observation_count % 10 == 0 and len(self._feature_history) >= 50:
            self._adapt_emissions()

        # ── Extract regime and metrics ──
        best_state = int(np.argmax(updated))
        regime = STATE_INDEX[best_state]
        confidence = float(updated[best_state])

        # Compute interpretable metrics for RegimeState compatibility
        trend_strength = self._compute_trend_strength(candle_list)
        volatility_pct = self._compute_volatility_percentile(candle_list)
        choppiness = self._compute_choppiness(candle_list)

        state = RegimeState(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_pct=volatility_pct,
            choppiness=choppiness,
        )
        self._last_regime = state
        return state

    def get_params(self, regime: Regime) -> dict:
        """Get parameter overrides for the given regime.

        Uses the same REGIME_PARAMS as the baseline detector.
        """
        return REGIME_PARAMS.get(regime, REGIME_PARAMS[Regime.RANGING])

    def get_state_probabilities(self) -> dict[str, float]:
        """Return current belief distribution over all regimes.

        Useful for dashboard display — shows how certain the HMM is.

        Returns:
            Dict mapping regime name to probability.
        """
        return {
            STATE_INDEX[i].value: float(self._state.alpha[i])
            for i in range(N_STATES)
        }

    def reset(self) -> None:
        """Reset HMM state to uniform prior."""
        self._state = HmmState()
        self._observation_count = 0
        self._feature_history.clear()

    # ── Initialisation ────────────────────────────────────────────────

    @staticmethod
    def _init_transition_matrix(stickiness: float) -> np.ndarray:
        """Build transition matrix with high self-transition probability.

        Off-diagonal transitions are split equally among other states,
        with some structure: trending states can transition to each other
        more easily than jumping straight to low_vol.
        """
        off_diag = (1.0 - stickiness) / (N_STATES - 1)
        A = np.full((N_STATES, N_STATES), off_diag)
        np.fill_diagonal(A, stickiness)

        # Slight structural bias: trending states transition to ranging
        # more easily than to low_vol
        # Index: 0=trend_up, 1=trend_down, 2=ranging, 3=high_vol, 4=low_vol
        # Trending → ranging is more likely than trending → low_vol
        boost = off_diag * 0.3
        A[0, 2] += boost   # trend_up → ranging
        A[1, 2] += boost   # trend_down → ranging
        A[0, 4] -= boost   # trend_up → low_vol (less likely)
        A[1, 4] -= boost   # trend_down → low_vol (less likely)

        # Normalise rows
        A /= A.sum(axis=1, keepdims=True)
        return A

    @staticmethod
    def _init_emission_params() -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Initialise emission means and covariances per state.

        Features: [return, volatility, directional]
            return:      (close - open) / open  (typically ±0.001 for 5 candles)
            volatility:  ATR / close  (typically 0.0001 to 0.005)
            directional: body_ratio  (-1 to 1, sign indicates direction)

        Each state has distinct emission characteristics:
            trending_up:   positive return, moderate vol, positive directional
            trending_down: negative return, moderate vol, negative directional
            ranging:       near-zero return, low-moderate vol, near-zero directional
            high_vol:      any return direction, high vol, any directional
            low_vol:       near-zero return, very low vol, near-zero directional
        """
        mu = [
            np.array([+0.0008, 0.0015, +0.40]),   # trending_up
            np.array([-0.0008, 0.0015, -0.40]),   # trending_down
            np.array([0.0000, 0.0008, 0.00]),      # ranging
            np.array([0.0000, 0.0035, 0.00]),      # high_vol
            np.array([0.0000, 0.0002, 0.00]),      # low_vol
        ]

        # Covariance matrices (diagonal for simplicity, can be adapted)
        base_var = np.array([1e-6, 5e-7, 0.08])

        sigma = [
            np.diag(base_var * np.array([1.0, 1.0, 1.0])),  # trending_up
            np.diag(base_var * np.array([1.0, 1.0, 1.0])),  # trending_down
            np.diag(base_var * np.array([2.0, 1.5, 2.0])),  # ranging (wider)
            np.diag(base_var * np.array([3.0, 4.0, 3.0])),  # high_vol (widest)
            np.diag(base_var * np.array([0.5, 0.3, 1.0])),  # low_vol (tight)
        ]

        return mu, sigma

    # ── Feature extraction ────────────────────────────────────────────

    def _extract_features(self, candles: list) -> np.ndarray | None:
        """Extract 3D observation vector from recent candles.

        Uses the last `feature_lookback` candles to compute:
            [return, volatility, directional]

        Returns:
            3D numpy array, or None if insufficient data.
        """
        n = self._feature_lookback
        recent = candles[-n:]
        if len(recent) < n:
            return None

        # Return: aggregate over lookback
        first_open = recent[0].open
        last_close = recent[-1].close
        if first_open <= 0:
            return None
        ret = (last_close - first_open) / first_open

        # Volatility: average true range / close
        tr_values = []
        for i in range(1, len(recent)):
            high = recent[i].high
            low = recent[i].low
            prev_close = recent[i - 1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        atr = sum(tr_values) / len(tr_values) if tr_values else 0.0
        vol = atr / last_close if last_close > 0 else 0.0

        # Directional: average body ratio (signed)
        body_ratios = []
        for c in recent:
            hl_range = c.high - c.low
            if hl_range > 0:
                body_ratios.append((c.close - c.open) / hl_range)
            else:
                body_ratios.append(0.0)
        directional = sum(body_ratios) / len(body_ratios)

        return np.array([ret, vol, directional])

    # ── Emission computation ──────────────────────────────────────────

    def _compute_emissions(self, obs: np.ndarray) -> np.ndarray:
        """Compute emission probability for each state given observation.

        Uses multivariate Gaussian PDF:
            P(obs | state_k) = N(obs; mu_k, sigma_k)

        Returns:
            Array of shape (N_STATES,) with emission probabilities.
        """
        emissions = np.zeros(N_STATES)

        for k in range(N_STATES):
            diff = obs - self._mu[k]
            # log P = -0.5 * (d*ln(2pi) + logdet(sigma) + diff^T sigma^-1 diff)
            mahal = diff @ self._sigma_inv[k] @ diff
            log_p = -0.5 * (
                N_FEATURES * np.log(2 * np.pi)
                + self._sigma_logdet[k]
                + mahal
            )
            emissions[k] = max(np.exp(log_p), self._emission_floor)

        return emissions

    # ── Adaptive emission updates ─────────────────────────────────────

    def _adapt_emissions(self) -> None:
        """Adapt emission parameters using soft state assignments.

        Uses the current belief distribution to weight recent observations,
        then updates emission means and covariances with exponential moving
        average. This is a simplified online EM step.
        """
        rate = self._adaptation_rate
        if rate <= 0:
            return

        # Use recent observations weighted by state beliefs
        recent_obs = list(self._feature_history)[-50:]
        if len(recent_obs) < 20:
            return

        obs_matrix = np.array(recent_obs)  # (T, N_FEATURES)

        for k in range(N_STATES):
            # Compute responsibilities (how much state k explains each obs)
            responsibilities = np.zeros(len(obs_matrix))
            for t, obs in enumerate(obs_matrix):
                emissions = self._compute_emissions(obs)
                total = emissions.sum()
                if total > 0:
                    responsibilities[t] = emissions[k] / total

            resp_sum = responsibilities.sum()
            if resp_sum < 1.0:
                continue

            # Weighted mean
            weighted_obs = (responsibilities[:, None] * obs_matrix).sum(axis=0)
            new_mu = weighted_obs / resp_sum

            # Weighted covariance
            diff = obs_matrix - new_mu
            weighted_diff = responsibilities[:, None] * diff
            new_sigma = (weighted_diff.T @ diff) / resp_sum

            # Ensure positive definite by adding small diagonal
            new_sigma += np.eye(N_FEATURES) * 1e-8

            # EMA update
            self._mu[k] = (1 - rate) * self._mu[k] + rate * new_mu
            self._sigma[k] = (1 - rate) * self._sigma[k] + rate * new_sigma

        # Recompute cached inverses and determinants
        self._sigma_inv = [np.linalg.inv(s) for s in self._sigma]
        self._sigma_logdet = [np.linalg.slogdet(s)[1] for s in self._sigma]

    # ── Compatibility metrics ─────────────────────────────────────────
    # These compute the same metrics as RegimeDetector for RegimeState
    # compatibility, but the HMM doesn't use them for classification —
    # they're just for dashboard display.

    def _compute_trend_strength(self, candles: list) -> float:
        """Compute ADX-like trend strength for RegimeState display."""
        period = 14
        recent = candles[-(period + 1):]
        if len(recent) < period + 1:
            return 0.0

        plus_dm_sum = 0.0
        minus_dm_sum = 0.0
        tr_sum = 0.0

        for i in range(1, len(recent)):
            high = recent[i].high
            low = recent[i].low
            prev_high = recent[i - 1].high
            prev_low = recent[i - 1].low
            prev_close = recent[i - 1].close

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_sum += tr

            up_move = high - prev_high
            down_move = prev_low - low

            if up_move > down_move and up_move > 0:
                plus_dm_sum += up_move
            if down_move > up_move and down_move > 0:
                minus_dm_sum += down_move

        if tr_sum == 0:
            return 0.0

        plus_di = plus_dm_sum / tr_sum
        minus_di = minus_dm_sum / tr_sum
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0

        return min(1.0, abs(plus_di - minus_di) / di_sum)

    def _compute_volatility_percentile(self, candles: list) -> float:
        """Compute ATR percentile for RegimeState display."""
        if len(candles) < 30:
            return 0.5

        atr_values = []
        for i in range(14, len(candles)):
            window = candles[i - 14:i + 1]
            tr_sum = 0.0
            for j in range(1, len(window)):
                tr = max(
                    window[j].high - window[j].low,
                    abs(window[j].high - window[j - 1].close),
                    abs(window[j].low - window[j - 1].close),
                )
                tr_sum += tr
            atr_values.append(tr_sum / 14)

        if len(atr_values) < 2:
            return 0.5

        current = atr_values[-1]
        rank = sum(1 for x in atr_values if x <= current)
        return rank / len(atr_values)

    def _compute_choppiness(self, candles: list) -> float:
        """Compute direction flip rate for RegimeState display."""
        recent = candles[-15:]
        if len(recent) < 3:
            return 0.5

        flips = 0
        for i in range(1, len(recent)):
            prev_dir = 1 if recent[i - 1].close >= recent[i - 1].open else -1
            curr_dir = 1 if recent[i].close >= recent[i].open else -1
            if curr_dir != prev_dir:
                flips += 1

        return flips / (len(recent) - 1)
