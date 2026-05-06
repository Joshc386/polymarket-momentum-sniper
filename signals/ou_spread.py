"""Ornstein-Uhlenbeck mean reversion signal.

Models the Binance-Oracle price spread as a mean-reverting OU process:
    dX = θ(μ - X)dt + σdW

where:
    X  = current spread (binance - oracle)
    θ  = mean reversion speed (higher = faster snap-back)
    μ  = long-run equilibrium spread
    σ  = volatility of the spread

When the spread exceeds 2σ from equilibrium, we expect reversion — the
oracle will catch up. This produces a directional signal for the binary
market: if Binance is above Oracle (positive spread), oracle will rise,
meaning price likely goes UP at resolution.

Parameters are calibrated online using maximum likelihood estimation
on a rolling window of spread observations.

Research basis: OU process on cross-source spreads captures the
structural mean-reversion of oracle price discovery. Academic work
shows MLE with noise correction significantly improves profitability
over naive OLS calibration.
"""

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class OUSpreadSignal:
    """Directional signal from OU-modeled Binance-Oracle spread.

    Calibrates OU parameters online and fires when the spread is
    far enough from equilibrium to expect profitable reversion.

    Args:
        calibration_window: Number of ticks for parameter estimation.
        min_observations: Minimum ticks before signal is active.
        entry_z_threshold: Z-score threshold to generate a signal.
        dt: Time step between observations in seconds.
    """

    calibration_window: int = 300     # ~5 minutes at 1 tick/sec
    min_observations: int = 60        # Need 1 minute of data minimum
    entry_z_threshold: float = 1.5    # Enter when spread exceeds 1.5σ
    dt: float = 1.0                   # 1 second between ticks

    def __post_init__(self) -> None:
        self._spread_history: deque[float] = deque(maxlen=self.calibration_window)

        # OU parameters (calibrated online)
        self._theta: float = 0.0    # Mean reversion speed
        self._mu: float = 0.0       # Long-run mean
        self._sigma: float = 0.0    # Volatility

        # Output state
        self._last_signal: float = 0.0
        self._last_z_score: float = 0.0
        self._last_half_life: float = 0.0

    def compute(
        self,
        binance_price: float,
        oracle_price: float,
    ) -> float:
        """Compute OU mean reversion signal in [-1.0, 1.0].

        Args:
            binance_price: Current Binance BTC spot price.
            oracle_price: Current Chainlink oracle price.

        Returns:
            Signal in [-1.0, 1.0].
            Positive = spread is positive and expected to revert (bullish
                for oracle catch-up → UP resolution).
            Negative = spread is negative and expected to revert (bearish
                for oracle catch-up → DOWN resolution).
            0.0 if insufficient data or spread within normal range.
        """
        if binance_price <= 0 or oracle_price <= 0:
            return 0.0

        spread = binance_price - oracle_price
        self._spread_history.append(spread)

        if len(self._spread_history) < self.min_observations:
            return 0.0

        # Recalibrate OU parameters every 30 ticks
        if len(self._spread_history) % 30 == 0 or self._theta == 0:
            self._calibrate()

        if self._sigma <= 0 or self._theta <= 0:
            return 0.0

        # Z-score: how far is current spread from equilibrium in σ units
        deviation = spread - self._mu
        z_score = deviation / self._sigma
        self._last_z_score = z_score

        # Half-life of mean reversion (in seconds)
        self._last_half_life = math.log(2) / self._theta if self._theta > 0 else float("inf")

        # Signal generation:
        # If spread > μ + threshold*σ → oracle will catch up → price goes UP → positive signal
        # If spread < μ - threshold*σ → oracle will catch down → price goes DOWN → negative signal
        if abs(z_score) < self.entry_z_threshold:
            self._last_signal = 0.0
            return 0.0

        # Ramp from 0 at threshold to ±1.0 at 3*threshold
        excess = abs(z_score) - self.entry_z_threshold
        max_excess = 2.0 * self.entry_z_threshold  # At 3x threshold → signal = 1.0
        ramp = min(1.0, excess / max_excess)

        # Positive spread (Binance > Oracle) → oracle will rise → UP
        signal = math.copysign(ramp, z_score)

        # Confidence adjustment: faster reversion (higher θ) = more confident
        # Half-life < 10s is very fast (high confidence)
        # Half-life > 60s is slow (low confidence, maybe not mean-reverting)
        if self._last_half_life > 60:
            signal *= 0.5  # Low confidence — slow reversion
        elif self._last_half_life < 10:
            signal *= 1.0  # Full confidence — fast reversion

        self._last_signal = max(-1.0, min(1.0, signal))
        return self._last_signal

    def _calibrate(self) -> None:
        """Calibrate OU parameters using MLE on the spread history.

        Uses the discrete-time OU MLE estimator:
            X_{t+1} = a + b*X_t + ε
            where b = exp(-θ*dt)
            θ = -ln(b) / dt
            μ = a / (1 - b)
            σ² = var(ε) * 2θ / (1 - b²)
        """
        spreads = list(self._spread_history)
        n = len(spreads)
        if n < self.min_observations:
            return

        # OLS regression: X_{t+1} = a + b * X_t
        # Using the standard formulas
        x = spreads[:-1]   # X_t
        y = spreads[1:]    # X_{t+1}
        n_pairs = len(x)

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        denom = n_pairs * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-20:
            return

        b = (n_pairs * sum_xy - sum_x * sum_y) / denom
        a = (sum_y - b * sum_x) / n_pairs

        # Clamp b to (0, 1) for mean-reverting process
        # b >= 1 means no mean reversion (random walk or explosive)
        # b <= 0 means oscillatory (unlikely for price spread)
        if b >= 1.0 or b <= 0.0:
            # Not mean-reverting — set theta to 0, signal stays 0
            self._theta = 0.0
            self._mu = sum(spreads) / n  # Use simple mean as fallback
            self._sigma = max(1e-6, self._sample_std(spreads))
            return

        # Extract OU parameters
        self._theta = -math.log(b) / self.dt
        self._mu = a / (1.0 - b)

        # Residual variance
        residuals = [yi - (a + b * xi) for xi, yi in zip(x, y)]
        var_resid = sum(r * r for r in residuals) / max(1, n_pairs - 2)

        # OU volatility: σ² = var(ε) * 2θ / (1 - b²)
        b2 = b * b
        if (1.0 - b2) > 1e-10 and self._theta > 0:
            ou_var = var_resid * 2.0 * self._theta / (1.0 - b2)
            self._sigma = math.sqrt(max(1e-10, ou_var))
        else:
            self._sigma = math.sqrt(max(1e-10, var_resid))

    @staticmethod
    def _sample_std(values: list[float]) -> float:
        """Compute sample standard deviation."""
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        return math.sqrt(var)

    def reset(self) -> None:
        """Clear all state."""
        self._spread_history.clear()
        self._theta = 0.0
        self._mu = 0.0
        self._sigma = 0.0
        self._last_signal = 0.0
        self._last_z_score = 0.0
        self._last_half_life = 0.0

    @property
    def theta(self) -> float:
        """Mean reversion speed parameter."""
        return self._theta

    @property
    def mu(self) -> float:
        """Long-run equilibrium spread."""
        return self._mu

    @property
    def sigma(self) -> float:
        """Spread volatility."""
        return self._sigma

    @property
    def z_score(self) -> float:
        """Current z-score of the spread."""
        return self._last_z_score

    @property
    def half_life(self) -> float:
        """Mean reversion half-life in seconds."""
        return self._last_half_life
