"""Cross-exchange divergence signal.

Measures the z-score of the Binance-Coinbase price spread against its
rolling distribution. When the spread exceeds 2σ, one exchange is
leading — this is a high-conviction directional signal.

Research basis: Cross-exchange spreads in crypto are larger and more
persistent than in traditional markets due to fragmented liquidity.
Divergence magnitude (not just direction) predicts short-term price
convergence.
"""

import math
from collections import deque
from dataclasses import dataclass


@dataclass
class CrossExchangeSignal:
    """Directional signal from Binance-Coinbase price divergence.

    Tracks the rolling mean and std of the spread. When the current
    spread exceeds z_threshold standard deviations, it outputs a
    strong directional signal (Binance leads price discovery).

    Args:
        rolling_window: Number of ticks to compute rolling stats over.
        z_threshold: Z-score threshold for "significant" divergence.
        max_spread_usd: Cap spread at this value to avoid outlier
            contamination (e.g. during exchange outages).
    """

    rolling_window: int = 120    # ~2 minutes at 1 tick/sec
    z_threshold: float = 1.5     # Lower than 2.0 — crypto spreads are noisy
    max_spread_usd: float = 200.0

    def __post_init__(self) -> None:
        self._spread_history: deque[float] = deque(maxlen=self.rolling_window)
        self._last_signal: float = 0.0

    def compute(
        self,
        binance_price: float,
        coinbase_price: float,
        coinbase_connected: bool = True,
    ) -> float:
        """Compute cross-exchange divergence signal in [-1.0, 1.0].

        Args:
            binance_price: Current Binance BTC spot price.
            coinbase_price: Current Coinbase BTC spot price.
            coinbase_connected: Whether Coinbase feed is live.

        Returns:
            Signal in [-1.0, 1.0]. Positive = Binance leading up
            (bullish), negative = Binance leading down (bearish).
            Returns 0.0 if Coinbase is disconnected or insufficient data.
        """
        if not coinbase_connected or coinbase_price <= 0 or binance_price <= 0:
            return 0.0

        # Raw spread: positive = Binance higher than Coinbase
        spread = binance_price - coinbase_price

        # Cap outliers
        spread = max(-self.max_spread_usd, min(self.max_spread_usd, spread))

        self._spread_history.append(spread)

        # Need enough history for meaningful stats
        if len(self._spread_history) < 30:
            return 0.0

        # Rolling mean and std
        n = len(self._spread_history)
        mean = sum(self._spread_history) / n
        variance = sum((s - mean) ** 2 for s in self._spread_history) / n
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std < 0.01:
            # Spread is flat — no signal
            return 0.0

        # Z-score of current spread
        z_score = (spread - mean) / std

        # Convert to signal: ramp from 0 at z_threshold to ±1.0 at 2*z_threshold
        if abs(z_score) < self.z_threshold:
            signal = 0.0
        else:
            # Linear ramp: z_threshold → 0.0, 2*z_threshold → ±1.0
            excess = abs(z_score) - self.z_threshold
            ramp = min(1.0, excess / self.z_threshold)
            signal = math.copysign(ramp, z_score)

        self._last_signal = signal
        return signal

    def reset(self) -> None:
        """Clear rolling history."""
        self._spread_history.clear()
        self._last_signal = 0.0

    @property
    def current_z_score(self) -> float:
        """Current z-score for dashboard display."""
        if len(self._spread_history) < 30:
            return 0.0
        n = len(self._spread_history)
        mean = sum(self._spread_history) / n
        variance = sum((s - mean) ** 2 for s in self._spread_history) / n
        std = math.sqrt(variance) if variance > 0 else 1.0
        spread = self._spread_history[-1]
        return (spread - mean) / std
