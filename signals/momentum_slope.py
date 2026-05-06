"""Multi-timeframe momentum slope signal.

Computes weighted linear regression slopes across 4 timeframes
(30s, 60s, 120s, 240s) using 1-minute candle close prices. A more
statistically grounded momentum measure than ROC/direction heuristics.

Research basis: Weighted multi-timeframe slope outperforms single-window
ROC for short-term crypto direction prediction. Longer timeframes provide
trend context; shorter timeframes provide entry timing.
"""

import math
from collections import deque
from dataclasses import dataclass


@dataclass
class MomentumSlopeSignal:
    """Layer 2 replacement: Multi-timeframe linear regression momentum.

    Computes normalised slope of price over 4 windows, weighted by
    recency. Output is in [-1, 1] like all other signals.

    Args:
        weights: Tuple of (30s, 60s, 120s, 240s) weights. Default
            emphasises shorter timeframes for entry timing.
        slope_normaliser: Dollar move per second that maps to ±1.0.
            BTC at ~$85K moving $50/sec would be extreme. Default
            $10/sec = ±1.0.
    """

    weight_30s: float = 0.35
    weight_60s: float = 0.30
    weight_120s: float = 0.20
    weight_240s: float = 0.15
    slope_normaliser: float = 10.0

    def compute(self, candles: deque, current_candle=None) -> float:
        """Compute multi-timeframe momentum slope in [-1.0, 1.0].

        Args:
            candles: Deque of closed 1-min Candle objects (most recent last).
            current_candle: The in-progress candle (may be None).

        Returns:
            Normalised momentum signal in [-1.0, 1.0].
            Positive = upward momentum, negative = downward.
        """
        # Build price series from candle closes + current
        prices = [c.close for c in candles if c.close > 0]
        if current_candle and current_candle.close > 0:
            prices.append(current_candle.close)

        if len(prices) < 2:
            return 0.0

        # Compute slopes over each timeframe
        # Each candle is ~60 seconds. We use the last N prices for each window.
        slopes = []
        timeframes = [
            (self.weight_30s, 1),    # ~30s: last 1 candle (current partial + prev)
            (self.weight_60s, 1),    # ~60s: last 1 full candle
            (self.weight_120s, 2),   # ~120s: last 2 candles
            (self.weight_240s, 4),   # ~240s: last 4 candles
        ]

        for weight, n_candles in timeframes:
            if len(prices) < n_candles + 1:
                slopes.append((weight, 0.0))
                continue

            window = prices[-(n_candles + 1):]
            slope = self._linear_regression_slope(window)
            slopes.append((weight, slope))

        # Weighted combination
        total_weight = sum(w for w, _ in slopes)
        if total_weight <= 0:
            return 0.0

        combined_slope = sum(w * s for w, s in slopes) / total_weight

        # Normalise to [-1, 1]
        # slope is in $/candle (~$/60s). Convert to $/sec then normalise.
        slope_per_sec = combined_slope / 60.0
        normalised = slope_per_sec / self.slope_normaliser
        return max(-1.0, min(1.0, normalised))

    @staticmethod
    def _linear_regression_slope(prices: list[float]) -> float:
        """Compute OLS slope of price series.

        Args:
            prices: List of prices (evenly spaced in time).

        Returns:
            Slope in price-units per time-step ($/candle).
        """
        n = len(prices)
        if n < 2:
            return 0.0

        # Simple OLS: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0

        for i, price in enumerate(prices):
            x = float(i)
            sum_x += x
            sum_y += price
            sum_xy += x * price
            sum_x2 += x * x

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-12:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denom
