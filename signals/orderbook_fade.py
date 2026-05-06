"""Layer 6: Orderbook Fade Signal — contrarian bet against extreme imbalances.

When the Polymarket CLOB is heavily skewed (e.g. 70%+ of top-5 depth
on the bid side of YES), it often represents market-maker liquidity
provision rather than informed flow. The price tends to move AGAINST
the imbalance — a "fade" opportunity.

Validated on 51,423 PolyBackTest Pro snapshots:
  - Imbalance > 0.3 with 60-180s remaining: 61-63% fade WR
  - Imbalance > 0.4: 55-56% overall, 63% in sweet spot
  - Signal degrades below 60s remaining (market converges)
  - Signal degrades above 240s (book still forming)

Output is [-1.0, 1.0]:
  Positive = fade says UP (bearish book is extreme, expect reversal up)
  Negative = fade says DOWN (bullish book is extreme, expect reversal down)
  Zero = imbalance not extreme enough to trigger fade
"""

from dataclasses import dataclass, field
from collections import deque


@dataclass
class OrderbookFadeSignal:
    """Contrarian orderbook signal that fires on extreme imbalances.

    Unlike the standard L4 orderbook signal which follows imbalance,
    this signal bets AGAINST it when depth skew exceeds a threshold.

    Args:
        imbalance_threshold: Minimum |imbalance| to trigger (0.0-1.0).
            Lower = more signals but lower WR. Default 0.35 balances
            frequency vs accuracy.
        min_secs_remaining: Don't fire below this — market too efficient.
        max_secs_remaining: Don't fire above this — book still forming.
        smoothing_window: Number of ticks to average imbalance over.
            Prevents firing on single-tick spikes.
        magnitude_scale: How quickly signal strength ramps above threshold.
            Higher = more aggressive scaling.
    """

    imbalance_threshold: float = 0.35
    min_secs_remaining: float = 60.0
    max_secs_remaining: float = 200.0
    smoothing_window: int = 5
    magnitude_scale: float = 2.0

    # Internal state
    _imbalance_history: deque = field(default_factory=lambda: deque(maxlen=30))
    _last_signal: float = 0.0

    def reset(self) -> None:
        """Reset between windows."""
        self._imbalance_history.clear()
        self._last_signal = 0.0

    def compute(
        self,
        yes_bid_depth: float,
        yes_ask_depth: float,
        seconds_remaining: float,
    ) -> float:
        """Compute the fade signal from Polymarket YES-side depth.

        Args:
            yes_bid_depth: Total bid depth on YES side (top-5 levels).
            yes_ask_depth: Total ask depth on YES side (top-5 levels).
            seconds_remaining: Seconds left in the 5-minute window.

        Returns:
            Signal in [-1.0, 1.0]. Positive = fade predicts UP,
            negative = fade predicts DOWN, zero = no extreme detected.
        """
        total = yes_bid_depth + yes_ask_depth
        if total <= 0:
            self._last_signal = 0.0
            return 0.0

        # Raw imbalance: positive = bids dominate (market looks bullish)
        raw_imbalance = (yes_bid_depth - yes_ask_depth) / total
        self._imbalance_history.append(raw_imbalance)

        # Need enough history to smooth
        if len(self._imbalance_history) < min(self.smoothing_window, 3):
            self._last_signal = 0.0
            return 0.0

        # Smoothed imbalance (last N ticks)
        recent = list(self._imbalance_history)[-self.smoothing_window:]
        smoothed = sum(recent) / len(recent)

        # Time gate: only fire in the validated sweet spot
        if (seconds_remaining < self.min_secs_remaining
                or seconds_remaining > self.max_secs_remaining):
            self._last_signal = 0.0
            return 0.0

        # Check if imbalance is extreme enough
        abs_imb = abs(smoothed)
        if abs_imb < self.imbalance_threshold:
            self._last_signal = 0.0
            return 0.0

        # Scale signal magnitude: ramps from 0 at threshold to 1.0 at
        # threshold + (1/magnitude_scale). Clamped to [-1, 1].
        excess = abs_imb - self.imbalance_threshold
        magnitude = min(excess * self.magnitude_scale, 1.0)

        # FADE: signal is opposite to imbalance direction
        # Bullish book (positive imbalance) -> fade says DOWN (negative signal)
        # Bearish book (negative imbalance) -> fade says UP (positive signal)
        if smoothed > 0:
            signal = -magnitude  # Bullish book -> fade DOWN
        else:
            signal = magnitude   # Bearish book -> fade UP

        self._last_signal = signal
        return signal

    @property
    def last_signal(self) -> float:
        """Most recently computed signal value."""
        return self._last_signal

    @property
    def current_imbalance(self) -> float:
        """Current smoothed imbalance (for diagnostics)."""
        if not self._imbalance_history:
            return 0.0
        recent = list(self._imbalance_history)[-self.smoothing_window:]
        return sum(recent) / len(recent)
