"""Layer 5: Cross-Exchange Sentiment Signal (Coinalyze).

Combines open interest changes, long/short ratios, and funding rates
from multiple exchanges to gauge market-wide sentiment and positioning.

Key edges for 5-min BTC prediction:
- Extreme long/short ratios → contrarian signal (crowded trades unwind)
- Rising OI + directional move → trend confirmation (new money backing it)
- Extreme funding → mean reversion pressure (expensive to hold position)
- OI drop + price drop → capitulation (bearish)
- OI drop + price rise → short squeeze (bullish)

Output: signal between -1.0 (bearish sentiment) and +1.0 (bullish sentiment)
"""

import logging

from data.coinalyze_feed import CoinalyzeSnapshot

logger = logging.getLogger(__name__)


class SentimentSignal:
    """Computes cross-exchange sentiment signal (Layer 5).

    Combines three sub-signals:
    1. Long/Short Ratio — contrarian when extreme
    2. Open Interest Change — confirms or contradicts price moves
    3. Funding Rate — mean reversion pressure at extremes
    """

    def __init__(
        self,
        # Long/short ratio thresholds
        lsr_extreme_long: float = 0.62,     # Above this = too many longs
        lsr_extreme_short: float = 0.62,    # Above this = too many shorts
        lsr_weight: float = 0.40,           # Weight in combined signal

        # OI change thresholds
        oi_significant_change_pct: float = 2.0,  # % change to be meaningful
        oi_weight: float = 0.30,

        # Funding rate thresholds
        funding_extreme: float = 0.03,       # Extreme funding rate (%)
        funding_weight: float = 0.30,
    ):
        self.lsr_extreme_long = lsr_extreme_long
        self.lsr_extreme_short = lsr_extreme_short
        self.lsr_weight = lsr_weight
        self.oi_significant_change_pct = oi_significant_change_pct
        self.oi_weight = oi_weight
        self.funding_extreme = funding_extreme
        self.funding_weight = funding_weight

    def compute(
        self,
        snapshot: CoinalyzeSnapshot,
        price_momentum: float = 0.0,    # From L2 momentum signal [-1, 1]
    ) -> float:
        """Compute cross-exchange sentiment signal.

        Args:
            snapshot: Current Coinalyze data snapshot.
            price_momentum: Current momentum signal for OI interpretation.

        Returns:
            Signal in [-1.0, 1.0]. Positive = bullish, negative = bearish.
        """
        if not snapshot.is_valid:
            return 0.0

        lsr_signal = self._compute_lsr_signal(snapshot)
        oi_signal = self._compute_oi_signal(snapshot, price_momentum)
        funding_signal = self._compute_funding_signal(snapshot)

        # Weighted combination
        combined = (
            self.lsr_weight * lsr_signal
            + self.oi_weight * oi_signal
            + self.funding_weight * funding_signal
        )

        return max(-1.0, min(1.0, combined))

    def _compute_lsr_signal(self, snapshot: CoinalyzeSnapshot) -> float:
        """Long/short ratio → contrarian signal.

        When too many traders are long, expect downside (bearish).
        When too many traders are short, expect upside (bullish).
        Near 50/50 = neutral.
        """
        if snapshot.long_ratio == 0.5 and snapshot.short_ratio == 0.5:
            return 0.0

        # How far from neutral (0.5)?
        long_excess = snapshot.long_ratio - 0.5
        short_excess = snapshot.short_ratio - 0.5

        signal = 0.0

        if snapshot.long_ratio > self.lsr_extreme_long:
            # Too many longs → contrarian bearish
            # Normalize: at extreme threshold = 0.5 signal, at 0.75 = 1.0
            intensity = (snapshot.long_ratio - self.lsr_extreme_long) / (0.75 - self.lsr_extreme_long)
            signal = -min(1.0, intensity)
        elif snapshot.short_ratio > self.lsr_extreme_short:
            # Too many shorts → contrarian bullish
            intensity = (snapshot.short_ratio - self.lsr_extreme_short) / (0.75 - self.lsr_extreme_short)
            signal = min(1.0, intensity)
        else:
            # Mild lean — weak directional signal
            # More shorts than longs = mild bullish (contrarian)
            signal = -long_excess * 2.0  # Scale mild leans

        return max(-1.0, min(1.0, signal))

    def _compute_oi_signal(
        self, snapshot: CoinalyzeSnapshot, price_momentum: float
    ) -> float:
        """Open interest change + price direction → trend/reversal signal.

        OI rising + price rising = new longs entering (bullish confirmation)
        OI rising + price falling = new shorts entering (bearish confirmation)
        OI falling + price rising = short squeeze / short covering (bullish)
        OI falling + price falling = long capitulation (bearish)
        """
        if abs(snapshot.oi_change_pct) < 0.5:
            # OI barely changed — no signal
            return 0.0

        # Normalize OI change to 0-1 scale
        oi_magnitude = min(
            abs(snapshot.oi_change_pct) / self.oi_significant_change_pct, 1.0
        )
        oi_rising = snapshot.oi_change_pct > 0

        signal = 0.0

        if oi_rising and price_momentum > 0.1:
            # New money + price up = trend confirmation (bullish)
            signal = oi_magnitude * 0.8
        elif oi_rising and price_momentum < -0.1:
            # New money + price down = trend confirmation (bearish)
            signal = -oi_magnitude * 0.8
        elif not oi_rising and price_momentum > 0.1:
            # Positions closing + price up = short squeeze (bullish)
            signal = oi_magnitude * 1.0
        elif not oi_rising and price_momentum < -0.1:
            # Positions closing + price down = capitulation (bearish)
            signal = -oi_magnitude * 1.0
        else:
            # Price flat — OI change less meaningful
            signal = 0.0

        return max(-1.0, min(1.0, signal))

    def _compute_funding_signal(self, snapshot: CoinalyzeSnapshot) -> float:
        """Funding rate → mean reversion pressure.

        High positive funding = longs paying shorts → expensive to be long
          → bearish pressure (contrarian)
        High negative funding = shorts paying longs → expensive to be short
          → bullish pressure (contrarian)
        Near zero = neutral
        """
        fr = snapshot.funding_rate

        if abs(fr) < 0.005:
            # Funding near zero — no signal
            return 0.0

        # Normalize to [-1, 1] based on extreme threshold
        intensity = min(abs(fr) / self.funding_extreme, 1.0)

        if fr > 0:
            # Positive funding → bearish pressure (contrarian)
            return -intensity
        else:
            # Negative funding → bullish pressure (contrarian)
            return intensity
