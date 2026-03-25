"""Layer 3: Liquidation Level Signal.

Assesses proximity of BTC price to liquidation clusters from CoinGlass data.

Logic:
- If price is rising AND nearest short liquidation cluster is close
  → cascade amplification → bullish signal (short squeeze)
- If price is falling AND nearest long liquidation cluster is close
  → cascade amplification → bearish signal (long cascade)
- Volume at each level matters: $50M cluster >> $5M cluster
- If data unavailable, output 0 (neutral)

Output: signal between -1.0 (bearish cascade) and +1.0 (bullish cascade)
"""

import logging

from data.coinglass_scraper import LiquidationData

logger = logging.getLogger(__name__)


class LiquidationSignal:
    """Computes the liquidation proximity signal (Layer 3).

    Measures how close BTC is to major liquidation clusters and whether
    price momentum would trigger a cascade.
    """

    def __init__(
        self,
        max_distance_pct: float = 0.02,    # Max distance to consider (2% from price)
        volume_scale: float = 100_000_000,  # $100M = max volume influence
        cascade_boost: float = 1.5,         # Multiplier when momentum aligns with liq direction
    ):
        self.max_distance_pct = max_distance_pct
        self.volume_scale = volume_scale
        self.cascade_boost = cascade_boost

    def compute(
        self,
        current_price: float,
        liq_data: LiquidationData,
        price_momentum: float = 0.0,       # From Layer 2, [-1, 1]
    ) -> float:
        """Compute liquidation proximity signal.

        Args:
            current_price: Current BTC price.
            liq_data: Liquidation data from CoinGlass scraper.
            price_momentum: Current momentum signal for cascade alignment.

        Returns:
            Signal in [-1.0, 1.0]. Positive = bullish (short squeeze likely).
            Negative = bearish (long cascade likely). 0 = neutral/no data.
        """
        if not liq_data.is_valid or current_price <= 0:
            return 0.0

        # Compute proximity scores for short and long liquidation clusters
        short_score = self._proximity_score(
            current_price, liq_data.short_levels, direction="above"
        )
        long_score = self._proximity_score(
            current_price, liq_data.long_levels, direction="below"
        )

        # Base signal: net pressure
        # Short liquidations above price → bullish if triggered (short squeeze)
        # Long liquidations below price → bearish if triggered (long cascade)
        signal = 0.0

        # Rising price + nearby short liquidations = bullish cascade
        if short_score > 0 and price_momentum > 0:
            signal += short_score * self.cascade_boost * min(price_momentum, 1.0)
        elif short_score > 0:
            # Shorts are nearby but price not moving up — mild bullish magnet
            signal += short_score * 0.3

        # Falling price + nearby long liquidations = bearish cascade
        if long_score > 0 and price_momentum < 0:
            signal -= long_score * self.cascade_boost * min(abs(price_momentum), 1.0)
        elif long_score > 0:
            # Longs are nearby but price not falling — mild bearish magnet
            signal -= long_score * 0.3

        # If both sides are roughly equal, dampen (balanced liq = no directional edge)
        if short_score > 0 and long_score > 0:
            balance_ratio = min(short_score, long_score) / max(short_score, long_score)
            if balance_ratio > 0.8:
                signal *= 0.3  # Heavily dampen if balanced

        return max(-1.0, min(1.0, signal))

    def _proximity_score(
        self,
        current_price: float,
        levels: list,
        direction: str,     # "above" or "below"
    ) -> float:
        """Compute a 0-1 proximity score for liquidation levels.

        Closer levels with higher volume score higher.
        """
        if not levels:
            return 0.0

        total_score = 0.0

        for level in levels:
            distance_pct = abs(level.price - current_price) / current_price

            # Skip levels too far away
            if distance_pct > self.max_distance_pct:
                continue

            # Check direction: short liqs should be above price, long liqs below
            if direction == "above" and level.price <= current_price:
                continue
            if direction == "below" and level.price >= current_price:
                continue

            # Distance factor: closer = stronger (1.0 at price, 0.0 at max_distance)
            distance_factor = 1.0 - (distance_pct / self.max_distance_pct)

            # Volume factor: scales logarithmically, capped at 1.0
            import math
            vol_ratio = level.volume_usd / self.volume_scale
            volume_factor = min(1.0, math.log1p(vol_ratio * 10) / math.log1p(10))

            # Combined score for this level
            level_score = distance_factor * volume_factor
            total_score += level_score

        # Cap total score at 1.0
        return min(1.0, total_score)
