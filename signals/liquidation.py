"""Layer 3: Liquidation Signal.

Two data sources combined:
1. CoinGlass: Static liquidation level clusters (where liquidations WOULD happen)
2. Binance forceOrder: Real-time liquidation events (what IS happening right now)

Logic:
- CoinGlass proximity: price near a cluster + aligned momentum = potential cascade
- Live liquidations: actual cascade happening = strongest signal
- Burst detection: sudden spike in one-sided liquidations = very high conviction

The live feed is weighted more heavily because it shows actual market impact,
not just potential. A $5M burst of long liquidations in 30 seconds is more
informative than a $50M cluster sitting 1% away.

Output: signal between -1.0 (bearish cascade) and +1.0 (bullish cascade)
"""

import logging
import math

from data.coinglass_scraper import LiquidationData
from data.binance_liquidations import LiquidationStats

logger = logging.getLogger(__name__)


class LiquidationSignal:
    """Computes the liquidation signal (Layer 3).

    Blends static level proximity (CoinGlass) with real-time
    liquidation flow (Binance forceOrder stream).
    """

    def __init__(
        self,
        # CoinGlass proximity params
        max_distance_pct: float = 0.02,
        volume_scale: float = 100_000_000,
        cascade_boost: float = 1.5,
        # Live feed params
        live_weight: float = 0.65,        # Weight given to live data vs static levels
        burst_boost: float = 2.0,         # Multiplier when burst detected
        significant_volume: float = 500_000,  # $500K in window = meaningful
    ):
        self.max_distance_pct = max_distance_pct
        self.volume_scale = volume_scale
        self.cascade_boost = cascade_boost
        self.live_weight = live_weight
        self.burst_boost = burst_boost
        self.significant_volume = significant_volume

    def compute(
        self,
        current_price: float,
        liq_data: LiquidationData,
        price_momentum: float = 0.0,
        live_stats: LiquidationStats | None = None,
    ) -> float:
        """Compute combined liquidation signal.

        Args:
            current_price: Current BTC price.
            liq_data: Static liquidation levels from CoinGlass.
            price_momentum: Current momentum signal [-1, 1].
            live_stats: Real-time liquidation stats from Binance.

        Returns:
            Signal in [-1.0, 1.0].
        """
        static_signal = self._compute_static(current_price, liq_data, price_momentum)
        live_signal = self._compute_live(live_stats, price_momentum)

        # If we have both, blend with live weighted more heavily
        if live_stats and live_stats.is_valid and live_stats.is_significant:
            combined = (
                (1 - self.live_weight) * static_signal
                + self.live_weight * live_signal
            )
        elif live_stats and live_stats.is_valid:
            # Live data exists but not significant volume — use it as mild modifier
            combined = static_signal + live_signal * 0.3
        else:
            # No live data — fall back to static only
            combined = static_signal

        return max(-1.0, min(1.0, combined))

    def _compute_live(
        self, stats: LiquidationStats | None, price_momentum: float
    ) -> float:
        """Compute signal from real-time liquidation events.

        Key insight: liquidations that are HAPPENING tell you more than
        liquidations that MIGHT happen.

        Long liquidations happening = forced selling = bearish pressure
        Short liquidations happening = forced buying = bullish pressure
        """
        if not stats or not stats.is_valid:
            return 0.0

        signal = 0.0

        # Base signal from imbalance
        # net_imbalance > 0 = more shorts liquidated = bullish
        # net_imbalance < 0 = more longs liquidated = bearish
        if stats.is_significant:
            # Scale by volume — more volume = stronger signal
            vol_factor = min(stats.total_volume / self.significant_volume, 3.0) / 3.0
            signal = stats.net_imbalance * vol_factor

        # Acceleration boost — cascade in progress
        if stats.acceleration > 0.5:
            # Liquidations are accelerating — amplify the direction
            accel_factor = min(stats.acceleration, 3.0) / 3.0
            signal *= (1.0 + accel_factor)

        # Burst override — strongest signal
        if stats.burst_detected:
            if stats.burst_side == "LONG":
                # Longs getting wiped out in a burst → strong bearish
                burst_signal = -1.0 * min(stats.total_volume / (self.significant_volume * 2), 1.0)
            else:
                # Shorts getting wiped out in a burst → strong bullish
                burst_signal = 1.0 * min(stats.total_volume / (self.significant_volume * 2), 1.0)

            # Burst signal takes priority if it's stronger
            if abs(burst_signal) > abs(signal):
                signal = burst_signal * self.burst_boost

        # Momentum alignment — when liquidations and momentum agree, it's stronger
        if signal != 0 and price_momentum != 0:
            agreement = signal * price_momentum
            if agreement > 0:
                # Momentum aligns with liquidation direction → cascade confirmation
                signal *= 1.3
            # Don't dampen on disagreement — live liquidations are hard data

        return max(-1.0, min(1.0, signal))

    def _compute_static(
        self, current_price: float, liq_data: LiquidationData, price_momentum: float
    ) -> float:
        """Compute signal from static CoinGlass liquidation levels.

        Original logic — proximity to liquidation clusters.
        """
        if not liq_data.is_valid or current_price <= 0:
            return 0.0

        short_score = self._proximity_score(
            current_price, liq_data.short_levels, direction="above"
        )
        long_score = self._proximity_score(
            current_price, liq_data.long_levels, direction="below"
        )

        signal = 0.0

        if short_score > 0 and price_momentum > 0:
            signal += short_score * self.cascade_boost * min(price_momentum, 1.0)
        elif short_score > 0:
            signal += short_score * 0.3

        if long_score > 0 and price_momentum < 0:
            signal -= long_score * self.cascade_boost * min(abs(price_momentum), 1.0)
        elif long_score > 0:
            signal -= long_score * 0.3

        if short_score > 0 and long_score > 0:
            balance_ratio = min(short_score, long_score) / max(short_score, long_score)
            if balance_ratio > 0.8:
                signal *= 0.3

        return max(-1.0, min(1.0, signal))

    def _proximity_score(
        self,
        current_price: float,
        levels: list,
        direction: str,
    ) -> float:
        """Compute a 0-1 proximity score for liquidation levels."""
        if not levels:
            return 0.0

        total_score = 0.0

        for level in levels:
            distance_pct = abs(level.price - current_price) / current_price

            if distance_pct > self.max_distance_pct:
                continue

            if direction == "above" and level.price <= current_price:
                continue
            if direction == "below" and level.price >= current_price:
                continue

            distance_factor = 1.0 - (distance_pct / self.max_distance_pct)
            vol_ratio = level.volume_usd / self.volume_scale
            volume_factor = min(1.0, math.log1p(vol_ratio * 10) / math.log1p(10))

            level_score = distance_factor * volume_factor
            total_score += level_score

        return min(1.0, total_score)
