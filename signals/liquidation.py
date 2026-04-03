"""Layer 3: Liquidation Signal.

Three data sources combined (priority order):
1. Binance forceOrder: Real-time liquidation events (what IS happening right now)
2. Coinalyze: Cross-exchange OI, L/S ratio, funding (where leverage is concentrated)
3. CoinGlass: Static liquidation level clusters (fallback, often offline)

Logic:
- Live liquidations: actual cascade happening = strongest signal
- Coinalyze positioning: crowded longs + expanding OI + high funding = squeeze risk
- Burst detection: sudden spike in one-sided liquidations = very high conviction

The live feed is weighted most heavily because it shows actual market impact.
Coinalyze positioning data predicts WHERE liquidations are likely to happen.

Output: signal between -1.0 (bearish cascade) and +1.0 (bullish cascade)
"""

import logging
import math
from typing import Any

from data.coinglass_scraper import LiquidationData
from data.binance_liquidations import LiquidationStats

logger = logging.getLogger(__name__)


class LiquidationSignal:
    """Computes the liquidation signal (Layer 3).

    Blends three sources:
    1. Real-time Binance liquidation flow (strongest weight)
    2. Coinalyze cross-exchange positioning (reliable API data)
    3. CoinGlass static levels (fallback when available)
    """

    def __init__(
        self,
        # CoinGlass proximity params
        max_distance_pct: float = 0.02,
        volume_scale: float = 100_000_000,
        cascade_boost: float = 1.5,
        # Live feed params
        live_weight: float = 0.50,        # Weight given to live Binance data
        positioning_weight: float = 0.35, # Weight given to Coinalyze positioning
        static_weight: float = 0.15,      # Weight given to CoinGlass levels
        burst_boost: float = 2.0,         # Multiplier when burst detected
        significant_volume: float = 500_000,  # $500K in window = meaningful
        # Coinalyze thresholds
        crowded_long_threshold: float = 0.58,   # L/S ratio where longs are crowded
        crowded_short_threshold: float = 0.42,  # L/S ratio where shorts are crowded
        high_funding_threshold: float = 0.01,   # Funding rate considered elevated
        oi_change_threshold: float = 2.0,       # OI change % considered significant
    ):
        self.max_distance_pct = max_distance_pct
        self.volume_scale = volume_scale
        self.cascade_boost = cascade_boost
        self.live_weight = live_weight
        self.positioning_weight = positioning_weight
        self.static_weight = static_weight
        self.burst_boost = burst_boost
        self.significant_volume = significant_volume
        self.crowded_long_threshold = crowded_long_threshold
        self.crowded_short_threshold = crowded_short_threshold
        self.high_funding_threshold = high_funding_threshold
        self.oi_change_threshold = oi_change_threshold

    def compute(
        self,
        current_price: float,
        liq_data: LiquidationData,
        price_momentum: float = 0.0,
        live_stats: LiquidationStats | None = None,
        coinalyze_snapshot: Any | None = None,
    ) -> float:
        """Compute combined liquidation signal.

        Args:
            current_price: Current BTC price.
            liq_data: Static liquidation levels from CoinGlass.
            price_momentum: Current momentum signal [-1, 1].
            live_stats: Real-time liquidation stats from Binance.
            coinalyze_snapshot: Cross-exchange positioning from Coinalyze.

        Returns:
            Signal in [-1.0, 1.0].
        """
        live_signal = self._compute_live(live_stats, price_momentum)
        positioning_signal = self._compute_positioning(
            coinalyze_snapshot, price_momentum
        )
        static_signal = self._compute_static(current_price, liq_data, price_momentum)

        # Determine which sources are available and blend accordingly
        has_live = live_stats and live_stats.is_valid and live_stats.is_significant
        has_positioning = (
            coinalyze_snapshot is not None
            and getattr(coinalyze_snapshot, "is_valid", False)
        )
        has_static = liq_data and liq_data.is_valid

        if has_live and has_positioning and has_static:
            # All three sources — use configured weights
            combined = (
                self.live_weight * live_signal
                + self.positioning_weight * positioning_signal
                + self.static_weight * static_signal
            )
        elif has_live and has_positioning:
            # Live + positioning (most common — CoinGlass often offline)
            combined = 0.55 * live_signal + 0.45 * positioning_signal
        elif has_live and has_static:
            # Live + static (Coinalyze offline)
            combined = 0.65 * live_signal + 0.35 * static_signal
        elif has_positioning and has_static:
            # No live events but have positioning + static
            combined = 0.65 * positioning_signal + 0.35 * static_signal
        elif has_live:
            combined = live_signal
        elif has_positioning:
            combined = positioning_signal
        elif has_static:
            combined = static_signal
        else:
            # No data at all
            return 0.0

        # Burst override — strongest possible signal from live data
        if (live_stats and live_stats.is_valid
                and live_stats.burst_detected):
            burst_signal = self._compute_burst(live_stats)
            if abs(burst_signal) > abs(combined):
                combined = burst_signal

        return max(-1.0, min(1.0, combined))

    def _compute_positioning(
        self, snapshot: Any | None, price_momentum: float
    ) -> float:
        """Compute signal from Coinalyze cross-exchange positioning data.

        Key insight: when the market is heavily positioned one way AND
        momentum is moving against those positions, liquidation cascades
        are likely. This replaces the CoinGlass static levels with
        real-time cross-exchange data.

        Signals:
        - Crowded longs + downward momentum = bearish (long squeeze incoming)
        - Crowded shorts + upward momentum = bullish (short squeeze incoming)
        - High funding + crowded position = amplified signal
        - Expanding OI + crowded position = more fuel for cascade
        """
        if snapshot is None or not getattr(snapshot, "is_valid", False):
            return 0.0

        signal = 0.0
        long_ratio = getattr(snapshot, "long_ratio", 0.5)
        funding_rate = getattr(snapshot, "funding_rate", 0.0)
        oi_change_pct = getattr(snapshot, "oi_change_pct", 0.0)

        # ── Crowding signal ──
        # How far from neutral (0.50) is the long/short split?
        crowding = long_ratio - 0.5  # Positive = more longs, negative = more shorts

        if long_ratio > self.crowded_long_threshold:
            # Market is crowded long
            # If momentum is bearish → long squeeze likely → bearish signal
            crowd_strength = (long_ratio - self.crowded_long_threshold) / (
                1.0 - self.crowded_long_threshold
            )
            crowd_strength = min(crowd_strength, 1.0)

            if price_momentum < 0:
                # Price moving against crowded longs — squeeze!
                signal = -crowd_strength * (1.0 + abs(price_momentum))
            else:
                # Price moving with longs — mild bearish contrarian signal
                # (crowded = vulnerable, even if momentum agrees for now)
                signal = -crowd_strength * 0.3

        elif long_ratio < self.crowded_short_threshold:
            # Market is crowded short
            crowd_strength = (self.crowded_short_threshold - long_ratio) / (
                self.crowded_short_threshold
            )
            crowd_strength = min(crowd_strength, 1.0)

            if price_momentum > 0:
                # Price moving against crowded shorts — squeeze!
                signal = crowd_strength * (1.0 + abs(price_momentum))
            else:
                # Mild bullish contrarian
                signal = crowd_strength * 0.3

        # ── Funding rate amplifier ──
        # High positive funding = longs paying shorts = longs are aggressive
        # High negative funding = shorts paying longs = shorts are aggressive
        if abs(funding_rate) > self.high_funding_threshold:
            funding_factor = min(abs(funding_rate) / 0.05, 1.0)  # Normalize to 0-1

            if funding_rate > 0 and signal < 0:
                # High positive funding + bearish signal = amplified
                # (aggressive longs paying premium = more vulnerable to squeeze)
                signal *= (1.0 + funding_factor * 0.5)
            elif funding_rate < 0 and signal > 0:
                # High negative funding + bullish signal = amplified
                signal *= (1.0 + funding_factor * 0.5)

        # ── OI expansion amplifier ──
        # Growing OI = new positions being opened = more fuel for liquidations
        if abs(oi_change_pct) > self.oi_change_threshold:
            oi_factor = min(abs(oi_change_pct) / 10.0, 1.0)

            if oi_change_pct > 0:
                # OI expanding = more leverage in system = bigger potential cascade
                signal *= (1.0 + oi_factor * 0.3)
            else:
                # OI contracting = positions closing = less cascade risk
                signal *= (1.0 - oi_factor * 0.2)

        return max(-1.0, min(1.0, signal))

    def _compute_burst(self, stats: LiquidationStats) -> float:
        """Compute burst signal — sudden spike in one-sided liquidations."""
        if not stats.burst_detected:
            return 0.0

        vol_factor = min(stats.total_volume / (self.significant_volume * 2), 1.0)

        if stats.burst_side == "LONG":
            # Longs getting wiped — strong bearish
            return -1.0 * vol_factor * self.burst_boost
        else:
            # Shorts getting wiped — strong bullish
            return 1.0 * vol_factor * self.burst_boost

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
