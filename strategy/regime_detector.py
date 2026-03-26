"""Regime detector — classifies current BTC market conditions.

Uses 1-minute candle data from Binance to detect whether the market is
trending, ranging, volatile, or dead. The detected regime adjusts signal
weights, edge thresholds, and position sizing.
"""

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Regime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"


@dataclass
class RegimeState:
    regime: Regime
    confidence: float       # 0.0-1.0
    trend_strength: float   # 0.0-1.0 (ADX-like)
    volatility_pct: float   # ATR percentile vs recent history
    choppiness: float       # 0.0-1.0 (direction flip rate)

    @property
    def label(self) -> str:
        return self.regime.value

    def __str__(self) -> str:
        return (
            f"{self.regime.value} (conf={self.confidence:.2f}, "
            f"trend={self.trend_strength:.2f}, vol={self.volatility_pct:.2f}, "
            f"chop={self.choppiness:.2f})"
        )


# Regime-specific parameter overrides
REGIME_PARAMS = {
    Regime.TRENDING_UP: {
        "weight_adjustments": {
            "oracle": -0.05,
            "momentum": +0.10,
            "liquidation": 0.0,
            "orderbook": 0.0,
            "sentiment": -0.05,
        },
        "edge_multiplier": 0.75,    # Lower edge required — go with the trend
        "size_multiplier": 1.0,
    },
    Regime.TRENDING_DOWN: {
        "weight_adjustments": {
            "oracle": -0.05,
            "momentum": +0.08,
            "liquidation": +0.05,
            "orderbook": 0.0,
            "sentiment": -0.08,
        },
        "edge_multiplier": 0.75,
        "size_multiplier": 1.0,
    },
    Regime.RANGING: {
        "weight_adjustments": {
            "oracle": 0.0,
            "momentum": -0.10,
            "liquidation": -0.03,
            "orderbook": +0.10,
            "sentiment": +0.03,
        },
        "edge_multiplier": 1.5,     # Need stronger signal in chop
        "size_multiplier": 0.75,
    },
    Regime.HIGH_VOLATILITY: {
        "weight_adjustments": {
            "oracle": -0.05,
            "momentum": 0.0,
            "liquidation": +0.10,
            "orderbook": -0.05,
            "sentiment": 0.0,
        },
        "edge_multiplier": 1.8,     # Very high bar — chaotic conditions
        "size_multiplier": 0.5,
    },
    Regime.LOW_VOLATILITY: {
        "weight_adjustments": {
            "oracle": 0.0,
            "momentum": 0.0,
            "liquidation": 0.0,
            "orderbook": 0.0,
            "sentiment": 0.0,
        },
        "edge_multiplier": 999.0,   # Effectively blocks trading
        "size_multiplier": 0.0,
    },
}


class RegimeDetector:
    """Detects BTC market regime from 1-minute candle data."""

    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        choppiness_lookback: int = 15,
        trend_threshold: float = 0.4,
        high_vol_percentile: float = 0.85,
        low_vol_percentile: float = 0.15,
        choppiness_threshold: float = 0.6,
        atr_history_len: int = 120,
    ):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.choppiness_lookback = choppiness_lookback
        self.trend_threshold = trend_threshold
        self.high_vol_pct = high_vol_percentile
        self.low_vol_pct = low_vol_percentile
        self.chop_threshold = choppiness_threshold

        # Rolling ATR history for percentile calculation
        self._atr_history = deque(maxlen=atr_history_len)
        self._last_regime = RegimeState(
            regime=Regime.RANGING, confidence=0.0,
            trend_strength=0.0, volatility_pct=0.5, choppiness=0.5,
        )

    def detect(self, candles) -> RegimeState:
        """Detect current market regime from recent candles.

        Args:
            candles: List/deque of Candle objects (1-min). Needs at least
                     max(adx_period, atr_period, choppiness_lookback) + 1 candles.

        Returns:
            RegimeState with detected regime and metrics.
        """
        candle_list = list(candles)
        min_required = max(self.adx_period, self.atr_period, self.choppiness_lookback) + 1

        if len(candle_list) < min_required:
            return self._last_regime

        # 1. Compute trend strength (simplified ADX)
        trend_strength, trend_direction = self._compute_trend(candle_list)

        # 2. Compute volatility (ATR percentile)
        current_atr = self._compute_atr(candle_list)
        self._atr_history.append(current_atr)
        vol_percentile = self._atr_percentile(current_atr)

        # 3. Compute choppiness (direction flip rate)
        choppiness = self._compute_choppiness(candle_list)

        # 4. Classify regime
        regime, confidence = self._classify(
            trend_strength, trend_direction, vol_percentile, choppiness
        )

        state = RegimeState(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_pct=vol_percentile,
            choppiness=choppiness,
        )
        self._last_regime = state
        return state

    def get_params(self, regime: Regime) -> dict:
        """Get parameter overrides for the given regime."""
        return REGIME_PARAMS.get(regime, REGIME_PARAMS[Regime.RANGING])

    def _compute_trend(self, candles) -> tuple:
        """Simplified ADX-like trend strength. Returns (strength, direction).

        strength: 0.0-1.0 (0 = no trend, 1 = strong trend)
        direction: +1 (up), -1 (down), 0 (flat)
        """
        period = self.adx_period
        recent = candles[-(period + 1):]

        if len(recent) < period + 1:
            return 0.0, 0

        # Compute directional movement
        plus_dm_sum = 0.0
        minus_dm_sum = 0.0
        tr_sum = 0.0

        for i in range(1, len(recent)):
            high = recent[i].high
            low = recent[i].low
            prev_high = recent[i - 1].high
            prev_low = recent[i - 1].low
            prev_close = recent[i - 1].close

            # True Range
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_sum += tr

            # Directional movement
            up_move = high - prev_high
            down_move = prev_low - low

            if up_move > down_move and up_move > 0:
                plus_dm_sum += up_move
            if down_move > up_move and down_move > 0:
                minus_dm_sum += down_move

        if tr_sum == 0:
            return 0.0, 0

        # Directional indicators
        plus_di = plus_dm_sum / tr_sum
        minus_di = minus_dm_sum / tr_sum

        # ADX approximation
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0, 0

        dx = abs(plus_di - minus_di) / di_sum
        # Normalize to 0-1 range (DX is already 0-1)
        trend_strength = min(1.0, dx)

        # Direction
        if plus_di > minus_di:
            direction = 1
        elif minus_di > plus_di:
            direction = -1
        else:
            direction = 0

        return trend_strength, direction

    def _compute_atr(self, candles) -> float:
        """Average True Range over recent candles."""
        period = self.atr_period
        recent = candles[-(period + 1):]

        if len(recent) < 2:
            return 0.0

        tr_values = []
        for i in range(1, len(recent)):
            high = recent[i].high
            low = recent[i].low
            prev_close = recent[i - 1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        return sum(tr_values) / len(tr_values) if tr_values else 0.0

    def _atr_percentile(self, current_atr: float) -> float:
        """Compute where current ATR sits in recent history (0.0-1.0)."""
        if len(self._atr_history) < 10:
            return 0.5  # Not enough history, assume middle

        sorted_history = sorted(self._atr_history)
        rank = sum(1 for x in sorted_history if x <= current_atr)
        return rank / len(sorted_history)

    def _compute_choppiness(self, candles) -> float:
        """Direction flip rate — how often candles alternate up/down.

        Returns 0.0 (all same direction) to 1.0 (alternating every candle).
        """
        recent = candles[-self.choppiness_lookback:]
        if len(recent) < 3:
            return 0.5

        flips = 0
        for i in range(1, len(recent)):
            prev_dir = 1 if recent[i - 1].close >= recent[i - 1].open else -1
            curr_dir = 1 if recent[i].close >= recent[i].open else -1
            if curr_dir != prev_dir:
                flips += 1

        return flips / (len(recent) - 1)

    def _classify(
        self, trend_strength: float, trend_direction: int,
        vol_percentile: float, choppiness: float,
    ) -> tuple:
        """Classify regime from metrics. Returns (Regime, confidence)."""

        # Low volatility takes priority — dead market, no edge
        if vol_percentile < self.low_vol_pct:
            confidence = 1.0 - (vol_percentile / self.low_vol_pct)
            return Regime.LOW_VOLATILITY, min(1.0, confidence)

        # High volatility — explosive conditions
        if vol_percentile > self.high_vol_pct:
            confidence = (vol_percentile - self.high_vol_pct) / (1.0 - self.high_vol_pct)
            return Regime.HIGH_VOLATILITY, min(1.0, confidence)

        # Trending — strong directional movement, low choppiness
        if trend_strength > self.trend_threshold and choppiness < self.chop_threshold:
            confidence = (
                (trend_strength - self.trend_threshold) / (1.0 - self.trend_threshold)
                * (1.0 - choppiness)
            )
            if trend_direction >= 0:
                return Regime.TRENDING_UP, min(1.0, confidence)
            else:
                return Regime.TRENDING_DOWN, min(1.0, confidence)

        # Default: ranging/choppy
        confidence = max(choppiness, 1.0 - trend_strength)
        return Regime.RANGING, min(1.0, confidence * 0.8)
