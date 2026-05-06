"""Multi-timeframe regime detector — drop-in replacement for RegimeDetector.

Extends regime detection with higher-timeframe trend alignment.
Self-aggregates 1m candles into 15m, 1h, 4h, and 1d bars internally,
so callers just pass 1m candles exactly like the original RegimeDetector.

Usage (identical to RegimeDetector):
    detector = MTFRegimeDetector()
    state = detector.detect(snapshot.binance_candles)
    params = detector.get_params(state.regime)

Parameters calibrated via walk-forward validation over 2 years of
BTC/USDT data (Apr 2024 - Apr 2026), 21 folds of 90d train / 30d test.
"""

import csv
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from strategy.regime_detector import Regime, RegimeState, REGIME_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class MTFTrend:
    """Trend state for a single timeframe."""
    direction: int = 0        # +1 up, -1 down, 0 flat
    strength: float = 0.0     # 0.0-1.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0


@dataclass
class MTFRegimeState(RegimeState):
    """Extended regime state with multi-timeframe context.

    Inherits from RegimeState for full backwards compatibility.
    Adds alignment fields for strategies that want to use them.
    """
    alignment_score: float = 0.0   # -1.0 (all down) to +1.0 (all up)
    alignment_count: int = 0       # How many TFs agree on direction
    htf_trend: int = 0             # Higher-timeframe dominant trend: +1/-1/0
    tf_details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.regime.value} (conf={self.confidence:.2f}, "
            f"align={self.alignment_score:+.2f}/{self.alignment_count}TF, "
            f"trend={self.trend_strength:.2f}, vol={self.volatility_pct:.2f}, "
            f"chop={self.choppiness:.2f})"
        )


@dataclass
class _AggCandle:
    """Internal aggregated candle for HTF computation."""
    open: float
    high: float
    low: float
    close: float
    timestamp_ms: int


class MTFRegimeDetector:
    """Multi-timeframe regime detector.

    Drop-in replacement for RegimeDetector. Accepts the same 1m candle
    input and returns a RegimeState-compatible object. Internally maintains
    rolling buffers of aggregated higher-timeframe candles and computes
    trend alignment across 15m, 1h, 4h, and 1d timeframes.

    Default parameters are optimised via walk-forward calibration:
    - trend_threshold: 0.30 (was 0.40) — catches more genuine trends
    - high_vol_percentile: 0.90 (was 0.85) — only blocks extreme volatility
    - low_vol_percentile: 0.20 (was 0.15) — blocks more dead markets
    - alignment_weight: 0.55 (was N/A) — HTF alignment strongly influences
    - stickiness: 3 — prevents flip-flopping between regimes
    """

    # HTF timeframes in minutes and their weights in alignment score
    _TF_CONFIGS = {
        "15m":  {"minutes": 15,   "weight": 0.15, "buffer_size": 100},
        "1h":   {"minutes": 60,   "weight": 0.25, "buffer_size": 100},
        "4h":   {"minutes": 240,  "weight": 0.35, "buffer_size": 60},
        "1d":   {"minutes": 1440, "weight": 0.25, "buffer_size": 40},
    }

    def __init__(
        self,
        # Base detector params (calibrated defaults)
        adx_period: int = 14,
        atr_period: int = 14,
        choppiness_lookback: int = 15,
        trend_threshold: float = 0.30,
        high_vol_percentile: float = 0.90,
        low_vol_percentile: float = 0.20,
        choppiness_threshold: float = 0.60,
        atr_history_len: int = 120,
        # MTF params (calibrated defaults)
        ema_fast_period: int = 10,
        ema_slow_period: int = 30,
        alignment_weight: float = 0.55,
        stickiness: int = 3,
    ):
        """Initialise the MTF regime detector.

        Args:
            adx_period: ADX calculation window (1m candles).
            atr_period: ATR calculation window (1m candles).
            choppiness_lookback: Direction flip rate window.
            trend_threshold: Min effective trend for trending classification.
            high_vol_percentile: ATR percentile for high vol cutoff.
            low_vol_percentile: ATR percentile for low vol cutoff.
            choppiness_threshold: Max effective choppiness for trending.
            atr_history_len: Rolling ATR history length.
            ema_fast_period: Fast EMA period for HTF trend.
            ema_slow_period: Slow EMA period for HTF trend.
            alignment_weight: How much HTF alignment influences (0-1).
            stickiness: Consecutive signals needed before regime change.
        """
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.choppiness_lookback = choppiness_lookback
        self.trend_threshold = trend_threshold
        self.high_vol_pct = high_vol_percentile
        self.low_vol_pct = low_vol_percentile
        self.chop_threshold = choppiness_threshold
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        self.alignment_weight = alignment_weight
        self.stickiness = stickiness

        # Rolling ATR history for percentile calculation
        self._atr_history: deque[float] = deque(maxlen=atr_history_len)

        # HTF candle buffers — self-aggregated from 1m input
        self._htf_buffers: dict[str, deque[_AggCandle]] = {}
        self._htf_partials: dict[str, list] = {}  # Accumulating 1m candles for current bar
        self._htf_trends: dict[str, MTFTrend] = {}
        for tf, cfg in self._TF_CONFIGS.items():
            self._htf_buffers[tf] = deque(maxlen=cfg["buffer_size"])
            self._htf_partials[tf] = []
            self._htf_trends[tf] = MTFTrend()

        # Stickiness state
        self._regime_counter: int = 0
        self._pending_regime: Regime | None = None
        self._last_state = MTFRegimeState(
            regime=Regime.RANGING, confidence=0.0,
            trend_strength=0.0, volatility_pct=0.5, choppiness=0.5,
        )

        # Track which 1m candles we've already processed (by timestamp)
        self._last_processed_ts: int = 0

    def detect(self, candles) -> MTFRegimeState:
        """Detect current market regime from 1-minute candles.

        Drop-in compatible with RegimeDetector.detect(). Accepts any
        iterable of candle objects with .open, .high, .low, .close attributes.

        Args:
            candles: List/deque/tuple of 1-min candle objects.

        Returns:
            MTFRegimeState (subclass of RegimeState) with regime and metrics.
        """
        candle_list = list(candles)
        min_required = max(self.adx_period, self.atr_period, self.choppiness_lookback) + 1

        if len(candle_list) < min_required:
            return self._last_state

        # 1. Aggregate new 1m candles into HTF bars
        self._update_htf_from_1m(candle_list)

        # 2. Compute HTF trends
        for tf in self._TF_CONFIGS:
            self._compute_htf_trend(tf)

        # 3. Single-timeframe metrics (from 1m candles)
        trend_strength, trend_direction = self._compute_trend(candle_list)
        current_atr = self._compute_atr(candle_list)
        self._atr_history.append(current_atr)
        vol_percentile = self._atr_percentile(current_atr)
        choppiness = self._compute_choppiness(candle_list)

        # 4. Multi-timeframe alignment
        alignment_score, alignment_count, htf_trend = self._compute_alignment()

        # 5. Classify regime (combined)
        regime, confidence = self._classify_mtf(
            trend_strength, trend_direction, vol_percentile,
            choppiness, alignment_score, htf_trend,
        )

        # 6. Apply stickiness
        if regime != self._last_state.regime:
            if self._pending_regime == regime:
                self._regime_counter += 1
            else:
                self._pending_regime = regime
                self._regime_counter = 1

            if self._regime_counter < self.stickiness:
                regime = self._last_state.regime
                confidence = self._last_state.confidence * 0.7 + confidence * 0.3
        else:
            self._pending_regime = None
            self._regime_counter = 0

        state = MTFRegimeState(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_pct=vol_percentile,
            choppiness=choppiness,
            alignment_score=alignment_score,
            alignment_count=alignment_count,
            htf_trend=htf_trend,
            tf_details={
                tf: {"dir": t.direction, "str": round(t.strength, 3)}
                for tf, t in self._htf_trends.items()
            },
        )
        self._last_state = state
        return state

    def get_params(self, regime: Regime) -> dict:
        """Get parameter overrides for the given regime."""
        return REGIME_PARAMS.get(regime, REGIME_PARAMS[Regime.RANGING])

    # ---- HTF aggregation from 1m candles ----

    def _get_candle_ts(self, candle) -> int:
        """Extract timestamp in ms from a candle object (duck-typed)."""
        # Try common attribute names
        if hasattr(candle, "open_time"):
            return int(candle.open_time)
        if hasattr(candle, "timestamp_ms"):
            return int(candle.timestamp_ms)
        if hasattr(candle, "open_time_ms"):
            return int(candle.open_time_ms)
        return 0

    def _update_htf_from_1m(self, candles_1m: list) -> None:
        """Aggregate 1m candles into HTF bars.

        Processes only new candles (after _last_processed_ts) to avoid
        recomputing on every tick. Groups candles into HTF buckets and
        closes completed bars into the HTF buffers.
        """
        for candle in candles_1m:
            ts = self._get_candle_ts(candle)
            if ts <= self._last_processed_ts:
                continue

            for tf, cfg in self._TF_CONFIGS.items():
                tf_ms = cfg["minutes"] * 60_000
                partial = self._htf_partials[tf]

                if partial:
                    # Check if this candle belongs to a new HTF bar
                    first_ts = self._get_candle_ts(partial[0])
                    bucket_start = (first_ts // tf_ms) * tf_ms
                    current_bucket = (ts // tf_ms) * tf_ms

                    if current_bucket != bucket_start:
                        # Close the completed bar
                        self._close_htf_bar(tf, partial, bucket_start)
                        self._htf_partials[tf] = [candle]
                    else:
                        partial.append(candle)
                else:
                    partial.append(candle)

            if ts > 0:
                self._last_processed_ts = ts

    def _close_htf_bar(self, tf: str, candles_1m: list, bucket_ms: int) -> None:
        """Close a completed HTF bar and add it to the buffer."""
        if not candles_1m:
            return

        bar = _AggCandle(
            open=candles_1m[0].open,
            high=max(c.high for c in candles_1m),
            low=min(c.low for c in candles_1m),
            close=candles_1m[-1].close,
            timestamp_ms=bucket_ms,
        )
        self._htf_buffers[tf].append(bar)

    def _compute_htf_trend(self, tf: str) -> None:
        """Compute EMA crossover trend for a timeframe."""
        buffer = self._htf_buffers[tf]
        if len(buffer) < self.ema_slow_period + 5:
            return

        closes = [c.close for c in buffer]
        ema_fast = self._compute_ema(closes, self.ema_fast_period)
        ema_slow = self._compute_ema(closes, self.ema_slow_period)

        if ema_fast > ema_slow:
            direction = 1
        elif ema_fast < ema_slow:
            direction = -1
        else:
            direction = 0

        price = closes[-1]
        if price > 0:
            gap_pct = abs(ema_fast - ema_slow) / price
            strength = min(1.0, gap_pct / 0.01)
        else:
            strength = 0.0

        self._htf_trends[tf] = MTFTrend(
            direction=direction,
            strength=strength,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
        )

    # ---- Multi-timeframe alignment ----

    def _compute_alignment(self) -> tuple[float, int, int]:
        """Compute weighted trend alignment across all higher timeframes.

        Returns:
            alignment_score: -1.0 (all strongly down) to +1.0 (all strongly up).
            alignment_count: Number of TFs agreeing on direction (0-4).
            htf_trend: Dominant HTF direction (+1/-1/0).
        """
        weighted_sum = 0.0
        total_weight = 0.0
        directions: list[int] = []

        for tf, cfg in self._TF_CONFIGS.items():
            trend = self._htf_trends[tf]
            weight = cfg["weight"]
            if trend.strength > 0.05:
                weighted_sum += trend.direction * trend.strength * weight
                total_weight += weight
                directions.append(trend.direction)

        if total_weight > 0:
            alignment_score = weighted_sum / total_weight
        else:
            alignment_score = 0.0

        if directions:
            up_count = sum(1 for d in directions if d > 0)
            down_count = sum(1 for d in directions if d < 0)
            alignment_count = max(up_count, down_count)
            htf_trend = 1 if up_count > down_count else (-1 if down_count > up_count else 0)
        else:
            alignment_count = 0
            htf_trend = 0

        return alignment_score, alignment_count, htf_trend

    # ---- Single-timeframe metrics ----

    def _compute_trend(self, candles: list) -> tuple[float, int]:
        """ADX-like trend strength from 1m candles."""
        period = self.adx_period
        recent = candles[-(period + 1):]
        if len(recent) < period + 1:
            return 0.0, 0

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
            return 0.0, 0

        plus_di = plus_dm_sum / tr_sum
        minus_di = minus_dm_sum / tr_sum
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0, 0

        dx = abs(plus_di - minus_di) / di_sum
        trend_strength = min(1.0, dx)
        direction = 1 if plus_di > minus_di else (-1 if minus_di > plus_di else 0)
        return trend_strength, direction

    def _compute_atr(self, candles: list) -> float:
        """Average True Range over recent 1m candles."""
        period = self.atr_period
        recent = candles[-(period + 1):]
        if len(recent) < 2:
            return 0.0

        tr_values = []
        for i in range(1, len(recent)):
            tr = max(
                recent[i].high - recent[i].low,
                abs(recent[i].high - recent[i - 1].close),
                abs(recent[i].low - recent[i - 1].close),
            )
            tr_values.append(tr)
        return sum(tr_values) / len(tr_values) if tr_values else 0.0

    def _atr_percentile(self, current_atr: float) -> float:
        """ATR percentile within rolling history."""
        if len(self._atr_history) < 10:
            return 0.5
        rank = sum(1 for x in self._atr_history if x <= current_atr)
        return rank / len(self._atr_history)

    def _compute_choppiness(self, candles: list) -> float:
        """Direction flip rate over recent candles."""
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

    # ---- Combined classification ----

    def _classify_mtf(
        self,
        trend_strength: float,
        trend_direction: int,
        vol_percentile: float,
        choppiness: float,
        alignment_score: float,
        htf_trend: int,
    ) -> tuple[Regime, float]:
        """Classify regime using both single-TF and multi-TF signals."""
        aw = self.alignment_weight

        # Low volatility — dead market, no edge
        if vol_percentile < self.low_vol_pct:
            confidence = 1.0 - (vol_percentile / self.low_vol_pct)
            return Regime.LOW_VOLATILITY, min(1.0, confidence)

        # High volatility — explosive, unreliable
        if vol_percentile > self.high_vol_pct:
            confidence = (vol_percentile - self.high_vol_pct) / (1.0 - self.high_vol_pct)
            return Regime.HIGH_VOLATILITY, min(1.0, confidence)

        # Effective trend: blend 1m ADX with HTF alignment
        abs_alignment = abs(alignment_score)
        effective_trend = trend_strength * (1.0 - aw) + abs_alignment * aw

        # Direction: prefer HTF when 1m is weak
        if trend_strength > 0.3:
            effective_direction = trend_direction
        elif abs_alignment > 0.3:
            effective_direction = htf_trend
        else:
            effective_direction = (
                trend_direction if trend_strength > abs_alignment else htf_trend
            )

        # Choppiness damped by strong HTF alignment
        effective_chop = choppiness * (1.0 - abs_alignment * aw)

        # Trending check
        if effective_trend > self.trend_threshold and effective_chop < self.chop_threshold:
            base_conf = (
                (effective_trend - self.trend_threshold)
                / (1.0 - self.trend_threshold)
            )
            alignment_bonus = abs_alignment * aw
            confidence = min(1.0, base_conf * (1.0 - effective_chop) + alignment_bonus)

            if effective_direction >= 0:
                return Regime.TRENDING_UP, confidence
            else:
                return Regime.TRENDING_DOWN, confidence

        # Default: ranging
        confidence = max(choppiness, 1.0 - effective_trend)
        return Regime.RANGING, min(1.0, confidence * 0.8)

    # ---- Utility ----

    @staticmethod
    def _compute_ema(values: list[float], period: int) -> float:
        """Compute EMA of a price series."""
        if len(values) < period:
            return values[-1] if values else 0.0
        multiplier = 2.0 / (period + 1)
        ema = sum(values[:period]) / period
        for price in values[period:]:
            ema = (price - ema) * multiplier + ema
        return ema


def load_candles_from_csv(filepath: str) -> list:
    """Load candles from a Binance klines CSV file.

    Returns list of _AggCandle objects (used by analysis scripts).
    """
    candles = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(_AggCandle(
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                timestamp_ms=int(row["open_time_ms"]),
            ))
    return candles
