"""Backtest using real PolyBackTest Pro pricing + Binance candle data.

Uses the EXACT same signal logic as the live bot:
- Layer 1: Oracle lag (Binance spot vs Polymarket implied price)
- Layer 2: Momentum (RSI, ROC, direction consistency, volume spike, body ratio)
- Layer 3: Regime detection (ADX, ATR percentile, choppiness)
- Full signal combiner with weight schedule and redistribution

The only layers missing vs live bot:
- Layer 3 Liquidation (no historical Coinalyze/CoinGlass data yet)
- Layer 4 Orderbook signal (separate from orderbook fade strategy)
- Layer 5 Sentiment
- Coinbase cross-confirmation

Usage:
    python -m backtest.backtest_real_pricing

Reads:
    backtest/data/polybacktest_markets.csv
    backtest/data/polybacktest_snapshots.csv
    backtest/data/binance_klines_1m.csv
"""

import csv
import logging
import os
from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ─── Strategy Parameters (match live bot config.yaml) ─────────────────
FEE_RATE = 0.02          # Polymarket 2% fee on winnings only
MIN_EDGE = 0.003         # Minimum required EV
MAX_EDGE = 0.025         # Maximum required EV (at earliest entry)
MIN_CONFIDENCE = 0.02    # Minimum signal confidence
MAX_ADJUSTMENT = 0.20    # Signal → probability scaling
EARLIEST_ENTRY = 270     # Seconds remaining — start scanning
LATEST_ENTRY = 60        # Seconds remaining — stop scanning


# ─── Candle Dataclass (matches live bot) ──────────────────────────────

@dataclass
class Candle:
    """Mirror of the live bot's BinanceFeed Candle."""
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = True


# ─── Signal Classes (imported logic from live bot) ────────────────────

class MomentumSignal:
    """Exact copy of signals/momentum.py logic."""

    def __init__(
        self,
        roc_weight: float = 0.30,
        direction_weight: float = 0.25,
        volume_weight: float = 0.25,
        body_ratio_weight: float = 0.10,
        rsi_weight: float = 0.10,
        rsi_period: int = 5,
        lookback_candles: int = 10,
    ):
        self.roc_weight = roc_weight
        self.direction_weight = direction_weight
        self.volume_weight = volume_weight
        self.body_ratio_weight = body_ratio_weight
        self.rsi_weight = rsi_weight
        self.rsi_period = rsi_period
        self.lookback_candles = lookback_candles

    def compute(self, candles: list[Candle], current_candle: Candle | None = None) -> float:
        working = list(candles)
        if current_candle:
            working.append(current_candle)
        if len(working) < 2:
            return 0.0

        recent = working[-self.lookback_candles:]
        roc = self._rate_of_change(recent)
        direction = self._direction_consistency(recent)
        volume = self._volume_spike(recent)
        body = self._candle_body_ratio(recent)
        rsi = self._rsi_signal(recent)

        raw = (
            self.roc_weight * roc
            + self.direction_weight * direction
            + self.volume_weight * volume
            + self.body_ratio_weight * body
            + self.rsi_weight * rsi
        )
        return max(-1.0, min(1.0, raw))

    def _rate_of_change(self, candles: list) -> float:
        if len(candles) < 2:
            return 0.0
        roc_1 = (candles[-1].close - candles[-2].close) / candles[-2].close if candles[-2].close else 0
        roc_3 = 0.0
        if len(candles) >= 4:
            roc_3 = (candles[-1].close - candles[-4].close) / candles[-4].close if candles[-4].close else 0
        avg_roc = (roc_1 + roc_3) / 2 if len(candles) >= 4 else roc_1
        return max(-1.0, min(1.0, avg_roc / 0.001))

    def _direction_consistency(self, candles: list) -> float:
        if len(candles) < 2:
            return 0.0
        ups = sum(1 for c in candles if c.close >= c.open)
        downs = len(candles) - ups
        total = len(candles)
        if ups > downs:
            return (ups / total - 0.5) * 2
        else:
            return -((downs / total - 0.5) * 2)

    def _volume_spike(self, candles: list) -> float:
        if len(candles) < 3:
            return 0.0
        volumes = [c.volume for c in candles if c.volume > 0]
        if not volumes:
            return 0.0
        avg_vol = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[0]
        if avg_vol <= 0:
            return 0.0
        current_vol = volumes[-1]
        ratio = current_vol / avg_vol
        direction = 1.0 if candles[-1].close >= candles[-1].open else -1.0
        spike = max(-1.0, min(1.0, (ratio - 1.0)))
        return spike * direction

    def _candle_body_ratio(self, candles: list) -> float:
        ratios = []
        for c in candles[-5:]:
            body = c.close - c.open
            range_ = c.high - c.low
            if range_ > 0:
                ratios.append(body / range_)
        if not ratios:
            return 0.0
        avg = sum(ratios) / len(ratios)
        return max(-1.0, min(1.0, avg))

    def _rsi_signal(self, candles: list) -> float:
        period = min(self.rsi_period, len(candles) - 1)
        if period < 2:
            return 0.0
        gains = []
        losses = []
        for i in range(-period, 0):
            change = candles[i].close - candles[i - 1].close
            if change >= 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(change))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        if rsi > 80:
            return -((rsi - 80) / 20)
        elif rsi < 20:
            return (20 - rsi) / 20
        else:
            return (rsi - 50) / 50


class OracleLagSignal:
    """Exact copy of signals/oracle_lag.py logic."""

    def __init__(self, max_expected_lag: float = 0.001):
        self.max_expected_lag = max_expected_lag

    def compute(
        self,
        exchange_price: float,
        oracle_price: float,
        oracle_open_price: float,
    ) -> float:
        if oracle_price <= 0 or exchange_price <= 0:
            return 0.0
        lag_delta = (exchange_price - oracle_price) / oracle_price
        lag_signal = self._clamp(lag_delta / self.max_expected_lag)
        if oracle_open_price > 0:
            open_delta = (exchange_price - oracle_open_price) / oracle_open_price
            open_signal = self._clamp(open_delta / self.max_expected_lag)
            return self._clamp(0.6 * lag_signal + 0.4 * open_signal)
        return lag_signal

    @staticmethod
    def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))


class RegimeDetector:
    """Exact copy of strategy/regime_detector.py logic."""

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
        self._atr_history: deque = deque(maxlen=atr_history_len)

    def reset_history(self) -> None:
        """Reset ATR history between independent markets in backtest."""
        self._atr_history.clear()

    def detect(self, candles: list[Candle]) -> tuple[str, float, dict]:
        """Returns (regime_name, confidence, params_dict)."""
        min_required = max(self.adx_period, self.atr_period, self.choppiness_lookback) + 1
        if len(candles) < min_required:
            return "ranging", 0.0, REGIME_PARAMS["ranging"]

        trend_strength, trend_direction = self._compute_trend(candles)
        current_atr = self._compute_atr(candles)
        self._atr_history.append(current_atr)
        vol_percentile = self._atr_percentile(current_atr)
        choppiness = self._compute_choppiness(candles)

        regime, confidence = self._classify(trend_strength, trend_direction, vol_percentile, choppiness)
        return regime, confidence, REGIME_PARAMS.get(regime, REGIME_PARAMS["ranging"])

    def _compute_trend(self, candles: list) -> tuple:
        period = self.adx_period
        recent = candles[-(period + 1):]
        if len(recent) < period + 1:
            return 0.0, 0
        plus_dm_sum = minus_dm_sum = tr_sum = 0.0
        for i in range(1, len(recent)):
            high, low = recent[i].high, recent[i].low
            prev_high, prev_low = recent[i-1].high, recent[i-1].low
            prev_close = recent[i-1].close
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
        direction = 1 if plus_di > minus_di else (-1 if minus_di > plus_di else 0)
        return min(1.0, dx), direction

    def _compute_atr(self, candles: list) -> float:
        period = self.atr_period
        recent = candles[-(period + 1):]
        if len(recent) < 2:
            return 0.0
        tr_values = []
        for i in range(1, len(recent)):
            high, low = recent[i].high, recent[i].low
            prev_close = recent[i-1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        return sum(tr_values) / len(tr_values) if tr_values else 0.0

    def _atr_percentile(self, current_atr: float) -> float:
        if len(self._atr_history) < 10:
            return 0.5
        sorted_history = sorted(self._atr_history)
        rank = sum(1 for x in sorted_history if x <= current_atr)
        return rank / len(sorted_history)

    def _compute_choppiness(self, candles: list) -> float:
        recent = candles[-self.choppiness_lookback:]
        if len(recent) < 3:
            return 0.5
        flips = 0
        for i in range(1, len(recent)):
            prev_dir = 1 if recent[i-1].close >= recent[i-1].open else -1
            curr_dir = 1 if recent[i].close >= recent[i].open else -1
            if curr_dir != prev_dir:
                flips += 1
        return flips / (len(recent) - 1)

    def _classify(self, trend_strength, trend_direction, vol_percentile, choppiness):
        if vol_percentile < self.low_vol_pct:
            confidence = 1.0 - (vol_percentile / self.low_vol_pct)
            return "low_vol", min(1.0, confidence)
        if vol_percentile > self.high_vol_pct:
            confidence = (vol_percentile - self.high_vol_pct) / (1.0 - self.high_vol_pct)
            return "high_vol", min(1.0, confidence)
        if trend_strength > self.trend_threshold and choppiness < self.chop_threshold:
            confidence = (trend_strength - self.trend_threshold) / (1.0 - self.trend_threshold) * (1.0 - choppiness)
            if trend_direction >= 0:
                return "trending_up", min(1.0, confidence)
            else:
                return "trending_down", min(1.0, confidence)
        confidence = max(choppiness, 1.0 - trend_strength)
        return "ranging", min(1.0, confidence * 0.8)


# Regime parameters matching live bot
REGIME_PARAMS = {
    "trending_up": {
        "weight_adjustments": {"oracle": -0.05, "momentum": +0.10, "liquidation": 0.0, "orderbook": 0.0, "sentiment": -0.05},
        "edge_multiplier": 0.75,
        "size_multiplier": 1.0,
    },
    "trending_down": {
        "weight_adjustments": {"oracle": -0.05, "momentum": +0.08, "liquidation": +0.05, "orderbook": 0.0, "sentiment": -0.08},
        "edge_multiplier": 0.75,
        "size_multiplier": 1.0,
    },
    "ranging": {
        "weight_adjustments": {"oracle": 0.0, "momentum": -0.10, "liquidation": -0.03, "orderbook": +0.10, "sentiment": +0.03},
        "edge_multiplier": 1.5,
        "size_multiplier": 0.75,
    },
    "high_vol": {
        "weight_adjustments": {"oracle": -0.05, "momentum": 0.0, "liquidation": +0.10, "orderbook": -0.05, "sentiment": 0.0},
        "edge_multiplier": 1.8,
        "size_multiplier": 0.5,
    },
    "low_vol": {
        "weight_adjustments": {"oracle": 0.0, "momentum": 0.0, "liquidation": 0.0, "orderbook": 0.0, "sentiment": 0.0},
        "edge_multiplier": 999.0,
        "size_multiplier": 0.0,
    },
}


# ─── L4 Orderbook Signal (from snapshot data) ────────────────────────

class OrderbookSignalBacktest:
    """Simplified L4 orderbook signal using available snapshot data.

    We have: up_bid_depth_5, up_ask_depth_5, down_bid_depth_5, down_ask_depth_5,
             up_best_bid, up_best_ask, down_best_bid, down_best_ask
    Missing: deltas (flow), weighted mid, total volume, level counts
    Available sub-signals (55% of live L4 weight):
      1. Bid/Ask Imbalance (30%)
      2. Top-of-Book Pressure (15%)
      3. Book Thickness (10%)
    """

    def compute(self, snap: dict) -> float:
        """Compute L4 from snapshot data."""
        up_bid = snap.get("up_bid_depth_5") or 0
        up_ask = snap.get("up_ask_depth_5") or 0
        down_bid = snap.get("down_bid_depth_5") or 0
        down_ask = snap.get("down_ask_depth_5") or 0

        if (up_bid + up_ask + down_bid + down_ask) <= 0:
            return 0.0

        # 1. Bid/Ask Imbalance (same logic as live L4)
        bullish_vol = up_bid + down_ask
        bearish_vol = up_ask + down_bid
        total_vol = bullish_vol + bearish_vol
        if total_vol > 0:
            imbalance = (bullish_vol - bearish_vol) / total_vol
            imbalance = max(-1.0, min(1.0, imbalance / 0.3))
        else:
            imbalance = 0.0

        # 2. Top-of-Book Pressure
        top_total = up_bid + up_ask
        if top_total > 0:
            top_pressure = (up_bid - up_ask) / top_total
            top_pressure = max(-1.0, min(1.0, top_pressure / 0.3))
        else:
            top_pressure = 0.0

        # 3. Thickness directional — aligned with live bot's _book_thickness
        # Live bot uses (total_bid - total_ask) / total, NOT the same
        # bullish/bearish grouping as imbalance. Here total_bid = bids on
        # both YES and NO books, total_ask = asks on both books.
        total_bid = up_bid + down_bid
        total_ask = up_ask + down_ask
        total = total_bid + total_ask
        if total > 0:
            thickness = (total_bid - total_ask) / total
            thickness = max(-1.0, min(1.0, thickness / 0.3))
        else:
            thickness = 0.0

        # Combine with live bot sub-signal weights (normalized for available signals)
        # Live: imbalance=0.30, flow=0.25, mid=0.20, pressure=0.15, thickness=0.10
        # Available: imbalance=0.30, pressure=0.15, thickness=0.10 → normalize to sum=1
        raw = (
            0.545 * imbalance      # 0.30 / 0.55
            + 0.273 * top_pressure  # 0.15 / 0.55
            + 0.182 * thickness     # 0.10 / 0.55
        )

        return max(-1.0, min(1.0, raw))


# ─── Coinalyze Historical Lookup (L3 + L5) ───────────────────────────

class CoinalyzeLookup:
    """Loads historical Coinalyze data and provides timestamp lookups.

    Reconstructs L3 (liquidation positioning) and L5 (sentiment) from:
    - Open interest history (5-min)
    - Long/short ratio history (5-min)
    - Funding rate history (1-hour)
    """

    def __init__(self, data_dir: str):
        self.oi_data: list[tuple[int, float]] = []         # (ts, oi_usd)
        self.lsr_data: list[tuple[int, float, float]] = []  # (ts, ls_ratio, long_pct)
        self.funding_data: list[tuple[int, float]] = []     # (ts, funding_rate)
        self.oi_timestamps: list[int] = []
        self.lsr_timestamps: list[int] = []
        self.funding_timestamps: list[int] = []
        self._load(data_dir)

    def _load(self, data_dir: str) -> None:
        oi_path = os.path.join(data_dir, "coinalyze_oi_history.csv")
        lsr_path = os.path.join(data_dir, "coinalyze_lsr_history.csv")
        funding_path = os.path.join(data_dir, "coinalyze_funding_history.csv")

        if os.path.exists(oi_path):
            with open(oi_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = int(row["timestamp"])
                    oi = float(row["total_oi_usd"])
                    self.oi_data.append((ts, oi))
                    self.oi_timestamps.append(ts)
            logger.info(f"Loaded {len(self.oi_data):,} OI history points")

        if os.path.exists(lsr_path):
            with open(lsr_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = int(row["timestamp"])
                    ratio = float(row["avg_ls_ratio"])
                    long_pct = float(row["avg_long_pct"])
                    self.lsr_data.append((ts, ratio, long_pct))
                    self.lsr_timestamps.append(ts)
            logger.info(f"Loaded {len(self.lsr_data):,} LSR history points")

        if os.path.exists(funding_path):
            with open(funding_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = int(row["timestamp"])
                    rate = float(row["avg_funding_rate"])
                    self.funding_data.append((ts, rate))
                    self.funding_timestamps.append(ts)
            logger.info(f"Loaded {len(self.funding_data):,} funding history points")

    @property
    def is_available(self) -> bool:
        return len(self.lsr_data) > 0 or len(self.oi_data) > 0

    def get_snapshot(self, timestamp_iso: str) -> dict:
        """Get Coinalyze data closest to timestamp.

        Returns dict with: ls_ratio, long_pct, oi_usd, oi_change_pct, funding_rate
        """
        try:
            dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
            ts = int(dt.timestamp())
        except (ValueError, AttributeError):
            return {}

        result = {}

        # LSR lookup
        if self.lsr_timestamps:
            idx = bisect_right(self.lsr_timestamps, ts) - 1
            if 0 <= idx < len(self.lsr_data):
                _, ratio, long_pct = self.lsr_data[idx]
                result["ls_ratio"] = ratio
                result["long_pct"] = long_pct
                result["short_pct"] = 100.0 - long_pct

        # OI lookup
        if self.oi_timestamps:
            idx = bisect_right(self.oi_timestamps, ts) - 1
            if 0 <= idx < len(self.oi_data):
                _, oi = self.oi_data[idx]
                result["oi_usd"] = oi
                # OI change: compare to ~5 min ago
                prev_idx = max(0, idx - 1)
                if prev_idx != idx:
                    _, prev_oi = self.oi_data[prev_idx]
                    if prev_oi > 0:
                        result["oi_change_pct"] = ((oi - prev_oi) / prev_oi) * 100
                    else:
                        result["oi_change_pct"] = 0.0
                else:
                    result["oi_change_pct"] = 0.0

        # Funding lookup
        if self.funding_timestamps:
            idx = bisect_right(self.funding_timestamps, ts) - 1
            if 0 <= idx < len(self.funding_data):
                _, rate = self.funding_data[idx]
                result["funding_rate"] = rate

        return result

    def compute_liquidation_signal(self, snapshot: dict, price_momentum: float = 0.0) -> float:
        """Compute L3 liquidation positioning signal from Coinalyze data.

        Replicates the Coinalyze portion (35% weight) of the live L3.
        Missing: Binance live stream (50%), CoinGlass static (15%).
        """
        ls_ratio = snapshot.get("ls_ratio", 1.0)
        long_pct = snapshot.get("long_pct", 50.0)
        funding_rate = snapshot.get("funding_rate", 0.0)
        oi_change = snapshot.get("oi_change_pct", 0.0)

        signal = 0.0

        # Crowded positioning (same thresholds as live bot)
        long_ratio = long_pct / 100.0 if long_pct > 1 else long_pct
        short_ratio = 1.0 - long_ratio

        if long_ratio > 0.58:  # Crowded longs
            # Bearish: longs vulnerable to squeeze down
            crowd_signal = -((long_ratio - 0.58) / 0.12)  # 0.58-0.70 -> 0 to -1
            signal += 0.4 * max(-1.0, min(0.0, crowd_signal))
        elif short_ratio > 0.58:  # Crowded shorts
            # Bullish: shorts vulnerable to squeeze up
            crowd_signal = (short_ratio - 0.58) / 0.12
            signal += 0.4 * max(0.0, min(1.0, crowd_signal))

        # Funding rate pressure
        if abs(funding_rate) > 0.01:
            # High positive funding = bearish (expensive to hold longs)
            # High negative funding = bullish (expensive to hold shorts)
            funding_signal = -funding_rate / 0.03
            signal += 0.3 * max(-1.0, min(1.0, funding_signal))

        # OI change + momentum confirmation
        if abs(oi_change) > 2.0:
            if oi_change > 0 and price_momentum > 0:
                signal += 0.3 * min(1.0, oi_change / 5.0)  # Rising OI + bullish = confirm
            elif oi_change > 0 and price_momentum < 0:
                signal -= 0.15 * min(1.0, oi_change / 5.0)  # Rising OI + bearish = mild bearish
            elif oi_change < 0 and price_momentum < 0:
                signal -= 0.3 * min(1.0, abs(oi_change) / 5.0)  # Falling OI + bearish = capitulation
            elif oi_change < 0 and price_momentum > 0:
                signal += 0.15 * min(1.0, abs(oi_change) / 5.0)  # Falling OI + bullish = short squeeze

        return max(-1.0, min(1.0, signal))

    def compute_sentiment_signal(self, snapshot: dict, price_momentum: float = 0.0) -> float:
        """Compute L5 sentiment signal from Coinalyze data.

        Replicates signals/sentiment_signal.py logic.
        """
        ls_ratio = snapshot.get("ls_ratio", 1.0)
        long_pct = snapshot.get("long_pct", 50.0)
        funding_rate = snapshot.get("funding_rate", 0.0)
        oi_change = snapshot.get("oi_change_pct", 0.0)

        long_ratio = long_pct / 100.0 if long_pct > 1 else long_pct

        # L/S ratio contrarian (40% weight in live L5)
        lsr_signal = 0.0
        if long_ratio > 0.62:
            lsr_signal = -((long_ratio - 0.62) / 0.08)
        elif (1.0 - long_ratio) > 0.62:
            lsr_signal = ((1.0 - long_ratio) - 0.62) / 0.08
        lsr_signal = max(-1.0, min(1.0, lsr_signal))

        # OI change (30% weight in live L5)
        oi_signal = 0.0
        if abs(oi_change) > 2.0:
            oi_signal = price_momentum * min(1.0, abs(oi_change) / 5.0)
        oi_signal = max(-1.0, min(1.0, oi_signal))

        # Funding (30% weight in live L5)
        funding_signal = 0.0
        if abs(funding_rate) > 0.03:
            funding_signal = -funding_rate / 0.05
        funding_signal = max(-1.0, min(1.0, funding_signal))

        raw = 0.40 * lsr_signal + 0.30 * oi_signal + 0.30 * funding_signal
        return max(-1.0, min(1.0, raw))


# ─── Signal Combiner (exact copy from combiner.py) ───────────────────

# (time_remaining_sec, oracle_w, momentum_w, liquidation_w, orderbook_w, sentiment_w)
WEIGHT_SCHEDULE = [
    (300, 0.25, 0.30, 0.15, 0.15, 0.15),
    (180, 0.20, 0.30, 0.12, 0.23, 0.15),
    (90,  0.15, 0.30, 0.08, 0.32, 0.15),
    (30,  0.10, 0.35, 0.05, 0.35, 0.15),
]


def get_weights(seconds_remaining: float) -> tuple[float, float, float, float, float]:
    """Get (oracle, momentum, liquidation, orderbook, sentiment) weights."""
    for threshold, w1, w2, w3, w4, w5 in WEIGHT_SCHEDULE:
        if seconds_remaining >= threshold:
            return w1, w2, w3, w4, w5
    last = WEIGHT_SCHEDULE[-1]
    return last[1], last[2], last[3], last[4], last[5]


def combine_signals(
    oracle_signal: float,
    momentum_signal: float,
    seconds_remaining: float,
    orderbook_signal: float = 0.0,
    liquidation_signal: float = 0.0,
    sentiment_signal: float = 0.0,
    regime_weight_adjustments: dict | None = None,
) -> tuple[float, float]:
    """Combine all available signals into (raw_signal, est_prob_up).

    Replicates combiner.py with unavailable signal redistribution.
    Any signal passed as 0.0 has its weight redistributed to active signals.
    """
    w1, w2, w3, w4, w5 = get_weights(seconds_remaining)

    # Apply regime adjustments
    if regime_weight_adjustments:
        w1 += regime_weight_adjustments.get("oracle", 0.0)
        w2 += regime_weight_adjustments.get("momentum", 0.0)
        w3 += regime_weight_adjustments.get("liquidation", 0.0)
        w4 += regime_weight_adjustments.get("orderbook", 0.0)
        w5 += regime_weight_adjustments.get("sentiment", 0.0)
        w1 = max(0.0, w1)
        w2 = max(0.0, w2)
        w3 = max(0.0, w3)
        w4 = max(0.0, w4)
        w5 = max(0.0, w5)
        total = w1 + w2 + w3 + w4 + w5
        if total > 0:
            w1 /= total
            w2 /= total
            w3 /= total
            w4 /= total
            w5 /= total

    signals = [
        (oracle_signal, w1),
        (momentum_signal, w2),
        (liquidation_signal, w3),
        (orderbook_signal, w4),
        (sentiment_signal, w5),
    ]

    # Redistribute unavailable signal weights
    unavailable_weight = 0.0
    active_weights = []
    for sig_val, w in signals:
        if sig_val == 0.0 and w > 0:
            unavailable_weight += w
            active_weights.append(0.0)
        else:
            active_weights.append(w)

    w1, w2, w3, w4, w5 = active_weights

    if unavailable_weight > 0:
        available_total = w1 + w2 + w3 + w4 + w5
        if available_total > 0:
            scale = (available_total + unavailable_weight) / available_total
            w1 *= scale
            w2 *= scale
            w3 *= scale
            w4 *= scale
            w5 *= scale

    raw_signal = (
        w1 * oracle_signal
        + w2 * momentum_signal
        + w3 * liquidation_signal
        + w4 * orderbook_signal
        + w5 * sentiment_signal
    )

    raw_signal = max(-1.0, min(1.0, raw_signal))
    est_prob_up = 0.5 + (raw_signal * MAX_ADJUSTMENT)
    est_prob_up = max(0.05, min(0.95, est_prob_up))

    return raw_signal, est_prob_up


# ─── EV Calculation ───────────────────────────────────────────────────

def compute_ev(est_prob_up: float, yes_ask: float, no_ask: float) -> tuple[float, float]:
    """Compute EV for both sides, matching live bot entry_logic.py."""
    profit_yes = 1.0 - yes_ask
    ev_yes = (est_prob_up * (profit_yes * (1.0 - FEE_RATE))) - ((1.0 - est_prob_up) * yes_ask)

    profit_no = 1.0 - no_ask
    ev_no = ((1.0 - est_prob_up) * (profit_no * (1.0 - FEE_RATE))) - (est_prob_up * no_ask)

    return ev_yes, ev_no


def required_edge(seconds_remaining: float, regime_edge_mult: float = 1.0) -> float:
    """Dynamic threshold matching live bot, with regime multiplier."""
    time_pct = seconds_remaining / 300.0
    base = MIN_EDGE + (MAX_EDGE - MIN_EDGE) * time_pct
    return base * regime_edge_mult


# ─── Orderbook Imbalance ─────────────────────────────────────────────

def compute_orderbook_imbalance(snapshot: dict) -> float:
    """Compute order book imbalance from snapshot data."""
    up_bid = snapshot.get("up_bid_depth_5") or 0
    up_ask = snapshot.get("up_ask_depth_5") or 0
    down_bid = snapshot.get("down_bid_depth_5") or 0
    down_ask = snapshot.get("down_ask_depth_5") or 0

    total = up_bid + up_ask + down_bid + down_ask
    if total == 0:
        return 0.0

    up_imbalance = (up_bid - up_ask) / (up_bid + up_ask) if (up_bid + up_ask) > 0 else 0
    down_imbalance = (down_bid - down_ask) / (down_bid + down_ask) if (down_bid + down_ask) > 0 else 0

    return up_imbalance - down_imbalance


# ─── Trade Result ────────────────────────────────────────────────────

@dataclass
class TradeResult:
    market_id: str
    strategy: str
    side: str
    entry_price: float
    seconds_remaining: float
    est_prob_up: float
    ev: float
    won: bool
    pnl: float
    start_time: str = ""
    regime: str = ""
    size_usdc: float = 1.0
    bankroll_after: float = 0.0


# ─── Backtest Risk & Sizing (mirrors live bot exactly) ───────────────

class BacktestAccount:
    """Tracks bankroll, Kelly sizing, and risk management per strategy.

    Uses the exact same logic as the live bot's PositionSizer and
    RiskManager, with injectable time for daily resets and streak pauses.
    """

    def __init__(
        self,
        initial_bankroll: float = 100.0,
        kelly_multiplier: float = 0.25,
        min_bet_usdc: float = 1.0,
        max_bet_usdc: float = 5.0,
        daily_loss_cap_pct: float = 0.20,
        daily_loss_warn_pct: float = 0.15,
        min_bankroll: float = 10.0,
        streak_reduce_at: int = 3,
        streak_reduce_factor: float = 0.75,
        streak_heavy_reduce_at: int = 5,
        streak_heavy_reduce_factor: float = 0.5,
        streak_pause_at: int = 7,
        streak_pause_minutes: int = 30,
        streak_reset_wins: int = 3,
        low_volatility_threshold: float = 0.0001,
        high_volatility_threshold: float = 0.02,
        high_volatility_size_factor: float = 0.5,
    ):
        self.bankroll = initial_bankroll
        self.kelly_multiplier = kelly_multiplier
        self.min_bet = min_bet_usdc
        self.max_bet = max_bet_usdc
        self.min_bankroll = min_bankroll

        # Daily loss tracking
        self.daily_loss_cap_pct = daily_loss_cap_pct
        self.daily_loss_warn_pct = daily_loss_warn_pct
        self.daily_pnl = 0.0
        self.daily_start_bankroll = initial_bankroll
        self.current_day = ""

        # Streak tracking
        self.streak_reduce_at = streak_reduce_at
        self.streak_reduce_factor = streak_reduce_factor
        self.streak_heavy_reduce_at = streak_heavy_reduce_at
        self.streak_heavy_reduce_factor = streak_heavy_reduce_factor
        self.streak_pause_at = streak_pause_at
        self.streak_pause_minutes = streak_pause_minutes
        self.streak_reset_wins = streak_reset_wins
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.paused_until_ts = 0.0  # Unix timestamp

        # Volatility
        self.low_vol_threshold = low_volatility_threshold
        self.high_vol_threshold = high_volatility_threshold
        self.high_vol_size_factor = high_volatility_size_factor

    def _reset_daily(self, day_str: str) -> None:
        """Reset daily counters on new UTC day."""
        if day_str != self.current_day:
            self.daily_pnl = 0.0
            self.daily_start_bankroll = self.bankroll
            self.consecutive_losses = 0
            self.consecutive_wins = 0
            self.paused_until_ts = 0.0
            self.current_day = day_str

    def can_trade(self, market_time_str: str, market_ts: float = 0.0,
                  volatility: float = 0.005) -> tuple[bool, float]:
        """Check risk rules. Returns (can_trade, size_multiplier).

        Args:
            market_time_str: ISO timestamp for daily reset.
            market_ts: Unix timestamp for streak pause expiry.
            volatility: Recent BTC volatility for vol filter.
        """
        day_str = market_time_str[:10] if market_time_str else ""
        self._reset_daily(day_str)

        # Minimum bankroll
        if self.bankroll < self.min_bankroll:
            return False, 0.0

        # Streak pause expiry
        if self.paused_until_ts > 0 and market_ts >= self.paused_until_ts:
            self.consecutive_losses = 0
            self.consecutive_wins = 0
            self.paused_until_ts = 0.0

        # Daily loss cap (hard stop)
        if self.daily_start_bankroll > 0:
            loss_cap = self.daily_start_bankroll * self.daily_loss_cap_pct
            if self.daily_pnl < -loss_cap:
                return False, 0.0

        # Streak pause (still active)
        if self.paused_until_ts > 0 and market_ts < self.paused_until_ts:
            return False, 0.0

        # Low volatility filter
        if volatility < self.low_vol_threshold:
            return False, 0.0

        # Size multiplier from streak
        multiplier = 1.0
        if self.consecutive_losses >= self.streak_heavy_reduce_at:
            multiplier = self.streak_heavy_reduce_factor
        elif self.consecutive_losses >= self.streak_reduce_at:
            multiplier = self.streak_reduce_factor

        # Soft daily loss warning
        if self.daily_start_bankroll > 0:
            warn_cap = self.daily_start_bankroll * self.daily_loss_warn_pct
            if self.daily_pnl < -warn_cap:
                multiplier *= 0.5

        # High volatility reduction
        if volatility > self.high_vol_threshold:
            multiplier *= self.high_vol_size_factor

        multiplier = max(0.25, multiplier)
        return True, multiplier

    def compute_size(self, est_prob_win: float, share_price: float,
                     size_multiplier: float = 1.0) -> float:
        """Quarter-Kelly sizing — exact match of live PositionSizer.compute()."""
        if share_price <= 0 or share_price >= 1.0 or est_prob_win <= 0:
            return 0.0

        payout_odds = (1.0 - share_price) / share_price
        kelly_fraction = (est_prob_win * payout_odds - (1.0 - est_prob_win)) / payout_odds

        if kelly_fraction <= 0:
            return 0.0

        adjusted = kelly_fraction * self.kelly_multiplier * size_multiplier
        bet_size = adjusted * self.bankroll
        bet_size = max(self.min_bet, min(self.max_bet, bet_size))

        if bet_size > self.bankroll:
            bet_size = self.bankroll
        if bet_size < self.min_bet:
            return 0.0

        return round(bet_size, 2)

    def record_trade(self, pnl: float, market_ts: float = 0.0) -> None:
        """Update bankroll, daily P&L, and streak counters."""
        self.bankroll += pnl
        self.daily_pnl += pnl

        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            if self.consecutive_losses >= self.streak_pause_at:
                self.paused_until_ts = market_ts + self.streak_pause_minutes * 60


# ─── Pre-computed signals for a single snapshot ─────────────────────

@dataclass
class SnapshotSignals:
    """Signals computed once per (market, offset) and shared across strategies."""
    momentum: float
    oracle: float
    ob_val: float
    liq_val: float
    sent_val: float
    regime_name: str
    regime_params: dict
    prob_up: float
    secs_remaining: float
    yes_ask: float
    no_ask: float
    btc_price: float
    snap: dict


def compute_snapshot_signals(
    snap: dict,
    offset_sec: int,
    btc_start: float,
    candle_lookup: 'CandleLookup | None',
    momentum_signal: MomentumSignal,
    oracle_lag_signal: OracleLagSignal,
    regime_detector: RegimeDetector,
    ob_signal: 'OrderbookSignalBacktest | None',
    coinalyze_lookup: 'CoinalyzeLookup | None',
) -> SnapshotSignals | None:
    """Compute all signals for a single snapshot. Returns None if snapshot is invalid."""
    secs_remaining = 300 - offset_sec

    if secs_remaining > EARLIEST_ENTRY or secs_remaining < LATEST_ENTRY:
        return None

    yes_ask = snap.get("up_best_ask")
    no_ask = snap.get("down_best_ask")
    if not yes_ask or not no_ask or yes_ask <= 0 or no_ask <= 0:
        return None

    btc_price = snap.get("btc_price") or btc_start
    if not btc_price or btc_price <= 0:
        return None

    # Get real candles for this moment in time
    snapshot_time = snap.get("snapshot_time", "")
    if candle_lookup and snapshot_time:
        candles = candle_lookup.get_candles(snapshot_time, count=30)
    else:
        candles = []

    # Compute signals using real candle data
    if candles and len(candles) >= 2:
        momentum = momentum_signal.compute(candles)
        regime_name, regime_conf, regime_params = regime_detector.detect(candles)
    else:
        momentum = 0.0
        regime_name = "ranging"
        regime_params = REGIME_PARAMS["ranging"]

    # Oracle lag: Binance spot vs Polymarket implied price
    oracle = oracle_lag_signal.compute(
        exchange_price=btc_price,
        oracle_price=btc_start,
        oracle_open_price=btc_start,
    )

    # L4: Orderbook signal from snapshot data
    ob_val = 0.0
    if ob_signal:
        ob_val = ob_signal.compute(snap)

    # L3/L5: Coinalyze positioning and sentiment
    liq_val = 0.0
    sent_val = 0.0
    if coinalyze_lookup and coinalyze_lookup.is_available and snapshot_time:
        ca_snap = coinalyze_lookup.get_snapshot(snapshot_time)
        if ca_snap:
            liq_val = coinalyze_lookup.compute_liquidation_signal(ca_snap, momentum)
            sent_val = coinalyze_lookup.compute_sentiment_signal(ca_snap, momentum)

    # Combine with full weight schedule and regime adjustments
    regime_adj = regime_params.get("weight_adjustments")
    _, prob_up = combine_signals(
        oracle_signal=oracle,
        momentum_signal=momentum,
        seconds_remaining=secs_remaining,
        orderbook_signal=ob_val,
        liquidation_signal=liq_val,
        sentiment_signal=sent_val,
        regime_weight_adjustments=regime_adj,
    )

    return SnapshotSignals(
        momentum=momentum, oracle=oracle, ob_val=ob_val,
        liq_val=liq_val, sent_val=sent_val,
        regime_name=regime_name, regime_params=regime_params,
        prob_up=prob_up, secs_remaining=secs_remaining,
        yes_ask=yes_ask, no_ask=no_ask, btc_price=btc_price, snap=snap,
    )


# ─── Strategy Definitions ────────────────────────────────────────────

def strategy_contrarian_ev(
    sorted_offsets: list[int],
    snapshots_by_offset: dict[int, dict],
    precomputed: dict[int, SnapshotSignals],
    winner: str,
    start_time: str,
) -> TradeResult | None:
    """Current live strategy: buy whichever side has best EV."""
    for offset_sec in sorted_offsets:
        sig = precomputed.get(offset_sec)
        if sig is None:
            continue

        if sig.regime_name == "low_vol":
            continue

        confidence = abs(sig.prob_up - 0.5)
        if confidence < MIN_CONFIDENCE:
            continue

        ev_yes, ev_no = compute_ev(sig.prob_up, sig.yes_ask, sig.no_ask)

        if ev_yes >= ev_no:
            side, best_ev, price = "YES", ev_yes, sig.yes_ask
        else:
            side, best_ev, price = "NO", ev_no, sig.no_ask

        regime_edge_mult = sig.regime_params.get("edge_multiplier", 1.0)
        req_edge = required_edge(sig.secs_remaining, regime_edge_mult)
        if best_ev <= req_edge:
            continue

        won = (side == "YES" and winner == "Up") or (side == "NO" and winner == "Down")
        if won:
            gross_profit = 1.0 - price
            pnl = gross_profit - (gross_profit * FEE_RATE)
        else:
            pnl = -price

        return TradeResult(
            market_id=sig.snap.get("market_id", ""),
            strategy="contrarian_ev",
            side=side,
            entry_price=price,
            seconds_remaining=sig.secs_remaining,
            est_prob_up=sig.prob_up,
            ev=best_ev,
            won=won,
            pnl=pnl,
            start_time=start_time,
            regime=sig.regime_name,
        )

    return None


def strategy_signal_follow(
    sorted_offsets: list[int],
    snapshots_by_offset: dict[int, dict],
    precomputed: dict[int, SnapshotSignals],
    winner: str,
    start_time: str,
) -> TradeResult | None:
    """Buy the side our signal points to (ignores EV of cheap side)."""
    for offset_sec in sorted_offsets:
        sig = precomputed.get(offset_sec)
        if sig is None:
            continue

        if sig.regime_name == "low_vol":
            continue

        confidence = abs(sig.prob_up - 0.5)
        if confidence < MIN_CONFIDENCE:
            continue

        if sig.prob_up >= 0.5:
            side, price = "YES", sig.yes_ask
        else:
            side, price = "NO", sig.no_ask

        ev_yes, ev_no = compute_ev(sig.prob_up, sig.yes_ask, sig.no_ask)
        ev = ev_yes if side == "YES" else ev_no

        regime_edge_mult = sig.regime_params.get("edge_multiplier", 1.0)
        req_edge = required_edge(sig.secs_remaining, regime_edge_mult)
        if ev <= req_edge:
            continue

        won = (side == "YES" and winner == "Up") or (side == "NO" and winner == "Down")
        if won:
            gross_profit = 1.0 - price
            pnl = gross_profit - (gross_profit * FEE_RATE)
        else:
            pnl = -price

        return TradeResult(
            market_id=sig.snap.get("market_id", ""),
            strategy="signal_follow",
            side=side,
            entry_price=price,
            seconds_remaining=sig.secs_remaining,
            est_prob_up=sig.prob_up,
            ev=ev,
            won=won,
            pnl=pnl,
            start_time=start_time,
            regime=sig.regime_name,
        )

    return None


def strategy_orderbook_fade(
    snapshots_by_offset: dict[int, dict],
    btc_start: float,
    winner: str,
    start_time: str,
) -> TradeResult | None:
    """Fade extreme order book imbalance (@polybacktest finding)."""
    IMBALANCE_THRESHOLD = 0.3

    for offset_sec in sorted(snapshots_by_offset.keys()):
        snap = snapshots_by_offset[offset_sec]
        secs_remaining = 300 - offset_sec

        if secs_remaining > EARLIEST_ENTRY or secs_remaining < LATEST_ENTRY:
            continue

        yes_ask = snap.get("up_best_ask")
        no_ask = snap.get("down_best_ask")
        if not yes_ask or not no_ask or yes_ask <= 0 or no_ask <= 0:
            continue

        imbalance = compute_orderbook_imbalance(snap)

        if abs(imbalance) < IMBALANCE_THRESHOLD:
            continue

        # FADE the imbalance: heavy UP bids → bet DOWN
        if imbalance > 0:
            side, price = "NO", no_ask
        else:
            side, price = "YES", yes_ask

        ev_yes, ev_no = compute_ev(0.5, yes_ask, no_ask)
        ev = ev_yes if side == "YES" else ev_no

        if ev <= 0.0:
            continue

        won = (side == "YES" and winner == "Up") or (side == "NO" and winner == "Down")
        if won:
            gross_profit = 1.0 - price
            pnl = gross_profit - (gross_profit * FEE_RATE)
        else:
            pnl = -price

        return TradeResult(
            market_id=snap.get("market_id", ""),
            strategy="orderbook_fade",
            side=side,
            entry_price=price,
            seconds_remaining=secs_remaining,
            est_prob_up=0.5,
            ev=ev,
            won=won,
            pnl=pnl,
            start_time=start_time,
        )

    return None


# ─── Candle Lookup ───────────────────────────────────────────────────

class CandleLookup:
    """Efficient lookup of historical candles by timestamp.

    Pre-loads all Binance 1-min candles and provides fast lookups
    of the N most recent candles at any given point in time.
    """

    def __init__(self, klines_path: str):
        self.candles: list[Candle] = []
        self.timestamps_ms: list[int] = []  # For binary search
        self._load(klines_path)

    def _load(self, path: str) -> None:
        """Load candles from CSV."""
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candle = Candle(
                    open_time=int(row["open_time_ms"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    is_closed=True,
                )
                self.candles.append(candle)
                self.timestamps_ms.append(candle.open_time)

        logger.info(f"Loaded {len(self.candles):,} Binance 1-min candles")

    def get_candles(self, timestamp_iso: str, count: int = 30) -> list[Candle]:
        """Get the most recent `count` closed candles as of `timestamp_iso`.

        Args:
            timestamp_iso: ISO timestamp (e.g. "2026-03-15T12:34:56.123Z")
            count: Number of candles to return.

        Returns:
            List of Candle objects, oldest first.
        """
        try:
            dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
            ts_ms = int(dt.timestamp() * 1000)
        except (ValueError, AttributeError):
            return []

        # Binary search: find the rightmost candle that opened before this timestamp
        idx = bisect_right(self.timestamps_ms, ts_ms) - 1
        if idx < 0:
            return []

        start_idx = max(0, idx - count + 1)
        return self.candles[start_idx:idx + 1]


# ─── Data Loading ─────────────────────────────────────────────────────

def load_markets(path: str) -> list[dict]:
    """Load market metadata CSV."""
    markets = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["btc_price_start"] = float(row["btc_price_start"]) if row["btc_price_start"] else 0
            row["btc_price_end"] = float(row["btc_price_end"]) if row["btc_price_end"] else 0
            row["final_volume"] = float(row["final_volume"]) if row["final_volume"] else 0
            markets.append(row)
    return markets


def load_snapshots(path: str) -> dict[str, dict[int, dict]]:
    """Load snapshots CSV, grouped by market_id → offset_sec → snapshot."""
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["market_id"]
            offset = int(row["offset_sec"])
            for key in ["btc_price", "price_up", "price_down",
                        "up_best_ask", "up_best_bid", "down_best_ask", "down_best_bid",
                        "up_bid_depth_5", "up_ask_depth_5", "down_bid_depth_5", "down_ask_depth_5"]:
                val = row.get(key)
                row[key] = float(val) if val and val != "None" else None
            grouped[mid][offset] = row
    return grouped


# ─── Results Reporting ────────────────────────────────────────────────

@dataclass
class StrategyStats:
    name: str
    trades: list[TradeResult] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.won)

    @property
    def win_rate(self) -> float:
        return self.wins / self.count if self.count else 0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.count if self.count else 0

    @property
    def avg_entry_price(self) -> float:
        return sum(t.entry_price for t in self.trades) / self.count if self.count else 0

    @property
    def avg_win_pnl(self) -> float:
        wins = [t.pnl for t in self.trades if t.won]
        return sum(wins) / len(wins) if wins else 0

    @property
    def avg_loss_pnl(self) -> float:
        losses = [t.pnl for t in self.trades if not t.won]
        return sum(losses) / len(losses) if losses else 0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        peak = 0.0
        cumulative = 0.0
        max_dd = 0.0
        for t in self.trades:
            cumulative += t.pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def side_breakdown(self) -> dict:
        yes_trades = [t for t in self.trades if t.side == "YES"]
        no_trades = [t for t in self.trades if t.side == "NO"]
        return {
            "YES": {
                "count": len(yes_trades),
                "wr": sum(1 for t in yes_trades if t.won) / len(yes_trades) if yes_trades else 0,
                "pnl": sum(t.pnl for t in yes_trades),
                "avg_price": sum(t.entry_price for t in yes_trades) / len(yes_trades) if yes_trades else 0,
            },
            "NO": {
                "count": len(no_trades),
                "wr": sum(1 for t in no_trades if t.won) / len(no_trades) if no_trades else 0,
                "pnl": sum(t.pnl for t in no_trades),
                "avg_price": sum(t.entry_price for t in no_trades) / len(no_trades) if no_trades else 0,
            },
        }

    def time_breakdown(self) -> dict:
        """WR and PnL by seconds remaining bucket."""
        buckets: dict[str, list[TradeResult]] = defaultdict(list)
        for t in self.trades:
            if t.seconds_remaining >= 240:
                buckets["240-270s"].append(t)
            elif t.seconds_remaining >= 180:
                buckets["180-240s"].append(t)
            elif t.seconds_remaining >= 120:
                buckets["120-180s"].append(t)
            else:
                buckets["60-120s"].append(t)

        result = {}
        for bucket, trades in sorted(buckets.items()):
            wins = sum(1 for t in trades if t.won)
            result[bucket] = {
                "count": len(trades),
                "wr": wins / len(trades) if trades else 0,
                "pnl": sum(t.pnl for t in trades),
            }
        return result

    def day_of_week_breakdown(self) -> dict:
        """WR and PnL by day of week."""
        DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        buckets: dict[str, list[TradeResult]] = defaultdict(list)
        for t in self.trades:
            if t.start_time:
                try:
                    dt = datetime.fromisoformat(t.start_time.replace("Z", "+00:00"))
                    day_name = DAY_NAMES[dt.weekday()]
                    buckets[day_name].append(t)
                except (ValueError, IndexError):
                    pass

        result = {}
        for day in DAY_NAMES:
            trades = buckets.get(day, [])
            if trades:
                wins = sum(1 for t in trades if t.won)
                result[day] = {
                    "count": len(trades),
                    "wr": wins / len(trades),
                    "pnl": sum(t.pnl for t in trades),
                    "avg_pnl": sum(t.pnl for t in trades) / len(trades),
                }
        return result

    def hour_breakdown(self) -> dict:
        """WR and PnL by hour of day (UTC)."""
        buckets: dict[int, list[TradeResult]] = defaultdict(list)
        for t in self.trades:
            if t.start_time:
                try:
                    dt = datetime.fromisoformat(t.start_time.replace("Z", "+00:00"))
                    buckets[dt.hour].append(t)
                except (ValueError, IndexError):
                    pass

        result = {}
        for hour in sorted(buckets.keys()):
            trades = buckets[hour]
            wins = sum(1 for t in trades if t.won)
            result[f"{hour:02d}:00"] = {
                "count": len(trades),
                "wr": wins / len(trades) if trades else 0,
                "pnl": sum(t.pnl for t in trades),
            }
        return result

    def regime_breakdown(self) -> dict:
        """WR and PnL by detected regime."""
        buckets: dict[str, list[TradeResult]] = defaultdict(list)
        for t in self.trades:
            if t.regime:
                buckets[t.regime].append(t)

        result = {}
        for regime, trades in sorted(buckets.items()):
            wins = sum(1 for t in trades if t.won)
            result[regime] = {
                "count": len(trades),
                "wr": wins / len(trades) if trades else 0,
                "pnl": sum(t.pnl for t in trades),
                "avg_pnl": sum(t.pnl for t in trades) / len(trades),
            }
        return result


def print_report(stats: StrategyStats, total_markets: int) -> None:
    """Print detailed strategy report."""
    print(f"\n{'='*60}")
    print(f"  Strategy: {stats.name}")
    print(f"{'='*60}")
    print(f"  Trades:           {stats.count:,}")
    print(f"  Participation:    {stats.count/total_markets*100:.1f}% ({stats.count}/{total_markets})")
    avg_size = sum(t.size_usdc for t in stats.trades) / stats.count if stats.count else 0
    final_bankroll = stats.trades[-1].bankroll_after if stats.trades else 100.0
    print(f"  Win Rate:         {stats.win_rate*100:.1f}%")
    print(f"  Total PnL:        ${stats.total_pnl:+,.2f}")
    print(f"  Final Bankroll:   ${final_bankroll:,.2f}  (ROI: {(final_bankroll - 100)/100*100:+.0f}%)")
    print(f"  Avg Bet Size:     ${avg_size:.2f}")
    print(f"  Avg PnL/trade:    ${stats.avg_pnl:+,.4f}")
    print(f"  Avg Entry Price:  ${stats.avg_entry_price:.3f}")
    print(f"  Avg Win:          ${stats.avg_win_pnl:+,.4f}")
    print(f"  Avg Loss:         ${stats.avg_loss_pnl:+,.4f}")
    print(f"  Profit Factor:    {stats.profit_factor:.3f}")
    print(f"  Max Drawdown:     ${stats.max_drawdown:,.2f}")

    sides = stats.side_breakdown()
    print(f"\n  Side Breakdown:")
    for side_name, s in sides.items():
        print(f"    {side_name}: {s['count']} trades, {s['wr']*100:.1f}% WR, "
              f"${s['pnl']:+,.2f} PnL, avg price ${s['avg_price']:.3f}")

    timing = stats.time_breakdown()
    print(f"\n  Entry Timing:")
    for bucket, b in timing.items():
        print(f"    {bucket}: {b['count']} trades, {b['wr']*100:.1f}% WR, ${b['pnl']:+,.2f}")

    days = stats.day_of_week_breakdown()
    print(f"\n  Day of Week:")
    for day, d in days.items():
        print(f"    {day}: {d['count']:>5} trades, {d['wr']*100:>5.1f}% WR, "
              f"${d['pnl']:>+9,.2f}, avg ${d['avg_pnl']:>+.4f}/trade")

    hours = stats.hour_breakdown()
    print(f"\n  Hour of Day (UTC):")
    for hour, h in hours.items():
        bar = "#" * max(1, int(h['count'] / 20))
        print(f"    {hour}: {h['count']:>5} trades, {h['wr']*100:>5.1f}% WR, "
              f"${h['pnl']:>+9,.2f}  {bar}")

    regimes = stats.regime_breakdown()
    if regimes:
        print(f"\n  Regime Breakdown:")
        for regime, r in regimes.items():
            print(f"    {regime:<16} {r['count']:>5} trades, {r['wr']*100:>5.1f}% WR, "
                  f"${r['pnl']:>+9,.2f}, avg ${r['avg_pnl']:>+.4f}/trade")

    print()


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    markets_path = os.path.join(DATA_DIR, "polybacktest_markets.csv")
    snapshots_path = os.path.join(DATA_DIR, "polybacktest_snapshots.csv")
    klines_path = os.path.join(DATA_DIR, "binance_klines_1m.csv")

    for path, name in [(markets_path, "Markets"), (snapshots_path, "Snapshots")]:
        if not os.path.exists(path):
            logger.error(f"{name} file not found: {path}")
            return

    logger.info("Loading data...")
    markets = load_markets(markets_path)
    snapshots = load_snapshots(snapshots_path)

    # Load Binance candle data if available
    candle_lookup = None
    if os.path.exists(klines_path):
        candle_lookup = CandleLookup(klines_path)
    else:
        logger.warning(f"Binance klines not found: {klines_path}")
        logger.warning("Running without real candle data — momentum signal will be degraded")

    logger.info(f"Loaded {len(markets)} markets, {sum(len(v) for v in snapshots.values())} snapshots")

    # Load Coinalyze historical data if available
    coinalyze = CoinalyzeLookup(DATA_DIR)

    # Initialize signal processors (same params as live bot)
    momentum_sig = MomentumSignal()
    oracle_sig = OracleLagSignal()
    regime_det = RegimeDetector()
    ob_sig = OrderbookSignalBacktest()

    # Run all strategies with independent bankroll/risk tracking
    strategies = {
        "contrarian_ev": StrategyStats("Contrarian EV (current live)"),
        "signal_follow": StrategyStats("Signal Follow"),
        "orderbook_fade": StrategyStats("Order Book Fade"),
    }

    # Each strategy gets its own BacktestAccount (independent bankroll + risk)
    accounts = {
        "contrarian_ev": BacktestAccount(),
        "signal_follow": BacktestAccount(),
        "orderbook_fade": BacktestAccount(),
    }

    skipped_no_snapshots = 0

    for i, market in enumerate(markets):
        if i % 1000 == 0 and i > 0:
            logger.info(f"Processing market {i:,}/{len(markets):,}...")

        mid = market["market_id"]
        winner = market.get("winner")
        btc_start = market.get("btc_price_start", 0)

        if not winner or not btc_start:
            continue

        snaps = snapshots.get(mid, {})
        if not snaps:
            skipped_no_snapshots += 1
            continue

        start_time = market.get("start_time", "")

        # Compute market timestamp for risk manager streak pauses
        market_ts = 0.0
        if start_time:
            try:
                from datetime import datetime, timezone
                market_ts = datetime.fromisoformat(start_time.replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                pass

        # Compute volatility from candles for risk manager
        volatility = 0.005  # Default mid-range
        if candle_lookup and start_time:
            vol_candles = candle_lookup.get_candles(start_time, count=15)
            if vol_candles and len(vol_candles) >= 2:
                high = max(c.high for c in vol_candles)
                low = min(c.low for c in vol_candles)
                mid_price = (high + low) / 2
                if mid_price > 0:
                    volatility = (high - low) / mid_price

        # Reset regime detector ATR history per market to prevent
        # cross-market data leakage in backtest
        regime_det.reset_history()

        # Pre-compute signals once per (market, offset) — shared by all strategies
        sorted_offsets = sorted(snaps.keys())
        precomputed: dict[int, SnapshotSignals] = {}
        for offset_sec in sorted_offsets:
            snap = snaps[offset_sec]
            sig = compute_snapshot_signals(
                snap, offset_sec, btc_start,
                candle_lookup, momentum_sig, oracle_sig, regime_det,
                ob_sig, coinalyze,
            )
            if sig is not None:
                precomputed[offset_sec] = sig

        # Strategy 1: Contrarian EV
        result = strategy_contrarian_ev(
            sorted_offsets, snaps, precomputed, winner, start_time,
        )
        if result:
            acct = accounts["contrarian_ev"]
            can, mult = acct.can_trade(start_time, market_ts, volatility)
            if can:
                # est_prob for sizing: probability of the CHOSEN side winning
                est_prob_win = result.est_prob_up if result.side == "YES" else (1.0 - result.est_prob_up)
                size = acct.compute_size(est_prob_win, result.entry_price, mult)
                if size > 0:
                    num_shares = size / result.entry_price
                    result.pnl *= num_shares
                    result.size_usdc = size
                    acct.record_trade(result.pnl, market_ts)
                    result.bankroll_after = acct.bankroll
                    strategies["contrarian_ev"].trades.append(result)

        # Strategy 2: Signal Follow
        result = strategy_signal_follow(
            sorted_offsets, snaps, precomputed, winner, start_time,
        )
        if result:
            acct = accounts["signal_follow"]
            can, mult = acct.can_trade(start_time, market_ts, volatility)
            if can:
                est_prob_win = result.est_prob_up if result.side == "YES" else (1.0 - result.est_prob_up)
                size = acct.compute_size(est_prob_win, result.entry_price, mult)
                if size > 0:
                    num_shares = size / result.entry_price
                    result.pnl *= num_shares
                    result.size_usdc = size
                    acct.record_trade(result.pnl, market_ts)
                    result.bankroll_after = acct.bankroll
                    strategies["signal_follow"].trades.append(result)

        # Strategy 3: Order Book Fade
        result = strategy_orderbook_fade(snaps, btc_start, winner, start_time)
        if result:
            acct = accounts["orderbook_fade"]
            can, mult = acct.can_trade(start_time, market_ts, volatility)
            if can:
                est_prob_win = result.est_prob_up if result.side == "YES" else (1.0 - result.est_prob_up)
                size = acct.compute_size(est_prob_win, result.entry_price, mult)
                if size > 0:
                    num_shares = size / result.entry_price
                    result.pnl *= num_shares
                    result.size_usdc = size
                    acct.record_trade(result.pnl, market_ts)
                    result.bankroll_after = acct.bankroll
                    strategies["orderbook_fade"].trades.append(result)

    total = len(markets) - skipped_no_snapshots
    logger.info(f"Backtested {total} markets ({skipped_no_snapshots} skipped - no snapshots)")

    # Build layer status
    layers = []
    layers.append("L1:Oracle" + (" (approx)" if True else ""))
    layers.append("L2:Momentum" + (" OK" if candle_lookup else " MISSING"))
    layers.append("L3:Liquidation" + (" OK" if coinalyze.is_available else " MISSING"))
    layers.append("L4:Orderbook OK")
    layers.append("L5:Sentiment" + (" OK" if coinalyze.is_available else " MISSING"))
    layer_str = " | ".join(layers)

    print(f"\n{'#'*60}")
    print(f"  BACKTEST RESULTS - {total} markets with real pricing")
    print(f"  Layers: {layer_str}")
    print(f"  Data: PolyBackTest Pro + Binance klines + Coinalyze")
    print(f"  Parameters: min_edge={MIN_EDGE}, max_edge={MAX_EDGE}, "
          f"fee={FEE_RATE}, max_adj={MAX_ADJUSTMENT}")
    print(f"  Sizing: Quarter-Kelly, $1-$5, bankroll=$100 start")
    print(f"  Risk: 20% daily cap, streak reduce@3/5, pause@7")
    print(f"{'#'*60}")

    for stats in strategies.values():
        print_report(stats, total)

    # Summary comparison table
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Strategy':<30} {'Trades':>7} {'WR':>7} {'PnL':>10} {'PF':>7} {'Final$':>8} {'AvgBet':>7}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*10} {'-'*7} {'-'*8} {'-'*7}")
    for key, stats in strategies.items():
        acct = accounts[key]
        avg_bet = sum(t.size_usdc for t in stats.trades) / stats.count if stats.count else 0
        print(f"  {stats.name:<30} {stats.count:>7,} {stats.win_rate*100:>6.1f}% "
              f"${stats.total_pnl:>+9,.2f} {stats.profit_factor:>7.3f} "
              f"${acct.bankroll:>7,.0f} ${avg_bet:>5.2f}")

    # Statistical significance
    print(f"\n{'='*60}")
    print(f"  STATISTICAL SIGNIFICANCE")
    print(f"{'='*60}")
    import math
    for stats in strategies.values():
        if stats.count > 0:
            z = (stats.wins - stats.count * 0.5) / math.sqrt(stats.count * 0.25)
            # Approximate p-value from z-score
            if z > 0:
                conf = "ABOVE 50%" if z > 1.96 else "not significant"
            else:
                conf = "BELOW 50%" if z < -1.96 else "not significant"
            print(f"  {stats.name:<30} z={z:>+6.2f}  ({conf})")


if __name__ == "__main__":
    main()
