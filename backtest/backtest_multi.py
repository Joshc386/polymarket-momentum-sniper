"""Multi-strategy backtest engine with real Chainlink oracle data.

Runs all 5 bot strategies through identical historical data and
produces per-strategy performance metrics for comparison.

Strategies:
  A: Contrarian EV (baseline)
  B: Kalman EV (KF-dominant weights + velocity filter)
  C: HMM Regime EV (trending-only filter)
  D: Enhanced EV (multi-timeframe + cross-exchange + funding)
  E: OU Reversion EV (mean-reversion on Binance-Oracle spread)

Data sources:
  - Binance 1-min klines (momentum, candle signals)
  - Chainlink oracle prices (real update timestamps)
  - Coinbase 1-min candles (cross-exchange, KF fusion)
  - PolyBackTest Pro snapshots (market odds for entry)
  - Coinalyze history (OI, L/S ratio, funding rates)

Usage:
    python -m backtest.backtest_multi
    python -m backtest.backtest_multi --strategies a,b,e
    python -m backtest.backtest_multi --start 2026-03-01 --end 2026-04-01
"""

import argparse
import csv
import json
import logging
import math
import os
from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# --- Strategy Parameters (defaults from config_multi.yaml) ----------

FEE_RATE = 0.02
MAX_ADJUSTMENT = 0.20
EARLIEST_ENTRY = 270  # seconds remaining
LATEST_ENTRY = 60


# --- Candle (shared, lightweight) ------------------------------------

@dataclass
class Candle:
    """1-min candle for signal computation."""
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = True


# --- Data Lookups ----------------------------------------------------

class CandleLookup:
    """Binary-searchable Binance candle history."""

    def __init__(self, path: str):
        self.candles: list[Candle] = []
        self.timestamps_ms: list[int] = []
        self._load(path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                c = Candle(
                    open_time=int(row["open_time_ms"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                self.candles.append(c)
                self.timestamps_ms.append(c.open_time)
        logger.info(f"Loaded {len(self.candles):,} Binance 1-min candles")

    def get_candles(self, ts_ms: int, count: int = 30) -> list[Candle]:
        """Get the most recent `count` closed candles as of `ts_ms`."""
        idx = bisect_right(self.timestamps_ms, ts_ms) - 1
        if idx < 0:
            return []
        start = max(0, idx - count + 1)
        return self.candles[start:idx + 1]

    def get_candles_iso(self, iso_str: str, count: int = 30) -> list[Candle]:
        """Get candles using ISO timestamp string."""
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return self.get_candles(int(dt.timestamp() * 1000), count)
        except (ValueError, AttributeError):
            return []

    def get_price_at(self, ts_ms: int) -> float:
        """Get the closest Binance close price at a timestamp."""
        idx = bisect_right(self.timestamps_ms, ts_ms) - 1
        if 0 <= idx < len(self.candles):
            return self.candles[idx].close
        return 0.0


class OracleLookup:
    """Binary-searchable historical Chainlink oracle prices."""

    def __init__(self, path: str):
        self.timestamps: list[int] = []
        self.prices: list[float] = []
        self._load(path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning(f"Oracle history not found: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.timestamps.append(int(row["timestamp"]))
                self.prices.append(float(row["price"]))
        logger.info(f"Loaded {len(self.timestamps):,} Chainlink oracle updates")

    @property
    def is_available(self) -> bool:
        return len(self.timestamps) > 0

    def get_price(self, ts: int) -> float:
        """Get oracle price at timestamp (forward-fill from last update)."""
        if not self.timestamps:
            return 0.0
        idx = bisect_right(self.timestamps, ts) - 1
        if idx < 0:
            return 0.0
        return self.prices[idx]

    def get_updates_in_range(self, start_ts: int, end_ts: int) -> list[tuple[int, float]]:
        """Get all oracle updates within [start_ts, end_ts]."""
        lo = bisect_right(self.timestamps, start_ts - 1)
        hi = bisect_right(self.timestamps, end_ts)
        return [(self.timestamps[i], self.prices[i]) for i in range(lo, hi)]


class CoinbaseLookup:
    """Binary-searchable historical Coinbase candle prices."""

    def __init__(self, path: str):
        self.timestamps: list[int] = []
        self.prices: list[float] = []  # close prices
        self._load(path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning(f"Coinbase history not found: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.timestamps.append(int(row["timestamp"]))
                self.prices.append(float(row["close"]))
        logger.info(f"Loaded {len(self.timestamps):,} Coinbase 1-min candles")

    @property
    def is_available(self) -> bool:
        return len(self.timestamps) > 0

    def get_price(self, ts: int) -> float:
        """Get Coinbase price at timestamp (forward-fill)."""
        if not self.timestamps:
            return 0.0
        idx = bisect_right(self.timestamps, ts) - 1
        if idx < 0:
            return 0.0
        return self.prices[idx]


# Reuse CoinalyzeLookup from existing backtest
from backtest.backtest_real_pricing import (
    CoinalyzeLookup,
    BacktestAccount,
    OrderbookSignalBacktest,
)


# --- Trade Result ----------------------------------------------------

@dataclass
class TradeResult:
    """Single trade from backtest."""
    market_id: str
    strategy: str
    side: str
    entry_price: float
    seconds_remaining: float
    est_prob_up: float
    prob_edge: float
    won: bool
    pnl: float
    start_time: str = ""
    regime: str = ""
    size_usdc: float = 1.0
    bankroll_after: float = 0.0
    signals: dict = field(default_factory=dict)


# --- Strategy Stats (extended from existing) ------------------------

@dataclass
class StrategyStats:
    """Comprehensive statistics for one strategy's backtest run."""
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
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        peak = cumulative = 0.0
        max_dd = 0.0
        for t in self.trades:
            cumulative += t.pnl
            peak = max(peak, cumulative)
            max_dd = max(max_dd, peak - cumulative)
        return max_dd

    @property
    def sharpe_ratio(self) -> float:
        """Annualised Sharpe (assuming ~288 trades/day at 5m windows)."""
        if self.count < 10:
            return 0.0
        pnls = [t.pnl for t in self.trades]
        mean = sum(pnls) / len(pnls)
        var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        std = math.sqrt(var) if var > 0 else 1e-10
        return (mean / std) * math.sqrt(288 * 365)

    def day_of_week_breakdown(self) -> dict:
        DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        buckets: dict[str, list] = defaultdict(list)
        for t in self.trades:
            if t.start_time:
                try:
                    dt = datetime.fromisoformat(t.start_time.replace("Z", "+00:00"))
                    buckets[DAYS[dt.weekday()]].append(t)
                except (ValueError, IndexError):
                    pass
        result = {}
        for day in DAYS:
            trades = buckets.get(day, [])
            if trades:
                wins = sum(1 for t in trades if t.won)
                result[day] = {
                    "count": len(trades),
                    "wr": wins / len(trades),
                    "pnl": sum(t.pnl for t in trades),
                }
        return result


# --- Signal Imports --------------------------------------------------

from signals.oracle_lag import OracleLagSignal
from signals.momentum import MomentumSignal
from signals.combiner import SignalCombiner
from signals.kalman_filter import KalmanFilterSignal, KalmanOutput
from signals.hmm_regime import HmmRegimeDetector
from signals.momentum_slope import MomentumSlopeSignal
from signals.cross_exchange import CrossExchangeSignal
from signals.funding_filter import FundingRegimeFilter
from signals.ou_spread import OUSpreadSignal


# --- Prob-Edge Entry Logic (matches live bot entry_logic.py) --------

def evaluate_entry(
    est_prob_up: float,
    yes_ask: float,
    no_ask: float,
    yes_bid: float,
    no_bid: float,
    seconds_remaining: float,
    min_edge: float,
    max_edge: float,
    fee_rate: float = FEE_RATE,
    min_confidence: float = 0.02,
    regime_edge_mult: float = 1.0,
) -> dict | None:
    """Evaluate entry using probability-edge (not raw dollar-EV).

    Returns dict with side, prob_edge, ev, price or None if no entry.
    """
    confidence = abs(est_prob_up - 0.5)
    if confidence < min_confidence:
        return None

    # Market-implied probability from orderbook midpoint
    market_mid_yes = (yes_bid + yes_ask) / 2.0
    market_prob_up = max(0.05, min(0.95, market_mid_yes))
    market_prob_down = 1.0 - market_prob_up

    # Probability edge
    prob_edge_yes = est_prob_up - market_prob_up
    prob_edge_no = market_prob_up - est_prob_up  # = -(prob_edge_yes)

    if prob_edge_yes >= 0:
        side = "YES"
        prob_edge = prob_edge_yes
        price = yes_ask
        est_prob_win = est_prob_up
    else:
        side = "NO"
        prob_edge = abs(prob_edge_no)
        price = no_ask
        est_prob_win = 1.0 - est_prob_up

    if price <= 0 or price >= 1.0:
        return None

    # Dollar EV (still computed for logging, not for gating)
    profit = 1.0 - price
    ev = (est_prob_win * profit * (1.0 - fee_rate)) - ((1.0 - est_prob_win) * price)

    # Dynamic edge threshold
    time_pct = seconds_remaining / 300.0
    required_edge = min_edge + (max_edge - min_edge) * time_pct
    required_edge *= regime_edge_mult

    if prob_edge < required_edge or ev <= 0:
        return None

    return {
        "side": side,
        "prob_edge": prob_edge,
        "ev": ev,
        "price": price,
        "est_prob_win": est_prob_win,
    }


# --- Strategy Runners -----------------------------------------------

class BaseStrategy:
    """Base class with common entry/sizing logic."""

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        entry = cfg.get("entry", {})
        self.min_edge = entry.get("min_edge", 0.01)
        self.max_edge = entry.get("max_edge", 0.06)
        self.fee_adjustment = entry.get("fee_adjustment", FEE_RATE)
        self.min_confidence = entry.get("min_confidence", 0.02)
        self.weekend_flip = cfg.get("weekend_flip", False)

        sizing = cfg.get("sizing", {})
        risk = cfg.get("risk", {})
        self.account = BacktestAccount(
            initial_bankroll=sizing.get("initial_bankroll", 100.0),
            kelly_multiplier=sizing.get("kelly_multiplier", 0.25),
            min_bet_usdc=sizing.get("min_bet_usdc", 1.0),
            max_bet_usdc=sizing.get("max_bet_usdc", 5.0),
            daily_loss_cap_pct=risk.get("daily_loss_cap_pct", 0.20),
            daily_loss_warn_pct=risk.get("daily_loss_warn_pct", 0.15),
            min_bankroll=risk.get("min_bankroll", 10.0),
            streak_reduce_at=risk.get("streak_reduce_at", 3),
            streak_reduce_factor=risk.get("streak_reduce_factor", 0.75),
            streak_heavy_reduce_at=risk.get("streak_heavy_reduce_at", 5),
            streak_heavy_reduce_factor=risk.get("streak_heavy_reduce_factor", 0.5),
            streak_pause_at=risk.get("streak_pause_at", 7),
            streak_pause_minutes=risk.get("streak_pause_minutes", 30),
            streak_reset_wins=risk.get("streak_reset_wins", 3),
        )
        self.stats = StrategyStats(name)

    def reset_window(self) -> None:
        """Override in subclass for per-window state reset."""
        pass

    def process_tick(
        self,
        ts: int,
        binance_price: float,
        oracle_price: float,
        coinbase_price: float,
        candles: list[Candle],
    ) -> None:
        """Override in subclass for continuous signal updates."""
        pass

    def _try_entry(
        self,
        snap: dict,
        est_prob_up: float,
        winner: str,
        start_time: str,
        market_id: str,
        regime_edge_mult: float = 1.0,
        regime_name: str = "",
        extra_signals: dict | None = None,
    ) -> TradeResult | None:
        """Shared entry evaluation + sizing + risk check."""
        secs_remaining = snap.get("secs_remaining", 0)
        yes_ask = snap.get("up_best_ask", 0)
        no_ask = snap.get("down_best_ask", 0)
        yes_bid = snap.get("up_best_bid", 0)
        no_bid = snap.get("down_best_bid", 0)

        if not yes_ask or not no_ask or yes_ask <= 0 or no_ask <= 0:
            return None

        entry = evaluate_entry(
            est_prob_up=est_prob_up,
            yes_ask=yes_ask,
            no_ask=no_ask,
            yes_bid=yes_bid if yes_bid else yes_ask,
            no_bid=no_bid if no_bid else no_ask,
            seconds_remaining=secs_remaining,
            min_edge=self.min_edge,
            max_edge=self.max_edge,
            fee_rate=self.fee_adjustment,
            min_confidence=self.min_confidence,
            regime_edge_mult=regime_edge_mult,
        )

        if not entry:
            return None

        # Risk check
        market_ts = 0.0
        if start_time:
            try:
                market_ts = datetime.fromisoformat(
                    start_time.replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, TypeError):
                pass

        can_trade, size_mult = self.account.can_trade(start_time, market_ts)
        if not can_trade:
            return None

        # Size position
        size = self.account.compute_size(
            entry["est_prob_win"], entry["price"], size_mult
        )
        if size <= 0:
            return None

        # Compute P&L
        side = entry["side"]
        price = entry["price"]
        won = (side == "YES" and winner == "Up") or (side == "NO" and winner == "Down")
        if won:
            gross_profit = 1.0 - price
            pnl_per_share = gross_profit - (gross_profit * self.fee_adjustment)
        else:
            pnl_per_share = -price

        num_shares = size / price
        pnl = pnl_per_share * num_shares

        self.account.record_trade(pnl, market_ts)

        result = TradeResult(
            market_id=market_id,
            strategy=self.name,
            side=side,
            entry_price=price,
            seconds_remaining=secs_remaining,
            est_prob_up=est_prob_up,
            prob_edge=entry["prob_edge"],
            won=won,
            pnl=pnl,
            start_time=start_time,
            regime=regime_name,
            size_usdc=size,
            bankroll_after=self.account.bankroll,
            signals=extra_signals or {},
        )
        self.stats.trades.append(result)
        return result


# -- Bot A: Contrarian EV ---------------------------------------------

class StrategyA(BaseStrategy):
    """Baseline contrarian EV strategy."""

    def __init__(self, cfg: dict):
        super().__init__("A: Contrarian EV", cfg)
        sig = cfg.get("signals", {})
        ol_cfg = sig.get("oracle_lag", {})
        self.oracle_lag = OracleLagSignal(
            max_expected_lag=ol_cfg.get("max_expected_lag", 0.001)
        )
        self.momentum = MomentumSignal()
        self.combiner = SignalCombiner(
            max_adjustment=sig.get("max_adjustment", 0.20),
        )
        self.ob_signal = OrderbookSignalBacktest()

    def evaluate_snapshot(
        self,
        snap: dict,
        btc_start: float,
        candles: list[Candle],
        oracle_price: float,
        winner: str,
        start_time: str,
        market_id: str,
        coinalyze: CoinalyzeLookup | None = None,
        is_weekend: bool = False,
    ) -> TradeResult | None:
        secs_remaining = snap.get("secs_remaining", 0)
        if secs_remaining > EARLIEST_ENTRY or secs_remaining < LATEST_ENTRY:
            return None

        btc_price = snap.get("btc_price") or btc_start

        # Signals
        oracle_sig = self.oracle_lag.compute(btc_price, oracle_price, btc_start)
        momentum_sig = self.momentum.compute(candles) if candles and len(candles) >= 2 else 0.0
        ob_sig = self.ob_signal.compute(snap)

        liq_sig = sent_sig = 0.0
        snapshot_time = snap.get("snapshot_time", "")
        if coinalyze and coinalyze.is_available and snapshot_time:
            ca = coinalyze.get_snapshot(snapshot_time)
            if ca:
                liq_sig = coinalyze.compute_liquidation_signal(ca, momentum_sig)
                sent_sig = coinalyze.compute_sentiment_signal(ca, momentum_sig)

        flip = self.weekend_flip and is_weekend
        _, est_prob_up = self.combiner.combine(
            oracle_lag_signal=oracle_sig,
            momentum_signal=momentum_sig,
            liquidation_signal=liq_sig,
            seconds_remaining=secs_remaining,
            orderbook_signal=ob_sig,
            sentiment_signal=sent_sig,
            flip_signal=flip,
        )

        return self._try_entry(
            snap, est_prob_up, winner, start_time, market_id,
            extra_signals={"oracle": oracle_sig, "momentum": momentum_sig,
                           "ob": ob_sig, "liq": liq_sig, "sent": sent_sig},
        )


# -- Bot B: Kalman EV -------------------------------------------------

class StrategyB(BaseStrategy):
    """Kalman filter EV strategy with velocity confirmation."""

    def __init__(self, cfg: dict):
        super().__init__("B: Kalman EV", cfg)
        sig = cfg.get("signals", {})
        kf_cfg = sig.get("kalman_filter", {})
        self.kf = KalmanFilterSignal(
            process_noise_scale=kf_cfg.get("process_noise_scale", 1e-5),
            measurement_noise_binance=kf_cfg.get("measurement_noise_binance", 1e-4),
            measurement_noise_coinbase=kf_cfg.get("measurement_noise_coinbase", 2e-4),
            measurement_noise_oracle=kf_cfg.get("measurement_noise_oracle", 5e-4),
            adaptive_window=kf_cfg.get("adaptive_window", 30),
            breakout_velocity_threshold=kf_cfg.get("breakout_velocity_threshold", 0.5),
            max_expected_lag=kf_cfg.get("max_expected_lag", 0.001),
        )
        self.momentum = MomentumSignal()
        self.combiner = SignalCombiner(
            max_adjustment=sig.get("max_adjustment", 0.20),
            weight_schedule_name=sig.get("weight_schedule", "kf_dominant"),
        )
        self.ob_signal = OrderbookSignalBacktest()
        self.require_velocity = cfg.get("entry", {}).get("require_velocity_confirm", True)
        self._last_kf_output = KalmanOutput()
        self._last_dt = 0.0
        self._last_ts = 0

    def reset_window(self) -> None:
        self.kf.reset()
        self._last_kf_output = KalmanOutput()
        self._last_ts = 0

    def process_tick(
        self, ts: int, binance_price: float, oracle_price: float,
        coinbase_price: float, candles: list[Candle],
    ) -> None:
        """Update Kalman filter with new price data."""
        dt = (ts - self._last_ts) if self._last_ts > 0 else 1.0
        dt = max(0.1, min(60.0, dt))
        self._last_ts = ts

        self._last_kf_output = self.kf.update(
            binance_price=binance_price,
            oracle_price=oracle_price,
            coinbase_price=coinbase_price if coinbase_price > 0 else 0.0,
            coinbase_connected=coinbase_price > 0,
            timestamp=float(ts),
        )

    def evaluate_snapshot(
        self, snap: dict, btc_start: float, candles: list[Candle],
        oracle_price: float, winner: str, start_time: str, market_id: str,
        coinalyze: CoinalyzeLookup | None = None, is_weekend: bool = False,
    ) -> TradeResult | None:
        secs_remaining = snap.get("secs_remaining", 0)
        if secs_remaining > EARLIEST_ENTRY or secs_remaining < LATEST_ENTRY:
            return None

        kf_signal = self._last_kf_output.signal
        momentum_sig = self.momentum.compute(candles) if candles and len(candles) >= 2 else 0.0
        ob_sig = self.ob_signal.compute(snap)

        liq_sig = sent_sig = 0.0
        snapshot_time = snap.get("snapshot_time", "")
        if coinalyze and coinalyze.is_available and snapshot_time:
            ca = coinalyze.get_snapshot(snapshot_time)
            if ca:
                liq_sig = coinalyze.compute_liquidation_signal(ca, momentum_sig)
                sent_sig = coinalyze.compute_sentiment_signal(ca, momentum_sig)

        flip = self.weekend_flip and is_weekend
        _, est_prob_up = self.combiner.combine(
            oracle_lag_signal=kf_signal,
            momentum_signal=momentum_sig,
            liquidation_signal=liq_sig,
            seconds_remaining=secs_remaining,
            orderbook_signal=ob_sig,
            sentiment_signal=sent_sig,
            flip_signal=flip,
        )

        # Try entry
        result = self._try_entry(
            snap, est_prob_up, winner, start_time, market_id,
            extra_signals={"kf": kf_signal, "velocity": self._last_kf_output.velocity,
                           "momentum": momentum_sig},
        )

        # Velocity gate
        if result and self.require_velocity:
            velocity = self._last_kf_output.velocity
            if (result.side == "YES" and velocity <= 0) or \
               (result.side == "NO" and velocity >= 0):
                self.stats.trades.remove(result)
                self.account.bankroll -= result.pnl  # Undo
                self.account.daily_pnl -= result.pnl
                # Undo streak
                if result.won:
                    self.account.consecutive_wins = max(0, self.account.consecutive_wins - 1)
                else:
                    self.account.consecutive_losses = max(0, self.account.consecutive_losses - 1)
                return None

        return result


# -- Bot C: HMM Regime EV ---------------------------------------------

class StrategyC(BaseStrategy):
    """HMM regime-filtered strategy — only trades during trends."""

    def __init__(self, cfg: dict):
        super().__init__("C: HMM Regime EV", cfg)
        sig = cfg.get("signals", {})
        hmm_cfg = sig.get("hmm_regime", {})
        self.hmm = HmmRegimeDetector(
            transition_stickiness=hmm_cfg.get("transition_stickiness", 0.90),
            adaptation_rate=hmm_cfg.get("adaptation_rate", 0.02),
            min_candles=hmm_cfg.get("min_candles", 16),
            feature_lookback=hmm_cfg.get("feature_lookback", 5),
        )
        ol_cfg = sig.get("oracle_lag", {})
        self.oracle_lag = OracleLagSignal(
            max_expected_lag=ol_cfg.get("max_expected_lag", 0.001)
        )
        self.momentum = MomentumSignal()
        self.combiner = SignalCombiner(
            max_adjustment=sig.get("max_adjustment", 0.20),
        )
        self.ob_signal = OrderbookSignalBacktest()
        entry_cfg = cfg.get("entry", {})
        self.only_trending = entry_cfg.get("only_trending", True)
        self.min_regime_confidence = entry_cfg.get("min_regime_confidence", 0.40)

    def reset_window(self) -> None:
        self.hmm.reset()

    def evaluate_snapshot(
        self, snap: dict, btc_start: float, candles: list[Candle],
        oracle_price: float, winner: str, start_time: str, market_id: str,
        coinalyze: CoinalyzeLookup | None = None, is_weekend: bool = False,
    ) -> TradeResult | None:
        secs_remaining = snap.get("secs_remaining", 0)
        if secs_remaining > EARLIEST_ENTRY or secs_remaining < LATEST_ENTRY:
            return None

        btc_price = snap.get("btc_price") or btc_start

        # HMM regime detection
        regime_state = self.hmm.detect(candles) if candles and len(candles) >= 16 else None

        # Regime gate
        if self.only_trending and regime_state:
            from strategy.regime_detector import Regime
            regime = regime_state.regime
            conf = regime_state.confidence
            is_trending = regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN)
            if not is_trending or conf < self.min_regime_confidence:
                return None

        oracle_sig = self.oracle_lag.compute(btc_price, oracle_price, btc_start)
        momentum_sig = self.momentum.compute(candles) if candles and len(candles) >= 2 else 0.0
        ob_sig = self.ob_signal.compute(snap)

        liq_sig = sent_sig = 0.0
        snapshot_time = snap.get("snapshot_time", "")
        if coinalyze and coinalyze.is_available and snapshot_time:
            ca = coinalyze.get_snapshot(snapshot_time)
            if ca:
                liq_sig = coinalyze.compute_liquidation_signal(ca, momentum_sig)
                sent_sig = coinalyze.compute_sentiment_signal(ca, momentum_sig)

        flip = self.weekend_flip and is_weekend
        _, est_prob_up = self.combiner.combine(
            oracle_lag_signal=oracle_sig,
            momentum_signal=momentum_sig,
            liquidation_signal=liq_sig,
            seconds_remaining=secs_remaining,
            orderbook_signal=ob_sig,
            sentiment_signal=sent_sig,
            flip_signal=flip,
        )

        regime_name = regime_state.regime.value if regime_state else "unknown"
        return self._try_entry(
            snap, est_prob_up, winner, start_time, market_id,
            regime_name=regime_name,
            extra_signals={"oracle": oracle_sig, "momentum": momentum_sig,
                           "regime": regime_name},
        )


# -- Bot D: Enhanced EV -----------------------------------------------

class StrategyD(BaseStrategy):
    """Enhanced EV with multi-timeframe momentum + cross-exchange + funding."""

    def __init__(self, cfg: dict):
        super().__init__("D: Enhanced EV", cfg)
        sig = cfg.get("signals", {})
        ol_cfg = sig.get("oracle_lag", {})
        self.oracle_lag = OracleLagSignal(
            max_expected_lag=ol_cfg.get("max_expected_lag", 0.001)
        )
        ms_cfg = sig.get("momentum_slope", {})
        self.momentum_slope = MomentumSlopeSignal(
            weight_30s=ms_cfg.get("weight_30s", 0.35),
            weight_60s=ms_cfg.get("weight_60s", 0.30),
            weight_120s=ms_cfg.get("weight_120s", 0.20),
            weight_240s=ms_cfg.get("weight_240s", 0.15),
            slope_normaliser=ms_cfg.get("slope_normaliser", 10.0),
        )
        cx_cfg = sig.get("cross_exchange", {})
        self.cross_exchange = CrossExchangeSignal(
            rolling_window=cx_cfg.get("rolling_window", 120),
            z_threshold=cx_cfg.get("z_threshold", 1.5),
            max_spread_usd=cx_cfg.get("max_spread_usd", 200.0),
        )
        ff_cfg = sig.get("funding_filter", {})
        self.funding_filter = FundingRegimeFilter(
            mild_threshold=ff_cfg.get("mild_threshold", 0.01),
            high_threshold=ff_cfg.get("high_threshold", 0.05),
            stress_threshold=ff_cfg.get("stress_threshold", 0.10),
        )
        self.combiner = SignalCombiner(
            max_adjustment=sig.get("max_adjustment", 0.20),
        )
        self.ob_signal = OrderbookSignalBacktest()

    def process_tick(
        self, ts: int, binance_price: float, oracle_price: float,
        coinbase_price: float, candles: list[Candle],
    ) -> None:
        """Update cross-exchange signal with tick data."""
        if coinbase_price > 0 and binance_price > 0:
            self.cross_exchange.compute(binance_price, coinbase_price, True)

    def evaluate_snapshot(
        self, snap: dict, btc_start: float, candles: list[Candle],
        oracle_price: float, winner: str, start_time: str, market_id: str,
        coinalyze: CoinalyzeLookup | None = None, is_weekend: bool = False,
    ) -> TradeResult | None:
        secs_remaining = snap.get("secs_remaining", 0)
        if secs_remaining > EARLIEST_ENTRY or secs_remaining < LATEST_ENTRY:
            return None

        btc_price = snap.get("btc_price") or btc_start

        oracle_sig = self.oracle_lag.compute(btc_price, oracle_price, btc_start)

        # MomentumSlope needs candle deque
        candle_deque = deque(candles, maxlen=30)
        slope_sig = self.momentum_slope.compute(candle_deque) if len(candles) >= 2 else 0.0

        # Cross-exchange: use last computed value from process_tick
        cx_sig = self.cross_exchange._last_signal if hasattr(self.cross_exchange, '_last_signal') else 0.0

        ob_sig = self.ob_signal.compute(snap)

        # Funding filter
        regime_edge_mult = 1.0
        size_factor = 1.0
        snapshot_time = snap.get("snapshot_time", "")
        if coinalyze and coinalyze.is_available and snapshot_time:
            ca = coinalyze.get_snapshot(snapshot_time)
            if ca:
                funding_rate = ca.get("funding_rate", 0.0)
                oi_change = ca.get("oi_change_pct", 0.0)
                funding_result = self.funding_filter.evaluate(
                    funding_rate=funding_rate,
                    oi_change_pct=oi_change,
                )
                regime_edge_mult = funding_result.get("edge_multiplier", 1.0)
                size_factor = funding_result.get("size_factor", 1.0)

        liq_sig = sent_sig = 0.0
        if coinalyze and coinalyze.is_available and snapshot_time:
            ca = coinalyze.get_snapshot(snapshot_time)
            if ca:
                liq_sig = coinalyze.compute_liquidation_signal(ca, slope_sig)
                sent_sig = coinalyze.compute_sentiment_signal(ca, slope_sig)

        flip = self.weekend_flip and is_weekend
        _, est_prob_up = self.combiner.combine(
            oracle_lag_signal=oracle_sig,
            momentum_signal=slope_sig,
            liquidation_signal=liq_sig,
            seconds_remaining=secs_remaining,
            coinbase_direction=cx_sig,
            orderbook_signal=ob_sig,
            sentiment_signal=sent_sig,
            flip_signal=flip,
        )

        return self._try_entry(
            snap, est_prob_up, winner, start_time, market_id,
            regime_edge_mult=regime_edge_mult,
            extra_signals={"oracle": oracle_sig, "slope": slope_sig,
                           "cx": cx_sig, "funding_mult": regime_edge_mult},
        )


# -- Bot E: OU Reversion EV -------------------------------------------

class StrategyE(BaseStrategy):
    """OU mean reversion on Binance-Oracle spread."""

    def __init__(self, cfg: dict):
        super().__init__("E: OU Reversion EV", cfg)
        sig = cfg.get("signals", {})
        ou_cfg = sig.get("ou_spread", {})
        self.ou_spread = OUSpreadSignal(
            calibration_window=ou_cfg.get("calibration_window", 300),
            min_observations=ou_cfg.get("min_observations", 60),
            entry_z_threshold=ou_cfg.get("entry_z_threshold", 1.5),
            dt=ou_cfg.get("dt", 1.0),
        )
        ms_cfg = sig.get("momentum_slope", {})
        self.momentum_slope = MomentumSlopeSignal(
            weight_30s=ms_cfg.get("weight_30s", 0.35),
            weight_60s=ms_cfg.get("weight_60s", 0.30),
            weight_120s=ms_cfg.get("weight_120s", 0.20),
            weight_240s=ms_cfg.get("weight_240s", 0.15),
            slope_normaliser=ms_cfg.get("slope_normaliser", 10.0),
        )
        self.momentum = MomentumSignal()
        self.combiner = SignalCombiner(
            max_adjustment=sig.get("max_adjustment", 0.20),
        )
        self.ob_signal = OrderbookSignalBacktest()
        self.min_ou_z = cfg.get("entry", {}).get("min_ou_z_score", 1.5)

    def reset_window(self) -> None:
        # Don't reset OU — it accumulates cross-window spread history
        pass

    def process_tick(
        self, ts: int, binance_price: float, oracle_price: float,
        coinbase_price: float, candles: list[Candle],
    ) -> None:
        """Feed the OU process with spread observations."""
        if binance_price > 0 and oracle_price > 0:
            self.ou_spread.compute(binance_price, oracle_price)

    def evaluate_snapshot(
        self, snap: dict, btc_start: float, candles: list[Candle],
        oracle_price: float, winner: str, start_time: str, market_id: str,
        coinalyze: CoinalyzeLookup | None = None, is_weekend: bool = False,
    ) -> TradeResult | None:
        secs_remaining = snap.get("secs_remaining", 0)
        if secs_remaining > EARLIEST_ENTRY or secs_remaining < LATEST_ENTRY:
            return None

        # OU z-score gate
        if abs(self.ou_spread.z_score) < self.min_ou_z:
            return None

        ou_signal = self.ou_spread._last_signal

        candle_deque = deque(candles, maxlen=30)
        slope_sig = self.momentum_slope.compute(candle_deque) if len(candles) >= 2 else 0.0
        momentum_sig = self.momentum.compute(candles) if candles and len(candles) >= 2 else 0.0
        ob_sig = self.ob_signal.compute(snap)

        liq_sig = sent_sig = 0.0
        snapshot_time = snap.get("snapshot_time", "")
        if coinalyze and coinalyze.is_available and snapshot_time:
            ca = coinalyze.get_snapshot(snapshot_time)
            if ca:
                liq_sig = coinalyze.compute_liquidation_signal(ca, momentum_sig)
                sent_sig = coinalyze.compute_sentiment_signal(ca, momentum_sig)

        flip = self.weekend_flip and is_weekend
        # OU signal goes into L1 slot (replaces oracle lag)
        _, est_prob_up = self.combiner.combine(
            oracle_lag_signal=ou_signal,
            momentum_signal=slope_sig,
            liquidation_signal=liq_sig,
            seconds_remaining=secs_remaining,
            orderbook_signal=ob_sig,
            sentiment_signal=sent_sig,
            flip_signal=flip,
        )

        return self._try_entry(
            snap, est_prob_up, winner, start_time, market_id,
            extra_signals={"ou": ou_signal, "z_score": self.ou_spread.z_score,
                           "theta": self.ou_spread.theta, "slope": slope_sig},
        )


# --- Strategy Factory -----------------------------------------------

STRATEGY_MAP = {
    "a": StrategyA,
    "b": StrategyB,
    "c": StrategyC,
    "d": StrategyD,
    "e": StrategyE,
}

# Default configs (matching config_multi.yaml)
DEFAULT_CONFIGS = {
    "a": {
        "weekend_flip": True,
        "signals": {"oracle_lag": {"max_expected_lag": 0.001}, "max_adjustment": 0.20},
        "entry": {"min_edge": 0.01, "max_edge": 0.06, "fee_adjustment": 0.02, "min_confidence": 0.02},
        "sizing": {"kelly_multiplier": 0.25, "min_bet_usdc": 1.0, "max_bet_usdc": 5.0, "initial_bankroll": 100.0},
        "risk": {"daily_loss_cap_pct": 0.20, "daily_loss_warn_pct": 0.15, "min_bankroll": 10.0,
                 "streak_reduce_at": 3, "streak_reduce_factor": 0.75,
                 "streak_heavy_reduce_at": 5, "streak_heavy_reduce_factor": 0.5,
                 "streak_pause_at": 7, "streak_pause_minutes": 30, "streak_reset_wins": 3},
    },
    "b": {
        "weekend_flip": True,
        "signals": {
            "kalman_filter": {"process_noise_scale": 1e-5, "measurement_noise_binance": 1e-4,
                              "measurement_noise_coinbase": 2e-4, "measurement_noise_oracle": 5e-4,
                              "adaptive_window": 30, "breakout_velocity_threshold": 0.5, "max_expected_lag": 0.001},
            "weight_schedule": "kf_dominant", "max_adjustment": 0.20,
        },
        "entry": {"min_edge": 0.02, "max_edge": 0.08, "fee_adjustment": 0.02, "min_confidence": 0.03,
                   "require_velocity_confirm": True},
        "sizing": {"kelly_multiplier": 0.25, "min_bet_usdc": 1.0, "max_bet_usdc": 5.0, "initial_bankroll": 100.0},
        "risk": {"daily_loss_cap_pct": 0.20, "daily_loss_warn_pct": 0.15, "min_bankroll": 10.0,
                 "streak_reduce_at": 3, "streak_reduce_factor": 0.75,
                 "streak_heavy_reduce_at": 5, "streak_heavy_reduce_factor": 0.5,
                 "streak_pause_at": 7, "streak_pause_minutes": 30, "streak_reset_wins": 3},
    },
    "c": {
        "weekend_flip": True,
        "signals": {"oracle_lag": {"max_expected_lag": 0.001},
                    "hmm_regime": {"transition_stickiness": 0.90, "adaptation_rate": 0.02,
                                   "min_candles": 16, "feature_lookback": 5},
                    "max_adjustment": 0.20},
        "entry": {"min_edge": 0.01, "max_edge": 0.06, "fee_adjustment": 0.02, "min_confidence": 0.02,
                   "only_trending": True, "min_regime_confidence": 0.40},
        "sizing": {"kelly_multiplier": 0.25, "min_bet_usdc": 1.0, "max_bet_usdc": 5.0, "initial_bankroll": 100.0},
        "risk": {"daily_loss_cap_pct": 0.20, "daily_loss_warn_pct": 0.15, "min_bankroll": 10.0,
                 "streak_reduce_at": 3, "streak_reduce_factor": 0.75,
                 "streak_heavy_reduce_at": 5, "streak_heavy_reduce_factor": 0.5,
                 "streak_pause_at": 7, "streak_pause_minutes": 30, "streak_reset_wins": 3},
    },
    "d": {
        "weekend_flip": True,
        "signals": {"oracle_lag": {"max_expected_lag": 0.001},
                    "momentum_slope": {"weight_30s": 0.35, "weight_60s": 0.30, "weight_120s": 0.20,
                                       "weight_240s": 0.15, "slope_normaliser": 10.0},
                    "cross_exchange": {"rolling_window": 120, "z_threshold": 1.5, "max_spread_usd": 200.0},
                    "funding_filter": {"mild_threshold": 0.01, "high_threshold": 0.05, "stress_threshold": 0.10},
                    "max_adjustment": 0.20},
        "entry": {"min_edge": 0.01, "max_edge": 0.06, "fee_adjustment": 0.02, "min_confidence": 0.02},
        "sizing": {"kelly_multiplier": 0.25, "min_bet_usdc": 1.0, "max_bet_usdc": 5.0, "initial_bankroll": 100.0},
        "risk": {"daily_loss_cap_pct": 0.20, "daily_loss_warn_pct": 0.15, "min_bankroll": 10.0,
                 "streak_reduce_at": 3, "streak_reduce_factor": 0.75,
                 "streak_heavy_reduce_at": 5, "streak_heavy_reduce_factor": 0.5,
                 "streak_pause_at": 7, "streak_pause_minutes": 30, "streak_reset_wins": 3},
    },
    "e": {
        "weekend_flip": True,
        "signals": {"ou_spread": {"calibration_window": 300, "min_observations": 60,
                                   "entry_z_threshold": 1.5, "dt": 1.0},
                    "momentum_slope": {"weight_30s": 0.35, "weight_60s": 0.30, "weight_120s": 0.20,
                                       "weight_240s": 0.15, "slope_normaliser": 10.0},
                    "max_adjustment": 0.20},
        "entry": {"min_edge": 0.01, "max_edge": 0.06, "fee_adjustment": 0.02, "min_confidence": 0.02,
                   "min_ou_z_score": 1.5},
        "sizing": {"kelly_multiplier": 0.25, "min_bet_usdc": 1.0, "max_bet_usdc": 5.0, "initial_bankroll": 100.0},
        "risk": {"daily_loss_cap_pct": 0.20, "daily_loss_warn_pct": 0.15, "min_bankroll": 10.0,
                 "streak_reduce_at": 3, "streak_reduce_factor": 0.75,
                 "streak_heavy_reduce_at": 5, "streak_heavy_reduce_factor": 0.5,
                 "streak_pause_at": 7, "streak_pause_minutes": 30, "streak_reset_wins": 3},
    },
}


# --- Data Loading ----------------------------------------------------

def load_markets(path: str) -> list[dict]:
    """Load market metadata, filtered and sorted by start_time."""
    markets = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["btc_price_start"] = float(row["btc_price_start"]) if row["btc_price_start"] else 0
            row["btc_price_end"] = float(row["btc_price_end"]) if row["btc_price_end"] else 0
            markets.append(row)
    # Sort chronologically
    markets.sort(key=lambda m: m.get("start_time", ""))
    return markets


def load_snapshots(path: str) -> dict[str, dict[int, dict]]:
    """Load snapshots grouped by market_id -> offset -> data."""
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
            # Add computed seconds_remaining
            row["secs_remaining"] = 300 - offset
            grouped[mid][offset] = row
    return grouped


# --- Reporting -------------------------------------------------------

def print_report(stats: StrategyStats, total_markets: int) -> None:
    """Print detailed strategy report."""
    print(f"\n{'='*65}")
    print(f"  {stats.name}")
    print(f"{'='*65}")
    if stats.count == 0:
        print("  No trades taken.")
        return

    final_bank = stats.trades[-1].bankroll_after if stats.trades else 100.0
    avg_size = sum(t.size_usdc for t in stats.trades) / stats.count
    avg_win = sum(t.pnl for t in stats.trades if t.won) / max(1, stats.wins)
    losses = [t for t in stats.trades if not t.won]
    avg_loss = sum(t.pnl for t in losses) / max(1, len(losses))

    print(f"  Trades:        {stats.count:>7,}   ({stats.count/total_markets*100:.1f}% participation)")
    print(f"  Win Rate:      {stats.win_rate*100:>7.1f}%")
    print(f"  Total PnL:     ${stats.total_pnl:>+9,.2f}")
    print(f"  Final Bank:    ${final_bank:>9,.2f}  (ROI: {(final_bank-100):>+.0f}%)")
    print(f"  Avg Bet:       ${avg_size:>9.2f}")
    print(f"  Avg Win:       ${avg_win:>+9.4f}")
    print(f"  Avg Loss:      ${avg_loss:>+9.4f}")
    print(f"  Profit Factor: {stats.profit_factor:>9.3f}")
    print(f"  Sharpe (ann):  {stats.sharpe_ratio:>+9.2f}")
    print(f"  Max Drawdown:  ${stats.max_drawdown:>9.2f}")

    days = stats.day_of_week_breakdown()
    if days:
        print(f"\n  Day of Week:")
        for day, d in days.items():
            print(f"    {day}: {d['count']:>5} trades, {d['wr']*100:>5.1f}% WR, ${d['pnl']:>+9,.2f}")


# --- Main Backtest ---------------------------------------------------

def run_backtest(
    strategy_keys: list[str],
    configs: dict[str, dict] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    warmup_days: int = 14,
) -> dict[str, StrategyStats]:
    """Run multi-strategy backtest.

    Args:
        strategy_keys: List of strategy letters to run (e.g. ["a", "b", "e"]).
        configs: Optional per-strategy config overrides.
        start_date: Optional ISO date filter (e.g. "2026-03-01").
        end_date: Optional ISO date filter.
        warmup_days: Days of data before start_date to feed signals for
            calibration (ticks processed but trades not counted). Default 2.

    Returns:
        Dict mapping strategy key to StrategyStats.
    """
    configs = configs or DEFAULT_CONFIGS

    # Compute warmup date (signals need calibration data before scoring starts)
    # OU/KF strategies need extensive warmup (14+ days) to calibrate properly.
    # Use all available data before start_date as warmup by default.
    warmup_date: str | None = None
    if start_date and warmup_days > 0:
        try:
            sd = datetime.strptime(start_date, "%Y-%m-%d")
            warmup_dt = sd - timedelta(days=warmup_days)
            warmup_date = warmup_dt.strftime("%Y-%m-%d")
        except ValueError:
            warmup_date = None

    # Load all data
    markets_path = os.path.join(DATA_DIR, "polybacktest_markets.csv")
    snapshots_path = os.path.join(DATA_DIR, "polybacktest_snapshots.csv")
    klines_path = os.path.join(DATA_DIR, "binance_klines_1m.csv")
    oracle_path = os.path.join(DATA_DIR, "chainlink_btc_history.csv")
    coinbase_path = os.path.join(DATA_DIR, "coinbase_btc_1m.csv")

    for path, name in [(markets_path, "Markets"), (snapshots_path, "Snapshots")]:
        if not os.path.exists(path):
            logger.error(f"{name} not found: {path}")
            return {}

    logger.info("Loading data...")
    markets = load_markets(markets_path)
    snapshots = load_snapshots(snapshots_path)
    candle_lookup = CandleLookup(klines_path) if os.path.exists(klines_path) else None
    oracle_lookup = OracleLookup(oracle_path)
    coinbase_lookup = CoinbaseLookup(coinbase_path)
    coinalyze = CoinalyzeLookup(DATA_DIR)

    logger.info(
        f"Data: {len(markets)} markets, "
        f"oracle={'OK' if oracle_lookup.is_available else 'MISSING'}, "
        f"coinbase={'OK' if coinbase_lookup.is_available else 'MISSING'}, "
        f"coinalyze={'OK' if coinalyze.is_available else 'MISSING'}"
    )

    # Date filter — include warmup period for signal calibration
    effective_start = warmup_date or start_date
    if effective_start or end_date:
        filtered = []
        for m in markets:
            st = m.get("start_time", "")
            if effective_start and st < effective_start:
                continue
            if end_date and st > end_date:
                continue
            filtered.append(m)
        logger.info(f"Date filter: {len(markets)} -> {len(filtered)} markets")
        markets = filtered

    # Instantiate strategies
    runners: dict[str, BaseStrategy] = {}
    for key in strategy_keys:
        key = key.lower()
        if key not in STRATEGY_MAP:
            logger.warning(f"Unknown strategy '{key}', skipping")
            continue
        cfg = configs.get(key, DEFAULT_CONFIGS.get(key, {}))
        runners[key] = STRATEGY_MAP[key](cfg)
        logger.info(f"Initialized strategy {runners[key].name}")

    # Main loop
    skipped = 0
    total_markets = len(markets)

    for i, market in enumerate(markets):
        if i % 1000 == 0 and i > 0:
            logger.info(f"Processing market {i:,}/{total_markets:,}...")

        mid = market["market_id"]
        winner = market.get("winner")
        btc_start = market.get("btc_price_start", 0)

        if not winner or not btc_start:
            continue

        snaps = snapshots.get(mid, {})
        if not snaps:
            skipped += 1
            continue

        start_time = market.get("start_time", "")

        # Is it a weekend?
        is_weekend = False
        if start_time:
            try:
                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                is_weekend = dt.weekday() >= 5
            except ValueError:
                pass

        # Parse window timestamps
        window_start_ts = 0
        window_end_ts = 0
        if start_time:
            try:
                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                window_start_ts = int(dt.timestamp())
                window_end_ts = window_start_ts + 300
            except ValueError:
                pass

        # Get oracle price at window open
        oracle_open_price = btc_start  # Fallback
        if oracle_lookup.is_available and window_start_ts > 0:
            op = oracle_lookup.get_price(window_start_ts)
            if op > 0:
                oracle_open_price = op

        # Reset per-window state
        for runner in runners.values():
            runner.reset_window()

        # -- Process tick-level data for continuous strategies --
        if window_start_ts > 0:
            # Build timeline of price events within the window
            events: list[tuple[int, float, float, float]] = []  # (ts, binance, oracle, coinbase)

            # Binance candle closes within window (every 60s)
            if candle_lookup:
                for sec_offset in range(60, 301, 60):
                    tick_ts = window_start_ts + sec_offset
                    tick_ms = tick_ts * 1000
                    bp = candle_lookup.get_price_at(tick_ms)
                    if bp > 0:
                        op = oracle_lookup.get_price(tick_ts) if oracle_lookup.is_available else oracle_open_price
                        cp = coinbase_lookup.get_price(tick_ts) if coinbase_lookup.is_available else 0.0
                        events.append((tick_ts, bp, op, cp))

            # Oracle updates within window
            if oracle_lookup.is_available:
                oracle_updates = oracle_lookup.get_updates_in_range(
                    window_start_ts, window_end_ts
                )
                for ou_ts, ou_price in oracle_updates:
                    bp = candle_lookup.get_price_at(ou_ts * 1000) if candle_lookup else btc_start
                    cp = coinbase_lookup.get_price(ou_ts) if coinbase_lookup.is_available else 0.0
                    events.append((ou_ts, bp, ou_price, cp))

            # Deduplicate and sort
            events.sort(key=lambda e: e[0])
            seen_ts: set[int] = set()
            unique_events = []
            for ev in events:
                if ev[0] not in seen_ts:
                    seen_ts.add(ev[0])
                    unique_events.append(ev)

            # Feed to continuous strategies
            for ts, bp, op, cp in unique_events:
                ts_ms = ts * 1000
                candles = candle_lookup.get_candles(ts_ms, 30) if candle_lookup else []
                for runner in runners.values():
                    runner.process_tick(ts, bp, op, cp, candles)

        # -- Skip trade evaluation during warmup period --
        if start_date and start_time < start_date:
            continue

        # -- Evaluate entry at snapshot timestamps --
        sorted_offsets = sorted(snaps.keys())
        for runner in runners.values():
            entered = False
            for offset_sec in sorted_offsets:
                if entered:
                    break

                snap = snaps[offset_sec]
                snapshot_time = snap.get("snapshot_time", "")

                # Get oracle price at snapshot time
                oracle_at_snap = oracle_open_price
                if oracle_lookup.is_available and snapshot_time:
                    try:
                        snap_dt = datetime.fromisoformat(
                            snapshot_time.replace("Z", "+00:00")
                        )
                        oracle_at_snap = oracle_lookup.get_price(int(snap_dt.timestamp()))
                    except ValueError:
                        pass

                # Get candles at snapshot time
                candles = []
                if candle_lookup and snapshot_time:
                    candles = candle_lookup.get_candles_iso(snapshot_time, 30)

                result = runner.evaluate_snapshot(
                    snap=snap,
                    btc_start=oracle_open_price,
                    candles=candles,
                    oracle_price=oracle_at_snap,
                    winner=winner,
                    start_time=start_time,
                    market_id=mid,
                    coinalyze=coinalyze,
                    is_weekend=is_weekend,
                )
                if result:
                    entered = True

    processed = total_markets - skipped
    logger.info(f"Backtested {processed:,} markets ({skipped} skipped)")

    return {key: runner.stats for key, runner in runners.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-strategy backtest")
    parser.add_argument(
        "--strategies", "-s", default="a,b,c,d,e",
        help="Comma-separated strategy letters (default: a,b,c,d,e)"
    )
    parser.add_argument("--start", default=None, help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date filter (YYYY-MM-DD)")
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML config (default: use built-in defaults)"
    )
    args = parser.parse_args()

    strategy_keys = [s.strip() for s in args.strategies.split(",")]

    # Load config from YAML if provided
    configs = None
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            raw = yaml.safe_load(f)
        # Map bot names to strategy letters
        bots = raw.get("bots", {})
        letter_map = {
            "bot_a_contrarian": "a", "bot_b_kalman": "b",
            "bot_c_hmm_regime": "c", "bot_d_enhanced": "d",
            "bot_e_ou_reversion": "e",
        }
        configs = {}
        for bot_name, bot_cfg in bots.items():
            letter = letter_map.get(bot_name)
            if letter:
                configs[letter] = bot_cfg

    results = run_backtest(
        strategy_keys=strategy_keys,
        configs=configs,
        start_date=args.start,
        end_date=args.end,
    )

    if not results:
        logger.error("No results")
        return

    # Find total markets from any strategy
    total_markets = 0
    for stats in results.values():
        if stats.count > 0:
            # Rough estimate from trade participation
            total_markets = max(total_markets, stats.count * 3)

    # Layers status
    oracle_path = os.path.join(DATA_DIR, "chainlink_btc_history.csv")
    coinbase_path = os.path.join(DATA_DIR, "coinbase_btc_1m.csv")
    layers = [
        f"L1:Oracle {'REAL' if os.path.exists(oracle_path) else 'approx'}",
        f"L2:Momentum OK",
        f"Coinbase {'OK' if os.path.exists(coinbase_path) else 'MISSING'}",
    ]

    print(f"\n{'#'*65}")
    print(f"  MULTI-STRATEGY BACKTEST RESULTS")
    print(f"  {' | '.join(layers)}")
    print(f"{'#'*65}")

    for key in sorted(results.keys()):
        print_report(results[key], max(1, total_markets))

    # Comparison table
    print(f"\n{'='*80}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*80}")
    header = f"  {'Strategy':<22} {'Trades':>7} {'WR':>7} {'PnL':>10} {'PF':>7} {'Sharpe':>7} {'Final$':>8} {'MaxDD':>7}"
    print(header)
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*10} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
    for key in sorted(results.keys()):
        s = results[key]
        final = s.trades[-1].bankroll_after if s.trades else 100.0
        print(
            f"  {s.name:<22} {s.count:>7,} {s.win_rate*100:>6.1f}% "
            f"${s.total_pnl:>+9,.2f} {s.profit_factor:>7.3f} "
            f"{s.sharpe_ratio:>+7.2f} ${final:>7,.0f} ${s.max_drawdown:>6.2f}"
        )

    # Statistical significance
    print(f"\n  Statistical Significance (vs 50% baseline):")
    for key in sorted(results.keys()):
        s = results[key]
        if s.count >= 30:
            z = (s.wins - s.count * 0.5) / math.sqrt(s.count * 0.25)
            sig = "significant" if abs(z) > 1.96 else "not significant"
            direction = "ABOVE" if z > 0 else "BELOW"
            print(f"    {s.name:<22} z={z:>+6.2f}  ({direction} 50%, {sig})")
        elif s.count > 0:
            print(f"    {s.name:<22} too few trades ({s.count}) for significance")

    # Save results to CSV
    output_path = os.path.join(DATA_DIR, "backtest_multi_results.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy", "market_id", "start_time", "side", "entry_price",
            "seconds_remaining", "est_prob_up", "prob_edge", "won", "pnl",
            "size_usdc", "bankroll_after", "regime", "signals",
        ])
        for key in sorted(results.keys()):
            for t in results[key].trades:
                writer.writerow([
                    t.strategy, t.market_id, t.start_time, t.side,
                    f"{t.entry_price:.4f}", f"{t.seconds_remaining:.0f}",
                    f"{t.est_prob_up:.4f}", f"{t.prob_edge:.4f}",
                    int(t.won), f"{t.pnl:.4f}", f"{t.size_usdc:.2f}",
                    f"{t.bankroll_after:.2f}", t.regime,
                    json.dumps(t.signals),
                ])
    logger.info(f"Trade log saved to {output_path}")


if __name__ == "__main__":
    main()
