"""Signal accuracy backtest — tests prediction quality over 6+ months.

This is the foundational test: forget about Polymarket prices, fills,
and execution for now. The only question that matters is:

    "Can our signal engine predict which direction BTC moves
     within a 5-minute window better than a coin flip?"

If the answer is YES with >53% accuracy, the strategy is viable
because early Polymarket prices are near 50/50.

Tests every 5-minute window in the historical dataset and measures:
- Overall directional accuracy
- Accuracy by signal strength (confident vs weak calls)
- Accuracy by regime (trending, ranging, volatile)
- Accuracy by time-of-day
- Accuracy by entry timing within the window
- Accuracy by BTC volatility level
- Optimal signal threshold (what confidence level maximizes edge?)

Usage:
    python -m backtest.signal_accuracy_test --data backtest/data/BTCUSDT_1m_6mo.csv
"""

import argparse
import csv
import logging
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from data.binance_feed import Candle
from signals.oracle_lag import OracleLagSignal
from signals.momentum import MomentumSignal
from signals.combiner import SignalCombiner
from strategy.regime_detector import RegimeDetector, Regime


@dataclass
class WindowPrediction:
    """One 5-minute window's prediction vs reality."""
    window_start: int
    open_price: float
    close_price: float
    actual_direction: str           # "UP" or "DOWN"

    # Signal values at different points in the window
    signals_by_minute: dict = field(default_factory=dict)
    # Format: {minute_remaining: {signal_name: value, "prediction": "UP"/"DOWN", "confidence": float}}

    regime: str = ""
    hour_utc: int = 0
    volatility: float = 0.0        # ATR-based
    price_change_pct: float = 0.0  # How much BTC moved in this window


def load_candles(csv_path: str) -> list[Candle]:
    """Load candles from CSV."""
    candles = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(Candle(
                open_time=int(row["open_time"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                is_closed=True,
            ))
    return candles


def chunk_into_windows(candles: list[Candle]) -> list[list[Candle]]:
    """Split candles into 5-minute windows."""
    windows = []
    current = []
    for c in candles:
        wid = c.open_time // 300000
        if current:
            prev_wid = current[0].open_time // 300000
            if wid != prev_wid:
                if len(current) >= 2:
                    windows.append(current)
                current = []
        current.append(c)
    if len(current) >= 2:
        windows.append(current)
    return windows


def run_signal_accuracy_test(candles: list[Candle]) -> list[WindowPrediction]:
    """Run signal engine over every window and record predictions."""

    # Signal engine
    oracle_lag = OracleLagSignal(max_expected_lag=0.001)
    momentum = MomentumSignal(
        roc_weight=0.30, direction_weight=0.25, volume_weight=0.25,
        body_ratio_weight=0.10, rsi_weight=0.10, rsi_period=5, lookback_candles=10,
    )
    combiner = SignalCombiner(max_adjustment=0.15)
    regime_detector = RegimeDetector()

    windows = chunk_into_windows(candles)
    logger.info(f"Testing {len(windows)} windows")

    history: deque = deque(maxlen=30)
    predictions = []

    for idx, window in enumerate(windows):
        if len(window) < 3:
            for c in window:
                if c.is_closed:
                    history.append(c)
            continue

        open_price = window[0].open
        close_price = window[-1].close
        actual_dir = "UP" if close_price >= open_price else "DOWN"
        price_change = (close_price - open_price) / open_price if open_price > 0 else 0

        hour_utc = datetime.fromtimestamp(
            window[0].open_time / 1000, tz=timezone.utc
        ).hour

        # Detect regime
        regime_state = regime_detector.detect(history)

        # Compute volatility from recent candles
        vol = 0.0
        if len(history) >= 5:
            recent = list(history)[-15:]
            if recent:
                h = max(c.high for c in recent)
                lo = min(c.low for c in recent)
                mid = (h + lo) / 2
                vol = (h - lo) / mid if mid > 0 else 0

        wp = WindowPrediction(
            window_start=window[0].open_time,
            open_price=open_price,
            close_price=close_price,
            actual_direction=actual_dir,
            regime=regime_state.label,
            hour_utc=hour_utc,
            volatility=vol,
            price_change_pct=price_change,
        )

        # Simulate oracle: use a slightly lagged/noisy version of the open price
        # In reality, oracle updates sporadically. For accuracy testing,
        # use the actual window open as the oracle reference.
        oracle_open = open_price

        # Process each candle in the window
        for i, candle in enumerate(window):
            secs_remaining = (len(window) - i - 1) * 60
            minute_remaining = secs_remaining // 60

            # Only record at specific timepoints
            if minute_remaining not in [4, 3, 2, 1, 0]:
                if candle.is_closed:
                    history.append(candle)
                continue

            # Oracle lag signal: compare current price to window open
            # (simplified: real oracle would be lagged/noisy)
            oracle_val = oracle_lag.compute(
                exchange_price=candle.close,
                oracle_price=oracle_open,  # Oracle still showing open price
                oracle_open_price=oracle_open,
            )

            momentum_val = momentum.compute(
                candles=history,
                current_candle=candle,
            )

            # Get regime weight adjustments
            regime_params = regime_detector.get_params(regime_state.regime)

            raw_signal, est_prob_up = combiner.combine(
                oracle_lag_signal=oracle_val,
                momentum_signal=momentum_val,
                liquidation_signal=0.0,
                seconds_remaining=secs_remaining,
                coinbase_direction=0.0,
                orderbook_signal=0.0,
                sentiment_signal=0.0,
                regime_weight_adjustments=regime_params.get("weight_adjustments"),
            )

            prediction = "UP" if est_prob_up > 0.5 else "DOWN"
            confidence = abs(est_prob_up - 0.5)

            wp.signals_by_minute[minute_remaining] = {
                "oracle_lag": oracle_val,
                "momentum": momentum_val,
                "raw_signal": raw_signal,
                "est_prob_up": est_prob_up,
                "prediction": prediction,
                "confidence": confidence,
            }

            if candle.is_closed:
                history.append(candle)

        # Add remaining candles to history
        for c in window:
            if c.is_closed and c not in list(history):
                history.append(c)

        predictions.append(wp)

        if (idx + 1) % 5000 == 0:
            logger.info(f"  Processed {idx + 1}/{len(windows)} windows")

    return predictions


def print_results(predictions: list[WindowPrediction]):
    """Print comprehensive accuracy analysis."""
    print(f"\n{'='*70}")
    print(f"  SIGNAL ACCURACY TEST — {len(predictions)} windows")
    print(f"{'='*70}")

    up_count = sum(1 for p in predictions if p.actual_direction == "UP")
    down_count = len(predictions) - up_count
    print(f"\n  Direction split: UP {up_count} ({100*up_count/len(predictions):.1f}%) / DOWN {down_count}")

    # 1. Overall accuracy at each timepoint
    print(f"\n  {'='*65}")
    print(f"  ACCURACY BY ENTRY TIMING")
    print(f"  {'='*65}")
    print(f"  {'Min Left':>8s}  {'N':>6s}  {'Accuracy':>8s}  {'Confident':>10s}  {'Conf Acc':>8s}  {'Avg Signal':>10s}")

    for minute in [4, 3, 2, 1, 0]:
        entries = [(p, p.signals_by_minute.get(minute)) for p in predictions if minute in p.signals_by_minute]
        if not entries:
            continue

        correct = sum(1 for p, s in entries if s["prediction"] == p.actual_direction)
        total = len(entries)
        accuracy = correct / total

        # Confident predictions (confidence > 0.02 = est_prob > 0.52)
        confident = [(p, s) for p, s in entries if s["confidence"] > 0.02]
        conf_correct = sum(1 for p, s in confident if s["prediction"] == p.actual_direction) if confident else 0
        conf_acc = conf_correct / len(confident) if confident else 0

        avg_signal = sum(abs(s["raw_signal"]) for _, s in entries) / total

        print(f"  {minute:>8d}  {total:>6d}  {accuracy:>7.1%}  {len(confident):>10d}  {conf_acc:>7.1%}  {avg_signal:>10.4f}")

    # 2. Accuracy by signal strength buckets
    print(f"\n  {'='*65}")
    print(f"  ACCURACY BY SIGNAL STRENGTH (at 2-min remaining)")
    print(f"  {'='*65}")
    print(f"  {'Confidence':>12s}  {'N':>6s}  {'Accuracy':>8s}  {'Avg Move':>9s}")

    entries_2m = [(p, p.signals_by_minute.get(2)) for p in predictions if 2 in p.signals_by_minute]

    conf_buckets = [
        (0.00, 0.005, "< 0.5%"),
        (0.005, 0.01, "0.5-1%"),
        (0.01, 0.02, "1-2%"),
        (0.02, 0.03, "2-3%"),
        (0.03, 0.05, "3-5%"),
        (0.05, 0.10, "5-10%"),
        (0.10, 1.00, "> 10%"),
    ]

    for lo, hi, label in conf_buckets:
        bucket = [(p, s) for p, s in entries_2m if s and lo <= s["confidence"] < hi]
        if len(bucket) < 10:
            continue
        correct = sum(1 for p, s in bucket if s["prediction"] == p.actual_direction)
        accuracy = correct / len(bucket)
        avg_move = sum(abs(p.price_change_pct) for p, _ in bucket) / len(bucket) * 100
        print(f"  {label:>12s}  {len(bucket):>6d}  {accuracy:>7.1%}  {avg_move:>8.3f}%")

    # 3. Accuracy by regime
    print(f"\n  {'='*65}")
    print(f"  ACCURACY BY MARKET REGIME (at 2-min remaining)")
    print(f"  {'='*65}")
    print(f"  {'Regime':>15s}  {'N':>6s}  {'Accuracy':>8s}  {'Conf Acc':>8s}  {'Windows':>8s}")

    regime_groups = defaultdict(list)
    for p in predictions:
        if 2 in p.signals_by_minute:
            regime_groups[p.regime].append((p, p.signals_by_minute[2]))

    for regime in sorted(regime_groups.keys()):
        entries = regime_groups[regime]
        total = len(entries)
        correct = sum(1 for p, s in entries if s["prediction"] == p.actual_direction)
        accuracy = correct / total

        confident = [(p, s) for p, s in entries if s["confidence"] > 0.02]
        conf_correct = sum(1 for p, s in confident if s["prediction"] == p.actual_direction) if confident else 0
        conf_acc = conf_correct / len(confident) if confident else 0

        pct = 100 * total / len(entries_2m) if entries_2m else 0
        print(f"  {regime:>15s}  {total:>6d}  {accuracy:>7.1%}  {conf_acc:>7.1%}  {pct:>7.1f}%")

    # 4. Accuracy by hour of day
    print(f"\n  {'='*65}")
    print(f"  ACCURACY BY HOUR (UTC) — at 2-min remaining")
    print(f"  {'='*65}")

    hour_groups = defaultdict(list)
    for p in predictions:
        if 2 in p.signals_by_minute:
            hour_groups[p.hour_utc].append((p, p.signals_by_minute[2]))

    for hour in sorted(hour_groups.keys()):
        entries = hour_groups[hour]
        total = len(entries)
        correct = sum(1 for p, s in entries if s["prediction"] == p.actual_direction)
        accuracy = correct / total
        bar = "#" * int(accuracy * 40) + "." * (40 - int(accuracy * 40))
        marker = " <-- edge" if accuracy > 0.53 else ""
        print(f"  {hour:02d}:00  N={total:>5d}  {accuracy:>6.1%}  [{bar}]{marker}")

    # 5. Accuracy by volatility bucket
    print(f"\n  {'='*65}")
    print(f"  ACCURACY BY VOLATILITY LEVEL")
    print(f"  {'='*65}")

    vol_buckets = [
        (0, 0.001, "Very Low"),
        (0.001, 0.003, "Low"),
        (0.003, 0.006, "Medium"),
        (0.006, 0.01, "High"),
        (0.01, 0.02, "Very High"),
        (0.02, 1.0, "Extreme"),
    ]

    for lo, hi, label in vol_buckets:
        entries = [(p, p.signals_by_minute.get(2)) for p in predictions
                   if 2 in p.signals_by_minute and lo <= p.volatility < hi]
        if len(entries) < 20:
            continue
        correct = sum(1 for p, s in entries if s and s["prediction"] == p.actual_direction)
        accuracy = correct / len(entries)
        conf = [(p, s) for p, s in entries if s and s["confidence"] > 0.02]
        conf_acc = sum(1 for p, s in conf if s["prediction"] == p.actual_direction) / len(conf) if conf else 0
        print(f"  {label:>12s}  N={len(entries):>5d}  Acc={accuracy:>6.1%}  ConfAcc={conf_acc:>6.1%}  (conf N={len(conf)})")

    # 6. Optimal threshold analysis
    print(f"\n  {'='*65}")
    print(f"  OPTIMAL THRESHOLD ANALYSIS")
    print(f"  What minimum confidence maximizes profit?")
    print(f"  {'='*65}")
    print(f"  {'Min Conf':>10s}  {'N Trades':>8s}  {'Accuracy':>8s}  {'Edge':>6s}  {'EV/trade':>9s}  {'Total EV':>9s}")

    # Simulate profit at different thresholds
    # Assume: entry at ~0.50, payout 1.00 on win, lose stake on loss
    for threshold in [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.10]:
        entries = [(p, p.signals_by_minute.get(2)) for p in predictions
                   if 2 in p.signals_by_minute and p.signals_by_minute[2]["confidence"] >= threshold]
        if len(entries) < 20:
            continue
        correct = sum(1 for p, s in entries if s["prediction"] == p.actual_direction)
        accuracy = correct / len(entries)
        edge = accuracy - 0.5
        # EV per $1 bet at 50% odds: win = $1 profit, lose = -$1
        ev_per_trade = (accuracy * 1.0) - ((1 - accuracy) * 1.0)
        total_ev = ev_per_trade * len(entries)

        marker = " <-- BEST" if ev_per_trade > 0 and total_ev > 0 else ""
        print(f"  {threshold:>10.3f}  {len(entries):>8d}  {accuracy:>7.1%}  {edge:>+5.1%}  ${ev_per_trade:>+7.3f}  ${total_ev:>+8.1f}{marker}")

    # 7. Summary
    print(f"\n  {'='*65}")
    print(f"  SUMMARY")
    print(f"  {'='*65}")

    # Best accuracy found
    best_acc = 0
    best_config = ""
    for minute in [4, 3, 2, 1]:
        for threshold in [0.02, 0.03, 0.05]:
            entries = [(p, p.signals_by_minute.get(minute)) for p in predictions
                       if minute in p.signals_by_minute and
                       p.signals_by_minute[minute]["confidence"] >= threshold]
            if len(entries) < 50:
                continue
            correct = sum(1 for p, s in entries if s["prediction"] == p.actual_direction)
            acc = correct / len(entries)
            if acc > best_acc:
                best_acc = acc
                best_config = f"minute={minute}, confidence>={threshold}, N={len(entries)}"

    print(f"  Best accuracy: {best_acc:.1%} at {best_config}")

    if best_acc > 0.53:
        print(f"\n  VIABLE: Signal engine shows predictive power.")
        print(f"  With early Polymarket entries near 50/50 prices,")
        print(f"  {best_acc:.1%} accuracy = real profit potential.")
    elif best_acc > 0.50:
        print(f"\n  MARGINAL: Slight edge detected but thin.")
        print(f"  May work with perfect execution but risky.")
    else:
        print(f"\n  NO EDGE: Signal engine cannot predict direction")
        print(f"  better than random at any threshold tested.")


def main():
    parser = argparse.ArgumentParser(description="Signal accuracy backtest")
    parser.add_argument("--data", type=str, default="backtest/data/BTCUSDT_1m_6mo.csv")
    args = parser.parse_args()

    logger.info(f"Loading candles from {args.data}")
    candles = load_candles(args.data)
    logger.info(f"Loaded {len(candles)} candles")

    predictions = run_signal_accuracy_test(candles)
    print_results(predictions)


if __name__ == "__main__":
    main()
