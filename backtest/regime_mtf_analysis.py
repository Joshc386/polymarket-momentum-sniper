"""Phase 1b: MTF regime analysis.

Runs the MTFRegimeDetector over the full 2-year kline dataset,
comparing single-TF vs multi-TF regime classifications and their
correlation with 5-minute BTC price outcomes.

This validates whether the MTF approach produces better regime
labels before we proceed to walk-forward calibration.
"""

import csv
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy.regime_detector import RegimeDetector, Regime
from strategy.mtf_regime_detector import (
    MTFRegimeDetector, Candle, load_candles_from_csv,
)

DATA_DIR = Path(__file__).resolve().parent / "data"


def aggregate_candles(candles_1m: list[Candle], tf_minutes: int) -> list[Candle]:
    """Aggregate 1m candles into higher timeframe candles.

    This creates HTF candles from 1m data so we have perfectly aligned
    timeframes. More reliable than using separate CSV files which may
    have slight alignment differences.

    Args:
        candles_1m: 1-minute candles sorted by timestamp.
        tf_minutes: Target timeframe in minutes (15, 60, 240, 1440).

    Returns:
        Aggregated candles for the target timeframe.
    """
    if not candles_1m:
        return []

    tf_ms = tf_minutes * 60_000
    result: list[Candle] = []
    bucket: list[Candle] = []
    bucket_start_ms = (candles_1m[0].timestamp_ms // tf_ms) * tf_ms

    for c in candles_1m:
        candle_bucket = (c.timestamp_ms // tf_ms) * tf_ms
        if candle_bucket != bucket_start_ms and bucket:
            # Close the current bucket
            result.append(Candle(
                open=bucket[0].open,
                high=max(b.high for b in bucket),
                low=min(b.low for b in bucket),
                close=bucket[-1].close,
                volume=sum(b.volume for b in bucket),
                timestamp_ms=bucket_start_ms,
            ))
            bucket = []
            bucket_start_ms = candle_bucket
        bucket.append(c)

    # Close last bucket
    if bucket:
        result.append(Candle(
            open=bucket[0].open,
            high=max(b.high for b in bucket),
            low=min(b.low for b in bucket),
            close=bucket[-1].close,
            volume=sum(b.volume for b in bucket),
            timestamp_ms=bucket_start_ms,
        ))

    return result


def run_comparison(
    candles_1m: list[Candle],
    candles_15m: list[Candle],
    candles_1h: list[Candle],
    candles_4h: list[Candle],
    candles_1d: list[Candle],
    start_idx: int = 0,
    window_minutes: int = 5,
) -> list[dict]:
    """Run both single-TF and MTF detectors side by side.

    For each 5-minute window, we:
    1. Feed 1m candles to both detectors
    2. Feed HTF candles to the MTF detector
    3. Record both classifications + actual BTC outcome

    Args:
        candles_1m: Full 1-minute candle series.
        candles_15m/1h/4h/1d: Pre-aggregated higher timeframe candles.
        start_idx: Index in candles_1m to start scoring from.
        window_minutes: Size of outcome window (5 for Polymarket).

    Returns:
        List of comparison dicts for each window.
    """
    base_detector = RegimeDetector()
    mtf_detector = MTFRegimeDetector()

    # Build timestamp indexes for HTF candles
    def build_index(candles: list[Candle]) -> dict[int, int]:
        idx = {}
        for i, c in enumerate(candles):
            idx[c.timestamp_ms] = i
        return idx

    idx_15m = build_index(candles_15m)
    idx_1h = build_index(candles_1h)
    idx_4h = build_index(candles_4h)
    idx_1d = build_index(candles_1d)

    # Helper: find the latest HTF candle at or before a given 1m timestamp
    def get_htf_slice(
        htf_candles: list[Candle],
        htf_index: dict[int, int],
        current_ms: int,
        tf_ms: int,
        lookback: int = 50,
    ) -> list[Candle]:
        """Get the most recent `lookback` HTF candles up to current_ms."""
        bucket_ms = (current_ms // tf_ms) * tf_ms
        # Find the nearest bucket at or before current
        while bucket_ms not in htf_index and bucket_ms > htf_candles[0].timestamp_ms:
            bucket_ms -= tf_ms
        if bucket_ms not in htf_index:
            return []
        end_idx = htf_index[bucket_ms] + 1
        start_idx_htf = max(0, end_idx - lookback)
        return htf_candles[start_idx_htf:end_idx]

    results = []
    n = len(candles_1m)
    warmup = 200  # 200 1m candles before we start scoring

    actual_start = max(start_idx, warmup)
    total_windows = (n - actual_start - window_minutes) // window_minutes
    report_every = max(1, total_windows // 20)

    for step, i in enumerate(range(actual_start, n - window_minutes, window_minutes)):
        current_ms = candles_1m[i].timestamp_ms

        # Feed 1m candles to both detectors (lookback 200)
        lookback_1m = candles_1m[max(0, i - 200):i]

        # Feed HTF candles to MTF detector
        htf_15m = get_htf_slice(candles_15m, idx_15m, current_ms, 15 * 60_000)
        htf_1h = get_htf_slice(candles_1h, idx_1h, current_ms, 60 * 60_000)
        htf_4h = get_htf_slice(candles_4h, idx_4h, current_ms, 240 * 60_000)
        htf_1d = get_htf_slice(candles_1d, idx_1d, current_ms, 1440 * 60_000)

        if htf_15m:
            mtf_detector.update_timeframe("15m", htf_15m)
        if htf_1h:
            mtf_detector.update_timeframe("1h", htf_1h)
        if htf_4h:
            mtf_detector.update_timeframe("4h", htf_4h)
        if htf_1d:
            mtf_detector.update_timeframe("1d", htf_1d)

        # Detect with both
        # Base detector expects objects with .high/.low/.close/.open attributes
        base_state = base_detector.detect(lookback_1m)
        mtf_state = mtf_detector.detect(lookback_1m)

        # 5-minute outcome
        window_open = candles_1m[i].open
        window_close = candles_1m[min(i + window_minutes - 1, n - 1)].close
        price_change = window_close - window_open
        pct_change = price_change / window_open * 100 if window_open > 0 else 0
        direction = "UP" if price_change > 0 else "DOWN"

        # Window volatility
        window_high = max(c.high for c in candles_1m[i:i + window_minutes])
        window_low = min(c.low for c in candles_1m[i:i + window_minutes])
        window_range = (window_high - window_low) / window_open * 100 if window_open > 0 else 0

        ts = datetime.fromtimestamp(current_ms / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        results.append({
            "timestamp": ts,
            "btc_price": window_open,
            "direction": direction,
            "pct_change": pct_change,
            "window_range_pct": window_range,
            # Base detector
            "base_regime": base_state.regime.value,
            "base_confidence": base_state.confidence,
            "base_trend_str": base_state.trend_strength,
            "base_vol_pct": base_state.volatility_pct,
            "base_chop": base_state.choppiness,
            # MTF detector
            "mtf_regime": mtf_state.regime.value,
            "mtf_confidence": mtf_state.confidence,
            "mtf_alignment": mtf_state.alignment_score,
            "mtf_align_count": mtf_state.alignment_count,
            "mtf_htf_trend": mtf_state.htf_trend,
        })

        if step % report_every == 0 and step > 0:
            print(f"  Progress: {step}/{total_windows} windows "
                  f"({step/total_windows*100:.0f}%) - {ts[:10]}")

    return results


def print_comparison_report(results: list[dict]) -> None:
    """Print detailed comparison of base vs MTF regime detection."""
    print("\n" + "=" * 95)
    print("BASE vs MTF REGIME DETECTION COMPARISON")
    print(f"Total 5-minute windows: {len(results)}")
    print(f"Date range: {results[0]['timestamp'][:10]} to {results[-1]['timestamp'][:10]}")
    print("=" * 95)

    # 1. Distribution comparison
    base_counts = defaultdict(int)
    mtf_counts = defaultdict(int)
    for r in results:
        base_counts[r["base_regime"]] += 1
        mtf_counts[r["mtf_regime"]] += 1

    total = len(results)
    print("\n--- REGIME DISTRIBUTION ---")
    print(f"{'Regime':<18} {'Base %':>8} {'MTF %':>8} {'Shift':>8}")
    print("-" * 46)
    for reg in ["trending_up", "trending_down", "ranging", "high_vol", "low_vol"]:
        base_pct = base_counts.get(reg, 0) / total * 100
        mtf_pct = mtf_counts.get(reg, 0) / total * 100
        shift = mtf_pct - base_pct
        print(f"{reg:<18} {base_pct:>7.1f}% {mtf_pct:>7.1f}% {shift:>+7.1f}%")

    # 2. Agreement matrix
    agreement = defaultdict(lambda: defaultdict(int))
    for r in results:
        agreement[r["base_regime"]][r["mtf_regime"]] += 1

    agree_count = sum(1 for r in results if r["base_regime"] == r["mtf_regime"])
    print(f"\nOverall agreement: {agree_count}/{total} ({agree_count/total*100:.1f}%)")

    print("\n--- AGREEMENT MATRIX (Base rows, MTF columns) ---")
    regimes = ["trending_up", "trending_down", "ranging", "high_vol", "low_vol"]
    print(f"{'Base \\ MTF':<16}", end="")
    for r in regimes:
        print(f"{r[:8]:>10}", end="")
    print()
    print("-" * 68)
    for base_r in regimes:
        row_total = sum(agreement[base_r].values())
        if row_total == 0:
            continue
        print(f"{base_r:<16}", end="")
        for mtf_r in regimes:
            count = agreement[base_r].get(mtf_r, 0)
            pct = count / row_total * 100 if row_total > 0 else 0
            print(f"{pct:>9.1f}%", end="")
        print()

    # 3. Directional accuracy per regime
    print("\n--- DIRECTIONAL BIAS BY REGIME (does the regime predict UP/DOWN?) ---")
    print("(A good regime detector should show strong directional bias in trending regimes)")
    print(f"\n{'Detector':<8} {'Regime':<18} {'Windows':>8} {'UP%':>6} {'DOWN%':>6} {'Avg Move%':>10}")
    print("-" * 62)

    for detector in ["base", "mtf"]:
        regime_key = f"{detector}_regime"
        stats = defaultdict(lambda: {"count": 0, "up": 0, "total_pct": 0.0})
        for r in results:
            reg = r[regime_key]
            s = stats[reg]
            s["count"] += 1
            s["total_pct"] += r["pct_change"]
            if r["direction"] == "UP":
                s["up"] += 1

        for reg in regimes:
            if reg not in stats:
                continue
            s = stats[reg]
            up_pct = s["up"] / s["count"] * 100
            down_pct = 100 - up_pct
            avg_move = s["total_pct"] / s["count"]
            label = detector.upper()
            print(f"{label:<8} {reg:<18} {s['count']:>8} {up_pct:>5.1f}% "
                  f"{down_pct:>5.1f}% {avg_move:>+9.5f}%")
        print()

    # 4. Streak analysis
    print("--- REGIME STREAK LENGTHS ---")
    for detector in ["base", "mtf"]:
        regime_key = f"{detector}_regime"
        streaks = defaultdict(list)
        current_streak = 1
        for i in range(1, len(results)):
            if results[i][regime_key] == results[i-1][regime_key]:
                current_streak += 1
            else:
                streaks[results[i-1][regime_key]].append(current_streak)
                current_streak = 1
        if results:
            streaks[results[-1][regime_key]].append(current_streak)

        print(f"\n  {detector.upper()} detector:")
        for reg in regimes:
            if reg not in streaks or not streaks[reg]:
                continue
            s = streaks[reg]
            avg = sum(s) / len(s)
            mx = max(s)
            med = sorted(s)[len(s) // 2]
            print(f"    {reg:<18} avg={avg:.1f}, median={med}, max={mx}, "
                  f"occurrences={len(s)}")

    # 5. MTF alignment analysis
    print("\n--- MTF ALIGNMENT SCORE ANALYSIS ---")
    align_buckets = {
        "Strong DOWN (< -0.5)": lambda a: a < -0.5,
        "Moderate DOWN (-0.5 to -0.2)": lambda a: -0.5 <= a < -0.2,
        "Weak/Flat (-0.2 to +0.2)": lambda a: -0.2 <= a <= 0.2,
        "Moderate UP (+0.2 to +0.5)": lambda a: 0.2 < a <= 0.5,
        "Strong UP (> +0.5)": lambda a: a > 0.5,
    }

    print(f"{'Alignment':<32} {'Windows':>8} {'UP%':>6} {'DOWN%':>6} "
          f"{'Avg Move%':>10} {'Avg Range%':>11}")
    print("-" * 80)

    for label, condition in align_buckets.items():
        matching = [r for r in results if condition(r["mtf_alignment"])]
        if not matching:
            continue
        count = len(matching)
        up_pct = sum(1 for r in matching if r["direction"] == "UP") / count * 100
        avg_move = sum(r["pct_change"] for r in matching) / count
        avg_range = sum(r["window_range_pct"] for r in matching) / count
        print(f"{label:<32} {count:>8} {up_pct:>5.1f}% {100-up_pct:>5.1f}% "
              f"{avg_move:>+9.5f}% {avg_range:>10.5f}%")

    # 6. Bot trading period analysis (Apr 5-9)
    bot_period = [r for r in results if "2026-04-05" <= r["timestamp"][:10] <= "2026-04-09"]
    if bot_period:
        print("\n--- BOT TRADING PERIOD (Apr 5-9) REGIME COMPARISON ---")
        print(f"{'Regime':<18} {'Base %':>8} {'MTF %':>8}")
        print("-" * 38)

        bp_base = defaultdict(int)
        bp_mtf = defaultdict(int)
        for r in bot_period:
            bp_base[r["base_regime"]] += 1
            bp_mtf[r["mtf_regime"]] += 1
        bp_total = len(bot_period)

        for reg in regimes:
            base_pct = bp_base.get(reg, 0) / bp_total * 100
            mtf_pct = bp_mtf.get(reg, 0) / bp_total * 100
            print(f"{reg:<18} {base_pct:>7.1f}% {mtf_pct:>7.1f}%")

    # 7. Profitable period analysis (Mar 27 - Apr 4, original bot)
    profit_period = [r for r in results if "2026-03-27" <= r["timestamp"][:10] <= "2026-04-04"]
    if profit_period:
        print("\n--- PROFITABLE PERIOD (Mar 27 - Apr 4, original bot) REGIME COMPARISON ---")
        print(f"{'Regime':<18} {'Base %':>8} {'MTF %':>8}")
        print("-" * 38)

        pp_base = defaultdict(int)
        pp_mtf = defaultdict(int)
        for r in profit_period:
            pp_base[r["base_regime"]] += 1
            pp_mtf[r["mtf_regime"]] += 1
        pp_total = len(profit_period)

        for reg in regimes:
            base_pct = pp_base.get(reg, 0) / pp_total * 100
            mtf_pct = pp_mtf.get(reg, 0) / pp_total * 100
            print(f"{reg:<18} {base_pct:>7.1f}% {mtf_pct:>7.1f}%")


def main() -> None:
    print("Loading 1-minute candles...")
    candles_1m = load_candles_from_csv(str(DATA_DIR / "binance_klines_1m.csv"))
    print(f"  Loaded {len(candles_1m)} 1m candles")
    print(f"  Range: {datetime.fromtimestamp(candles_1m[0].timestamp_ms/1000, tz=timezone.utc).date()} "
          f"to {datetime.fromtimestamp(candles_1m[-1].timestamp_ms/1000, tz=timezone.utc).date()}")

    print("\nAggregating higher timeframes from 1m data...")
    candles_15m = aggregate_candles(candles_1m, 15)
    candles_1h = aggregate_candles(candles_1m, 60)
    candles_4h = aggregate_candles(candles_1m, 240)
    candles_1d = aggregate_candles(candles_1m, 1440)
    print(f"  15m: {len(candles_15m)} candles")
    print(f"  1h:  {len(candles_1h)} candles")
    print(f"  4h:  {len(candles_4h)} candles")
    print(f"  1d:  {len(candles_1d)} candles")

    print("\nRunning base vs MTF comparison over full dataset...")
    results = run_comparison(
        candles_1m, candles_15m, candles_1h, candles_4h, candles_1d,
    )
    print(f"\nScored {len(results)} 5-minute windows")

    print_comparison_report(results)

    # Save full results
    output_path = DATA_DIR / "regime_mtf_comparison.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nFull comparison saved to {output_path}")
    print(f"({len(results)} rows)")


if __name__ == "__main__":
    main()
