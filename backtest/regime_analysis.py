"""Phase 1: Regime ground truth analysis.

Runs the RegimeDetector over the full kline dataset (Mar 2 - Apr 9),
labels every 5-minute window, and cross-references with actual BTC
price outcomes to identify which regimes predict profitable conditions.
"""

import csv
import sqlite3
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy.regime_detector import RegimeDetector, Regime, RegimeState


@dataclass
class Candle:
    """Minimal candle for regime detector."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: str


def load_klines(csv_path: str) -> list[Candle]:
    """Load Binance 1m klines from CSV."""
    candles = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(Candle(
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                timestamp=row["open_time_utc"],
            ))
    return candles


def run_regime_timeline(
    candles: list[Candle],
    detector: RegimeDetector,
    window_minutes: int = 5,
) -> list[dict]:
    """Run regime detector over candles and label each 5-min window.

    For each 5-minute window:
    - Regime is detected using all candles up to the window start
    - Outcome is the price change over the 5-minute window
    """
    results = []
    n = len(candles)

    # We need at least 120 candles (ATR history) before we start scoring
    warmup = 120

    for i in range(warmup, n - window_minutes, window_minutes):
        # Feed all candles up to window start for regime detection
        lookback_candles = candles[max(0, i - 200):i]
        state = detector.detect(lookback_candles)

        # 5-minute window outcome
        window_start_price = candles[i].open
        window_end_price = candles[i + window_minutes - 1].close
        price_change = window_end_price - window_start_price
        pct_change = price_change / window_start_price * 100

        # Intra-window volatility
        window_high = max(c.high for c in candles[i:i + window_minutes])
        window_low = min(c.low for c in candles[i:i + window_minutes])
        window_range_pct = (window_high - window_low) / window_start_price * 100

        # Did BTC go up or down?
        direction = "UP" if price_change > 0 else "DOWN"

        results.append({
            "timestamp": candles[i].timestamp,
            "regime": state.regime.value,
            "confidence": state.confidence,
            "trend_strength": state.trend_strength,
            "volatility_pct": state.volatility_pct,
            "choppiness": state.choppiness,
            "btc_open": window_start_price,
            "btc_close": window_end_price,
            "price_change": price_change,
            "pct_change": pct_change,
            "window_range_pct": window_range_pct,
            "direction": direction,
        })

    return results


def load_bot_trades(db_path: str) -> list[dict]:
    """Load trades from a bot database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT timestamp, regime, regime_confidence, pnl, side, edge, "
        "combined_signal, btc_price_at_entry FROM trades"
    ).fetchall()
    trades = [dict(r) for r in rows]
    conn.close()
    return trades


def analyse_regime_distribution(results: list[dict]) -> None:
    """Print regime distribution and outcome stats."""
    regime_stats = defaultdict(lambda: {
        "count": 0, "up": 0, "down": 0,
        "total_pct_change": 0.0, "total_range": 0.0,
        "pct_changes": [],
    })

    for r in results:
        reg = r["regime"]
        s = regime_stats[reg]
        s["count"] += 1
        s["total_pct_change"] += r["pct_change"]
        s["total_range"] += r["window_range_pct"]
        s["pct_changes"].append(r["pct_change"])
        if r["direction"] == "UP":
            s["up"] += 1
        else:
            s["down"] += 1

    total = sum(s["count"] for s in regime_stats.values())

    print("\n" + "=" * 90)
    print("REGIME DISTRIBUTION & OUTCOME ANALYSIS")
    print(f"Total 5-minute windows: {total}")
    print(f"Date range: {results[0]['timestamp']} to {results[-1]['timestamp']}")
    print("=" * 90)

    print(f"\n{'Regime':<18} {'Count':>6} {'%':>6} {'Up%':>6} {'Down%':>6} "
          f"{'Avg Move%':>10} {'Avg Range%':>11} {'Bias':>6}")
    print("-" * 80)

    for regime in ["trending_up", "trending_down", "ranging", "high_vol", "low_vol"]:
        if regime not in regime_stats:
            continue
        s = regime_stats[regime]
        pct = s["count"] / total * 100
        up_pct = s["up"] / s["count"] * 100 if s["count"] > 0 else 0
        down_pct = s["down"] / s["count"] * 100 if s["count"] > 0 else 0
        avg_move = s["total_pct_change"] / s["count"] if s["count"] > 0 else 0
        avg_range = s["total_range"] / s["count"] if s["count"] > 0 else 0
        bias = "UP" if up_pct > 52 else ("DOWN" if down_pct > 52 else "NEUTRAL")

        print(f"{regime:<18} {s['count']:>6} {pct:>5.1f}% {up_pct:>5.1f}% {down_pct:>5.1f}% "
              f"{avg_move:>+9.4f}% {avg_range:>10.4f}% {bias:>7}")


def analyse_regime_transitions(results: list[dict]) -> None:
    """Analyse how regimes transition over time."""
    transitions = defaultdict(lambda: defaultdict(int))
    streaks = defaultdict(list)

    current_streak = 1
    for i in range(1, len(results)):
        prev = results[i - 1]["regime"]
        curr = results[i]["regime"]
        transitions[prev][curr] += 1

        if curr == prev:
            current_streak += 1
        else:
            streaks[prev].append(current_streak)
            current_streak = 1
    # Don't forget the last streak
    if results:
        streaks[results[-1]["regime"]].append(current_streak)

    print("\n" + "=" * 90)
    print("REGIME TRANSITION MATRIX (row -> col)")
    print("=" * 90)

    regimes = ["trending_up", "trending_down", "ranging", "high_vol", "low_vol"]
    print(f"{'From / To':<18}", end="")
    for r in regimes:
        print(f"{r[:8]:>10}", end="")
    print()
    print("-" * 70)

    for from_r in regimes:
        if from_r not in transitions:
            continue
        total = sum(transitions[from_r].values())
        print(f"{from_r:<18}", end="")
        for to_r in regimes:
            count = transitions[from_r].get(to_r, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"{pct:>9.1f}%", end="")
        print()

    print("\n" + "=" * 90)
    print("REGIME STREAK LENGTHS")
    print("=" * 90)
    for regime in regimes:
        if regime not in streaks or not streaks[regime]:
            continue
        s = streaks[regime]
        avg = sum(s) / len(s)
        mx = max(s)
        med = sorted(s)[len(s) // 2]
        print(f"{regime:<18} avg={avg:.1f} windows, median={med}, max={mx}, "
              f"occurrences={len(s)}")


def analyse_time_of_day(results: list[dict]) -> None:
    """Analyse regime distribution by hour of day."""
    hour_regimes = defaultdict(lambda: defaultdict(int))

    for r in results:
        try:
            hour = int(r["timestamp"][11:13])
        except (IndexError, ValueError):
            continue
        hour_regimes[hour][r["regime"]] += 1

    print("\n" + "=" * 90)
    print("REGIME DISTRIBUTION BY HOUR (UTC)")
    print("=" * 90)

    regimes = ["trending_up", "trending_down", "ranging", "high_vol", "low_vol"]
    print(f"{'Hour':<6}", end="")
    for r in regimes:
        print(f"{r[:8]:>10}", end="")
    print(f"{'Total':>8}")
    print("-" * 70)

    for hour in range(24):
        if hour not in hour_regimes:
            continue
        total = sum(hour_regimes[hour].values())
        print(f"{hour:>4}h ", end="")
        for regime in regimes:
            count = hour_regimes[hour].get(regime, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"{pct:>9.1f}%", end="")
        print(f"{total:>8}")


def analyse_weekly_periods(results: list[dict]) -> None:
    """Break down regime distribution by week."""
    week_stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "up": 0}))

    for r in results:
        date_str = r["timestamp"][:10]
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        week_start = dt - timedelta(days=dt.weekday())
        week_label = week_start.strftime("%b %d")

        reg = r["regime"]
        week_stats[week_label][reg]["count"] += 1
        if r["direction"] == "UP":
            week_stats[week_label][reg]["up"] += 1

    print("\n" + "=" * 90)
    print("REGIME DISTRIBUTION BY WEEK")
    print("=" * 90)

    regimes = ["trending_up", "trending_down", "ranging", "high_vol", "low_vol"]
    print(f"{'Week':<12}", end="")
    for r in regimes:
        print(f"{r[:8]:>10}", end="")
    print(f"{'Total':>8}")
    print("-" * 75)

    for week in sorted(week_stats.keys(), key=lambda w: datetime.strptime(w, "%b %d")):
        total = sum(week_stats[week][r]["count"] for r in regimes)
        print(f"{week:<12}", end="")
        for regime in regimes:
            count = week_stats[week][regime]["count"]
            pct = count / total * 100 if total > 0 else 0
            print(f"{pct:>9.1f}%", end="")
        print(f"{total:>8}")


def analyse_bot_performance_by_regime(data_dir: str) -> None:
    """Cross-reference bot trades with their stored regime labels."""
    bot_dbs = {
        "Bot A (Contrarian)": "bot_a_contrarian.db",
        "Bot B (Kalman)": "bot_b_kalman.db",
        "Bot C (HMM Regime)": "bot_c_hmm_regime.db",
        "Bot D (Enhanced)": "bot_d_enhanced.db",
        "Bot E (OU Revert)": "bot_e_ou_reversion.db",
    }

    print("\n" + "=" * 90)
    print("BOT PERFORMANCE BY REGIME (from live trade data)")
    print("=" * 90)

    all_bot_regime_stats = {}

    for bot_name, db_file in bot_dbs.items():
        db_path = os.path.join(data_dir, db_file)
        if not os.path.exists(db_path):
            continue

        trades = load_bot_trades(db_path)
        if not trades:
            continue

        regime_stats = defaultdict(lambda: {
            "count": 0, "wins": 0, "pnl": 0.0, "edges": [],
        })

        for t in trades:
            reg = t["regime"] or "unknown"
            s = regime_stats[reg]
            s["count"] += 1
            s["pnl"] += t["pnl"] or 0
            if (t["pnl"] or 0) > 0:
                s["wins"] += 1
            if t["edge"]:
                s["edges"].append(t["edge"])

        all_bot_regime_stats[bot_name] = regime_stats

        print(f"\n--- {bot_name} ({len(trades)} trades) ---")
        print(f"{'Regime':<18} {'Trades':>7} {'WR%':>6} {'PnL':>10} {'Avg Edge':>10}")
        print("-" * 55)

        for regime in ["trending_up", "trending_down", "ranging", "high_vol", "low_vol", "unknown"]:
            if regime not in regime_stats:
                continue
            s = regime_stats[regime]
            wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
            avg_edge = sum(s["edges"]) / len(s["edges"]) if s["edges"] else 0
            marker = " <-- profitable" if s["pnl"] > 0 else ""
            print(f"{regime:<18} {s['count']:>7} {wr:>5.1f}% ${s['pnl']:>+8.2f} "
                  f"{avg_edge:>9.4f}{marker}")

    # Cross-bot regime summary
    print("\n" + "=" * 90)
    print("CROSS-BOT REGIME PROFITABILITY MATRIX")
    print("=" * 90)

    regimes = ["trending_up", "trending_down", "ranging", "high_vol"]
    print(f"{'Bot':<25}", end="")
    for r in regimes:
        print(f"{r[:10]:>12}", end="")
    print()
    print("-" * 75)

    for bot_name, regime_stats in all_bot_regime_stats.items():
        print(f"{bot_name:<25}", end="")
        for regime in regimes:
            if regime in regime_stats and regime_stats[regime]["count"] > 0:
                pnl = regime_stats[regime]["pnl"]
                print(f"${pnl:>+10.2f}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()


def analyse_threshold_sensitivity(candles: list[Candle]) -> None:
    """Test different threshold values to see how regime distribution shifts."""
    print("\n" + "=" * 90)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("How regime distribution changes with different parameter values")
    print("=" * 90)

    # Test trend_threshold values
    print("\n--- Trend Threshold Sensitivity ---")
    print(f"{'trend_thresh':<14} {'trending_up%':>12} {'trending_dn%':>12} "
          f"{'ranging%':>10} {'high_vol%':>10} {'low_vol%':>10}")
    print("-" * 72)

    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        det = RegimeDetector(trend_threshold=thresh)
        results = run_regime_timeline(candles, det)
        counts = defaultdict(int)
        for r in results:
            counts[r["regime"]] += 1
        total = len(results)
        print(f"{thresh:<14.1f}", end="")
        for reg in ["trending_up", "trending_down", "ranging", "high_vol", "low_vol"]:
            pct = counts.get(reg, 0) / total * 100
            print(f"{pct:>11.1f}%", end="")
        print()

    # Test high_vol_percentile values
    print("\n--- High Vol Percentile Sensitivity ---")
    print(f"{'high_vol_pct':<14} {'trending_up%':>12} {'trending_dn%':>12} "
          f"{'ranging%':>10} {'high_vol%':>10} {'low_vol%':>10}")
    print("-" * 72)

    for pct in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        det = RegimeDetector(high_vol_percentile=pct)
        results = run_regime_timeline(candles, det)
        counts = defaultdict(int)
        for r in results:
            counts[r["regime"]] += 1
        total = len(results)
        print(f"{pct:<14.2f}", end="")
        for reg in ["trending_up", "trending_down", "ranging", "high_vol", "low_vol"]:
            pct_val = counts.get(reg, 0) / total * 100
            print(f"{pct_val:>11.1f}%", end="")
        print()

    # Test choppiness_threshold values
    print("\n--- Choppiness Threshold Sensitivity ---")
    print(f"{'chop_thresh':<14} {'trending_up%':>12} {'trending_dn%':>12} "
          f"{'ranging%':>10} {'high_vol%':>10} {'low_vol%':>10}")
    print("-" * 72)

    for chop in [0.4, 0.5, 0.6, 0.7, 0.8]:
        det = RegimeDetector(choppiness_threshold=chop)
        results = run_regime_timeline(candles, det)
        counts = defaultdict(int)
        for r in results:
            counts[r["regime"]] += 1
        total = len(results)
        print(f"{chop:<14.1f}", end="")
        for reg in ["trending_up", "trending_down", "ranging", "high_vol", "low_vol"]:
            pct_val = counts.get(reg, 0) / total * 100
            print(f"{pct_val:>11.1f}%", end="")
        print()


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    kline_path = base_dir / "backtest" / "data" / "binance_klines_1m.csv"
    data_runtime = base_dir / "data_runtime"

    print("Loading klines...")
    candles = load_klines(str(kline_path))
    print(f"Loaded {len(candles)} 1-minute candles")
    print(f"Range: {candles[0].timestamp} to {candles[-1].timestamp}")

    # Run with default parameters
    print("\nRunning regime detector with DEFAULT parameters...")
    detector = RegimeDetector()
    results = run_regime_timeline(candles, detector)
    print(f"Labelled {len(results)} 5-minute windows")

    # Core analyses
    analyse_regime_distribution(results)
    analyse_regime_transitions(results)
    analyse_time_of_day(results)
    analyse_weekly_periods(results)

    # Bot performance cross-reference
    analyse_bot_performance_by_regime(str(data_runtime))

    # Threshold sensitivity
    analyse_threshold_sensitivity(candles)

    # Save timeline to CSV for further analysis
    output_path = base_dir / "backtest" / "data" / "regime_timeline.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nRegime timeline saved to {output_path}")
    print(f"({len(results)} rows)")


if __name__ == "__main__":
    main()
