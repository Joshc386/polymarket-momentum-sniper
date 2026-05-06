"""Market Regime Researcher Agent — classifies market conditions and correlates with strategy performance.

Commands:
    python -m agents.regime_researcher analyse                     — Run regime detection over historical data
    python -m agents.regime_researcher correlate --strategy <name> — Cross-reference regimes with strategy P&L
    python -m agents.regime_researcher report                      — Generate regime summary with recommendations
"""

import argparse
import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.common import (
    DATA_DIR,
    format_table,
    load_backtest_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agents.regime_researcher")


def _build_regime_detector() -> 'RegimeDetector':
    """Import and instantiate the backtest's RegimeDetector."""
    # Import from backtest module to reuse exact same logic
    sys.path.insert(0, str(Path(PROJECT_ROOT) / "backtest"))
    from backtest.backtest_real_pricing import Candle, RegimeDetector
    return RegimeDetector(), Candle


def _load_klines_as_candles(klines_path: Path) -> list:
    """Load Binance klines CSV and convert to Candle objects."""
    _, Candle = _build_regime_detector()

    candles = []
    with open(klines_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(Candle(
                open_time=int(row["open_time_ms"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                is_closed=True,
            ))
    return candles


def cmd_analyse(args: argparse.Namespace) -> None:
    """Run regime detection over historical kline data and output a regime timeline."""
    klines_path = DATA_DIR / "binance_klines_1m.csv"
    if not klines_path.exists():
        logger.error(f"Klines not found: {klines_path}")
        logger.error("Run: python -m agents.data_pipeline fetch --source binance")
        sys.exit(1)

    detector, Candle = _build_regime_detector()
    candles = _load_klines_as_candles(klines_path)
    logger.info(f"Loaded {len(candles):,} candles")

    # Classify regime at each 5-minute window boundary
    window_size = 5  # 5 candles = 5 minutes
    regime_counts: dict[str, int] = defaultdict(int)
    regime_timeline: list[dict] = []

    # Slide through candles in 5-minute windows
    for i in range(30, len(candles), window_size):
        history = candles[max(0, i - 30):i]
        if len(history) < 16:  # Need at least ADX/ATR period + 1
            continue

        regime_name, confidence, params = detector.detect(history)
        regime_counts[regime_name] += 1

        ts_ms = candles[i - 1].open_time
        ts_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        regime_timeline.append({
            "timestamp": ts_utc.isoformat(),
            "regime": regime_name,
            "confidence": round(confidence, 4),
        })

    # Write timeline CSV
    output_path = DATA_DIR / "regime_timeline.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "regime", "confidence"])
        writer.writeheader()
        writer.writerows(regime_timeline)

    total_windows = sum(regime_counts.values())
    print(f"\n  REGIME ANALYSIS")
    print(f"  {'=' * 50}")
    print(f"  Total 5-min windows analysed: {total_windows:,}")
    print()

    headers = ["Regime", "Count", "Percentage"]
    rows = []
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        pct = count / total_windows * 100
        rows.append([regime, f"{count:,}", f"{pct:.1f}%"])

    print(format_table(headers, rows))
    print(f"\n  Timeline saved to: {output_path}")

    # Show regime transitions
    transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(1, len(regime_timeline)):
        prev = regime_timeline[i - 1]["regime"]
        curr = regime_timeline[i]["regime"]
        transitions[prev][curr] += 1

    if transitions:
        print(f"\n  REGIME TRANSITIONS (from -> to)")
        print(f"  {'-' * 40}")
        for from_regime in sorted(transitions):
            for to_regime, count in sorted(transitions[from_regime].items(), key=lambda x: -x[1]):
                if from_regime != to_regime:
                    print(f"    {from_regime:12s} -> {to_regime:12s}: {count:,}")
    print()


def cmd_correlate(args: argparse.Namespace) -> None:
    """Cross-reference regime labels with strategy P&L from backtest results."""
    import pandas as pd

    strategy = args.strategy
    timeline_path = DATA_DIR / "regime_timeline.csv"
    snaps_path = DATA_DIR / "polybacktest_snapshots.csv"
    markets_path = DATA_DIR / "polybacktest_markets.csv"

    if not timeline_path.exists():
        logger.error("Regime timeline not found. Run: python -m agents.regime_researcher analyse")
        sys.exit(1)

    if not snaps_path.exists() or not markets_path.exists():
        logger.error("Backtest data not found. Run: python -m agents.data_pipeline fetch")
        sys.exit(1)

    # Load regime timeline
    regimes = pd.read_csv(timeline_path, encoding="utf-8")
    regimes["timestamp"] = pd.to_datetime(regimes["timestamp"], utc=True)
    logger.info(f"Loaded {len(regimes):,} regime classifications")

    # Load market data to get winners
    markets = pd.read_csv(markets_path, encoding="utf-8")
    markets["start_time"] = pd.to_datetime(
        markets["start_time"].str.replace("Z", "+00:00", regex=False), utc=True
    )

    # Map each market to its nearest regime
    regime_performance: dict[str, dict] = defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0})

    for _, market in markets.iterrows():
        start = market["start_time"]
        winner = market.get("winner", "")
        if not winner:
            continue

        # Find regime at market start time
        mask = regimes["timestamp"] <= start
        if mask.any():
            nearest_regime = regimes.loc[mask, "regime"].iloc[-1]
        else:
            nearest_regime = "unknown"

        regime_performance[nearest_regime]["total"] += 1
        # For correlation, we'd need actual trade results per strategy
        # Here we just show the distribution of market outcomes per regime
        if winner.lower() in ["up", "yes"]:
            regime_performance[nearest_regime]["wins"] += 1
        else:
            regime_performance[nearest_regime]["losses"] += 1

    print(f"\n  REGIME-OUTCOME CORRELATION")
    print(f"  Strategy: {strategy}")
    print(f"  {'=' * 60}")

    headers = ["Regime", "Markets", "UP Wins", "DOWN Wins", "UP %"]
    rows = []
    for regime in sorted(regime_performance):
        stats = regime_performance[regime]
        up_pct = stats["wins"] / stats["total"] * 100 if stats["total"] > 0 else 0
        rows.append([
            regime,
            str(stats["total"]),
            str(stats["wins"]),
            str(stats["losses"]),
            f"{up_pct:.1f}%",
        ])

    print(format_table(headers, rows))
    print()
    print("  NOTE: For full strategy-specific correlation, run the backtest")
    print("  with regime labels and pass results through this agent.")
    print()


def cmd_report(args: argparse.Namespace) -> None:
    """Generate a regime summary with recommended strategy weights per regime."""
    import pandas as pd

    timeline_path = DATA_DIR / "regime_timeline.csv"
    if not timeline_path.exists():
        logger.error("Regime timeline not found. Run: python -m agents.regime_researcher analyse")
        sys.exit(1)

    regimes = pd.read_csv(timeline_path, encoding="utf-8")
    regimes["timestamp"] = pd.to_datetime(regimes["timestamp"], utc=True)

    print(f"\n  REGIME REPORT")
    print(f"  {'=' * 60}")

    # Distribution
    regime_counts = regimes["regime"].value_counts()
    total = len(regimes)

    print(f"\n  Distribution ({total:,} windows):")
    for regime, count in regime_counts.items():
        bar_len = int(count / total * 40)
        bar = "#" * bar_len
        print(f"    {regime:12s}: {count:>6,} ({count/total*100:5.1f}%) {bar}")

    # Average confidence by regime
    print(f"\n  Average Confidence:")
    for regime in regime_counts.index:
        mask = regimes["regime"] == regime
        avg_conf = regimes.loc[mask, "confidence"].mean()
        print(f"    {regime:12s}: {avg_conf:.3f}")

    # Time-of-day analysis
    regimes["hour"] = regimes["timestamp"].dt.hour
    print(f"\n  Regime by Hour (UTC) — top 3 hours per regime:")
    for regime in regime_counts.index:
        mask = regimes["regime"] == regime
        hour_counts = regimes.loc[mask, "hour"].value_counts().head(3)
        hours_str = ", ".join(f"{h:02d}:00 ({c})" for h, c in hour_counts.items())
        print(f"    {regime:12s}: {hours_str}")

    # Strategy recommendations based on regime characteristics
    print(f"\n  RECOMMENDED STRATEGY WEIGHTS")
    print(f"  {'-' * 50}")
    recommendations = {
        "trending": {
            "description": "Strong directional movement detected",
            "contrarian_ev": 0.5,
            "signal_follow": 0.4,
            "orderbook_fade": 0.1,
            "notes": "Signal follow benefits from momentum; contrarian EV still valuable",
        },
        "ranging": {
            "description": "Low trend strength, choppy price action",
            "contrarian_ev": 0.4,
            "signal_follow": 0.2,
            "orderbook_fade": 0.4,
            "notes": "Orderbook fade works well in mean-reverting conditions",
        },
        "volatile": {
            "description": "High ATR relative to history",
            "contrarian_ev": 0.5,
            "signal_follow": 0.3,
            "orderbook_fade": 0.2,
            "notes": "Wider spreads create EV opportunities; reduce size via risk manager",
        },
        "low_vol": {
            "description": "Very low volatility — thin edges",
            "contrarian_ev": 0.3,
            "signal_follow": 0.3,
            "orderbook_fade": 0.4,
            "notes": "Consider skipping trades; edges are minimal",
        },
    }

    for regime, rec in recommendations.items():
        print(f"\n    {regime.upper()}: {rec['description']}")
        print(f"      Contrarian EV: {rec['contrarian_ev']:.0%}")
        print(f"      Signal Follow: {rec['signal_follow']:.0%}")
        print(f"      OB Fade:       {rec['orderbook_fade']:.0%}")
        print(f"      Notes: {rec['notes']}")

    print()


def main() -> None:
    """CLI entry point for the Market Regime Researcher Agent."""
    parser = argparse.ArgumentParser(
        prog="python -m agents.regime_researcher",
        description="Market Regime Researcher — classifies conditions and correlates with strategy performance",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyse
    subparsers.add_parser("analyse", help="Run regime detection over historical kline data")

    # correlate
    corr_parser = subparsers.add_parser("correlate", help="Cross-reference regimes with strategy P&L")
    corr_parser.add_argument(
        "--strategy",
        default="contrarian_ev",
        help="Strategy name to correlate (default: contrarian_ev)",
    )

    # report
    subparsers.add_parser("report", help="Generate regime summary with recommendations")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "analyse": cmd_analyse,
        "correlate": cmd_correlate,
        "report": cmd_report,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
