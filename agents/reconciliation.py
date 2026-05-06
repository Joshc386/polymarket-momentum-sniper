"""Backtest Reconciliation Agent — compares backtest predictions against live bot trades.

Quantifies the approximation gap between the backtest and live paper trading bot
to help calibrate the backtest and identify where it diverges most.

Commands:
    python -m agents.reconciliation compare    — Match backtest to live trades, compute divergence metrics
    python -m agents.reconciliation gaps       — Identify where backtest diverges most from live
    python -m agents.reconciliation calibrate  — Suggest correction factors for the backtest
"""

import argparse
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.common import (
    DATA_DIR,
    DEFAULT_DB_PATH,
    format_table,
    load_backtest_data,
    load_trade_db,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agents.reconciliation")


def _parse_timestamp(ts_str: str) -> datetime | None:
    """Parse various timestamp formats to timezone-aware datetime."""
    if not ts_str or pd.isna(ts_str):
        return None
    try:
        ts_str = str(ts_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _load_live_trades() -> pd.DataFrame:
    """Load live/paper trades from the SQLite database."""
    if not DEFAULT_DB_PATH.exists():
        logger.error(f"Trade database not found: {DEFAULT_DB_PATH}")
        logger.error("The bot must run for a while to generate trade data.")
        sys.exit(1)

    df = load_trade_db()
    if df.empty:
        logger.error("No trades found in database.")
        sys.exit(1)

    # Parse timestamps
    df["parsed_ts"] = df["timestamp"].apply(_parse_timestamp)
    df = df.dropna(subset=["parsed_ts"])
    return df


def _load_backtest_markets() -> pd.DataFrame:
    """Load backtest market data."""
    markets_path = DATA_DIR / "polybacktest_markets.csv"
    if not markets_path.exists():
        logger.error(f"Markets file not found: {markets_path}")
        sys.exit(1)
    return load_backtest_data(markets_path)


def cmd_compare(args: argparse.Namespace) -> None:
    """Match backtest snapshots to live trades and compute divergence metrics."""
    live = _load_live_trades()
    markets = _load_backtest_markets()

    snaps_path = DATA_DIR / "polybacktest_snapshots.csv"
    has_snaps = snaps_path.exists()
    if has_snaps:
        snaps = load_backtest_data(snaps_path)
    else:
        snaps = pd.DataFrame()

    print(f"\n  BACKTEST vs LIVE RECONCILIATION")
    print(f"  {'=' * 60}")

    # Basic stats
    print(f"\n  Live trades: {len(live):,}")
    print(f"  Backtest markets: {len(markets):,}")
    if has_snaps:
        print(f"  Backtest snapshots: {len(snaps):,}")

    # Live trade summary
    if "pnl" in live.columns:
        resolved = live[live["pnl"].notna()]
        wins = (resolved["pnl"] > 0).sum()
        losses = (resolved["pnl"] <= 0).sum()
        total_pnl = resolved["pnl"].sum()
        win_rate = wins / len(resolved) * 100 if len(resolved) > 0 else 0

        print(f"\n  LIVE PERFORMANCE:")
        print(f"    Resolved trades: {len(resolved)}")
        print(f"    Wins: {wins}, Losses: {losses}")
        print(f"    Win rate: {win_rate:.1f}%")
        print(f"    Total P&L: ${total_pnl:.2f}")
        if len(resolved) > 0:
            print(f"    Avg win: ${resolved[resolved['pnl'] > 0]['pnl'].mean():.3f}" if wins > 0 else "")
            print(f"    Avg loss: ${resolved[resolved['pnl'] <= 0]['pnl'].mean():.3f}" if losses > 0 else "")

    # Signal comparison
    signal_cols = ["oracle_lag_signal", "momentum_signal", "liquidation_signal", "combined_signal"]
    available_signals = [c for c in signal_cols if c in live.columns]

    if available_signals:
        print(f"\n  SIGNAL STATISTICS (from live trades):")
        headers = ["Signal", "Mean", "Std", "Min", "Max", "Non-zero %"]
        rows = []
        for col in available_signals:
            vals = live[col].dropna()
            if len(vals) == 0:
                continue
            non_zero = (vals != 0).sum() / len(vals) * 100
            rows.append([
                col.replace("_signal", ""),
                f"{vals.mean():.4f}",
                f"{vals.std():.4f}",
                f"{vals.min():.4f}",
                f"{vals.max():.4f}",
                f"{non_zero:.1f}%",
            ])
        print(format_table(headers, rows))

    # Direction analysis
    if "side" in live.columns and "resolution" in live.columns:
        resolved = live[live["resolution"].notna()]
        if len(resolved) > 0:
            print(f"\n  DIRECTION ANALYSIS:")
            yes_trades = resolved[resolved["side"] == "YES"]
            no_trades = resolved[resolved["side"] == "NO"]

            for label, subset in [("YES", yes_trades), ("NO", no_trades)]:
                if len(subset) > 0:
                    correct = subset["pnl"].apply(lambda x: x > 0 if pd.notna(x) else False).sum()
                    wr = correct / len(subset) * 100
                    print(f"    {label} trades: {len(subset):,} (win rate: {wr:.1f}%)")

    # Time-of-day analysis
    if "parsed_ts" in live.columns and "pnl" in live.columns:
        resolved = live[live["pnl"].notna()].copy()
        if len(resolved) > 0:
            resolved["hour"] = resolved["parsed_ts"].dt.hour
            print(f"\n  PERFORMANCE BY HOUR (UTC):")
            hour_stats = resolved.groupby("hour")["pnl"].agg(["count", "sum", "mean"])
            headers = ["Hour", "Trades", "Total P&L", "Avg P&L"]
            rows = []
            for hour in sorted(hour_stats.index):
                row = hour_stats.loc[hour]
                rows.append([
                    f"{hour:02d}:00",
                    str(int(row["count"])),
                    f"${row['sum']:.2f}",
                    f"${row['mean']:.3f}",
                ])
            print(format_table(headers, rows))

    print()


def cmd_gaps(args: argparse.Namespace) -> None:
    """Identify where the backtest diverges most from live trading."""
    live = _load_live_trades()

    print(f"\n  DIVERGENCE ANALYSIS")
    print(f"  {'=' * 60}")

    if len(live) == 0:
        print("  No live trades to analyse.")
        return

    resolved = live[live["pnl"].notna()].copy()
    if len(resolved) == 0:
        print("  No resolved trades to analyse.")
        return

    # 1. Entry timing analysis
    if "time_remaining_secs" in resolved.columns:
        print(f"\n  ENTRY TIMING:")
        time_remaining = resolved["time_remaining_secs"].dropna()
        if len(time_remaining) > 0:
            print(f"    Mean time remaining at entry: {time_remaining.mean():.0f}s")
            print(f"    Median: {time_remaining.median():.0f}s")
            print(f"    Std: {time_remaining.std():.0f}s")
            print(f"    Range: {time_remaining.min():.0f}s - {time_remaining.max():.0f}s")

            # Bucket by time remaining
            buckets = [(240, 300, "4-5 min"), (180, 240, "3-4 min"),
                       (120, 180, "2-3 min"), (60, 120, "1-2 min"), (0, 60, "0-1 min")]
            headers = ["Window", "Trades", "Win Rate", "Avg P&L"]
            rows = []
            for lo, hi, label in buckets:
                mask = (time_remaining >= lo) & (time_remaining < hi)
                bucket_trades = resolved[mask]
                if len(bucket_trades) > 0:
                    wr = (bucket_trades["pnl"] > 0).sum() / len(bucket_trades) * 100
                    avg_pnl = bucket_trades["pnl"].mean()
                    rows.append([label, str(len(bucket_trades)), f"{wr:.1f}%", f"${avg_pnl:.3f}"])
            if rows:
                print(format_table(headers, rows))

    # 2. Edge analysis (estimated prob vs market implied)
    if "estimated_prob_up" in resolved.columns and "market_implied_prob" in resolved.columns:
        print(f"\n  EDGE ANALYSIS:")
        est = resolved["estimated_prob_up"].dropna()
        mkt = resolved["market_implied_prob"].dropna()
        if len(est) > 0 and len(mkt) > 0:
            edge = (est - mkt).abs()
            print(f"    Mean absolute edge: {edge.mean():.4f}")
            print(f"    Median edge: {edge.median():.4f}")
            print(f"    Max edge: {edge.max():.4f}")

    # 3. Streak patterns
    if "pnl" in resolved.columns:
        print(f"\n  STREAK ANALYSIS:")
        results = (resolved["pnl"] > 0).tolist()
        max_win_streak = max_loss_streak = current_streak = 0
        is_winning = None
        for r in results:
            if r == is_winning:
                current_streak += 1
            else:
                if is_winning is True:
                    max_win_streak = max(max_win_streak, current_streak)
                elif is_winning is False:
                    max_loss_streak = max(max_loss_streak, current_streak)
                current_streak = 1
                is_winning = r
        if is_winning is True:
            max_win_streak = max(max_win_streak, current_streak)
        elif is_winning is False:
            max_loss_streak = max(max_loss_streak, current_streak)

        print(f"    Max winning streak: {max_win_streak}")
        print(f"    Max losing streak: {max_loss_streak}")

    # 4. Key known divergence points
    print(f"\n  KNOWN BACKTEST LIMITATIONS:")
    print(f"    1. Oracle lag: backtest uses btc_start as proxy for both oracle prices")
    print(f"       (live bot has real Chainlink oracle with actual lag)")
    print(f"    2. Orderbook: backtest uses point-in-time snapshots")
    print(f"       (live bot has continuous websocket feed)")
    print(f"    3. Missing signals: L3 Liquidation, L5 Sentiment limited in backtest")
    print(f"    4. No Coinbase cross-confirmation in backtest")
    print(f"    5. Execution: backtest assumes instant fill at best ask")
    print(f"       (live bot may get slippage or partial fills)")
    print()


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Suggest correction factors to bring backtest closer to live performance."""
    live = _load_live_trades()

    print(f"\n  CALIBRATION SUGGESTIONS")
    print(f"  {'=' * 60}")

    resolved = live[live["pnl"].notna()].copy()
    if len(resolved) == 0:
        print("  No resolved trades to calibrate against.")
        return

    wins = (resolved["pnl"] > 0).sum()
    losses = (resolved["pnl"] <= 0).sum()
    live_wr = wins / len(resolved) if len(resolved) > 0 else 0.5
    total_pnl = resolved["pnl"].sum()

    print(f"\n  Live baseline:")
    print(f"    Trades: {len(resolved)}, Win rate: {live_wr:.1%}, P&L: ${total_pnl:.2f}")

    # Estimate signal accuracy
    if "combined_signal" in resolved.columns and "resolution" in resolved.columns:
        resolved_with_signal = resolved[resolved["combined_signal"].notna()]
        if len(resolved_with_signal) > 0:
            correct_direction = 0
            for _, row in resolved_with_signal.iterrows():
                signal = row["combined_signal"]
                resolution = str(row.get("resolution", "")).upper()
                if (signal > 0 and resolution == "UP") or (signal < 0 and resolution == "DOWN"):
                    correct_direction += 1
            signal_accuracy = correct_direction / len(resolved_with_signal)
            print(f"    Signal directional accuracy: {signal_accuracy:.1%}")

    # Calibration factors
    print(f"\n  SUGGESTED CORRECTIONS:")

    # 1. Win rate adjustment
    print(f"\n  1. Win Rate Gap:")
    print(f"     If backtest shows ~50% and live shows {live_wr:.1%},")
    print(f"     the oracle lag approximation accounts for most of this gap.")
    print(f"     Correction: add {(live_wr - 0.5) * 0.5:.3f} to signal before combining")
    print(f"     (represents the real-time information advantage)")

    # 2. Sizing impact
    if "size_usdc" in resolved.columns:
        avg_size = resolved["size_usdc"].mean()
        print(f"\n  2. Position Sizing:")
        print(f"     Avg live trade size: ${avg_size:.2f}")
        print(f"     Ensure backtest uses identical Kelly parameters")
    else:
        print(f"\n  2. Position Sizing:")
        print(f"     size_usdc column not found — ensure backtest logs sizes")

    # 3. Time-based edge
    if "time_remaining_secs" in resolved.columns:
        tr = resolved["time_remaining_secs"].dropna()
        if len(tr) > 0:
            print(f"\n  3. Entry Timing:")
            print(f"     Live avg entry: {tr.mean():.0f}s remaining")
            print(f"     Backtest uses fixed offset snapshots (30, 60, 120, 180, 240s)")
            print(f"     Consider weighting backtest offsets to match live entry distribution")

    # 4. Fee calibration
    if "pnl" in resolved.columns:
        win_pnl = resolved[resolved["pnl"] > 0]["pnl"]
        loss_pnl = resolved[resolved["pnl"] <= 0]["pnl"]
        if len(win_pnl) > 0 and len(loss_pnl) > 0:
            avg_win = win_pnl.mean()
            avg_loss = abs(loss_pnl.mean())
            implied_fee = 1.0 - (avg_win / avg_loss) if avg_loss > 0 else 0
            print(f"\n  4. Fee Calibration:")
            print(f"     Avg win: ${avg_win:.4f}, Avg loss: ${avg_loss:.4f}")
            print(f"     Win/Loss ratio: {avg_win / avg_loss:.3f}" if avg_loss > 0 else "")
            print(f"     Backtest fee rate: 2.0% (on winnings only)")

    print()


def main() -> None:
    """CLI entry point for the Backtest Reconciliation Agent."""
    parser = argparse.ArgumentParser(
        prog="python -m agents.reconciliation",
        description="Backtest Reconciliation — compare backtest vs live trading performance",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # compare
    subparsers.add_parser("compare", help="Match backtest to live trades, compute divergence metrics")

    # gaps
    subparsers.add_parser("gaps", help="Identify where backtest diverges most from live")

    # calibrate
    subparsers.add_parser("calibrate", help="Suggest correction factors for the backtest")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "compare": cmd_compare,
        "gaps": cmd_gaps,
        "calibrate": cmd_calibrate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
