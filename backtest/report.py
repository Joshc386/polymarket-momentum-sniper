"""Generate performance reports from backtest results.

Outputs detailed stats, time-of-day analysis, signal calibration,
and daily equity curves.
"""

import csv
import os
from collections import defaultdict
from datetime import datetime, timezone

from backtest.replay_engine import BacktestResult, WindowResult


def print_summary(result: BacktestResult):
    """Print a comprehensive backtest summary to stdout."""
    print()
    print("=" * 70)
    print("  BACKTEST RESULTS")
    print("=" * 70)
    print()

    # Overview
    if result.total_windows > 0:
        start_dt = datetime.fromtimestamp(
            result.windows[0].window_start_time / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d")
        end_dt = datetime.fromtimestamp(
            result.windows[-1].window_start_time / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d")
        print(f"  Period:           {start_dt} -> {end_dt}")

    print(f"  Total Windows:    {result.total_windows:,}")
    print(f"  Trades Attempted: {result.trades_attempted:,}")
    print(f"  Trades Filled:    {result.trades_placed:,}")
    if result.trades_attempted > 0:
        print(f"  Fill Rate:        {result.fill_rate:.1%}  ({result.fill_failures} failed)")
    print(f"  Skip Rate:        {result.skip_rate:.1%}")
    print()

    # P&L
    print("  -- Performance --")
    print(f"  Initial Bankroll: ${result.initial_bankroll:.2f}")
    print(f"  Final Bankroll:   ${result.final_bankroll:.2f}")
    print(f"  Total P&L:        ${result.total_pnl:+.2f}")
    print(f"  ROI:              {result.roi:+.1%}")
    print(f"  Avg P&L/Trade:    ${result.avg_pnl_per_trade:+.4f}")
    print()

    # Win/Loss
    print("  -- Win/Loss --")
    print(f"  Wins:             {result.wins}")
    print(f"  Losses:           {result.losses}")
    print(f"  Win Rate:         {result.win_rate:.1%}")
    print()

    # Risk
    print("  -- Risk --")
    print(f"  Max Bankroll:     ${result.max_bankroll:.2f}")
    print(f"  Min Bankroll:     ${result.min_bankroll:.2f}")
    print(f"  Max Drawdown:     {result.max_drawdown:.1%}")
    print()

    # Streaks
    trades = [w for w in result.windows if w.trade_placed]
    if trades:
        max_win_streak = _max_streak(trades, won=True)
        max_loss_streak = _max_streak(trades, won=False)
        print(f"  Max Win Streak:   {max_win_streak}")
        print(f"  Max Loss Streak:  {max_loss_streak}")
        print()

    # Direction bias
    up_trades = [w for w in trades if w.trade_side == "YES"]
    down_trades = [w for w in trades if w.trade_side == "NO"]
    print("  -- Direction Bias --")
    print(f"  YES (Up) trades:  {len(up_trades)}  "
          f"(WR: {_wr(up_trades):.1%})")
    print(f"  NO (Down) trades: {len(down_trades)}  "
          f"(WR: {_wr(down_trades):.1%})")
    print()


def print_timing_analysis(result: BacktestResult):
    """Analyse performance by entry timing within the 5-min window."""
    trades = [w for w in result.windows if w.trade_placed]
    if not trades:
        return

    print("  -- Entry Timing Analysis --")
    buckets = {
        "5:00-3:00 (early)": (120, 300),
        "3:00-1:30 (mid)":   (90, 120),
        "1:30-0:30 (late)":  (30, 90),
        "0:30-0:05 (final)": (5, 30),
    }

    for label, (lo, hi) in buckets.items():
        bucket_trades = [t for t in trades if lo <= t.entry_time_remaining <= hi]
        if not bucket_trades:
            print(f"  {label:25s}  No trades")
            continue
        wins = sum(1 for t in bucket_trades if t.won)
        pnl = sum(t.trade_pnl for t in bucket_trades)
        wr = wins / len(bucket_trades)
        avg_edge = sum(t.trade_edge for t in bucket_trades) / len(bucket_trades)
        print(
            f"  {label:25s}  "
            f"N={len(bucket_trades):4d}  "
            f"WR={wr:.1%}  "
            f"P&L=${pnl:+.2f}  "
            f"AvgEdge={avg_edge:.4f}"
        )
    print()


def print_hourly_analysis(result: BacktestResult):
    """Analyse performance by hour of day (UTC)."""
    trades = [w for w in result.windows if w.trade_placed]
    if not trades:
        return

    print("  -- Hourly Analysis (UTC) --")
    by_hour = defaultdict(list)
    for t in trades:
        hour = datetime.fromtimestamp(
            t.window_start_time / 1000, tz=timezone.utc
        ).hour
        by_hour[hour].append(t)

    for hour in sorted(by_hour.keys()):
        ht = by_hour[hour]
        wins = sum(1 for t in ht if t.won)
        pnl = sum(t.trade_pnl for t in ht)
        wr = wins / len(ht) if ht else 0
        print(
            f"  {hour:02d}:00  "
            f"N={len(ht):4d}  "
            f"WR={wr:.1%}  "
            f"P&L=${pnl:+.2f}"
        )
    print()


def print_signal_calibration(result: BacktestResult):
    """Check if est_prob_up correlates with actual outcomes."""
    trades = [w for w in result.windows if w.trade_placed]
    if not trades:
        return

    print("  -- Signal Calibration --")
    # Bucket by estimated probability
    buckets = [
        ("35-40%", 0.35, 0.40),
        ("40-45%", 0.40, 0.45),
        ("45-50%", 0.45, 0.50),
        ("50-55%", 0.50, 0.55),
        ("55-60%", 0.55, 0.60),
        ("60-65%", 0.60, 0.65),
    ]

    for label, lo, hi in buckets:
        bucket = [w for w in result.windows
                  if lo <= w.est_prob_up < hi and w.actual_direction != ""]
        if not bucket:
            continue
        actual_up = sum(1 for w in bucket if w.actual_direction == "UP")
        actual_pct = actual_up / len(bucket)
        avg_est = sum(w.est_prob_up for w in bucket) / len(bucket)
        cal_error = abs(avg_est - actual_pct)
        print(
            f"  Est {label:8s}  "
            f"N={len(bucket):5d}  "
            f"Actual Up={actual_pct:.1%}  "
            f"Avg Est={avg_est:.1%}  "
            f"CalErr={cal_error:.3f}"
        )
    print()


def print_daily_equity(result: BacktestResult, last_n_days: int = 30):
    """Print daily P&L for the last N days."""
    if not result.daily_pnl:
        return

    print(f"  -- Daily P&L (last {last_n_days} days) --")
    sorted_days = sorted(result.daily_pnl.keys())
    recent = sorted_days[-last_n_days:] if len(sorted_days) > last_n_days else sorted_days

    cumulative = 0.0
    for day in recent:
        pnl = result.daily_pnl[day]
        cumulative += pnl
        bar_width = int(abs(pnl) * 5)
        bar = "+" * bar_width if pnl >= 0 else "-" * bar_width
        print(f"  {day}  ${pnl:+6.2f}  Cum: ${cumulative:+7.2f}  {bar}")
    print()


def export_trades_csv(result: BacktestResult, output_path: str):
    """Export all trades to a CSV file for external analysis."""
    trades = [w for w in result.windows if w.trade_placed]
    if not trades:
        print("No trades to export.")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window_start", "direction", "side", "entry_price", "size",
            "edge", "pnl", "won", "time_remaining",
            "oracle_lag", "momentum", "orderbook", "combined", "est_prob_up",
            "btc_open", "btc_close",
        ])
        for t in trades:
            ts = datetime.fromtimestamp(
                t.window_start_time / 1000, tz=timezone.utc
            ).isoformat()
            writer.writerow([
                ts, t.actual_direction, t.trade_side, f"{t.trade_price:.4f}",
                f"{t.trade_size:.2f}", f"{t.trade_edge:.4f}",
                f"{t.trade_pnl:.4f}", int(t.won), f"{t.entry_time_remaining:.0f}",
                f"{t.oracle_lag_signal:.4f}", f"{t.momentum_signal:.4f}",
                f"{t.orderbook_signal_val:.4f}",
                f"{t.combined_signal:.4f}", f"{t.est_prob_up:.4f}",
                f"{t.open_price:.2f}", f"{t.close_price:.2f}",
            ])

    print(f"  Exported {len(trades)} trades to {output_path}")


def full_report(result: BacktestResult, export_csv_path: str | None = None):
    """Print the complete backtest report."""
    print_summary(result)
    print_timing_analysis(result)
    print_hourly_analysis(result)
    print_signal_calibration(result)
    print_daily_equity(result)

    if export_csv_path:
        export_trades_csv(result, export_csv_path)


# ── Helpers ────────────────────────────────────────────────────────────

def _max_streak(trades: list[WindowResult], won: bool) -> int:
    streak = 0
    max_s = 0
    for t in trades:
        if t.won == won:
            streak += 1
            max_s = max(max_s, streak)
        else:
            streak = 0
    return max_s


def _wr(trades: list[WindowResult]) -> float:
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.won) / len(trades)
