"""Phase 4: Analytics — session summaries, signal calibration, performance reports.

Can be imported into main.py or run standalone:
    python -m logging_db.analytics [--db PATH] [--days N] [--export-csv]
"""

import csv
import sqlite3
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class SessionSummary:
    period: str                     # "all", "today", "24h", etc.
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_edge: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    profit_factor: float = 0.0     # gross wins / gross losses
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    current_streak: int = 0        # positive = wins, negative = losses
    roi_pct: float = 0.0
    # By timing bucket
    pnl_by_bucket: dict = field(default_factory=dict)
    trades_by_bucket: dict = field(default_factory=dict)
    wr_by_bucket: dict = field(default_factory=dict)
    # By side
    yes_trades: int = 0
    yes_wins: int = 0
    no_trades: int = 0
    no_wins: int = 0
    # Signal layer averages (for winning vs losing trades)
    avg_oracle_signal_wins: float = 0.0
    avg_oracle_signal_losses: float = 0.0
    avg_momentum_signal_wins: float = 0.0
    avg_momentum_signal_losses: float = 0.0


@dataclass
class CalibrationBucket:
    """Model probability bucket for calibration analysis."""
    range_low: float
    range_high: float
    label: str
    total_signals: int = 0
    actual_up_count: int = 0
    actual_up_rate: float = 0.0
    avg_model_prob: float = 0.0
    avg_market_prob: float = 0.0


# ── Timing buckets ────────────────────────────────────────────────────

TIMING_BUCKETS = [
    ("5:00-3:00", 300, 180),
    ("3:00-1:30", 180, 90),
    ("1:30-0:30", 90, 30),
    ("0:30-0:05", 30, 5),
]

def _bucket_for_time(secs_remaining: float) -> str:
    for label, high, low in TIMING_BUCKETS:
        if low <= secs_remaining < high:
            return label
    if secs_remaining >= 300:
        return "5:00-3:00"
    return "0:30-0:05"


# ── Analytics class ───────────────────────────────────────────────────

class Analytics:
    """Generates performance reports from the SQLite trade/signal database."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    # ── Session Summary ───────────────────────────────────────────────

    def session_summary(
        self,
        mode: str = "paper",
        since: str | None = None,
        initial_bankroll: float = 100.0,
    ) -> SessionSummary:
        """Generate a full session summary.

        Args:
            mode: 'paper' or 'live'
            since: ISO timestamp filter (trades after this time). None = all.
            initial_bankroll: For ROI calculation.
        """
        is_paper = 1 if mode == "paper" else 0
        params: list = [is_paper]
        where = "WHERE is_paper = ? AND resolution IS NOT NULL"
        if since:
            where += " AND timestamp >= ?"
            params.append(since)

        rows = self.conn.execute(
            f"SELECT * FROM trades {where} ORDER BY timestamp ASC", params
        ).fetchall()

        s = SessionSummary(period=since or "all")
        if not rows:
            return s

        s.total_trades = len(rows)
        gross_wins = 0.0
        gross_losses = 0.0
        pnl_list = []
        edge_list = []

        # Streak tracking
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        # Bucket accumulators
        bucket_pnl: dict[str, float] = {}
        bucket_trades: dict[str, int] = {}
        bucket_wins: dict[str, int] = {}

        # Signal averages
        oracle_wins, oracle_losses = [], []
        mom_wins, mom_losses = [], []

        for row in rows:
            pnl = row["pnl"] or 0.0
            pnl_list.append(pnl)
            won = pnl > 0

            if won:
                s.wins += 1
                gross_wins += pnl
                current_streak = max(current_streak + 1, 1) if current_streak >= 0 else 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                s.losses += 1
                gross_losses += abs(pnl)
                current_streak = min(current_streak - 1, -1) if current_streak <= 0 else -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))

            edge = row["edge"]
            if edge is not None:
                edge_list.append(edge)

            # Side tracking
            side = row["side"]
            if side == "YES":
                s.yes_trades += 1
                if won: s.yes_wins += 1
            else:
                s.no_trades += 1
                if won: s.no_wins += 1

            # Timing bucket
            tr = row["time_remaining_secs"] or 0
            bucket = _bucket_for_time(tr)
            bucket_pnl[bucket] = bucket_pnl.get(bucket, 0.0) + pnl
            bucket_trades[bucket] = bucket_trades.get(bucket, 0) + 1
            if won:
                bucket_wins[bucket] = bucket_wins.get(bucket, 0) + 1

            # Signal layer tracking
            olag = row["oracle_lag_signal"]
            mom = row["momentum_signal"]
            if won:
                if olag is not None: oracle_wins.append(olag)
                if mom is not None: mom_wins.append(mom)
            else:
                if olag is not None: oracle_losses.append(olag)
                if mom is not None: mom_losses.append(mom)

        s.total_pnl = sum(pnl_list)
        s.win_rate = s.wins / s.total_trades if s.total_trades > 0 else 0
        s.avg_pnl = s.total_pnl / s.total_trades if s.total_trades > 0 else 0
        s.avg_edge = statistics.mean(edge_list) if edge_list else 0
        win_pnls = [p for p in pnl_list if p > 0]
        loss_pnls = [p for p in pnl_list if p <= 0]
        s.avg_win = statistics.mean(win_pnls) if win_pnls else 0
        s.avg_loss = statistics.mean(loss_pnls) if loss_pnls else 0
        s.best_trade = max(pnl_list) if pnl_list else 0
        s.worst_trade = min(pnl_list) if pnl_list else 0
        s.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")
        s.longest_win_streak = max_win_streak
        s.longest_loss_streak = max_loss_streak
        s.current_streak = current_streak
        s.roi_pct = (s.total_pnl / initial_bankroll) * 100 if initial_bankroll > 0 else 0

        s.pnl_by_bucket = bucket_pnl
        s.trades_by_bucket = bucket_trades
        s.wr_by_bucket = {
            k: bucket_wins.get(k, 0) / v if v > 0 else 0
            for k, v in bucket_trades.items()
        }

        s.avg_oracle_signal_wins = statistics.mean(oracle_wins) if oracle_wins else 0
        s.avg_oracle_signal_losses = statistics.mean(oracle_losses) if oracle_losses else 0
        s.avg_momentum_signal_wins = statistics.mean(mom_wins) if mom_wins else 0
        s.avg_momentum_signal_losses = statistics.mean(mom_losses) if mom_losses else 0

        return s

    # ── Signal Calibration ────────────────────────────────────────────

    def signal_calibration(self, mode: str = "paper") -> list[CalibrationBucket]:
        """Analyse model probability calibration vs actual outcomes.

        Groups signal_log entries by model probability buckets and compares
        predicted P(Up) vs actual resolution rate.
        """
        # We need signal_log entries with actual_resolution filled in.
        # Join signal_log to trades by market_id to get resolutions.
        # Alternatively, use signal_log entries that have actual_resolution set.
        # For now, join approach — get resolution from trades table.
        sql = """
            SELECT
                s.estimated_prob_up,
                s.market_implied_prob,
                s.oracle_lag_signal,
                s.momentum_signal,
                s.liquidation_signal,
                s.time_remaining_secs,
                s.trade_placed,
                t.resolution
            FROM signal_log s
            LEFT JOIN (
                SELECT market_id, resolution
                FROM trades
                WHERE resolution IS NOT NULL
                GROUP BY market_id
            ) t ON s.market_id = t.market_id
            WHERE t.resolution IS NOT NULL
        """
        rows = self.conn.execute(sql).fetchall()

        # Define probability buckets
        bucket_defs = [
            (0.00, 0.40, "<40%"),
            (0.40, 0.45, "40-45%"),
            (0.45, 0.50, "45-50%"),
            (0.50, 0.55, "50-55%"),
            (0.55, 0.60, "55-60%"),
            (0.60, 1.01, ">60%"),
        ]
        buckets = {
            label: CalibrationBucket(range_low=lo, range_high=hi, label=label)
            for lo, hi, label in bucket_defs
        }
        prob_sums: dict[str, list[float]] = {label: [] for _, _, label in bucket_defs}
        mkt_sums: dict[str, list[float]] = {label: [] for _, _, label in bucket_defs}

        for row in rows:
            prob = row["estimated_prob_up"] or 0.5
            resolution = row["resolution"]
            actual_up = 1 if resolution == "UP" else 0

            for lo, hi, label in bucket_defs:
                if lo <= prob < hi:
                    b = buckets[label]
                    b.total_signals += 1
                    b.actual_up_count += actual_up
                    prob_sums[label].append(prob)
                    mp = row["market_implied_prob"]
                    if mp is not None:
                        mkt_sums[label].append(mp)
                    break

        result = []
        for lo, hi, label in bucket_defs:
            b = buckets[label]
            if b.total_signals > 0:
                b.actual_up_rate = b.actual_up_count / b.total_signals
                b.avg_model_prob = statistics.mean(prob_sums[label])
                b.avg_market_prob = statistics.mean(mkt_sums[label]) if mkt_sums[label] else 0
            result.append(b)

        return result

    # ── Hourly P&L Breakdown ──────────────────────────────────────────

    def hourly_pnl(self, mode: str = "paper", since: str | None = None) -> dict[int, dict]:
        """P&L grouped by hour-of-day (UTC). Useful for finding best/worst trading hours."""
        is_paper = 1 if mode == "paper" else 0
        params: list = [is_paper]
        where = "WHERE is_paper = ? AND resolution IS NOT NULL"
        if since:
            where += " AND timestamp >= ?"
            params.append(since)

        rows = self.conn.execute(
            f"SELECT timestamp, pnl FROM trades {where}", params
        ).fetchall()

        hours: dict[int, dict] = {}
        for row in rows:
            try:
                ts = datetime.fromisoformat(row["timestamp"])
                h = ts.hour
            except (ValueError, TypeError):
                continue
            if h not in hours:
                hours[h] = {"trades": 0, "pnl": 0.0, "wins": 0}
            hours[h]["trades"] += 1
            hours[h]["pnl"] += row["pnl"] or 0
            if (row["pnl"] or 0) > 0:
                hours[h]["wins"] += 1

        return dict(sorted(hours.items()))

    # ── Export Signal Log to CSV ──────────────────────────────────────

    def export_signal_csv(self, output_path: str, since: str | None = None):
        """Export signal_log table to CSV for external analysis."""
        params: list = []
        where = ""
        if since:
            where = "WHERE timestamp >= ?"
            params.append(since)

        rows = self.conn.execute(
            f"SELECT * FROM signal_log {where} ORDER BY timestamp ASC", params
        ).fetchall()

        if not rows:
            print(f"No signal data to export.")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        columns = rows[0].keys()
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([row[c] for c in columns])

        print(f"Exported {len(rows)} signal entries to {output_path}")

    # ── Export Trades to CSV ──────────────────────────────────────────

    def export_trades_csv(self, output_path: str, mode: str = "paper"):
        """Export trades table to CSV."""
        is_paper = 1 if mode == "paper" else 0
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE is_paper = ? ORDER BY timestamp ASC",
            (is_paper,),
        ).fetchall()

        if not rows:
            print("No trade data to export.")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        columns = rows[0].keys()
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([row[c] for c in columns])

        print(f"Exported {len(rows)} trades to {output_path}")


# ── Terminal Report Renderer ──────────────────────────────────────────

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
RESET = "\033[0m"


def _pnl_color(val: float) -> str:
    if val > 0: return GREEN
    if val < 0: return RED
    return DIM


def print_session_report(s: SessionSummary, initial_bankroll: float = 100.0):
    """Print a formatted session report to terminal."""
    print()
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print(f"{BOLD}{CYAN}  SESSION PERFORMANCE REPORT{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print(f"  Period: {s.period}")
    print()

    # Overview
    print(f"  {BOLD}-- Overview --{RESET}")
    print(f"  Total trades:    {s.total_trades}")
    print(f"  Wins / Losses:   {GREEN}{s.wins}{RESET} / {RED}{s.losses}{RESET}")
    wr_c = GREEN if s.win_rate >= 0.55 else RED if s.win_rate < 0.45 else YELLOW
    print(f"  Win rate:        {wr_c}{s.win_rate:.1%}{RESET}")
    pnl_c = _pnl_color(s.total_pnl)
    print(f"  Total P&L:       {pnl_c}${s.total_pnl:+.2f}{RESET}")
    roi_c = _pnl_color(s.roi_pct)
    print(f"  ROI:             {roi_c}{s.roi_pct:+.1f}%{RESET}")
    print(f"  Avg edge:        {s.avg_edge:+.4f}")
    pf_c = GREEN if s.profit_factor > 1.0 else RED
    pf_str = f"{s.profit_factor:.2f}" if s.profit_factor != float("inf") else "inf"
    print(f"  Profit factor:   {pf_c}{pf_str}{RESET}")
    print()

    # Trade stats
    print(f"  {BOLD}-- Trade Stats --{RESET}")
    print(f"  Avg P&L/trade:   {_pnl_color(s.avg_pnl)}${s.avg_pnl:+.4f}{RESET}")
    print(f"  Avg win:         {GREEN}${s.avg_win:+.4f}{RESET}")
    print(f"  Avg loss:        {RED}${s.avg_loss:+.4f}{RESET}")
    print(f"  Best trade:      {GREEN}${s.best_trade:+.2f}{RESET}")
    print(f"  Worst trade:     {RED}${s.worst_trade:+.2f}{RESET}")
    print()

    # Streaks
    print(f"  {BOLD}-- Streaks --{RESET}")
    print(f"  Longest win:     {GREEN}{s.longest_win_streak}{RESET}")
    print(f"  Longest loss:    {RED}{s.longest_loss_streak}{RESET}")
    streak_sign = "W" if s.current_streak > 0 else "L"
    streak_c = GREEN if s.current_streak > 0 else RED if s.current_streak < 0 else DIM
    print(f"  Current:         {streak_c}{abs(s.current_streak)}{streak_sign}{RESET}")
    print()

    # By side
    print(f"  {BOLD}-- By Side --{RESET}")
    yes_wr = s.yes_wins / s.yes_trades if s.yes_trades > 0 else 0
    no_wr = s.no_wins / s.no_trades if s.no_trades > 0 else 0
    print(f"  YES:  {s.yes_trades} trades, {s.yes_wins} wins ({yes_wr:.0%})")
    print(f"  NO:   {s.no_trades} trades, {s.no_wins} wins ({no_wr:.0%})")
    print()

    # By timing bucket
    if s.pnl_by_bucket:
        print(f"  {BOLD}-- P&L by Entry Timing --{RESET}")
        print(f"  {'Bucket':<14} {'Trades':>7} {'WR':>7} {'P&L':>10}")
        print(f"  {'-'*40}")
        for label, _, _ in TIMING_BUCKETS:
            t = s.trades_by_bucket.get(label, 0)
            wr = s.wr_by_bucket.get(label, 0)
            pnl = s.pnl_by_bucket.get(label, 0)
            if t > 0:
                pc = _pnl_color(pnl)
                print(f"  {label:<14} {t:>7} {wr:>6.0%} {pc}${pnl:>+9.2f}{RESET}")
        print()

    # Signal layer comparison
    print(f"  {BOLD}-- Signal Quality (avg by outcome) --{RESET}")
    print(f"  {'Signal':<18} {'Wins':>10} {'Losses':>10} {'Delta':>10}")
    print(f"  {'-'*50}")
    o_delta = s.avg_oracle_signal_wins - s.avg_oracle_signal_losses
    m_delta = s.avg_momentum_signal_wins - s.avg_momentum_signal_losses
    print(
        f"  {'Oracle lag':<18} {s.avg_oracle_signal_wins:>+10.4f} "
        f"{s.avg_oracle_signal_losses:>+10.4f} {_pnl_color(o_delta)}{o_delta:>+10.4f}{RESET}"
    )
    print(
        f"  {'Momentum':<18} {s.avg_momentum_signal_wins:>+10.4f} "
        f"{s.avg_momentum_signal_losses:>+10.4f} {_pnl_color(m_delta)}{m_delta:>+10.4f}{RESET}"
    )
    print()

    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print()


def print_calibration_report(buckets: list[CalibrationBucket]):
    """Print signal calibration table."""
    print()
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print(f"{BOLD}{CYAN}  SIGNAL CALIBRATION REPORT{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print(f"  Compares model's P(Up) vs actual UP rate.")
    print(f"  Perfect calibration: model prob == actual rate.")
    print()
    print(f"  {'Bucket':<10} {'N':>6} {'Model':>8} {'Market':>8} {'Actual':>8} {'Gap':>8}")
    print(f"  {'-'*52}")

    for b in buckets:
        if b.total_signals == 0:
            print(f"  {b.label:<10} {0:>6}   {'--':>6}   {'--':>6}   {'--':>6}   {'--':>6}")
            continue
        gap = b.avg_model_prob - b.actual_up_rate
        gc = GREEN if abs(gap) < 0.03 else YELLOW if abs(gap) < 0.06 else RED
        print(
            f"  {b.label:<10} {b.total_signals:>6} "
            f"{b.avg_model_prob:>7.1%} {b.avg_market_prob:>7.1%} "
            f"{b.actual_up_rate:>7.1%} {gc}{gap:>+7.1%}{RESET}"
        )

    print()
    print(f"  {DIM}Gap = model - actual. Green (<3%) = well calibrated.{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print()


def print_hourly_report(hours: dict[int, dict]):
    """Print P&L by hour-of-day."""
    print()
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print(f"{BOLD}{CYAN}  P&L BY HOUR (UTC){RESET}")
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print(f"  {'Hour':>6} {'Trades':>8} {'Wins':>6} {'WR':>7} {'P&L':>10}")
    print(f"  {'-'*40}")

    for h in range(24):
        if h not in hours:
            continue
        d = hours[h]
        wr = d["wins"] / d["trades"] if d["trades"] > 0 else 0
        pc = _pnl_color(d["pnl"])
        print(f"  {h:>4}:00 {d['trades']:>8} {d['wins']:>6} {wr:>6.0%} {pc}${d['pnl']:>+9.2f}{RESET}")

    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print()


# ── CLI entry point ───────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Polymarket Bot — Performance Analytics")
    parser.add_argument("--db", default="./data_runtime/trades.db", help="Path to SQLite database")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"], help="Trading mode to analyse")
    parser.add_argument("--days", type=int, default=0, help="Limit to last N days (0 = all)")
    parser.add_argument("--bankroll", type=float, default=100.0, help="Initial bankroll for ROI calc")
    parser.add_argument("--export-csv", action="store_true", help="Export trades + signals to CSV")
    parser.add_argument("--calibration", action="store_true", help="Show signal calibration report")
    parser.add_argument("--hourly", action="store_true", help="Show P&L by hour")
    parser.add_argument("--all", action="store_true", help="Show all reports")
    args = parser.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    a = Analytics(db_path)
    a.connect()

    since = None
    if args.days > 0:
        since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()

    # Always show session summary
    s = a.session_summary(mode=args.mode, since=since, initial_bankroll=args.bankroll)
    print_session_report(s, initial_bankroll=args.bankroll)

    if args.calibration or args.all:
        buckets = a.signal_calibration(mode=args.mode)
        print_calibration_report(buckets)

    if args.hourly or args.all:
        hours = a.hourly_pnl(mode=args.mode, since=since)
        print_hourly_report(hours)

    if args.export_csv or args.all:
        trades_path = str(Path(db_path).parent / "trades_export.csv")
        signals_path = str(Path(db_path).parent / "signals_export.csv")
        a.export_trades_csv(trades_path, mode=args.mode)
        a.export_signal_csv(signals_path, since=since)

    a.close()


if __name__ == "__main__":
    main()
