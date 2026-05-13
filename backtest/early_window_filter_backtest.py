"""Early-Window Filter Backtest — does delaying entry help?

The high-EV investigation found that trades fired with 240-300s remaining
in the window (i.e., 0-60s into the window) have:
  - 337 trades, 27.9% WR, -$213 PnL
  - Single biggest dimensional leak in Bot G's data

Hypothesis: very early entries have noisier signals because momentum has
no history, orderbook hasn't settled, and the bot pulls the trigger on
whatever fires first. Delaying the earliest entry should improve outcomes.

Current config:
  earliest_entry_secs: 270   (can enter when 270s remain = 30s into window)
  latest_entry_secs:    60   (must enter by 60s remain = 240s into window)

This backtest simulates tightening `earliest_entry_secs` to various
thresholds and reports the resulting PnL/WR/Sharpe/DD.

Caveat: this assumes the bot SKIPS the early trade entirely. In reality,
the bot would likely re-fire later in the same window when the next valid
signal appears. So this gives a lower-bound estimate (we'd capture some
of those re-fires in reality). But the captured signal would be DIFFERENT
from the skipped one — possibly better, possibly worse. Can't know without
simulating signal history per window.

Usage:
    python -m backtest.early_window_filter_backtest
"""

import sqlite3
import statistics
from collections import defaultdict
from typing import NamedTuple

DB_PATH = "data_runtime/bot_g_signal_aligned.db"


def fmt_pnl(x: float) -> str:
    return f"${x:+.2f}"


def fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def load_trades() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, side, entry_price, size_usdc, pnl, edge,
               estimated_prob_up, regime, time_remaining_secs,
               combined_signal
        FROM trades
        WHERE pnl IS NOT NULL AND size_usdc > 0
        ORDER BY timestamp
    """)
    cols = [d[0] for d in cur.description]
    out = []
    for r in cur.fetchall():
        d = dict(zip(cols, r))
        d["won"] = d["pnl"] > 0
        d["pnl_per_dollar"] = d["pnl"] / d["size_usdc"] if d["size_usdc"] else 0
        # secs INTO window = 300 - time_remaining
        d["secs_into_window"] = 300.0 - (d["time_remaining_secs"] or 0)
        out.append(d)
    conn.close()
    return out


def metrics(pnls: list[float]) -> dict:
    if not pnls:
        return {"total_pnl": 0, "n": 0, "wr": 0, "sharpe": 0, "max_dd": 0,
                "avg_pnl": 0, "std": 0}
    n = len(pnls)
    total = sum(pnls)
    mean = total / n
    std = statistics.stdev(pnls) if n > 1 else 0
    sharpe = mean / std if std > 0 else 0
    # Drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return {
        "total_pnl": total, "n": n, "avg_pnl": mean, "std": std,
        "sharpe": sharpe, "max_dd": max_dd,
        "wr": sum(1 for p in pnls if p > 0) / n,
    }


def main() -> None:
    print("=" * 85)
    print("EARLY-WINDOW FILTER BACKTEST")
    print("Does delaying earliest_entry_secs improve performance?")
    print("=" * 85)

    trades = load_trades()
    n = len(trades)
    print(f"\nLoaded {n:,} Bot G trades")
    print(f"Current config: earliest_entry_secs=270 (30s into window)")
    print(f"                latest_entry_secs=60   (240s into window)")

    # ── Time-into-window distribution ────────────────────────────────

    print(f"\n--- Distribution of entries by secs into window ---")
    print(f"  {'Range':<22} {'N':>6} {'%':>6} {'WR':>7} {'PnL':>9} {'Avg':>8}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*8}")

    intervals = [
        (0, 30, "0-30s into window"),
        (30, 60, "30-60s"),
        (60, 90, "60-90s"),
        (90, 120, "90-120s"),
        (120, 150, "120-150s"),
        (150, 180, "150-180s"),
        (180, 210, "180-210s"),
        (210, 240, "210-240s"),
        (240, 270, "240-270s"),
    ]
    for lo, hi, lbl in intervals:
        bucket = [t for t in trades if lo <= t["secs_into_window"] < hi]
        if not bucket:
            continue
        m = metrics([t["pnl"] for t in bucket])
        print(f"  {lbl:<22} {m['n']:>6,} {m['n']/n:>5.1%} "
              f"{fmt_pct(m['wr']):>7} {fmt_pnl(m['total_pnl']):>9} "
              f"{fmt_pnl(m['avg_pnl']):>8}")

    # ── Simulated filter: tighten earliest_entry_secs ────────────────

    print(f"\n{'=' * 85}")
    print("FILTER SIMULATION: tighten earliest_entry_secs threshold")
    print(f"{'=' * 85}\n")
    print("If we'd required at least X seconds into window before entering,")
    print("here's what Bot G's PnL would have been:\n")

    # We're simulating: trades where secs_into_window < threshold get DROPPED
    # (treated as "didn't enter"). We don't model re-entry later in window.

    thresholds = [30, 45, 60, 75, 90, 105, 120, 150, 180]

    baseline = metrics([t["pnl"] for t in trades])
    print(f"  Baseline (no change):     "
          f"N={baseline['n']:>5,} PnL {fmt_pnl(baseline['total_pnl']):>9} "
          f"WR {fmt_pct(baseline['wr']):>6} "
          f"Sharpe {baseline['sharpe']:.4f} MaxDD {fmt_pnl(baseline['max_dd'])}")
    print()
    print(f"  {'Min secs in window':<22} {'N':>6} {'PnL':>10} {'vs Base':>10} "
          f"{'WR':>7} {'Sharpe':>8} {'MaxDD':>10}")
    print(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*10} {'-'*7} {'-'*8} {'-'*10}")

    for thresh in thresholds:
        kept = [t for t in trades if t["secs_into_window"] >= thresh]
        m = metrics([t["pnl"] for t in kept])
        delta = m["total_pnl"] - baseline["total_pnl"]
        print(f"  >={thresh:>3}s into window     "
              f"{m['n']:>6,} {fmt_pnl(m['total_pnl']):>10} "
              f"{fmt_pnl(delta):>10} {fmt_pct(m['wr']):>7} "
              f"{m['sharpe']:>8.4f} {fmt_pnl(m['max_dd']):>10}")

    # ── Combined: by EV bucket × time bucket ────────────────────────

    print(f"\n{'=' * 85}")
    print("WHERE DOES THE LEAK CONCENTRATE? (EV bucket x time bucket)")
    print(f"{'=' * 85}\n")

    ev_buckets = [
        (-99, 0.05, "0-0.05"),
        (0.05, 0.10, "0.05-0.10"),
        (0.10, 0.15, "0.10-0.15"),
        (0.15, 0.20, "0.15-0.20"),
        (0.20, 99, "0.20+"),
    ]
    time_buckets = [
        (0, 60, "0-60s (very early)"),
        (60, 120, "60-120s (early)"),
        (120, 180, "120-180s (mid)"),
        (180, 240, "180-240s (late)"),
    ]

    print(f"  {'EV bucket':<15} | " + " ".join(f"{lbl:<22}" for _, _, lbl in time_buckets))
    print("  " + "-" * 105)
    for evlo, evhi, evlbl in ev_buckets:
        row_cells = []
        for tlo, thi, tlbl in time_buckets:
            bucket = [t for t in trades
                      if evlo <= t["edge"] < evhi
                      and tlo <= t["secs_into_window"] < thi]
            if bucket:
                m = metrics([t["pnl"] for t in bucket])
                cell = f"N={m['n']:>3} WR{fmt_pct(m['wr']):>5} {fmt_pnl(m['total_pnl']):>8}"
            else:
                cell = "(empty)              "
            row_cells.append(cell)
        print(f"  {evlbl:<15} | " + " ".join(f"{c:<22}" for c in row_cells))

    # ── Slice: high-EV early entries vs high-EV late entries ─────────

    print(f"\n{'=' * 85}")
    print("HIGH-EV (>=0.15) TRADES: where in window do they lose?")
    print(f"{'=' * 85}\n")
    high_ev = [t for t in trades if t["edge"] >= 0.15]
    print(f"  Total high-EV trades: {len(high_ev):,}")
    print(f"  Total high-EV PnL:    {fmt_pnl(sum(t['pnl'] for t in high_ev))}\n")
    for tlo, thi, tlbl in time_buckets:
        bucket = [t for t in high_ev if tlo <= t["secs_into_window"] < thi]
        if not bucket:
            continue
        m = metrics([t["pnl"] for t in bucket])
        print(f"  {tlbl:<22} N={m['n']:>4,} WR {fmt_pct(m['wr']):>6} "
              f"PnL {fmt_pnl(m['total_pnl']):>9} Avg {fmt_pnl(m['avg_pnl']):>8}")

    # ── Slice: low-EV (already profitable) trades by time ────────────

    print(f"\n  LOW-EV (<0.15) trades by time:")
    low_ev = [t for t in trades if t["edge"] < 0.15]
    print(f"  Total low-EV trades:  {len(low_ev):,}")
    print(f"  Total low-EV PnL:     {fmt_pnl(sum(t['pnl'] for t in low_ev))}\n")
    for tlo, thi, tlbl in time_buckets:
        bucket = [t for t in low_ev if tlo <= t["secs_into_window"] < thi]
        if not bucket:
            continue
        m = metrics([t["pnl"] for t in bucket])
        print(f"  {tlbl:<22} N={m['n']:>4,} WR {fmt_pct(m['wr']):>6} "
              f"PnL {fmt_pnl(m['total_pnl']):>9} Avg {fmt_pnl(m['avg_pnl']):>8}")

    # ── What about late trades? Should we also tighten latest_entry_secs?

    print(f"\n{'=' * 85}")
    print("LATE ENTRIES — should latest_entry_secs be tightened too?")
    print(f"{'=' * 85}\n")

    late_thresholds = [60, 75, 90, 120, 150, 180]
    print(f"  Filter: skip trades entered too late (secs_remaining < threshold)\n")
    print(f"  {'Max secs into window':<22} {'N kept':>8} {'PnL':>10} {'vs Base':>10} "
          f"{'WR':>7} {'Sharpe':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*7} {'-'*8}")

    for thresh in [180, 210, 240, 270]:  # secs into window
        # secs_into_window = 300 - time_remaining. So thresh=240 means we
        # only keep trades that entered when AT LEAST (300-thresh)=60s remain.
        kept = [t for t in trades if t["secs_into_window"] <= thresh]
        m = metrics([t["pnl"] for t in kept])
        delta = m["total_pnl"] - baseline["total_pnl"]
        print(f"  <={thresh:>3}s into window     "
              f"{m['n']:>8,} {fmt_pnl(m['total_pnl']):>10} "
              f"{fmt_pnl(delta):>10} {fmt_pct(m['wr']):>7} "
              f"{m['sharpe']:>8.4f}")

    # ── Combined filter: tighten both ends ──────────────────────────

    print(f"\n{'=' * 85}")
    print("COMBINED: tighten both earliest and latest entry windows")
    print(f"{'=' * 85}\n")

    combos = [
        (30, 240, "current (>=30s, <=240s)"),
        (45, 240, "delay early to >=45s"),
        (60, 240, "delay early to >=60s"),
        (75, 240, "delay early to >=75s"),
        (90, 240, "delay early to >=90s"),
        (60, 210, "delay early >=60s, latest <=210s"),
        (60, 180, "delay early >=60s, latest <=180s"),
        (90, 180, "tight: >=90s and <=180s"),
    ]

    print(f"  {'Config':<35} {'N':>6} {'PnL':>10} {'vs Base':>10} "
          f"{'WR':>7} {'Sharpe':>8} {'MaxDD':>10}")
    print(f"  {'-'*35} {'-'*6} {'-'*10} {'-'*10} {'-'*7} {'-'*8} {'-'*10}")
    for min_t, max_t, lbl in combos:
        kept = [t for t in trades
                if min_t <= t["secs_into_window"] <= max_t]
        m = metrics([t["pnl"] for t in kept])
        delta = m["total_pnl"] - baseline["total_pnl"]
        print(f"  {lbl:<35} {m['n']:>6,} {fmt_pnl(m['total_pnl']):>10} "
              f"{fmt_pnl(delta):>10} {fmt_pct(m['wr']):>7} "
              f"{m['sharpe']:>8.4f} {fmt_pnl(m['max_dd']):>10}")

    # ── Verdict ──────────────────────────────────────────────────────

    print(f"\n{'=' * 85}")
    print("VERDICT")
    print(f"{'=' * 85}\n")

    # Compute the "skipped" PnL for the best earliest-entry filter
    for thresh in [45, 60, 75, 90]:
        skipped = [t for t in trades if t["secs_into_window"] < thresh]
        if not skipped:
            continue
        s_m = metrics([t["pnl"] for t in skipped])
        print(f"  Trades skipped with min={thresh}s: "
              f"N={s_m['n']:>4} PnL {fmt_pnl(s_m['total_pnl']):>9} "
              f"WR {fmt_pct(s_m['wr']):>6}")


if __name__ == "__main__":
    main()
