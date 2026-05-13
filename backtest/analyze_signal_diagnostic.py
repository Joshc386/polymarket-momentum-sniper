"""Analyze per-tick signal diagnostic data.

Run this AFTER restarting Bot G (and Bot K) with signal_diagnostic.enabled=true
and collecting a few hours of tick data. Answers the questions:

1. What's the distribution of L1 / L4 / combined_signal across ALL ticks
   (not just trade ticks)?
2. Which L4 sub-component (imbalance/flow/mid_dev/top_pressure/thickness)
   is most responsible for L4's bias?
3. What's the L1 lag_component vs open_component breakdown? Is the
   exchange-vs-oracle component the dominant driver?
4. How many ticks have est_prob_up > 0.5 vs < 0.5? If balanced but bot
   only traded YES, the bug is in the entry filter chain.
5. What entry_reason / filter_blocked values are most common when est_prob
   is bearish? This tells us what's gating NO trades.

Usage:
    python -m backtest.analyze_signal_diagnostic [--bot bot_g_signal_aligned]
                                                  [--hours 4]
"""

import argparse
import os
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict

DEFAULT_DB = "data_runtime/bot_g_signal_aligned_signal_diag.db"


def fmt(x, decimals=4):
    if x is None:
        return "  None"
    return f"{x:+.{decimals}f}"


def stats(values, name, decimals=4):
    """Mean / median / std / p10 / p90 / pct_pos."""
    if not values:
        return f"  {name}: (no data)"
    vs = sorted(values)
    n = len(vs)
    avg = sum(vs) / n
    std = statistics.stdev(vs) if n > 1 else 0.0
    pos_pct = sum(1 for v in vs if v > 0) / n
    return (
        f"  {name}: "
        f"avg {fmt(avg, decimals)}  "
        f"median {fmt(vs[n//2], decimals)}  "
        f"std {fmt(std, decimals)}  "
        f"p10 {fmt(vs[int(n*0.10)], decimals)}  "
        f"p90 {fmt(vs[int(n*0.90)], decimals)}  "
        f"pct_pos {pos_pct:.1%}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--hours", type=float, default=None,
                        help="Only analyse the most recent N hours")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Diagnostic DB not found at {args.db}")
        print("Start Bot G with signal_diagnostic.enabled=true and let it run for a few hours first.")
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    # Determine time range
    cur.execute("SELECT MIN(unix_time), MAX(unix_time), COUNT(*) FROM signal_ticks")
    min_t, max_t, total = cur.fetchone()
    if total == 0:
        print("Diagnostic DB is empty. Run Bot G for a while first.")
        return

    from datetime import datetime, timezone
    min_dt = datetime.fromtimestamp(min_t, tz=timezone.utc).isoformat()
    max_dt = datetime.fromtimestamp(max_t, tz=timezone.utc).isoformat()
    span_hours = (max_t - min_t) / 3600

    print("=" * 90)
    print("SIGNAL DIAGNOSTIC ANALYSIS")
    print(f"DB: {args.db}")
    print(f"Time range: {min_dt} → {max_dt} ({span_hours:.1f}h)")
    print(f"Total ticks: {total:,}")
    print("=" * 90)

    # Optional time filter
    where = ""
    params = []
    if args.hours:
        cutoff = max_t - args.hours * 3600
        where = "WHERE unix_time >= ?"
        params = [cutoff]
        print(f"\nFiltering to last {args.hours}h (since {datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()})")

    # ── Load all rows ────────────────────────────────────────────────
    cols_query = """
        SELECT regime, schedule_override,
               l1_oracle_lag, l1_lag_component, l1_open_component,
               l2_momentum, l3_liquidation,
               l4_orderbook, l4_imbalance, l4_flow, l4_mid_dev,
               l4_top_pressure, l4_thickness,
               l5_sentiment, l6_fade, l7_taker_ratio, l8_clob_flow,
               l9b_absorption, l10_exhaustion, l11_trade_size, l12_wallet_flow,
               coinbase_direction, combined_signal,
               est_prob_up, market_implied_prob, prob_edge, required_edge,
               would_pick_side, would_enter, entry_reason,
               filter_blocked, risk_can_trade, risk_reason, trade_placed,
               btc_price, oracle_price, oracle_open_price,
               secs_remaining
        FROM signal_ticks
    """
    cur.execute(cols_query + " " + where, params)
    col_names = [d[0] for d in cur.description]
    rows = cur.fetchall()
    print(f"Analysing {len(rows):,} rows\n")

    def col(name):
        i = col_names.index(name)
        return [r[i] for r in rows if r[i] is not None]

    # ── 1. Distribution of core signals ──────────────────────────────
    print("=" * 90)
    print("1. SIGNAL DISTRIBUTIONS across ALL ticks")
    print("=" * 90)
    print(stats(col("l1_oracle_lag"), "L1 oracle_lag      "))
    print(stats(col("l2_momentum"), "L2 momentum        "))
    print(stats(col("l3_liquidation"), "L3 liquidation     "))
    print(stats(col("l4_orderbook"), "L4 orderbook       "))
    print(stats(col("l5_sentiment"), "L5 sentiment       "))
    print(stats(col("combined_signal"), "combined_signal    "))
    print(stats(col("est_prob_up"), "est_prob_up        ", decimals=4))

    # ── 2. L1 sub-component breakdown ────────────────────────────────
    print("\n" + "=" * 90)
    print("2. L1 SUB-COMPONENT BREAKDOWN")
    print("   l1 = 0.6 * lag_component + 0.4 * open_component")
    print("   lag_component = (binance - oracle_NOW) / oracle / 0.001  (60%)")
    print("   open_component = (binance - oracle_OPEN) / oracle / 0.001  (40%)")
    print("=" * 90)
    print(stats(col("l1_lag_component"), "L1 lag_component   "))
    print(stats(col("l1_open_component"), "L1 open_component  "))

    # If lag_component is consistently positive while open_component varies,
    # the lag component is the bug source (exchange-vs-oracle offset)
    lag = col("l1_lag_component")
    opn = col("l1_open_component")
    if lag and opn:
        print("\n  Diagnostic:")
        lag_pct_pos = sum(1 for v in lag if v > 0) / len(lag)
        opn_pct_pos = sum(1 for v in opn if v > 0) / len(opn)
        print(f"    lag_component  positive: {lag_pct_pos:.1%}")
        print(f"    open_component positive: {opn_pct_pos:.1%}")
        if lag_pct_pos > 0.85 and abs(opn_pct_pos - 0.5) < 0.2:
            print("    → lag_component has persistent bias (likely feed offset)")
            print("       open_component is balanced (real signal)")
            print("       FIX: remove lag_component from L1 (use only open_component)")

    # ── 3. L4 sub-component breakdown ────────────────────────────────
    print("\n" + "=" * 90)
    print("3. L4 SUB-COMPONENT BREAKDOWN")
    print("   l4 = 0.30*imbalance + 0.25*flow + 0.20*mid_dev")
    print("        + 0.15*top_pressure + 0.10*thickness")
    print("=" * 90)
    print(stats(col("l4_imbalance"), "L4 imbalance       "))
    print(stats(col("l4_flow"), "L4 flow            "))
    print(stats(col("l4_mid_dev"), "L4 mid_dev         "))
    print(stats(col("l4_top_pressure"), "L4 top_pressure    "))
    print(stats(col("l4_thickness"), "L4 thickness       "))

    # Weighted contribution to L4 — find the dominant biased component
    weights = {"imbalance": 0.30, "flow": 0.25, "mid_dev": 0.20,
               "top_pressure": 0.15, "thickness": 0.10}
    print("\n  Average weighted contribution to L4:")
    for name, w in weights.items():
        vals = col(f"l4_{name}")
        if vals:
            avg = sum(vals) / len(vals)
            print(f"    {name:>13}: {fmt(w * avg)}  (weight {w}, mean {fmt(avg)})")

    # ── 4. est_prob_up distribution & would_pick_side ────────────────
    print("\n" + "=" * 90)
    print("4. WHAT THE BOT WOULD DO (every tick, not just trades)")
    print("=" * 90)
    est_probs = col("est_prob_up")
    if est_probs:
        yes_probs = sum(1 for p in est_probs if p > 0.5) / len(est_probs)
        no_probs = sum(1 for p in est_probs if p < 0.5) / len(est_probs)
        flat = sum(1 for p in est_probs if p == 0.5) / len(est_probs)
        print(f"  est_prob_up > 0.5: {yes_probs:.1%}  (would pick YES)")
        print(f"  est_prob_up < 0.5: {no_probs:.1%}  (would pick NO)")
        print(f"  est_prob_up == 0.5: {flat:.1%}  (no signal)")

    sides = Counter(col("would_pick_side"))
    print(f"\n  would_pick_side counts: {dict(sides)}")

    enters = Counter(col("would_enter"))
    print(f"  would_enter (1=yes, 0=no): {dict(enters)}")

    trades = Counter(col("trade_placed"))
    print(f"  trade_placed (1=yes, 0=no): {dict(trades)}")

    # ── 5. When est_prob says NO, what blocks the trade? ─────────────
    print("\n" + "=" * 90)
    print("5. WHEN est_prob < 0.5 (signal says NO), WHAT IS BLOCKING THE TRADE?")
    print("=" * 90)
    cur.execute("""
        SELECT entry_reason, filter_blocked, risk_can_trade, risk_reason,
               COUNT(*) as n
        FROM signal_ticks
        WHERE est_prob_up < 0.5
        GROUP BY entry_reason, filter_blocked, risk_can_trade, risk_reason
        ORDER BY n DESC
        LIMIT 20
    """)
    grouped = cur.fetchall()
    if not grouped:
        print("  (no rows with est_prob_up < 0.5 — all ticks predicted YES)")
    else:
        print(f"  {'N':>6} {'risk_ok':>7} {'entry_reason':<50} {'filter_block':<25}")
        print(f"  {'-'*6} {'-'*7} {'-'*50} {'-'*25}")
        for r in grouped:
            entry_reason = (r[0] or "")[:50]
            filter_b = (r[1] or "")[:25]
            print(f"  {r[4]:>6,} {r[2]:>7} {entry_reason:<50} {filter_b:<25}")

    # ── 6. When est_prob says YES, what happens? ─────────────────────
    print("\n" + "=" * 90)
    print("6. WHEN est_prob > 0.5 (signal says YES), what happens?")
    print("=" * 90)
    cur.execute("""
        SELECT
            CASE
                WHEN trade_placed = 1 THEN '1. trade_placed'
                WHEN filter_blocked != '' THEN '2. filtered: ' || substr(filter_blocked, 0, 50)
                WHEN would_enter = 0 THEN '3. would_not_enter'
                WHEN risk_can_trade = 0 THEN '4. risk_blocked'
                ELSE '5. other'
            END as outcome,
            COUNT(*) as n
        FROM signal_ticks
        WHERE est_prob_up > 0.5
        GROUP BY outcome
        ORDER BY n DESC
    """)
    for row in cur.fetchall():
        print(f"  {row[0]:<60} {row[1]:>6,}")

    # ── 7. BTC trajectory (sanity) ───────────────────────────────────
    print("\n" + "=" * 90)
    print("7. BTC vs oracle relationship (sanity)")
    print("=" * 90)
    btc = col("btc_price")
    orc = col("oracle_price")
    if btc and orc and len(btc) == len(orc):
        deltas = [b - o for b, o in zip(btc, orc) if b > 0 and o > 0]
        print(stats(deltas, "BTC - oracle_NOW ($)", decimals=2))
        if deltas:
            n_above = sum(1 for d in deltas if d > 0)
            print(f"  BTC > oracle in {n_above:,}/{len(deltas):,} ticks "
                  f"({n_above/len(deltas):.1%})")
            print("  Expected: ~50% in noise. >90% or <10% = persistent feed offset.")

    conn.close()


if __name__ == "__main__":
    main()
