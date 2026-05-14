"""L3 historical audit — when did L3 fire and was it predictive?

We can't replay L3 with different thresholds on historical data because
the tick database doesn't store Coinalyze snapshots or Binance liquidation
stats. But Bot G's trade history DOES store the final L3 value
(liquidation_signal column) for every trade.

This script:
1. Distribution of L3 values across all Bot G trades
2. WR + PnL by L3 signal bucket — does L3 firing predict outcomes?
3. Conditional analysis: when L3 agrees with the direction of est_prob,
   does WR improve?
4. Timeline: when did L3 fire historically?

Output answers: is L3 worth tuning, or is it dead weight?
"""

import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime

DB = "data_runtime/bot_g_signal_aligned.db"


def fmt_pnl(x):
    return f"${x:+.2f}"


def fmt_pct(x):
    return f"{x:.1%}"


def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
        SELECT timestamp, side, entry_price, size_usdc, pnl,
               liquidation_signal, momentum_signal, combined_signal,
               estimated_prob_up, regime
        FROM trades
        WHERE pnl IS NOT NULL AND size_usdc > 0
        ORDER BY timestamp
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No trades found.")
        return

    print("=" * 85)
    print("L3 HISTORICAL AUDIT — Bot G")
    print(f"Total completed trades: {len(rows):,}")
    print("=" * 85)

    # ── 1. Distribution of L3 values ─────────────────────────────────

    l3_values = [r[5] for r in rows if r[5] is not None]
    nonzero = [v for v in l3_values if v != 0]
    zero = len(l3_values) - len(nonzero)

    print(f"\n--- L3 value distribution ---")
    print(f"  L3 == 0:     {zero:>5,} trades ({zero/len(l3_values):.1%})")
    print(f"  L3 != 0:     {len(nonzero):>5,} trades ({len(nonzero)/len(l3_values):.1%})")
    if nonzero:
        pos = sum(1 for v in nonzero if v > 0)
        neg = len(nonzero) - pos
        print(f"    L3 > 0:    {pos:>5,} trades ({pos/len(nonzero):.1%} of nonzero)")
        print(f"    L3 < 0:    {neg:>5,} trades ({neg/len(nonzero):.1%} of nonzero)")
        print(f"  Avg non-zero L3: {sum(nonzero)/len(nonzero):+.4f}")
        print(f"  Min L3: {min(nonzero):+.4f}")
        print(f"  Max L3: {max(nonzero):+.4f}")

    if not nonzero:
        print("\n*** L3 never fired historically. No more analysis possible.")
        return

    # ── 2. WR/PnL by L3 bucket ───────────────────────────────────────

    print(f"\n--- WR/PnL by L3 bucket ---")
    print(f"  {'Bucket':<20} {'N':>6} {'%':>6} {'WR':>7} "
          f"{'PnL':>10} {'Avg':>9}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*7} {'-'*10} {'-'*9}")

    buckets = [
        ("L3 = 0", lambda v: v == 0),
        ("L3 > 0.5 (strong+)", lambda v: v > 0.5),
        ("L3 in 0.2-0.5", lambda v: 0.2 <= v <= 0.5),
        ("L3 in 0-0.2", lambda v: 0 < v < 0.2),
        ("L3 in -0.2-0", lambda v: -0.2 < v < 0),
        ("L3 in -0.5--0.2", lambda v: -0.5 <= v <= -0.2),
        ("L3 < -0.5 (strong-)", lambda v: v < -0.5),
    ]

    for label, predicate in buckets:
        matching = [r for r in rows if r[5] is not None and predicate(r[5])]
        if not matching:
            continue
        n = len(matching)
        wins = sum(1 for r in matching if r[4] > 0)
        wr = wins / n
        pnl = sum(r[4] for r in matching)
        avg = pnl / n
        pct = n / len(rows)
        print(f"  {label:<20} {n:>6,} {pct:>5.1%} "
              f"{fmt_pct(wr):>7} {fmt_pnl(pnl):>10} {fmt_pnl(avg):>9}")

    # ── 3. L3 directional accuracy ───────────────────────────────────

    print(f"\n--- L3 directional accuracy (when L3 fires) ---")
    print("  Does L3's sign predict outcome?")
    nonzero_trades = [r for r in rows if r[5] is not None and r[5] != 0]

    # Resolution sign: YES win = bullish outcome (+1), NO win = bearish (-1)
    # but it depends on side. If side=YES and pnl>0, outcome was UP.
    # If side=NO and pnl>0, outcome was DOWN.
    correct_l3 = 0  # L3 sign matched eventual outcome direction
    wrong_l3 = 0
    for r in nonzero_trades:
        _, side, _, _, pnl, l3, _, _, _, _ = r
        if pnl is None:
            continue
        if side == "YES":
            outcome_up = pnl > 0  # YES won = UP happened
        else:
            outcome_up = pnl < 0  # NO lost = UP happened
        l3_says_up = l3 > 0
        if l3_says_up == outcome_up:
            correct_l3 += 1
        else:
            wrong_l3 += 1

    if correct_l3 + wrong_l3 > 0:
        print(f"  L3 sign predicted outcome correctly: "
              f"{correct_l3:,}/{correct_l3 + wrong_l3:,} "
              f"({correct_l3/(correct_l3 + wrong_l3):.1%})")
        print(f"  Random would be ~50%. >55% = L3 has predictive power.")

    # ── 4. L3 agreement with combined signal ─────────────────────────

    print(f"\n--- When L3 agrees with combined_signal direction ---")
    agree = []
    disagree = []
    for r in nonzero_trades:
        _, _, _, _, pnl, l3, _, combined, _, _ = r
        if combined is None or combined == 0 or pnl is None:
            continue
        if (l3 > 0) == (combined > 0):
            agree.append(r)
        else:
            disagree.append(r)

    if agree:
        a_pnl = sum(r[4] for r in agree)
        a_wins = sum(1 for r in agree if r[4] > 0)
        print(f"  Agree:    {len(agree):>5,} trades  WR {fmt_pct(a_wins/len(agree)):>6}  "
              f"PnL {fmt_pnl(a_pnl):>9}  Avg {fmt_pnl(a_pnl/len(agree)):>8}")
    if disagree:
        d_pnl = sum(r[4] for r in disagree)
        d_wins = sum(1 for r in disagree if r[4] > 0)
        print(f"  Disagree: {len(disagree):>5,} trades  WR {fmt_pct(d_wins/len(disagree)):>6}  "
              f"PnL {fmt_pnl(d_pnl):>9}  Avg {fmt_pnl(d_pnl/len(disagree)):>8}")

    # ── 5. Timeline of L3 firing ─────────────────────────────────────

    print(f"\n--- L3 firing timeline (daily) ---")
    by_day = defaultdict(lambda: {"n": 0, "n_fired": 0, "fired_pnl": 0.0})
    for r in rows:
        try:
            d = datetime.fromisoformat(r[0].replace("Z", "+00:00")).date()
        except Exception:
            continue
        by_day[d]["n"] += 1
        if r[5] is not None and r[5] != 0:
            by_day[d]["n_fired"] += 1
            by_day[d]["fired_pnl"] += r[4]

    print(f"  {'Date':<12} {'N':>5} {'L3 fired':>10} {'%fired':>8} {'fired PnL':>11}")
    for d in sorted(by_day.keys()):
        b = by_day[d]
        if b["n_fired"] == 0:
            continue
        pct = b["n_fired"] / b["n"]
        print(f"  {str(d):<12} {b['n']:>5} {b['n_fired']:>10} "
              f"{pct:>7.1%} {fmt_pnl(b['fired_pnl']):>11}")

    # ── 6. Overall comparison: fired vs didn't fire ──────────────────

    print(f"\n--- Overall: fired vs didn't fire ---")
    fired = [r for r in rows if r[5] is not None and r[5] != 0]
    not_fired = [r for r in rows if r[5] is not None and r[5] == 0]

    def summarise(label, ts):
        if not ts:
            return f"  {label}: (no trades)"
        n = len(ts)
        wins = sum(1 for r in ts if r[4] > 0)
        pnl = sum(r[4] for r in ts)
        return (f"  {label}: N={n:,}  WR {fmt_pct(wins/n)}  "
                f"PnL {fmt_pnl(pnl)}  Avg {fmt_pnl(pnl/n)}")

    print(summarise("L3 fired      ", fired))
    print(summarise("L3 did NOT fire", not_fired))


if __name__ == "__main__":
    main()
