"""Check for systematic offset between exchange price (Binance) and oracle price.

The L1 oracle lag signal has two parts:
  60% on (binance - oracle_NOW) / oracle_NOW
  40% on (binance - oracle_OPEN) / oracle_OPEN

If binance is systematically HIGHER than oracle (e.g., because oracle =
0.5*binance + 0.5*coinbase, and coinbase is lagging binance), then the
'lag_signal' component would be persistently positive even when there's
no real price move within the window.

Test: compute (btc_price_at_entry - oracle_price_at_entry) for recent
trades and see if it's systematically positive.
"""

import sqlite3
from datetime import datetime

DB = "data_runtime/bot_g_signal_aligned.db"

conn = sqlite3.connect(DB)
cur = conn.cursor()

# Periods to compare
periods = [
    ("HISTORICAL (Apr 20-25)", "2026-04-20", "2026-04-26"),
    ("RECENT (May 11-13)", "2026-05-11", "2026-05-14"),
]

for label, start, end in periods:
    cur.execute("""
        SELECT btc_price_at_entry, oracle_price_at_entry, oracle_price_at_open,
               oracle_lag_signal, side
        FROM trades
        WHERE timestamp >= ? AND timestamp < ?
          AND pnl IS NOT NULL
          AND btc_price_at_entry IS NOT NULL
          AND oracle_price_at_entry IS NOT NULL
        ORDER BY timestamp
    """, (start, end))
    rows = cur.fetchall()

    if not rows:
        print(f"\n{label}: no data")
        continue

    print(f"\n{'=' * 80}")
    print(f"{label}: {len(rows):,} trades")
    print(f"{'=' * 80}")

    # btc - oracle_at_entry (the "lag" component, 60% weight in L1)
    btc_vs_oracle_now = [r[0] - r[1] for r in rows if r[0] and r[1]]
    btc_vs_oracle_open = [r[0] - r[2] for r in rows if r[0] and r[2]]
    btc_vs_oracle_now_pct = [(r[0] - r[1]) / r[1] * 100 for r in rows if r[0] and r[1] and r[1] > 0]

    def stats(name, arr):
        if not arr:
            return
        s = sorted(arr)
        n = len(s)
        avg = sum(s) / n
        median = s[n // 2]
        p10 = s[int(n * 0.10)]
        p90 = s[int(n * 0.90)]
        pos = sum(1 for x in s if x > 0) / n
        print(f"  {name}:")
        print(f"    avg  {avg:+.3f}   median {median:+.3f}")
        print(f"    p10  {p10:+.3f}   p90    {p90:+.3f}")
        print(f"    fraction positive: {pos:.1%}")

    stats("BTC - oracle_NOW ($)", btc_vs_oracle_now)
    stats("BTC - oracle_NOW (% of price)", btc_vs_oracle_now_pct)
    stats("BTC - oracle_OPEN ($)", btc_vs_oracle_open)

    # Verify recorded L1 against what we'd compute
    # L1 = clamp(0.6 * clamp((btc-oracle_now)/oracle_now / 0.001, -1, 1)
    #          + 0.4 * clamp((btc-oracle_open)/oracle_open / 0.001, -1, 1),
    #          -1, 1)
    def compute_l1(btc, o_now, o_open):
        def clamp(x):
            return max(-1.0, min(1.0, x))
        if o_now <= 0 or btc <= 0:
            return 0.0
        lag = (btc - o_now) / o_now
        lag_sig = clamp(lag / 0.001)
        if o_open > 0:
            open_d = (btc - o_open) / o_open
            open_sig = clamp(open_d / 0.001)
            return clamp(0.6 * lag_sig + 0.4 * open_sig)
        return lag_sig

    recorded_l1 = [r[3] for r in rows if r[3] is not None]
    computed_l1 = [
        compute_l1(r[0], r[1], r[2])
        for r in rows
        if r[0] and r[1]
    ]

    print(f"\n  L1 signal — recorded vs recomputed:")
    if recorded_l1:
        print(f"    Recorded avg:   {sum(recorded_l1)/len(recorded_l1):+.4f}")
    if computed_l1:
        print(f"    Recomputed avg: {sum(computed_l1)/len(computed_l1):+.4f}")
        pos_pct = sum(1 for x in computed_l1 if x > 0) / len(computed_l1)
        print(f"    Recomputed % positive: {pos_pct:.1%}")

conn.close()
