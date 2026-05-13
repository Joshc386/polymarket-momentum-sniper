"""Check BTC price trajectory in recent vs historical Bot G trades.

If BTC has been in a sustained bull run since May 11, L1 oracle lag
would be persistently positive (driving YES bias). That'd be a market
condition, not a bug. Check by computing (btc_at_entry - oracle_open)
for recent vs historical trades.
"""

import sqlite3
from datetime import datetime
from collections import defaultdict

DB = "data_runtime/bot_g_signal_aligned.db"

conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute("""
    SELECT timestamp, side, btc_price_at_entry, oracle_price_at_open,
           oracle_lag_signal, orderbook_signal, combined_signal
    FROM trades
    WHERE pnl IS NOT NULL
      AND btc_price_at_entry IS NOT NULL
      AND oracle_price_at_open IS NOT NULL
    ORDER BY timestamp
""")
rows = cur.fetchall()
conn.close()

print(f"Loaded {len(rows):,} trades with BTC price data\n")


# Per-day stats: average BTC - oracle_open delta, L1 signal, L4 signal
by_day = defaultdict(lambda: {
    "n": 0, "yes": 0, "no": 0,
    "delta_sum": 0.0, "abs_delta_sum": 0.0,
    "btc_above_open": 0,
    "l1_sum": 0.0, "l4_sum": 0.0, "combined_sum": 0.0,
})

for ts, side, btc, oracle_open, l1, l4, combined in rows:
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
    except Exception:
        continue
    delta = btc - oracle_open
    b = by_day[d]
    b["n"] += 1
    if side == "YES":
        b["yes"] += 1
    else:
        b["no"] += 1
    b["delta_sum"] += delta
    b["abs_delta_sum"] += abs(delta)
    if btc > oracle_open:
        b["btc_above_open"] += 1
    b["l1_sum"] += l1 or 0.0
    b["l4_sum"] += l4 or 0.0
    b["combined_sum"] += combined or 0.0

print(f"  {'Date':<12} {'N':>4} {'YES%':>6} "
      f"{'AvgDelta':>10} {'BTC>Open%':>10} "
      f"{'AvgL1':>8} {'AvgL4':>8} {'AvgCombined':>12}")
print("  " + "-" * 78)
for d in sorted(by_day.keys()):
    b = by_day[d]
    n = b["n"]
    if n == 0:
        continue
    yes_pct = b["yes"] / n
    avg_delta = b["delta_sum"] / n
    btc_above_pct = b["btc_above_open"] / n
    avg_l1 = b["l1_sum"] / n
    avg_l4 = b["l4_sum"] / n
    avg_combined = b["combined_sum"] / n
    print(f"  {str(d):<12} {n:>4} {yes_pct:>5.1%} "
          f"${avg_delta:>+9.2f} {btc_above_pct:>9.1%} "
          f"{avg_l1:>+7.3f} {avg_l4:>+7.3f} {avg_combined:>+11.3f}")

# Also: what was the BTC price range over these dates?
print(f"\n--- BTC price range over time ---")
by_date_btc = defaultdict(list)
for ts, side, btc, oracle_open, l1, l4, combined in rows:
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
    except Exception:
        continue
    by_date_btc[d].append(btc)

print(f"  {'Date':<12} {'BTC min':>10} {'BTC avg':>10} {'BTC max':>10}")
for d in sorted(by_date_btc.keys()):
    prices = by_date_btc[d]
    print(f"  {str(d):<12} ${min(prices):>9,.0f} ${sum(prices)/len(prices):>9,.0f} ${max(prices):>9,.0f}")
