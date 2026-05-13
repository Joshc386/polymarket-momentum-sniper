"""Check if oracle_window_open_price is being computed correctly.

If oracle_open is stale or wrong (e.g., always reflects last window's
close), L1 would be systematically biased. Test by checking:
1. How does oracle_open change between consecutive trades in the same
   window vs different windows?
2. Is btc_price_at_entry / oracle_price_at_entry near oracle_open
   (which means we just opened) or far from it (window has progressed)?
"""

import sqlite3
from datetime import datetime

DB = "data_runtime/bot_g_signal_aligned.db"

conn = sqlite3.connect(DB)
cur = conn.cursor()

# Look at recent trades in detail
print("=" * 90)
print("RECENT TRADES — Bot G — oracle/BTC relationship")
print("=" * 90)
cur.execute("""
    SELECT timestamp, market_slug, side, entry_price,
           btc_price_at_entry, oracle_price_at_entry, oracle_price_at_open,
           time_remaining_secs, oracle_lag_signal, regime
    FROM trades
    WHERE pnl IS NOT NULL
    ORDER BY timestamp DESC
    LIMIT 30
""")
rows = cur.fetchall()
rows.reverse()

print(f"\n  {'Time':<19} {'Window':<10} {'Side':<4} "
      f"{'btc_entry':>10} {'oracle_open':>11} {'delta':>8} "
      f"{'tRem':>5} {'L1':>8} {'regime':<13}")
for r in rows:
    ts, slug, side, eprice, btc, oracle_at_entry, oracle_open, trem, l1, regime = r
    ts_short = ts[:19] if ts else "?"
    # Strip prefix from slug
    slug_short = slug[-10:] if slug else "?"
    delta = (btc or 0) - (oracle_open or 0)
    print(f"  {ts_short:<19} {slug_short:<10} {side:<4} "
          f"${btc or 0:>9,.0f} ${oracle_open or 0:>10,.0f} "
          f"${delta:>+7.0f} {trem or 0:>5.0f} {l1 or 0:>+8.4f} "
          f"{regime or '?':<13}")

# Check: how many distinct windows in the last 30 trades?
windows = set()
for r in rows:
    if r[1]:
        windows.add(r[1])
print(f"\n  Distinct windows in last 30 trades: {len(windows)}")

# Check: is BTC consistently above oracle_open?
above_open = sum(1 for r in rows if (r[4] or 0) > (r[6] or 0))
print(f"  BTC > oracle_open: {above_open}/{len(rows)} ({above_open/len(rows):.1%})")

# Compare to the OLD data (April)
print("\n" + "=" * 90)
print("APRIL TRADES — same query for historical comparison")
print("=" * 90)
cur.execute("""
    SELECT timestamp, side, btc_price_at_entry, oracle_price_at_open,
           oracle_lag_signal, regime
    FROM trades
    WHERE pnl IS NOT NULL
      AND timestamp >= '2026-04-20'
      AND timestamp < '2026-04-21'
    ORDER BY timestamp
""")
old_rows = cur.fetchall()
above_open_old = sum(1 for r in old_rows if (r[2] or 0) > (r[3] or 0))
sides = {"YES": 0, "NO": 0}
for r in old_rows:
    sides[r[1]] = sides.get(r[1], 0) + 1

print(f"\n  April 20: {len(old_rows)} trades")
print(f"  YES: {sides.get('YES', 0)} | NO: {sides.get('NO', 0)}")
print(f"  BTC > oracle_open: {above_open_old}/{len(old_rows)} "
      f"({above_open_old/len(old_rows):.1%} of trades)")

# Histogram of BTC-oracle deltas for both periods
def delta_histogram(label: str, query: str):
    cur.execute(query)
    rs = cur.fetchall()
    deltas = [(r[0] or 0) - (r[1] or 0) for r in rs if r[0] and r[1]]
    if not deltas:
        return
    print(f"\n  {label} — BTC minus oracle_open distribution:")
    bins = [-1000, -100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 1000]
    counts = [0] * (len(bins) - 1)
    for d in deltas:
        for i in range(len(bins) - 1):
            if bins[i] <= d < bins[i + 1]:
                counts[i] += 1
                break
    for i in range(len(bins) - 1):
        if counts[i] > 0:
            pct = counts[i] / len(deltas) * 100
            bar = "█" * int(pct / 2)
            print(f"    ${bins[i]:>+5} to ${bins[i+1]:>+5}: {counts[i]:>4} ({pct:>5.1f}%) {bar}")

delta_histogram(
    "RECENT (May 11-13)",
    """SELECT btc_price_at_entry, oracle_price_at_open FROM trades
       WHERE timestamp >= '2026-05-11' AND pnl IS NOT NULL"""
)
delta_histogram(
    "HISTORICAL (April)",
    """SELECT btc_price_at_entry, oracle_price_at_open FROM trades
       WHERE timestamp >= '2026-04-14' AND timestamp < '2026-05-01'
       AND pnl IS NOT NULL"""
)

conn.close()
