"""Snapshot-level signal audit — does the bot SEE NO opportunities?

Computes what Bot G and Bot K signals would say at signal-firing time
across EVERY window in the tick database (not just windows the bot
traded). This answers: is the bot picking 100% YES because no NO
signals exist (market regime) or because something is blocking NO
detection (code bug)?

For each window snapshot in the entry window:
1. Compute L1 (oracle lag), L2 (momentum), L4 (orderbook imbalance)
2. Combine with Bot G's ranging schedule and Bot K's optimised schedule
3. Determine est_prob_up under each
4. Tabulate the distribution

If recent windows show 50/50 est_prob distribution but the bot only
took YES trades, there's a code-level selection issue. If recent
windows show 99%+ above 0.5, the bias is structural to the market.

Usage:
    python -m backtest.snapshot_signal_audit
"""

import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "tick_data.db")

# Bot G ranging schedule (used when regime is ranging, the majority case)
# (oracle, momentum, liquidation, orderbook, sentiment) at 180s remaining
BOT_G_RANGING_180 = (0.05, 0.25, 0.05, 0.60, 0.05)

# Bot K optimised schedule at 180s
BOT_K_OPTIMISED_180 = (0.11, 0.38, 0.12, 0.24, 0.15)

# Entry timing window: bot enters at 30-270s remaining
ENTRY_SECS_MIN = 60   # 30s into window (60s remaining = end)
ENTRY_SECS_MAX = 270  # 30s into window (start of entry window)
# i.e., we sample windows where secs_remaining is 60-270


def compute_l1_oracle(btc_price: float, btc_open: float) -> float:
    """Oracle lag: clamp((btc - open) / open / 0.001, -1, 1)."""
    if btc_open <= 0 or btc_price <= 0:
        return 0.0
    pct = (btc_price - btc_open) / btc_open
    if abs(pct) < 0.0001:
        return 0.0
    raw = pct / 0.001
    return max(-1.0, min(1.0, raw))


def compute_l4_orderbook(yes_bid_depth: float, yes_ask_depth: float,
                          no_bid_depth: float, no_ask_depth: float) -> float:
    """Simplified L4: 4-level imbalance signal.
    YES bid depth + NO ask depth = bullish
    YES ask depth + NO bid depth = bearish
    """
    total = yes_bid_depth + yes_ask_depth + no_bid_depth + no_ask_depth
    if total < 1.0:
        return 0.0
    bull = yes_bid_depth + no_ask_depth
    bear = yes_ask_depth + no_bid_depth
    imb = (bull - bear) / total
    return max(-1.0, min(1.0, imb / 0.3))  # match orderbook_signal default norm


def compute_l2_momentum(spot_rows: list) -> float:
    """Simplified L2 from recent spot candles."""
    if len(spot_rows) < 3:
        return 0.0
    recent = spot_rows[-10:] if len(spot_rows) >= 10 else spot_rows
    # roc + direction + body ratio
    o, c, h, l = recent[-1][0], recent[-1][1], recent[-1][2], recent[-1][3]
    rng = h - l
    if rng < 1e-9:
        return 0.0
    direction = 1.0 if c > o else (-1.0 if c < o else 0.0)
    body_ratio = abs(c - o) / rng
    roc = 0.0
    if len(recent) >= 2:
        prev_c = recent[-2][1]
        if prev_c > 0:
            roc = (c - prev_c) / prev_c * 1000
    raw = 0.40 * roc + 0.30 * direction + 0.30 * body_ratio * direction
    return max(-1.0, min(1.0, raw))


def combine(weights: tuple, l1: float, l2: float, l3: float, l4: float, l5: float) -> float:
    """Weighted blend. Returns raw_signal in [-1, 1]."""
    w1, w2, w3, w4, w5 = weights
    return w1 * l1 + w2 * l2 + w3 * l3 + w4 * l4 + w5 * l5


def main() -> None:
    print("=" * 90)
    print("SNAPSHOT SIGNAL AUDIT")
    print("Computes what signals would say across ALL windows, not just bot trades")
    print("=" * 90)

    if not os.path.exists(DB_PATH):
        print(f"Tick DB not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Determine date range in DB
    cur.execute("SELECT MIN(start_time), MAX(start_time) FROM markets")
    min_t, max_t = cur.fetchone()
    print(f"\nTick DB date range: {min_t} to {max_t}")

    # Define two periods
    periods = [
        ("HISTORICAL (Apr 20-25)", "2026-04-20", "2026-04-26"),
        ("RECENT (May 9-13)", "2026-05-09", "2026-05-14"),
    ]

    # Load spot data once
    print("\nLoading spot trades for momentum computation...")
    cur.execute("""
        SELECT timestamp, price_open, price_close, price_high, price_low
        FROM spot_trades
        ORDER BY timestamp
    """)
    spot_rows = cur.fetchall()
    spot_by_ts = {}
    for row in spot_rows:
        if row[0]:
            spot_by_ts[row[0][:19]] = row[1:]
    print(f"  Loaded {len(spot_rows):,} spot candles")

    for label, start, end in periods:
        print(f"\n{'=' * 90}")
        print(f"{label}: {start} to {end}")
        print(f"{'=' * 90}")

        # Load markets in this period
        cur.execute("""
            SELECT market_id, start_time, btc_price_start, btc_price_end, winner
            FROM markets
            WHERE start_time >= ? AND start_time < ?
              AND winner IN ('Up', 'Down')
            ORDER BY start_time
        """, (start, end))
        markets = cur.fetchall()
        print(f"\n  Markets in period: {len(markets):,}")

        if not markets:
            print("  No markets — skipping.")
            continue

        # Take one snapshot per market at ~entry time (60s into window)
        # We pick the snapshot with secs_into_window closest to 60
        market_ids = [m[0] for m in markets]
        market_open_btc = {str(m[0]): m[2] for m in markets}
        market_winner = {str(m[0]): m[4] for m in markets}

        # Collect one signal-firing snapshot per market
        snapshot_data = []
        batch_size = 500
        for batch_start in range(0, len(market_ids), batch_size):
            batch = market_ids[batch_start:batch_start + batch_size]
            placeholders = ",".join("?" * len(batch))
            cur.execute(f"""
                SELECT market_id, seconds_into_window, btc_price,
                       up_best_bid, up_best_ask, down_best_bid, down_best_ask,
                       up_bid_depth_5, up_ask_depth_5,
                       down_bid_depth_5, down_ask_depth_5, time
                FROM snapshots
                WHERE market_id IN ({placeholders})
                  AND seconds_into_window BETWEEN 30 AND 240
                ORDER BY market_id, seconds_into_window
            """, batch)
            rows = cur.fetchall()
            # Take the first snapshot per market (around 60s into window)
            seen = set()
            for r in rows:
                mid = str(r[0])
                if mid in seen:
                    continue
                seen.add(mid)
                snapshot_data.append(r)

        print(f"  Snapshots collected (one per market): {len(snapshot_data):,}")

        if not snapshot_data:
            continue

        # Compute signals for each
        bot_g_combined = []
        bot_k_combined = []
        l1_values = []
        l2_values = []
        l4_values = []
        btc_vs_open = []
        side_bot_g = []
        side_bot_k = []

        for snap in snapshot_data:
            (mid, sec_into, btc_price, y_bid, y_ask, n_bid, n_ask,
             y_bid_depth, y_ask_depth, n_bid_depth, n_ask_depth, snap_time) = snap

            btc_price = btc_price or 0
            btc_open = market_open_btc.get(str(mid), 0) or 0

            l1 = compute_l1_oracle(btc_price, btc_open)
            l4 = compute_l4_orderbook(
                y_bid_depth or 0, y_ask_depth or 0,
                n_bid_depth or 0, n_ask_depth or 0
            )

            # L2 momentum from spot trades around snapshot time
            mom_rows = []
            if snap_time:
                try:
                    t_sec = datetime.fromisoformat(snap_time[:19])
                    for offset_s in range(-10, 1):
                        t_key = (t_sec + timedelta(seconds=offset_s)).strftime("%Y-%m-%dT%H:%M:%S")
                        if t_key in spot_by_ts:
                            mom_rows.append(spot_by_ts[t_key])
                except Exception:
                    pass
            l2 = compute_l2_momentum(mom_rows)

            l1_values.append(l1)
            l2_values.append(l2)
            l4_values.append(l4)
            btc_vs_open.append(btc_price - btc_open)

            combined_g = combine(BOT_G_RANGING_180, l1, l2, 0.0, l4, 0.0)
            combined_k = combine(BOT_K_OPTIMISED_180, l1, l2, 0.0, l4, 0.0)
            bot_g_combined.append(combined_g)
            bot_k_combined.append(combined_k)

            # est_prob_up > 0.5 -> YES side, < 0.5 -> NO side
            side_bot_g.append("YES" if combined_g > 0 else ("NO" if combined_g < 0 else "FLAT"))
            side_bot_k.append("YES" if combined_k > 0 else ("NO" if combined_k < 0 else "FLAT"))

        # Distribution stats
        def pct(arr, p):
            arr_sorted = sorted(arr)
            return arr_sorted[int(len(arr_sorted) * p)] if arr_sorted else 0

        def summary(name: str, values: list) -> str:
            if not values:
                return f"  {name}: (no data)"
            avg = sum(values) / len(values)
            return (f"  {name}: avg {avg:+.4f}  "
                    f"p10 {pct(values, 0.10):+.4f}  "
                    f"p50 {pct(values, 0.50):+.4f}  "
                    f"p90 {pct(values, 0.90):+.4f}  "
                    f"pct_pos {sum(1 for x in values if x > 0)/len(values):.1%}")

        print(f"\n--- Signal value distributions ---")
        print(summary("L1 oracle", l1_values))
        print(summary("L2 momentum", l2_values))
        print(summary("L4 orderbook", l4_values))
        print(summary("BTC - open ($)", btc_vs_open))

        print(f"\n--- Bot G ranging schedule combined_signal ---")
        print(summary("Bot G combined", bot_g_combined))
        g_yes = sum(1 for s in side_bot_g if s == "YES")
        g_no = sum(1 for s in side_bot_g if s == "NO")
        print(f"  Side breakdown: YES={g_yes:,} ({g_yes/len(side_bot_g):.1%})  "
              f"NO={g_no:,} ({g_no/len(side_bot_g):.1%})")

        print(f"\n--- Bot K optimised schedule combined_signal ---")
        print(summary("Bot K combined", bot_k_combined))
        k_yes = sum(1 for s in side_bot_k if s == "YES")
        k_no = sum(1 for s in side_bot_k if s == "NO")
        print(f"  Side breakdown: YES={k_yes:,} ({k_yes/len(side_bot_k):.1%})  "
              f"NO={k_no:,} ({k_no/len(side_bot_k):.1%})")

        # Histogram of combined_signal
        print(f"\n--- Bot G combined_signal histogram ---")
        bins = [-1.0, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
        counts = [0] * (len(bins) - 1)
        for v in bot_g_combined:
            for i in range(len(bins) - 1):
                if bins[i] <= v < bins[i + 1]:
                    counts[i] += 1
                    break
        for i in range(len(bins) - 1):
            if counts[i] > 0:
                pct_v = counts[i] / len(bot_g_combined) * 100
                bar = "#" * int(pct_v / 2)
                print(f"    {bins[i]:>+5.2f} to {bins[i+1]:>+5.2f}: "
                      f"{counts[i]:>5,} ({pct_v:>5.1f}%) {bar}")

    conn.close()


if __name__ == "__main__":
    main()
