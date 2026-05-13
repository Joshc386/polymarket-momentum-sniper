"""Investigate the YES/NO side bias issue.

User report: both Bot G and Bot K are trading exclusively YES contracts.
Historically it was more balanced and NO produced most of the profit.

This script:
1. Verifies the bias empirically from both bot DBs
2. Plots side distribution over time
3. Identifies when the bias started
4. Cross-tabs with signal values to find the cause
"""

import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta

BOT_G_DB = "data_runtime/bot_g_signal_aligned.db"
BOT_K_DB = "data_runtime/bot_k_sm_confirmation.db"


def load_trades(db_path: str, bot_label: str):
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        print(f"  Could not open {db_path}: {e}")
        return []
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT timestamp, side, entry_price, pnl,
                   estimated_prob_up, market_implied_prob, edge,
                   oracle_lag_signal, momentum_signal, liquidation_signal,
                   orderbook_signal, sentiment_signal, combined_signal,
                   regime, time_remaining_secs
            FROM trades
            WHERE pnl IS NOT NULL AND size_usdc > 0
            ORDER BY timestamp
        """)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    except sqlite3.Error as e:
        print(f"  Error reading {db_path}: {e}")
        return []
    conn.close()
    print(f"  {bot_label}: {len(rows):,} completed trades")
    return rows


def date_of(t):
    try:
        return datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00")).date()
    except Exception:
        return None


def main():
    print("=" * 80)
    print("SIDE BIAS INVESTIGATION")
    print("=" * 80)

    print("\nLoading trades...")
    g_trades = load_trades(BOT_G_DB, "Bot G")
    k_trades = load_trades(BOT_K_DB, "Bot K")

    # ── Overall side distribution per bot ────────────────────────────

    for bot_label, trades in [("Bot G", g_trades), ("Bot K", k_trades)]:
        if not trades:
            continue
        yes = [t for t in trades if t["side"] == "YES"]
        no = [t for t in trades if t["side"] == "NO"]
        print(f"\n--- {bot_label} overall (n={len(trades):,}) ---")
        print(f"  YES: {len(yes):,} ({len(yes)/len(trades):.1%}) "
              f"PnL ${sum(t['pnl'] for t in yes):+.2f}")
        print(f"  NO:  {len(no):,} ({len(no)/len(trades):.1%}) "
              f"PnL ${sum(t['pnl'] for t in no):+.2f}")

    # ── Side distribution per day (Bot G) ────────────────────────────

    print(f"\n{'=' * 80}")
    print("BOT G: side distribution by DAY (last 30 days)")
    print(f"{'=' * 80}")

    if g_trades:
        by_day = defaultdict(lambda: {"YES": 0, "NO": 0, "yes_pnl": 0.0, "no_pnl": 0.0})
        for t in g_trades:
            d = date_of(t)
            if d:
                by_day[d][t["side"]] += 1
                if t["side"] == "YES":
                    by_day[d]["yes_pnl"] += t["pnl"]
                else:
                    by_day[d]["no_pnl"] += t["pnl"]

        days = sorted(by_day.keys())[-30:]
        print(f"\n  {'Date':<12} {'YES':>5} {'NO':>5} {'YES%':>6} "
              f"{'YES PnL':>9} {'NO PnL':>9}")
        for d in days:
            row = by_day[d]
            total = row["YES"] + row["NO"]
            if total == 0:
                continue
            yes_pct = row["YES"] / total
            print(f"  {str(d):<12} {row['YES']:>5,} {row['NO']:>5,} "
                  f"{yes_pct:>5.1%} ${row['yes_pnl']:>+8.2f} "
                  f"${row['no_pnl']:>+8.2f}")

    # ── Side distribution per day (Bot K) ────────────────────────────

    if k_trades:
        print(f"\n{'=' * 80}")
        print("BOT K: side distribution by DAY")
        print(f"{'=' * 80}")
        by_day = defaultdict(lambda: {"YES": 0, "NO": 0, "yes_pnl": 0.0, "no_pnl": 0.0})
        for t in k_trades:
            d = date_of(t)
            if d:
                by_day[d][t["side"]] += 1
                if t["side"] == "YES":
                    by_day[d]["yes_pnl"] += t["pnl"]
                else:
                    by_day[d]["no_pnl"] += t["pnl"]

        days = sorted(by_day.keys())
        print(f"\n  {'Date':<12} {'YES':>5} {'NO':>5} {'YES%':>6} "
              f"{'YES PnL':>9} {'NO PnL':>9}")
        for d in days:
            row = by_day[d]
            total = row["YES"] + row["NO"]
            if total == 0:
                continue
            yes_pct = row["YES"] / total
            print(f"  {str(d):<12} {row['YES']:>5,} {row['NO']:>5,} "
                  f"{yes_pct:>5.1%} ${row['yes_pnl']:>+8.2f} "
                  f"${row['no_pnl']:>+8.2f}")

    # ── Recent trades signal investigation (Bot G) ───────────────────

    print(f"\n{'=' * 80}")
    print("BOT G: signal values on RECENT trades (last 50)")
    print(f"{'=' * 80}")

    if g_trades:
        recent = g_trades[-50:]
        yes_recent = [t for t in recent if t["side"] == "YES"]
        no_recent = [t for t in recent if t["side"] == "NO"]

        def avg(ts, key):
            vals = [t[key] for t in ts if t[key] is not None]
            return sum(vals) / len(vals) if vals else 0.0

        print(f"\n  Last 50 Bot G trades: {len(yes_recent)} YES, {len(no_recent)} NO")
        print(f"\n  {'Signal':<22} {'YES (recent)':>14} {'NO (recent)':>14} "
              f"{'All-time YES':>14} {'All-time NO':>14}")
        print("  " + "-" * 80)

        all_yes = [t for t in g_trades if t["side"] == "YES"]
        all_no = [t for t in g_trades if t["side"] == "NO"]

        signals = [
            ("est_prob_up", "estimated_prob_up"),
            ("market_implied_p", "market_implied_prob"),
            ("combined_signal", "combined_signal"),
            ("L1 oracle", "oracle_lag_signal"),
            ("L2 momentum", "momentum_signal"),
            ("L3 liquidation", "liquidation_signal"),
            ("L4 orderbook", "orderbook_signal"),
            ("L5 sentiment", "sentiment_signal"),
            ("edge (best_ev)", "edge"),
        ]
        for lbl, key in signals:
            yr = avg(yes_recent, key)
            nr = avg(no_recent, key)
            ya = avg(all_yes, key)
            na = avg(all_no, key)
            print(f"  {lbl:<22} {yr:>+14.4f} {nr:>+14.4f} "
                  f"{ya:>+14.4f} {na:>+14.4f}")

    # ── Last 20 trades verbatim ──────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("BOT G: LAST 20 TRADES (verbatim)")
    print(f"{'=' * 80}")
    if g_trades:
        print(f"\n  {'Time':<20} {'Side':<5} {'Price':>7} {'est_P':>6} "
              f"{'impl_P':>7} {'edge':>7} {'combined':>9} {'regime':<13}")
        for t in g_trades[-20:]:
            ts = t["timestamp"][:19] if t["timestamp"] else "?"
            print(f"  {ts:<20} {t['side']:<5} ${t['entry_price']:>6.3f} "
                  f"{t.get('estimated_prob_up', 0) or 0:>5.3f} "
                  f"{t.get('market_implied_prob', 0) or 0:>6.3f} "
                  f"{t.get('edge', 0) or 0:>+6.3f} "
                  f"{t.get('combined_signal', 0) or 0:>+8.4f} "
                  f"{(t.get('regime') or '?'):<13}")

    print(f"\n{'=' * 80}")
    print("BOT K: LAST 20 TRADES (verbatim)")
    print(f"{'=' * 80}")
    if k_trades:
        print(f"\n  {'Time':<20} {'Side':<5} {'Price':>7} {'est_P':>6} "
              f"{'impl_P':>7} {'edge':>7} {'combined':>9} {'regime':<13}")
        for t in k_trades[-20:]:
            ts = t["timestamp"][:19] if t["timestamp"] else "?"
            print(f"  {ts:<20} {t['side']:<5} ${t['entry_price']:>6.3f} "
                  f"{t.get('estimated_prob_up', 0) or 0:>5.3f} "
                  f"{t.get('market_implied_prob', 0) or 0:>6.3f} "
                  f"{t.get('edge', 0) or 0:>+6.3f} "
                  f"{t.get('combined_signal', 0) or 0:>+8.4f} "
                  f"{(t.get('regime') or '?'):<13}")


if __name__ == "__main__":
    main()
