"""Quick check of which signals are stuck at constant/zero values.

User reported L3, L6, L9b stuck at 0. Verify and investigate which other
signals might be broken.
"""

import os
import sqlite3
import statistics

DBS = [
    "data_runtime/bot_g_signal_aligned_signal_diag.db",
    "data_runtime/bot_k_sm_confirmation_signal_diag.db",
]

SIGNAL_COLS = [
    ("l1_oracle_lag", "L1 oracle_lag"),
    ("l2_momentum", "L2 momentum"),
    ("l3_liquidation", "L3 liquidation"),
    ("l4_orderbook", "L4 orderbook"),
    ("l5_sentiment", "L5 sentiment"),
    ("l6_fade", "L6 fade"),
    ("l7_taker_ratio", "L7 taker_ratio"),
    ("l8_clob_flow", "L8 clob_flow"),
    ("l9b_absorption", "L9b absorption"),
    ("l10_exhaustion", "L10 exhaustion"),
    ("l11_trade_size", "L11 trade_size"),
    ("l12_wallet_flow", "L12 wallet_flow"),
]


def analyze(db_path: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {db_path}")
    print(f"{'=' * 80}")

    if not os.path.exists(db_path):
        print(f"  (not found)")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM signal_ticks")
    n = cur.fetchone()[0]
    print(f"  {n:,} ticks\n")

    print(f"  {'Signal':<18} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} "
          f"{'distinct':>9} {'%nonzero':>9} {'status':<15}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*9} {'-'*9} {'-'*15}")

    for col, label in SIGNAL_COLS:
        cur.execute(f"""
            SELECT AVG({col}), MIN({col}), MAX({col}),
                   COUNT(DISTINCT {col}),
                   SUM(CASE WHEN {col} != 0 THEN 1 ELSE 0 END)
            FROM signal_ticks
        """)
        avg, mn, mx, distinct, nonzero = cur.fetchone()

        # Compute std via second pass (SQLite doesn't have stdev built in)
        cur.execute(f"SELECT {col} FROM signal_ticks")
        vals = [r[0] for r in cur.fetchall() if r[0] is not None]
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0

        nonzero_pct = nonzero / n if n else 0
        if distinct == 1:
            status = "STUCK at one value"
        elif distinct <= 5:
            status = f"Only {distinct} values"
        elif std < 0.001:
            status = "near-zero variance"
        elif nonzero_pct < 0.01:
            status = "mostly zero"
        else:
            status = "OK"

        print(f"  {label:<18} {avg or 0:>+10.4f} {std:>10.4f} "
              f"{mn or 0:>+10.4f} {mx or 0:>+10.4f} "
              f"{distinct:>9} {nonzero_pct:>8.1%} {status:<15}")

    conn.close()


for db in DBS:
    analyze(db)
