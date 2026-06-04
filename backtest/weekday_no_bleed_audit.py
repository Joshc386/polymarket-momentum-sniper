"""
Weekday NO bleed audit — split by L1 magnitude and J13 deployment date.

Goal: locate WHERE the -$210 weekday NO bleed is concentrated, so we
know whether it's a J13 residual (no action), a J13 hole (gate needs
fixing), or a hidden cell in mild-L1 weekday NO.

J13 deployment: commit 9ef0523, 2026-05-21 21:37 +0100 = 2026-05-21
20:37 UTC. Conservative cutoff for "post-J13" = 2026-05-22 00:00 UTC
(allows for bot restart latency).

Read-only.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRADES_DB = REPO / "data_runtime" / "bot_k_sm_confirmation.db"

J13_CUTOFF_UTC = pd.Timestamp("2026-05-22 00:00:00", tz="UTC")


def load_trades() -> pd.DataFrame:
    con = sqlite3.connect(TRADES_DB)
    df = pd.read_sql("""
        SELECT id, timestamp, side, pnl, entry_price,
               oracle_lag_signal, momentum_signal, combined_signal,
               estimated_prob_up, market_implied_prob, edge,
               resolution, ob_ask_depth
        FROM trades
        WHERE pnl IS NOT NULL AND oracle_lag_signal IS NOT NULL
    """, con)
    con.close()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["weekend"] = df["ts"].dt.dayofweek >= 5
    df["mild_l1"] = df["oracle_lag_signal"].abs() <= 0.4
    df["post_j13"] = df["ts"] >= J13_CUTOFF_UTC
    df["win"] = (df["pnl"] > 0).astype(int)
    return df


def cell(df: pd.DataFrame, **filters) -> dict:
    sub = df.copy()
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    n = len(sub)
    if n == 0:
        return {"n": 0, "WR_%": None, "pnl": 0.0, "pnl_per_day": 0.0}
    days = max(1, (sub["ts"].max() - sub["ts"].min()).days + 1)
    return {
        "n": n,
        "WR_%": round(100 * sub["win"].mean(), 1),
        "pnl": round(sub["pnl"].sum(), 2),
        "pnl_per_day": round(sub["pnl"].sum() / days, 2),
        "days_covered": days,
    }


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    df = load_trades()
    print()
    print(f"Date range          : {df['ts'].min()} to {df['ts'].max()}")
    print(f"J13 cutoff (UTC)    : {J13_CUTOFF_UTC}")
    print(f"Pre-J13 trades      : {(~df['post_j13']).sum()}")
    print(f"Post-J13 trades     : {df['post_j13'].sum()}")
    print()

    section("WEEKDAY NO -- 4-cell cross-tab (L1 magnitude × J13 status)")
    rows = []
    for mild in [True, False]:
        for post in [False, True]:
            r = cell(df, side="NO", weekend=False, mild_l1=mild, post_j13=post)
            r["L1"] = "mild (|L1|<=0.4)" if mild else "strong (|L1|>0.4)"
            r["era"] = "post-J13" if post else "pre-J13"
            rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    section("WEEKDAY NO -- aggregated by L1 magnitude (both eras)")
    rows = []
    for mild in [True, False]:
        r = cell(df, side="NO", weekend=False, mild_l1=mild)
        r["L1"] = "mild (|L1|<=0.4)" if mild else "strong (|L1|>0.4)"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    section("WEEKDAY NO -- aggregated by J13 era (both L1 mags)")
    rows = []
    for post in [False, True]:
        r = cell(df, side="NO", weekend=False, post_j13=post)
        r["era"] = "post-J13" if post else "pre-J13"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    section("SANITY: all sides × weekend × J13 era")
    rows = []
    for side in ["YES", "NO"]:
        for weekend in [False, True]:
            for post in [False, True]:
                r = cell(df, side=side, weekend=weekend, post_j13=post)
                r["side"] = side
                r["weekend"] = weekend
                r["era"] = "post-J13" if post else "pre-J13"
                rows.append(r)
    out = pd.DataFrame(rows)
    # Reorder columns for readability
    cols_order = ["side", "weekend", "era", "n", "WR_%", "pnl",
                  "pnl_per_day", "days_covered"]
    print(out[cols_order].to_string(index=False))
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
