"""
Locate Bot K's -$36/day post-J13 degradation.

Headline: Bot K pre-J13 +$58.43/day, post-J13 +$47.26/day. Difference
-$11.17/day. Weekday NO improved by +$69.41/day, so other categories
must have degraded by ~$80/day combined.

Decompose by side × weekend (and L1 magnitude where useful) to find
which cell(s) account for the degradation.

Compare to Bot G as control (Bot G has J13 OFF; any degradation
present on Bot G too is regime, not J13-induced).

Read-only.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DB_BOT_K = REPO / "data_runtime" / "bot_k_sm_confirmation.db"
DB_BOT_G = REPO / "data_runtime" / "bot_g_signal_aligned.db"

J13_CUTOFF_UTC = pd.Timestamp("2026-05-22 00:00:00", tz="UTC")


def load(db_path: Path, start: pd.Timestamp | None = None,
         end: pd.Timestamp | None = None) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT id, timestamp, side, pnl, oracle_lag_signal
        FROM trades WHERE pnl IS NOT NULL AND oracle_lag_signal IS NOT NULL
    """, con)
    con.close()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    if start is not None:
        df = df[df["ts"] >= start]
    if end is not None:
        df = df[df["ts"] <= end]
    df["weekend"] = df["ts"].dt.dayofweek >= 5
    df["mild_l1"] = df["oracle_lag_signal"].abs() <= 0.4
    df["post_j13"] = df["ts"] >= J13_CUTOFF_UTC
    df["win"] = (df["pnl"] > 0).astype(int)
    return df


def cell(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {"n": 0, "WR_%": None, "pnl": 0.0, "pnl_per_day": 0.0}
    days = max(1, (sub["ts"].max() - sub["ts"].min()).days + 1)
    return {
        "n": n,
        "WR_%": round(100 * sub["win"].mean(), 1),
        "pnl": round(sub["pnl"].sum(), 2),
        "pnl_per_day": round(sub["pnl"].sum() / days, 2),
        "days": days,
    }


def decompose(bot_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Pre vs post-J13 PnL/day by side × weekend."""
    rows = []
    for side in ["YES", "NO"]:
        for weekend in [False, True]:
            for era_name, mask in [("pre-J13", ~df["post_j13"]),
                                    ("post-J13", df["post_j13"])]:
                sub = df[(df["side"] == side) & (df["weekend"] == weekend) & mask]
                rows.append({
                    "bot": bot_name, "side": side,
                    "weekend": "wknd" if weekend else "wkdy",
                    "era": era_name, **cell(sub),
                })
    return pd.DataFrame(rows)


def delta_table(decomp: pd.DataFrame) -> pd.DataFrame:
    """Pivot to show pre vs post side-by-side with delta."""
    pre = decomp[decomp["era"] == "pre-J13"][["bot", "side", "weekend",
                                                "n", "WR_%", "pnl_per_day"]]
    post = decomp[decomp["era"] == "post-J13"][["bot", "side", "weekend",
                                                  "n", "WR_%", "pnl_per_day"]]
    m = pre.merge(post, on=["bot", "side", "weekend"],
                  suffixes=("_pre", "_post"))
    m["delta_pnl_day"] = (m["pnl_per_day_post"] - m["pnl_per_day_pre"]).round(2)
    m["delta_WR"] = (m["WR_%_post"] - m["WR_%_pre"]).round(1)
    return m[["bot", "side", "weekend", "n_pre", "WR_%_pre",
              "pnl_per_day_pre", "n_post", "WR_%_post", "pnl_per_day_post",
              "delta_WR", "delta_pnl_day"]]


def main() -> int:
    print()
    bot_k = load(DB_BOT_K)
    # Restrict Bot G to Bot K's window for fair comparison
    bot_g_full = load(DB_BOT_G)
    bot_g = bot_g_full[(bot_g_full["ts"] >= bot_k["ts"].min())
                       & (bot_g_full["ts"] <= bot_k["ts"].max())].copy()

    print(f"Bot K trades: {len(bot_k)} ({bot_k['ts'].min()} to {bot_k['ts'].max()})")
    print(f"Bot G trades in K's window: {len(bot_g)}")
    print()

    print("=" * 78)
    print("BOT K DECOMPOSITION (side × weekend × era)")
    print("=" * 78)
    print()
    k_decomp = decompose("Bot K", bot_k)
    k_delta = delta_table(k_decomp)
    print(k_delta.to_string(index=False))
    print()

    print("=" * 78)
    print("BOT G DECOMPOSITION (control — J13 OFF)")
    print("=" * 78)
    print()
    g_decomp = decompose("Bot G", bot_g)
    g_delta = delta_table(g_decomp)
    print(g_delta.to_string(index=False))
    print()

    print("=" * 78)
    print("J13 EFFECT ISOLATED (Bot K delta - Bot G delta per cell)")
    print("=" * 78)
    print()
    isolated = k_delta[["side", "weekend", "delta_pnl_day"]].merge(
        g_delta[["side", "weekend", "delta_pnl_day"]],
        on=["side", "weekend"], suffixes=("_K", "_G"))
    isolated["J13_effect_per_day"] = (
        isolated["delta_pnl_day_K"] - isolated["delta_pnl_day_G"]
    ).round(2)
    print(isolated.to_string(index=False))
    print()

    print("=" * 78)
    print("BOT K TOTALS — verify the -$36/day degradation accounting")
    print("=" * 78)
    print()
    rows = []
    for era_name, mask in [("pre-J13", ~bot_k["post_j13"]),
                            ("post-J13", bot_k["post_j13"])]:
        rows.append({"era": era_name, **cell(bot_k[mask])})
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("Sum of cell deltas should equal whole-bot delta:")
    print(f"  Sum of cell delta_pnl_day: ${k_delta['delta_pnl_day'].sum():+.2f}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
