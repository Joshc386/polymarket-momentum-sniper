"""Bot K vs Bot K2 A/B monitor + Bot K recent-performance / floor surface area.

Option 1 (2026-05-29): the directional L1 floor's incremental value is
measured by the live paper A/B (Bot K2 = Bot K + floor vs unchanged Bot K).
This is the recurring evaluation tool. Don't draw conclusions before ~week 2.

While Bot K2 accumulates trades, this also answers two questions about
current Bot K:
  1. Is recent profitability broad, or concentrated in WITH-the-line trades
     (L1 on the side of the bet)? If concentrated, it's a favourable
     directional regime and the floor has little to remove right now.
  2. How many AGAINST-the-line trades is Bot K still taking (beyond the
     0.1 deadband)? That is the floor's surface area = expected A/B
     divergence. Their PnL is a first-order estimate of the floor's effect.

L1 (oracle_lag_signal) = displacement from the window-open resolution line.
WITH the line: YES & L1>=0, or NO & L1<=0. AGAINST: YES & L1<0, NO & L1>0.
"floor would block" uses the deadband (default 0.1) matching Bot K2 config.

Read-only.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DB_BOT_K = REPO / "data_runtime" / "bot_k_sm_confirmation.db"
DB_BOT_K2 = REPO / "data_runtime" / "bot_k2_l1_floor.db"
DEADBAND = 0.1


def load(db: Path) -> pd.DataFrame:
    if not db.exists():
        return pd.DataFrame()
    con = sqlite3.connect(db)
    df = pd.read_sql("""
        SELECT timestamp, side, pnl, oracle_lag_signal, entry_price
        FROM trades WHERE pnl IS NOT NULL
    """, con)
    con.close()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["win"] = (df["pnl"] > 0).astype(int)
    df["against_line"] = (
        ((df["side"] == "YES") & (df["oracle_lag_signal"] < 0))
        | ((df["side"] == "NO") & (df["oracle_lag_signal"] > 0))
    )
    df["floor_would_block"] = (
        ((df["side"] == "YES") & (df["oracle_lag_signal"] < -DEADBAND))
        | ((df["side"] == "NO") & (df["oracle_lag_signal"] > DEADBAND))
    )
    return df


def perf(df: pd.DataFrame) -> str:
    if df.empty:
        return "no trades"
    n = len(df)
    return (f"n={n}, WR={100*df['win'].mean():.1f}%, "
            f"PnL=${df['pnl'].sum():+.2f}, ${df['pnl'].mean():+.3f}/trade")


def window(df: pd.DataFrame, days: int, now: pd.Timestamp) -> pd.DataFrame:
    return df[df["ts"] >= now - pd.Timedelta(days=days)]


def main() -> int:
    k = load(DB_BOT_K)
    k2 = load(DB_BOT_K2)
    if k.empty:
        print("No Bot K data.")
        return 1
    now = k["ts"].max()
    print()
    print("=" * 78)
    print(f"BOT K RECENT PERFORMANCE (as of {now:%Y-%m-%d %H:%M} UTC)")
    print("=" * 78)
    for d in (7, 14):
        print(f"  last {d:>2}d: {perf(window(k, d, now))}")
    print(f"  all    : {perf(k)}")
    print()

    # Split both ways across windows. last 7d is entirely POST-J13 (J13
    # deployed 2026-05-21 20:37); last 14d straddles it, so its against-line
    # bleed is inflated by the worse pre-J13 NO zone. last 7d = honest "now".
    for label, days in [("last 7d (post-J13, current regime)", 7),
                        ("last 14d (straddles J13 — inflated)", 14)]:
        kw = window(k, days, now)
        if kw.empty:
            continue
        with_line = kw[~kw["against_line"]]
        against = kw[kw["against_line"]]
        blocked = kw[kw["floor_would_block"]]
        kept = kw[~kw["floor_would_block"]]
        per_day = len(blocked) / max(1, (kw['ts'].max() - kw['ts'].min()).days + 1)
        print("=" * 78)
        print(f"DECOMPOSITION — {label}")
        print("=" * 78)
        print(f"  WITH the line   : {perf(with_line)}")
        print(f"  AGAINST the line: {perf(against)}  "
              f"({100*len(against)/len(kw):.0f}% of trades)")
        print(f"  -- floor (deadband {DEADBAND}) --")
        print(f"  would BLOCK: {perf(blocked)}  (~{per_day:.1f}/day, "
              f"{100*len(blocked)/len(kw):.0f}%)")
        print(f"  would KEEP : {perf(kept)}")
        print(f"  first-order PnL effect: ${-blocked['pnl'].sum():+.2f}")
        print()
    print("  NOTE: first-order only — the live A/B (Bot K2) captures the true")
    print("  counterfactual incl. re-selected entries. This is the surface area.")
    print()

    print("=" * 78)
    print("LIVE A/B — Bot K vs Bot K2 (overlapping period only)")
    print("=" * 78)
    if k2.empty:
        print("  Bot K2 has no trades yet (just deployed). Re-run after it")
        print("  accumulates. Do NOT evaluate before ~week 2 (~2026-06-12).")
    else:
        start = k2["ts"].min()
        k_ov = k[k["ts"] >= start]
        print(f"  overlap since {start:%Y-%m-%d %H:%M}:")
        print(f"    Bot K : {perf(k_ov)}")
        print(f"    Bot K2: {perf(k2)}")
        for side in ("YES", "NO"):
            print(f"    [{side}] K : {perf(k_ov[k_ov['side']==side])}")
            print(f"    [{side}] K2: {perf(k2[k2['side']==side])}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
