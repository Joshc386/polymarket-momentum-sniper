"""
Test 2 (handoff 26-05-28): Bot K YES weekday selection diff, pre vs post-J13.

Bot K only (Bot G dropped per user 2026-05-29 — Bot K is the go-live
candidate, Bot G to be disabled).

Goal: characterise WHAT is different about the YES weekday trades Bot K
takes pre-J13 vs post-J13. J13 kills ~45% of YES weekday trades by volume
(83/day -> 46/day). Did it cut winners or losers?

CODE-CONFIRMED MECHANISM (signals/combiner.py:240-285):
J13 directional_signal_gating is SYMMETRIC. In a strong-bearish-L1 state
(oracle_lag_signal < -0.4) it clamps L4 mid_dev / L4 top_pressure / L7
taker_ratio to be non-positive and takes the MORE BEARISH of the passed /
re-aggregated orderbook signal. That LOWERS est_prob_up. Any YES trade that
was previously firing in a strong-bearish-L1 state (driven by bullish
orderbook/taker micro-signals overriding bearish L1) can therefore be
suppressed post-J13. The "YES strictly preserved" invariant only holds in
the BULLISH-L1 branch.

DECISIVE CUT: bucket YES weekday trades by oracle_lag_signal (L1) sign /
magnitude, pre vs post, with n / WR / pnl-per-trade. If a profitable
YES-in-bearish-L1 population existed pre-J13 and vanished post-J13, the
hole materially cost the YES path. If those trades were losers, J13 is
doing its job and the degradation is benign (regime + correct suppression).

Sign convention (verified combiner.py + gating code): oracle_lag_signal > 0
= bullish (favours UP/YES); < 0 = bearish (favours DOWN/NO). A YES trade is
a bet on UP. YES trade with L1 < 0 = bot bet UP against bearish oracle lag,
driven by other layers.

Read-only.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DB_BOT_K = REPO / "data_runtime" / "bot_k_sm_confirmation.db"

# Commit-derived boundaries (verified via git log 2026-05-28/29).
RANGING_FIX  = pd.Timestamp("2026-05-15 09:32:00", tz="UTC")  # 46b7a2b
J13_DEPLOYED = pd.Timestamp("2026-05-21 20:37:00", tz="UTC")  # 9ef0523
GATE_THRESHOLD = 0.4  # J13 gate_threshold (combiner default)


def load() -> pd.DataFrame:
    con = sqlite3.connect(DB_BOT_K)
    df = pd.read_sql("""
        SELECT id, timestamp, side, pnl, edge, size_usdc, time_remaining_secs,
               oracle_lag_signal, combined_signal, regime, entry_price,
               momentum_signal, orderbook_signal
        FROM trades WHERE pnl IS NOT NULL
    """, con)
    con.close()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["weekend"] = df["ts"].dt.dayofweek >= 5
    df["win"] = (df["pnl"] > 0).astype(int)
    df = df[(df["ts"] >= RANGING_FIX)].copy()
    df["era"] = np.where(df["ts"] < J13_DEPLOYED, "pre_J13", "post_J13")
    return df


def yes_weekday(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["side"] == "YES") & (~df["weekend"])].copy()


def summ(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {"n": 0}
    days = max(1, (sub["ts"].max() - sub["ts"].min()).days + 1)
    return {
        "n": n,
        "trades_per_day": round(n / days, 1),
        "WR_%": round(100 * sub["win"].mean(), 1),
        "pnl_per_trade": round(sub["pnl"].mean(), 3),
        "pnl_total": round(sub["pnl"].sum(), 2),
    }


def dist_row(label: str, s: pd.Series) -> dict:
    return {
        "field": label,
        "mean": round(s.mean(), 4),
        "p10": round(s.quantile(0.10), 4),
        "p50": round(s.quantile(0.50), 4),
        "p90": round(s.quantile(0.90), 4),
        "max": round(s.max(), 4),
    }


def l1_bucket(x: float) -> str:
    if x < -GATE_THRESHOLD:
        return "1: L1 strong bearish (<-0.4) [J13-GATED]"
    if x < 0:
        return "2: L1 mild bearish (-0.4..0)"
    if x <= GATE_THRESHOLD:
        return "3: L1 mild bullish (0..0.4)"
    return "4: L1 strong bullish (>0.4) [J13-AMPLIFIED]"


def main() -> int:
    df = load()
    yw = yes_weekday(df)
    pre = yw[yw["era"] == "pre_J13"]
    post = yw[yw["era"] == "post_J13"]

    print()
    print("=" * 84)
    print("TEST 2 — Bot K YES WEEKDAY selection diff (pre-J13 vs post-J13)")
    print("=" * 84)
    print(f"pre-J13  window: {RANGING_FIX}  ->  {J13_DEPLOYED}")
    print(f"post-J13 window: {J13_DEPLOYED}  ->  {df['ts'].max()}")
    print()

    print("HEADLINE (YES weekday):")
    print(f"  pre-J13 : {summ(pre)}")
    print(f"  post-J13: {summ(post)}")
    print()

    # ── DECISIVE CUT: L1 bucket decomposition ──
    print("=" * 84)
    print("DECISIVE CUT — YES weekday by L1 (oracle_lag_signal) bucket")
    print("=" * 84)
    print("If a profitable 'L1 strong bearish [J13-GATED]' population existed")
    print("pre-J13 and vanished/degraded post-J13, J13's hole cost the YES path.")
    print()
    rows = []
    for era_name, sub in [("pre_J13", pre), ("post_J13", post)]:
        sub = sub.copy()
        sub["l1b"] = sub["oracle_lag_signal"].apply(l1_bucket)
        for b in sorted(sub["l1b"].unique()):
            cell = sub[sub["l1b"] == b]
            rows.append({
                "era": era_name, "L1_bucket": b, "n": len(cell),
                "WR_%": round(100 * cell["win"].mean(), 1),
                "pnl_per_trade": round(cell["pnl"].mean(), 3),
                "pnl_total": round(cell["pnl"].sum(), 2),
            })
    bucket_df = pd.DataFrame(rows).sort_values(["L1_bucket", "era"])
    print(bucket_df.to_string(index=False))
    print()

    # Focused: the J13-gated (strong bearish L1) YES population
    pre_gated = pre[pre["oracle_lag_signal"] < -GATE_THRESHOLD]
    post_gated = post[post["oracle_lag_signal"] < -GATE_THRESHOLD]
    print("FOCUS — YES weekday trades in strong-bearish-L1 (the J13-gated zone):")
    print(f"  pre-J13 : n={len(pre_gated)}  "
          f"WR={100*pre_gated['win'].mean() if len(pre_gated) else float('nan'):.1f}%  "
          f"pnl/trade=${pre_gated['pnl'].mean() if len(pre_gated) else 0:.3f}  "
          f"total=${pre_gated['pnl'].sum():.2f}")
    print(f"  post-J13: n={len(post_gated)}  "
          f"WR={100*post_gated['win'].mean() if len(post_gated) else float('nan'):.1f}%  "
          f"pnl/trade=${post_gated['pnl'].mean() if len(post_gated) else 0:.3f}  "
          f"total=${post_gated['pnl'].sum():.2f}")
    print()
    pre_per_day = len(pre_gated) / max(1, (pre["ts"].max() - pre["ts"].min()).days + 1)
    post_per_day = len(post_gated) / max(1, (post["ts"].max() - post["ts"].min()).days + 1)
    print(f"  J13-gated-zone YES trades/day: {pre_per_day:.1f} pre -> {post_per_day:.1f} post "
          f"({100*(post_per_day-pre_per_day)/pre_per_day if pre_per_day else 0:+.0f}%)")
    print()

    # ── Distribution comparisons ──
    print("=" * 84)
    print("DISTRIBUTION SHIFTS (YES weekday, pre vs post)")
    print("=" * 84)
    for field in ["edge", "size_usdc", "time_remaining_secs",
                  "combined_signal", "oracle_lag_signal", "entry_price"]:
        print(f"\n  {field}:")
        d = pd.DataFrame([
            {"era": "pre_J13", **dist_row(field, pre[field])},
            {"era": "post_J13", **dist_row(field, post[field])},
        ])
        print(d.to_string(index=False))
    print()

    # ── Regime composition ──
    print("=" * 84)
    print("REGIME COMPOSITION (YES weekday, share of trades + pnl/trade)")
    print("=" * 84)
    rows = []
    for era_name, sub in [("pre_J13", pre), ("post_J13", post)]:
        for reg in sorted(sub["regime"].dropna().unique()):
            cell = sub[sub["regime"] == reg]
            rows.append({
                "era": era_name, "regime": reg, "n": len(cell),
                "share_%": round(100 * len(cell) / len(sub), 1),
                "WR_%": round(100 * cell["win"].mean(), 1),
                "pnl_per_trade": round(cell["pnl"].mean(), 3),
            })
    print(pd.DataFrame(rows).sort_values(["regime", "era"]).to_string(index=False))
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
