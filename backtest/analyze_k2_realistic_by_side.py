"""Realistic maker-fill edge, SPLIT BY SIDE — does the NO-only thesis survive fills?

The scorecard (maker_fill_realism.py) reports the whole strategy. Our paper-edge
analysis (analyze_k2_paper_edge.py) showed the edge is one-sided: NO makes money,
YES loses. This re-runs the SAME fill model but breaks NO vs YES (and NO by L1
zone) so we can see whether a NO-only min-size probe clears realistic fills.

Reuses the validated pure helpers from maker_fill_realism (would_fill, taker_fee).
Read-only. Run with system python (needs pandas).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from maker_fill_realism import (  # same dir
    GTC_TIMEOUT_S,
    TAKER_FLOOR_SECS,
    taker_fee,
    would_fill,
)

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "data_runtime"
BOT = "bot_k2_l1_floor"
SINCE = "2026-05-29"  # match scorecard A/B window


def load_trades() -> pd.DataFrame:
    q = (
        "SELECT timestamp, market_slug, side, entry_price, ob_spread, "
        "time_remaining_secs, resolution, size_usdc, oracle_lag_signal FROM trades "
        "WHERE resolution IN ('UP','DOWN') AND entry_price IS NOT NULL "
        "AND ob_spread IS NOT NULL AND timestamp >= ? AND oracle_price_at_open != 0"
    )
    with sqlite3.connect(str(RUNTIME / f"{BOT}.db")) as c:
        df = pd.read_sql_query(q, c, params=(SINCE,))
    df["won"] = (
        ((df["side"] == "YES") & (df["resolution"] == "UP"))
        | ((df["side"] == "NO") & (df["resolution"] == "DOWN"))
    ).astype(int)
    return df


def load_mid_paths() -> dict:
    with sqlite3.connect(str(RUNTIME / f"{BOT}_signal_diag.db")) as c:
        st = pd.read_sql_query(
            "SELECT market_slug, secs_remaining, market_implied_prob AS mip "
            "FROM signal_ticks WHERE secs_remaining BETWEEN 0 AND 300 "
            "AND market_implied_prob IS NOT NULL",
            c,
        )
    return {s: g.sort_values("secs_remaining", ascending=False) for s, g in st.groupby("market_slug")}


def simulate(trades: pd.DataFrame, paths: dict, window: str) -> pd.DataFrame:
    rows = []
    for t in trades.itertuples():
        path = paths.get(t.market_slug)
        if path is None:
            continue
        lo = TAKER_FLOOR_SECS if window == "optimistic" else t.time_remaining_secs - GTC_TIMEOUT_S
        seg = path[(path["secs_remaining"] < t.time_remaining_secs) & (path["secs_remaining"] >= lo)]
        p_side_path = seg["mip"].to_numpy() if t.side == "YES" else (1.0 - seg["mip"]).to_numpy()
        p_side_entry = t.entry_price - t.ob_spread
        filled = would_fill(p_side_entry, t.ob_spread, p_side_path)
        maker_L = t.entry_price - t.ob_spread
        rows.append({
            "side": t.side, "L1": t.oracle_lag_signal, "won": t.won, "filled": filled,
            "maker_L": maker_L,
            "pnl_maker": (t.won - maker_L) if filled else np.nan,
            "pnl_paper": t.won - t.entry_price - taker_fee(t.entry_price),
        })
    return pd.DataFrame(rows)


def line(label: str, g: pd.DataFrame) -> None:
    n = len(g)
    if n == 0:
        print(f"  {label:26s} n=0")
        return
    fills = g[g["filled"]]
    miss = g[~g["filled"]]
    fr = len(fills) / n
    if len(fills):
        ev = fills["pnl_maker"].mean()
        tot = fills["pnl_maker"].sum()
        fwin = fills["won"].mean()
    else:
        ev = tot = fwin = float("nan")
    mwin = miss["won"].mean() if len(miss) else float("nan")
    print(
        f"  {label:26s} n={n:4d} fill%={fr:4.0%} | FILLED win={fwin:4.0%} "
        f"EV/sh={ev*100:+5.2f}c tot={tot*100:+7.1f}c | MISSED win={mwin:4.0%}"
    )


def main() -> None:
    trades = load_trades()
    paths = load_mid_paths()
    print(f"{BOT}  since {SINCE}   resolved UP/DOWN trades: {len(trades)}\n")
    for window in ("strict", "optimistic"):
        sim = simulate(trades, paths, window)
        cov = len(sim)
        print(f"=== {window.upper()} window  (covered={cov}) ===")
        line("ALL", sim)
        line("NO / short", sim[sim["side"] == "NO"])
        line("YES / long", sim[sim["side"] == "YES"])
        no = sim[sim["side"] == "NO"]
        line("  NO  strong L1<=-0.4", no[no["L1"] <= -0.4])
        line("  NO  weak  -0.4..0.4", no[(no["L1"] > -0.4) & (no["L1"] < 0.4)])
        print()


if __name__ == "__main__":
    main()
