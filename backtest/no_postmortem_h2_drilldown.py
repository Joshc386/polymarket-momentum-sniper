"""
NO mild-L1 post-mortem -- PATH A drill-down.

Goal: identify WHICH signal layer is producing the NO call in the 51
thin-book bleed trades (displ>0 AND mom>=0 AND ob_ask_depth < ~1255,
WR 19.6%, total PnL -$99.92).

Natural A/B: the same logical cell (displ>0 AND mom>=0) in thick books
has n=18, WR 50%, +$30.10. Same model decision, opposite outcome.
Whatever signal differs between these two groups in the wrong direction
is the bug.

Join fix: use `trade_placed=1` within market_id (signal_ticks records a
placement tick at ~same-microsecond as the trade row).

Read-only. No live data touched. No code paths modified.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRADES_DB = REPO / "data_runtime" / "bot_k_sm_confirmation.db"
DIAG_DB = REPO / "data_runtime" / "bot_k_sm_confirmation_signal_diag.db"

L1_MILD_THRESHOLD = 0.4
THIN_BOOK_THRESHOLD = 1255  # median ob_ask_depth in the NO mild-L1 sample


# ---------- data ----------

def load_no_mild_with_signals() -> pd.DataFrame:
    """Load NO mild-L1 trades with their entry-tick signal sub-components."""
    con_t = sqlite3.connect(TRADES_DB)
    trades = pd.read_sql("""
        SELECT id, timestamp, market_id, side, pnl, edge,
               btc_price_at_entry, oracle_price_at_open,
               oracle_lag_signal, momentum_signal, liquidation_signal,
               orderbook_signal, sentiment_signal, coinbase_direction,
               ob_ask_depth, ob_yes_depth, ob_spread,
               time_remaining_secs, resolution
        FROM trades
        WHERE side = 'NO'
          AND ABS(oracle_lag_signal) <= ?
          AND pnl IS NOT NULL
          AND oracle_price_at_open > 0
          AND btc_price_at_entry IS NOT NULL
    """, con_t, params=(L1_MILD_THRESHOLD,))
    con_t.close()

    market_ids = trades["market_id"].unique().tolist()
    con_d = sqlite3.connect(DIAG_DB)
    placeholders = ",".join("?" * len(market_ids))
    ticks = pd.read_sql(f"""
        SELECT timestamp AS tick_ts, market_id,
               l1_oracle_lag, l1_lag_component, l1_open_component,
               l2_momentum, l3_liquidation,
               l4_orderbook, l4_imbalance, l4_flow, l4_mid_dev,
               l4_top_pressure, l4_thickness,
               l5_sentiment, l6_fade, l7_taker_ratio,
               l8_clob_flow, l9b_absorption, l10_exhaustion,
               l11_trade_size, l12_wallet_flow,
               coinbase_direction AS cb_dir_tick,
               combined_signal AS combined_tick,
               est_prob_up, market_implied_prob, prob_edge,
               trade_placed
        FROM signal_ticks
        WHERE trade_placed = 1
          AND market_id IN ({placeholders})
    """, con_d, params=market_ids)
    con_d.close()

    # 1 placement tick per market in our set
    print(f"Trades (NO mild-L1)         : {len(trades)}")
    print(f"Markets in trade set        : {len(market_ids)}")
    print(f"Placement ticks matched     : {len(ticks)}")

    merged = trades.merge(ticks, on="market_id", how="left", suffixes=("", "_tick"))
    matched = merged["l8_clob_flow"].notna().sum()
    print(f"Trades with signal join     : {matched} / {len(trades)} "
          f"({100*matched/len(trades):.1f}%)")
    print()

    merged["win"] = (merged["pnl"] > 0).astype(int)
    merged["displacement"] = merged["btc_price_at_entry"] - merged["oracle_price_at_open"]
    return merged


# ---------- analysis ----------

def classify_cells(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["book"] = np.where(df["ob_ask_depth"] < THIN_BOOK_THRESHOLD, "thin", "thick")
    df["cell"] = np.select(
        [
            (df["displacement"] <= 0) & (df["momentum_signal"] >= 0),
            (df["displacement"] <= 0) & (df["momentum_signal"] < 0),
            (df["displacement"] > 0) & (df["momentum_signal"] >= 0),
            (df["displacement"] > 0) & (df["momentum_signal"] < 0),
        ],
        ["d<=0_m>=0", "d<=0_m<0", "d>0_m>=0", "d>0_m<0"],
        default="?",
    )
    return df


def summarise_group(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"group": label, "n": 0}
    sub = df.dropna(subset=["l8_clob_flow"])  # only signal-joined rows
    if sub.empty:
        return {"group": label, "n": len(df), "n_joined": 0}
    return {
        "group": label,
        "n": len(df),
        "n_joined": len(sub),
        "WR_%": round(100 * df["win"].mean(), 1),
        "tot_pnl": round(df["pnl"].sum(), 2),
    }


def compare_signal_layers(bleed: pd.DataFrame, control: pd.DataFrame) -> None:
    """Compare mean signal sub-components between bleed group and control group."""
    print("=" * 78)
    print("SIGNAL LAYER COMPARISON")
    print("=" * 78)
    print()
    print(f"BLEED group:   thin book + wrong-side cell  (n={len(bleed)})")
    print(f"CONTROL group: thick book + same cell logic (n={len(control)})")
    print()
    print("Same combiner decision (NO entry), opposite outcomes.")
    print("Whatever pushes BLEED toward NO but CONTROL toward (correctly priced)")
    print("YES is the suspect.")
    print()

    cols = [
        "l1_oracle_lag", "l1_lag_component", "l1_open_component",
        "l2_momentum", "l3_liquidation",
        "l4_orderbook", "l4_imbalance", "l4_flow", "l4_mid_dev",
        "l4_top_pressure", "l4_thickness",
        "l5_sentiment", "l6_fade", "l7_taker_ratio",
        "l8_clob_flow", "l9b_absorption", "l10_exhaustion",
        "l11_trade_size", "l12_wallet_flow",
        "cb_dir_tick", "combined_tick", "est_prob_up",
        "market_implied_prob", "prob_edge",
    ]

    rows = []
    for c in cols:
        if c not in bleed.columns:
            continue
        b = bleed[c].dropna()
        k = control[c].dropna()
        if len(b) == 0 or len(k) == 0:
            continue
        b_mean = b.mean()
        k_mean = k.mean()
        # Welch t-stat (rough, just to rank by separation)
        b_var = b.var(ddof=1) if len(b) > 1 else 0
        k_var = k.var(ddof=1) if len(k) > 1 else 0
        se = ((b_var / len(b)) + (k_var / len(k))) ** 0.5
        t = (b_mean - k_mean) / se if se > 0 else 0.0
        rows.append({
            "layer": c,
            "bleed_mean": round(b_mean, 4),
            "ctrl_mean": round(k_mean, 4),
            "diff (bleed - ctrl)": round(b_mean - k_mean, 4),
            "|t|": round(abs(t), 2),
            "direction": "MORE NEG in bleed" if (b_mean - k_mean) < 0 else "more pos in bleed",
        })
    out = pd.DataFrame(rows).sort_values("|t|", ascending=False)
    print("Sorted by |t| -- biggest separations at top:")
    print(out.to_string(index=False))
    print()


def detail_top_bleeders(df: pd.DataFrame, group_label: str, n_show: int = 10) -> None:
    """Print the most expensive bleed trades with all signal layers visible."""
    print("=" * 78)
    print(f"TOP {n_show} INDIVIDUAL BLEED TRADES IN {group_label}")
    print("=" * 78)
    print()
    sub = df.sort_values("pnl").head(n_show)
    cols_to_show = [
        "id", "timestamp", "pnl", "displacement", "ob_ask_depth",
        "l1_oracle_lag", "l2_momentum",
        "l4_orderbook", "l4_imbalance", "l4_flow",
        "l4_mid_dev", "l4_top_pressure",
        "l5_sentiment", "l7_taker_ratio", "l8_clob_flow",
        "l9b_absorption", "l11_trade_size",
        "combined_tick", "est_prob_up", "market_implied_prob",
    ]
    cols_to_show = [c for c in cols_to_show if c in sub.columns]
    print(sub[cols_to_show].to_string(index=False))
    print()


# ---------- main ----------

def main() -> int:
    print()
    df = load_no_mild_with_signals()
    df = classify_cells(df)

    print("Group counts (post-join):")
    grp = df.groupby(["book", "cell"]).agg(
        n=("id", "count"),
        n_joined=("l8_clob_flow", lambda s: s.notna().sum()),
        WR_pct=("win", lambda s: round(100 * s.mean(), 1)),
        pnl_sum=("pnl", lambda s: round(s.sum(), 2)),
    ).reset_index()
    print(grp.to_string(index=False))
    print()

    # Define groups
    wrong_side = (df["displacement"] > 0) & (df["momentum_signal"] >= 0)
    bleed = df[wrong_side & (df["book"] == "thin") & df["l8_clob_flow"].notna()]
    control = df[wrong_side & (df["book"] == "thick") & df["l8_clob_flow"].notna()]

    print(f"BLEED   (thin × wrong-side, signal-joined) : n={len(bleed)}")
    print(f"CONTROL (thick × wrong-side, signal-joined): n={len(control)}")
    print()

    if len(bleed) < 10 or len(control) < 5:
        print("WARN: groups too small for stable comparison")

    compare_signal_layers(bleed, control)

    # Optional: also compare bleed group vs the THIN winners as a within-thin contrast
    thin_winners = df[(df["book"] == "thin") & (df["win"] == 1)
                      & df["l8_clob_flow"].notna()]
    print(f"Additional contrast -- BLEED vs THIN WINNERS (within thin book):")
    print(f"  thin winners n={len(thin_winners)}")
    print()
    compare_signal_layers(bleed, thin_winners)

    detail_top_bleeders(bleed, "thin-book bleed cell", n_show=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
