"""
NO-side post-mortem, hypotheses H2 and H2b.

H2  -- L8 CLOB-flow miscalibration in thin-book windows.
H2b -- "Selloff-but-still-above-open" trap: NO fires when recent momentum
       is negative but BTC is still above the window-open resolution
       anchor, which still favours YES.

Scope: paper NO trades with |oracle_lag_signal| <= 0.4 (the mild-L1
bucket that's still bleeding ~$330/wk at ~25% WR per the post-mortem).

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
JOIN_TOLERANCE_SECONDS = 5
MIN_BUCKET_N = 30


# ---------- data loaders ----------

def load_no_mild_trades() -> pd.DataFrame:
    con = sqlite3.connect(TRADES_DB)
    q = """
    SELECT id, timestamp, market_id, side, entry_price, pnl,
           btc_price_at_entry, oracle_price_at_open, oracle_price_at_entry,
           oracle_lag_signal, momentum_signal, liquidation_signal,
           orderbook_signal, combined_signal,
           estimated_prob_up, market_implied_prob, edge,
           time_remaining_secs, resolution,
           ob_ask_depth, ob_yes_depth, ob_spread
    FROM trades
    WHERE side = 'NO'
      AND ABS(oracle_lag_signal) <= ?
      AND pnl IS NOT NULL
      AND oracle_price_at_open IS NOT NULL
      AND oracle_price_at_open > 0
      AND btc_price_at_entry IS NOT NULL
    """
    df = pd.read_sql(q, con, params=(L1_MILD_THRESHOLD,))
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["win"] = (df["pnl"] > 0).astype(int)
    df["displacement"] = df["btc_price_at_entry"] - df["oracle_price_at_open"]
    return df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)


def load_signal_ticks_for(market_ids: list[str]) -> pd.DataFrame:
    """Load signal_ticks rows only for markets in our trade set."""
    if not market_ids:
        return pd.DataFrame()
    con = sqlite3.connect(DIAG_DB)
    placeholders = ",".join("?" * len(market_ids))
    q = f"""
    SELECT timestamp, market_id,
           l1_oracle_lag, l2_momentum, l4_orderbook, l4_thickness,
           l4_imbalance, l4_flow,
           l8_clob_flow, l9b_absorption, l10_exhaustion, l11_trade_size,
           combined_signal, est_prob_up, prob_edge
    FROM signal_ticks
    WHERE market_id IN ({placeholders})
    """
    df = pd.read_sql(q, con, params=market_ids)
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)


def join_l8_to_trades(trades: pd.DataFrame, ticks: pd.DataFrame) -> pd.DataFrame:
    """merge_asof: for each trade, attach the nearest signal_tick within tolerance."""
    if ticks.empty:
        trades["l8_clob_flow"] = np.nan
        trades["l4_thickness"] = np.nan
        return trades

    tol = pd.Timedelta(seconds=JOIN_TOLERANCE_SECONDS)
    # merge_asof requires both sides sorted on the 'on' key globally
    t = trades.sort_values("timestamp").reset_index(drop=True)
    k = ticks.sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(
        t, k,
        on="timestamp",
        by="market_id",
        tolerance=tol,
        direction="nearest",
        suffixes=("", "_tick"),
    )
    return merged


# ---------- helpers ----------

def ci95_pct(p: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    se = (p * (1.0 - p) / n) ** 0.5
    return (p - 1.96 * se) * 100, (p + 1.96 * se) * 100


def bucket_stats(df: pd.DataFrame, label: str) -> dict:
    n = len(df)
    if n == 0:
        return {"bucket": label, "n": 0}
    wr = df["win"].mean()
    lo, hi = ci95_pct(wr, n)
    return {
        "bucket": label,
        "n": n,
        "WR_%": round(100 * wr, 2),
        "ci_lo_%": round(lo, 2),
        "ci_hi_%": round(hi, 2),
        "mean_pnl": round(df["pnl"].mean(), 3),
        "total_pnl": round(df["pnl"].sum(), 2),
        "mean_edge": round(df["edge"].mean(), 4),
        "mean_displ": round(df["displacement"].mean(), 1),
        "mean_mom": round(df["momentum_signal"].mean(), 3),
    }


# ---------- Test A: L8 in thin books ----------

def test_a_l8_thin_books(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("TEST A -- L8 CLOB-flow miscalibration vs book thickness")
    print("=" * 78)
    print()
    print("Bucketing NO mild-L1 trades by ob_ask_depth quartile (proxy for NO-side")
    print("liquidity at entry). If L8 saturates in thin books, the thin-book bucket")
    print("should show low WR while |L8| at entry stays elevated.")
    print()

    sub = df.dropna(subset=["ob_ask_depth"]).copy()
    sub["depth_q"] = pd.qcut(sub["ob_ask_depth"], 4, labels=["Q1 thinnest", "Q2", "Q3", "Q4 thickest"])

    rows = []
    for label, grp in sub.groupby("depth_q", observed=True):
        s = bucket_stats(grp, str(label))
        s["depth_range"] = f"{grp['ob_ask_depth'].min():.0f}-{grp['ob_ask_depth'].max():.0f}"
        if "l8_clob_flow" in grp.columns and grp["l8_clob_flow"].notna().any():
            s["mean_abs_l8"] = round(grp["l8_clob_flow"].abs().mean(), 4)
            s["mean_l4_thickness"] = round(grp["l4_thickness"].mean(), 4) if "l4_thickness" in grp else None
        rows.append(s)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print()

    # Sanity check: overall NO mild-L1 stats
    overall = bucket_stats(df, "ALL NO mild-L1")
    print("Reference (whole filter):")
    for k, v in overall.items():
        print(f"  {k:>10}: {v}")
    print()


# ---------- Test B: selloff-but-above-open trap ----------

def test_b_selloff_trap(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("TEST B -- 'Selloff but BTC still above the open' trap (H2b)")
    print("=" * 78)
    print()
    print("displacement = btc_price_at_entry - oracle_price_at_open")
    print("  > 0 means BTC trading above the window's resolution anchor")
    print("  (which favours YES, against a NO position)")
    print()
    print("momentum = momentum_signal at entry")
    print("  < 0 means recent ticks down -- 'selloff in progress'")
    print()
    print("Trap cell = (displacement > 0) AND (momentum < 0)")
    print()

    df = df.copy()
    df["displ_pos"] = df["displacement"] > 0
    df["mom_neg"] = df["momentum_signal"] < 0

    cells = [
        ("displ<=0, mom>=0  (selloff confirmed, NO is on-side)", ~df["displ_pos"] & ~df["mom_neg"]),
        ("displ<=0, mom<0   (selloff continuation, NO on-side)", ~df["displ_pos"] & df["mom_neg"]),
        ("displ>0,  mom>=0  (above open + recent up, NO wrong-side)", df["displ_pos"] & ~df["mom_neg"]),
        ("displ>0,  mom<0   (TRAP: above open, selloff fires NO)", df["displ_pos"] & df["mom_neg"]),
    ]

    rows = []
    for label, mask in cells:
        s = bucket_stats(df[mask], label)
        rows.append(s)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print()

    trap_mask = df["displ_pos"] & df["mom_neg"]
    safe_mask = ~trap_mask
    n_trap, n_safe = trap_mask.sum(), safe_mask.sum()
    if n_trap >= MIN_BUCKET_N and n_safe >= MIN_BUCKET_N:
        wr_trap = df.loc[trap_mask, "win"].mean()
        wr_safe = df.loc[safe_mask, "win"].mean()
        pnl_trap = df.loc[trap_mask, "pnl"].sum()
        pnl_safe = df.loc[safe_mask, "pnl"].sum()
        # 2-proportion z-test
        p_pool = df["win"].mean()
        se = (p_pool * (1 - p_pool) * (1/n_trap + 1/n_safe)) ** 0.5
        z = (wr_trap - wr_safe) / se if se > 0 else 0.0
        print(f"Trap vs rest:")
        print(f"  trap : n={n_trap:3d}  WR={100*wr_trap:5.2f}%  total_pnl=${pnl_trap:+.2f}")
        print(f"  rest : n={n_safe:3d}  WR={100*wr_safe:5.2f}%  total_pnl=${pnl_safe:+.2f}")
        print(f"  WR diff: {100*(wr_trap-wr_safe):+.2f} pp   z-score: {z:+.2f}")
        if abs(z) >= 1.96:
            verdict = "STATISTICALLY SIGNIFICANT (|z| >= 1.96)"
        elif abs(z) >= 1.28:
            verdict = "suggestive (|z| >= 1.28)"
        else:
            verdict = "not significant"
        print(f"  verdict: {verdict}")
    else:
        print(f"Trap n={n_trap}, rest n={n_safe} -- one cell below MIN_BUCKET_N={MIN_BUCKET_N}, skipping z-test")
    print()


# ---------- Test B+: deeper view by displacement bin ----------

def test_b_plus_displacement_bins(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("TEST B+ -- WR by displacement bin (momentum sign overlaid)")
    print("=" * 78)
    print()
    print("Same NO mild-L1 trades. Bins of displacement in $; rows are momentum sign.")
    print()

    df = df.copy()
    bins = [-1e9, -100, -25, 0, 25, 100, 1e9]
    labels = ["< -$100", "-$100..-$25", "-$25..$0", "$0..$25", "$25..$100", "> $100"]
    df["displ_bin"] = pd.cut(df["displacement"], bins=bins, labels=labels)
    df["mom_sign"] = np.where(df["momentum_signal"] >= 0, "mom>=0", "mom<0")

    pivot_n = df.pivot_table(index="mom_sign", columns="displ_bin", values="win",
                              aggfunc="count", observed=False, fill_value=0)
    pivot_wr = df.pivot_table(index="mom_sign", columns="displ_bin", values="win",
                                aggfunc="mean", observed=False) * 100

    print("Counts per cell:")
    print(pivot_n.to_string())
    print()
    print("WR% per cell (NaN = empty):")
    print(pivot_wr.round(1).to_string())
    print()


# ---------- main ----------

def main() -> int:
    print()
    print(f"DB trades : {TRADES_DB}")
    print(f"DB diag   : {DIAG_DB}")
    print(f"Filter    : side='NO' AND |oracle_lag_signal| <= {L1_MILD_THRESHOLD}")
    print()

    trades = load_no_mild_trades()
    print(f"Loaded {len(trades)} NO mild-L1 trades.")
    print(f"  date range: {trades['timestamp'].min()} to {trades['timestamp'].max()}")
    print(f"  overall WR: {100*trades['win'].mean():.2f}%   total PnL: ${trades['pnl'].sum():+.2f}")
    print()

    market_ids = trades["market_id"].unique().tolist()
    print(f"Loading signal_ticks for {len(market_ids)} markets...")
    ticks = load_signal_ticks_for(market_ids)
    print(f"  loaded {len(ticks):,} tick rows")
    print()

    trades = join_l8_to_trades(trades, ticks)
    matched_l8 = trades["l8_clob_flow"].notna().sum()
    print(f"Trades matched to a signal_tick within {JOIN_TOLERANCE_SECONDS}s: "
          f"{matched_l8} / {len(trades)} ({100*matched_l8/len(trades):.1f}%)")
    print()

    test_a_l8_thin_books(trades)
    test_b_selloff_trap(trades)
    test_b_plus_displacement_bins(trades)
    test_c_crosstab(trades)

    return 0


# ---------- Test C: cross-tab book thickness x displacement/momentum cell ----------

def test_c_crosstab(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("TEST C -- Cross-tab: book thickness x displacement/momentum cell")
    print("=" * 78)
    print()
    print("Unifies Finding 1 (thin books bleed) with Finding 2 (NO fires while")
    print("displacement+momentum say YES). If wrong-side bleed disappears in thick")
    print("books, both findings are the same phenomenon and a thin-book NO veto")
    print("would fix it.")
    print()

    sub = df.dropna(subset=["ob_ask_depth"]).copy()
    # Coarse 2-way split first for readability
    median_depth = sub["ob_ask_depth"].median()
    sub["book"] = np.where(sub["ob_ask_depth"] < median_depth, "thin", "thick")

    sub["cell"] = np.select(
        [
            (sub["displacement"] <= 0) & (sub["momentum_signal"] >= 0),
            (sub["displacement"] <= 0) & (sub["momentum_signal"] < 0),
            (sub["displacement"] > 0)  & (sub["momentum_signal"] >= 0),
            (sub["displacement"] > 0)  & (sub["momentum_signal"] < 0),
        ],
        [
            "displ<=0, mom>=0 (NO on-side)",
            "displ<=0, mom<0  (selloff cont.)",
            "displ>0,  mom>=0 (wrong-side both)",
            "displ>0,  mom<0  (your trap)",
        ],
        default="?",
    )

    print(f"Median ob_ask_depth split: thin < {median_depth:.0f} <= thick")
    print()
    print("Per-cell WR% (with n) by book thickness:")
    print()

    rows = []
    for cell in [
        "displ<=0, mom>=0 (NO on-side)",
        "displ<=0, mom<0  (selloff cont.)",
        "displ>0,  mom>=0 (wrong-side both)",
        "displ>0,  mom<0  (your trap)",
    ]:
        for book in ["thin", "thick"]:
            grp = sub[(sub["cell"] == cell) & (sub["book"] == book)]
            n = len(grp)
            if n == 0:
                rows.append({"cell": cell, "book": book, "n": 0,
                             "WR_%": None, "total_pnl": 0.0, "mean_edge": None})
                continue
            wr = grp["win"].mean()
            rows.append({
                "cell": cell,
                "book": book,
                "n": n,
                "WR_%": round(100 * wr, 2),
                "total_pnl": round(grp["pnl"].sum(), 2),
                "mean_edge": round(grp["edge"].mean(), 4),
            })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print()

    # Compact pivot view
    pivot_wr = sub.pivot_table(index="cell", columns="book", values="win",
                                aggfunc="mean", observed=False) * 100
    pivot_n = sub.pivot_table(index="cell", columns="book", values="win",
                               aggfunc="count", observed=False, fill_value=0)
    pivot_pnl = sub.pivot_table(index="cell", columns="book", values="pnl",
                                 aggfunc="sum", observed=False, fill_value=0)
    print("WR% pivot (rows = displ/mom cell, cols = book thickness):")
    print(pivot_wr.round(1).to_string())
    print()
    print("n pivot:")
    print(pivot_n.to_string())
    print()
    print("Total PnL pivot ($):")
    print(pivot_pnl.round(2).to_string())
    print()

    # Bottom line: where is the bleed concentrated?
    bleed = sub[sub["pnl"] < 0]
    bleed_summary = (
        bleed.groupby(["book", "cell"])["pnl"].agg(["count", "sum"])
        .reset_index().sort_values("sum")
    )
    print("Losing trades, sorted by total loss (most bleed first):")
    print(bleed_summary.to_string(index=False))
    print()


if __name__ == "__main__":
    sys.exit(main())
