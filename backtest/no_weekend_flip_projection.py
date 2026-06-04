"""
No-flip projection: what happens to Bot K's PnL if weekend_flip is disabled?

Approach:
  Bot K runs signal_aligned + weekend_flip. On weekends, raw_signal gets
  negated, inverting side selection. Hypothesis: this destroys value on
  weekends because the model is genuinely informative.

  For every actual Bot K trade, reverse-engineer the pre-flip raw_signal
  from the logged post-flip est_prob_up. Then project what side / PnL the
  bot would have produced under weekend_flip=False.

Scope: ALL Bot K trades (n=1786), not just NO mild-L1. Covers both the
mild-L1 NO bleed and the bigger weekday/YES interactions.

Assumptions / caveats (printed at top of run):
  * max_adjustment = 0.195 (per config_multi.yaml line 570)
  * min_edge = 0.003 (per config line 587)
  * Fixed $5 trade size — ignores Kelly resizing under different edge
  * Opposite-side fill price approximated as 1 - same_side_price
    (ignores actual spread on the other side of the book)
  * EARLY_EXIT resolutions: assumed to recur with the same PnL regardless
    of side choice (stop-loss machinery is side-agnostic in spirit;
    not strictly true but conservative — likely understates no-flip benefit)
  * Only covers trades that ACTUALLY HAPPENED. Trades that no-flip would
    have taken but original config didn't are NOT captured. To capture
    those would need re-simulation over all 635k signal_ticks.
  * Winner fee = 2% per config (fee_adjustment: 0.02)

Read-only. No live data touched.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRADES_DB = REPO / "data_runtime" / "bot_k_sm_confirmation.db"

MAX_ADJUSTMENT = 0.195
MIN_EDGE = 0.003
TRADE_SIZE = 5.0
WINNER_FEE = 0.02


# ---------- data ----------

def load_trades() -> pd.DataFrame:
    con = sqlite3.connect(TRADES_DB)
    df = pd.read_sql("""
        SELECT id, timestamp, market_id, side, entry_price,
               estimated_prob_up, market_implied_prob, edge,
               combined_signal,
               resolution, pnl, size_usdc, num_shares,
               oracle_lag_signal
        FROM trades
        WHERE pnl IS NOT NULL
          AND estimated_prob_up IS NOT NULL
          AND market_implied_prob IS NOT NULL
          AND entry_price > 0
    """, con)
    con.close()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["weekend"] = df["ts"].dt.dayofweek >= 5
    df["mild_l1"] = df["oracle_lag_signal"].abs() <= 0.4
    return df


# ---------- projection ----------

def project_no_flip(df: pd.DataFrame) -> pd.DataFrame:
    """For each actual trade, compute what no-flip would have produced."""
    df = df.copy()

    # The logged combined_signal IS the post-flip raw_signal.
    # Reverse-engineer pre-flip: on weekends, multiply by -1; on weekdays
    # leave alone (the flip only fires on weekends).
    df["raw_post_flip"] = df["combined_signal"]
    df["raw_no_flip"] = np.where(df["weekend"], -df["raw_post_flip"],
                                  df["raw_post_flip"])

    # Recompute est_prob_up under no-flip
    df["est_prob_up_no_flip"] = (0.5 + df["raw_no_flip"] * MAX_ADJUSTMENT).clip(0.05, 0.95)

    # signal_aligned side selection
    df["side_no_flip"] = np.where(df["est_prob_up_no_flip"] >= 0.5, "YES", "NO")

    # Edge under no-flip (magnitude of model vs market disagreement)
    df["edge_no_flip"] = np.abs(df["est_prob_up_no_flip"] - df["market_implied_prob"])

    # Would the trade be taken under no-flip? min_edge filter
    df["taken_no_flip"] = df["edge_no_flip"] >= MIN_EDGE

    # Side changed?
    df["side_flipped"] = df["side"] != df["side_no_flip"]

    return df


def project_pnl(row) -> tuple[float, str]:
    """Project PnL for a single trade under no-flip.

    Returns (projected_pnl, classification).
    """
    res = row["resolution"]
    side_new = row["side_no_flip"]
    side_old = row["side"]
    taken = row["taken_no_flip"]
    price_old = row["entry_price"]  # same-side price at original entry

    # If trade wouldn't be taken under no-flip (edge below min_edge)
    if not taken:
        return (0.0, "DROPPED (edge<min)")

    # Side is unchanged → same PnL (same trade, same outcome)
    if side_new == side_old:
        return (row["pnl"], "SAME_SIDE")

    # Side flipped. Need to project new PnL.
    # Approximate opposite-side entry price as (1 - same_side_price)
    price_new = max(0.05, min(0.95, 1.0 - price_old))
    shares_new = TRADE_SIZE / price_new

    # EARLY_EXIT resolutions — keep original PnL (stop-loss conservative)
    if isinstance(res, str) and res.startswith("EARLY_EXIT"):
        return (row["pnl"], "EARLY_EXIT_KEPT")

    # Standard UP / DOWN resolution
    win = (side_new == "YES" and res == "UP") or (side_new == "NO" and res == "DOWN")
    if win:
        gross = shares_new * 1.0 - TRADE_SIZE  # profit before fee
        fee = WINNER_FEE * shares_new  # 2% fee on payout
        pnl = gross - fee
        return (pnl, "FLIPPED_WIN")
    else:
        return (-TRADE_SIZE, "FLIPPED_LOSS")


def apply_projection(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    out = df.apply(project_pnl, axis=1, result_type="expand")
    df["pnl_no_flip"] = out[0]
    df["projection_class"] = out[1]
    df["pnl_delta"] = df["pnl_no_flip"] - df["pnl"]
    return df


# ---------- reporting ----------

def summarise(df: pd.DataFrame, title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"  n trades                   : {len(df):,}")
    print(f"  ORIGINAL total PnL         : ${df['pnl'].sum():+.2f}")
    print(f"  PROJECTED no-flip PnL      : ${df['pnl_no_flip'].sum():+.2f}")
    print(f"  Delta                      : ${df['pnl_delta'].sum():+.2f}")
    print()
    print(f"  ORIG WR                    : {100*(df['pnl']>0).mean():.2f}%")
    print(f"  PROJ WR (excl. dropped)    : "
          f"{100*(df[df['taken_no_flip']]['pnl_no_flip']>0).mean():.2f}%")
    print(f"  trades dropped under no-flip: {(~df['taken_no_flip']).sum()}")
    print(f"  trades with side flipped   : {df['side_flipped'].sum()}")
    print()


def breakdown(df: pd.DataFrame, title: str, by_cols: list[str]) -> None:
    print("-" * 78)
    print(f"BREAKDOWN: {title}")
    print("-" * 78)
    g = df.groupby(by_cols, observed=True).agg(
        n=("id", "count"),
        orig_pnl=("pnl", "sum"),
        new_pnl=("pnl_no_flip", "sum"),
        delta=("pnl_delta", "sum"),
        orig_wr=("pnl", lambda s: round(100 * (s > 0).mean(), 1)),
        new_wr=("pnl_no_flip", lambda s: round(100 * (s > 0).mean(), 1)),
        n_flipped=("side_flipped", "sum"),
        n_dropped=("taken_no_flip", lambda s: (~s).sum()),
    ).round(2).reset_index()
    print(g.to_string(index=False))
    print()


def classification_breakdown(df: pd.DataFrame) -> None:
    print("-" * 78)
    print("PROJECTION CLASSIFICATION (which trades drove the delta?)")
    print("-" * 78)
    g = df.groupby("projection_class").agg(
        n=("id", "count"),
        orig_pnl=("pnl", "sum"),
        new_pnl=("pnl_no_flip", "sum"),
        delta=("pnl_delta", "sum"),
    ).round(2).reset_index().sort_values("delta", ascending=False)
    print(g.to_string(index=False))
    print()


# ---------- main ----------

def main() -> int:
    print()
    print("ASSUMPTIONS:")
    print(f"  max_adjustment      = {MAX_ADJUSTMENT}")
    print(f"  min_edge            = {MIN_EDGE}")
    print(f"  trade_size          = ${TRADE_SIZE}")
    print(f"  winner_fee          = {WINNER_FEE*100:.0f}%")
    print(f"  opposite-side price = 1 - same_side_price (approx)")
    print(f"  EARLY_EXIT outcomes assumed to recur unchanged")
    print()

    df = load_trades()
    df = project_no_flip(df)
    df = apply_projection(df)

    print(f"date range: {df['ts'].min()} to {df['ts'].max()}")
    print()

    summarise(df, "WHOLE BOT K (all sides, all weekday/weekend)")
    breakdown(df, "by weekend × side", ["weekend", "side"])
    breakdown(df, "by weekend × side × mild_l1", ["weekend", "side", "mild_l1"])
    classification_breakdown(df)

    # Zoom into the mild-L1 NO bleed cell that started this whole investigation
    print("=" * 78)
    print("ZOOM: NO mild-L1 (the original post-mortem sample)")
    print("=" * 78)
    sub = df[(df["side"] == "NO") & df["mild_l1"]]
    summarise(sub, "NO mild-L1 only")
    breakdown(sub, "by weekend", ["weekend"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
