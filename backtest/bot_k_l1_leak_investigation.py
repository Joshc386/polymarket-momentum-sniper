"""
L1 leak investigation (2026-05-29): does Bot K leak money on YES trades
taken when BTC is below the window-open resolution line?

Context: L1 (oracle_lag_signal) = clamp((btc_ref - window_open)/window_open
/ 0.1%) = BTC's displacement from the window-open resolution line. L1 > 0
means price is ABOVE the line (YES/UP currently winning); L1 < 0 means
BELOW (YES/UP currently losing). J13 does NOT modify L1 and is inert in
the -0.4..0 zone (it only acts at |L1| > 0.4).

Test 2 surfaced: post-J13, 45% of YES weekday trades sit in L1 in -0.4..0
(price below the line) at ~45% WR, losing money. This script:

  1. Profiles YES trades by FINE L1 bins: n, WR, breakeven-WR (from entry
     price), pnl/trade, pnl_total. Finds where YES turns unprofitable.
  2. Simulates an L1-floor filter (require L1 >= theta for YES), sweeping
     theta, measuring PnL / WR / trades-kept on Bot K live trades.
  3. Checks which signal layer drives combined_signal positive in the
     leak zone (why does the bot enter at all?).
  4. Symmetric check: is there a mild-bullish-L1 NO leak (NO bet when
     price is above the line)?

Sample: Bot K trades from 2026-05-15 09:32 (ranging-fix, clean optimised
weights) onward. All days (not weekday-only) since a filter applies to all.

FIRST-ORDER ESTIMATE on live trades: summing removed-trade PnL is a fair
direct estimate of a "don't take these" filter (no capital redeployment
assumed). Validate on the backtest snapshot pipeline before any code.

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
CLEAN_START = pd.Timestamp("2026-05-15 09:32:00", tz="UTC")  # 46b7a2b ranging-fix
J13_DEPLOYED = pd.Timestamp("2026-05-21 20:37:00", tz="UTC")  # 9ef0523


def load() -> pd.DataFrame:
    con = sqlite3.connect(DB_BOT_K)
    df = pd.read_sql("""
        SELECT id, timestamp, side, pnl, edge, size_usdc, entry_price,
               time_remaining_secs, oracle_lag_signal, momentum_signal,
               liquidation_signal, orderbook_signal, sentiment_signal,
               combined_signal, regime
        FROM trades WHERE pnl IS NOT NULL
    """, con)
    con.close()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["ts"] >= CLEAN_START].copy()
    df["win"] = (df["pnl"] > 0).astype(int)
    df["era"] = np.where(df["ts"] < J13_DEPLOYED, "pre_J13", "post_J13")
    return df


def fine_bins(df_side: pd.DataFrame, side_label: str) -> pd.DataFrame:
    edges = [-1.01, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["<-0.6", "-0.6..-0.4", "-0.4..-0.2", "-0.2..0",
              "0..0.2", "0.2..0.4", "0.4..0.6", "0.6..0.8", "0.8..1.0"]
    b = df_side.copy()
    b["bin"] = pd.cut(b["oracle_lag_signal"], bins=edges, labels=labels, right=False)
    rows = []
    for lab in labels:
        cell = b[b["bin"] == lab]
        n = len(cell)
        if n == 0:
            rows.append({"L1_bin": lab, "n": 0})
            continue
        be_wr = 100 * cell["entry_price"].mean()  # breakeven WR ~ entry price
        wr = 100 * cell["win"].mean()
        rows.append({
            "L1_bin": lab, "n": n,
            "WR_%": round(wr, 1),
            "breakeven_WR_%": round(be_wr, 1),
            "WR_minus_BE": round(wr - be_wr, 1),
            "pnl_per_trade": round(cell["pnl"].mean(), 3),
            "pnl_total": round(cell["pnl"].sum(), 2),
            "mean_entry": round(cell["entry_price"].mean(), 3),
        })
    return pd.DataFrame(rows)


def sweep_filter(yes: pd.DataFrame) -> pd.DataFrame:
    """Require L1 >= theta for YES. Kept trades' PnL vs dropped."""
    total_pnl = yes["pnl"].sum()
    total_n = len(yes)
    rows = []
    for theta in [-0.6, -0.4, -0.2, 0.0, 0.1, 0.2, 0.3]:
        kept = yes[yes["oracle_lag_signal"] >= theta]
        dropped = yes[yes["oracle_lag_signal"] < theta]
        rows.append({
            "theta": theta,
            "kept_n": len(kept),
            "kept_%": round(100 * len(kept) / total_n, 1),
            "dropped_n": len(dropped),
            "dropped_pnl": round(dropped["pnl"].sum(), 2),
            "dropped_WR_%": round(100 * dropped["win"].mean(), 1) if len(dropped) else None,
            "kept_pnl": round(kept["pnl"].sum(), 2),
            "kept_WR_%": round(100 * kept["win"].mean(), 1) if len(kept) else None,
            "kept_pnl_per_trade": round(kept["pnl"].mean(), 3) if len(kept) else None,
            "pnl_gain_vs_nofilter": round(kept["pnl"].sum() - total_pnl, 2),
        })
    return pd.DataFrame(rows)


def layer_drivers(leak: pd.DataFrame) -> pd.DataFrame:
    """In the leak zone, what are the signal layers doing on average?"""
    cols = ["oracle_lag_signal", "momentum_signal", "liquidation_signal",
            "orderbook_signal", "sentiment_signal", "combined_signal"]
    rows = []
    for c in cols:
        s = leak[c]
        rows.append({
            "layer": c, "mean": round(s.mean(), 4),
            "share_positive_%": round(100 * (s > 0).mean(), 1),
            "p50": round(s.median(), 4),
        })
    return pd.DataFrame(rows)


def main() -> int:
    df = load()
    yes = df[df["side"] == "YES"].copy()
    no = df[df["side"] == "NO"].copy()

    print()
    print("=" * 84)
    print("L1 LEAK INVESTIGATION — Bot K, all days, from 2026-05-15 09:32")
    print("=" * 84)
    print(f"Total trades: {len(df)}  (YES={len(yes)}, NO={len(no)})")
    print(f"Window: {df['ts'].min()} -> {df['ts'].max()}")
    print("L1 = displacement from window-open line. >0 price above, <0 below.")
    print()

    print("=" * 84)
    print("(1) YES trades by fine L1 bin — where does YES turn unprofitable?")
    print("=" * 84)
    print("breakeven_WR ~ mean entry price; WR_minus_BE > 0 = profitable edge")
    print()
    print(fine_bins(yes, "YES").to_string(index=False))
    print()

    print("=" * 84)
    print("(2) FILTER SWEEP — require L1 >= theta for YES entries")
    print("=" * 84)
    print(f"No-filter YES baseline: n={len(yes)}, "
          f"pnl=${yes['pnl'].sum():.2f}, WR={100*yes['win'].mean():.1f}%, "
          f"${yes['pnl'].mean():.3f}/trade")
    print()
    print(sweep_filter(yes).to_string(index=False))
    print()

    print("=" * 84)
    print("(3) LEAK-ZONE LAYER DRIVERS — why does the bot enter YES at L1<0?")
    print("=" * 84)
    leak = yes[(yes["oracle_lag_signal"] >= -0.4) & (yes["oracle_lag_signal"] < 0)]
    print(f"Leak zone (YES, L1 in -0.4..0): n={len(leak)}, "
          f"WR={100*leak['win'].mean():.1f}%, ${leak['pnl'].mean():.3f}/trade, "
          f"total=${leak['pnl'].sum():.2f}")
    print()
    print(layer_drivers(leak).to_string(index=False))
    print()

    print("=" * 84)
    print("(4) SYMMETRIC CHECK — NO trades by fine L1 bin (mild-bullish NO leak?)")
    print("=" * 84)
    print("For NO (bet DOWN), the analogous leak is L1 > 0 (price above line).")
    print()
    print(fine_bins(no, "NO").to_string(index=False))
    print()

    print("=" * 84)
    print("(5) ERA SPLIT — is the NO-at-L1>0.4 leak pre-fix residual or a")
    print("    live post-J13 hole? (J13's whole job is to kill these.)")
    print("=" * 84)
    for era_name in ["pre_J13", "post_J13"]:
        sub = no[(no["era"] == era_name) & (no["oracle_lag_signal"] > 0.4)]
        days = max(1, (sub["ts"].max() - sub["ts"].min()).days + 1) if len(sub) else 1
        print(f"\n  NO, L1>0.4, {era_name}: n={len(sub)}, "
              f"WR={100*sub['win'].mean() if len(sub) else float('nan'):.1f}%, "
              f"pnl=${sub['pnl'].sum():.2f}, "
              f"${sub['pnl'].mean() if len(sub) else 0:.3f}/trade, "
              f"~${sub['pnl'].sum()/days:.2f}/day over {days}d")
    print()
    print("  Also: YES leak (L1 in -0.2..0) by era — should persist post-J13")
    print("  since J13 does not touch this zone:")
    for era_name in ["pre_J13", "post_J13"]:
        sub = yes[(yes["era"] == era_name)
                  & (yes["oracle_lag_signal"] >= -0.2)
                  & (yes["oracle_lag_signal"] < 0)]
        print(f"    YES L1[-0.2,0) {era_name}: n={len(sub)}, "
              f"WR={100*sub['win'].mean() if len(sub) else float('nan'):.1f}%, "
              f"pnl=${sub['pnl'].sum():.2f}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
