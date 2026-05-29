"""
Test 4 (handoff 26-05-28): Bot K pre-J13 OOS-only restriction.

Hypothesis: The -$40/day Bot-K-specific YES weekday degradation post-J13
is OOS drift on the walk-forward-optimised weights (train end 2026-05-11),
NOT a J13 cost on the YES path.

Method: Split Bot K trades into three windows:
  - IN-SAMPLE pre-J13:   2026-05-09 to 2026-05-11  (optimiser saw these days)
  - OOS pre-J13:         2026-05-12 to 2026-05-21  (post-train, pre-J13)
  - OOS post-J13:        2026-05-22 to 2026-05-28  (post-train, post-J13)

Compare YES weekday cell PnL/trade across (OOS pre-J13) vs (OOS post-J13).
If the gap is similar to (full pre-J13) vs (post-J13), OOS drift is NOT
the dominant driver. If the gap shrinks substantially, OOS drift confirmed.

Bot G included as regime control. Bot G uses DEFAULT weights (no
walk-forward optimisation per config_multi.yaml:406-494), so Bot G has
no in-sample/OOS concern of its own — it's pure regime control.

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

# Precise commit-derived boundaries (verified via `git log` 2026-05-28).
# Before BOT_K_WEIGHTS_DEPLOYED: Bot K ran DEFAULT weights = Bot G's setup.
# Between BOT_K_WEIGHTS_DEPLOYED and RANGING_FIX: Bot K's optimised schedule
#   was clobbered by the ranging-regime override (decision J11). Treat as
#   "mixed" — neither pure default nor pure optimised.
# Between RANGING_FIX and J13_DEPLOYED: Bot K running PURE optimised
#   weights, pre-J13. This is the cleanest "Bot K-with-optimised-weights,
#   pre-J13" baseline.
# After J13_DEPLOYED: Bot K running optimised weights + J13 gating.
BOT_K_WEIGHTS_DEPLOYED = pd.Timestamp("2026-05-12 12:52:00", tz="UTC")  # 40ed014
RANGING_FIX            = pd.Timestamp("2026-05-15 09:32:00", tz="UTC")  # 46b7a2b
J13_DEPLOYED           = pd.Timestamp("2026-05-21 20:37:00", tz="UTC")  # 9ef0523


def load(db_path: Path) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT id, timestamp, side, pnl, oracle_lag_signal
        FROM trades WHERE pnl IS NOT NULL
    """, con)
    con.close()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["weekend"] = df["ts"].dt.dayofweek >= 5
    df["win"] = (df["pnl"] > 0).astype(int)
    return df


def label_window(ts: pd.Series) -> pd.Series:
    out = pd.Series("unknown", index=ts.index, dtype=object)
    out[ts < BOT_K_WEIGHTS_DEPLOYED] = "01_default_weights"
    out[(ts >= BOT_K_WEIGHTS_DEPLOYED) & (ts < RANGING_FIX)] = "02_mixed_ranging_clobber"
    out[(ts >= RANGING_FIX) & (ts < J13_DEPLOYED)] = "03_pure_optimised_pre_J13"
    out[ts >= J13_DEPLOYED] = "04_pure_optimised_post_J13"
    return out


def cell(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {"n": 0, "WR_%": None, "pnl_total": 0.0,
                "pnl_per_trade": 0.0, "pnl_per_day": 0.0, "days": 0}
    days = max(1, (sub["ts"].max() - sub["ts"].min()).days + 1)
    return {
        "n": n,
        "WR_%": round(100 * sub["win"].mean(), 1),
        "pnl_total": round(sub["pnl"].sum(), 2),
        "pnl_per_trade": round(sub["pnl"].mean(), 3),
        "pnl_per_day": round(sub["pnl"].sum() / days, 2),
        "days": days,
    }


WINDOWS = ["01_default_weights", "02_mixed_ranging_clobber",
           "03_pure_optimised_pre_J13", "04_pure_optimised_post_J13"]


def yes_weekday_table(bot_name: str, df: pd.DataFrame) -> pd.DataFrame:
    yes_wkdy = df[(df["side"] == "YES") & (~df["weekend"])].copy()
    yes_wkdy["window"] = label_window(yes_wkdy["ts"])
    rows = []
    for w in WINDOWS:
        rows.append({"bot": bot_name, "window": w, **cell(yes_wkdy[yes_wkdy["window"] == w])})
    return pd.DataFrame(rows)


def full_side_weekend_table(bot_name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["window"] = label_window(df["ts"])
    rows = []
    for side in ["YES", "NO"]:
        for wknd in [False, True]:
            for w in WINDOWS:
                sub = df[(df["side"] == side) & (df["weekend"] == wknd)
                         & (df["window"] == w)]
                rows.append({
                    "bot": bot_name, "side": side,
                    "wk": "wknd" if wknd else "wkdy",
                    "window": w, **cell(sub),
                })
    return pd.DataFrame(rows)


def main() -> int:
    print()
    bot_k = load(DB_BOT_K)
    bot_g_full = load(DB_BOT_G)
    # Restrict Bot G to Bot K's window for fair regime comparison.
    bot_g = bot_g_full[(bot_g_full["ts"] >= bot_k["ts"].min())
                       & (bot_g_full["ts"] <= bot_k["ts"].max())].copy()

    print(f"Bot K trades: {len(bot_k)} ({bot_k['ts'].min()} - {bot_k['ts'].max()})")
    print(f"Bot G trades in K's window: {len(bot_g)}")
    print(f"Bot K weights deployed (40ed014): {BOT_K_WEIGHTS_DEPLOYED}")
    print(f"Ranging-fix (46b7a2b):             {RANGING_FIX}")
    print(f"J13 deployed (9ef0523):            {J13_DEPLOYED}")
    print()

    print("=" * 84)
    print("YES WEEKDAY — the cell user flagged as alarming")
    print("=" * 84)
    print()
    yk = yes_weekday_table("Bot K", bot_k)
    yg = yes_weekday_table("Bot G", bot_g)
    print("Bot K YES weekday by window:")
    print(yk.to_string(index=False))
    print()
    print("Bot G YES weekday by window (regime control — default weights):")
    print(yg.to_string(index=False))
    print()

    print("=" * 84)
    print("KEY COMPARISONS — Test 4 verdict")
    print("=" * 84)
    print()

    def get_ppt(tbl, win):
        row = tbl[tbl["window"] == win].iloc[0]
        return row["pnl_per_trade"], row["n"], row["pnl_per_day"]

    k_pre_ppt, k_pre_n, k_pre_ppd   = get_ppt(yk, "03_pure_optimised_pre_J13")
    k_post_ppt, k_post_n, k_post_ppd = get_ppt(yk, "04_pure_optimised_post_J13")
    g_pre_ppt, g_pre_n, g_pre_ppd    = get_ppt(yg, "03_pure_optimised_pre_J13")
    g_post_ppt, g_post_n, g_post_ppd = get_ppt(yg, "04_pure_optimised_post_J13")

    print(f"  Bot K YES wkdy PURE pre-J13 (May 15 09:32 - May 21 20:37):")
    print(f"    ${k_pre_ppt:+.3f}/trade   n={k_pre_n}   ${k_pre_ppd:+.2f}/day")
    print(f"  Bot K YES wkdy POST-J13 (May 21 20:37 - May 28):")
    print(f"    ${k_post_ppt:+.3f}/trade   n={k_post_n}   ${k_post_ppd:+.2f}/day")
    print()
    print(f"  Bot G YES wkdy SAME pre-J13 window:")
    print(f"    ${g_pre_ppt:+.3f}/trade   n={g_pre_n}   ${g_pre_ppd:+.2f}/day")
    print(f"  Bot G YES wkdy SAME post-J13 window:")
    print(f"    ${g_post_ppt:+.3f}/trade   n={g_post_n}   ${g_post_ppd:+.2f}/day")
    print()

    k_oos_delta_ppt = k_post_ppt - k_pre_ppt
    g_oos_delta_ppt = g_post_ppt - g_pre_ppt
    bot_k_specific_oos_ppt = k_oos_delta_ppt - g_oos_delta_ppt

    k_oos_delta_ppd = k_post_ppd - k_pre_ppd
    g_oos_delta_ppd = g_post_ppd - g_pre_ppd
    bot_k_specific_oos_ppd = k_oos_delta_ppd - g_oos_delta_ppd

    print(f"  Bot K Δ (OOS post − OOS pre):  ${k_oos_delta_ppt:+.3f}/trade   "
          f"${k_oos_delta_ppd:+.2f}/day")
    print(f"  Bot G Δ (OOS post − OOS pre):  ${g_oos_delta_ppt:+.3f}/trade   "
          f"${g_oos_delta_ppd:+.2f}/day")
    print(f"  Bot K-specific Δ on pure OOS:  ${bot_k_specific_oos_ppt:+.3f}/trade   "
          f"${bot_k_specific_oos_ppd:+.2f}/day")
    print()
    print("  Reference from prior session (full pre-J13 incl. in-sample days):")
    print("    Bot K-specific Δ: ~-$40/day, ~-$0.76/trade (the alarming number)")
    print()
    print("  If pure-OOS Bot K-specific Δ is materially smaller in magnitude")
    print("  (e.g. shrinks toward zero or Bot G's level), OOS drift on Bot K's")
    print("  optimised weights is the dominant driver — NOT J13.")
    print("  If it's similar magnitude (-$30 to -$40/day), OOS drift is NOT")
    print("  the explanation and J13 implication stands.")
    print()

    print("=" * 84)
    print("FULL SIDE × WEEKEND × WINDOW DECOMPOSITION (for the record)")
    print("=" * 84)
    print()
    print("Bot K:")
    print(full_side_weekend_table("Bot K", bot_k).to_string(index=False))
    print()
    print("Bot G (regime control, restricted to K's window):")
    print(full_side_weekend_table("Bot G", bot_g).to_string(index=False))
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
