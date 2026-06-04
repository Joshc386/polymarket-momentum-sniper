"""
Base-rate audit for the J13 mild-L1 NO post-mortem.

Question: in 5-min BTC windows, what's the actual UP rate? If it's materially
above 50%, the combiner's symmetric prior (est_prob_up = 0.5 + signal*adj)
is mis-anchored and reads positive drift as "edge for NO".

Three slices:
  1. Full 2-year sample           -> anchor base rate, tight CI
  2. Rolling 90-day windows       -> regime stability check
  3. By UTC hour-of-day           -> does drift vary by trading window?

UP definition: close[t] > close[t-1] (consecutive 5-min closes).
Tied bars (close == previous close) are counted separately, not as UP.

Read-only. No code changes. No live data touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).parent / "data" / "binance_klines_5m.csv"


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(
        CSV,
        usecols=["open_time_utc", "close"],
        parse_dates=["open_time_utc"],
    )
    df = df.sort_values("open_time_utc").reset_index(drop=True)
    df["prev_close"] = df["close"].shift(1)
    df = df.dropna(subset=["prev_close"]).reset_index(drop=True)
    df["direction"] = np.where(
        df["close"] > df["prev_close"], "UP",
        np.where(df["close"] < df["prev_close"], "DOWN", "FLAT"),
    )
    return df


def ci95_pct(p: float, n: int) -> tuple[float, float]:
    """95% normal-approx CI for a proportion, returned in percentage points."""
    se = (p * (1.0 - p) / n) ** 0.5
    return (p - 1.96 * se) * 100, (p + 1.96 * se) * 100


def slice1_whole_sample(df: pd.DataFrame) -> None:
    n = len(df)
    up = (df["direction"] == "UP").sum()
    down = (df["direction"] == "DOWN").sum()
    flat = (df["direction"] == "FLAT").sum()

    p_up_all = up / n
    p_up_directional = up / (up + down)
    lo_all, hi_all = ci95_pct(p_up_all, n)
    lo_dir, hi_dir = ci95_pct(p_up_directional, up + down)

    print("=" * 70)
    print("SLICE 1 -- WHOLE SAMPLE (2024-04-10 to 2026-04-10)")
    print("=" * 70)
    print(f"  bars              : {n:,}")
    print(f"  date range        : {df['open_time_utc'].min()} to {df['open_time_utc'].max()}")
    print(f"  UP                : {up:,} ({100*up/n:.3f}%)")
    print(f"  DOWN              : {down:,} ({100*down/n:.3f}%)")
    print(f"  FLAT              : {flat:,} ({100*flat/n:.3f}%)")
    print()
    print(f"  P(UP) incl. flats : {100*p_up_all:.3f}%   95% CI [{lo_all:.3f}%, {hi_all:.3f}%]")
    print(f"  P(UP | not flat)  : {100*p_up_directional:.3f}%   95% CI [{lo_dir:.3f}%, {hi_dir:.3f}%]")
    print()
    drift = 100 * (p_up_directional - 0.5)
    print(f"  >>> directional drift from 50/50: {drift:+.3f} pp")
    print()


def slice2_rolling_90d(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("SLICE 2 -- ROLLING 90-DAY WINDOWS (regime stability)")
    print("=" * 70)
    df = df.set_index("open_time_utc")
    df["is_up"] = (df["direction"] == "UP").astype(int)
    df["is_directional"] = (df["direction"] != "FLAT").astype(int)

    # Resample to non-overlapping 90-day buckets for readability
    buckets = df.resample("90D")
    rows = []
    for ts, grp in buckets:
        n_total = len(grp)
        n_dir = int(grp["is_directional"].sum())
        n_up = int(grp["is_up"].sum())
        if n_dir == 0:
            continue
        p_up = n_up / n_dir
        lo, hi = ci95_pct(p_up, n_dir)
        rows.append({
            "window_start": ts.date(),
            "bars": n_total,
            "P(UP|dir)_%": round(100 * p_up, 3),
            "ci_lo_%": round(lo, 3),
            "ci_hi_%": round(hi, 3),
            "drift_pp": round(100 * (p_up - 0.5), 3),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print()
    print(f"  mean drift across windows : {out['drift_pp'].mean():+.3f} pp")
    print(f"  stdev across windows      : {out['drift_pp'].std():.3f} pp")
    print(f"  min / max drift           : {out['drift_pp'].min():+.3f} pp / {out['drift_pp'].max():+.3f} pp")
    print()


def slice3_by_hour(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("SLICE 3 -- BY UTC HOUR-OF-DAY")
    print("=" * 70)
    df = df.copy()
    df["hour"] = df["open_time_utc"].dt.hour
    df["is_up"] = (df["direction"] == "UP").astype(int)
    df["is_directional"] = (df["direction"] != "FLAT").astype(int)

    rows = []
    for h, grp in df.groupby("hour"):
        n_dir = int(grp["is_directional"].sum())
        n_up = int(grp["is_up"].sum())
        p_up = n_up / n_dir
        lo, hi = ci95_pct(p_up, n_dir)
        rows.append({
            "hour_utc": h,
            "n_directional": n_dir,
            "P(UP|dir)_%": round(100 * p_up, 3),
            "ci_lo_%": round(lo, 3),
            "ci_hi_%": round(hi, 3),
            "drift_pp": round(100 * (p_up - 0.5), 3),
        })
    out = pd.DataFrame(rows).sort_values("hour_utc")
    print(out.to_string(index=False))
    print()
    print(f"  hour-of-day drift range : "
          f"{out['drift_pp'].min():+.3f} pp to {out['drift_pp'].max():+.3f} pp")
    print(f"  hour-of-day drift stdev : {out['drift_pp'].std():.3f} pp")
    print()


def main() -> int:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found", file=sys.stderr)
        return 1
    df = load_klines()
    slice1_whole_sample(df)
    slice2_rolling_90d(df)
    slice3_by_hour(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
