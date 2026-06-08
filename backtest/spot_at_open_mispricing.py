"""Spot-at-open mispricing test — is there an edge when current price ≈ open?

Hypothesis (user): on a 5-min BTC Up/Down market, whenever the spot price is back
within ±$10 of the window's opening price, the true outcome is ~50/50 regardless of
the path taken to get there. If the market prices a side away from 0.5 in that
moment, the underpriced side carries positive EV.

Counter-view (efficient market): 5-min BTC returns are NOT a pure martingale; the
market price away from 0.5 reflects real serial correlation (momentum/reversion),
so realized P(Up) tracks the price, not a flat 0.5.

This script settles it on GROUND-TRUTH resolved windows. Read-only: it loads two
CSVs, joins them, and writes charts + a scorecard. Nothing is written to any DB and
no bot code is touched.

Data (Mar–Apr era; the only self-contained source with fillable asks + outcomes):
  backtest/data/polybacktest_snapshots.csv  — intra-window snapshots (spot, asks)
  backtest/data/polybacktest_markets.csv    — per-window open price + winner

Outputs:
  backtest/spot_at_open_test/calibration_at_open.png   <- THE verdict chart
  backtest/spot_at_open_test/ev_by_friction.png
  backtest/spot_at_open_test/scorecard.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "backtest" / "data"
OUT = REPO / "backtest" / "spot_at_open_test"
OUT.mkdir(exist_ok=True)

# --- test parameters (user spec) ------------------------------------------- #
DEV_THRESHOLD = 10.0        # |spot - open| <= $10 counts as "at the open"
FAIR = 0.5                  # naive fair value when spot == open
# friction model, pulled from config.yaml (live-bot assumptions):
FEE = 0.02                  # fee_adjustment — "Realistic Polymarket fee (~2%)" per share
SLIPPAGE = 0.005            # fok_slippage on a taker fill


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, centre - half, centre + half)


def load() -> pd.DataFrame:
    """Join intra-window snapshots to their window metadata, restricted to
    moments where spot is within DEV_THRESHOLD of the window open."""
    snaps = pd.read_csv(
        DATA / "polybacktest_snapshots.csv",
        usecols=["market_id", "offset_sec", "seconds_remaining", "snapshot_time",
                 "btc_price", "up_best_ask", "down_best_ask"],
    )
    mkts = pd.read_csv(
        DATA / "polybacktest_markets.csv",
        usecols=["market_id", "start_time", "btc_price_start", "winner"],
    )
    df = snaps.merge(mkts, on="market_id", how="inner")
    df = df.dropna(subset=["btc_price", "btc_price_start", "winner",
                           "up_best_ask", "down_best_ask"])
    df = df[df["winner"].isin(["Up", "Down"])].copy()
    df["dev"] = df["btc_price"] - df["btc_price_start"]
    df["y_up"] = (df["winner"] == "Up").astype(int)
    df["ts"] = pd.to_datetime(df["snapshot_time"], utc=True, format="ISO8601")
    df["day"] = df["ts"].dt.date
    near = df[df["dev"].abs() <= DEV_THRESHOLD].copy()
    return near.sort_values("ts").reset_index(drop=True)


# --------------------------------------------------------------------------- #
def plot_calibration(df: pd.DataFrame) -> list[str]:
    """The verdict chart: among 'spot ≈ open' moments, does realized P(Up)
    sit on a flat 0.5 line (naive view) or on the price diagonal (efficient)?"""
    p = df["up_best_ask"].to_numpy()
    y = df["y_up"].to_numpy()
    edges = np.linspace(max(0.0, p.min()), min(1.0, p.max()), 11)
    xs, ph, lo, hi, ns = [], [], [], [], []
    for b in range(len(edges) - 1):
        m = (p >= edges[b]) & (p < edges[b + 1]) if b < len(edges) - 2 else \
            (p >= edges[b]) & (p <= edges[b + 1])
        n = int(m.sum())
        if n < 30:
            continue
        prop, l, h = wilson(int(y[m].sum()), n)
        xs.append(p[m].mean()); ph.append(prop); lo.append(l); hi.append(h); ns.append(n)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0.2, 0.8], [0.2, 0.8], "k--", lw=1, label="efficient market (realized = price)")
    ax.axhline(FAIR, color="tab:green", lw=1.2, ls=":", label="naive view (flat 0.5)")
    ax.errorbar(xs, ph,
                yerr=[np.array(ph) - np.array(lo), np.array(hi) - np.array(ph)],
                fmt="o-", color="tab:blue", capsize=3, label="realized P(Up)")
    for x, yv, n in zip(xs, ph, ns):
        ax.annotate(f"n={n}", (x, yv), textcoords="offset points", xytext=(4, 6),
                    fontsize=7, color="grey")
    ax.set_xlabel("Market price to buy UP (up_best_ask)")
    ax.set_ylabel("Realized P(Up)  (Wilson 95% CI)")
    ax.set_title(f"Calibration when |spot − open| ≤ ${DEV_THRESHOLD:.0f}\n"
                 f"N={len(df)} snapshots — dots on dashed = market right, "
                 f"dots on green = mispriced")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "calibration_at_open.png", dpi=130)
    plt.close(fig)

    L = ["  -- conditional calibration (realized P(Up) by UP price) --"]
    for x, yv, n in zip(xs, ph, ns):
        verdict = "mispriced↑" if yv - x > 0.03 else ("mispriced↓" if x - yv > 0.03 else "fair")
        L.append(f"     price~{x:.3f}  realized={yv:.3f}  n={n:5d}   {verdict}")
    return L


def simulate(df: pd.DataFrame, fee: float, slip: float) -> dict:
    """Strategy: buy whichever side is priced below fair 0.5 (the 'underpriced'
    side per the hypothesis), pay ask + slippage + fee, hold to resolution."""
    up_a = df["up_best_ask"].to_numpy()
    dn_a = df["down_best_ask"].to_numpy()
    y_up = df["y_up"].to_numpy()

    pick_up = (up_a < FAIR) & ((dn_a >= FAIR) | (up_a <= dn_a))
    pick_dn = (dn_a < FAIR) & ((up_a >= FAIR) | (dn_a < up_a))
    traded = pick_up | pick_dn

    entry = np.where(pick_up, up_a, np.where(pick_dn, dn_a, np.nan))
    won = np.where(pick_up, y_up == 1, np.where(pick_dn, y_up == 0, False))
    cost = entry + slip + fee
    pnl = np.where(won, 1.0 - cost, -cost)
    pnl = np.where(traded, pnl, np.nan)

    d = df.assign(pnl=pnl, traded=traded)
    t = d[d["traded"]].copy()
    n = len(t)
    if n == 0:
        return {"n": 0}
    daily = t.groupby("day")["pnl"].sum()
    dse = daily.std(ddof=1) / np.sqrt(len(daily)) if len(daily) > 1 else np.nan
    # window-clustered: one mean PnL per market, then t over windows
    win = t.groupby("market_id")["pnl"].mean()
    wse = win.std(ddof=1) / np.sqrt(len(win)) if len(win) > 1 else np.nan
    return {
        "n": n,
        "mean": t["pnl"].mean(),
        "total": t["pnl"].sum(),
        "winrate": float(won[traded].mean()),
        "daily_mean": daily.mean(),
        "daily_t": daily.mean() / dse if dse and not np.isnan(dse) else np.nan,
        "n_days": len(daily),
        "pos_days": int((daily > 0).sum()),
        "win_t": win.mean() / wse if wse and not np.isnan(wse) else np.nan,
        "n_windows": len(win),
    }


def plot_ev_by_friction(rows: list[tuple[str, dict]]) -> None:
    labels = [r[0] for r in rows if r[1].get("n")]
    means = [r[1]["mean"] for r in rows if r[1].get("n")]
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = ["tab:green" if m > 0 else "tab:red" for m in means]
    ax.bar(labels, means, color=colours)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Mean EV per trade ($/share)")
    ax.set_title("Underpriced-side strategy EV vs friction level")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "ev_by_friction.png", dpi=130)
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = load()
    L = [
        "SPOT-AT-OPEN MISPRICING TEST  (read-only, ground-truth resolved windows)",
        f"data era: {df['day'].min()} .. {df['day'].max()}   "
        f"threshold |spot-open| <= ${DEV_THRESHOLD:.0f}",
        "=" * 72,
        f"  qualifying snapshots : {len(df)}   across {df['market_id'].nunique()} windows"
        f"  / {df['day'].nunique()} days",
        f"  realized P(Up) here  : {df['y_up'].mean():.4f}   "
        f"(naive view says this should be ~0.5000)",
        f"  mean UP ask / DOWN ask: {df['up_best_ask'].mean():.4f} / "
        f"{df['down_best_ask'].mean():.4f}   (mean implied, both sides)",
    ]
    L += plot_calibration(df)

    # robustness cut: one (first) qualifying snapshot per window
    first = df.sort_values("offset_sec").drop_duplicates("market_id", keep="first")

    L += ["", "  -- underpriced-side strategy (buy the side priced < 0.5) --"]
    friction_rows = []
    for tag, fee, slip in [("gross (no cost)", 0.0, 0.0),
                           ("slippage only", 0.0, SLIPPAGE),
                           ("full (fee+slip)", FEE, SLIPPAGE)]:
        r = simulate(df, fee, slip)
        friction_rows.append((tag, r))
        if r.get("n"):
            L.append(
                f"   [{tag:16s}] N={r['n']:6d}  win%={r['winrate']:.3f}  "
                f"EV/trade={r['mean']:+.4f}  total={r['total']:+.0f}")
            L.append(
                f"   {'':18s} day-clustered t={r['daily_t']:.2f} "
                f"({r['pos_days']}/{r['n_days']} +days)   "
                f"window-clustered t={r['win_t']:.2f} (N={r['n_windows']})  <- honest")

    rf = simulate(first, FEE, SLIPPAGE)
    if rf.get("n"):
        L += ["", "  -- robustness: first qualifying snapshot per window, full friction --",
              f"   N={rf['n']:6d}  win%={rf['winrate']:.3f}  EV/trade={rf['mean']:+.4f}  "
              f"total={rf['total']:+.0f}  day-t={rf['daily_t']:.2f} "
              f"({rf['pos_days']}/{rf['n_days']} +days)"]

    plot_ev_by_friction(friction_rows)

    L += [
        "", "  CAVEATS:",
        "   - Mar–Apr era; predates the current net_ev config (tests the MARKET, not the bot).",
        "   - spot & open are Binance prices; settlement oracle may differ (see signal_ticks",
        "     oracle_open_price for a recency/reference cross-check if an edge shows here).",
        "   - 'any moment' => many correlated snapshots per window; trust the window/day-",
        "     clustered t-stats, not raw N.",
    ]
    text = "\n".join(L)
    print(text)
    (OUT / "scorecard.txt").write_text(text, encoding="utf-8")
    print(f"\nCharts + scorecard written to {OUT}")


if __name__ == "__main__":
    main()
