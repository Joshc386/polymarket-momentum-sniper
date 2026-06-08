"""Edge validation for the LIVE bots — is the paper-trading edge real?

Read-only diagnostic over GROUND-TRUTH resolved trades (no outcome reconstruction:
every trade carries its real resolution and PnL). For each bot it answers:

  1. Is realized PnL significantly positive once we de-correlate within-day trades?
  2. Does the bot's est_prob_up beat the market-implied price (Brier)?  i.e. does the
     signal engine add information, or would "just trust the order book" do as well?
  3. Where does the edge live / where does it leak (regime, config era, calibration)?

Targets the two currently-live bots (Bot K, Bot K2).

Outputs (read-only — nothing written back to any DB):
  backtest/edge_validation/<bot>/calibration.png
  backtest/edge_validation/<bot>/edge_test.png
  backtest/edge_validation/<bot>/equity.png
  backtest/edge_validation/scorecard.txt
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "data_runtime"
OUT = REPO / "backtest" / "edge_validation"
OUT.mkdir(exist_ok=True)

# the two currently-live bots
BOTS = [
    ("bot_k_sm_confirmation", "Bot K (SM confirmation) [LIVE]"),
    ("bot_k2_l1_floor", "Bot K2 (L1 floor) [LIVE]"),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, centre - half, centre + half)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def load(bot: str) -> pd.DataFrame:
    with sqlite3.connect(RUNTIME / f"{bot}.db") as c:
        df = pd.read_sql_query(
            "SELECT timestamp, estimated_prob_up, market_implied_prob, resolution, "
            "side, pnl, size_usdc, regime, net_ev_per_share FROM trades "
            "WHERE resolution IN ('UP','DOWN') AND pnl IS NOT NULL",
            c,
        )
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["day"] = df["ts"].dt.date
    df["y"] = (df["resolution"] == "UP").astype(int)
    return df.sort_values("ts").reset_index(drop=True)


# --------------------------------------------------------------------------- #
def reliability(p, y, bins=10):
    edges = np.linspace(p.min(), p.max(), bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        if n < 10:
            continue
        ph, lo, hi = wilson(int(y[m].sum()), n)
        out.append((float(p[m].mean()), ph, lo, hi, n))
    return out


def plot_calibration(df, name, slug):
    y = df["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0.3, 0.7], [0.3, 0.7], "k--", lw=1, label="perfect")
    for col, colour, lbl in (
        ("estimated_prob_up", "tab:blue", "Bot est_prob_up"),
        ("market_implied_prob", "tab:orange", "Market implied"),
    ):
        rows = reliability(df[col].to_numpy(), y)
        if not rows:
            continue
        xs, ph, lo, hi, ns = zip(*rows)
        ax.errorbar(xs, ph,
                    yerr=[np.array(ph) - np.array(lo), np.array(hi) - np.array(ph)],
                    fmt="o-", color=colour, capsize=3, label=lbl)
    ax.set_xlabel("Predicted P(UP)")
    ax.set_ylabel("Realized P(UP)  (Wilson 95% CI)")
    ax.set_title(f"Calibration — {name}\nN={len(df)} resolved trades")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / slug / "calibration.png", dpi=130)
    plt.close(fig)


def plot_edge_test(df, name, slug):
    d = df.copy()
    d["edge"] = d["estimated_prob_up"] - d["market_implied_prob"]
    try:
        d["bin"] = pd.qcut(d["edge"], 8, duplicates="drop")
    except ValueError:
        return
    xs, real, lo, hi, imp = [], [], [], [], []
    for _, g in d.groupby("bin", observed=True):
        ph, l, h = wilson(int(g["y"].sum()), len(g))
        xs.append(g["edge"].mean()); real.append(ph); lo.append(l); hi.append(h)
        imp.append(g["market_implied_prob"].mean())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(xs, real,
                yerr=[np.array(real) - np.array(lo), np.array(hi) - np.array(real)],
                fmt="o-", color="tab:blue", capsize=3, label="Realized P(UP)")
    ax.plot(xs, imp, "s--", color="tab:orange", label="Market implied")
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel("Bot edge claim (est − market implied)")
    ax.set_ylabel("P(UP)")
    ax.set_title(f"Edge test — {name}\nblue above orange on the right = real edge")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / slug / "edge_test.png", dpi=130)
    plt.close(fig)


def plot_equity(df, name, slug):
    eq = df["pnl"].cumsum().to_numpy()
    peak = np.maximum.accumulate(eq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["ts"], eq, color="tab:green", lw=1.5, label="cumulative PnL")
    ax.fill_between(df["ts"], eq, peak, color="tab:red", alpha=0.2, label="drawdown")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title(f"Equity curve — {name}\ntotal={eq[-1]:+.0f}  maxDD={(eq-peak).min():.0f}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / slug / "equity.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def scorecard(df, name) -> str:
    L = [f"\n{'='*68}\n{name}", "-" * 68]
    n = len(df)
    pnl = df["pnl"].to_numpy()
    se = pnl.std(ddof=1) / np.sqrt(n)
    daily = df.groupby("day")["pnl"].sum()
    dse = daily.std(ddof=1) / np.sqrt(len(daily))
    roi = 100 * pnl.sum() / df["size_usdc"].sum() if df["size_usdc"].sum() else float("nan")
    eq = pnl.cumsum()
    dd = (eq - np.maximum.accumulate(eq)).min()
    bm = brier(df["estimated_prob_up"].to_numpy(), df["y"].to_numpy())
    bk = brier(df["market_implied_prob"].to_numpy(), df["y"].to_numpy())
    L += [
        f"  resolved trades : {n}   ({df['timestamp'].min()[:10]} .. {df['timestamp'].max()[:10]})",
        f"  total PnL       : {pnl.sum():+.2f}   ROI {roi:+.2f}%   win% {(pnl>0).mean():.3f}",
        f"  mean PnL/trade  : {pnl.mean():+.4f}  t={pnl.mean()/se:.2f}  (trade-level, inflated)",
        f"  DAILY PnL       : {daily.mean():+.2f}/day  t={daily.mean()/dse:.2f}  "
        f"({(daily>0).sum()}/{len(daily)} positive days)  <- honest significance",
        f"  max drawdown    : {dd:.0f}   return/DD {abs(eq[-1]/dd):.1f}x" if dd < 0 else
        f"  max drawdown    : none",
        f"  Brier model/mkt : {bm:.4f} / {bk:.4f}   "
        f"=> {'MODEL beats market (edge)' if bm < bk else 'market beats model (no edge)'}",
        f"  realized P(UP)  : {df['y'].mean():.3f}   est {df['estimated_prob_up'].mean():.3f}"
        f"   mkt {df['market_implied_prob'].mean():.3f}",
    ]
    # config era split
    if df["net_ev_per_share"].notna().any() and df["net_ev_per_share"].isna().any():
        L.append("  -- config eras --")
        for era, g in df.groupby(df["net_ev_per_share"].notna()):
            s = g["pnl"].std(ddof=1) / np.sqrt(len(g))
            L.append(f"     net_ev {'present' if era else 'absent '}: N={len(g):4d}"
                     f"  mean={g['pnl'].mean():+.3f} t={g['pnl'].mean()/s:.2f}"
                     f"  total={g['pnl'].sum():+.0f}")
    return "\n".join(L)


def main():
    report = ["EDGE VALIDATION — live K bots (Bot K, Bot K2)",
              "(ground-truth resolved trades; read-only)"]
    for slug, name in BOTS:
        (OUT / slug).mkdir(exist_ok=True)
        df = load(slug)
        plot_calibration(df, name, slug)
        plot_edge_test(df, name, slug)
        plot_equity(df, name, slug)
        report.append(scorecard(df, name))
    text = "\n".join(report)
    print(text)
    (OUT / "scorecard.txt").write_text(text, encoding="utf-8")
    print(f"\nCharts + scorecard written to {OUT}")


if __name__ == "__main__":
    main()
