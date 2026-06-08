"""Regime Monitor Agent — daily L1 / volatility / regime drift watch on the LIVE bots.

The live trades already carry per-trade tags (oracle_lag_signal = L1, regime,
btc_price_at_entry). This agent AGGREGATES those tags so a regime shift becomes
visible *before* it masquerades as signal degradation — the exact failure mode
behind the late-May YES scare (a profitable strong-bullish-L1 regime quietly
evaporated and PnL followed).

Read-only: reads data_runtime/<bot>.db, writes a dated scorecard + CSVs to
data_runtime/regime_monitor/. Nothing is written back to any bot DB.

Phase 1 (profile)  — per-day L1 distribution, BTC level/vol, regime mix.
Phase 2 (pnl)      — win% / PnL bucketed by L1 zone and by regime label.
(Phase 3 drift alarm is a planned follow-up.)

Commands:
    python -m agents.regime_monitor profile [--bot NAME] [--days N]
    python -m agents.regime_monitor pnl     [--bot NAME] [--days N]
    python -m agents.regime_monitor report  [--bot NAME] [--days N]   (profile + pnl)
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.common import RUNTIME_DIR, format_table

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("agents.regime_monitor")

LIVE_BOTS = ["bot_k_sm_confirmation", "bot_k2_l1_floor"]
OUT = RUNTIME_DIR / "regime_monitor"

# L1 zones — thresholds echo the J13 / to-do language. Symmetric:
# |L1| >= 0.4 = strong, 0.1 <= |L1| < 0.4 = mild, |L1| < 0.1 = neutral.
L1_ZONE_ORDER = ["strong_bear", "mild_bear", "neutral", "mild_bull", "strong_bull"]
STRONG = 0.4
MILD = 0.1


# --------------------------------------------------------------------------- #
# Pure functions (regression-tested in tests/test_regime_monitor.py)
# --------------------------------------------------------------------------- #
def l1_bucket(value: float) -> str:
    """Map an L1 (oracle_lag) value in [-1, 1] to its directional zone.

    Symmetric boundaries: |L1| >= 0.4 strong, 0.1 <= |L1| < 0.4 mild, else neutral.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    if value <= -STRONG:
        return "strong_bear"
    if value <= -MILD:
        return "mild_bear"
    if value < MILD:
        return "neutral"
    if value < STRONG:
        return "mild_bull"
    return "strong_bull"


def load_trades(bot: str) -> pd.DataFrame:
    """Read a live bot's trades with the columns the monitor needs."""
    db = RUNTIME_DIR / f"{bot}.db"
    if not db.exists():
        raise FileNotFoundError(f"Bot DB not found: {db}")
    with sqlite3.connect(str(db)) as c:
        df = pd.read_sql_query(
            "SELECT timestamp, oracle_lag_signal, regime, btc_price_at_entry, "
            "side, resolution, pnl, is_paper FROM trades",
            c,
        )
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["day"] = df["ts"].dt.date
    df["l1"] = pd.to_numeric(df["oracle_lag_signal"], errors="coerce")
    df["l1_zone"] = df["l1"].map(l1_bucket)
    return df.sort_values("ts").reset_index(drop=True)


def daily_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day regime profile: L1 distribution, BTC level/vol, regime mix."""
    rows = []
    for day, g in df.groupby("day"):
        l1 = g["l1"].dropna().to_numpy()
        btc = g["btc_price_at_entry"].dropna().to_numpy()
        regime_mix = g["regime"].value_counts(normalize=True)
        dom = regime_mix.index[0] if len(regime_mix) else "n/a"
        rows.append({
            "day": day,
            "n": len(g),
            "l1_mean": round(float(l1.mean()), 3) if l1.size else np.nan,
            "l1_std": round(float(l1.std(ddof=0)), 3) if l1.size else np.nan,
            "pct_bull": round(float((l1 > 0).mean()), 3) if l1.size else np.nan,
            "pct_bear": round(float((l1 < 0).mean()), 3) if l1.size else np.nan,
            "l1_p10": round(float(np.percentile(l1, 10)), 3) if l1.size else np.nan,
            "l1_p50": round(float(np.percentile(l1, 50)), 3) if l1.size else np.nan,
            "l1_p90": round(float(np.percentile(l1, 90)), 3) if l1.size else np.nan,
            "btc_mean": round(float(btc.mean()), 0) if btc.size else np.nan,
            "btc_range_pct": round(float((btc.max() - btc.min()) / btc.mean() * 100), 2)
            if btc.size and btc.mean() else np.nan,
            "dom_regime": dom,
            "dom_regime_pct": round(float(regime_mix.iloc[0]), 2) if len(regime_mix) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


def regime_pnl(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Win% / PnL aggregated by a grouping column over RESOLVED trades only.

    Args:
        by: 'l1_zone' or 'regime'.
    """
    res = df[df["resolution"].isin(["UP", "DOWN"]) & df["pnl"].notna()].copy()
    rows = []
    for key, g in res.groupby(by):
        pnl = g["pnl"].to_numpy()
        rows.append({
            by: key,
            "n": len(g),
            "win_pct": round(float((pnl > 0).mean()), 3),
            "mean_pnl": round(float(pnl.mean()), 4),
            "total_pnl": round(float(pnl.sum()), 1),
        })
    out = pd.DataFrame(rows)
    if by == "l1_zone" and not out.empty:
        order = {name: i for i, name in enumerate(L1_ZONE_ORDER)}
        out["_o"] = out[by].map(lambda k: order.get(k, 99))
        out = out.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    else:
        out = out.sort_values("total_pnl", ascending=False).reset_index(drop=True)
    return out


def _window(df: pd.DataFrame, days: int | None) -> pd.DataFrame:
    if not days:
        return df
    cutoff = df["ts"].max() - pd.Timedelta(days=days)
    return df[df["ts"] >= cutoff].copy()


# --------------------------------------------------------------------------- #
# Phase 3 — drift detection (recent window vs trailing baseline)
# --------------------------------------------------------------------------- #
RECENT_DAYS = 3
BASELINE_DAYS = 14
PSI_ALERT = 0.25        # >0.25 = major distribution shift (standard PSI band)
L1Z_ALERT = 2.0         # recent L1 mean this many baseline-daily-std from baseline
PCTBULL_ALERT = 0.20    # absolute shift in bullish fraction
VOL_RATIO_ALERT = 1.5   # recent vol >= 1.5x baseline (or <= 1/1.5x)
EDGE_DROP_ALERT = 0.40  # recent expected edge fell >= 40% below baseline


def zone_fractions(df: pd.DataFrame) -> dict[str, float]:
    """Fraction of trades in each L1 zone, in canonical order."""
    n = len(df)
    counts = df["l1_zone"].value_counts()
    return {z: (float(counts.get(z, 0)) / n if n else 0.0) for z in L1_ZONE_ORDER}


def psi(expected: dict[str, float], actual: dict[str, float], eps: float = 1e-4) -> float:
    """Population Stability Index between two zone-fraction distributions.

    <0.1 stable · 0.1-0.25 moderate shift · >0.25 major shift.
    """
    total = 0.0
    for z in L1_ZONE_ORDER:
        e = max(expected.get(z, 0.0), eps)
        a = max(actual.get(z, 0.0), eps)
        total += (a - e) * np.log(a / e)
    return float(total)


def zone_mean_pnl(df: pd.DataFrame) -> dict[str, float]:
    """Mean PnL per L1 zone over resolved trades (the baseline payoff profile)."""
    res = df[df["resolution"].isin(["UP", "DOWN"]) & df["pnl"].notna()]
    out = {}
    for z in L1_ZONE_ORDER:
        g = res[res["l1_zone"] == z]
        out[z] = float(g["pnl"].mean()) if len(g) else 0.0
    return out


def expected_edge(zone_fracs: dict[str, float], zone_pnl: dict[str, float]) -> float:
    """Regime-weighted expected PnL/trade = Σ frac[z] · payoff[z]."""
    return float(sum(zone_fracs.get(z, 0.0) * zone_pnl.get(z, 0.0)
                     for z in L1_ZONE_ORDER))


def drift_signals(baseline: pd.DataFrame, recent: pd.DataFrame) -> dict:
    """Compare recent vs baseline windows; return drift metrics + alert flags."""
    if baseline.empty or recent.empty:
        return {"ok": None, "reason": "insufficient data "
                f"(baseline={len(baseline)}, recent={len(recent)})", "signals": []}

    b_l1 = baseline["l1"].dropna()
    r_l1 = recent["l1"].dropna()
    b_daily = baseline.groupby("day")["l1"].mean()
    dsd = float(b_daily.std(ddof=0)) if len(b_daily) > 1 else float("nan")
    d_l1 = float(r_l1.mean() - b_l1.mean())
    l1_z = abs(d_l1) / dsd if (dsd and not np.isnan(dsd) and dsd > 0) else float("nan")
    d_bull = float((r_l1 > 0).mean() - (b_l1 > 0).mean())

    bf, rf = zone_fractions(baseline), zone_fractions(recent)
    psi_val = psi(bf, rf)

    b_vol = float(daily_profile(baseline)["btc_range_pct"].mean())
    r_vol = float(daily_profile(recent)["btc_range_pct"].mean())
    vol_ratio = r_vol / b_vol if b_vol else float("nan")

    zp = zone_mean_pnl(baseline)
    ee_b, ee_r = expected_edge(bf, zp), expected_edge(rf, zp)
    edge_drop = (ee_b - ee_r) / abs(ee_b) if ee_b else float("nan")

    def alert(cond: float | bool) -> bool:
        return bool(cond) if not (isinstance(cond, float) and np.isnan(cond)) else False

    signals = [
        {"name": "L1 mean shift", "alert": alert(l1_z >= L1Z_ALERT),
         "value": f"ΔL1={d_l1:+.3f}  ({l1_z:.1f}σ of baseline daily)"
         if not np.isnan(l1_z) else f"ΔL1={d_l1:+.3f}  (σ n/a)",
         "thresh": f">={L1Z_ALERT:.0f}σ"},
        {"name": "%bull shift", "alert": alert(abs(d_bull) >= PCTBULL_ALERT),
         "value": f"Δ%bull={d_bull:+.2f}", "thresh": f">={PCTBULL_ALERT:.2f}"},
        {"name": "L1 zone-mix (PSI)", "alert": alert(psi_val >= PSI_ALERT),
         "value": f"PSI={psi_val:.3f}", "thresh": f">={PSI_ALERT:.2f}"},
        {"name": "volatility shift", "alert": alert(
            vol_ratio >= VOL_RATIO_ALERT or vol_ratio <= 1 / VOL_RATIO_ALERT),
         "value": f"range% {b_vol:.2f}→{r_vol:.2f} ({vol_ratio:.2f}x)",
         "thresh": f">={VOL_RATIO_ALERT:.1f}x"},
        {"name": "expected-edge-at-risk", "alert": alert(edge_drop >= EDGE_DROP_ALERT),
         "value": f"E[PnL/trade] {ee_b:+.3f}→{ee_r:+.3f} "
         f"({edge_drop*100:+.0f}%)" if not np.isnan(edge_drop)
         else f"E[PnL/trade] {ee_b:+.3f}→{ee_r:+.3f}",
         "thresh": f"drop>={EDGE_DROP_ALERT*100:.0f}%"},
    ]
    return {"ok": not any(s["alert"] for s in signals), "signals": signals,
            "n_baseline": len(baseline), "n_recent": len(recent),
            "edge_b": ee_b, "edge_r": ee_r, "edge_drop": edge_drop}


# --------------------------------------------------------------------------- #
# Report sections
# --------------------------------------------------------------------------- #
def _profile_section(bot: str, df: pd.DataFrame) -> str:
    prof = daily_profile(df)
    OUT.mkdir(parents=True, exist_ok=True)
    prof.to_csv(OUT / f"{bot}_daily_profile.csv", index=False)
    headers = ["day", "n", "L1mean", "%bull", "%bear", "p10", "p90",
               "btc", "rng%", "regime", "reg%"]
    rows = [[str(r.day), r.n, r.l1_mean, r.pct_bull, r.pct_bear, r.l1_p10,
             r.l1_p90, r.btc_mean, r.btc_range_pct, r.dom_regime, r.dom_regime_pct]
            for r in prof.itertuples()]
    return (f"\n[{bot}]  daily regime profile  ({len(df)} trades)\n"
            + format_table(headers, rows, align="right"))


def _pnl_section(bot: str, df: pd.DataFrame) -> str:
    zone = regime_pnl(df, "l1_zone")
    reg = regime_pnl(df, "regime")
    OUT.mkdir(parents=True, exist_ok=True)
    zone.to_csv(OUT / f"{bot}_pnl_by_l1zone.csv", index=False)
    reg.to_csv(OUT / f"{bot}_pnl_by_regime.csv", index=False)
    zr = [[r.l1_zone, r.n, r.win_pct, r.mean_pnl, r.total_pnl] for r in zone.itertuples()]
    rr = [[r.regime, r.n, r.win_pct, r.mean_pnl, r.total_pnl] for r in reg.itertuples()]
    hdr = ["bucket", "n", "win%", "meanPnL", "totPnL"]
    return (f"\n[{bot}]  PnL by L1 zone (resolved trades)\n"
            + format_table(hdr, zr, align="right")
            + f"\n\n[{bot}]  PnL by regime label (resolved trades)\n"
            + format_table(hdr, rr, align="right"))


def _drift_section(bot: str, full_df: pd.DataFrame) -> str:
    """Phase 3: recent window vs trailing baseline, on FULL history."""
    t_max = full_df["ts"].max()
    recent_start = t_max - pd.Timedelta(days=RECENT_DAYS)
    baseline_start = recent_start - pd.Timedelta(days=BASELINE_DAYS)
    recent = full_df[full_df["ts"] >= recent_start]
    baseline = full_df[(full_df["ts"] >= baseline_start) & (full_df["ts"] < recent_start)]
    d = drift_signals(baseline, recent)

    head = (f"\n[{bot}]  DRIFT ALARM  "
            f"(recent {RECENT_DAYS}d vs baseline {BASELINE_DAYS}d)\n")
    if d["ok"] is None:
        return head + f"  {d['reason']}"
    verdict = "OK — no material drift" if d["ok"] else "*** DRIFT DETECTED ***"
    lines = [head + f"  n: baseline={d['n_baseline']} recent={d['n_recent']}   "
             f"verdict: {verdict}"]
    rows = [["ALERT" if s["alert"] else "ok", s["name"], s["value"], s["thresh"]]
            for s in d["signals"]]
    lines.append(format_table(["", "signal", "value", "threshold"], rows, align="left"))
    return "\n".join(lines)


def _run(bots: list[str], days: int | None,
         profile: bool, pnl: bool, drift: bool) -> str:
    L = ["REGIME MONITOR  (read-only; live paper bots)"]
    if days:
        L.append(f"window: last {days} days (profile/pnl)")
    L.append("=" * 76)
    for bot in bots:
        try:
            full = load_trades(bot)
        except FileNotFoundError as e:
            L.append(f"\n[{bot}]  SKIPPED — {e}")
            continue
        if full.empty:
            L.append(f"\n[{bot}]  no trades")
            continue
        df = _window(full, days)
        if profile:
            L.append(_profile_section(bot, df))
        if pnl:
            L.append(_pnl_section(bot, df))
        if drift:
            L.append(_drift_section(bot, full))
    text = "\n".join(L)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scorecard.txt").write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------- #
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Regime drift monitor for live bots")
    sub = p.add_subparsers(dest="command", required=True)
    for cmd, helptext in [("profile", "daily L1 / vol / regime profile"),
                          ("pnl", "win% & PnL by L1 zone and regime"),
                          ("drift", "recent-vs-baseline drift alarm"),
                          ("report", "profile + pnl + drift")]:
        sp = sub.add_parser(cmd, help=helptext)
        sp.add_argument("--bot", default=None, help="single bot (default: all live bots)")
        sp.add_argument("--days", type=int, default=None, help="lookback window in days")
    args = p.parse_args()

    bots = [args.bot] if args.bot else LIVE_BOTS
    text = _run(bots, args.days,
                profile=args.command in ("profile", "report"),
                pnl=args.command in ("pnl", "report"),
                drift=args.command in ("drift", "report"))
    print(text)
    print(f"\nCSVs + scorecard written to {OUT}")


if __name__ == "__main__":
    main()
