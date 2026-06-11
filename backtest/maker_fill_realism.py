"""Maker-fill realism — re-price the LIVE bots' validated edge under realistic fills.

THE GO-LIVE GATE. Paper trading fills every entry at the ASK with 100% certainty
(`PaperExecutionEngine`: fill_price = price, order_id = None). But live Bot K is a
MAKER: it posts a GTC limit at the BID (every entry is >60s-remaining → 100% maker
regime, confirmed) and only fills when the market trades down to it — a better
price, but adverse-selected and not guaranteed. This script asks: how much of the
validated paper edge survives once we model that?

Track B of the maker-fill plan: uses the bots' ACTUAL logged trades (directly
comparable to validate_edge) + the `signal_ticks` mid-price path + per-trade
`ob_spread`. (Track A = the high-fidelity tick_data backtest on Mar–May 11.)

Fill model (side-agnostic):
  - bot-side price p_side(t): YES -> market_implied_prob, NO -> 1 - mip.
  - maker rests at the bid L = entry_price - spread = p_side(entry).
  - fills iff p_side drops by >= one spread within the window
    (that drop IS the adverse selection: you fill right as your side weakens).
  - filled  -> maker fill at L, 0% maker fee.
  - unfilled-> MISSED (no trade); tracked for adverse-selection analysis.
  - window: STRICT = first 10s (one GTC timeout); OPTIMISTIC = entry..60s-left
    (continuous re-post). Reported as a band.

PnL is recomputed from the real resolution (not the heterogeneous paper `pnl`,
which spans the Jun-3 fee-model change):
  - realistic maker: won - L                       (0% maker fee)
  - paper baseline : won - entry_price - taker_fee  (fill at ask, 7.2% fee)

Read-only. Outputs: backtest/maker_fill_realism/<bot>_*.csv + scorecard.txt
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "data_runtime"
OUT = REPO / "backtest" / "maker_fill_realism"
OUT.mkdir(parents=True, exist_ok=True)

BOTS = [
    ("bot_k_sm_confirmation", "Bot K (SM confirmation)"),
    ("bot_k2_l1_floor", "Bot K2 (L1 floor)"),
]
TAKER_FEE_RATE = 0.072       # crypto-category taker fee: 0.072 * p * (1-p)
GTC_TIMEOUT_S = 10.0         # strict window: one GTC timeout
TAKER_FLOOR_SECS = 60.0      # optimistic window extends to here (then bot would FOK)

# Optional A/B-window mode: pass an ISO date as argv[1] (e.g. 2026-05-29) to
# restrict to trades >= that date AND exclude zero-open (L1-dead) windows,
# matching backtest/k2_ab_eval.py conventions. Outputs get a suffix so the
# full-period J26 artifacts are never overwritten.
SINCE: str | None = None
SUFFIX = ""


def taker_fee(p: float) -> float:
    return TAKER_FEE_RATE * p * (1.0 - p)


def would_fill(p_side_entry: float, spread: float, p_side_path: np.ndarray) -> bool:
    """A resting maker buy at the bid fills iff the bot-side price drops by >=
    one spread anywhere on the path (i.e. the side weakens to our limit).

    Pure function — regression-tested in tests/test_maker_fill_realism.py.
    """
    if p_side_path.size == 0:
        return False
    return bool(np.any(p_side_path <= p_side_entry - spread))


def _load_trades(bot: str) -> pd.DataFrame:
    query = (
        "SELECT timestamp, market_slug, side, entry_price, ob_spread, "
        "time_remaining_secs, resolution, size_usdc FROM trades "
        "WHERE resolution IN ('UP','DOWN') AND entry_price IS NOT NULL "
        "AND ob_spread IS NOT NULL"
    )
    params: tuple = ()
    if SINCE:
        query += " AND timestamp >= ? AND oracle_price_at_open != 0"
        params = (SINCE,)
    with sqlite3.connect(str(RUNTIME / f"{bot}.db")) as c:
        df = pd.read_sql_query(query, c, params=params)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["day"] = df["ts"].dt.date
    df["won"] = ((df["side"] == "YES") & (df["resolution"] == "UP")) | \
                ((df["side"] == "NO") & (df["resolution"] == "DOWN"))
    df["won"] = df["won"].astype(int)
    df["p_side_entry"] = np.where(df["side"] == "YES",
                                  df["entry_price"] - df["ob_spread"],
                                  df["entry_price"] - df["ob_spread"])
    return df


def _load_mid_paths(bot: str) -> dict[str, pd.DataFrame]:
    """market_slug -> DataFrame(secs_remaining, mip) of in-window mid ticks."""
    with sqlite3.connect(str(RUNTIME / f"{bot}_signal_diag.db")) as c:
        st = pd.read_sql_query(
            "SELECT market_slug, secs_remaining, market_implied_prob AS mip "
            "FROM signal_ticks WHERE secs_remaining BETWEEN 0 AND 300 "
            "AND market_implied_prob IS NOT NULL",
            c,
        )
    return {slug: g.sort_values("secs_remaining", ascending=False)
            for slug, g in st.groupby("market_slug")}


def _simulate(trades: pd.DataFrame, paths: dict, window: str) -> pd.DataFrame:
    """Return per-trade fill outcome under STRICT or OPTIMISTIC window."""
    rows = []
    for t in trades.itertuples():
        path = paths.get(t.market_slug)
        if path is None:
            rows.append({"covered": False})
            continue
        entry_sr = t.time_remaining_secs
        lo = TAKER_FLOOR_SECS if window == "optimistic" else entry_sr - GTC_TIMEOUT_S
        # ticks strictly after entry (less time remaining), within the window floor
        seg = path[(path["secs_remaining"] < entry_sr) & (path["secs_remaining"] >= lo)]
        p_side_entry = t.entry_price - t.ob_spread
        if t.side == "YES":
            p_side_path = seg["mip"].to_numpy()
        else:
            p_side_path = (1.0 - seg["mip"]).to_numpy()
        filled = would_fill(p_side_entry, t.ob_spread, p_side_path)
        maker_L = t.entry_price - t.ob_spread
        rows.append({
            "covered": True, "filled": filled, "won": t.won, "day": t.day,
            "size": t.size_usdc, "entry": t.entry_price, "maker_L": maker_L,
            "pnl_maker": (t.won - maker_L) if filled else np.nan,
            "pnl_paper": t.won - t.entry_price - taker_fee(t.entry_price),
        })
    return pd.DataFrame(rows)


def _day_t(pnl_by_day: pd.Series) -> tuple[float, int, int]:
    if len(pnl_by_day) < 2:
        return (float("nan"), len(pnl_by_day), int((pnl_by_day > 0).sum()))
    se = pnl_by_day.std(ddof=1) / np.sqrt(len(pnl_by_day))
    t = pnl_by_day.mean() / se if se else float("nan")
    return (t, len(pnl_by_day), int((pnl_by_day > 0).sum()))


def _report_bot(bot: str, name: str) -> str:
    trades = _load_trades(bot)
    paths = _load_mid_paths(bot)
    L = [f"\n{'='*72}\n{name}  [{bot}]\n{'-'*72}",
         f"  resolved trades w/ entry+spread : {len(trades)}"]

    # paper baseline (recomputed: fill at ask + taker fee, all trades)
    sim_any = _simulate(trades, paths, "strict")
    covered = sim_any[sim_any["covered"]].copy() if "covered" in sim_any else sim_any
    n_cov = len(covered)
    L.append(f"  covered by signal_ticks mid path: {n_cov} "
             f"({n_cov/len(trades)*100:.0f}%)   "
             f"[uncovered = silent signal_diag gap / pre-logging]")
    if n_cov == 0:
        L.append("  -- no mid-path coverage; cannot run Track B for this bot --")
        return "\n".join(L)

    base_pnl = covered["pnl_paper"].to_numpy()
    base_total = float(base_pnl.sum())
    base_roi = 100 * base_total / float(covered["entry"].sum())
    bt, bd, bp = _day_t(covered.groupby("day")["pnl_paper"].sum())
    L += [
        "",
        "  PAPER BASELINE (recomputed: fill@ask + 7.2% taker fee, 100% fill)",
        f"    N={n_cov}  win%={covered['won'].mean():.3f}  "
        f"EV/sh={base_pnl.mean():+.4f}  total={base_total:+.1f}  ROI={base_roi:+.1f}%",
        f"    day-clustered t={bt:.2f} ({bp}/{bd} +days)",
    ]

    for window in ("strict", "optimistic"):
        sim = _simulate(trades, paths, window)
        cov = sim[sim["covered"]].copy()
        cov["filled"] = cov["filled"].astype(bool)
        fills = cov[cov["filled"]]
        miss = cov[~cov["filled"]]
        fr = len(fills) / len(cov) if len(cov) else 0.0
        if len(fills):
            fp = fills["pnl_maker"].to_numpy()
            total = float(fp.sum())
            roi = 100 * total / float(fills["maker_L"].sum())
            ft, fd, fpd = _day_t(fills.groupby("day")["pnl_maker"].sum())
            fill_win = fills["won"].mean()
        else:
            total = roi = fill_win = float("nan"); ft = fd = fpd = 0
        miss_win = miss["won"].mean() if len(miss) else float("nan")
        cov.to_csv(OUT / f"{bot}_{window}{SUFFIX}.csv", index=False)
        L += [
            "",
            f"  REALISTIC MAKER — {window.upper()} window "
            f"({'10s GTC' if window=='strict' else 'entry..60s-left, re-post'})",
            f"    fill rate={fr:.3f}  ({len(fills)}/{len(cov)} fill)",
            f"    FILLED:  win%={fill_win:.3f}  EV/sh={ (total/len(fills)) if len(fills) else float('nan'):+.4f}  "
            f"total={total:+.1f}  ROI={roi:+.1f}%   day-t={ft:.2f} ({fpd}/{fd}+)",
            f"    MISSED:  win%={miss_win:.3f}  (n={len(miss)})   "
            f"<- {'ADVERSE (missed are winners)' if (miss_win and not np.isnan(miss_win) and miss_win > fill_win) else 'ok'}",
            f"    VERDICT: realistic total {total:+.1f} vs paper {base_total:+.1f}  "
            f"=> edge {'SURVIVES' if (not np.isnan(total) and total > 0 and total >= 0.5*base_total) else ('DEGRADED' if (not np.isnan(total) and total>0) else 'GONE')}",
        ]
    return "\n".join(L)


def main() -> None:
    global SINCE, SUFFIX
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) > 1:
        SINCE = sys.argv[1]
        SUFFIX = f"_since_{SINCE}"
    out = ["MAKER-FILL REALISM — Track B (actual logged trades, mid-path fill model)",
           "read-only; paper assumes 100% fill at ask, reality is maker-at-bid + adverse selection"]
    if SINCE:
        out.append(f"A/B-WINDOW MODE: trades >= {SINCE}, zero-open (L1-dead) windows excluded")
    for bot, name in BOTS:
        try:
            out.append(_report_bot(bot, name))
        except FileNotFoundError as e:
            out.append(f"\n[{bot}] SKIPPED — {e}")
    text = "\n".join(out)
    print(text)
    (OUT / f"scorecard{SUFFIX}.txt").write_text(text, encoding="utf-8")
    print(f"\nScorecard + per-trade CSVs -> {OUT}")


if __name__ == "__main__":
    main()
