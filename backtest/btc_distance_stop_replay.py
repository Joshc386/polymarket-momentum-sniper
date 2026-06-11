"""BTC distance-stop threshold replay: would tightening $100 -> $X help?

Replays the live stop rule (exit when BTC moves $X against the position
vs window open, only while >= 60s remain) on the bots' actual trades and
signal_ticks paths, for X in {60, 70, 80, 90}.

Two populations per bot (since 2026-05-29, zero-open windows excluded):
- HELD trades (resolution UP/DOWN, never stopped): a tighter stop converts
  some into early exits. delta = shares * (exit_price - won). Negative if
  the trade would have recovered and won; positive if it saved a loss.
- STOPPED trades (EARLY_EXIT:BTC_DISTANCE_STOP_*): already exited ~$100;
  a tighter stop exits EARLIER at a better salvage price.
  delta = shares * (p@X_cross - p@100_cross), tick-model on both sides.

Exit price model: our-side mid (YES -> mip, NO -> 1-mip) at the first
crossing tick. GROSS basis: entry fee identical across scenarios, so it
cancels; a spread haircut variant (exit at mid - entry ob_spread/2) is
reported for held-trade exits, where the haircut does not cancel.

Read-only. Usage: python -m backtest.btc_distance_stop_replay
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "data_runtime"

SINCE = "2026-05-29"
MIN_SECS_REMAINING = 60.0
THRESHOLDS = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]
CURRENT_STOP = 100.0

BOTS = [
    ("bot_k_sm_confirmation", "Bot K"),
    ("bot_k2_l1_floor", "Bot K2"),
]


def load_trades(bot: str) -> pd.DataFrame:
    with sqlite3.connect(str(RUNTIME / f"{bot}.db")) as c:
        df = pd.read_sql_query(
            "SELECT market_slug, side, entry_price, size_usdc, ob_spread, "
            "time_remaining_secs, resolution, oracle_price_at_open FROM trades "
            "WHERE pnl IS NOT NULL AND timestamp >= ? AND oracle_price_at_open != 0 "
            "AND entry_price > 0 AND size_usdc > 0 "
            "AND (resolution IN ('UP','DOWN') "
            "     OR resolution LIKE 'EARLY_EXIT:BTC_DISTANCE_STOP%')",
            c, params=(SINCE,),
        )
    df["shares"] = df["size_usdc"] / df["entry_price"]
    df["won"] = ((df["side"] == "YES") & (df["resolution"] == "UP")) | \
                ((df["side"] == "NO") & (df["resolution"] == "DOWN"))
    df["stopped"] = df["resolution"].str.startswith("EARLY_EXIT")
    return df


def load_paths(bot: str, slugs: list[str]) -> dict[str, pd.DataFrame]:
    paths: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(str(RUNTIME / f"{bot}_signal_diag.db")) as c:
        for i in range(0, len(slugs), 500):
            chunk = slugs[i:i + 500]
            q = (
                "SELECT market_slug, secs_remaining, btc_price, market_implied_prob "
                "FROM signal_ticks WHERE btc_price > 0 AND market_implied_prob IS NOT NULL "
                f"AND market_slug IN ({','.join('?' * len(chunk))})"
            )
            st = pd.read_sql_query(q, c, params=chunk)
            for slug, g in st.groupby("market_slug"):
                paths[slug] = g.sort_values("secs_remaining", ascending=False)
    return paths


def first_crossing(seg: pd.DataFrame, side: str, open_price: float,
                   threshold: float) -> tuple[float, float] | None:
    """First tick where BTC is >= threshold against `side` vs open.

    Returns (our_side_price, secs_remaining) at that tick, or None.
    Mirrors RiskManager.should_stop_btc_distance: YES stops on
    distance < -threshold, NO on distance > +threshold.
    """
    dist = seg["btc_price"].to_numpy() - open_price
    hit = dist < -threshold if side == "YES" else dist > threshold
    idx = np.argmax(hit) if hit.any() else None
    if idx is None:
        return None
    mip = float(seg["market_implied_prob"].to_numpy()[idx])
    p_side = mip if side == "YES" else 1.0 - mip
    return p_side, float(seg["secs_remaining"].to_numpy()[idx])


def replay_bot(bot: str, label: str) -> None:
    trades = load_trades(bot)
    paths = load_paths(bot, trades["market_slug"].unique().tolist())

    held = trades[~trades["stopped"]]
    stopped = trades[trades["stopped"]]
    print(f"\n=== {label}: {len(held)} held + {len(stopped)} stopped "
          f"(since {SINCE}, zero-open excluded) ===")

    # Per-trade in-window segment (after entry, while the stop is armed)
    def segment(t) -> pd.DataFrame | None:
        path = paths.get(t.market_slug)
        if path is None:
            return None
        seg = path[(path["secs_remaining"] < t.time_remaining_secs) &
                   (path["secs_remaining"] >= MIN_SECS_REMAINING)]
        return seg if len(seg) else None

    cov_held = sum(segment(t) is not None for t in held.itertuples())
    cov_stop = sum(segment(t) is not None for t in stopped.itertuples())
    print(f"tick coverage: held {cov_held}/{len(held)}, stopped {cov_stop}/{len(stopped)}")

    # Model consistency: held trades should NOT cross $100 (live stop was on)
    phantom = sum(
        1 for t in held.itertuples()
        if (seg := segment(t)) is not None
        and first_crossing(seg, t.side, t.oracle_price_at_open, CURRENT_STOP)
    )
    print(f"model check: held trades crossing $100 in ticks (should be ~0): {phantom}")

    results = defaultdict(lambda: dict(n_new=0, n_recovered=0, d_held=0.0,
                                       d_held_haircut=0.0, d_stopped=0.0,
                                       n_stop_earlier=0))
    for t in held.itertuples():
        seg = segment(t)
        if seg is None:
            continue
        for x in THRESHOLDS:
            hit = first_crossing(seg, t.side, t.oracle_price_at_open, x)
            if hit is None:
                continue
            exit_p, _ = hit
            r = results[x]
            r["n_new"] += 1
            r["n_recovered"] += int(t.won)
            r["d_held"] += t.shares * (exit_p - t.won)
            r["d_held_haircut"] += t.shares * (max(exit_p - t.ob_spread / 2, 0.0) - t.won)

    for t in stopped.itertuples():
        seg = segment(t)
        if seg is None:
            continue
        base = first_crossing(seg, t.side, t.oracle_price_at_open, CURRENT_STOP)
        if base is None:
            continue  # latency/coverage: $100 crossing not visible in ticks
        for x in THRESHOLDS:
            hit = first_crossing(seg, t.side, t.oracle_price_at_open, x)
            if hit is None:
                continue
            r = results[x]
            r["n_stop_earlier"] += 1
            r["d_stopped"] += t.shares * (hit[0] - base[0])

    print(f"\n{'X':>5} {'new stops':>10} {'recovered':>10} {'d_held':>9} "
          f"{'d_held(hc)':>11} {'d_stopped':>10} {'TOTAL':>9} {'TOTAL(hc)':>10}")
    for x in THRESHOLDS:
        r = results[x]
        rec = f"{r['n_recovered']}/{r['n_new']}"
        tot = r["d_held"] + r["d_stopped"]
        tot_hc = r["d_held_haircut"] + r["d_stopped"]
        print(f"{x:>5.0f} {r['n_new']:>10} {rec:>10} {r['d_held']:>+9.2f} "
              f"{r['d_held_haircut']:>+11.2f} {r['d_stopped']:>+10.2f} "
              f"{tot:>+9.2f} {tot_hc:>+10.2f}")
    print("(d_held: tighter stop on never-stopped trades; recovered = would "
          "have won if held. d_stopped: earlier exit on already-stopped "
          "trades. hc = exit haircut of half entry spread.)")


def self_test() -> None:
    seg = pd.DataFrame({
        "secs_remaining": [200.0, 150.0, 100.0],
        "btc_price": [59950.0, 59910.0, 59880.0],   # open 60000: -50, -90, -120
        "market_implied_prob": [0.40, 0.25, 0.10],
    })
    # YES, threshold 80: first crossing at -90 -> mip 0.25
    assert first_crossing(seg, "YES", 60000.0, 80.0) == (0.25, 150.0)
    # YES, threshold 100: crossing at -120 -> mip 0.10
    assert first_crossing(seg, "YES", 60000.0, 100.0) == (0.10, 100.0)
    # NO never stopped here (distance always negative)
    assert first_crossing(seg, "NO", 60000.0, 80.0) is None
    # threshold deeper than any move -> None
    assert first_crossing(seg, "YES", 60000.0, 200.0) is None


def main() -> None:
    self_test()
    for bot, label in BOTS:
        replay_bot(bot, label)


if __name__ == "__main__":
    main()
