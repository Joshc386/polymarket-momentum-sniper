"""Capture-missed-fills study — is there residual EV in chasing the misses?

Follow-up to the maker-fill realism gate (J26 / Phase 23). That study showed
62–92% of MISSED maker orders are winners — the adverse-selection signature:
a resting bid systematically misses the trades where price moved your way.

The trap this script tests directly: by the time the GTC times out, the price
has ALREADY repriced toward the winning outcome. So the question is not
"missed trades win 70–90%, capture them" but:

    At the CHASE price (the repriced bot-side ask at timeout) PLUS the 7.2%
    taker fee, scored against the real resolution, is there residual EV?

Three policies compared, on the bots' actual logged trades:
  1. MAKER-ONLY (status quo)      — fills as maker, misses dropped (Track B).
  2. UNCONDITIONAL CHASE          — on 10s GTC timeout, cross at the new ask
                                    + taker fee (what J7 tested and rejected
                                    on the weak pre-live signal).
  3. CONDITIONAL CHASE (the idea) — chase ONLY if the price hasn't run more
                                    than `cap` above the original ask
                                    (premium cap grid: 0.00–0.05, inf).

Fill/miss decided by the same strict-10s model as maker_fill_realism (matches
live GTC behaviour). Chase ask reconstructed at the timeout tick as
p_side(timeout) + at-entry spread (mid+spread proxy, same fidelity caveat).

Read-only. Outputs: backtest/capture_missed_fills/scorecard.txt + per-bot CSVs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtest.maker_fill_realism import (  # noqa: E402
    BOTS,
    GTC_TIMEOUT_S,
    RUNTIME,
    _day_t,
    _load_trades,
    taker_fee,
    would_fill,
)

OUT = REPO / "backtest" / "capture_missed_fills"
OUT.mkdir(parents=True, exist_ok=True)

PREMIUM_CAPS = [0.00, 0.01, 0.02, 0.03, 0.05, np.inf]
TIMEOUT_TOL_S = 10.0   # accept the tick nearest (timeout ± tol)
MAX_CHASE_ASK = 0.97   # don't model chasing into a near-resolved book
ENTRY_FLOOR_S = 60.0   # latest_entry_secs — no posting below this
POST_TOL_S = 5.0       # tick must be within ±5s of the re-post moment


def capture_pnl(won: int, chase_ask: float) -> float:
    """Taker PnL per share for a chased fill: resolution payout minus the
    repriced ask minus the real taker fee. Pure — regression-tested."""
    return float(won) - chase_ask - taker_fee(chase_ask)


def chase_ask_at_timeout(p_side_timeout: float, spread: float) -> float:
    """Reconstruct the bot-side ask at the timeout moment (mid+spread proxy)."""
    return min(p_side_timeout + spread, 0.99)


def repost_fill(entry_sr: float, entry_bid: float, spread: float,
                secs: np.ndarray, p_side: np.ndarray,
                gate_ok: np.ndarray | None = None,
                floor: float = ENTRY_FLOOR_S,
                timeout: float = GTC_TIMEOUT_S) -> tuple[bool, float]:
    """Sequential GTC re-post — what live ALREADY does on a miss.

    Round 0 posts at the entry bid. Each timeout, cancel and re-post at the
    then-current touch (nearest tick to the re-post moment), until fill or the
    60s entry floor. A round's order fills iff the bot-side price trades
    through the resting bid (drops >= one spread) within that round.

    gate_ok (optional) aligned with secs: live only re-posts when the entry
    gate still passes (prob_edge >= required_edge, side unchanged). None =
    price-only re-posting (maximal baseline, conservative against the chase).

    Returns (filled, fill_price). Pure — regression-tested.
    """
    t_post = entry_sr
    bid = entry_bid
    while t_post > floor:
        lo = max(t_post - timeout, floor)
        if bid is not None:
            m = (secs < t_post) & (secs >= lo)
            if m.any() and (p_side[m] <= bid - spread).any():
                return True, bid
        t_post -= timeout
        if t_post <= floor:
            break
        # re-post at the touch nearest the new post moment (if gate allows)
        d = np.abs(secs - t_post)
        if d.size == 0 or d.min() > POST_TOL_S:
            continue  # no tick near the re-post moment; nothing rests this round
        i = int(d.argmin())
        if gate_ok is not None and not gate_ok[i]:
            bid = None  # gate failed — live posts nothing this round
        else:
            bid = float(p_side[i])
    return False, float("nan")


def _load_paths_with_gate(bot: str) -> dict[str, pd.DataFrame]:
    """Mid path + per-tick entry-gate fields (position-independent: from the
    snapshot builder, which runs every tick). would_enter is NOT usable here —
    it is gated on `not self._has_position` so it reads 0 after the paper entry."""
    import sqlite3
    with sqlite3.connect(str(RUNTIME / f"{bot}_signal_diag.db")) as c:
        st = pd.read_sql_query(
            "SELECT market_slug, secs_remaining, market_implied_prob AS mip, "
            "prob_edge, required_edge, would_pick_side "
            "FROM signal_ticks WHERE secs_remaining BETWEEN 0 AND 300 "
            "AND market_implied_prob IS NOT NULL",
            c,
        )
    return {slug: g.sort_values("secs_remaining", ascending=False)
            for slug, g in st.groupby("market_slug")}


def _simulate_bot(bot: str) -> tuple[pd.DataFrame, dict] | None:
    trades = _load_trades(bot)
    paths = _load_paths_with_gate(bot)
    rows = []
    for t in trades.itertuples():
        path = paths.get(t.market_slug)
        if path is None:
            continue
        entry_sr = t.time_remaining_secs
        timeout_sr = entry_sr - GTC_TIMEOUT_S
        seg = path[(path["secs_remaining"] < entry_sr)
                   & (path["secs_remaining"] >= timeout_sr)]
        p_side_entry = t.entry_price - t.ob_spread
        if t.side == "YES":
            p_path = seg["mip"].to_numpy()
        else:
            p_path = (1.0 - seg["mip"]).to_numpy()
        filled = would_fill(p_side_entry, t.ob_spread, p_path)

        # sequential re-post baselines (full in-window path after entry)
        full = path[path["secs_remaining"] < entry_sr]
        f_secs = full["secs_remaining"].to_numpy()
        f_pside = (full["mip"].to_numpy() if t.side == "YES"
                   else 1.0 - full["mip"].to_numpy())
        gate_ok = ((full["prob_edge"].to_numpy() >= full["required_edge"].to_numpy())
                   & (full["would_pick_side"].to_numpy() == t.side))
        rp_fill, rp_price = repost_fill(entry_sr, p_side_entry, t.ob_spread,
                                        f_secs, f_pside, gate_ok=None)
        rg_fill, rg_price = repost_fill(entry_sr, p_side_entry, t.ob_spread,
                                        f_secs, f_pside, gate_ok=gate_ok)

        row = {
            "day": t.day, "side": t.side, "won": t.won, "entry": t.entry_price,
            "spread": t.ob_spread, "filled": filled,
            "pnl_maker": (t.won - p_side_entry) if filled else np.nan,
            "repost_filled": rp_fill,
            "pnl_repost": (t.won - rp_price) if rp_fill else np.nan,
            "repost_gated_filled": rg_fill,
            "pnl_repost_gated": (t.won - rg_price) if rg_fill else np.nan,
            "chase_ask": np.nan, "chase_premium": np.nan, "pnl_chase": np.nan,
        }
        if not filled:
            # locate the tick nearest the timeout moment
            near = path[(path["secs_remaining"] >= timeout_sr - TIMEOUT_TOL_S)
                        & (path["secs_remaining"] <= timeout_sr + TIMEOUT_TOL_S)]
            if len(near):
                idx = (near["secs_remaining"] - timeout_sr).abs().idxmin()
                mip_t = float(near.loc[idx, "mip"])
                p_side_t = mip_t if t.side == "YES" else 1.0 - mip_t
                ask = chase_ask_at_timeout(p_side_t, t.ob_spread)
                if ask <= MAX_CHASE_ASK:
                    row["chase_ask"] = ask
                    row["chase_premium"] = ask - t.entry_price
                    row["pnl_chase"] = capture_pnl(t.won, ask)
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows), {"n_trades": len(trades)}


def _policy_line(tag: str, pnl: pd.Series, days: pd.Series, won: pd.Series) -> str:
    total = float(pnl.sum())
    n = len(pnl)
    t, nd, pos = _day_t(pnl.groupby(days).sum())
    return (f"    [{tag:<26s}] N={n:5d}  win%={won.mean():.3f}  "
            f"EV/sh={pnl.mean():+.4f}  total={total:+.1f}  "
            f"day-t={t:.2f} ({pos}/{nd}+)")


def _report_bot(bot: str, name: str) -> str:
    sim = _simulate_bot(bot)
    if sim is None:
        return f"\n[{bot}] no coverage"
    df, meta = sim
    fills = df[df["filled"]]
    miss = df[~df["filled"]]
    chaseable = miss.dropna(subset=["pnl_chase"])

    L = [f"\n{'='*72}\n{name}  [{bot}]\n{'-'*72}",
         f"  covered trades: {len(df)} (of {meta['n_trades']} resolved)   "
         f"maker fills: {len(fills)}  misses: {len(miss)}  "
         f"chase-priceable misses: {len(chaseable)}",
         f"  missed win%: {miss['won'].mean():.3f}   "
         f"avg chase premium (ask moved): {chaseable['chase_premium'].mean():+.4f}",
         "",
         "  -- policies (per-share PnL, real resolutions, day-clustered t) --"]

    # 1. maker baselines: single-shot, then sequential re-post (live as-is)
    L.append(_policy_line("maker 1-shot (miss=drop)", fills["pnl_maker"],
                          fills["day"], fills["won"]))
    rp = df[df["repost_filled"]]
    L.append(_policy_line("re-post price-only (max)", rp["pnl_repost"],
                          rp["day"], rp["won"]))
    rg = df[df["repost_gated_filled"]]
    L.append(_policy_line("re-post gate-checked LIVE", rg["pnl_repost_gated"],
                          rg["day"], rg["won"]))

    # 2/3. chase policies: maker fills + chased subset
    grid_rows = []
    for cap in PREMIUM_CAPS:
        take = chaseable[chaseable["chase_premium"] <= cap]
        combo_pnl = pd.concat([fills["pnl_maker"], take["pnl_chase"]])
        combo_day = pd.concat([fills["day"], take["day"]])
        combo_won = pd.concat([fills["won"], take["won"]])
        tag = ("unconditional chase" if np.isinf(cap)
               else f"chase if premium<={cap:.2f}")
        L.append(_policy_line(tag, combo_pnl, combo_day, combo_won))
        # incremental view: the chased subset alone
        if len(take):
            grid_rows.append({
                "cap": cap, "n_chased": len(take),
                "chase_win%": round(float(take["won"].mean()), 3),
                "chase_EV/sh": round(float(take["pnl_chase"].mean()), 4),
                "chase_total": round(float(take["pnl_chase"].sum()), 1),
            })

    L += ["", "  -- the chased subset alone (incremental on top of maker fills) --"]
    for r in grid_rows:
        cap_s = "inf " if np.isinf(r["cap"]) else f"{r['cap']:.2f}"
        L.append(f"    cap={cap_s}: n={r['n_chased']:5d}  win%={r['chase_win%']:.3f}  "
                 f"EV/sh={r['chase_EV/sh']:+.4f}  total={r['chase_total']:+.1f}")
    df.to_csv(OUT / f"{bot}_capture.csv", index=False)
    return "\n".join(L)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = ["CAPTURE MISSED FILLS — residual EV after the market repriced?",
           "(read-only; strict 10s GTC fill model; chase at timeout ask + 7.2% fee)"]
    for bot, name in BOTS:
        try:
            out.append(_report_bot(bot, name))
        except FileNotFoundError as e:
            out.append(f"\n[{bot}] SKIPPED — {e}")
    text = "\n".join(out)
    print(text)
    (OUT / "scorecard.txt").write_text(text, encoding="utf-8")
    print(f"\nScorecard + CSVs -> {OUT}")


if __name__ == "__main__":
    main()
