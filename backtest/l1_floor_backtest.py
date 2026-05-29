"""
Backtest validation of the directional L1 floor (2026-05-29).

Hypothesis (from live-trade investigation, bot_k_l1_leak_investigation.py):
the bot leaks money when it bets AGAINST the window-open resolution line —
YES (UP) while L1<0 (price below line), or NO (DOWN) while L1>0 (above).
Proposed fix: require L1>=0 for YES, L1<=0 for NO entries.

This validates the principle on an INDEPENDENT dataset: the polybacktest
snapshot pipeline covers 2026-03-08 to 2026-04-08 — a different month and
regime from the live trades (May), ~9k markets, balanced 4500/4482 Up/Down.
The engine uses DEFAULT weights and has NO J13. So this is NOT a faithful
Bot K replica — it is a robustness check: if the floor helps here too (on
different config, period, and regime), the effect is a robust principle,
not an artifact of Bot K's May weights or the 14-day live sample.

Reuses the real engine (signals, sizing, BacktestAccount, risk) from
backtest_real_pricing.py — only the entry-decision loop is reproduced here
with a gate toggle. The gated loop mirrors strategy_contrarian_ev exactly
except for the L1 gate, placed after side selection (where a live filter
would sit). Read-only — touches no live bot code.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime

from backtest.backtest_real_pricing import (
    DATA_DIR, FEE_RATE, MIN_CONFIDENCE,
    BacktestAccount, CandleLookup, CoinalyzeLookup,
    MomentumSignal, OracleLagSignal, OrderbookSignalBacktest, RegimeDetector,
    SnapshotSignals, TradeResult,
    compute_ev, compute_snapshot_signals, load_markets, load_snapshots,
    required_edge,
)


def gated_contrarian_entry(
    sorted_offsets: list[int],
    precomputed: dict[int, SnapshotSignals],
    winner: str,
    start_time: str,
    gate: bool,
) -> TradeResult | None:
    """Mirror of strategy_contrarian_ev with an optional L1 directional gate.

    Gate: skip a YES entry when L1 (sig.oracle) < 0, or a NO entry when
    L1 > 0 — i.e. never bet against the window-open resolution line. The
    loop continues scanning later offsets, exactly as the live bot would.
    """
    for offset_sec in sorted_offsets:
        sig = precomputed.get(offset_sec)
        if sig is None:
            continue
        if sig.regime_name == "low_vol":
            continue
        confidence = abs(sig.prob_up - 0.5)
        if confidence < MIN_CONFIDENCE:
            continue

        ev_yes, ev_no = compute_ev(sig.prob_up, sig.yes_ask, sig.no_ask)
        if ev_yes >= ev_no:
            side, best_ev, price = "YES", ev_yes, sig.yes_ask
        else:
            side, best_ev, price = "NO", ev_no, sig.no_ask

        # ── directional L1 floor (the change under test) ──
        if gate:
            if side == "YES" and sig.oracle < 0:
                continue
            if side == "NO" and sig.oracle > 0:
                continue

        regime_edge_mult = sig.regime_params.get("edge_multiplier", 1.0)
        req_edge = required_edge(sig.secs_remaining, regime_edge_mult)
        if best_ev <= req_edge:
            continue

        won = (side == "YES" and winner == "Up") or (side == "NO" and winner == "Down")
        if won:
            gross = 1.0 - price
            pnl = gross - (gross * FEE_RATE)
        else:
            pnl = -price

        r = TradeResult(
            market_id=sig.snap.get("market_id", ""),
            strategy="contrarian_ev_gated" if gate else "contrarian_ev",
            side=side, entry_price=price, seconds_remaining=sig.secs_remaining,
            est_prob_up=sig.prob_up, ev=best_ev, won=won, pnl=pnl,
            start_time=start_time, regime=sig.regime_name,
        )
        r.oracle = sig.oracle  # stash L1 for bucket analysis
        return r
    return None


def run(markets: list[dict], snapshots, candle_lookup, coinalyze, gate: bool,
        flat: bool = False):
    """Replicate main()'s contrarian_ev path with sizing + risk, gate toggle.

    flat=True bypasses BacktestAccount (can_trade + sizing) and takes every
    entry at a flat $1 stake — isolates SIGNAL QUALITY from the risk
    manager's survival path-dependence (which otherwise lets the gated run
    stay alive and take more trades than the bleeding ungated run).
    """
    momentum_sig = MomentumSignal()
    oracle_sig = OracleLagSignal()
    regime_det = RegimeDetector()
    ob_sig = OrderbookSignalBacktest()
    acct = BacktestAccount()
    trades: list[TradeResult] = []

    for market in markets:
        mid = market["market_id"]
        winner = market.get("winner")
        btc_start = market.get("btc_price_start", 0)
        if not winner or not btc_start:
            continue
        snaps = snapshots.get(mid, {})
        if not snaps:
            continue
        start_time = market.get("start_time", "")

        market_ts = 0.0
        if start_time:
            try:
                market_ts = datetime.fromisoformat(
                    start_time.replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                pass

        volatility = 0.005
        if candle_lookup and start_time:
            vc = candle_lookup.get_candles(start_time, count=15)
            if vc and len(vc) >= 2:
                hi = max(c.high for c in vc); lo = min(c.low for c in vc)
                mp = (hi + lo) / 2
                if mp > 0:
                    volatility = (hi - lo) / mp

        regime_det.reset_history()
        sorted_offsets = sorted(snaps.keys())
        precomputed: dict[int, SnapshotSignals] = {}
        for off in sorted_offsets:
            s = compute_snapshot_signals(
                snaps[off], off, btc_start, candle_lookup,
                momentum_sig, oracle_sig, regime_det, ob_sig, coinalyze)
            if s is not None:
                precomputed[off] = s

        r = gated_contrarian_entry(sorted_offsets, precomputed, winner, start_time, gate)
        if not r:
            continue
        r.raw_pnl = r.pnl  # per-$1 pnl before sizing (signal-quality view)
        if flat:
            r.size_usdc = 1.0
            trades.append(r)
            continue
        can, mult = acct.can_trade(start_time, market_ts, volatility)
        if not can:
            continue
        est_prob_win = r.est_prob_up if r.side == "YES" else (1.0 - r.est_prob_up)
        size = acct.compute_size(est_prob_win, r.entry_price, mult)
        if size <= 0:
            continue
        num_shares = size / r.entry_price
        r.pnl *= num_shares
        r.size_usdc = size
        acct.record_trade(r.pnl, market_ts)
        trades.append(r)

    return acct, trades


def stats(trades: list[TradeResult], acct: BacktestAccount) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0}
    wins = sum(1 for t in trades if t.won)
    raw = sum(t.raw_pnl for t in trades)        # unsized, per-$1
    sized = sum(t.pnl for t in trades)          # risk-managed
    return {
        "n": n,
        "WR_%": round(100 * wins / n, 1),
        "raw_pnl_per_$1": round(raw, 2),
        "raw_per_trade": round(raw / n, 4),
        "sized_pnl": round(sized, 2),
        "final_bankroll": round(acct.bankroll, 2),
    }


def bucket_by_l1(trades: list[TradeResult]) -> None:
    """Confirm the losing-against-the-line pattern exists in THIS data too."""
    edges = [(-1.01, -0.4), (-0.4, -0.2), (-0.2, 0.0),
             (0.0, 0.2), (0.2, 0.4), (0.4, 1.01)]
    for side in ["YES", "NO"]:
        print(f"\n  {side} (gate OFF) by L1 bin — raw per-$1 pnl:")
        print(f"    {'bin':>14} {'n':>5} {'WR%':>6} {'raw_total':>10} {'per_trade':>10}")
        for lo, hi in edges:
            cell = [t for t in trades if t.side == side
                    and lo <= getattr(t, "oracle", 0.0) < hi]
            if not cell:
                continue
            n = len(cell)
            wr = 100 * sum(1 for t in cell if t.won) / n
            raw = sum(t.raw_pnl for t in cell)
            flag = "  <-- bets against line" if (
                (side == "YES" and hi <= 0.0) or (side == "NO" and lo >= 0.0)
            ) else ""
            print(f"    {f'{lo:.1f}..{hi:.1f}':>14} {n:>5} {wr:>6.1f} "
                  f"{raw:>10.2f} {raw/n:>10.4f}{flag}")


def main() -> int:
    print()
    print("Loading polybacktest data (2026-03-08 to 2026-04-08, independent of live)...")
    markets = load_markets(os.path.join(DATA_DIR, "polybacktest_markets.csv"))
    snapshots = load_snapshots(os.path.join(DATA_DIR, "polybacktest_snapshots.csv"))
    klines = os.path.join(DATA_DIR, "binance_klines_1m.csv")
    candle_lookup = CandleLookup(klines) if os.path.exists(klines) else None
    coinalyze = CoinalyzeLookup(DATA_DIR)
    for m in markets:
        m["date"] = (m.get("start_time", "") or "")[:10]
    print(f"  {len(markets)} markets loaded.")
    print()

    print("=" * 78)
    print("(A) CLEAN SIGNAL QUALITY — flat $1 stake, NO risk manager")
    print("    (isolates the gate's effect from survival path-dependence)")
    print("=" * 78)
    af_off, tf_off = run(markets, snapshots, candle_lookup, coinalyze, gate=False, flat=True)
    af_on, tf_on = run(markets, snapshots, candle_lookup, coinalyze, gate=True, flat=True)
    fo, fn = stats(tf_off, af_off), stats(tf_on, af_on)
    print(f"  gate OFF: n={fo['n']}, WR={fo['WR_%']}%, "
          f"total raw pnl/$1={fo['raw_pnl_per_$1']:+.2f}, per-trade={fo['raw_per_trade']:+.4f}")
    print(f"  gate ON : n={fn['n']}, WR={fn['WR_%']}%, "
          f"total raw pnl/$1={fn['raw_pnl_per_$1']:+.2f}, per-trade={fn['raw_per_trade']:+.4f}")
    if fo["n"] and fn["n"]:
        print(f"  --> per-trade EV {fo['raw_per_trade']:+.4f} -> {fn['raw_per_trade']:+.4f}, "
              f"WR {fo['WR_%']}% -> {fn['WR_%']}%, "
              f"dropped {fo['n']-fn['n']} of {fo['n']} entries "
              f"({100*(fo['n']-fn['n'])/fo['n']:.0f}%)")
    print()

    print("=" * 78)
    print("(B) REALISTIC — full BacktestAccount (Kelly + risk caps), gate OFF vs ON")
    print("    NOTE: trade-count rises with gate ON because the ungated run hits")
    print("    loss caps / streak pauses and stops trading (survival effect).")
    print("=" * 78)
    acct_off, tr_off = run(markets, snapshots, candle_lookup, coinalyze, gate=False)
    acct_on, tr_on = run(markets, snapshots, candle_lookup, coinalyze, gate=True)
    s_off, s_on = stats(tr_off, acct_off), stats(tr_on, acct_on)
    print(f"  gate OFF: {s_off}")
    print(f"  gate ON : {s_on}")
    print()

    print("=" * 78)
    print("L1-BUCKET CHECK (flat, gate OFF) — does 'betting against the line' lose here too?")
    print("=" * 78)
    bucket_by_l1(tf_off)
    print()

    print("=" * 78)
    print("WALK-FORWARD by week — gate ON vs OFF (parameter-free floor)")
    print("=" * 78)
    weeks = defaultdict(list)
    for m in markets:
        d = m["date"]
        if not d:
            continue
        try:
            wk = datetime.fromisoformat(d).isocalendar()[1]
        except ValueError:
            continue
        weeks[wk].append(m)
    print(f"\n  {'week':>5} {'n_mkts':>7} {'raw_off':>9} {'raw_on':>9} {'delta':>8} "
          f"{'WR_off':>7} {'WR_on':>7}")
    for wk in sorted(weeks):
        wm = weeks[wk]
        if len(wm) < 50:
            continue
        a_off, t_off = run(wm, snapshots, candle_lookup, coinalyze, gate=False)
        a_on, t_on = run(wm, snapshots, candle_lookup, coinalyze, gate=True)
        so, sn = stats(t_off, a_off), stats(t_on, a_on)
        if not so["n"] or not sn["n"]:
            continue
        print(f"  {wk:>5} {len(wm):>7} {so['raw_pnl_per_$1']:>9.2f} "
              f"{sn['raw_pnl_per_$1']:>9.2f} "
              f"{sn['raw_pnl_per_$1']-so['raw_pnl_per_$1']:>+8.2f} "
              f"{so['WR_%']:>7.1f} {sn['WR_%']:>7.1f}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
