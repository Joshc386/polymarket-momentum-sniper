"""Maker Fallback Backtest — maker-then-taker and theoretical ceiling.

Two strategies tested against the taker baseline:

1. THEORETICAL CEILING (offset $0.00): every entry fills at the taker's
   price but with 0% maker fee. Shows the maximum possible fee savings.
   Not achievable in practice (post-only at the ask is rejected).

2. MAKER-THEN-TAKER FALLBACK: place a maker order below the ask for
   N seconds. If filled -> 0% fee at the better price. If NOT filled ->
   fall back to FOK taker at the CURRENT ask (which may have moved).
   This solves the adverse selection problem: you never miss a trade,
   you just sometimes get better economics.

Fee formula: fee_per_share = 0.072 x p x (1-p)  [crypto category]
Maker fee: 0% (+ 20% rebate of taker fee pool, not modelled here)

Usage:
    python -m backtest.maker_fallback_backtest
"""

import os
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import NamedTuple

# ── Constants ────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "tick_data.db")

# Fee structure -- crypto category
FEE_RATE = 0.072

# Bot K optimised weights
ORACLE_W = 0.145
MOMENTUM_W = 0.524
ORDERBOOK_W = 0.331
TAKER_W = 0.131
MAX_ADJ = 0.195

# Entry thresholds
MIN_EDGE = 0.003
MAX_EDGE = 0.025
MIN_CONFIDENCE = 0.02
YES_MIN_PRICE = 0.40

# Maker-then-taker parameters
OFFSETS = [0.01, 0.02, 0.03, 0.05]
PATIENCE_WINDOWS = [5, 10, 15, 20, 30]

# Entry timing
PREFERRED_ENTRY_SECS = 180
LATEST_ENTRY_SECS = 60
EARLIEST_ENTRY_SECS = 270


def taker_fee(price: float) -> float:
    """Exact Polymarket taker fee per share for crypto category."""
    return FEE_RATE * price * (1.0 - price)


# ── Signal computation (identical to maker_execution_backtest) ───────

def compute_oracle_lag(btc_price: float, btc_open: float) -> float:
    if btc_open <= 0 or btc_price <= 0:
        return 0.0
    pct_move = (btc_price - btc_open) / btc_open
    max_lag = 0.001
    if abs(pct_move) < max_lag * 0.1:
        return 0.0
    raw = pct_move / max_lag
    return max(-1.0, min(1.0, raw))


def compute_momentum_from_trades(rows: list) -> float:
    if len(rows) < 3:
        return 0.0
    recent = rows[-10:] if len(rows) >= 10 else rows
    o, c, h, l = recent[-1][0], recent[-1][1], recent[-1][2], recent[-1][3]
    rng = h - l
    if rng < 1e-9:
        return 0.0
    body_ratio = abs(c - o) / rng
    direction = 1.0 if c > o else (-1.0 if c < o else 0.0)
    roc = 0.0
    if len(recent) >= 2:
        prev_c = recent[-2][1]
        if prev_c > 0:
            roc = (c - prev_c) / prev_c * 1000
    total_buy = sum(r[5] for r in recent if len(r) > 5)
    total_sell = sum(r[6] for r in recent if len(r) > 6)
    total_vol = total_buy + total_sell
    vol_signal = 0.0
    if total_vol > 0:
        vol_signal = (total_buy / total_vol - 0.5) * 2.0
    rsi_val = 0.0
    if len(recent) >= 5:
        gains, losses = [], []
        for i in range(1, len(recent)):
            chg = recent[i][1] - recent[i - 1][1]
            if chg > 0:
                gains.append(chg)
            else:
                losses.append(abs(chg))
        avg_gain = sum(gains) / len(recent) if gains else 0.0
        avg_loss = sum(losses) / len(recent) if losses else 0.0
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - 100 / (1 + rs)
            rsi_val = (rsi - 50) / 50
    raw = (0.30 * roc + 0.25 * direction + 0.25 * vol_signal
           + 0.10 * body_ratio * direction + 0.10 * rsi_val)
    return max(-1.0, min(1.0, raw))


def compute_ob_imbalance(
    bid_depth: float, ask_depth: float,
    no_bid_depth: float, no_ask_depth: float,
) -> float:
    total = bid_depth + ask_depth + no_bid_depth + no_ask_depth
    if total < 1.0:
        return 0.0
    bull_depth = bid_depth + no_ask_depth
    bear_depth = ask_depth + no_bid_depth
    imb = (bull_depth - bear_depth) / total
    return max(-1.0, min(1.0, imb * 2.0))


def compute_taker_ratio_from_trades(rows: list) -> float:
    if len(rows) < 3:
        return 0.0
    window = rows[-30:] if len(rows) >= 30 else rows
    total_buy = sum(r[5] for r in window if len(r) > 5)
    total_sell = sum(r[6] for r in window if len(r) > 6)
    total = total_buy + total_sell
    if total < 1e-9:
        return 0.0
    dev = total_buy / total - 0.5
    nb = 0.05
    if abs(dev) < nb:
        return 0.0
    if dev > 0:
        return min((dev - nb) / (0.5 - nb), 1.0)
    return max((dev + nb) / (0.5 - nb), -1.0)


def combine_signals(
    oracle_sig: float, momentum_sig: float,
    ob_sig: float, taker_sig: float,
) -> float:
    total_core = ORACLE_W + MOMENTUM_W + ORDERBOOK_W
    if total_core < 1e-9:
        return 0.5
    core = (
        ORACLE_W * oracle_sig
        + MOMENTUM_W * momentum_sig
        + ORDERBOOK_W * ob_sig
    ) / total_core
    additive = TAKER_W * taker_sig
    combined = core + min(MAX_ADJ, max(-MAX_ADJ, additive))
    combined = max(-1.0, min(1.0, combined))
    return 0.5 + combined * 0.5


def compute_ev(est_prob_up: float, yes_ask: float, no_ask: float) -> tuple:
    fee_yes = taker_fee(yes_ask)
    profit_yes = 1.0 - yes_ask - fee_yes
    ev_yes = est_prob_up * profit_yes - (1.0 - est_prob_up) * yes_ask

    fee_no = taker_fee(no_ask)
    profit_no = 1.0 - no_ask - fee_no
    ev_no = (1.0 - est_prob_up) * profit_no - est_prob_up * no_ask

    return ev_yes, ev_no


# ── Data structures ──────────────────────────────────────────────────

class SignalEntry(NamedTuple):
    market_idx: int
    secs_remaining: float
    signal_time_unix: float
    yes_ask: float
    no_ask: float
    yes_bid: float
    no_bid: float
    est_prob_up: float
    ev_yes: float
    ev_no: float
    side: str
    entry_price: float
    winner_is_up: bool


class FallbackResult(NamedTuple):
    """Result of maker-then-taker fallback for one entry."""
    maker_filled: bool
    fill_price: float       # maker price if filled, taker fallback price if not
    fill_time_secs: float   # seconds after signal (0 if taker fallback)
    original_ask: float     # ask at signal time
    fee_paid: float         # 0 if maker, taker_fee(fallback_ask) if taker
    pnl: float
    execution_type: str     # "maker", "taker_fallback", "taker_baseline"
    winner_is_up: bool
    side: str
    fallback_ask: float     # ask at fallback time (0 if maker filled)
    ask_slippage: float     # fallback_ask - original_ask (cost of waiting)


# ── Main backtest logic ──────────────────────────────────────────────

def main() -> None:
    print("=" * 80)
    print("MAKER FALLBACK BACKTEST")
    print("Strategy: maker order for N seconds, then FOK taker if not filled")
    print("=" * 80)
    print(f"\nFee formula: {FEE_RATE} x p x (1-p)  [crypto category]")
    print(f"At p=0.45: ${taker_fee(0.45):.4f}/share ({taker_fee(0.45)/0.45*100:.1f}% of price)")
    print(f"At p=0.50: ${taker_fee(0.50):.4f}/share ({taker_fee(0.50)/0.50*100:.1f}% of price)")
    print(f"Offsets tested: {OFFSETS}")
    print(f"Patience windows: {PATIENCE_WINDOWS}s")
    print()

    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = None
    cur = conn.cursor()

    # ── Load markets ─────────────────────────────────────────────────
    cur.execute("""
        SELECT market_id, start_time, btc_price_start, btc_price_end, winner
        FROM markets
        WHERE winner IN ('Up', 'Down')
        ORDER BY start_time
    """)
    markets_raw = cur.fetchall()
    markets = []
    market_id_to_idx = {}
    for i, (mid, st, btc_open, btc_close, winner) in enumerate(markets_raw):
        mid_str = str(mid)
        try:
            dt = datetime.fromisoformat(st.replace("Z", "+00:00"))
            start_unix = int(dt.timestamp())
        except Exception:
            continue
        markets.append({
            "market_id": mid_str,
            "date": dt.date(),
            "winner": winner,
            "btc_open": btc_open or 0.0,
            "start_unix": start_unix,
        })
        market_id_to_idx[mid_str] = len(markets) - 1
    print(f"  {len(markets):,} markets loaded")

    # ── Load spot trades ─────────────────────────────────────────────
    cur.execute("""
        SELECT timestamp, price_open, price_close, price_high, price_low,
               total_volume, aggressive_buy_volume, aggressive_sell_volume,
               num_trades
        FROM spot_trades ORDER BY timestamp
    """)
    spot_raw = cur.fetchall()
    spot_by_ts = {}
    for row in spot_raw:
        ts_str = row[0]
        spot_by_ts[ts_str[:19]] = row[1:]
    print(f"  {len(spot_raw):,} spot candles loaded")

    # ── Find signal entries + tick paths ──────────────────────────────
    print("\n  Finding signal entries + collecting tick paths...")

    signal_entries = []
    tick_paths = []

    batch_size = 500
    market_ids = [m["market_id"] for m in markets]

    for batch_start in range(0, len(market_ids), batch_size):
        batch = market_ids[batch_start:batch_start + batch_size]
        placeholders = ",".join("?" * len(batch))

        cur.execute(f"""
            SELECT market_id, seconds_into_window, btc_price,
                   up_best_bid, up_best_ask, down_best_bid, down_best_ask,
                   up_bid_depth_5, up_ask_depth_5,
                   down_bid_depth_5, down_ask_depth_5, time
            FROM snapshots
            WHERE market_id IN ({placeholders})
            ORDER BY market_id, seconds_into_window
        """, batch)

        current_mid = None
        current_snaps = []

        def process_market(mid_str, snaps):
            idx = market_id_to_idx.get(mid_str)
            if idx is None:
                return
            mkt = markets[idx]
            btc_open = mkt["btc_open"]
            winner = mkt["winner"]
            winner_is_up = winner == "Up"
            start_unix = mkt["start_unix"]

            entry_snap = None

            for snap in snaps:
                secs_into = snap[1]
                secs_remaining = 300.0 - secs_into
                if secs_remaining > EARLIEST_ENTRY_SECS or secs_remaining < LATEST_ENTRY_SECS:
                    continue

                btc_price = snap[2] or 0.0
                yes_bid = snap[3] or 0.0
                yes_ask = snap[4] or 0.0
                no_bid = snap[5] or 0.0
                no_ask = snap[6] or 0.0
                bid_depth = snap[7] or 0.0
                ask_depth = snap[8] or 0.0
                no_bid_depth = snap[9] or 0.0
                no_ask_depth = snap[10] or 0.0

                if yes_ask < YES_MIN_PRICE or yes_ask > 0.99:
                    continue
                if yes_bid <= 0 or no_bid <= 0:
                    continue

                oracle_sig = compute_oracle_lag(btc_price, btc_open)

                snap_time = snap[11]
                snap_ts = snap_time[:19] if snap_time else ""
                mom_rows = []
                if snap_ts:
                    for offset_s in range(-10, 1):
                        try:
                            t_sec = datetime.fromisoformat(
                                snap_ts.replace("Z", "+00:00")
                            )
                            t_key = (t_sec + timedelta(seconds=offset_s)).strftime(
                                "%Y-%m-%dT%H:%M:%S"
                            )
                            if t_key in spot_by_ts:
                                mom_rows.append(spot_by_ts[t_key])
                        except Exception:
                            pass

                momentum_sig = compute_momentum_from_trades(mom_rows)
                ob_sig = compute_ob_imbalance(
                    bid_depth, ask_depth, no_bid_depth, no_ask_depth
                )
                taker_sig = compute_taker_ratio_from_trades(mom_rows)
                est_prob_up = combine_signals(
                    oracle_sig, momentum_sig, ob_sig, taker_sig
                )
                ev_yes, ev_no = compute_ev(est_prob_up, yes_ask, no_ask)

                if ev_yes >= MIN_EDGE and ev_yes <= MAX_EDGE:
                    confidence = abs(est_prob_up - 0.5) * 2.0
                    if confidence >= MIN_CONFIDENCE:
                        entry_snap = snap
                        entry_data = SignalEntry(
                            market_idx=idx,
                            secs_remaining=secs_remaining,
                            signal_time_unix=start_unix + secs_into,
                            yes_ask=yes_ask, no_ask=no_ask,
                            yes_bid=yes_bid, no_bid=no_bid,
                            est_prob_up=est_prob_up,
                            ev_yes=ev_yes, ev_no=ev_no,
                            side="YES", entry_price=yes_ask,
                            winner_is_up=winner_is_up,
                        )
                        break
                elif ev_no >= MIN_EDGE and ev_no <= MAX_EDGE:
                    confidence = abs(est_prob_up - 0.5) * 2.0
                    if confidence >= MIN_CONFIDENCE:
                        entry_snap = snap
                        entry_data = SignalEntry(
                            market_idx=idx,
                            secs_remaining=secs_remaining,
                            signal_time_unix=start_unix + secs_into,
                            yes_ask=yes_ask, no_ask=no_ask,
                            yes_bid=yes_bid, no_bid=no_bid,
                            est_prob_up=est_prob_up,
                            ev_yes=ev_yes, ev_no=ev_no,
                            side="NO", entry_price=no_ask,
                            winner_is_up=winner_is_up,
                        )
                        break

            if entry_snap is None:
                return

            entry_secs_into = entry_snap[1]
            ticks = []
            for snap in snaps:
                if snap[1] < entry_secs_into:
                    continue
                secs_after_signal = snap[1] - entry_secs_into
                secs_remaining = 300.0 - snap[1]
                yes_bid = snap[3] or 0.0
                yes_ask = snap[4] or 0.0
                no_bid = snap[5] or 0.0
                no_ask = snap[6] or 0.0
                ticks.append((secs_after_signal, secs_remaining,
                              yes_bid, yes_ask, no_bid, no_ask))

            if len(ticks) >= 2:
                signal_entries.append(entry_data)
                tick_paths.append(ticks)

        for row in cur:
            mid_str = str(row[0])
            if mid_str != current_mid:
                if current_mid is not None:
                    process_market(current_mid, current_snaps)
                current_mid = mid_str
                current_snaps = [row]
            else:
                current_snaps.append(row)

        if current_mid is not None:
            process_market(current_mid, current_snaps)
            current_mid = None
            current_snaps = []

        if (batch_start + batch_size) % 2000 == 0 or batch_start + batch_size >= len(market_ids):
            print(f"    {min(batch_start + batch_size, len(market_ids)):,}/{len(market_ids):,} "
                  f"markets processed ({len(signal_entries):,} entries found)",
                  flush=True)

    conn.close()

    total = len(signal_entries)
    print(f"\n  Total signal entries: {total:,}")
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    if not signal_entries:
        print("No entries found. Check thresholds.")
        return

    # ── Compute taker baseline PnL ───────────────────────────────────

    def compute_pnl(entry_price: float, fee: float, won: bool) -> float:
        if won:
            return 1.0 - entry_price - fee
        else:
            return -entry_price - fee

    taker_pnls = []
    taker_fees_total = 0.0
    taker_wins = 0
    for entry in signal_entries:
        won = (entry.side == "YES" and entry.winner_is_up) or \
              (entry.side == "NO" and not entry.winner_is_up)
        fee = taker_fee(entry.entry_price)
        pnl = compute_pnl(entry.entry_price, fee, won)
        taker_pnls.append(pnl)
        taker_fees_total += fee
        if won:
            taker_wins += 1

    taker_total_pnl = sum(taker_pnls)
    taker_wr = taker_wins / total

    # ── Strategy 1: Theoretical ceiling (0% fee at taker price) ──────

    ceiling_pnls = []
    ceiling_fee_savings = 0.0
    for entry in signal_entries:
        won = (entry.side == "YES" and entry.winner_is_up) or \
              (entry.side == "NO" and not entry.winner_is_up)
        # Same price as taker, but 0% fee
        pnl = compute_pnl(entry.entry_price, 0.0, won)
        ceiling_pnls.append(pnl)
        ceiling_fee_savings += taker_fee(entry.entry_price)

    ceiling_total_pnl = sum(ceiling_pnls)

    # ── Strategy 2: Maker-then-taker fallback ────────────────────────

    print("\n  Simulating maker-then-taker fallback...")

    # Results: (offset, patience) -> list of FallbackResult
    fallback_results = {}

    for offset in OFFSETS:
        for patience in PATIENCE_WINDOWS:
            results_list = []

            for entry, ticks in zip(signal_entries, tick_paths):
                won = (entry.side == "YES" and entry.winner_is_up) or \
                      (entry.side == "NO" and not entry.winner_is_up)

                # Maker limit price
                if entry.side == "YES":
                    limit_price = entry.yes_ask - offset
                    original_ask = entry.yes_ask
                else:
                    limit_price = entry.no_ask - offset
                    original_ask = entry.no_ask

                if limit_price <= 0.01:
                    # Nonsensical limit -- just taker
                    fee = taker_fee(original_ask)
                    pnl = compute_pnl(original_ask, fee, won)
                    results_list.append(FallbackResult(
                        maker_filled=False,
                        fill_price=original_ask,
                        fill_time_secs=0.0,
                        original_ask=original_ask,
                        fee_paid=fee,
                        pnl=pnl,
                        execution_type="taker_fallback",
                        winner_is_up=entry.winner_is_up,
                        side=entry.side,
                        fallback_ask=original_ask,
                        ask_slippage=0.0,
                    ))
                    continue

                # Phase 1: Try maker fill within patience window
                maker_filled = False
                fill_time = 0.0

                # Track the ask at the end of patience window for fallback
                fallback_ask = original_ask  # default if no ticks in window
                last_tick_in_window = None

                for tick in ticks:
                    secs_after, secs_rem, y_bid, y_ask, n_bid, n_ask = tick

                    # Skip tick 0 (signal tick itself)
                    if secs_after < 0.5:
                        continue

                    # Still in patience window?
                    if secs_after <= patience:
                        # Update fallback ask to latest known
                        if entry.side == "YES" and y_ask > 0:
                            fallback_ask = y_ask
                        elif entry.side == "NO" and n_ask > 0:
                            fallback_ask = n_ask
                        last_tick_in_window = tick

                        # Check maker fill: ask drops to our limit
                        if entry.side == "YES":
                            if y_ask <= limit_price and y_ask > 0:
                                maker_filled = True
                                fill_time = secs_after
                                break
                        else:
                            if n_ask <= limit_price and n_ask > 0:
                                maker_filled = True
                                fill_time = secs_after
                                break
                    else:
                        # Past patience -- find the first tick after patience
                        # for the fallback ask
                        if entry.side == "YES" and y_ask > 0:
                            fallback_ask = y_ask
                        elif entry.side == "NO" and n_ask > 0:
                            fallback_ask = n_ask
                        break

                if maker_filled:
                    # Maker fill: 0% fee at limit_price
                    pnl = compute_pnl(limit_price, 0.0, won)
                    results_list.append(FallbackResult(
                        maker_filled=True,
                        fill_price=limit_price,
                        fill_time_secs=fill_time,
                        original_ask=original_ask,
                        fee_paid=0.0,
                        pnl=pnl,
                        execution_type="maker",
                        winner_is_up=entry.winner_is_up,
                        side=entry.side,
                        fallback_ask=0.0,
                        ask_slippage=0.0,
                    ))
                else:
                    # Phase 2: Taker fallback at current ask
                    # If fallback ask is unreasonable, use original
                    if fallback_ask <= 0.01 or fallback_ask > 0.99:
                        fallback_ask = original_ask

                    fee = taker_fee(fallback_ask)
                    pnl = compute_pnl(fallback_ask, fee, won)
                    slippage = fallback_ask - original_ask

                    results_list.append(FallbackResult(
                        maker_filled=False,
                        fill_price=fallback_ask,
                        fill_time_secs=0.0,
                        original_ask=original_ask,
                        fee_paid=fee,
                        pnl=pnl,
                        execution_type="taker_fallback",
                        winner_is_up=entry.winner_is_up,
                        side=entry.side,
                        fallback_ask=fallback_ask,
                        ask_slippage=slippage,
                    ))

            fallback_results[(offset, patience)] = results_list

    # ── Report ───────────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Baseline
    print(f"\n--- TAKER BASELINE (FOK at signal ask) ---")
    print(f"  Entries:        {total:,}")
    print(f"  Total PnL:      ${taker_total_pnl:+.2f}")
    print(f"  Avg PnL/trade:  ${taker_total_pnl / total:+.4f}")
    print(f"  Win rate:       {taker_wr:.1%}")
    print(f"  Total fees:     ${taker_fees_total:.2f}")
    print(f"  Avg fee/trade:  ${taker_fees_total / total:.4f}")

    # Theoretical ceiling
    print(f"\n--- THEORETICAL CEILING (0% fee at taker price) ---")
    print(f"  Total PnL:      ${ceiling_total_pnl:+.2f}")
    print(f"  Avg PnL/trade:  ${ceiling_total_pnl / total:+.4f}")
    print(f"  Fee savings:    ${ceiling_fee_savings:.2f}")
    print(f"  PnL uplift:     ${ceiling_total_pnl - taker_total_pnl:+.2f}")
    if taker_total_pnl != 0:
        print(f"  % improvement:  {(ceiling_total_pnl - taker_total_pnl) / abs(taker_total_pnl) * 100:+.1f}%")

    # Maker-then-taker results
    print(f"\n{'=' * 80}")
    print("MAKER-THEN-TAKER FALLBACK RESULTS")
    print(f"{'=' * 80}")

    header = (
        f"{'Offset':>7} {'Wait':>5} "
        f"{'MkrFill%':>8} {'MkrFills':>8} "
        f"{'TotalPnL':>10} {'d_vs_Taker':>11} {'d_vs_Ceil':>10} "
        f"{'AvgSlip':>8} {'WR':>6} {'AvgFillT':>8}"
    )
    print(header)
    print("-" * 95)

    best_pnl_delta = -9999
    best_combo = None

    for patience in PATIENCE_WINDOWS:
        for offset in OFFSETS:
            rlist = fallback_results[(offset, patience)]

            n_maker = sum(1 for r in rlist if r.maker_filled)
            maker_rate = n_maker / total
            total_pnl = sum(r.pnl for r in rlist)
            delta_vs_taker = total_pnl - taker_total_pnl
            delta_vs_ceiling = total_pnl - ceiling_total_pnl

            # Win rate
            wins = sum(1 for r in rlist if r.pnl > 0)
            wr = wins / total

            # Average slippage on taker fallbacks
            taker_fallbacks = [r for r in rlist if not r.maker_filled]
            avg_slip = 0.0
            if taker_fallbacks:
                avg_slip = sum(r.ask_slippage for r in taker_fallbacks) / len(taker_fallbacks)

            # Average maker fill time
            maker_fills = [r for r in rlist if r.maker_filled]
            avg_fill_t = 0.0
            if maker_fills:
                avg_fill_t = sum(r.fill_time_secs for r in maker_fills) / len(maker_fills)

            print(
                f"${offset:.2f}   {patience:>3}s "
                f"{maker_rate:>8.1%} {n_maker:>8,} "
                f"{total_pnl:>+10.2f} {delta_vs_taker:>+11.2f} {delta_vs_ceiling:>+10.2f} "
                f"${avg_slip:>+7.4f} {wr:>5.1%} {avg_fill_t:>7.1f}s"
            )

            if delta_vs_taker > best_pnl_delta:
                best_pnl_delta = delta_vs_taker
                best_combo = {
                    "offset": offset,
                    "patience": patience,
                    "maker_rate": maker_rate,
                    "n_maker": n_maker,
                    "total_pnl": total_pnl,
                    "delta": delta_vs_taker,
                    "avg_slip": avg_slip,
                    "wr": wr,
                    "avg_fill_t": avg_fill_t,
                }

        print()  # blank between patience groups

    # ── Optimal result ───────────────────────────────────────────────

    if best_combo:
        print("=" * 80)
        print("OPTIMAL MAKER-THEN-TAKER STRATEGY")
        print("=" * 80)
        bc = best_combo
        print(f"  Offset:           ${bc['offset']:.2f} below ask")
        print(f"  Patience:         {bc['patience']}s")
        print(f"  Maker fill rate:  {bc['maker_rate']:.1%} ({bc['n_maker']:,}/{total:,})")
        print(f"  Total PnL:        ${bc['total_pnl']:+.2f}")
        print(f"  vs Taker:         ${bc['delta']:+.2f}")
        print(f"  vs Ceiling:       ${bc['total_pnl'] - ceiling_total_pnl:+.2f}")
        print(f"  Win rate:         {bc['wr']:.1%}")
        print(f"  Avg fill time:    {bc['avg_fill_t']:.1f}s")
        print(f"  Avg slip on FB:   ${bc['avg_slip']:+.4f}")

    # ── Detailed breakdown for best combo ────────────────────────────

    if best_combo:
        bc = best_combo
        rlist = fallback_results[(bc["offset"], bc["patience"])]

        print(f"\n{'=' * 80}")
        print(f"DETAILED BREAKDOWN: ${bc['offset']:.2f} offset, {bc['patience']}s patience")
        print(f"{'=' * 80}")

        # Maker fills: analyse PnL
        maker_fills = [r for r in rlist if r.maker_filled]
        taker_fbs = [r for r in rlist if not r.maker_filled]

        if maker_fills:
            mk_pnl = sum(r.pnl for r in maker_fills)
            mk_wins = sum(1 for r in maker_fills if r.pnl > 0)
            mk_wr = mk_wins / len(maker_fills)

            # What would those same trades have been as taker?
            mk_as_taker_pnl = 0.0
            for r in maker_fills:
                won = r.pnl > 0  # crude but correct for binaries
                fee = taker_fee(r.original_ask)
                if won:
                    mk_as_taker_pnl += 1.0 - r.original_ask - fee
                else:
                    mk_as_taker_pnl += -r.original_ask - fee

            print(f"\n  MAKER FILLS ({len(maker_fills):,} trades):")
            print(f"    PnL as maker:   ${mk_pnl:+.2f}")
            print(f"    PnL as taker:   ${mk_as_taker_pnl:+.2f}")
            print(f"    Fee savings:    ${mk_pnl - mk_as_taker_pnl:+.2f}")
            print(f"    Win rate:       {mk_wr:.1%}")
            print(f"    Avg fill time:  {sum(r.fill_time_secs for r in maker_fills) / len(maker_fills):.1f}s")
            # Price improvement (maker got a better price)
            avg_price_imp = sum(r.original_ask - r.fill_price for r in maker_fills) / len(maker_fills)
            print(f"    Avg price improvement: ${avg_price_imp:.4f}/share")

        if taker_fbs:
            fb_pnl = sum(r.pnl for r in taker_fbs)
            fb_wins = sum(1 for r in taker_fbs if r.pnl > 0)
            fb_wr = fb_wins / len(taker_fbs)

            # Slippage analysis
            slip_positive = [r for r in taker_fbs if r.ask_slippage > 0.001]
            slip_negative = [r for r in taker_fbs if r.ask_slippage < -0.001]
            slip_neutral = [r for r in taker_fbs if abs(r.ask_slippage) <= 0.001]

            print(f"\n  TAKER FALLBACKS ({len(taker_fbs):,} trades):")
            print(f"    PnL:            ${fb_pnl:+.2f}")
            print(f"    Win rate:       {fb_wr:.1%}")
            print(f"    Avg slippage:   ${sum(r.ask_slippage for r in taker_fbs)/len(taker_fbs):+.4f}")
            print(f"    Ask moved up:   {len(slip_positive):,} ({len(slip_positive)/len(taker_fbs):.1%}) -- paid more")
            print(f"    Ask moved down: {len(slip_negative):,} ({len(slip_negative)/len(taker_fbs):.1%}) -- paid less")
            print(f"    Ask unchanged:  {len(slip_neutral):,} ({len(slip_neutral)/len(taker_fbs):.1%})")

            if slip_positive:
                avg_up = sum(r.ask_slippage for r in slip_positive) / len(slip_positive)
                print(f"    When up:        avg +${avg_up:.4f}/share")
            if slip_negative:
                avg_dn = sum(r.ask_slippage for r in slip_negative) / len(slip_negative)
                print(f"    When down:      avg ${avg_dn:.4f}/share")

        # ── Are the maker fills biased? ──────────────────────────────
        print(f"\n  ADVERSE SELECTION CHECK:")
        if maker_fills:
            mk_win_pct = sum(1 for r in maker_fills if r.pnl > 0) / len(maker_fills)
            fb_win_pct = sum(1 for r in taker_fbs if r.pnl > 0) / len(taker_fbs) if taker_fbs else 0
            print(f"    Maker fill WR:     {mk_win_pct:.1%}")
            print(f"    Taker fallback WR: {fb_win_pct:.1%}")
            print(f"    Baseline WR:       {taker_wr:.1%}")
            if mk_win_pct < fb_win_pct:
                print(f"    >> Maker fills have LOWER WR than fallbacks (adverse selection present)")
            else:
                print(f"    >> Maker fills have HIGHER WR than fallbacks (no adverse selection)")

    # ── Summary comparison ───────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 80}")
    print(f"  {'Strategy':<35} {'PnL':>10} {'vs Taker':>10} {'WR':>7}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*7}")
    print(f"  {'Taker baseline (current)':<35} ${taker_total_pnl:>+9.2f} {'---':>10} {taker_wr:>6.1%}")
    print(f"  {'Theoretical ceiling (0% fee)':<35} ${ceiling_total_pnl:>+9.2f} ${ceiling_total_pnl-taker_total_pnl:>+9.2f} {taker_wr:>6.1%}")
    if best_combo:
        bc = best_combo
        lbl = f"Maker-taker ${bc['offset']:.2f}/{bc['patience']}s"
        print(f"  {lbl:<35} ${bc['total_pnl']:>+9.2f} ${bc['delta']:>+9.2f} {bc['wr']:>6.1%}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
