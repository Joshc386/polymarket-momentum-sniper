"""Maker Execution Backtest — simulate limit order placement vs FOK taker.

For every entry that Bot K's signals would have taken, instead of buying
at the ask (FOK taker, 7.2% fee), simulate placing a resting limit order
at various offsets BELOW the ask and check whether the ask drops to fill
the order before resolution.

Key questions answered:
1. At each offset, what % of signals result in a fill?
2. What's the net PnL accounting for 0% maker fee vs 7.2% taker fee?
3. Where is the optimal offset that maximises risk-adjusted return?

Fee formula: fee_per_share = 0.072 x p x (1-p)  [crypto category]
Maker fee: 0% (+ 20% rebate of taker fees, not modelled here)

Usage:
    python -m backtest.maker_execution_backtest
"""

import os
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from typing import NamedTuple

# ── Constants ────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "tick_data.db")

# Fee structure — crypto category
FEE_RATE = 0.072  # 7.2% taker fee rate

# Bot K optimised weights (from walk-forward backtest)
ORACLE_W = 0.145
MOMENTUM_W = 0.524
ORDERBOOK_W = 0.331
TAKER_W = 0.131
MAX_ADJ = 0.195

# Entry thresholds (Bot G/K)
MIN_EDGE = 0.003
MAX_EDGE = 0.025
MIN_CONFIDENCE = 0.02
YES_MIN_PRICE = 0.40

# Offsets to test (cents below the ask at signal time)
# Only below-ask — above-ask is pointless because worse price eats
# more than the fee savings (max fee is ~3.6% of price at p=0.50)
OFFSETS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]

# Patience windows — how long to wait for fill (seconds)
PATIENCE_WINDOWS = [30, 60, 120, 999]  # 999 = until resolution

# Entry timing
PREFERRED_ENTRY_SECS = 180
LATEST_ENTRY_SECS = 60
EARLIEST_ENTRY_SECS = 270


def taker_fee(price: float) -> float:
    """Exact Polymarket taker fee per share for crypto category."""
    return FEE_RATE * price * (1.0 - price)


# ── Signal computation (reused from weight_optimiser) ────────────────

def compute_oracle_lag(btc_price: float, btc_open: float) -> float:
    """L1: oracle lag signal."""
    if btc_open <= 0 or btc_price <= 0:
        return 0.0
    pct_move = (btc_price - btc_open) / btc_open
    max_lag = 0.001
    if abs(pct_move) < max_lag * 0.1:
        return 0.0
    raw = pct_move / max_lag
    return max(-1.0, min(1.0, raw))


def compute_momentum_from_trades(rows: list) -> float:
    """L2: momentum from Binance 1s candles."""
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
    """L4: orderbook imbalance signal."""
    total = bid_depth + ask_depth + no_bid_depth + no_ask_depth
    if total < 1.0:
        return 0.0
    bull_depth = bid_depth + no_ask_depth
    bear_depth = ask_depth + no_bid_depth
    imb = (bull_depth - bear_depth) / total
    return max(-1.0, min(1.0, imb * 2.0))


def compute_taker_ratio_from_trades(rows: list) -> float:
    """L7: taker buy/sell ratio from Binance trades."""
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
    """Combine L1/L2/L4 core + L7 additive → est_prob_up."""
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
    """Compute EV for YES and NO sides (taker fees included)."""
    # Taker EV (what Bot G currently does)
    fee_yes = taker_fee(yes_ask)
    profit_yes = 1.0 - yes_ask - fee_yes
    ev_yes = est_prob_up * profit_yes - (1.0 - est_prob_up) * yes_ask

    fee_no = taker_fee(no_ask)
    profit_no = 1.0 - no_ask - fee_no
    ev_no = (1.0 - est_prob_up) * profit_no - est_prob_up * no_ask

    return ev_yes, ev_no


# ── Entry candidate with tick-level data ─────────────────────────────

class SignalEntry(NamedTuple):
    """A point where Bot K's signals would trigger an entry."""
    market_idx: int
    secs_remaining: float
    signal_time_unix: float    # unix timestamp of signal
    yes_ask: float             # ask price at signal time
    no_ask: float
    yes_bid: float
    no_bid: float
    est_prob_up: float
    ev_yes: float
    ev_no: float
    side: str                  # "YES" or "NO"
    entry_price: float         # the ask we'd pay as taker
    winner_is_up: bool


class FillResult(NamedTuple):
    """Result of simulating a limit order for one entry."""
    filled: bool
    fill_price: float          # limit price (or 0 if not filled)
    fill_time_secs: float      # seconds after signal when filled
    taker_price: float         # what FOK would have paid
    taker_fee_paid: float      # fee the taker would pay
    pnl_maker: float           # PnL if filled as maker (0% fee)
    pnl_taker: float           # PnL if filled as taker
    winner_is_up: bool
    side: str


# ── Main backtest logic ──────────────────────────────────────────────

def main() -> None:
    print("=" * 75)
    print("MAKER EXECUTION BACKTEST")
    print("Simulating limit order fills vs FOK taker entries")
    print("=" * 75)
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

    # Load markets
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

    # Load spot trades (for momentum/taker signals)
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
        spot_by_ts[ts_str[:19]] = row[1:]  # key by second
    print(f"  {len(spot_raw):,} spot candles loaded")

    # ── Phase 1: Find signal entries AND collect tick data per market ──
    print("\n  Finding signal entries + collecting tick paths...")

    # For each market, we need:
    # 1. The signal entry point (if any)
    # 2. ALL subsequent ticks from signal to resolution (for fill simulation)

    signal_entries = []
    tick_paths = []  # parallel to signal_entries: list of (secs_remaining, yes_bid, yes_ask, no_bid, no_ask)

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

        # Group by market
        current_mid = None
        current_snaps = []

        def process_market(mid_str, snaps):
            """Find signal entry and collect tick path for one market."""
            idx = market_id_to_idx.get(mid_str)
            if idx is None:
                return
            mkt = markets[idx]
            btc_open = mkt["btc_open"]
            winner = mkt["winner"]
            winner_is_up = winner == "Up"
            start_unix = mkt["start_unix"]

            # Find the entry point: first valid signal in the entry window
            entry_snap = None
            entry_secs_remaining = None

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

                # Compute signals
                oracle_sig = compute_oracle_lag(btc_price, btc_open)

                # Get spot trades for this timestamp
                snap_time = snap[11]
                snap_ts = snap_time[:19] if snap_time else ""
                mom_rows = []
                if snap_ts:
                    for offset in range(-10, 1):
                        try:
                            t_sec = datetime.fromisoformat(
                                snap_ts.replace("Z", "+00:00")
                            )
                            from datetime import timedelta
                            t_key = (t_sec + timedelta(seconds=offset)).strftime(
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

                # Check entry conditions
                if ev_yes >= MIN_EDGE and ev_yes <= MAX_EDGE:
                    confidence = abs(est_prob_up - 0.5) * 2.0
                    if confidence >= MIN_CONFIDENCE:
                        entry_snap = snap
                        entry_secs_remaining = secs_remaining
                        entry_data = SignalEntry(
                            market_idx=idx,
                            secs_remaining=secs_remaining,
                            signal_time_unix=start_unix + secs_into,
                            yes_ask=yes_ask,
                            no_ask=no_ask,
                            yes_bid=yes_bid,
                            no_bid=no_bid,
                            est_prob_up=est_prob_up,
                            ev_yes=ev_yes,
                            ev_no=ev_no,
                            side="YES",
                            entry_price=yes_ask,
                            winner_is_up=winner_is_up,
                        )
                        break
                elif ev_no >= MIN_EDGE and ev_no <= MAX_EDGE:
                    confidence = abs(est_prob_up - 0.5) * 2.0
                    if confidence >= MIN_CONFIDENCE:
                        entry_snap = snap
                        entry_secs_remaining = secs_remaining
                        entry_data = SignalEntry(
                            market_idx=idx,
                            secs_remaining=secs_remaining,
                            signal_time_unix=start_unix + secs_into,
                            yes_ask=yes_ask,
                            no_ask=no_ask,
                            yes_bid=yes_bid,
                            no_bid=no_bid,
                            est_prob_up=est_prob_up,
                            ev_yes=ev_yes,
                            ev_no=ev_no,
                            side="NO",
                            entry_price=no_ask,
                            winner_is_up=winner_is_up,
                        )
                        break

            if entry_snap is None:
                return

            # Collect tick path from entry onwards
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
                  f"markets processed ({len(signal_entries):,} entries found)", flush=True)

    conn.close()

    print(f"\n  Total signal entries: {len(signal_entries):,}")
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    if not signal_entries:
        print("No entries found. Check thresholds.")
        return

    # ── Phase 2: Simulate limit order fills ───────────────────────────

    print("\n" + "=" * 75)
    print("FILL SIMULATION")
    print("=" * 75)

    # For each (offset, patience) combination, simulate all entries
    results = {}  # (offset, patience) -> list of FillResult

    for offset in OFFSETS:
        for patience in PATIENCE_WINDOWS:
            fills = []
            for entry, ticks in zip(signal_entries, tick_paths):
                # Limit price = ask at signal time - offset
                if entry.side == "YES":
                    limit_price = entry.yes_ask - offset
                    taker_price = entry.yes_ask
                else:
                    limit_price = entry.no_ask - offset
                    taker_price = entry.no_ask

                if limit_price <= 0.01:
                    # Nonsensical limit price
                    fills.append(FillResult(
                        filled=False, fill_price=0.0, fill_time_secs=0.0,
                        taker_price=taker_price,
                        taker_fee_paid=taker_fee(taker_price),
                        pnl_maker=0.0, pnl_taker=0.0,
                        winner_is_up=entry.winner_is_up, side=entry.side,
                    ))
                    continue

                # Scan ticks for fill
                filled = False
                fill_time = 0.0
                for tick in ticks:
                    secs_after, secs_rem, y_bid, y_ask, n_bid, n_ask = tick

                    # Check patience window
                    if patience < 999 and secs_after > patience:
                        break

                    # Don't fill in last 5 seconds (can't place orders)
                    if secs_rem < 5:
                        break

                    # Fill condition: the ask drops to or below our limit
                    # (someone is willing to sell at our price or cheaper)
                    if entry.side == "YES":
                        if y_ask <= limit_price and y_ask > 0:
                            filled = True
                            fill_time = secs_after
                            break
                    else:
                        if n_ask <= limit_price and n_ask > 0:
                            filled = True
                            fill_time = secs_after
                            break

                # Compute PnL
                won = (entry.side == "YES" and entry.winner_is_up) or \
                      (entry.side == "NO" and not entry.winner_is_up)

                # Taker PnL (baseline)
                fee_t = taker_fee(taker_price)
                if won:
                    pnl_taker = 1.0 - taker_price - fee_t
                else:
                    pnl_taker = -taker_price - fee_t

                # Maker PnL (0% fee)
                if filled:
                    if won:
                        pnl_maker = 1.0 - limit_price  # no fee
                    else:
                        pnl_maker = -limit_price  # no fee
                else:
                    pnl_maker = 0.0  # no fill = no trade = no PnL

                fills.append(FillResult(
                    filled=filled,
                    fill_price=limit_price if filled else 0.0,
                    fill_time_secs=fill_time,
                    taker_price=taker_price,
                    taker_fee_paid=fee_t,
                    pnl_maker=pnl_maker,
                    pnl_taker=pnl_taker,
                    winner_is_up=entry.winner_is_up,
                    side=entry.side,
                ))

            results[(offset, patience)] = fills

    # ── Phase 3: Report ───────────────────────────────────────────────

    total_entries = len(signal_entries)

    # Taker baseline (same for all combos)
    all_taker_pnl = sum(r.pnl_taker for r in results[(OFFSETS[0], PATIENCE_WINDOWS[0])])
    avg_taker_fee = sum(r.taker_fee_paid for r in results[(OFFSETS[0], PATIENCE_WINDOWS[0])]) / total_entries
    taker_wr = sum(1 for r in results[(OFFSETS[0], PATIENCE_WINDOWS[0])] if r.pnl_taker > 0) / total_entries

    print(f"\n{'TAKER BASELINE (FOK):':}")
    print(f"  Entries: {total_entries:,}")
    print(f"  Total PnL: ${all_taker_pnl:+.2f}")
    print(f"  Avg PnL/trade: ${all_taker_pnl / total_entries:+.4f}")
    print(f"  Win rate: {taker_wr:.1%}")
    print(f"  Avg fee/trade: ${avg_taker_fee:.4f}")
    print(f"  Total fees paid: ${avg_taker_fee * total_entries:.2f}")

    # Results table
    print(f"\n{'=' * 100}")
    print(f"{'MAKER RESULTS BY OFFSET AND PATIENCE':}")
    print(f"{'=' * 100}")

    header = (f"{'Offset':>7} {'Patience':>9} {'Fill%':>7} {'Fills':>7} "
              f"{'MakerPnL':>10} {'TakerPnL':>10} {'d_PnL':>10} "
              f"{'Avg$/fill':>10} {'MakerWR':>8} {'AvgFillT':>9}")
    print(header)
    print("-" * 100)

    best_delta = -999
    best_combo = None

    for patience in PATIENCE_WINDOWS:
        for offset in OFFSETS:
            fills = results[(offset, patience)]
            n_filled = sum(1 for f in fills if f.filled)
            fill_rate = n_filled / total_entries if total_entries > 0 else 0
            maker_pnl = sum(f.pnl_maker for f in fills)
            taker_pnl = sum(f.pnl_taker for f in fills)
            delta_pnl = maker_pnl - taker_pnl

            maker_wins = sum(1 for f in fills if f.filled and f.pnl_maker > 0)
            maker_wr = maker_wins / n_filled if n_filled > 0 else 0

            avg_pnl_per_fill = maker_pnl / n_filled if n_filled > 0 else 0

            filled_fills = [f for f in fills if f.filled]
            avg_fill_time = (sum(f.fill_time_secs for f in filled_fills)
                            / len(filled_fills)) if filled_fills else 0

            patience_str = f"{patience}s" if patience < 999 else "all"
            print(f"${offset:.2f}   {patience_str:>9} {fill_rate:>7.1%} "
                  f"{n_filled:>7,} {maker_pnl:>+10.2f} {taker_pnl:>+10.2f} "
                  f"{delta_pnl:>+10.2f} {avg_pnl_per_fill:>+10.4f} "
                  f"{maker_wr:>7.1%} {avg_fill_time:>8.1f}s")

            if delta_pnl > best_delta:
                best_delta = delta_pnl
                best_combo = (offset, patience, fill_rate, n_filled,
                              maker_pnl, maker_wr, avg_fill_time)

        print()  # Blank line between patience groups

    # ── Best result ───────────────────────────────────────────────────
    if best_combo:
        offset, patience, fill_rate, n_filled, maker_pnl, maker_wr, avg_ft = best_combo
        patience_str = f"{patience}s" if patience < 999 else "until resolution"
        print("=" * 75)
        print("OPTIMAL MAKER STRATEGY")
        print("=" * 75)
        print(f"  Offset: ${offset:.2f} below ask")
        print(f"  Patience: {patience_str}")
        print(f"  Fill rate: {fill_rate:.1%} ({n_filled:,}/{total_entries:,})")
        print(f"  Maker PnL: ${maker_pnl:+.2f}")
        print(f"  Taker PnL: ${all_taker_pnl:+.2f}")
        print(f"  Improvement: ${best_delta:+.2f} ({best_delta/abs(all_taker_pnl)*100:+.1f}%)" if all_taker_pnl != 0 else f"  Improvement: ${best_delta:+.2f}")
        print(f"  Maker win rate: {maker_wr:.1%}")
        print(f"  Avg fill time: {avg_ft:.1f}s after signal")

    # ── Price improvement analysis ────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("PRICE IMPROVEMENT BREAKDOWN (patience=all)")
    print(f"{'=' * 75}")

    for offset in OFFSETS:
        fills = results[(offset, 999)]
        filled = [f for f in fills if f.filled]
        if not filled:
            print(f"  ${offset:.2f}: no fills")
            continue

        avg_improvement = sum(f.taker_price - f.fill_price for f in filled) / len(filled)
        avg_fee_saved = sum(f.taker_fee_paid for f in filled) / len(filled)
        total_saved = sum(f.taker_fee_paid for f in filled)
        total_improvement = sum(f.taker_price - f.fill_price for f in filled)

        print(f"  ${offset:.2f}: {len(filled):,} fills | "
              f"avg price improvement: ${avg_improvement:.4f}/share | "
              f"avg fee saved: ${avg_fee_saved:.4f}/share | "
              f"total saved (fees+price): ${total_saved + total_improvement:.2f}")

    # ── Missed trade analysis ─────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("MISSED TRADE ANALYSIS (patience=all)")
    print(f"{'=' * 75}")
    print("What happened to trades that didn't fill as maker?")

    for offset in OFFSETS:
        fills = results[(offset, 999)]
        missed = [f for f in fills if not f.filled]
        if not missed:
            print(f"  ${offset:.2f}: all trades filled!")
            continue

        missed_wins = sum(1 for f in missed if f.pnl_taker > 0)
        missed_losses = len(missed) - missed_wins
        missed_pnl = sum(f.pnl_taker for f in missed)

        print(f"  ${offset:.2f}: {len(missed):,} missed | "
              f"would have won {missed_wins} / lost {missed_losses} | "
              f"missed PnL: ${missed_pnl:+.2f} "
              f"({'good' if missed_pnl < 0 else 'bad'} to miss)")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
