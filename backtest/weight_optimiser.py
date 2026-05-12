"""Walk-forward signal weight optimiser for L1/L2/L4/L7.

Phase 1: Precompute signal values for every market's best entry point.
          Stores as lightweight tuples — ~14 floats per market, not 42M objects.
Phase 2: Grid search over weight combinations using only the precomputed signals.
          Each weight combo is pure arithmetic — no snapshot re-reading.

Walk-forward: 7-day train / 5-day test folds, scored by Sharpe ratio.
Uses Bot G's entry thresholds (min_edge=0.003, max_edge=0.025).
Results applied to Bot K's config only.

Usage:
    python -m backtest.weight_optimiser
    python -m backtest.weight_optimiser --train-days 7 --test-days 5
"""

import itertools
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "tick_data.db")

# Bot G entry thresholds
MIN_EDGE = 0.003
MAX_EDGE = 0.025
FEE_RATE = 0.02
MIN_CONFIDENCE = 0.02
EARLIEST_ENTRY_SECS = 270
LATEST_ENTRY_SECS = 60
YES_MIN_PRICE = 0.40


# ── Precomputed entry candidate ─────────────────────────────────────

class EntryCandidate(NamedTuple):
    """Precomputed signal values + orderbook state at one candidate entry point."""
    market_idx: int          # index into markets list
    secs_remaining: float
    oracle_signal: float     # L1
    momentum_signal: float   # L2
    orderbook_signal: float  # L4
    taker_signal: float      # L7
    up_ask: float
    down_ask: float
    up_bid: float
    down_bid: float
    winner_is_up: bool


# ── Signal computation ──────────────────────────────────────────────

def compute_oracle_lag(btc_price: float, btc_open: float,
                       max_lag: float = 0.001) -> float:
    """L1: Oracle lag [-1, 1]."""
    if btc_open <= 0 or btc_price <= 0:
        return 0.0
    delta = (btc_price - btc_open) / btc_open
    return max(-1.0, min(1.0, delta / max_lag))


def compute_momentum_from_trades(spot_rows: list[tuple],
                                 lookback: int = 10) -> float:
    """L2: Momentum [-1, 1] from 1s spot trade rows.

    spot_rows: list of (open, close, high, low, volume, buy_vol, sell_vol) tuples.
    """
    if len(spot_rows) < 3:
        return 0.0

    recent = spot_rows[-lookback:] if len(spot_rows) >= lookback else spot_rows

    # Rate of change
    first_close = recent[0][1]
    last_close = recent[-1][1]
    if first_close > 0:
        roc = (last_close - first_close) / first_close
        roc_sig = max(-1.0, min(1.0, roc / 0.002))
    else:
        roc_sig = 0.0

    # Direction consistency
    up = sum(1 for r in recent if r[1] >= r[0])  # close >= open
    dir_sig = (up / len(recent) - 0.5) * 2.0

    # Volume trend
    mid = len(recent) // 2
    if mid > 0:
        old_vol = sum(r[4] for r in recent[:mid]) / mid
        new_vol = sum(r[4] for r in recent[mid:]) / max(len(recent) - mid, 1)
        vol_sig = max(-1.0, min(1.0, ((new_vol / old_vol) - 1.0) * 2.0)) if old_vol > 0 else 0.0
    else:
        vol_sig = 0.0

    # Body ratio (last candle)
    o, c, h, l = recent[-1][0], recent[-1][1], recent[-1][2], recent[-1][3]
    hl = h - l
    if hl > 0:
        body = abs(c - o) / hl
        body_sig = body if c >= o else -body
    else:
        body_sig = 0.0

    raw = 0.30 * roc_sig + 0.25 * dir_sig + 0.25 * vol_sig + 0.10 * body_sig
    return max(-1.0, min(1.0, raw))


def compute_ob_imbalance(up_bid_d: float, up_ask_d: float,
                         dn_bid_d: float, dn_ask_d: float) -> float:
    """L4: Orderbook imbalance [-1, 1]."""
    total_bid = up_bid_d + dn_bid_d
    total_ask = up_ask_d + dn_ask_d
    total = total_bid + total_ask
    if total <= 0:
        return 0.0
    return max(-1.0, min(1.0, (total_bid - total_ask) / total * 2.0))


def compute_taker_ratio_from_trades(spot_rows: list[tuple]) -> float:
    """L7: Taker ratio [-1, 1] from 1s spot trade rows.

    spot_rows: list of (open, close, high, low, volume, buy_vol, sell_vol).
    """
    if len(spot_rows) < 2:
        return 0.0
    total_buy = sum(r[5] for r in spot_rows)
    total_sell = sum(r[6] for r in spot_rows)
    total = total_buy + total_sell
    if total <= 0:
        return 0.0
    dev = total_buy / total - 0.5
    nb = 0.05
    if abs(dev) < nb:
        return 0.0
    if dev > 0:
        return min((dev - nb) / (0.5 - nb), 1.0)
    return max((dev + nb) / (0.5 - nb), -1.0)


# ── Data loading + precomputation ───────────────────────────────────

def load_and_precompute() -> tuple[list[dict], list[list[EntryCandidate]]]:
    """Load data from tick_data.db and precompute signals per market.

    Instead of holding 42M snapshot objects in memory, we:
    1. Load spot_trades into a dict (2.6M rows, ~400MB as tuples)
    2. Stream snapshots market-by-market via SQL GROUP
    3. For each market, find up to 3 candidate entry points (early/mid/late)
    4. Precompute L1/L2/L4/L7 signals at each candidate
    5. Store as lightweight EntryCandidate tuples (~14 floats each)

    Peak memory: ~1-2GB instead of 19GB.

    Returns:
        markets: list of {market_id, date, winner, btc_open, start_unix}
        candidates_by_market: list of list of EntryCandidate per market
    """
    print("Phase 1: Loading data and precomputing signals...", flush=True)
    t0 = time.time()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = None
    cur = conn.cursor()

    # Step 1: Load markets
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
        except (ValueError, TypeError):
            continue
        markets.append({
            "market_id": mid_str,
            "date": st[:10],
            "winner": winner,
            "btc_open": btc_open or 0.0,
            "start_unix": start_unix,
        })
        market_id_to_idx[mid_str] = len(markets) - 1

    print(f"  {len(markets):,} markets loaded", flush=True)

    # Step 2: Load spot trades as tuples (open, close, high, low, vol, buy_vol, sell_vol)
    print("  Loading spot trades...", flush=True)
    cur.execute("""
        SELECT timestamp, price_open, price_close, price_high, price_low,
               total_volume, aggressive_buy_volume, aggressive_sell_volume
        FROM spot_trades ORDER BY timestamp
    """)
    spot_by_sec: dict[int, tuple] = {}
    for row in cur.fetchall():
        try:
            dt = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
            ts = int(dt.timestamp())
        except (ValueError, TypeError):
            continue
        # (open, close, high, low, volume, buy_vol, sell_vol)
        spot_by_sec[ts] = (
            row[1] or 0.0, row[2] or 0.0, row[3] or 0.0, row[4] or 0.0,
            row[5] or 0.0, row[6] or 0.0, row[7] or 0.0,
        )
    print(f"  {len(spot_by_sec):,} spot candles loaded", flush=True)

    # Step 3: Stream snapshots market-by-market and precompute signals
    # We sample ~3 entry points per market (early=240s rem, mid=180s, late=90s)
    # to capture time-varying signal quality without storing all 2000+ ticks
    print("  Precomputing signals per market (streaming snapshots)...", flush=True)

    SAMPLE_SECS_INTO = [30, 60, 90, 120, 150, 180, 210, 240]
    # These correspond to secs_remaining = 300 - secs_into:
    # 270, 240, 210, 180, 150, 120, 90, 60

    candidates_by_market: list[list[EntryCandidate]] = [[] for _ in range(len(markets))]
    processed = 0
    total_candidates = 0

    # Process each market individually to avoid loading all snapshots
    for m_idx, m in enumerate(markets):
        mid = m["market_id"]
        btc_open = m["btc_open"]
        start_unix = m["start_unix"]
        winner_up = m["winner"] == "Up"

        # Query only the snapshots we need for this market
        # Sample one snapshot near each target secs_into
        cur.execute("""
            SELECT seconds_into_window, btc_price,
                   price_up, price_down,
                   up_best_bid, up_best_ask, down_best_bid, down_best_ask,
                   up_bid_depth_5, up_ask_depth_5, down_bid_depth_5, down_ask_depth_5
            FROM snapshots
            WHERE market_id = ?
              AND seconds_into_window >= 25
              AND seconds_into_window <= 245
            ORDER BY seconds_into_window
        """, (mid,))

        # Pick snapshots closest to our target times
        all_snaps = cur.fetchall()
        if not all_snaps:
            continue

        # Build index for quick lookup
        snap_by_approx: dict[int, tuple] = {}
        for snap in all_snaps:
            bucket = round(snap[0] / 30) * 30  # bucket to nearest 30s
            if bucket not in snap_by_approx:
                snap_by_approx[bucket] = snap

        for target_secs_into in SAMPLE_SECS_INTO:
            bucket = round(target_secs_into / 30) * 30
            snap = snap_by_approx.get(bucket)
            if not snap:
                continue

            secs_into = snap[0]
            secs_remaining = 300.0 - secs_into

            if secs_remaining > EARLIEST_ENTRY_SECS or secs_remaining < LATEST_ENTRY_SECS:
                continue

            btc_price = snap[1] or 0.0
            up_ask = snap[5] or 0.0
            down_ask = snap[7] or 0.0
            up_bid = snap[4] or 0.0
            down_bid = snap[6] or 0.0

            if up_ask <= 0 or down_ask <= 0:
                continue

            # L1: Oracle lag
            oracle_sig = compute_oracle_lag(btc_price, btc_open)

            # L2: Momentum (last 60s of spot trades)
            current_unix = start_unix + int(secs_into)
            mom_rows = []
            for offset in range(-60, 1):
                t = current_unix + offset
                if t in spot_by_sec:
                    mom_rows.append(spot_by_sec[t])
            momentum_sig = compute_momentum_from_trades(mom_rows)

            # L4: Orderbook imbalance
            ob_sig = compute_ob_imbalance(
                snap[8] or 0.0, snap[9] or 0.0,
                snap[10] or 0.0, snap[11] or 0.0,
            )

            # L7: Taker ratio (last 30s)
            taker_rows = []
            for offset in range(-30, 1):
                t = current_unix + offset
                if t in spot_by_sec:
                    taker_rows.append(spot_by_sec[t])
            taker_sig = compute_taker_ratio_from_trades(taker_rows)

            candidates_by_market[m_idx].append(EntryCandidate(
                market_idx=m_idx,
                secs_remaining=secs_remaining,
                oracle_signal=oracle_sig,
                momentum_signal=momentum_sig,
                orderbook_signal=ob_sig,
                taker_signal=taker_sig,
                up_ask=up_ask,
                down_ask=down_ask,
                up_bid=up_bid,
                down_bid=down_bid,
                winner_is_up=winner_up,
            ))
            total_candidates += 1

        processed += 1
        if processed % 2000 == 0:
            print(f"    {processed:,}/{len(markets):,} markets processed "
                  f"({total_candidates:,} candidates)", flush=True)

    conn.close()
    elapsed = time.time() - t0
    print(f"  Done: {processed:,} markets, {total_candidates:,} entry candidates "
          f"in {elapsed:.1f}s", flush=True)

    return markets, candidates_by_market


# ── Fast weight evaluation (pure arithmetic) ────────────────────────

def evaluate_weights(
    markets: list[dict],
    candidates_by_market: list[list[EntryCandidate]],
    market_indices: list[int],
    oracle_w: float,
    momentum_w: float,
    orderbook_w: float,
    taker_w: float,
    max_adj: float,
) -> tuple[float, float, int, float]:
    """Evaluate one weight combo on a set of markets.

    Pure arithmetic on precomputed signals — no I/O, no object creation.
    Takes the first valid entry in each market (earliest secs_remaining).

    Returns (sharpe, total_pnl, num_trades, win_rate).
    """
    pnls = []
    wins = 0

    for m_idx in market_indices:
        candidates = candidates_by_market[m_idx]
        if not candidates:
            continue

        # Try candidates in order (earliest entry first = highest secs_remaining)
        for c in sorted(candidates, key=lambda x: -x.secs_remaining):
            # Combine signals
            # Core weights with redistribution for zero signals
            sigs = [
                (c.oracle_signal, oracle_w),
                (c.momentum_signal, momentum_w),
                (c.orderbook_signal, orderbook_w),
            ]
            unavail = 0.0
            active_w = []
            for sv, w in sigs:
                if sv == 0.0 and w > 0:
                    unavail += w
                    active_w.append(0.0)
                else:
                    active_w.append(w)

            tw = sum(active_w)
            if tw > 0 and unavail > 0:
                sc = (tw + unavail) / tw
                active_w = [w * sc for w in active_w]

            raw = (active_w[0] * c.oracle_signal
                   + active_w[1] * c.momentum_signal
                   + active_w[2] * c.orderbook_signal)

            if c.taker_signal != 0.0 and taker_w > 0:
                raw += taker_w * c.taker_signal

            raw = max(-1.0, min(1.0, raw))
            est_prob_up = 0.5 + raw * max_adj
            est_prob_up = max(0.05, min(0.95, est_prob_up))

            # Entry check
            confidence = abs(est_prob_up - 0.5)
            if confidence < MIN_CONFIDENCE:
                continue

            market_mid = (c.up_bid + c.up_ask) / 2.0
            market_prob = max(0.05, min(0.95, market_mid))

            if est_prob_up >= 0.5:
                side = "YES"
                price = c.up_ask
            else:
                side = "NO"
                price = c.down_ask

            prob_edge = abs(est_prob_up - market_prob)
            time_pct = c.secs_remaining / 300.0
            req_edge = MIN_EDGE + (MAX_EDGE - MIN_EDGE) * time_pct

            if prob_edge < req_edge:
                continue

            if side == "YES" and price < YES_MIN_PRICE:
                continue

            # Trade! Compute PnL
            won = (side == "YES" and c.winner_is_up) or \
                  (side == "NO" and not c.winner_is_up)

            if won:
                pnl = (1.0 - price) / price - FEE_RATE
            else:
                pnl = -1.0

            pnls.append(pnl)
            if won:
                wins += 1
            break  # one trade per market

    # Sharpe
    n = len(pnls)
    if n < 5:
        return -999.0, sum(pnls), n, (wins / n if n > 0 else 0.0)

    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / n
    std = math.sqrt(var) if var > 0 else 1e-9
    sharpe = mean / std

    return sharpe, sum(pnls), n, wins / n


# ── Walk-forward engine ─────────────────────────────────────────────

def build_param_grid() -> list[tuple[float, float, float, float, float]]:
    """Build deduplicated parameter grid as tuples for speed.

    Returns list of (oracle_w, momentum_w, orderbook_w, taker_w, max_adj).
    """
    oracle_opts = [0.10, 0.20, 0.30, 0.40, 0.50]
    momentum_opts = [0.20, 0.30, 0.40, 0.50, 0.60]
    ob_opts = [0.05, 0.10, 0.20, 0.30, 0.40]
    taker_opts = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20]
    max_adj_opts = [0.15, 0.20, 0.25]

    seen = set()
    grid = []
    for o_raw, m_raw, ob_raw in itertools.product(oracle_opts, momentum_opts, ob_opts):
        total = o_raw + m_raw + ob_raw
        o_w = round(o_raw / total, 4)
        m_w = round(m_raw / total, 4)
        ob_w = round(ob_raw / total, 4)
        for t_w in taker_opts:
            for ma in max_adj_opts:
                key = (o_w, m_w, ob_w, t_w, ma)
                if key not in seen:
                    seen.add(key)
                    grid.append(key)

    return grid


def generate_folds(
    markets: list[dict],
    train_days: int = 7,
    test_days: int = 5,
    warmup_days: int = 3,
) -> list[tuple[str, str, str, str]]:
    """Generate walk-forward folds."""
    dates = sorted(set(m["date"] for m in markets))
    if not dates:
        return []

    earliest = datetime.strptime(dates[0], "%Y-%m-%d")
    latest = datetime.strptime(dates[-1], "%Y-%m-%d")
    start = earliest + timedelta(days=warmup_days)

    folds = []
    while True:
        train_end = start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        if test_end > latest:
            break
        folds.append((
            start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))
        start += timedelta(days=test_days)

    return folds


def get_market_indices(markets: list[dict], date_start: str, date_end: str) -> list[int]:
    """Get indices of markets in [date_start, date_end)."""
    return [i for i, m in enumerate(markets) if date_start <= m["date"] < date_end]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Walk-forward signal weight optimiser")
    parser.add_argument("--train-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=5)
    parser.add_argument("--warmup-days", type=int, default=3)
    args = parser.parse_args()

    # Phase 1: Load + precompute
    markets, candidates_by_market = load_and_precompute()
    print(f"\nTotal markets: {len(markets):,}")
    print(f"Date range: {markets[0]['date']} to {markets[-1]['date']}")

    # Phase 2: Grid search with walk-forward validation
    param_grid = build_param_grid()
    print(f"Parameter combinations: {len(param_grid):,}")

    folds = generate_folds(markets, args.train_days, args.test_days, args.warmup_days)
    print(f"Walk-forward folds: {len(folds)}")

    if not folds:
        print("ERROR: Not enough data for walk-forward folds")
        return

    print("\nFold schedule:")
    for i, (ts, te, vs, ve) in enumerate(folds, 1):
        n_train = len(get_market_indices(markets, ts, te))
        n_test = len(get_market_indices(markets, vs, ve))
        print(f"  Fold {i}: train {ts} → {te} ({n_train} mkts) | test {vs} → {ve} ({n_test} mkts)")

    # Default weights (Bot G baseline at ~180s remaining)
    DEFAULT = (0.25, 0.30, 0.23, 0.08, 0.20)
    # (oracle, momentum, orderbook, taker, max_adj)
    # Note: L3 (0.12) and L5 (0.15) get redistributed into L1/L2/L4 when zero

    print("\n" + "=" * 75)
    print("WALK-FORWARD OPTIMISATION")
    print("=" * 75)

    fold_results = []

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds, 1):
        print(f"\n--- Fold {fold_idx}/{len(folds)}: "
              f"train {train_start}→{train_end} | test {test_start}→{test_end} ---")

        train_idx = get_market_indices(markets, train_start, train_end)
        test_idx = get_market_indices(markets, test_start, test_end)

        if not train_idx:
            print("  No train markets — skipping")
            continue

        # Grid search on train
        t0 = time.time()
        best = None
        best_sharpe = -999.0

        for pi, (o_w, m_w, ob_w, t_w, ma) in enumerate(param_grid):
            sharpe, pnl, trades, wr = evaluate_weights(
                markets, candidates_by_market, train_idx,
                o_w, m_w, ob_w, t_w, ma,
            )
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best = (o_w, m_w, ob_w, t_w, ma, pnl, trades, wr)

            if (pi + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"    {pi+1}/{len(param_grid)} combos ({elapsed:.1f}s)...", flush=True)

        elapsed = time.time() - t0
        o_w, m_w, ob_w, t_w, ma, tr_pnl, tr_trades, tr_wr = best
        print(f"  Train: {elapsed:.1f}s | Sharpe={best_sharpe:.4f} "
              f"PnL={tr_pnl:+.2f} trades={tr_trades} WR={tr_wr:.1%}")
        print(f"  Best: L1={o_w:.3f} L2={m_w:.3f} L4={ob_w:.3f} "
              f"L7={t_w:.3f} max_adj={ma:.3f}")

        # Evaluate on test
        test_sharpe, test_pnl, test_trades, test_wr = evaluate_weights(
            markets, candidates_by_market, test_idx,
            o_w, m_w, ob_w, t_w, ma,
        )

        # Default on test
        def_sharpe, def_pnl, def_trades, def_wr = evaluate_weights(
            markets, candidates_by_market, test_idx,
            *DEFAULT,
        )

        print(f"  Test (optimised): Sharpe={test_sharpe:.4f} "
              f"PnL={test_pnl:+.2f} trades={test_trades} WR={test_wr:.1%}")
        print(f"  Test (default):   Sharpe={def_sharpe:.4f} "
              f"PnL={def_pnl:+.2f} trades={def_trades} WR={def_wr:.1%}")

        fold_results.append({
            "fold": fold_idx,
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "best": {"oracle_w": o_w, "momentum_w": m_w, "orderbook_w": ob_w,
                      "taker_w": t_w, "max_adj": ma},
            "train_sharpe": best_sharpe, "train_pnl": tr_pnl,
            "train_trades": tr_trades, "train_wr": tr_wr,
            "test_sharpe": test_sharpe, "test_pnl": test_pnl,
            "test_trades": test_trades, "test_wr": test_wr,
            "def_sharpe": def_sharpe, "def_pnl": def_pnl,
            "def_trades": def_trades, "def_wr": def_wr,
        })

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    if not fold_results:
        print("No results")
        return

    print(f"\n{'Fold':<6} {'Train Sharpe':<14} {'Test PnL(opt)':<15} "
          f"{'Test PnL(def)':<15} {'Δ PnL':<10} {'Test WR(opt)':<13}")
    print("-" * 73)

    total_opt = 0.0
    total_def = 0.0
    fold_wins = 0
    for fr in fold_results:
        delta = fr["test_pnl"] - fr["def_pnl"]
        total_opt += fr["test_pnl"]
        total_def += fr["def_pnl"]
        if delta > 0:
            fold_wins += 1
        print(f"{fr['fold']:<6} {fr['train_sharpe']:<14.4f} {fr['test_pnl']:<+15.2f} "
              f"{fr['def_pnl']:<+15.2f} {delta:<+10.2f} {fr['test_wr']:<13.1%}")

    print("-" * 73)
    print(f"{'Total':<6} {'':<14} {total_opt:<+15.2f} {total_def:<+15.2f} "
          f"{total_opt - total_def:<+10.2f}")

    print(f"\n\nBest weights per fold:")
    print(f"{'Fold':<6} {'L1(oracle)':<11} {'L2(momentum)':<13} "
          f"{'L4(orderbook)':<14} {'L7(taker)':<11} {'max_adj':<9}")
    print("-" * 64)
    for fr in fold_results:
        p = fr["best"]
        print(f"{fr['fold']:<6} {p['oracle_w']:<11.3f} {p['momentum_w']:<13.3f} "
              f"{p['orderbook_w']:<14.3f} {p['taker_w']:<11.3f} {p['max_adj']:<9.3f}")

    # Trade-weighted average
    total_tt = sum(fr["test_trades"] for fr in fold_results) or 1
    avg = {
        "oracle_w": sum(fr["best"]["oracle_w"] * fr["test_trades"] for fr in fold_results) / total_tt,
        "momentum_w": sum(fr["best"]["momentum_w"] * fr["test_trades"] for fr in fold_results) / total_tt,
        "orderbook_w": sum(fr["best"]["orderbook_w"] * fr["test_trades"] for fr in fold_results) / total_tt,
        "taker_w": sum(fr["best"]["taker_w"] * fr["test_trades"] for fr in fold_results) / total_tt,
        "max_adj": sum(fr["best"]["max_adj"] * fr["test_trades"] for fr in fold_results) / total_tt,
    }

    print(f"\nTrade-weighted average (recommended for Bot K):")
    print(f"  L1 (oracle):    {avg['oracle_w']:.3f}")
    print(f"  L2 (momentum):  {avg['momentum_w']:.3f}")
    print(f"  L4 (orderbook): {avg['orderbook_w']:.3f}")
    print(f"  L7 (taker):     {avg['taker_w']:.3f}")
    print(f"  max_adjustment: {avg['max_adj']:.3f}")

    print(f"\nCurrent defaults (Bot G):")
    print(f"  L1 (oracle):    {DEFAULT[0]:.3f}")
    print(f"  L2 (momentum):  {DEFAULT[1]:.3f}")
    print(f"  L4 (orderbook): {DEFAULT[2]:.3f}")
    print(f"  L7 (taker):     {DEFAULT[3]:.3f}")
    print(f"  max_adjustment: {DEFAULT[4]:.3f}")

    improvement = total_opt - total_def
    pct = (improvement / abs(total_def) * 100) if total_def != 0 else 0
    print(f"\nTotal OOS PnL improvement: {improvement:+.2f} ({pct:+.1f}%)")
    print(f"Folds where optimised > default: {fold_wins}/{len(fold_results)}")

    if fold_wins < len(fold_results) * 0.6:
        print("\n⚠️  Optimised weights did NOT consistently beat defaults.")
        print("   Improvement may be noise. Consider keeping defaults for Bot K.")
    else:
        print("\n✅ Optimised weights consistently outperformed defaults.")
        print(f"   Apply to Bot K's config_multi.yaml signals section.")


if __name__ == "__main__":
    main()
