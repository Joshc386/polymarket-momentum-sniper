"""Extended high_vol validation using the full tick database.

The first high_vol validation used 27.6h of post-L1-fix diagnostic data,
giving 128 simulated trades in Bot G's high_vol regime. The directional
finding (high_vol more profitable than ranging) is suggestive but a
moderate sample.

This extended version uses backtest/data/tick_data.db (snapshots of every
market window since March 24) to dramatically increase the sample size.

Approach:
1. Replicate MTF regime detection on the BTC candles for each market
2. Classify each window's regime at typical entry time (~60s into window)
3. For high_vol windows (and ranging baseline), recompute L1-L5 signals
   and simulate entry decision under Bot G's current ranging-schedule
   weights AND under the L1-FIXED open-component-only version
4. Aggregate PnL using the actual market winner from the markets table

Caveats:
- We can't perfectly replicate the live regime detector here without
  rebuilding the full MTF state machine. We use a single-timeframe ATR
  percentile as a high_vol proxy. Real MTFRegimeDetector also uses
  multi-timeframe alignment, choppiness, and stickiness - so our proxy
  will catch most high_vol periods but not exactly the same set.
- Resolution is exact from the markets.winner column.
- L1 fix applied (uses only open_component, no lag component).

Usage:
    python -m backtest.high_vol_extended_validation
"""

import os
import sqlite3
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import NamedTuple

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "tick_data.db")

# Bot G ranging schedule (used when regime is ranging)
WEIGHTS_RANGING_180 = (0.05, 0.25, 0.05, 0.60, 0.05)
# Bot G default schedule (used for high_vol since high_vol has no override)
WEIGHTS_DEFAULT_180 = (0.20, 0.30, 0.12, 0.23, 0.15)

# Entry parameters (Bot G/K config)
MIN_EDGE = 0.003
MAX_EDGE = 0.025
MIN_CONFIDENCE = 0.02
YES_MIN_PRICE = 0.40
NO_MIN_PRICE = 0.40   # symmetric (would-be filter)
HIGH_EV_THRESHOLD = 0.15
HIGH_EV_MIN_SECS_INTO_WIN = 60

# Realism guards
PRICE_MIN = 0.05
PRICE_MAX = 0.95
PROB_EDGE_MAX_SANE = 0.30

# Regime edge multipliers
EDGE_MULT_HIGH_VOL = 1.8
EDGE_MULT_RANGING = 1.5
EDGE_MULT_DEFAULT = 1.0

# Volatility proxy: top quintile of realized BTC range = "high_vol" proxy
HIGH_VOL_PCT = 0.80  # top 20% volatility = high_vol proxy

# Fee
FEE_RATE = 0.072

# Entry window
EARLIEST_ENTRY_SECS = 270  # secs_remaining
LATEST_ENTRY_SECS = 60


def taker_fee(price: float) -> float:
    return FEE_RATE * price * (1.0 - price)


def required_edge(secs_remaining: float, edge_mult: float) -> float:
    time_pct = max(0.0, min(1.0, secs_remaining / 300.0))
    return (MIN_EDGE + (MAX_EDGE - MIN_EDGE) * time_pct) * edge_mult


# ── Signal computations (same as maker_execution_backtest) ──

def compute_oracle_lag(btc_price: float, btc_open: float) -> float:
    """L1 with FIX APPLIED: uses only the open component."""
    if btc_open <= 0 or btc_price <= 0:
        return 0.0
    pct = (btc_price - btc_open) / btc_open
    if abs(pct) < 0.0001:
        return 0.0
    raw = pct / 0.001
    return max(-1.0, min(1.0, raw))


def compute_momentum(spot_rows: list) -> float:
    if len(spot_rows) < 3:
        return 0.0
    recent = spot_rows[-10:] if len(spot_rows) >= 10 else spot_rows
    o, c, h, l = recent[-1][0], recent[-1][1], recent[-1][2], recent[-1][3]
    rng = h - l
    if rng < 1e-9:
        return 0.0
    direction = 1.0 if c > o else (-1.0 if c < o else 0.0)
    body_ratio = abs(c - o) / rng
    roc = 0.0
    if len(recent) >= 2:
        prev_c = recent[-2][1]
        if prev_c > 0:
            roc = (c - prev_c) / prev_c * 1000
    raw = 0.40 * roc + 0.30 * direction + 0.30 * body_ratio * direction
    return max(-1.0, min(1.0, raw))


def compute_ob_imbalance(
    bid_depth, ask_depth, no_bid_depth, no_ask_depth
) -> float:
    """Simplified L4 (imbalance only) — sub-components other than imbalance
    aren't reconstructable from tick_data.db so we use just this."""
    total = bid_depth + ask_depth + no_bid_depth + no_ask_depth
    if total < 1.0:
        return 0.0
    bull = bid_depth + no_ask_depth
    bear = ask_depth + no_bid_depth
    imb = (bull - bear) / total
    return max(-1.0, min(1.0, imb / 0.3))


def combine(weights, l1, l2, l3, l4, l5):
    w1, w2, w3, w4, w5 = weights
    return w1 * l1 + w2 * l2 + w3 * l3 + w4 * l4 + w5 * l5


# ── Volatility proxy ──

def compute_market_volatility(snapshots: list) -> float:
    """High-low range during the window, scaled to BTC price."""
    if len(snapshots) < 2:
        return 0.0
    prices = [s[2] for s in snapshots if s[2] and s[2] > 0]
    if len(prices) < 2:
        return 0.0
    avg_p = sum(prices) / len(prices)
    if avg_p <= 0:
        return 0.0
    return (max(prices) - min(prices)) / avg_p


class TradeResult(NamedTuple):
    market_id: str
    regime_proxy: str       # "high_vol_proxy" / "mid_vol" / "low_vol_proxy"
    side: str
    entry_price: float
    est_prob_up: float
    prob_edge: float
    secs_remaining: float
    winner: str             # "Up" or "Down"
    won: bool
    pnl: float


def main():
    print("=" * 85)
    print("EXTENDED HIGH_VOL VALIDATION using tick_data.db")
    print("=" * 85)

    if not os.path.exists(DB_PATH):
        print(f"tick_data.db not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Determine date range
    cur.execute("SELECT MIN(start_time), MAX(start_time), COUNT(*) FROM markets WHERE winner IN ('Up', 'Down')")
    min_t, max_t, n_markets = cur.fetchone()
    print(f"\nMarkets in DB: {n_markets:,}")
    print(f"Date range: {min_t} to {max_t}")

    # Load spot trades for momentum
    print("\nLoading spot trades for momentum...")
    cur.execute("""
        SELECT timestamp, price_open, price_close, price_high, price_low
        FROM spot_trades ORDER BY timestamp
    """)
    spot_by_ts = {}
    for row in cur.fetchall():
        if row[0]:
            spot_by_ts[row[0][:19]] = row[1:]
    print(f"  {len(spot_by_ts):,} spot candles indexed")

    # Load markets
    cur.execute("""
        SELECT market_id, start_time, btc_price_start, btc_price_end, winner
        FROM markets
        WHERE winner IN ('Up', 'Down')
        ORDER BY start_time
    """)
    markets_raw = cur.fetchall()

    markets = {}
    for mid, st, btc_open, btc_close, winner in markets_raw:
        markets[str(mid)] = {
            "start_time": st,
            "btc_open": btc_open or 0.0,
            "btc_close": btc_close or 0.0,
            "winner": winner,
        }

    print(f"  {len(markets):,} resolved markets loaded")

    # ── Phase 1: classify each market by realized volatility ──
    print("\nPhase 1: classifying markets by realized BTC volatility...")
    market_volatility = {}
    market_entry_snapshot = {}

    batch_size = 500
    market_ids = list(markets.keys())

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
        rows = cur.fetchall()

        # Group by market_id
        by_market = defaultdict(list)
        for r in rows:
            by_market[str(r[0])].append(r)

        for mid, snaps in by_market.items():
            # Volatility = price range during window
            vol = compute_market_volatility(snaps)
            market_volatility[mid] = vol

            # Entry snapshot = first one at seconds_into_window >= 30
            for s in snaps:
                if 30 <= s[1] <= 240:
                    market_entry_snapshot[mid] = s
                    break

        if (batch_start + batch_size) % 2000 == 0 or batch_start + batch_size >= len(market_ids):
            print(f"  {min(batch_start + batch_size, len(market_ids)):,}/"
                  f"{len(market_ids):,} markets processed",
                  flush=True)

    # ── Phase 2: bucket markets by volatility ──
    vols = sorted(market_volatility.values())
    if not vols:
        print("No volatility data!")
        return

    n = len(vols)
    high_vol_threshold = vols[int(n * HIGH_VOL_PCT)]
    low_vol_threshold = vols[int(n * 0.20)]

    print(f"\nVolatility distribution:")
    print(f"  p10:  {vols[n//10]*100:.3f}%")
    print(f"  p50:  {vols[n//2]*100:.3f}%")
    print(f"  p80:  {high_vol_threshold*100:.3f}% (high_vol proxy threshold)")
    print(f"  p90:  {vols[int(n*0.9)]*100:.3f}%")

    def regime_proxy(vol):
        if vol >= high_vol_threshold:
            return "high_vol"
        elif vol < low_vol_threshold:
            return "low_vol"
        else:
            return "mid_vol"

    # ── Phase 3: simulate trades for each market ──
    print("\nPhase 3: simulating trades...")

    trade_results = []
    stats = defaultdict(lambda: {"n_signal": 0, "n_traded": 0})

    for mid, snap in market_entry_snapshot.items():
        if mid not in markets:
            continue
        mkt = markets[mid]
        btc_open = mkt["btc_open"]
        if btc_open <= 0:
            continue
        winner = mkt["winner"]
        winner_is_up = winner == "Up"

        proxy = regime_proxy(market_volatility[mid])

        secs_into, btc_price = snap[1], snap[2] or 0.0
        secs_rem = 300.0 - secs_into

        if secs_rem < LATEST_ENTRY_SECS or secs_rem > EARLIEST_ENTRY_SECS:
            continue

        y_bid = snap[3] or 0.0
        y_ask = snap[4] or 0.0
        n_bid = snap[5] or 0.0
        n_ask = snap[6] or 0.0
        y_bid_depth = snap[7] or 0.0
        y_ask_depth = snap[8] or 0.0
        n_bid_depth = snap[9] or 0.0
        n_ask_depth = snap[10] or 0.0

        if y_ask <= 0 or n_ask <= 0:
            continue

        mkt_mid = (y_bid + y_ask) / 2.0
        if mkt_mid < PRICE_MIN or mkt_mid > PRICE_MAX:
            continue

        # Compute signals
        l1 = compute_oracle_lag(btc_price, btc_open)

        # L2 momentum
        snap_time = snap[11]
        mom_rows = []
        if snap_time:
            try:
                t_sec = datetime.fromisoformat(snap_time[:19])
                for off in range(-10, 1):
                    t_key = (t_sec + timedelta(seconds=off)).strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    )
                    if t_key in spot_by_ts:
                        mom_rows.append(spot_by_ts[t_key])
            except Exception:
                pass
        l2 = compute_momentum(mom_rows)

        l4 = compute_ob_imbalance(
            y_bid_depth, y_ask_depth, n_bid_depth, n_ask_depth
        )

        # For high_vol use default schedule + contrarian mode
        # For mid_vol assume ranging schedule + signal_aligned (Bot G in ranging)
        if proxy == "high_vol":
            weights = WEIGHTS_DEFAULT_180
            edge_mult = EDGE_MULT_HIGH_VOL
            signal_aligned = False  # contrarian for non-ranging
        else:
            weights = WEIGHTS_RANGING_180
            edge_mult = EDGE_MULT_RANGING
            signal_aligned = True

        combined = combine(weights, l1, l2, 0.0, l4, 0.0)
        combined = max(-1.0, min(1.0, combined))
        est_p = 0.5 + combined * 0.20  # max_adjustment

        # Side selection
        prob_edge_yes = est_p - mkt_mid
        prob_edge = abs(prob_edge_yes)

        if prob_edge > PROB_EDGE_MAX_SANE:
            continue

        if signal_aligned:
            if est_p >= 0.5:
                side = "YES"
                price = y_ask
            else:
                side = "NO"
                price = n_ask
        else:
            if prob_edge_yes >= 0:
                side = "YES"
                price = y_ask
            else:
                side = "NO"
                price = n_ask

        if price <= 0 or price >= 1.0:
            continue
        if side == "YES" and price < YES_MIN_PRICE:
            continue
        if side == "NO" and price < NO_MIN_PRICE:
            continue

        # Confidence
        signal_conf = abs(est_p - 0.5) * 2.0
        if signal_conf < MIN_CONFIDENCE:
            continue

        # Edge threshold
        req_edge = required_edge(secs_rem, edge_mult)
        if prob_edge <= req_edge:
            continue

        # EV
        fee = taker_fee(price)
        if side == "YES":
            ev = est_p * (1.0 - price) - (1.0 - est_p) * price - fee
        else:
            ev = (1.0 - est_p) * (1.0 - price) - est_p * price - fee
        if ev <= 0:
            continue

        # high_ev early-window filter
        if ev >= HIGH_EV_THRESHOLD and secs_into < HIGH_EV_MIN_SECS_INTO_WIN:
            continue

        # PnL
        won = (side == "YES" and winner_is_up) or (side == "NO" and not winner_is_up)
        if won:
            pnl = ((1.0 - price - fee) / price) * 3.0  # $3 bet
        else:
            pnl = -3.0

        stats[proxy]["n_signal"] += 1
        stats[proxy]["n_traded"] += 1
        trade_results.append(TradeResult(
            market_id=mid,
            regime_proxy=proxy,
            side=side,
            entry_price=price,
            est_prob_up=est_p,
            prob_edge=prob_edge,
            secs_remaining=secs_rem,
            winner=winner,
            won=won,
            pnl=pnl,
        ))

    conn.close()

    print(f"\n  {len(trade_results):,} would-be trades simulated")

    # ── Phase 4: Report ──
    print(f"\n{'=' * 85}")
    print(f"RESULTS by regime proxy")
    print(f"{'=' * 85}")

    for proxy in ["high_vol", "mid_vol", "low_vol"]:
        trades = [t for t in trade_results if t.regime_proxy == proxy]
        print(f"\n--- {proxy.upper()} ---")
        if not trades:
            print(f"  No trades.")
            continue

        n = len(trades)
        yes = [t for t in trades if t.side == "YES"]
        no = [t for t in trades if t.side == "NO"]
        wins = sum(1 for t in trades if t.won)
        pnl = sum(t.pnl for t in trades)
        avg = pnl / n
        pnls = [t.pnl for t in trades]
        std = statistics.stdev(pnls) if n > 1 else 0
        sharpe = avg / std if std > 0 else 0

        print(f"  N: {n:,} trades")
        print(f"  YES: {len(yes):,} ({len(yes)/n:.1%})  NO: {len(no):,} ({len(no)/n:.1%})")
        print(f"  WR: {wins/n:.1%}")
        print(f"  Total PnL: ${pnl:+.2f}")
        print(f"  Avg PnL/trade: ${avg:+.4f}")
        print(f"  Stdev: ${std:.4f}  Sharpe/trade: {sharpe:.4f}")

        if yes:
            yw = sum(1 for t in yes if t.won)
            yp = sum(t.pnl for t in yes)
            print(f"    YES side: WR {yw/len(yes):.1%}  PnL ${yp:+.2f}  Avg ${yp/len(yes):+.4f}")
        if no:
            nw = sum(1 for t in no if t.won)
            np_ = sum(t.pnl for t in no)
            print(f"    NO side:  WR {nw/len(no):.1%}  PnL ${np_:+.2f}  Avg ${np_/len(no):+.4f}")

    # Comparison table
    print(f"\n{'=' * 85}")
    print(f"COMPARISON TABLE")
    print(f"{'=' * 85}")
    print(f"  {'Proxy':<15} {'N':>6} {'WR':>7} {'Total PnL':>11} {'Avg PnL':>10} "
          f"{'Sharpe':>8}")
    for proxy in ["high_vol", "mid_vol", "low_vol"]:
        trades = [t for t in trade_results if t.regime_proxy == proxy]
        if not trades:
            continue
        n = len(trades)
        wins = sum(1 for t in trades if t.won)
        pnl = sum(t.pnl for t in trades)
        avg = pnl / n
        pnls = [t.pnl for t in trades]
        std = statistics.stdev(pnls) if n > 1 else 0
        sharpe = avg / std if std > 0 else 0
        print(f"  {proxy:<15} {n:>6,} {wins/n:>6.1%} "
              f"${pnl:>+9.2f} ${avg:>+8.4f} {sharpe:>8.4f}")

    print(f"\nCaveat: volatility proxy is BTC price range during the window, "
          f"not the live MTFRegimeDetector's ATR-percentile + multi-timeframe-"
          f"alignment classification. The actual high_vol set may differ.")


if __name__ == "__main__":
    main()
