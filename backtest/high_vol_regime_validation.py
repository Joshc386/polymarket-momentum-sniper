"""Validate the high_vol regime block empirically.

The regime detector blocks all trades when ATR is in the top 10% (high_vol).
In the May 13-14 diagnostic capture, this blocked 11,055 NO ticks and a
similar number of YES ticks - roughly 22,000 evaluation points per bot.

This script asks: would those trades have been profitable if allowed?

Approach:
1. From signal_diag DB, find all high_vol-regime ticks
2. For each market that had high_vol-blocked signal opportunities, find
   the FIRST tick where the bot WOULD have entered (prob_edge clears the
   high_vol-multiplied threshold, EV positive)
3. Reconstruct the would-be resolution from the same DB:
   - Find the last tick in that window
   - If btc_price > oracle_open_price -> UP wins (YES side wins)
   - Else -> DOWN wins (NO side wins)
4. Compute would-be PnL with full Polymarket crypto fee structure
5. Compare aggregate PnL of high_vol-allowed vs the current block

Same logic optionally for low_vol (though edge_multiplier 999 makes it
effectively dead).

Limitations:
- Resolution approximated from Binance price at last tick, not Chainlink
  close. Could be off by ~$10 in absolute price (rarely affects direction).
- Uses market_implied_prob (midpoint) as price proxy rather than ask.
  Real entries pay slightly more (spread/2). PnL will be slightly
  optimistic.
- 27.6h is a small sample. Results suggestive, not definitive.

Usage:
    python -m backtest.high_vol_regime_validation
"""

import os
import sqlite3
import statistics
from collections import defaultdict
from typing import NamedTuple

DBS = [
    ("Bot G", "data_runtime/bot_g_signal_aligned_signal_diag.db"),
    ("Bot K", "data_runtime/bot_k_sm_confirmation_signal_diag.db"),
]

# Entry threshold params (Bot G/K config)
MIN_EDGE = 0.003
MAX_EDGE = 0.025
MIN_CONFIDENCE = 0.02
EARLIEST_ENTRY_SECS = 270  # latest seconds_remaining for entry
LATEST_ENTRY_SECS = 60     # cutoff before window end

# Filters that Bot G/K actually apply (must be replicated in simulation
# or we'll count trades the bot wouldn't have taken)
YES_MIN_PRICE = 0.40            # block YES below this price
NO_MIN_PRICE = 0.40             # block NO below this price (symmetric — used here)
HIGH_EV_THRESHOLD = 0.15        # if EV (best_ev) >= this, also enforce
HIGH_EV_MIN_SECS_INTO_WIN = 60  # ...minimum seconds into window

# Realism guards
PRICE_MIN = 0.05  # below this is junk (one-sided book / window transition)
PRICE_MAX = 0.95  # above this is junk (mirror)
PROB_EDGE_MAX_SANE = 0.30  # disagreement larger than this is spurious

# Regime edge multipliers (from regime_detector.py REGIME_PARAMS)
EDGE_MULT_HIGH_VOL = 1.8
EDGE_MULT_LOW_VOL = 999.0  # effectively blocks
EDGE_MULT_TRENDING_UP = 0.75
EDGE_MULT_TRENDING_DOWN = 0.75
EDGE_MULT_RANGING = 1.5

# Fees: Polymarket crypto category 7.2 percent * p * (1-p) per share
FEE_RATE = 0.072


class WouldBeTrade(NamedTuple):
    market_id: str
    side: str
    entry_price: float       # market_implied_prob proxy
    est_prob_up: float
    prob_edge: float
    required_edge: float
    secs_remaining: float
    oracle_open_price: float
    btc_resolution_price: float
    won: bool
    pnl: float


def taker_fee(price: float) -> float:
    return FEE_RATE * price * (1.0 - price)


def required_edge(secs_remaining: float, edge_mult: float) -> float:
    time_pct = max(0.0, min(1.0, secs_remaining / 300.0))
    return (MIN_EDGE + (MAX_EDGE - MIN_EDGE) * time_pct) * edge_mult


def simulate_for_regime(
    db_path: str,
    target_regime: str,
    edge_multiplier: float,
) -> tuple[list, dict]:
    """Find first valid would-be entry per market in the given regime,
    compute would-be PnL based on observed window close. Returns
    (list of WouldBeTrade, stats dict)."""

    if not os.path.exists(db_path):
        return [], {"db_missing": True}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Step 1: load ticks in target regime, ordered by market and time
    cur.execute("""
        SELECT market_id, secs_remaining, btc_price, oracle_open_price,
               est_prob_up, market_implied_prob, regime
        FROM signal_ticks
        WHERE regime = ?
          AND secs_remaining BETWEEN ? AND ?
          AND market_id != ''
          AND oracle_open_price > 0
          AND btc_price > 0
        ORDER BY market_id, secs_remaining DESC
    """, (target_regime, LATEST_ENTRY_SECS, EARLIEST_ENTRY_SECS))
    candidate_rows = cur.fetchall()

    # Step 2: also load resolution info — for each market_id, find the
    # last tick (smallest secs_remaining) to get the closing BTC price.
    cur.execute("""
        SELECT market_id, btc_price, oracle_open_price, secs_remaining
        FROM signal_ticks
        WHERE market_id != ''
          AND btc_price > 0
        ORDER BY market_id, secs_remaining ASC
    """)
    last_tick_by_market = {}
    for mid, btc, opn, secs_rem in cur.fetchall():
        # Keep the tick with smallest secs_remaining (closest to resolution)
        existing = last_tick_by_market.get(mid)
        if existing is None or secs_rem < existing[2]:
            last_tick_by_market[mid] = (btc, opn, secs_rem)

    conn.close()

    # Step 3: for each market in target regime, find FIRST tick where the
    # bot would have entered. "First" by time (= largest secs_remaining).
    # Group ticks by market.
    market_ticks = defaultdict(list)
    for r in candidate_rows:
        market_ticks[r[0]].append(r)

    trades = []
    stats = {
        "n_markets_in_regime": len(market_ticks),
        "n_markets_with_signal": 0,
        "n_markets_with_resolution": 0,
        "n_skipped_no_resolution": 0,
        "n_skipped_no_signal": 0,
    }

    for market_id, ticks in market_ticks.items():
        # Find first valid entry: iterate in time order (largest secs_remaining first)
        # ticks are already ORDER BY market_id, secs_remaining DESC
        entry = None
        for mid, secs_rem, btc, opn, est_p, mkt_p, regime in ticks:
            if est_p is None or mkt_p is None:
                continue

            # REALISM GUARD: skip junk-data ticks where midpoint is at extreme.
            # These indicate one-sided book / window transition, not tradeable.
            if mkt_p < PRICE_MIN or mkt_p > PRICE_MAX:
                continue

            # signal_aligned vs contrarian. For high_vol/low_vol the regime
            # has no schedule_override -> contrarian mode is used.
            # For contrarian:
            #   side = YES if (est_p - mkt_p) >= 0 else NO
            #   prob_edge = abs(est_p - mkt_p)
            prob_edge_yes = est_p - mkt_p
            prob_edge = abs(prob_edge_yes)

            # REALISM GUARD: skip extreme prob_edges (likely spurious)
            if prob_edge > PROB_EDGE_MAX_SANE:
                continue

            if prob_edge_yes >= 0:
                side = "YES"
                price = mkt_p  # ask proxy
            else:
                side = "NO"
                price = 1.0 - mkt_p  # NO ask ~ 1 - YES mid

            # BOT G/K FILTER: yes_min_price (and symmetric NO version).
            # The actual bot has only YES filter; we apply both here so the
            # "lifted block" simulation matches what we'd reasonably allow
            # if we also did Fix 4 (symmetric filter). Mark which scenarios.
            if side == "YES" and price < YES_MIN_PRICE:
                continue
            if side == "NO" and price < NO_MIN_PRICE:
                continue

            # Confidence check
            signal_conf = abs(est_p - 0.5) * 2.0
            if signal_conf < MIN_CONFIDENCE:
                continue

            # Edge threshold (with regime multiplier)
            req_edge = required_edge(secs_rem, edge_multiplier)
            if prob_edge <= req_edge:
                continue

            # EV check
            fee = taker_fee(price)
            if side == "YES":
                ev = est_p * (1.0 - price) - (1.0 - est_p) * price - fee
            else:
                ev = (1.0 - est_p) * (1.0 - price) - est_p * price - fee
            if ev <= 0:
                continue

            # BOT G FILTER: high_ev_early — block high-EV trades fired early
            secs_into_window = 300.0 - secs_rem
            if (ev >= HIGH_EV_THRESHOLD
                    and secs_into_window < HIGH_EV_MIN_SECS_INTO_WIN):
                continue

            # All conditions met — this is the entry
            entry = (mid, secs_rem, btc, opn, est_p, mkt_p, side, price, ev)
            break

        if entry is None:
            stats["n_skipped_no_signal"] += 1
            continue
        stats["n_markets_with_signal"] += 1

        # Look up resolution
        last_tick = last_tick_by_market.get(market_id)
        if last_tick is None:
            stats["n_skipped_no_resolution"] += 1
            continue
        last_btc, last_opn, last_secs_rem = last_tick
        # Resolution: BTC > open at last tick -> UP wins (YES wins)
        # last_secs_rem should be near 0 for a complete window
        # opn from last tick should match entry opn (same market)
        oracle_open = entry[3]
        if oracle_open <= 0:
            stats["n_skipped_no_resolution"] += 1
            continue

        up_wins = last_btc > oracle_open
        stats["n_markets_with_resolution"] += 1

        # Compute PnL
        side = entry[6]
        price = entry[7]
        if price <= 0 or price >= 1.0:
            # Skip nonsensical prices
            continue
        won = (side == "YES" and up_wins) or (side == "NO" and not up_wins)
        fee = taker_fee(price)
        # Bet size = $3 (avg Bot G bet)
        # PnL per dollar bet on a binary outcome at price p:
        if won:
            pnl_per_dollar = (1.0 - price - fee) / price
        else:
            pnl_per_dollar = -1.0  # lose entire bet
        # Multiply by typical $3 bet (avg Bot G bet size)
        pnl_3 = pnl_per_dollar * 3.0
        pnl_5 = pnl_per_dollar * 5.0  # max bet equivalent

        trades.append(WouldBeTrade(
            market_id=market_id,
            side=side,
            entry_price=price,
            est_prob_up=entry[4],
            prob_edge=abs(entry[4] - entry[5]),
            required_edge=required_edge(entry[1], edge_multiplier),
            secs_remaining=entry[1],
            oracle_open_price=oracle_open,
            btc_resolution_price=last_btc,
            won=won,
            pnl=pnl_3,
        ))

    return trades, stats


def summary(label: str, trades: list[WouldBeTrade], stats: dict) -> None:
    print(f"\n{'-' * 75}")
    print(f"  {label}")
    print(f"{'-' * 75}")

    if stats.get("db_missing"):
        print(f"  (DB not found)")
        return

    print(f"  Markets in regime: {stats['n_markets_in_regime']:,}")
    print(f"  Markets with valid signal: {stats['n_markets_with_signal']:,}")
    print(f"  Markets with resolution data: {stats['n_markets_with_resolution']:,}")

    if not trades:
        print(f"  No would-be trades to evaluate.")
        return

    n = len(trades)
    wins = sum(1 for t in trades if t.won)
    yes_trades = [t for t in trades if t.side == "YES"]
    no_trades = [t for t in trades if t.side == "NO"]
    total_pnl = sum(t.pnl for t in trades)
    avg_pnl = total_pnl / n
    wr = wins / n

    print(f"\n  Would-be trades: {n:,}")
    print(f"    YES side: {len(yes_trades):,} ({len(yes_trades)/n:.1%})")
    print(f"    NO side:  {len(no_trades):,} ({len(no_trades)/n:.1%})")
    print(f"    Win rate: {wr:.1%}")
    print(f"    Total PnL (at $3 avg): ${total_pnl:+.2f}")
    print(f"    Avg PnL/trade:         ${avg_pnl:+.4f}")

    if yes_trades:
        yes_wins = sum(1 for t in yes_trades if t.won)
        yes_pnl = sum(t.pnl for t in yes_trades)
        print(f"\n    YES side: WR {yes_wins/len(yes_trades):.1%}  "
              f"PnL ${yes_pnl:+.2f}  Avg ${yes_pnl/len(yes_trades):+.4f}")
    if no_trades:
        no_wins = sum(1 for t in no_trades if t.won)
        no_pnl = sum(t.pnl for t in no_trades)
        print(f"    NO side:  WR {no_wins/len(no_trades):.1%}  "
              f"PnL ${no_pnl:+.2f}  Avg ${no_pnl/len(no_trades):+.4f}")

    # Per-trade std and Sharpe
    pnls = [t.pnl for t in trades]
    if len(pnls) > 1:
        std = statistics.stdev(pnls)
        sharpe = avg_pnl / std if std > 0 else 0
        print(f"\n    Per-trade stdev: ${std:.4f}")
        print(f"    Sharpe/trade:    {sharpe:.4f}")

    # Price distribution
    prices = [t.entry_price for t in trades]
    print(f"\n    Entry price (midpoint proxy): "
          f"min ${min(prices):.3f}  median ${sorted(prices)[len(prices)//2]:.3f}  "
          f"max ${max(prices):.3f}")

    # Top winners and losers
    sorted_trades = sorted(trades, key=lambda t: t.pnl)
    print(f"\n    Top 5 WINS:")
    for t in sorted_trades[-5:][::-1]:
        won_str = "WIN" if t.won else "LOSS"
        print(f"      {t.side} @ ${t.entry_price:.3f}  "
              f"est_p={t.est_prob_up:.3f}  edge={t.prob_edge:.3f}  "
              f"BTC: ${t.oracle_open_price:.0f} -> ${t.btc_resolution_price:.0f}  "
              f"{won_str}  PnL ${t.pnl:+.2f}")
    print(f"\n    Top 5 LOSSES:")
    for t in sorted_trades[:5]:
        won_str = "WIN" if t.won else "LOSS"
        print(f"      {t.side} @ ${t.entry_price:.3f}  "
              f"est_p={t.est_prob_up:.3f}  edge={t.prob_edge:.3f}  "
              f"BTC: ${t.oracle_open_price:.0f} -> ${t.btc_resolution_price:.0f}  "
              f"{won_str}  PnL ${t.pnl:+.2f}")


def main():
    print("=" * 80)
    print("HIGH_VOL REGIME BLOCK VALIDATION")
    print("Would high_vol-blocked trades have been profitable?")
    print("=" * 80)

    for bot_label, db_path in DBS:
        print(f"\n{'=' * 80}")
        print(f"  {bot_label}  ({db_path})")
        print(f"{'=' * 80}")

        # high_vol simulation
        hv_trades, hv_stats = simulate_for_regime(
            db_path, "high_vol", EDGE_MULT_HIGH_VOL
        )
        summary("HIGH_VOL regime (currently BLOCKED)", hv_trades, hv_stats)

        # For comparison: ranging trades (currently allowed)
        # This is a sanity check — should be roughly profitable like Bot G's history
        rng_trades, rng_stats = simulate_for_regime(
            db_path, "ranging", EDGE_MULT_RANGING
        )
        summary("RANGING regime (currently ALLOWED, comparison baseline)",
                rng_trades, rng_stats)

        # And trending_up which is also allowed
        tu_trades, tu_stats = simulate_for_regime(
            db_path, "trending_up", EDGE_MULT_TRENDING_UP
        )
        summary("TRENDING_UP regime (currently ALLOWED)", tu_trades, tu_stats)

        # Side-by-side comparison
        print(f"\n{'-' * 75}")
        print(f"  COMPARISON ({bot_label})")
        print(f"{'-' * 75}")
        print(f"  {'Regime':<20} {'N':>6} {'WR':>7} {'Total PnL':>12} {'Avg PnL':>10}")
        for label, ts in [
            ("ranging", rng_trades),
            ("trending_up", tu_trades),
            ("high_vol (blocked)", hv_trades),
        ]:
            if not ts:
                print(f"  {label:<20} {0:>6} {'---':>7} {'---':>12} {'---':>10}")
                continue
            n = len(ts)
            wins = sum(1 for t in ts if t.won)
            pnl = sum(t.pnl for t in ts)
            avg = pnl / n
            print(f"  {label:<20} {n:>6,} {wins/n:>6.1%} "
                  f"${pnl:>+10.2f} ${avg:>+8.4f}")

    print(f"\n{'=' * 80}")
    print("INTERPRETATION")
    print(f"{'=' * 80}")
    print("""
  If high_vol PnL is POSITIVE -> lifting the block could improve performance.
  If high_vol PnL is NEGATIVE -> the block is doing its job, keep it.
  If high_vol PnL is around zero -> block is removing noise but not edge.

  Sample size caveat: 27.6h is small. Confidence depends on N. >100 trades
  per regime = directionally meaningful. <30 trades = inconclusive.

  Comparison baseline: ranging is the regime where Bot G earned its
  historical edge. If ranging would-be-PnL is positive in this dataset,
  the simulation framework is at least internally consistent.
""")


if __name__ == "__main__":
    main()
