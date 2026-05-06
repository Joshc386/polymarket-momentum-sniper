"""Backtest: buy the favourite side in the last N seconds of a BTC window.

Supports both 5m and 15m market types via --market-type flag.

Strategy hypothesis:
    When there's only a minute (or less) left in a BTC up/down window and one
    side is heavily favoured (e.g. ask >= $0.80), buying that side may have
    positive expectancy — especially if BTC has moved in that direction
    confirming the market's view.

Method:
    1. Load all resolved markets (winner, btc_price_start).
    2. Filter snapshots to the target time-to-close window.
    3. For each snapshot, check strategy entry conditions.
    4. Simulate trade: buy favourite at ask, hold to resolution.
    5. Apply Polymarket crypto-category fees (7.2% × p × (1-p)).
    6. Sweep parameter grid and report.

Run:
    python -m backtest.backtest_late_favourite --market-type 5m
    python -m backtest.backtest_late_favourite --market-type 15m
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Optional

# Data paths resolved per market-type in main()
SNAPSHOTS_CSV = ""
MARKETS_CSV = ""
WINDOW_SECS = 300  # Set by --market-type

# Polymarket crypto category fee rate
CRYPTO_FEE_RATE = 0.072

# Default position size per trade (USDC)
BET_SIZE = 5.0


@dataclass
class Market:
    market_id: str
    winner: str           # "Up" or "Down"
    btc_price_start: float
    btc_price_end: float


@dataclass
class Trade:
    market_id: str
    side: str             # "UP" or "DOWN"
    entry_price: float
    shares: float
    fee: float
    won: bool
    pnl: float
    btc_price: float      # At entry (snapshot time)
    btc_price_start: float
    move_bps: float       # Signed: +ve if BTC moved up, -ve if down


def load_markets() -> dict[str, Market]:
    """Load resolved markets keyed by market_id."""
    markets: dict[str, Market] = {}
    with open(MARKETS_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            winner = (row.get("winner") or "").strip()
            if winner not in ("Up", "Down"):
                continue
            try:
                start = float(row["btc_price_start"])
                end = float(row["btc_price_end"])
            except (KeyError, TypeError, ValueError):
                continue
            markets[row["market_id"]] = Market(
                market_id=row["market_id"],
                winner=winner,
                btc_price_start=start,
                btc_price_end=end,
            )
    return markets


def iter_snapshots(seconds_remaining_max: int = 60):
    """Yield snapshot rows where seconds_remaining <= max."""
    with open(SNAPSHOTS_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sr = int(float(row["seconds_remaining"]))
            except (KeyError, TypeError, ValueError):
                continue
            if sr <= seconds_remaining_max:
                yield row


def compute_fee(shares: float, price: float) -> float:
    """Polymarket taker fee (crypto category).

    fee = shares × feeRate × p × (1 - p)
    Peaks at p=0.5, approaches 0 near 0 or 1.
    """
    return shares * CRYPTO_FEE_RATE * price * (1.0 - price)


def try_enter(
    row: dict,
    market: Market,
    min_ask: float,
    max_ask: float,
    min_move_bps: float,
) -> Optional[Trade]:
    """Apply strategy filters; if passed, return a simulated Trade."""
    try:
        btc_price = float(row["btc_price"])
        up_ask = float(row["up_best_ask"])
        down_ask = float(row["down_best_ask"])
    except (KeyError, TypeError, ValueError):
        return None

    if up_ask <= 0 or down_ask <= 0:
        return None
    if market.btc_price_start <= 0:
        return None

    # Determine favourite (lower-probability side would be pricier for
    # the winning contract — actually favourite = cheapest losing side...
    # No: on Polymarket, YES/NO sum ≈ $1. Favourite = side with HIGHER ask
    # because higher ask means higher implied probability of winning.
    if up_ask >= down_ask:
        side = "UP"
        fav_ask = up_ask
    else:
        side = "DOWN"
        fav_ask = down_ask

    # Ask price band
    if fav_ask < min_ask or fav_ask > max_ask:
        return None

    # Price move from window open (signed)
    move_bps = (btc_price - market.btc_price_start) / market.btc_price_start * 10_000

    # Movement must confirm favourite's direction
    if side == "UP":
        if move_bps < min_move_bps:
            return None
    else:  # DOWN
        if -move_bps < min_move_bps:
            return None

    # Simulate trade
    shares = BET_SIZE / fav_ask
    fee = compute_fee(shares, fav_ask)

    won = (side == "UP" and market.winner == "Up") or \
          (side == "DOWN" and market.winner == "Down")

    if won:
        gross = shares * (1.0 - fav_ask)
    else:
        gross = -shares * fav_ask
    pnl = gross - fee

    return Trade(
        market_id=market.market_id,
        side=side,
        entry_price=fav_ask,
        shares=shares,
        fee=fee,
        won=won,
        pnl=pnl,
        btc_price=btc_price,
        btc_price_start=market.btc_price_start,
        move_bps=move_bps,
    )


def run_backtest(
    markets: dict[str, Market],
    min_ask: float,
    max_ask: float,
    min_move_bps: float,
    seconds_remaining_max: int = 60,
) -> list[Trade]:
    """Run backtest for one parameter set. Returns list of simulated trades."""
    # Keep one trade per market (first qualifying snapshot in the window)
    traded_markets: set[str] = set()
    trades: list[Trade] = []

    for row in iter_snapshots(seconds_remaining_max):
        mid = row["market_id"]
        if mid in traded_markets:
            continue
        market = markets.get(mid)
        if market is None:
            continue
        trade = try_enter(row, market, min_ask, max_ask, min_move_bps)
        if trade is not None:
            trades.append(trade)
            traded_markets.add(mid)

    return trades


def summarise(trades: list[Trade]) -> dict:
    """Compute summary stats for a list of trades."""
    if not trades:
        return {"n": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t.won)
    total_pnl = sum(t.pnl for t in trades)
    avg_pnl = total_pnl / n
    avg_entry = sum(t.entry_price for t in trades) / n
    # Sharpe-per-trade
    if n > 1:
        mean = avg_pnl
        var = sum((t.pnl - mean) ** 2 for t in trades) / (n - 1)
        stdev = var ** 0.5
        sharpe = mean / stdev if stdev > 0 else 0.0
    else:
        sharpe = 0.0

    up_n = sum(1 for t in trades if t.side == "UP")
    up_wins = sum(1 for t in trades if t.side == "UP" and t.won)
    up_pnl = sum(t.pnl for t in trades if t.side == "UP")
    dn_n = n - up_n
    dn_wins = wins - up_wins
    dn_pnl = total_pnl - up_pnl

    return {
        "n": n,
        "wins": wins,
        "wr_pct": wins / n * 100,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "avg_entry": avg_entry,
        "sharpe_trade": sharpe,
        "up_n": up_n,
        "up_wr_pct": up_wins / up_n * 100 if up_n else 0,
        "up_pnl": up_pnl,
        "dn_n": dn_n,
        "dn_wr_pct": dn_wins / dn_n * 100 if dn_n else 0,
        "dn_pnl": dn_pnl,
    }


def format_row(params: dict, stats: dict) -> str:
    """One-line summary."""
    if stats["n"] == 0:
        return (
            f"  ask[{params['min_ask']:.2f}-{params['max_ask']:.2f}]  "
            f"move>={params['min_move_bps']:>2}bps  "
            f"-> 0 trades"
        )
    return (
        f"  ask[{params['min_ask']:.2f}-{params['max_ask']:.2f}]  "
        f"move>={params['min_move_bps']:>2}bps  "
        f"n={stats['n']:>4}  WR={stats['wr_pct']:>4.1f}%  "
        f"PnL=${stats['total_pnl']:>+7.2f}  "
        f"avg=${stats['avg_pnl']:>+.3f}  "
        f"sharpe={stats['sharpe_trade']:>+.3f}  "
        f"UP[n={stats['up_n']:>3} WR={stats['up_wr_pct']:.0f}% ${stats['up_pnl']:>+6.2f}]  "
        f"DN[n={stats['dn_n']:>3} WR={stats['dn_wr_pct']:.0f}% ${stats['dn_pnl']:>+6.2f}]"
    )


def run_for_window(
    markets: dict[str, Market],
    seconds_remaining_max: int,
    label: str,
) -> None:
    """Run the full parameter sweep for one entry timing."""
    min_asks = [0.75, 0.80, 0.85, 0.90]
    max_asks = [0.92, 0.95, 0.98]
    min_moves = [0, 5, 10, 20, 30, 50, 75]

    print(f"\n{'#'*120}\n# {label}  |  seconds_remaining <= {seconds_remaining_max}s\n{'#'*120}")

    results = []
    for min_ask in min_asks:
        for max_ask in max_asks:
            if max_ask <= min_ask:
                continue
            for mv in min_moves:
                trades = run_backtest(
                    markets, min_ask, max_ask, mv, seconds_remaining_max
                )
                stats = summarise(trades)
                params = {
                    "min_ask": min_ask,
                    "max_ask": max_ask,
                    "min_move_bps": mv,
                }
                results.append((params, stats))

    # Top 10 by total PnL (min 30 trades)
    print(f"\nTop 10 by total PnL (n >= 30):")
    eligible = [(p, s) for p, s in results if s.get("n", 0) >= 30]
    eligible.sort(key=lambda x: x[1]["total_pnl"], reverse=True)
    for params, stats in eligible[:10]:
        print(format_row(params, stats))

    # Top 10 by Sharpe
    print(f"\nTop 10 by Sharpe-per-trade (n >= 30):")
    eligible.sort(key=lambda x: x[1].get("sharpe_trade", -999), reverse=True)
    for params, stats in eligible[:10]:
        print(format_row(params, stats))

    # Baseline (no move filter)
    print(f"\nBaseline (no move filter):")
    for p, s in results:
        if p["min_move_bps"] == 0:
            print(format_row(p, s))

    # Does move filter progression show a real signal?
    print(f"\nMove filter sensitivity (ask band 0.80-0.95):")
    for p, s in results:
        if p["min_ask"] == 0.80 and p["max_ask"] == 0.95:
            print(format_row(p, s))


def main():
    global SNAPSHOTS_CSV, MARKETS_CSV, WINDOW_SECS

    parser = argparse.ArgumentParser()
    parser.add_argument("--market-type", choices=["5m", "15m"], default="5m",
                        help="Which market type to backtest")
    parser.add_argument("--offsets", nargs="*", type=int, default=None,
                        help="seconds_remaining_max values to sweep "
                             "(e.g. 60 120 300). Default = sensible per market-type.")
    args = parser.parse_args()

    if args.market_type == "5m":
        MARKETS_CSV = "backtest/data/polybacktest_markets.csv"
        SNAPSHOTS_CSV = "backtest/data/polybacktest_snapshots.csv"
        WINDOW_SECS = 300
        default_offsets = [60]  # 5m only has snapshot at offset_sec=240 = 60 remaining
    else:  # 15m
        MARKETS_CSV = "backtest/data/polybacktest_markets_15m.csv"
        SNAPSHOTS_CSV = "backtest/data/polybacktest_snapshots_15m.csv"
        WINDOW_SECS = 900
        default_offsets = [60, 120, 300]  # last 1/2/5 min

    offsets = args.offsets or default_offsets

    if not os.path.exists(MARKETS_CSV):
        print(f"ERROR: {MARKETS_CSV} not found. Run fetcher first.")
        return
    if not os.path.exists(SNAPSHOTS_CSV):
        print(f"ERROR: {SNAPSHOTS_CSV} not found. Run fetcher first.")
        return

    print(f"Market type: {args.market_type}")
    print(f"Loading markets from {MARKETS_CSV}...")
    markets = load_markets()
    print(f"  {len(markets):,} resolved markets")
    print(f"Bet size: ${BET_SIZE:.0f}/trade  |  "
          f"Fee rate: {CRYPTO_FEE_RATE:.1%} (crypto category)")
    print(f"Window: {WINDOW_SECS}s  |  Offsets to test: {offsets}")

    for off in offsets:
        label = f"Entry at {off}s remaining ({(WINDOW_SECS-off)//60}:{(WINDOW_SECS-off)%60:02d} into window)"
        run_for_window(markets, off, label)


if __name__ == "__main__":
    main()
