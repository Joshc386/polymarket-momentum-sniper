"""Backtest using real PolyBackTest Pro pricing data.

Reconstructs signals from BTC price history and uses real Polymarket
order book prices for entry/exit. This is the ground-truth backtest.

Usage:
    python -m backtest.backtest_real_pricing

Reads:
    backtest/data/polybacktest_markets.csv
    backtest/data/polybacktest_snapshots.csv
    backtest/data/polybacktest_spot.csv (optional — for taker buy ratio)

Strategies tested:
    1. Contrarian EV — buy whichever side has best EV (current live strategy)
    2. Signal-follow — buy the side our signal points to
    3. Order book fade — bet against extreme imbalance
    4. Contrarian EV + order book fade combined
"""

import csv
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ─── Strategy Parameters (match live bot config.yaml) ─────────────────
FEE_RATE = 0.02          # Polymarket 2% fee on winnings only
MIN_EDGE = 0.003         # Minimum required EV
MAX_EDGE = 0.025         # Maximum required EV (at earliest entry)
MIN_CONFIDENCE = 0.02    # Minimum signal confidence
MAX_ADJUSTMENT = 0.20    # Signal → probability scaling
EARLIEST_ENTRY = 270     # Seconds remaining — start scanning
LATEST_ENTRY = 60        # Seconds remaining — stop scanning


# ─── Signal Reconstruction ────────────────────────────────────────────

def compute_momentum_signal(
    prices: list[float],
    current_price: float,
    start_price: float,
) -> float:
    """Simplified momentum signal from BTC prices.

    Uses rate-of-change and direction to approximate the live bot's
    momentum layer. In the live bot this uses 1-min candles with RSI,
    body ratio, etc. — here we approximate from available BTC prices.

    Args:
        prices: BTC prices at earlier timepoints in this window.
        current_price: BTC price at the entry timepoint.
        start_price: BTC price at window open.

    Returns:
        Signal in [-1.0, 1.0].
    """
    if start_price <= 0:
        return 0.0

    # Rate of change from window open
    roc = (current_price - start_price) / start_price

    # Scale: 0.1% move = moderate signal, 0.3%+ = strong
    roc_signal = max(-1.0, min(1.0, roc / 0.003))

    # Direction consistency: are all prices trending one way?
    if len(prices) >= 2:
        up_moves = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i - 1])
        down_moves = sum(1 for i in range(1, len(prices)) if prices[i] < prices[i - 1])
        total_moves = up_moves + down_moves
        if total_moves > 0:
            direction = (up_moves - down_moves) / total_moves
            roc_signal = 0.6 * roc_signal + 0.4 * direction
        else:
            roc_signal *= 0.6

    return max(-1.0, min(1.0, roc_signal))


def compute_oracle_lag_signal(
    btc_price: float,
    start_price: float,
    price_up: float,
    price_down: float,
) -> float:
    """Simplified oracle lag signal.

    Compares what BTC price implies vs what the market is pricing.
    If BTC has moved up significantly but the market still prices
    UP cheaply, there's a lag to exploit.

    Args:
        btc_price: Current BTC price from Binance.
        start_price: BTC price at window open (oracle reference).
        price_up: Current Polymarket UP token price.
        price_down: Current Polymarket DOWN token price.

    Returns:
        Signal in [-1.0, 1.0].
    """
    if start_price <= 0:
        return 0.0

    # What BTC price implies
    btc_move = (btc_price - start_price) / start_price
    # Scale: 0.05% = mild, 0.2%+ = strong
    implied_direction = max(-1.0, min(1.0, btc_move / 0.002))

    # What the market is pricing
    if price_up and price_down and (price_up + price_down) > 0:
        market_direction = (price_up - price_down) / (price_up + price_down)
    else:
        market_direction = 0.0

    # The lag is the gap between BTC-implied and market-priced
    lag = implied_direction - market_direction
    lag_signal = max(-1.0, min(1.0, lag * 0.5))

    # Blend: mostly BTC direction, some lag
    signal = 0.7 * implied_direction + 0.3 * lag_signal

    return max(-1.0, min(1.0, signal))


def compute_orderbook_imbalance(snapshot: dict) -> float:
    """Compute order book imbalance from snapshot data.

    Positive = more bid depth on UP side (market expects UP).
    Uses the @polybacktest finding: heavy imbalance is CONTRARIAN.

    Returns:
        Imbalance ratio. >0 means UP side has more bids.
    """
    up_bid = snapshot.get("up_bid_depth_5", 0) or 0
    up_ask = snapshot.get("up_ask_depth_5", 0) or 0
    down_bid = snapshot.get("down_bid_depth_5", 0) or 0
    down_ask = snapshot.get("down_ask_depth_5", 0) or 0

    total = up_bid + up_ask + down_bid + down_ask
    if total == 0:
        return 0.0

    # Imbalance: positive means UP side has more bids relative to asks
    up_imbalance = (up_bid - up_ask) / (up_bid + up_ask) if (up_bid + up_ask) > 0 else 0
    down_imbalance = (down_bid - down_ask) / (down_bid + down_ask) if (down_bid + down_ask) > 0 else 0

    return up_imbalance - down_imbalance


def combine_signals(
    oracle_signal: float,
    momentum_signal: float,
    seconds_remaining: float,
) -> tuple[float, float]:
    """Simplified signal combiner matching live bot logic.

    Only uses oracle + momentum (no liquidation/sentiment in backtest).
    Redistributes weights to available signals.

    Returns:
        (raw_signal, est_prob_up)
    """
    # Weight schedule (simplified from combiner.py)
    if seconds_remaining >= 180:
        w_oracle, w_momentum = 0.35, 0.65
    elif seconds_remaining >= 90:
        w_oracle, w_momentum = 0.30, 0.70
    else:
        w_oracle, w_momentum = 0.25, 0.75

    raw_signal = w_oracle * oracle_signal + w_momentum * momentum_signal
    raw_signal = max(-1.0, min(1.0, raw_signal))

    est_prob_up = 0.5 + (raw_signal * MAX_ADJUSTMENT)
    est_prob_up = max(0.05, min(0.95, est_prob_up))

    return raw_signal, est_prob_up


# ─── EV Calculation ───────────────────────────────────────────────────

def compute_ev(est_prob_up: float, yes_ask: float, no_ask: float) -> tuple[float, float]:
    """Compute EV for both sides, matching live bot entry_logic.py."""
    profit_yes = 1.0 - yes_ask
    ev_yes = (est_prob_up * (profit_yes * (1.0 - FEE_RATE))) - ((1.0 - est_prob_up) * yes_ask)

    profit_no = 1.0 - no_ask
    ev_no = ((1.0 - est_prob_up) * (profit_no * (1.0 - FEE_RATE))) - (est_prob_up * no_ask)

    return ev_yes, ev_no


def required_edge(seconds_remaining: float) -> float:
    """Dynamic threshold matching live bot."""
    time_pct = seconds_remaining / 300.0
    return MIN_EDGE + (MAX_EDGE - MIN_EDGE) * time_pct


# ─── Strategy Definitions ─────────────────────────────────────────────

@dataclass
class TradeResult:
    market_id: str
    strategy: str
    side: str
    entry_price: float
    seconds_remaining: float
    est_prob_up: float
    ev: float
    won: bool
    pnl: float


def strategy_contrarian_ev(
    est_prob_up: float,
    snapshots_by_offset: dict[int, dict],
    btc_start: float,
    winner: str,
) -> TradeResult | None:
    """Current live strategy: buy whichever side has best EV."""
    # Try each timepoint from earliest to latest
    for offset_sec in sorted(snapshots_by_offset.keys()):
        snap = snapshots_by_offset[offset_sec]
        secs_remaining = 300 - offset_sec

        if secs_remaining > EARLIEST_ENTRY or secs_remaining < LATEST_ENTRY:
            continue

        yes_ask = snap.get("up_best_ask")
        no_ask = snap.get("down_best_ask")
        if not yes_ask or not no_ask or yes_ask <= 0 or no_ask <= 0:
            continue

        # Build price history for momentum signal
        earlier_prices = []
        for earlier_offset in sorted(snapshots_by_offset.keys()):
            if earlier_offset <= offset_sec:
                p = snapshots_by_offset[earlier_offset].get("btc_price")
                if p:
                    earlier_prices.append(p)

        btc_price = snap.get("btc_price", btc_start)
        price_up = snap.get("price_up", 0.5)
        price_down = snap.get("price_down", 0.5)

        # Compute signals
        momentum = compute_momentum_signal(earlier_prices, btc_price, btc_start)
        oracle = compute_oracle_lag_signal(btc_price, btc_start, price_up, price_down)
        _, prob_up = combine_signals(oracle, momentum, secs_remaining)

        # Signal confidence
        confidence = abs(prob_up - 0.5)
        if confidence < MIN_CONFIDENCE:
            continue

        # EV calculation
        ev_yes, ev_no = compute_ev(prob_up, yes_ask, no_ask)

        # Pick best EV side (contrarian — buys cheap side)
        if ev_yes >= ev_no:
            side, best_ev, price = "YES", ev_yes, yes_ask
        else:
            side, best_ev, price = "NO", ev_no, no_ask

        # Threshold check
        req_edge = required_edge(secs_remaining)
        if best_ev <= req_edge:
            continue

        # Trade taken — compute PnL
        won = (side == "YES" and winner == "Up") or (side == "NO" and winner == "Down")
        if won:
            gross_profit = 1.0 - price
            pnl = gross_profit - (gross_profit * FEE_RATE)
        else:
            pnl = -price

        return TradeResult(
            market_id=snap["market_id"],
            strategy="contrarian_ev",
            side=side,
            entry_price=price,
            seconds_remaining=secs_remaining,
            est_prob_up=prob_up,
            ev=best_ev,
            won=won,
            pnl=pnl,
        )

    return None


def strategy_signal_follow(
    est_prob_up: float,
    snapshots_by_offset: dict[int, dict],
    btc_start: float,
    winner: str,
) -> TradeResult | None:
    """Buy the side our signal points to (ignores EV of cheap side)."""
    for offset_sec in sorted(snapshots_by_offset.keys()):
        snap = snapshots_by_offset[offset_sec]
        secs_remaining = 300 - offset_sec

        if secs_remaining > EARLIEST_ENTRY or secs_remaining < LATEST_ENTRY:
            continue

        yes_ask = snap.get("up_best_ask")
        no_ask = snap.get("down_best_ask")
        if not yes_ask or not no_ask or yes_ask <= 0 or no_ask <= 0:
            continue

        earlier_prices = []
        for earlier_offset in sorted(snapshots_by_offset.keys()):
            if earlier_offset <= offset_sec:
                p = snapshots_by_offset[earlier_offset].get("btc_price")
                if p:
                    earlier_prices.append(p)

        btc_price = snap.get("btc_price", btc_start)
        price_up = snap.get("price_up", 0.5)
        price_down = snap.get("price_down", 0.5)

        momentum = compute_momentum_signal(earlier_prices, btc_price, btc_start)
        oracle = compute_oracle_lag_signal(btc_price, btc_start, price_up, price_down)
        _, prob_up = combine_signals(oracle, momentum, secs_remaining)

        confidence = abs(prob_up - 0.5)
        if confidence < MIN_CONFIDENCE:
            continue

        # Follow signal direction
        if prob_up >= 0.5:
            side, price = "YES", yes_ask
        else:
            side, price = "NO", no_ask

        ev_yes, ev_no = compute_ev(prob_up, yes_ask, no_ask)
        ev = ev_yes if side == "YES" else ev_no

        req_edge = required_edge(secs_remaining)
        if ev <= req_edge:
            continue

        won = (side == "YES" and winner == "Up") or (side == "NO" and winner == "Down")
        if won:
            gross_profit = 1.0 - price
            pnl = gross_profit - (gross_profit * FEE_RATE)
        else:
            pnl = -price

        return TradeResult(
            market_id=snap["market_id"],
            strategy="signal_follow",
            side=side,
            entry_price=price,
            seconds_remaining=secs_remaining,
            est_prob_up=prob_up,
            ev=ev,
            won=won,
            pnl=pnl,
        )

    return None


def strategy_orderbook_fade(
    snapshots_by_offset: dict[int, dict],
    btc_start: float,
    winner: str,
) -> TradeResult | None:
    """Fade extreme order book imbalance (@polybacktest finding)."""
    IMBALANCE_THRESHOLD = 0.3  # Moderate+ imbalance

    for offset_sec in sorted(snapshots_by_offset.keys()):
        snap = snapshots_by_offset[offset_sec]
        secs_remaining = 300 - offset_sec

        if secs_remaining > EARLIEST_ENTRY or secs_remaining < LATEST_ENTRY:
            continue

        yes_ask = snap.get("up_best_ask")
        no_ask = snap.get("down_best_ask")
        if not yes_ask or not no_ask or yes_ask <= 0 or no_ask <= 0:
            continue

        imbalance = compute_orderbook_imbalance(snap)

        if abs(imbalance) < IMBALANCE_THRESHOLD:
            continue

        # FADE the imbalance: heavy UP bids → bet DOWN
        if imbalance > 0:
            side, price = "NO", no_ask
        else:
            side, price = "YES", yes_ask

        # Require minimum EV even for fade trades
        btc_price = snap.get("btc_price", btc_start)
        # Use a flat 0.5 prob for EV check (we're not using signal direction)
        ev_yes, ev_no = compute_ev(0.5, yes_ask, no_ask)
        ev = ev_yes if side == "YES" else ev_no

        # Lower threshold for fade trades (the imbalance IS the edge)
        if ev <= 0.0:
            continue

        won = (side == "YES" and winner == "Up") or (side == "NO" and winner == "Down")
        if won:
            gross_profit = 1.0 - price
            pnl = gross_profit - (gross_profit * FEE_RATE)
        else:
            pnl = -price

        return TradeResult(
            market_id=snap["market_id"],
            strategy="orderbook_fade",
            side=side,
            entry_price=price,
            seconds_remaining=secs_remaining,
            est_prob_up=0.5,
            ev=ev,
            won=won,
            pnl=pnl,
        )

    return None


# ─── Data Loading ─────────────────────────────────────────────────────

def load_markets(path: str) -> list[dict]:
    """Load market metadata CSV."""
    markets = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["btc_price_start"] = float(row["btc_price_start"]) if row["btc_price_start"] else 0
            row["btc_price_end"] = float(row["btc_price_end"]) if row["btc_price_end"] else 0
            row["final_volume"] = float(row["final_volume"]) if row["final_volume"] else 0
            markets.append(row)
    return markets


def load_snapshots(path: str) -> dict[str, dict[int, dict]]:
    """Load snapshots CSV, grouped by market_id → offset_sec → snapshot."""
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["market_id"]
            offset = int(row["offset_sec"])
            # Parse numeric fields
            for key in ["btc_price", "price_up", "price_down",
                        "up_best_ask", "up_best_bid", "down_best_ask", "down_best_bid",
                        "up_bid_depth_5", "up_ask_depth_5", "down_bid_depth_5", "down_ask_depth_5"]:
                val = row.get(key)
                row[key] = float(val) if val and val != "None" else None
            grouped[mid][offset] = row
    return grouped


# ─── Results Reporting ────────────────────────────────────────────────

@dataclass
class StrategyStats:
    name: str
    trades: list[TradeResult] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.won)

    @property
    def win_rate(self) -> float:
        return self.wins / self.count if self.count else 0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.count if self.count else 0

    @property
    def avg_entry_price(self) -> float:
        return sum(t.entry_price for t in self.trades) / self.count if self.count else 0

    @property
    def avg_win_pnl(self) -> float:
        wins = [t.pnl for t in self.trades if t.won]
        return sum(wins) / len(wins) if wins else 0

    @property
    def avg_loss_pnl(self) -> float:
        losses = [t.pnl for t in self.trades if not t.won]
        return sum(losses) / len(losses) if losses else 0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        peak = 0.0
        cumulative = 0.0
        max_dd = 0.0
        for t in self.trades:
            cumulative += t.pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def participation_rate(self) -> float:
        """Requires total_markets to be set externally."""
        return 0.0  # Computed in report

    def side_breakdown(self) -> dict:
        yes_trades = [t for t in self.trades if t.side == "YES"]
        no_trades = [t for t in self.trades if t.side == "NO"]
        return {
            "YES": {
                "count": len(yes_trades),
                "wr": sum(1 for t in yes_trades if t.won) / len(yes_trades) if yes_trades else 0,
                "pnl": sum(t.pnl for t in yes_trades),
                "avg_price": sum(t.entry_price for t in yes_trades) / len(yes_trades) if yes_trades else 0,
            },
            "NO": {
                "count": len(no_trades),
                "wr": sum(1 for t in no_trades if t.won) / len(no_trades) if no_trades else 0,
                "pnl": sum(t.pnl for t in no_trades),
                "avg_price": sum(t.entry_price for t in no_trades) / len(no_trades) if no_trades else 0,
            },
        }

    def time_breakdown(self) -> dict:
        """WR and PnL by seconds remaining bucket."""
        buckets: dict[str, list[TradeResult]] = defaultdict(list)
        for t in self.trades:
            if t.seconds_remaining >= 240:
                buckets["240-270s"].append(t)
            elif t.seconds_remaining >= 180:
                buckets["180-240s"].append(t)
            elif t.seconds_remaining >= 120:
                buckets["120-180s"].append(t)
            else:
                buckets["60-120s"].append(t)

        result = {}
        for bucket, trades in sorted(buckets.items()):
            wins = sum(1 for t in trades if t.won)
            result[bucket] = {
                "count": len(trades),
                "wr": wins / len(trades) if trades else 0,
                "pnl": sum(t.pnl for t in trades),
            }
        return result


def print_report(stats: StrategyStats, total_markets: int) -> None:
    """Print detailed strategy report."""
    print(f"\n{'='*60}")
    print(f"  Strategy: {stats.name}")
    print(f"{'='*60}")
    print(f"  Trades:           {stats.count:,}")
    print(f"  Participation:    {stats.count/total_markets*100:.1f}% ({stats.count}/{total_markets})")
    print(f"  Win Rate:         {stats.win_rate*100:.1f}%")
    print(f"  Total PnL:        ${stats.total_pnl:+,.2f}")
    print(f"  Avg PnL/trade:    ${stats.avg_pnl:+,.4f}")
    print(f"  Avg Entry Price:  ${stats.avg_entry_price:.3f}")
    print(f"  Avg Win:          ${stats.avg_win_pnl:+,.4f}")
    print(f"  Avg Loss:         ${stats.avg_loss_pnl:+,.4f}")
    print(f"  Profit Factor:    {stats.profit_factor:.3f}")
    print(f"  Max Drawdown:     ${stats.max_drawdown:,.2f}")

    sides = stats.side_breakdown()
    print(f"\n  Side Breakdown:")
    for side_name, s in sides.items():
        print(f"    {side_name}: {s['count']} trades, {s['wr']*100:.1f}% WR, "
              f"${s['pnl']:+,.2f} PnL, avg price ${s['avg_price']:.3f}")

    timing = stats.time_breakdown()
    print(f"\n  Entry Timing:")
    for bucket, b in timing.items():
        print(f"    {bucket}: {b['count']} trades, {b['wr']*100:.1f}% WR, ${b['pnl']:+,.2f}")

    print()


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    markets_path = os.path.join(DATA_DIR, "polybacktest_markets.csv")
    snapshots_path = os.path.join(DATA_DIR, "polybacktest_snapshots.csv")

    if not os.path.exists(markets_path):
        logger.error(f"Markets file not found: {markets_path}")
        logger.error("Run: python -m backtest.fetch_polybacktest_pro")
        return

    if not os.path.exists(snapshots_path):
        logger.error(f"Snapshots file not found: {snapshots_path}")
        logger.error("Run: python -m backtest.fetch_polybacktest_pro")
        return

    logger.info("Loading data...")
    markets = load_markets(markets_path)
    snapshots = load_snapshots(snapshots_path)
    logger.info(f"Loaded {len(markets)} markets, {sum(len(v) for v in snapshots.values())} snapshots")

    # Run all strategies
    strategies = {
        "contrarian_ev": StrategyStats("Contrarian EV (current live)"),
        "signal_follow": StrategyStats("Signal Follow"),
        "orderbook_fade": StrategyStats("Order Book Fade"),
    }

    skipped_no_snapshots = 0

    for market in markets:
        mid = market["market_id"]
        winner = market.get("winner")
        btc_start = market.get("btc_price_start", 0)

        if not winner or not btc_start:
            continue

        snaps = snapshots.get(mid, {})
        if not snaps:
            skipped_no_snapshots += 1
            continue

        # Strategy 1: Contrarian EV
        result = strategy_contrarian_ev(0.0, snaps, btc_start, winner)
        if result:
            strategies["contrarian_ev"].trades.append(result)

        # Strategy 2: Signal Follow
        result = strategy_signal_follow(0.0, snaps, btc_start, winner)
        if result:
            strategies["signal_follow"].trades.append(result)

        # Strategy 3: Order Book Fade
        result = strategy_orderbook_fade(snaps, btc_start, winner)
        if result:
            strategies["orderbook_fade"].trades.append(result)

    total = len(markets) - skipped_no_snapshots
    logger.info(f"Backtested {total} markets ({skipped_no_snapshots} skipped — no snapshots)")

    print(f"\n{'#'*60}")
    print(f"  BACKTEST RESULTS — {total} markets with real pricing")
    print(f"  Data: PolyBackTest Pro (BTC 5m)")
    print(f"  Parameters: min_edge={MIN_EDGE}, max_edge={MAX_EDGE}, "
          f"fee={FEE_RATE}, max_adj={MAX_ADJUSTMENT}")
    print(f"{'#'*60}")

    for stats in strategies.values():
        print_report(stats, total)

    # Summary comparison table
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Strategy':<30} {'Trades':>7} {'WR':>7} {'PnL':>10} {'PF':>7}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*10} {'-'*7}")
    for stats in strategies.values():
        print(f"  {stats.name:<30} {stats.count:>7,} {stats.win_rate*100:>6.1f}% "
              f"${stats.total_pnl:>+9,.2f} {stats.profit_factor:>7.3f}")


if __name__ == "__main__":
    main()
