"""Strategy Prototyper Agent — end-to-end strategy backtesting with full risk management.

Takes a strategy definition, runs it through the backtest pipeline with Kelly sizing,
daily loss caps, and streak management, then produces a go/no-go recommendation.

Commands:
    python -m agents.strategy_prototyper run --strategy <name>         — Full backtest
    python -m agents.strategy_prototyper compare --strategies A,B,C    — Side-by-side comparison
    python -m agents.strategy_prototyper promote --strategy <name>     — Generate deployable module
    python -m agents.strategy_prototyper list                          — List available strategies
"""

import argparse
import csv
import importlib
import inspect
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.common import (
    DATA_DIR,
    format_table,
    load_backtest_data,
    load_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agents.strategy_prototyper")


# ─── Strategy Protocol ──────────────────────────────────────────────

class StrategyPrototype(Protocol):
    """Protocol that all strategy prototypes must implement."""
    name: str
    description: str

    def should_enter(
        self,
        snap: dict,
        regime: str,
        secs_remaining: float,
    ) -> tuple[bool, str, float]:
        """Decide whether to enter a trade.

        Args:
            snap: Snapshot data dict with prices, orderbook depths, etc.
            regime: Current market regime name.
            secs_remaining: Seconds remaining in the 5-min window.

        Returns:
            (should_enter, side, confidence)
            side: "YES" or "NO"
            confidence: 0.0-1.0 strength of conviction
        """
        ...

    def entry_params(self) -> dict:
        """Return strategy-specific parameters.

        Expected keys:
            min_edge: float — minimum required EV
            max_spread: float — maximum allowed spread
            min_time_remaining: float — don't enter below this
            max_time_remaining: float — don't enter above this
        """
        ...


# ─── Built-in Strategy Wrappers ────────────────────────────────────

class ContrairianEVStrategy:
    """Wraps the existing contrarian EV strategy from backtest_real_pricing.py."""
    name = "contrarian_ev"
    description = "Buy whichever side has best EV above dynamic threshold"

    def should_enter(self, snap: dict, regime: str, secs_remaining: float) -> tuple[bool, str, float]:
        if regime == "low_vol":
            return False, "", 0.0

        yes_ask = float(snap.get("up_best_ask", 0))
        no_ask = float(snap.get("down_best_ask", 0))
        if yes_ask <= 0 or no_ask <= 0:
            return False, "", 0.0

        price_up = float(snap.get("price_up", 0.5))
        prob_up = max(0.05, min(0.95, price_up))
        confidence = abs(prob_up - 0.5)
        if confidence < 0.02:
            return False, "", 0.0

        ev_yes = prob_up * (1.0 - yes_ask) * 0.98 - (1.0 - prob_up) * yes_ask
        ev_no = (1.0 - prob_up) * (1.0 - no_ask) * 0.98 - prob_up * no_ask

        if ev_yes >= ev_no and ev_yes > 0.003:
            return True, "YES", confidence
        elif ev_no > ev_yes and ev_no > 0.003:
            return True, "NO", confidence
        return False, "", 0.0

    def entry_params(self) -> dict:
        return {
            "min_edge": 0.003,
            "max_spread": 0.10,
            "min_time_remaining": 60,
            "max_time_remaining": 270,
        }


class SignalFollowStrategy:
    """Wraps the signal follow strategy."""
    name = "signal_follow"
    description = "Buy the side our signal points to"

    def should_enter(self, snap: dict, regime: str, secs_remaining: float) -> tuple[bool, str, float]:
        if regime == "low_vol":
            return False, "", 0.0

        price_up = float(snap.get("price_up", 0.5))
        prob_up = max(0.05, min(0.95, price_up))
        confidence = abs(prob_up - 0.5)
        if confidence < 0.02:
            return False, "", 0.0

        yes_ask = float(snap.get("up_best_ask", 0))
        no_ask = float(snap.get("down_best_ask", 0))
        if yes_ask <= 0 or no_ask <= 0:
            return False, "", 0.0

        if prob_up >= 0.5:
            side = "YES"
            ev = prob_up * (1.0 - yes_ask) * 0.98 - (1.0 - prob_up) * yes_ask
        else:
            side = "NO"
            ev = (1.0 - prob_up) * (1.0 - no_ask) * 0.98 - prob_up * no_ask

        if ev > 0.003:
            return True, side, confidence
        return False, "", 0.0

    def entry_params(self) -> dict:
        return {
            "min_edge": 0.003,
            "max_spread": 0.10,
            "min_time_remaining": 60,
            "max_time_remaining": 270,
        }


class OrderbookFadeStrategy:
    """Wraps the orderbook fade strategy."""
    name = "orderbook_fade"
    description = "Fade extreme orderbook imbalance"

    def should_enter(self, snap: dict, regime: str, secs_remaining: float) -> tuple[bool, str, float]:
        up_bid = float(snap.get("up_bid_depth_5", 0))
        up_ask = float(snap.get("up_ask_depth_5", 0))
        down_bid = float(snap.get("down_bid_depth_5", 0))
        down_ask = float(snap.get("down_ask_depth_5", 0))

        total = up_bid + up_ask + down_bid + down_ask
        if total <= 0:
            return False, "", 0.0

        imbalance = ((up_bid + down_ask) - (up_ask + down_bid)) / total
        threshold = 0.3

        if abs(imbalance) < threshold:
            return False, "", 0.0

        # Fade: heavy UP bids -> bet DOWN
        if imbalance > 0:
            side = "NO"
        else:
            side = "YES"

        confidence = min(abs(imbalance), 1.0)
        return True, side, confidence

    def entry_params(self) -> dict:
        return {
            "min_edge": 0.0,
            "max_spread": 0.15,
            "min_time_remaining": 60,
            "max_time_remaining": 270,
        }


BUILTIN_STRATEGIES: dict[str, type] = {
    "contrarian_ev": ContrairianEVStrategy,
    "signal_follow": SignalFollowStrategy,
    "orderbook_fade": OrderbookFadeStrategy,
}


def _discover_strategies() -> dict[str, StrategyPrototype]:
    """Discover all available strategies (built-in + user-defined)."""
    strategies: dict[str, StrategyPrototype] = {}

    for name, cls in BUILTIN_STRATEGIES.items():
        strategies[name] = cls()

    # User-defined strategies from strategies/
    strat_dir = Path(PROJECT_ROOT) / "strategies"
    if strat_dir.exists():
        for py_file in strat_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                module_name = f"strategies.{py_file.stem}"
                module = importlib.import_module(module_name)
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if (hasattr(obj, "name") and hasattr(obj, "should_enter")
                            and hasattr(obj, "entry_params")):
                        if obj.__module__ == module_name:
                            instance = obj()
                            strategies[instance.name] = instance
                            logger.info(f"Discovered strategy: {instance.name}")
            except Exception as e:
                logger.warning(f"Failed to load strategy from {py_file.name}: {e}")

    return strategies


# ─── Backtest Engine ────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Results from running a strategy through the backtest."""
    strategy_name: str
    total_trades: int
    wins: int
    losses: int
    total_pnl: float
    final_bankroll: float
    initial_bankroll: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_bet_size: float
    sharpe: float
    trades: list[dict]


def _run_backtest(
    strategy: StrategyPrototype,
    config: dict,
) -> BacktestResult:
    """Run a full backtest with Kelly sizing and risk management.

    Reuses BacktestAccount from backtest_real_pricing.py for exact
    parity with the live bot's risk management.
    """
    from backtest.backtest_real_pricing import BacktestAccount

    # Load data
    snaps_path = DATA_DIR / "polybacktest_snapshots.csv"
    markets_path = DATA_DIR / "polybacktest_markets.csv"

    if not snaps_path.exists() or not markets_path.exists():
        logger.error("Required data files not found.")
        sys.exit(1)

    snaps = load_backtest_data(snaps_path)
    markets = load_backtest_data(markets_path)

    # Build market lookup
    market_info = {}
    for _, row in markets.iterrows():
        mid = row["market_id"]
        market_info[mid] = {
            "winner": row.get("winner", ""),
            "start_time": row.get("start_time", ""),
            "btc_start": float(row.get("btc_start", 0)) if "btc_start" in row else 0,
        }

    # Group snapshots by market
    market_snapshots: dict[str, dict[int, dict]] = defaultdict(dict)
    for _, snap in snaps.iterrows():
        mid = snap["market_id"]
        offset = int(snap.get("offset_sec", 0))
        market_snapshots[mid][offset] = snap.to_dict()

    # Load regime timeline if available
    regime_timeline: dict[str, str] = {}
    regime_path = DATA_DIR / "regime_timeline.csv"
    if regime_path.exists():
        regime_df = load_backtest_data(regime_path)
        for _, row in regime_df.iterrows():
            regime_timeline[str(row.get("timestamp", ""))] = row.get("regime", "ranging")

    # Init account
    sizing_config = config.get("sizing", {})
    risk_config = config.get("risk", {})
    initial_bankroll = sizing_config.get("initial_bankroll", 100.0)

    account = BacktestAccount(
        initial_bankroll=initial_bankroll,
        kelly_multiplier=sizing_config.get("kelly_multiplier", 0.25),
        min_bet_usdc=sizing_config.get("min_bet_usdc", 1.0),
        max_bet_usdc=sizing_config.get("max_bet_usdc", 5.0),
        daily_loss_cap_pct=risk_config.get("daily_loss_cap_pct", 0.20),
        daily_loss_warn_pct=risk_config.get("daily_loss_warn_pct", 0.15),
        min_bankroll=risk_config.get("min_bankroll", 10.0),
        streak_reduce_at=risk_config.get("streak_reduce_at", 3),
        streak_reduce_factor=risk_config.get("streak_reduce_factor", 0.75),
        streak_heavy_reduce_at=risk_config.get("streak_heavy_reduce_at", 5),
        streak_heavy_reduce_factor=risk_config.get("streak_heavy_reduce_factor", 0.5),
        streak_pause_at=risk_config.get("streak_pause_at", 7),
        streak_pause_minutes=risk_config.get("streak_pause_minutes", 30),
        streak_reset_wins=risk_config.get("streak_reset_wins", 3),
    )

    # Strategy params
    params = strategy.entry_params()
    fee_rate = config.get("entry", {}).get("fee_adjustment", 0.02)

    # Run backtest
    trades: list[dict] = []
    peak_bankroll = initial_bankroll
    max_drawdown = 0.0

    sorted_markets = sorted(
        market_info.keys(),
        key=lambda mid: market_info[mid].get("start_time", ""),
    )

    for market_id in sorted_markets:
        info = market_info[market_id]
        winner = info["winner"]
        start_time = info.get("start_time", "")
        if not winner:
            continue

        snapshots = market_snapshots.get(market_id, {})
        if not snapshots:
            continue

        # Estimate market timestamp for risk management
        market_ts = 0.0
        if start_time:
            try:
                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                market_ts = dt.timestamp()
            except ValueError:
                pass

        # Check risk limits
        can_trade, size_mult = account.can_trade(start_time, market_ts)
        if not can_trade:
            continue

        # Find nearest regime
        regime = "ranging"
        # Simple: use default regime if timeline not available

        # Try each offset (sorted by time into window)
        traded = False
        for offset in sorted(snapshots.keys()):
            secs_remaining = 300 - offset
            if secs_remaining < params.get("min_time_remaining", 60):
                continue
            if secs_remaining > params.get("max_time_remaining", 270):
                continue

            snap = snapshots[offset]
            should_enter, side, confidence = strategy.should_enter(snap, regime, secs_remaining)

            if not should_enter:
                continue

            # Determine entry price
            price = float(snap.get("up_best_ask", 0)) if side == "YES" else float(snap.get("down_best_ask", 0))
            if price <= 0 or price >= 1.0:
                continue

            # Kelly sizing
            est_prob_win = 0.5 + confidence * 0.2  # Map confidence to probability
            bet_size = account.compute_size(est_prob_win, price, size_mult)
            if bet_size <= 0:
                continue

            # Determine outcome
            won = (side == "YES" and winner.lower() in ["up", "yes"]) or \
                  (side == "NO" and winner.lower() in ["down", "no"])

            if won:
                gross_profit = (1.0 - price) * (bet_size / price)
                pnl = gross_profit - (gross_profit * fee_rate)
            else:
                pnl = -bet_size

            account.record_trade(pnl, market_ts)

            trades.append({
                "market_id": market_id,
                "side": side,
                "price": price,
                "size": bet_size,
                "confidence": confidence,
                "secs_remaining": secs_remaining,
                "won": won,
                "pnl": pnl,
                "bankroll_after": account.bankroll,
                "regime": regime,
                "start_time": start_time,
            })

            # Track drawdown
            if account.bankroll > peak_bankroll:
                peak_bankroll = account.bankroll
            dd = peak_bankroll - account.bankroll
            if dd > max_drawdown:
                max_drawdown = dd

            traded = True
            break  # One trade per market

    # Compute summary stats
    total_trades = len(trades)
    wins = sum(1 for t in trades if t["won"])
    losses = total_trades - wins
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = wins / total_trades if total_trades > 0 else 0

    win_pnls = [t["pnl"] for t in trades if t["pnl"] > 0]
    loss_pnls = [abs(t["pnl"]) for t in trades if t["pnl"] <= 0]
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
    profit_factor = sum(win_pnls) / sum(loss_pnls) if loss_pnls and sum(loss_pnls) > 0 else float("inf")

    sizes = [t["size"] for t in trades]
    avg_bet_size = sum(sizes) / len(sizes) if sizes else 0

    max_dd_pct = max_drawdown / peak_bankroll * 100 if peak_bankroll > 0 else 0

    # Sharpe ratio
    pnls = [t["pnl"] for t in trades]
    if len(pnls) > 1:
        mean_pnl = sum(pnls) / len(pnls)
        var_pnl = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std_pnl = math.sqrt(var_pnl) if var_pnl > 0 else 1.0
        sharpe = (mean_pnl / std_pnl) * math.sqrt(len(pnls)) if std_pnl > 0 else 0.0
    else:
        sharpe = 0.0

    return BacktestResult(
        strategy_name=strategy.name,
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        total_pnl=total_pnl,
        final_bankroll=account.bankroll,
        initial_bankroll=initial_bankroll,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_dd_pct,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_bet_size=avg_bet_size,
        sharpe=sharpe,
        trades=trades,
    )


def _print_result(result: BacktestResult) -> None:
    """Pretty-print a backtest result."""
    roi = (result.final_bankroll - result.initial_bankroll) / result.initial_bankroll * 100

    print(f"\n    Strategy: {result.strategy_name}")
    print(f"    {'─' * 45}")
    print(f"    Trades: {result.total_trades}")
    print(f"    Wins: {result.wins}, Losses: {result.losses}")
    print(f"    Win rate: {result.win_rate:.1%}")
    print(f"    Total P&L: ${result.total_pnl:.2f}")
    print(f"    Bankroll: ${result.initial_bankroll:.2f} -> ${result.final_bankroll:.2f} ({roi:+.1f}%)")
    print(f"    Max drawdown: ${result.max_drawdown:.2f} ({result.max_drawdown_pct:.1f}%)")
    print(f"    Profit factor: {result.profit_factor:.2f}")
    print(f"    Avg win: ${result.avg_win:.3f}, Avg loss: ${result.avg_loss:.3f}")
    print(f"    Avg bet size: ${result.avg_bet_size:.2f}")
    print(f"    Sharpe: {result.sharpe:.2f}")

    # Go/no-go recommendation
    print(f"\n    RECOMMENDATION:")
    if result.total_trades < 50:
        print(f"    INCONCLUSIVE — only {result.total_trades} trades (need 50+)")
    elif result.win_rate > 0.55 and result.profit_factor > 1.2 and result.sharpe > 0.5:
        print(f"    GO — strong performance across all metrics")
    elif result.win_rate > 0.52 and result.total_pnl > 0:
        print(f"    CAUTIOUS GO — positive but marginal edge")
    elif result.total_pnl > 0:
        print(f"    MONITOR — profitable but weak metrics")
    else:
        print(f"    NO GO — negative P&L")


def cmd_run(args: argparse.Namespace) -> None:
    """Run a full backtest for a single strategy."""
    strategies = _discover_strategies()
    name = args.strategy

    if name not in strategies:
        available = ", ".join(sorted(strategies.keys()))
        logger.error(f"Unknown strategy: {name}")
        logger.error(f"Available: {available}")
        sys.exit(1)

    config = load_config()
    strategy = strategies[name]

    print(f"\n  STRATEGY PROTOTYPER — FULL BACKTEST")
    print(f"  {'=' * 50}")
    print(f"  Strategy: {name}")
    print(f"  Description: {strategy.description}")

    result = _run_backtest(strategy, config)
    _print_result(result)

    # Write trade log
    if result.trades:
        output_path = DATA_DIR / f"prototype_{name}_trades.csv"
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=result.trades[0].keys())
            writer.writeheader()
            writer.writerows(result.trades)
        print(f"\n    Trade log: {output_path}")
    print()


def cmd_compare(args: argparse.Namespace) -> None:
    """Side-by-side comparison of multiple strategies."""
    strategies = _discover_strategies()
    names = [s.strip() for s in args.strategies.split(",")]

    missing = [n for n in names if n not in strategies]
    if missing:
        available = ", ".join(sorted(strategies.keys()))
        logger.error(f"Unknown strategies: {', '.join(missing)}")
        logger.error(f"Available: {available}")
        sys.exit(1)

    config = load_config()

    print(f"\n  STRATEGY COMPARISON")
    print(f"  {'=' * 60}")

    results: list[BacktestResult] = []
    for name in names:
        logger.info(f"Running backtest for {name}...")
        result = _run_backtest(strategies[name], config)
        results.append(result)

    # Comparison table
    headers = ["Metric"] + names
    metrics = [
        ("Trades", lambda r: str(r.total_trades)),
        ("Win Rate", lambda r: f"{r.win_rate:.1%}"),
        ("Total P&L", lambda r: f"${r.total_pnl:.2f}"),
        ("Final Bankroll", lambda r: f"${r.final_bankroll:.2f}"),
        ("ROI", lambda r: f"{(r.final_bankroll - r.initial_bankroll) / r.initial_bankroll * 100:+.1f}%"),
        ("Max Drawdown", lambda r: f"${r.max_drawdown:.2f}"),
        ("Profit Factor", lambda r: f"{r.profit_factor:.2f}"),
        ("Avg Win", lambda r: f"${r.avg_win:.3f}"),
        ("Avg Loss", lambda r: f"${r.avg_loss:.3f}"),
        ("Avg Bet", lambda r: f"${r.avg_bet_size:.2f}"),
        ("Sharpe", lambda r: f"{r.sharpe:.2f}"),
    ]

    rows = []
    for label, fn in metrics:
        row = [label] + [fn(r) for r in results]
        rows.append(row)

    print()
    print(format_table(headers, rows))

    # Winner
    best = max(results, key=lambda r: r.sharpe)
    print(f"\n  Best strategy by Sharpe: {best.strategy_name} ({best.sharpe:.2f})")

    best_pnl = max(results, key=lambda r: r.total_pnl)
    print(f"  Best strategy by P&L: {best_pnl.strategy_name} (${best_pnl.total_pnl:.2f})")
    print()


def cmd_promote(args: argparse.Namespace) -> None:
    """Generate a ready-to-integrate strategy module in strategies/."""
    strategies = _discover_strategies()
    name = args.strategy

    if name not in strategies:
        available = ", ".join(sorted(strategies.keys()))
        logger.error(f"Unknown strategy: {name}")
        logger.error(f"Available: {available}")
        sys.exit(1)

    strategy = strategies[name]
    params = strategy.entry_params()

    output_path = Path(PROJECT_ROOT) / "strategies" / f"{name}.py"

    template = f'''"""Strategy: {name} — promoted from prototyper.

{strategy.description}

Auto-generated by: python -m agents.strategy_prototyper promote --strategy {name}
"""


class {name.replace("_", " ").title().replace(" ", "")}Strategy:
    """Promoted strategy: {strategy.description}"""

    name = "{name}"
    description = """{strategy.description}"""

    def should_enter(
        self,
        snap: dict,
        regime: str,
        secs_remaining: float,
    ) -> tuple[bool, str, float]:
        """Decide whether to enter a trade.

        Args:
            snap: Snapshot data with prices and orderbook depths.
            regime: Current market regime name.
            secs_remaining: Seconds remaining in the window.

        Returns:
            (should_enter, side, confidence)
        """
        # TODO: implement strategy logic
        raise NotImplementedError("Fill in strategy logic before deploying")

    def entry_params(self) -> dict:
        """Strategy parameters locked from backtest optimisation."""
        return {{
            "min_edge": {params.get("min_edge", 0.003)},
            "max_spread": {params.get("max_spread", 0.10)},
            "min_time_remaining": {params.get("min_time_remaining", 60)},
            "max_time_remaining": {params.get("max_time_remaining", 270)},
        }}
'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template)

    print(f"\n  Strategy module promoted to: {output_path}")
    print(f"  Edit the should_enter() method with your strategy logic,")
    print(f"  then run: python -m agents.strategy_prototyper run --strategy {name}")
    print()


def cmd_list(args: argparse.Namespace) -> None:
    """List all available strategies."""
    strategies = _discover_strategies()

    print(f"\n  AVAILABLE STRATEGIES")
    print(f"  {'=' * 50}")

    headers = ["Name", "Description", "Source"]
    rows = []
    for name in sorted(strategies):
        strat = strategies[name]
        source = "built-in" if name in BUILTIN_STRATEGIES else "strategies/"
        rows.append([name, strat.description, source])

    print(format_table(headers, rows))
    print()


def main() -> None:
    """CLI entry point for the Strategy Prototyper Agent."""
    parser = argparse.ArgumentParser(
        prog="python -m agents.strategy_prototyper",
        description="Strategy Prototyper — end-to-end strategy backtesting and promotion",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = subparsers.add_parser("run", help="Run full backtest for a strategy")
    run_parser.add_argument("--strategy", required=True, help="Strategy name")

    # compare
    cmp_parser = subparsers.add_parser("compare", help="Side-by-side strategy comparison")
    cmp_parser.add_argument("--strategies", required=True, help="Comma-separated strategy names")

    # promote
    promo_parser = subparsers.add_parser("promote", help="Generate deployable strategy module")
    promo_parser.add_argument("--strategy", required=True, help="Strategy name to promote")

    # list
    subparsers.add_parser("list", help="List all available strategies")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "run": cmd_run,
        "compare": cmd_compare,
        "promote": cmd_promote,
        "list": cmd_list,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
