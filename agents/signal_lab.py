"""Signal Lab Agent — rapid prototyping and evaluation of new signal hypotheses.

Tests signal ideas against historical data before they enter the live signal engine.
Signal hypotheses live in signals/hypotheses/ and implement the SignalHypothesis protocol.

Commands:
    python -m agents.signal_lab test --hypothesis <name>          — Test a single hypothesis
    python -m agents.signal_lab scan                              — Rank all hypotheses by predictive power
    python -m agents.signal_lab combine --signals L1,L2,new_sig   — Test signal combinations
"""

import argparse
import csv
import importlib
import inspect
import logging
import math
import sys
from collections import defaultdict
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
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agents.signal_lab")


class SignalHypothesis(Protocol):
    """Protocol that all signal hypotheses must implement."""
    name: str

    def compute(self, row: dict) -> float:
        """Return signal value in [-1, 1]. 0.0 = no signal."""
        ...

    def required_fields(self) -> list[str]:
        """CSV columns this signal needs."""
        ...


# ─── Built-in signal wrappers (reuse existing signal logic) ─────────

class L1OracleLagWrapper:
    """Wraps the backtest's OracleLagSignal for use in the signal lab."""
    name = "L1_oracle_lag"

    def __init__(self) -> None:
        from backtest.backtest_real_pricing import OracleLagSignal
        self._signal = OracleLagSignal()

    def compute(self, row: dict) -> float:
        btc_price = float(row.get("btc_price", 0))
        # In backtest, oracle = btc_start (approximation)
        btc_start = float(row.get("btc_start", btc_price))
        if btc_price <= 0 or btc_start <= 0:
            return 0.0
        return self._signal.compute(btc_price, btc_start, btc_start)

    def required_fields(self) -> list[str]:
        return ["btc_price"]


class L2MomentumWrapper:
    """Wraps the backtest's MomentumSignal — requires candle data."""
    name = "L2_momentum"

    def compute(self, row: dict) -> float:
        # Momentum requires candle history, so we use the pre-computed value if available
        return float(row.get("momentum_signal", 0))

    def required_fields(self) -> list[str]:
        return ["momentum_signal"]


class L4OrderbookWrapper:
    """Wraps orderbook imbalance from snapshot data."""
    name = "L4_orderbook"

    def compute(self, row: dict) -> float:
        up_bid = float(row.get("up_bid_depth_5", 0))
        up_ask = float(row.get("up_ask_depth_5", 0))
        down_bid = float(row.get("down_bid_depth_5", 0))
        down_ask = float(row.get("down_ask_depth_5", 0))

        total_bid = up_bid + down_bid
        total_ask = up_ask + down_ask
        total = total_bid + total_ask
        if total <= 0:
            return 0.0

        thickness = (total_bid - total_ask) / total
        return max(-1.0, min(1.0, thickness * 3.0))

    def required_fields(self) -> list[str]:
        return ["up_bid_depth_5", "up_ask_depth_5", "down_bid_depth_5", "down_ask_depth_5"]


BUILTIN_SIGNALS: dict[str, type] = {
    "L1_oracle_lag": L1OracleLagWrapper,
    "L2_momentum": L2MomentumWrapper,
    "L4_orderbook": L4OrderbookWrapper,
}


def _discover_hypotheses() -> dict[str, SignalHypothesis]:
    """Discover all signal hypotheses from signals/hypotheses/ directory."""
    hypotheses: dict[str, SignalHypothesis] = {}

    # Built-in wrappers
    for name, cls in BUILTIN_SIGNALS.items():
        hypotheses[name] = cls()

    # User-defined hypotheses from signals/hypotheses/
    hyp_dir = Path(PROJECT_ROOT) / "signals" / "hypotheses"
    if not hyp_dir.exists():
        return hypotheses

    for py_file in hyp_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        try:
            module_name = f"signals.hypotheses.{py_file.stem}"
            module = importlib.import_module(module_name)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if hasattr(obj, "name") and hasattr(obj, "compute") and hasattr(obj, "required_fields"):
                    if obj.__module__ == module_name:
                        instance = obj()
                        hypotheses[instance.name] = instance
                        logger.info(f"Discovered hypothesis: {instance.name}")
        except Exception as e:
            logger.warning(f"Failed to load hypothesis from {py_file.name}: {e}")

    return hypotheses


def _load_enriched_data() -> list[dict]:
    """Load snapshot data enriched with market outcomes for signal testing."""
    snaps_path = DATA_DIR / "polybacktest_snapshots.csv"
    markets_path = DATA_DIR / "polybacktest_markets.csv"

    if not snaps_path.exists() or not markets_path.exists():
        logger.error("Required data files not found.")
        logger.error("Run: python -m agents.data_pipeline fetch --source polybacktest")
        sys.exit(1)

    # Load and merge
    snaps = load_backtest_data(snaps_path)
    markets = load_backtest_data(markets_path)

    # Build market_id -> winner map
    winner_map = {}
    btc_start_map = {}
    for _, row in markets.iterrows():
        mid = row.get("market_id", "")
        winner_map[mid] = row.get("winner", "")
        btc_start_map[mid] = float(row.get("btc_start", 0)) if "btc_start" in row else 0

    # Enrich snapshots
    rows = []
    for _, snap in snaps.iterrows():
        d = snap.to_dict()
        mid = d.get("market_id", "")
        d["winner"] = winner_map.get(mid, "")
        d["btc_start"] = btc_start_map.get(mid, d.get("btc_price", 0))
        d["outcome_up"] = 1 if d["winner"].lower() in ["up", "yes"] else 0
        rows.append(d)

    logger.info(f"Enriched {len(rows):,} snapshots with outcomes")
    return rows


def _evaluate_signal(
    hypothesis: SignalHypothesis,
    data: list[dict],
) -> dict:
    """Evaluate a signal hypothesis against historical data.

    Returns metrics: accuracy, sharpe, mean_signal, correlation with outcome.
    """
    predictions: list[float] = []
    outcomes: list[int] = []
    signals: list[float] = []

    for row in data:
        # Check required fields
        required = hypothesis.required_fields()
        if any(row.get(f) is None or row.get(f) == "" for f in required):
            continue

        try:
            signal = hypothesis.compute(row)
        except Exception:
            continue

        if signal == 0.0:
            continue  # No signal = skip

        outcome = int(row.get("outcome_up", 0))
        predicted_up = 1 if signal > 0 else 0

        signals.append(signal)
        predictions.append(predicted_up)
        outcomes.append(outcome)

    if len(signals) < 10:
        return {
            "name": hypothesis.name,
            "samples": len(signals),
            "accuracy": 0.0,
            "mean_signal": 0.0,
            "signal_std": 0.0,
            "sharpe": 0.0,
            "correlation": 0.0,
        }

    # Directional accuracy
    correct = sum(1 for p, o in zip(predictions, outcomes) if p == o)
    accuracy = correct / len(predictions)

    # Mean and std of signal values
    mean_sig = sum(signals) / len(signals)
    variance = sum((s - mean_sig) ** 2 for s in signals) / len(signals)
    std_sig = math.sqrt(variance) if variance > 0 else 0.0

    # Sharpe-like ratio: mean accuracy excess / std
    # (treat each prediction as +1 if correct, -1 if wrong)
    pnls = [1.0 if p == o else -1.0 for p, o in zip(predictions, outcomes)]
    mean_pnl = sum(pnls) / len(pnls)
    pnl_var = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
    pnl_std = math.sqrt(pnl_var) if pnl_var > 0 else 1.0
    sharpe = mean_pnl / pnl_std if pnl_std > 0 else 0.0

    # Correlation between signal and outcome
    mean_out = sum(outcomes) / len(outcomes)
    cov = sum((s - mean_sig) * (o - mean_out) for s, o in zip(signals, outcomes)) / len(signals)
    out_var = sum((o - mean_out) ** 2 for o in outcomes) / len(outcomes)
    out_std = math.sqrt(out_var) if out_var > 0 else 1.0
    correlation = cov / (std_sig * out_std) if std_sig > 0 and out_std > 0 else 0.0

    return {
        "name": hypothesis.name,
        "samples": len(signals),
        "accuracy": accuracy,
        "mean_signal": mean_sig,
        "signal_std": std_sig,
        "sharpe": sharpe,
        "correlation": correlation,
    }


def cmd_test(args: argparse.Namespace) -> None:
    """Test a single signal hypothesis against historical data."""
    hypotheses = _discover_hypotheses()

    name = args.hypothesis
    if name not in hypotheses:
        available = ", ".join(sorted(hypotheses.keys()))
        logger.error(f"Unknown hypothesis: {name}")
        logger.error(f"Available: {available}")
        sys.exit(1)

    hypothesis = hypotheses[name]
    data = _load_enriched_data()

    print(f"\n  SIGNAL HYPOTHESIS TEST: {name}")
    print(f"  {'=' * 50}")
    print(f"  Required fields: {hypothesis.required_fields()}")

    result = _evaluate_signal(hypothesis, data)

    print(f"\n  Results:")
    print(f"    Samples with signal: {result['samples']:,}")
    print(f"    Directional accuracy: {result['accuracy']:.1%}")
    print(f"    Mean signal: {result['mean_signal']:.4f}")
    print(f"    Signal std: {result['signal_std']:.4f}")
    print(f"    Sharpe ratio: {result['sharpe']:.3f}")
    print(f"    Outcome correlation: {result['correlation']:.4f}")

    # Interpretation
    print(f"\n  Interpretation:")
    if result["samples"] < 30:
        print(f"    WARNING: Only {result['samples']} samples — results not statistically significant")
    if result["accuracy"] > 0.55:
        print(f"    Signal shows predictive value (accuracy > 55%)")
    elif result["accuracy"] > 0.50:
        print(f"    Signal shows marginal edge (50-55%)")
    else:
        print(f"    Signal does not appear predictive")

    if abs(result["correlation"]) > 0.1:
        direction = "bullish bias" if result["correlation"] > 0 else "bearish bias"
        print(f"    Notable {direction} (correlation: {result['correlation']:.3f})")
    print()


def cmd_scan(args: argparse.Namespace) -> None:
    """Run all hypotheses and rank by standalone predictive power."""
    hypotheses = _discover_hypotheses()
    data = _load_enriched_data()

    print(f"\n  SIGNAL HYPOTHESIS SCAN")
    print(f"  {'=' * 60}")
    print(f"  Testing {len(hypotheses)} hypotheses against {len(data):,} snapshots")

    results = []
    for name, hypothesis in sorted(hypotheses.items()):
        result = _evaluate_signal(hypothesis, data)
        results.append(result)
        logger.info(f"  {name}: accuracy={result['accuracy']:.1%}, sharpe={result['sharpe']:.3f}")

    # Sort by Sharpe ratio (best risk-adjusted prediction)
    results.sort(key=lambda r: r["sharpe"], reverse=True)

    print()
    headers = ["Rank", "Signal", "Samples", "Accuracy", "Sharpe", "Correlation"]
    rows = []
    for i, r in enumerate(results, 1):
        rows.append([
            str(i),
            r["name"],
            f"{r['samples']:,}",
            f"{r['accuracy']:.1%}",
            f"{r['sharpe']:.3f}",
            f"{r['correlation']:.4f}",
        ])
    print(format_table(headers, rows))

    # Highlight best
    if results and results[0]["sharpe"] > 0:
        print(f"\n  Best signal: {results[0]['name']} (Sharpe: {results[0]['sharpe']:.3f})")
    print()


def cmd_combine(args: argparse.Namespace) -> None:
    """Test combining multiple signals with weight optimisation."""
    hypotheses = _discover_hypotheses()
    signal_names = [s.strip() for s in args.signals.split(",")]

    # Validate signal names
    missing = [s for s in signal_names if s not in hypotheses]
    if missing:
        available = ", ".join(sorted(hypotheses.keys()))
        logger.error(f"Unknown signals: {', '.join(missing)}")
        logger.error(f"Available: {available}")
        sys.exit(1)

    data = _load_enriched_data()

    print(f"\n  SIGNAL COMBINATION TEST")
    print(f"  {'=' * 50}")
    print(f"  Signals: {', '.join(signal_names)}")

    # Compute all signals for each data point
    signal_values: dict[str, list[float]] = {name: [] for name in signal_names}
    outcomes: list[int] = []
    valid_indices: list[int] = []

    for i, row in enumerate(data):
        all_valid = True
        values = {}
        for name in signal_names:
            hyp = hypotheses[name]
            required = hyp.required_fields()
            if any(row.get(f) is None or row.get(f) == "" for f in required):
                all_valid = False
                break
            try:
                val = hyp.compute(row)
                values[name] = val
            except Exception:
                all_valid = False
                break

        if all_valid and any(v != 0.0 for v in values.values()):
            for name in signal_names:
                signal_values[name].append(values[name])
            outcomes.append(int(row.get("outcome_up", 0)))
            valid_indices.append(i)

    n = len(outcomes)
    print(f"  Valid samples: {n:,}")

    if n < 30:
        print(f"  WARNING: Too few samples for meaningful combination analysis")
        print()
        return

    # Equal-weight combination
    print(f"\n  EQUAL-WEIGHT COMBINATION:")
    equal_weight = 1.0 / len(signal_names)
    correct = 0
    for j in range(n):
        combined = sum(signal_values[name][j] * equal_weight for name in signal_names)
        predicted_up = 1 if combined > 0 else 0
        if predicted_up == outcomes[j]:
            correct += 1
    eq_accuracy = correct / n
    print(f"    Accuracy: {eq_accuracy:.1%}")

    # Grid search over weight combinations (simple for 2-3 signals)
    if len(signal_names) <= 3:
        print(f"\n  WEIGHT OPTIMISATION (grid search):")
        best_accuracy = 0.0
        best_weights: list[float] = []

        # Generate weight combinations in steps of 0.1
        step = 0.1
        weight_range = [round(i * step, 1) for i in range(0, 11)]

        from itertools import product

        if len(signal_names) == 2:
            for w1 in weight_range:
                w2 = round(1.0 - w1, 1)
                if w2 < 0:
                    continue
                weights = [w1, w2]
                correct = 0
                for j in range(n):
                    combined = sum(
                        signal_values[signal_names[k]][j] * weights[k]
                        for k in range(len(signal_names))
                    )
                    if (1 if combined > 0 else 0) == outcomes[j]:
                        correct += 1
                acc = correct / n
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_weights = weights

        elif len(signal_names) == 3:
            for w1 in weight_range:
                for w2 in weight_range:
                    w3 = round(1.0 - w1 - w2, 1)
                    if w3 < 0 or w3 > 1.0:
                        continue
                    weights = [w1, w2, w3]
                    correct = 0
                    for j in range(n):
                        combined = sum(
                            signal_values[signal_names[k]][j] * weights[k]
                            for k in range(len(signal_names))
                        )
                        if (1 if combined > 0 else 0) == outcomes[j]:
                            correct += 1
                    acc = correct / n
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_weights = weights

        if best_weights:
            print(f"    Best accuracy: {best_accuracy:.1%}")
            for name, w in zip(signal_names, best_weights):
                print(f"      {name}: {w:.1f}")
            print(f"    Improvement over equal weight: {best_accuracy - eq_accuracy:+.1%}")
    else:
        print(f"\n  (Grid search skipped for {len(signal_names)} signals — use fewer for optimisation)")

    # Correlation matrix between signals
    print(f"\n  SIGNAL CORRELATIONS:")
    for i, name_a in enumerate(signal_names):
        for j, name_b in enumerate(signal_names):
            if j <= i:
                continue
            vals_a = signal_values[name_a]
            vals_b = signal_values[name_b]
            mean_a = sum(vals_a) / n
            mean_b = sum(vals_b) / n
            cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(vals_a, vals_b)) / n
            std_a = math.sqrt(sum((a - mean_a) ** 2 for a in vals_a) / n)
            std_b = math.sqrt(sum((b - mean_b) ** 2 for b in vals_b) / n)
            corr = cov / (std_a * std_b) if std_a > 0 and std_b > 0 else 0.0
            quality = "LOW" if abs(corr) < 0.3 else "MODERATE" if abs(corr) < 0.6 else "HIGH"
            print(f"    {name_a} vs {name_b}: {corr:.3f} ({quality} correlation)")

    print()


def main() -> None:
    """CLI entry point for the Signal Lab Agent."""
    parser = argparse.ArgumentParser(
        prog="python -m agents.signal_lab",
        description="Signal Lab — rapid prototyping and evaluation of signal hypotheses",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # test
    test_parser = subparsers.add_parser("test", help="Test a single signal hypothesis")
    test_parser.add_argument("--hypothesis", required=True, help="Hypothesis name to test")

    # scan
    subparsers.add_parser("scan", help="Rank all hypotheses by predictive power")

    # combine
    combine_parser = subparsers.add_parser("combine", help="Test signal combinations")
    combine_parser.add_argument(
        "--signals",
        required=True,
        help="Comma-separated signal names (e.g. L1_oracle_lag,L2_momentum)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "test": cmd_test,
        "scan": cmd_scan,
        "combine": cmd_combine,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
