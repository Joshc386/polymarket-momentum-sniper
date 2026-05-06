"""Walk-forward parameter optimiser for multi-strategy backtest.

Finds optimal parameters for each strategy using walk-forward validation:
  1. Split data into rolling train/test windows (14d train, 7d test)
  2. Grid search over key parameters on each train window
  3. Score by Sharpe ratio on the train period
  4. Evaluate best params on the test period
  5. Report aggregate out-of-sample performance

Prevents overfitting: parameters are ALWAYS tested on data the
optimiser hasn't seen. If a strategy only works with specific
parameters and fails out-of-sample, it's noise, not edge.

Usage:
    python -m backtest.optimiser
    python -m backtest.optimiser --strategies a,b --train-days 14 --test-days 7
"""

import argparse
import copy
import csv
import itertools
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

from typing import Any

from backtest.backtest_multi import (
    DEFAULT_CONFIGS,
    STRATEGY_MAP,
    StrategyStats,
    run_backtest,
)


# --- Parameter Search Spaces ----------------------------------------

# Each strategy defines which parameters to tune and their candidate values.
# Format: list of (param_path, values) where param_path is a dot-separated
# key into the config dict (e.g. "entry.min_edge").

SEARCH_SPACES: dict[str, list[tuple[str, list]]] = {
    "a": [
        ("entry.min_edge",        [0.005, 0.01, 0.015, 0.02]),
        ("entry.max_edge",        [0.04, 0.06, 0.08]),
        ("entry.min_confidence",  [0.01, 0.02, 0.03]),
        ("sizing.kelly_multiplier", [0.15, 0.25, 0.35]),
        ("signals.max_adjustment", [0.15, 0.20, 0.25]),
    ],
    "b": [
        ("entry.min_edge",        [0.01, 0.02, 0.03]),
        ("entry.min_confidence",  [0.02, 0.03, 0.05]),
        ("signals.kalman_filter.process_noise_scale", [5e-6, 1e-5, 5e-5]),
        ("signals.kalman_filter.measurement_noise_oracle", [2e-4, 5e-4, 1e-3]),
        ("sizing.kelly_multiplier", [0.15, 0.25, 0.35]),
    ],
    "c": [
        ("entry.min_edge",        [0.005, 0.01, 0.02]),
        ("entry.min_confidence",  [0.01, 0.02, 0.03]),
        ("entry.min_regime_confidence", [0.30, 0.40, 0.50, 0.60]),
        ("signals.hmm_regime.transition_stickiness", [0.85, 0.90, 0.95]),
        ("sizing.kelly_multiplier", [0.15, 0.25, 0.35]),
    ],
    "d": [
        ("entry.min_edge",        [0.005, 0.01, 0.015]),
        ("entry.min_confidence",  [0.01, 0.02, 0.03]),
        ("signals.cross_exchange.z_threshold", [1.0, 1.5, 2.0]),
        ("signals.max_adjustment", [0.15, 0.20, 0.25]),
        ("sizing.kelly_multiplier", [0.15, 0.25, 0.35]),
    ],
    "e": [
        ("entry.min_edge",        [0.005, 0.01, 0.02]),
        ("signals.ou_spread.entry_z_threshold", [1.0, 1.5, 2.0]),
        ("entry.min_ou_z_score",  [1.0, 1.5, 2.0]),
        ("signals.ou_spread.calibration_window", [200, 300, 500]),
        ("sizing.kelly_multiplier", [0.15, 0.25, 0.35]),
    ],
}


# --- Config Manipulation --------------------------------------------

def set_nested(d: dict, path: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated path."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def get_nested(d: dict, path: str, default=None):
    """Get a value from a nested dict using dot-separated path."""
    keys = path.split(".")
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d



# --- Walk-Forward Fold Generation ------------------------------------

@dataclass
class WalkForwardFold:
    """One train/test split."""
    fold_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def generate_folds(
    markets_path: str,
    train_days: int = 7,
    test_days: int = 5,
    warmup_days: int = 14,
) -> list[WalkForwardFold]:
    """Generate walk-forward folds from market date range.

    Signal calibration (OU/KF) needs ~14 days of warmup before
    producing useful signals. The warmup period is handled by
    run_backtest's warmup_days parameter. Folds start after the
    warmup period to ensure all strategies can trade.

    Walks forward by test_days each fold:
      Fold 1: train=[warmup+0, warmup+7), test=[warmup+7, warmup+12)
      Fold 2: train=[warmup+5, warmup+12), test=[warmup+12, warmup+17)
      ...
    """
    # Find date range
    earliest = latest = None
    with open(markets_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["start_time", "end_time"]:
                t = row.get(key, "")
                if t:
                    try:
                        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
                        if earliest is None or dt < earliest:
                            earliest = dt
                        if latest is None or dt > latest:
                            latest = dt
                    except ValueError:
                        pass

    if not earliest or not latest:
        return []

    # Skip warmup period — signals aren't reliable before this
    scoring_start = earliest + timedelta(days=warmup_days)

    folds = []
    fold_num = 1
    current_start = scoring_start

    while True:
        train_end = current_start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)

        if test_end > latest:
            break

        folds.append(WalkForwardFold(
            fold_num=fold_num,
            train_start=current_start.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
        ))

        current_start += timedelta(days=test_days)
        fold_num += 1

    return folds


# --- Scoring ---------------------------------------------------------

@dataclass
class ParamScore:
    """Score for one parameter combination on one fold."""
    params: dict = field(default_factory=dict)
    train_sharpe: float = 0.0
    train_pf: float = 0.0
    train_wr: float = 0.0
    train_pnl: float = 0.0
    train_trades: int = 0
    test_sharpe: float = 0.0
    test_pf: float = 0.0
    test_wr: float = 0.0
    test_pnl: float = 0.0
    test_trades: int = 0


def score_stats(stats: StrategyStats) -> dict:
    """Extract key metrics from a StrategyStats."""
    return {
        "sharpe": stats.sharpe_ratio,
        "pf": stats.profit_factor,
        "wr": stats.win_rate,
        "pnl": stats.total_pnl,
        "trades": stats.count,
        "max_dd": stats.max_drawdown,
    }


# --- Main Optimiser -------------------------------------------------

def optimise_strategy(
    strategy_key: str,
    folds: list[WalkForwardFold],
    base_config: dict | None = None,
) -> dict:
    """Run walk-forward optimisation for one strategy.

    Returns dict with:
        best_params: The parameter combination with best aggregate OOS performance
        fold_results: Per-fold train/test scores
        all_combos: All parameter combinations tested (for analysis)
    """
    if strategy_key not in SEARCH_SPACES:
        logger.error(f"No search space defined for strategy '{strategy_key}'")
        return {}

    search_space = SEARCH_SPACES[strategy_key]
    base_cfg = base_config or DEFAULT_CONFIGS.get(strategy_key, {})

    # Generate all parameter combinations
    param_names = [name for name, _ in search_space]
    param_values = [values for _, values in search_space]
    combinations = list(itertools.product(*param_values))

    logger.info(
        f"Strategy {strategy_key.upper()}: {len(combinations)} param combinations "
        f"× {len(folds)} folds = {len(combinations) * len(folds)} backtest runs"
    )

    # Track scores per combination across folds
    combo_oos_scores: dict[int, list[float]] = defaultdict(list)  # combo_idx -> [oos_sharpe per fold]
    combo_oos_pnl: dict[int, list[float]] = defaultdict(list)
    combo_oos_trades: dict[int, list[int]] = defaultdict(list)
    fold_results: list[dict] = []

    for fold in folds:
        logger.info(
            f"\n  Fold {fold.fold_num}: train=[{fold.train_start}, {fold.train_end}) "
            f"test=[{fold.test_start}, {fold.test_end})"
        )

        best_train_sharpe = -999.0
        best_combo_idx = 0

        # Phase 1: Grid search on train period
        for combo_idx, values in enumerate(combinations):
            if combo_idx % 50 == 0:
                logger.info(
                    f"    Testing combo {combo_idx+1}/{len(combinations)} "
                    f"(best train Sharpe so far: {best_train_sharpe:+.2f})"
                )

            # Build config with this parameter combination
            cfg = copy.deepcopy(base_cfg)
            for (param_name, _), value in zip(search_space, values):
                set_nested(cfg, param_name, value)

            # Run backtest on train period (with warmup for signal calibration)
            try:
                results = run_backtest(
                    strategy_keys=[strategy_key],
                    configs={strategy_key: cfg},
                    start_date=fold.train_start,
                    end_date=fold.train_end,
                    warmup_days=14,
                )
            except Exception as e:
                logger.debug(f"  Combo {combo_idx} failed: {e}")
                continue

            stats = results.get(strategy_key)
            if not stats or stats.count < 5:
                continue

            train_score = score_stats(stats)

            if train_score["sharpe"] > best_train_sharpe:
                best_train_sharpe = train_score["sharpe"]
                best_combo_idx = combo_idx

        # Phase 2: Evaluate best params on test period
        best_values = combinations[best_combo_idx]
        best_cfg = copy.deepcopy(base_cfg)
        best_params_dict = {}
        for (param_name, _), value in zip(search_space, best_values):
            set_nested(best_cfg, param_name, value)
            best_params_dict[param_name] = value

        try:
            test_results = run_backtest(
                strategy_keys=[strategy_key],
                configs={strategy_key: best_cfg},
                start_date=fold.test_start,
                end_date=fold.test_end,
                warmup_days=14,
            )
        except Exception as e:
            logger.error(f"  Test run failed: {e}")
            continue

        test_stats = test_results.get(strategy_key)
        test_score = score_stats(test_stats) if test_stats and test_stats.count > 0 else {
            "sharpe": 0, "pf": 0, "wr": 0, "pnl": 0, "trades": 0, "max_dd": 0,
        }

        combo_oos_scores[best_combo_idx].append(test_score["sharpe"])
        combo_oos_pnl[best_combo_idx].append(test_score["pnl"])
        combo_oos_trades[best_combo_idx].append(test_score["trades"])

        fold_result = {
            "fold": fold.fold_num,
            "train_period": f"{fold.train_start} -> {fold.train_end}",
            "test_period": f"{fold.test_start} -> {fold.test_end}",
            "best_params": best_params_dict,
            "train_sharpe": best_train_sharpe,
            "test_sharpe": test_score["sharpe"],
            "test_pf": test_score["pf"],
            "test_wr": test_score["wr"],
            "test_pnl": test_score["pnl"],
            "test_trades": test_score["trades"],
        }
        fold_results.append(fold_result)

        logger.info(
            f"  Fold {fold.fold_num} result: "
            f"train Sharpe={best_train_sharpe:+.2f}, "
            f"test Sharpe={test_score['sharpe']:+.2f}, "
            f"test PnL=${test_score['pnl']:+.2f} ({test_score['trades']} trades)"
        )
        logger.info(f"  Best params: {best_params_dict}")

    # Aggregate: find the combo that appeared most often as "best" and had
    # the best average OOS performance
    if not fold_results:
        logger.warning(f"No fold results for strategy {strategy_key}")
        return {}

    # Find most commonly selected parameter combo
    from collections import Counter
    param_selections = Counter()
    for fr in fold_results:
        key = tuple(sorted(fr["best_params"].items()))
        param_selections[key] += 1

    most_common_params = dict(param_selections.most_common(1)[0][0])

    # Aggregate OOS metrics
    avg_oos_sharpe = sum(fr["test_sharpe"] for fr in fold_results) / len(fold_results)
    avg_oos_pnl = sum(fr["test_pnl"] for fr in fold_results) / len(fold_results)
    total_oos_trades = sum(fr["test_trades"] for fr in fold_results)
    positive_folds = sum(1 for fr in fold_results if fr["test_pnl"] > 0)

    return {
        "strategy": strategy_key,
        "best_params": most_common_params,
        "avg_oos_sharpe": avg_oos_sharpe,
        "avg_oos_pnl": avg_oos_pnl,
        "total_oos_trades": total_oos_trades,
        "positive_folds": f"{positive_folds}/{len(fold_results)}",
        "fold_results": fold_results,
        "param_stability": param_selections.most_common(3),
        "n_combos_tested": len(combinations),
    }


# --- Reporting -------------------------------------------------------

def print_optimisation_report(results: dict) -> None:
    """Print a comprehensive optimisation report."""
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD OPTIMISATION: Strategy {results['strategy'].upper()}")
    print(f"{'='*70}")
    print(f"  Combinations tested: {results['n_combos_tested']}")
    print(f"  Folds: {len(results['fold_results'])}")
    print(f"\n  Aggregate Out-of-Sample Performance:")
    print(f"    Avg Sharpe:     {results['avg_oos_sharpe']:>+8.2f}")
    print(f"    Avg PnL/fold:   ${results['avg_oos_pnl']:>+8.2f}")
    print(f"    Total trades:   {results['total_oos_trades']:>8}")
    print(f"    Positive folds: {results['positive_folds']}")

    print(f"\n  Recommended Parameters:")
    for param, value in results["best_params"].items():
        current = get_nested(
            DEFAULT_CONFIGS.get(results["strategy"], {}), param, "?"
        )
        changed = " <-- CHANGED" if value != current else ""
        print(f"    {param:<50} {value}{changed}")

    print(f"\n  Parameter Stability (top 3 most-selected combos):")
    for params_tuple, count in results["param_stability"]:
        params_dict = dict(params_tuple)
        print(f"    Selected {count}x: {params_dict}")

    print(f"\n  Per-Fold Results:")
    print(f"    {'Fold':>4}  {'Train Sharpe':>13}  {'Test Sharpe':>12}  {'Test PnL':>10}  {'Trades':>7}")
    print(f"    {'-'*4}  {'-'*13}  {'-'*12}  {'-'*10}  {'-'*7}")
    for fr in results["fold_results"]:
        print(
            f"    {fr['fold']:>4}  {fr['train_sharpe']:>+13.2f}  "
            f"{fr['test_sharpe']:>+12.2f}  ${fr['test_pnl']:>+9.2f}  "
            f"{fr['test_trades']:>7}"
        )

    # Overfitting detection
    train_sharpes = [fr["train_sharpe"] for fr in results["fold_results"]]
    test_sharpes = [fr["test_sharpe"] for fr in results["fold_results"]]
    if train_sharpes and test_sharpes:
        avg_train = sum(train_sharpes) / len(train_sharpes)
        avg_test = sum(test_sharpes) / len(test_sharpes)
        decay = avg_train - avg_test if avg_train != 0 else 0
        print(f"\n  Overfitting Check:")
        print(f"    Avg train Sharpe: {avg_train:+.2f}")
        print(f"    Avg test Sharpe:  {avg_test:+.2f}")
        print(f"    Sharpe decay:     {decay:+.2f}", end="")
        if decay > 1.0:
            print("  [!]  HIGH — likely overfitting")
        elif decay > 0.5:
            print("  [!]  moderate decay")
        else:
            print("  [OK] stable")


# --- Main ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward parameter optimiser")
    parser.add_argument(
        "--strategies", "-s", default="a,b,c,d,e",
        help="Comma-separated strategy letters (default: a,b,c,d,e)"
    )
    parser.add_argument("--train-days", type=int, default=7, help="Train window size in days")
    parser.add_argument("--test-days", type=int, default=5, help="Test window size in days")
    args = parser.parse_args()

    strategy_keys = [s.strip().lower() for s in args.strategies.split(",")]

    # Generate folds
    markets_path = os.path.join(DATA_DIR, "polybacktest_markets.csv")
    if not os.path.exists(markets_path):
        logger.error(f"Markets file not found: {markets_path}")
        return

    folds = generate_folds(markets_path, args.train_days, args.test_days)
    if not folds:
        logger.error("Could not generate walk-forward folds — insufficient data range")
        return

    logger.info(f"Generated {len(folds)} walk-forward folds ({args.train_days}d train / {args.test_days}d test)")
    for fold in folds:
        logger.info(f"  Fold {fold.fold_num}: train [{fold.train_start}, {fold.train_end}) -> test [{fold.test_start}, {fold.test_end})")

    # Optimise each strategy
    all_results = {}
    for key in strategy_keys:
        if key not in STRATEGY_MAP:
            logger.warning(f"Unknown strategy '{key}', skipping")
            continue
        logger.info(f"\n{'='*50}")
        logger.info(f"  Optimising Strategy {key.upper()}")
        logger.info(f"{'='*50}")
        result = optimise_strategy(key, folds)
        if result:
            all_results[key] = result
            print_optimisation_report(result)

    # Final comparison
    if len(all_results) > 1:
        print(f"\n{'#'*70}")
        print(f"  CROSS-STRATEGY COMPARISON (Out-of-Sample)")
        print(f"{'#'*70}")
        print(f"  {'Strategy':<22} {'OOS Sharpe':>11} {'OOS PnL':>10} {'Trades':>8} {'+ Folds':>8}")
        print(f"  {'-'*22} {'-'*11} {'-'*10} {'-'*8} {'-'*8}")
        for key in sorted(all_results.keys()):
            r = all_results[key]
            print(
                f"  {key.upper():<22} {r['avg_oos_sharpe']:>+11.2f} "
                f"${r['avg_oos_pnl']:>+9.2f} {r['total_oos_trades']:>8} "
                f"{r['positive_folds']:>8}"
            )

    # Save results
    output_path = os.path.join(DATA_DIR, "optimiser_results.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy", "param_name", "current_value", "optimised_value",
            "avg_oos_sharpe", "avg_oos_pnl", "positive_folds",
        ])
        for key, result in all_results.items():
            for param, value in result["best_params"].items():
                current = get_nested(DEFAULT_CONFIGS.get(key, {}), param, "")
                writer.writerow([
                    key.upper(), param, current, value,
                    f"{result['avg_oos_sharpe']:.4f}",
                    f"{result['avg_oos_pnl']:.4f}",
                    result["positive_folds"],
                ])
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
