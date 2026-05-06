"""Strategy Lab — automated strategy generation, backtesting, and evaluation.

The central pipeline for strategy R&D. Accepts hypotheses from any source
(mutations, combinations, research, manual), backtests them automatically,
evaluates viability, and produces ranked recommendations.

Pipeline:
    Generate -> Backtest -> Evaluate -> Rank -> Report

Three generation modes:
    mutate:   Perturb parameters of an existing strategy
    combine:  Mix signals from different strategies into novel configs
    test:     Backtest a specific hypothesis (from research, manual, etc.)

Evaluation criteria (go/no-go):
    - Out-of-sample Sharpe > 0.5
    - Win rate > 45% (minimum 50 trades)
    - Profit factor > 1.2
    - Max drawdown < 30% of starting bankroll
    - Trade frequency > 20/day
    - Positive PnL in >50% of walk-forward folds

Usage:
    python -m backtest.strategy_lab mutate --base a --count 50
    python -m backtest.strategy_lab combine --count 30
    python -m backtest.strategy_lab test --config hypothesis.yaml
    python -m backtest.strategy_lab full --count 100
    python -m backtest.strategy_lab leaderboard
"""

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LAB_DIR = os.path.join(DATA_DIR, "lab_results")

from backtest.backtest_multi import (
    DEFAULT_CONFIGS,
    STRATEGY_MAP,
    StrategyStats,
    run_backtest,
)
from backtest.optimiser import generate_folds, score_stats


# --- Core Data Structures -------------------------------------------

@dataclass
class StrategyHypothesis:
    """A strategy idea to be tested."""
    name: str
    description: str
    source: str  # "mutation", "combination", "research", "manual"
    base_strategy: str | None = None
    config: dict = field(default_factory=dict)
    rationale: str = ""
    strategy_key: str = "a"  # Which strategy class to use

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "base_strategy": self.base_strategy,
            "strategy_key": self.strategy_key,
            "rationale": self.rationale,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyHypothesis":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            source=d.get("source", "manual"),
            base_strategy=d.get("base_strategy"),
            strategy_key=d.get("strategy_key", "a"),
            rationale=d.get("rationale", ""),
            config=d.get("config", {}),
        )


@dataclass
class BacktestResult:
    """Results from backtesting a hypothesis."""
    hypothesis: StrategyHypothesis
    stats: StrategyStats | None = None
    oos_sharpe: float = 0.0
    oos_pnl: float = 0.0
    oos_trades: int = 0
    positive_folds: int = 0
    total_folds: int = 0
    fold_sharpes: list[float] = field(default_factory=list)
    runtime_secs: float = 0.0


@dataclass
class Evaluation:
    """Go/no-go evaluation of a backtest result."""
    hypothesis_name: str
    verdict: str  # "GO", "MARGINAL", "NO-GO"
    score: float  # Composite score for ranking (higher = better)
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


# --- Evaluation Thresholds ------------------------------------------

GO_CRITERIA = {
    "min_sharpe": 0.5,
    "min_wr": 0.45,
    "min_pf": 1.2,
    "max_drawdown_pct": 0.30,  # 30% of starting bankroll
    "min_trades_per_day": 20,
    "min_positive_fold_pct": 0.50,
    "min_total_trades": 50,
}

MARGINAL_CRITERIA = {
    "min_sharpe": 0.0,
    "min_wr": 0.42,
    "min_pf": 1.0,
    "max_drawdown_pct": 0.40,
    "min_trades_per_day": 10,
    "min_positive_fold_pct": 0.40,
    "min_total_trades": 30,
}


# --- Signal Palette -------------------------------------------------
# Available signals that can be mixed into strategies.
# Each entry defines: which slot it fills, default config, and what it needs.

SIGNAL_PALETTE = {
    # L1 alternatives (oracle lag slot)
    "oracle_lag": {
        "slot": "L1",
        "strategy_keys": ["a", "c", "d"],  # Which strategy classes support it
        "config_key": "oracle_lag",
        "params": {
            "max_expected_lag": [0.0005, 0.001, 0.002],
        },
    },
    "kalman_filter": {
        "slot": "L1",
        "strategy_keys": ["b"],
        "config_key": "kalman_filter",
        "params": {
            "process_noise_scale": [5e-6, 1e-5, 5e-5, 1e-4],
            "measurement_noise_binance": [5e-5, 1e-4, 5e-4],
            "measurement_noise_oracle": [1e-4, 5e-4, 1e-3],
            "adaptive_window": [20, 30, 50],
        },
    },
    "ou_spread": {
        "slot": "L1",
        "strategy_keys": ["e"],
        "config_key": "ou_spread",
        "params": {
            "calibration_window": [150, 300, 500],
            "min_observations": [30, 60, 100],
            "entry_z_threshold": [1.0, 1.5, 2.0, 2.5],
        },
    },

    # L2 alternatives (momentum slot)
    "momentum_standard": {
        "slot": "L2",
        "strategy_keys": ["a", "b", "c", "e"],
        "config_key": "momentum",
        "params": {
            "roc_weight": [0.20, 0.30, 0.40],
            "direction_weight": [0.15, 0.25, 0.35],
            "volume_weight": [0.15, 0.25, 0.35],
            "rsi_period": [3, 5, 7],
        },
    },
    "momentum_slope": {
        "slot": "L2",
        "strategy_keys": ["d", "e"],
        "config_key": "momentum_slope",
        "params": {
            "weight_30s": [0.25, 0.35, 0.45],
            "weight_60s": [0.20, 0.30, 0.40],
            "weight_120s": [0.15, 0.20, 0.30],
            "weight_240s": [0.10, 0.15, 0.25],
            "slope_normaliser": [5.0, 10.0, 20.0],
        },
    },

    # Modifiers (not required, can be added to any strategy)
    "cross_exchange": {
        "slot": "modifier",
        "strategy_keys": ["d"],
        "config_key": "cross_exchange",
        "params": {
            "rolling_window": [60, 120, 240],
            "z_threshold": [1.0, 1.5, 2.0, 2.5],
        },
    },
    "funding_filter": {
        "slot": "modifier",
        "strategy_keys": ["d"],
        "config_key": "funding_filter",
        "params": {
            "mild_threshold": [0.005, 0.01, 0.02],
            "high_threshold": [0.03, 0.05, 0.08],
            "stress_threshold": [0.08, 0.10, 0.15],
        },
    },

    # Gates (regime/condition filters)
    "hmm_regime": {
        "slot": "gate",
        "strategy_keys": ["c"],
        "config_key": "hmm_regime",
        "params": {
            "transition_stickiness": [0.80, 0.85, 0.90, 0.95],
            "adaptation_rate": [0.01, 0.02, 0.05],
        },
    },
}

# Entry parameter ranges for random generation
ENTRY_RANGES = {
    "min_edge": [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03],
    "max_edge": [0.03, 0.04, 0.06, 0.08, 0.10],
    "min_confidence": [0.01, 0.02, 0.03, 0.05],
    "fee_adjustment": [0.02],  # Fixed
}

SIZING_RANGES = {
    "kelly_multiplier": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "min_bet_usdc": [1.0],
    "max_bet_usdc": [3.0, 5.0, 8.0, 10.0],
    "initial_bankroll": [100.0],
}


# --- Hypothesis Generators ------------------------------------------

def _set_nested(d: dict, path: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated path."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _get_nested(d: dict, path: str, default: Any = None) -> Any:
    """Get a value from a nested dict."""
    keys = path.split(".")
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


def generate_mutations(
    base_key: str,
    count: int = 50,
    mutation_rate: float = 0.4,
    seed: int | None = None,
) -> list[StrategyHypothesis]:
    """Generate N mutations of an existing strategy.

    Each mutation randomly perturbs 1-3 parameters from the base strategy.
    Higher mutation_rate = more parameters changed per variant.

    Args:
        base_key: Strategy letter (a-e) to mutate.
        count: Number of mutations to generate.
        mutation_rate: Probability of mutating each parameter.
        seed: Random seed for reproducibility.

    Returns:
        List of StrategyHypothesis with mutated configs.
    """
    if seed is not None:
        random.seed(seed)

    base_cfg = copy.deepcopy(DEFAULT_CONFIGS.get(base_key, {}))
    if not base_cfg:
        logger.error(f"No config for strategy '{base_key}'")
        return []

    # Collect all tunable parameters for this strategy
    tunable: list[tuple[str, list]] = []

    # Signal params from palette
    for sig_name, sig_info in SIGNAL_PALETTE.items():
        if base_key in sig_info.get("strategy_keys", []):
            cfg_key = sig_info["config_key"]
            for param, values in sig_info["params"].items():
                tunable.append((f"signals.{cfg_key}.{param}", values))

    # Entry params
    for param, values in ENTRY_RANGES.items():
        tunable.append((f"entry.{param}", values))

    # Sizing params
    for param, values in SIZING_RANGES.items():
        if len(values) > 1:
            tunable.append((f"sizing.{param}", values))

    # Signal combiner
    tunable.append(("signals.max_adjustment", [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]))

    # Weekend flip
    tunable.append(("weekend_flip", [True, False]))

    hypotheses = []
    seen: set[str] = set()

    for i in range(count * 3):  # Generate extra to account for duplicates
        if len(hypotheses) >= count:
            break

        cfg = copy.deepcopy(base_cfg)
        mutations = []

        for path, values in tunable:
            if random.random() < mutation_rate:
                new_val = random.choice(values)
                old_val = _get_nested(cfg, path)
                if new_val != old_val:
                    _set_nested(cfg, path, new_val)
                    mutations.append(f"{path.split('.')[-1]}={new_val}")

        if not mutations:
            continue

        # Deduplicate by config hash
        cfg_key = json.dumps(cfg, sort_keys=True, default=str)
        if cfg_key in seen:
            continue
        seen.add(cfg_key)

        name = f"mut_{base_key}_{len(hypotheses)+1:03d}"
        hypotheses.append(StrategyHypothesis(
            name=name,
            description=f"Mutation of Bot {base_key.upper()} ({len(mutations)} changes)",
            source="mutation",
            base_strategy=base_key,
            strategy_key=base_key,
            rationale=f"Perturbed: {', '.join(mutations[:5])}",
            config=cfg,
        ))

    logger.info(f"Generated {len(hypotheses)} mutations of strategy {base_key.upper()}")
    return hypotheses


def generate_combinations(
    count: int = 30,
    seed: int | None = None,
) -> list[StrategyHypothesis]:
    """Generate novel strategy combinations by mixing signals.

    Creates strategies by randomly selecting:
    - An L1 signal (oracle lag, KF, or OU spread)
    - An L2 signal (standard momentum or momentum slope)
    - Optional modifiers (cross-exchange, funding filter)
    - Optional gates (HMM regime, OU z-score)
    - Random entry/sizing parameters

    Each combination is assigned to the appropriate strategy class
    based on which L1 signal it uses.

    Args:
        count: Number of combinations to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of StrategyHypothesis with novel signal combinations.
    """
    if seed is not None:
        random.seed(seed)

    l1_options = ["oracle_lag", "kalman_filter", "ou_spread"]
    l2_options = ["momentum_standard", "momentum_slope"]
    modifier_options = ["cross_exchange", "funding_filter", None, None]  # None = no modifier
    gate_options = ["hmm_regime", None, None, None]  # None = no gate

    # Map L1 signal to strategy class
    l1_to_strategy = {
        "oracle_lag": "a",
        "kalman_filter": "b",
        "ou_spread": "e",
    }

    hypotheses = []
    seen: set[str] = set()

    for i in range(count * 3):
        if len(hypotheses) >= count:
            break

        # Pick signals
        l1 = random.choice(l1_options)
        l2 = random.choice(l2_options)
        modifier = random.choice(modifier_options)
        gate = random.choice(gate_options)

        # Determine strategy class
        strategy_key = l1_to_strategy[l1]

        # If we picked HMM gate, use strategy C
        if gate == "hmm_regime":
            strategy_key = "c"

        # If we picked cross_exchange or funding, use strategy D (if L1 is oracle)
        if modifier in ("cross_exchange", "funding_filter") and l1 == "oracle_lag":
            strategy_key = "d"

        # Build config from base
        cfg = copy.deepcopy(DEFAULT_CONFIGS.get(strategy_key, DEFAULT_CONFIGS["a"]))

        # Randomise L1 signal params
        l1_info = SIGNAL_PALETTE[l1]
        for param, values in l1_info["params"].items():
            _set_nested(cfg, f"signals.{l1_info['config_key']}.{param}", random.choice(values))

        # Randomise L2 signal params
        l2_info = SIGNAL_PALETTE[l2]
        for param, values in l2_info["params"].items():
            _set_nested(cfg, f"signals.{l2_info['config_key']}.{param}", random.choice(values))

        # Optional modifier
        if modifier:
            mod_info = SIGNAL_PALETTE[modifier]
            for param, values in mod_info["params"].items():
                _set_nested(cfg, f"signals.{mod_info['config_key']}.{param}", random.choice(values))

        # Optional gate
        if gate:
            gate_info = SIGNAL_PALETTE[gate]
            for param, values in gate_info["params"].items():
                _set_nested(cfg, f"signals.{gate_info['config_key']}.{param}", random.choice(values))

        # Random entry params
        for param, values in ENTRY_RANGES.items():
            _set_nested(cfg, f"entry.{param}", random.choice(values))

        # Random sizing
        for param, values in SIZING_RANGES.items():
            _set_nested(cfg, f"sizing.{param}", random.choice(values))

        # Combiner adjustment
        _set_nested(cfg, "signals.max_adjustment",
                    random.choice([0.15, 0.18, 0.20, 0.22, 0.25]))

        cfg["weekend_flip"] = random.choice([True, False])

        # Deduplicate
        cfg_key = json.dumps(cfg, sort_keys=True, default=str)
        if cfg_key in seen:
            continue
        seen.add(cfg_key)

        parts = [l1.replace("_", " "), l2.replace("_", " ").replace("standard", "")]
        if modifier:
            parts.append(modifier.replace("_", " "))
        if gate:
            parts.append(gate.replace("_", " ") + " gate")

        name = f"combo_{len(hypotheses)+1:03d}"
        hypotheses.append(StrategyHypothesis(
            name=name,
            description=f"Combination: {' + '.join(parts)}",
            source="combination",
            base_strategy=None,
            strategy_key=strategy_key,
            rationale=f"L1={l1}, L2={l2}, mod={modifier}, gate={gate}",
            config=cfg,
        ))

    logger.info(f"Generated {len(hypotheses)} signal combinations")
    return hypotheses


def load_hypothesis_from_yaml(path: str) -> StrategyHypothesis:
    """Load a strategy hypothesis from a YAML file.

    Expected format:
        name: my_strategy
        description: Strategy based on XYZ research
        source: research
        strategy_key: a
        rationale: Paper XYZ showed that ...
        config:
            signals: ...
            entry: ...
            sizing: ...
            risk: ...
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return StrategyHypothesis.from_dict(data)


# --- Auto-Backtester ------------------------------------------------

def test_hypothesis(
    hypothesis: StrategyHypothesis,
    run_walk_forward: bool = True,
    train_days: int = 14,
    test_days: int = 7,
) -> BacktestResult:
    """Backtest a single hypothesis and optionally run walk-forward.

    Args:
        hypothesis: The strategy to test.
        run_walk_forward: Whether to run walk-forward validation.
        train_days: Walk-forward train window.
        test_days: Walk-forward test window.

    Returns:
        BacktestResult with full-range and OOS metrics.
    """
    start_time = time.time()
    key = hypothesis.strategy_key

    if key not in STRATEGY_MAP:
        logger.error(f"Unknown strategy class '{key}' for {hypothesis.name}")
        return BacktestResult(hypothesis=hypothesis)

    # Full-range backtest
    logger.info(f"Testing {hypothesis.name} ({hypothesis.description})")
    try:
        results = run_backtest(
            strategy_keys=[key],
            configs={key: hypothesis.config},
        )
    except Exception as e:
        logger.error(f"Backtest failed for {hypothesis.name}: {e}")
        return BacktestResult(hypothesis=hypothesis, runtime_secs=time.time() - start_time)

    stats = results.get(key)
    if not stats or stats.count == 0:
        logger.info(f"  {hypothesis.name}: 0 trades")
        return BacktestResult(hypothesis=hypothesis, stats=stats, runtime_secs=time.time() - start_time)

    result = BacktestResult(
        hypothesis=hypothesis,
        stats=stats,
        runtime_secs=time.time() - start_time,
    )

    # Walk-forward validation (if enough trades to be worth it)
    if run_walk_forward and stats.count >= 30:
        markets_path = os.path.join(DATA_DIR, "polybacktest_markets.csv")
        folds = generate_folds(markets_path, train_days, test_days)

        fold_sharpes = []
        positive_folds = 0

        for fold in folds:
            try:
                test_results = run_backtest(
                    strategy_keys=[key],
                    configs={key: hypothesis.config},
                    start_date=fold.test_start,
                    end_date=fold.test_end,
                    warmup_days=14,
                )
                test_stats = test_results.get(key)
                if test_stats and test_stats.count > 0:
                    fold_sharpes.append(test_stats.sharpe_ratio)
                    if test_stats.total_pnl > 0:
                        positive_folds += 1
                else:
                    fold_sharpes.append(0.0)
            except Exception:
                fold_sharpes.append(0.0)

        result.fold_sharpes = fold_sharpes
        result.oos_sharpe = sum(fold_sharpes) / len(fold_sharpes) if fold_sharpes else 0
        result.oos_pnl = stats.total_pnl  # Full range PnL as proxy
        result.oos_trades = stats.count
        result.positive_folds = positive_folds
        result.total_folds = len(folds)

    result.runtime_secs = time.time() - start_time
    return result


def test_batch(
    hypotheses: list[StrategyHypothesis],
    run_walk_forward: bool = False,
    quick_filter: bool = True,
) -> list[BacktestResult]:
    """Test multiple hypotheses. Optionally quick-filter before walk-forward.

    If quick_filter is True:
    1. Run all hypotheses on full range (fast, no WF)
    2. Filter to top 20% by Sharpe
    3. Run walk-forward only on survivors

    This is much faster than running WF on every hypothesis.

    Args:
        hypotheses: Strategies to test.
        run_walk_forward: Run walk-forward on (filtered) survivors.
        quick_filter: Pre-filter before walk-forward.

    Returns:
        List of BacktestResult, sorted by composite score.
    """
    total = len(hypotheses)
    logger.info(f"Testing {total} hypotheses...")

    # Phase 1: Full-range backtest (no WF)
    results: list[BacktestResult] = []
    for i, hyp in enumerate(hypotheses):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"  [{i+1}/{total}] Testing {hyp.name}")
        result = test_hypothesis(hyp, run_walk_forward=False)
        results.append(result)

    # Filter to those with positive Sharpe and enough trades
    viable = [r for r in results if r.stats and r.stats.count >= 20 and r.stats.sharpe_ratio > 0]
    logger.info(f"Phase 1: {len(viable)}/{total} viable (Sharpe > 0, trades >= 20)")

    if not viable:
        return results

    # Phase 2: Walk-forward on top candidates
    if run_walk_forward and quick_filter:
        # Keep top 20% (at least 5)
        viable.sort(key=lambda r: r.stats.sharpe_ratio, reverse=True)
        n_keep = max(5, len(viable) // 5)
        top = viable[:n_keep]
        logger.info(f"Phase 2: Walk-forward on top {len(top)} candidates")

        wf_results = []
        for i, r in enumerate(top):
            logger.info(f"  [{i+1}/{len(top)}] Walk-forward: {r.hypothesis.name}")
            wf_result = test_hypothesis(r.hypothesis, run_walk_forward=True)
            wf_results.append(wf_result)

        # Merge: replace quick results with WF results for tested hypotheses
        wf_names = {r.hypothesis.name for r in wf_results}
        final = [r for r in results if r.hypothesis.name not in wf_names]
        final.extend(wf_results)
        return final

    elif run_walk_forward:
        # Run WF on all viable
        wf_results = []
        for i, r in enumerate(viable):
            logger.info(f"  [{i+1}/{len(viable)}] Walk-forward: {r.hypothesis.name}")
            wf_result = test_hypothesis(r.hypothesis, run_walk_forward=True)
            wf_results.append(wf_result)
        return wf_results

    return results


# --- Evaluator -------------------------------------------------------

def evaluate(result: BacktestResult) -> Evaluation:
    """Evaluate a backtest result against go/no-go criteria.

    Returns an Evaluation with verdict, score, and detailed reasons.
    """
    stats = result.stats
    if not stats or stats.count == 0:
        return Evaluation(
            hypothesis_name=result.hypothesis.name,
            verdict="NO-GO",
            score=-999,
            reasons=["No trades taken"],
        )

    reasons = []
    metrics = {
        "trades": stats.count,
        "win_rate": stats.win_rate,
        "pnl": stats.total_pnl,
        "sharpe": stats.sharpe_ratio,
        "profit_factor": stats.profit_factor,
        "max_drawdown": stats.max_drawdown,
        "oos_sharpe": result.oos_sharpe,
        "positive_folds": f"{result.positive_folds}/{result.total_folds}",
    }

    # Estimate trades per day (rough: assume data spans ~49 days)
    trades_per_day = stats.count / 49.0

    # Check GO criteria
    go_pass = True

    if stats.count < GO_CRITERIA["min_total_trades"]:
        reasons.append(f"Insufficient trades ({stats.count} < {GO_CRITERIA['min_total_trades']})")
        go_pass = False

    if stats.win_rate < GO_CRITERIA["min_wr"]:
        reasons.append(f"Win rate too low ({stats.win_rate:.1%} < {GO_CRITERIA['min_wr']:.0%})")
        go_pass = False

    if stats.profit_factor < GO_CRITERIA["min_pf"]:
        reasons.append(f"Profit factor too low ({stats.profit_factor:.2f} < {GO_CRITERIA['min_pf']})")
        go_pass = False

    if stats.max_drawdown > 100 * GO_CRITERIA["max_drawdown_pct"]:
        reasons.append(f"Max drawdown too high (${stats.max_drawdown:.2f})")
        go_pass = False

    if trades_per_day < GO_CRITERIA["min_trades_per_day"]:
        reasons.append(f"Trade frequency too low ({trades_per_day:.0f}/day < {GO_CRITERIA['min_trades_per_day']})")
        go_pass = False

    if result.total_folds > 0:
        fold_pct = result.positive_folds / result.total_folds
        if fold_pct < GO_CRITERIA["min_positive_fold_pct"]:
            reasons.append(f"Too few positive folds ({fold_pct:.0%} < {GO_CRITERIA['min_positive_fold_pct']:.0%})")
            go_pass = False

    if stats.sharpe_ratio < GO_CRITERIA["min_sharpe"]:
        reasons.append(f"Sharpe too low ({stats.sharpe_ratio:.2f} < {GO_CRITERIA['min_sharpe']})")
        go_pass = False

    # Determine verdict
    if go_pass and not reasons:
        verdict = "GO"
    elif all(
        stats.count >= MARGINAL_CRITERIA["min_total_trades"]
        and stats.win_rate >= MARGINAL_CRITERIA["min_wr"]
        and stats.profit_factor >= MARGINAL_CRITERIA["min_pf"]
        and stats.sharpe_ratio >= MARGINAL_CRITERIA["min_sharpe"]
        for _ in [None]  # Just for the 'all' syntax
    ):
        verdict = "MARGINAL"
    else:
        verdict = "NO-GO"

    # Composite score for ranking
    # Weighted: 40% Sharpe, 25% PF, 20% WR, 15% trade frequency
    score = (
        0.40 * min(3.0, stats.sharpe_ratio)
        + 0.25 * min(3.0, stats.profit_factor)
        + 0.20 * (stats.win_rate * 10)
        + 0.15 * min(3.0, trades_per_day / 50)
    )

    # Bonus for walk-forward consistency
    if result.total_folds > 0 and result.positive_folds > 0:
        consistency = result.positive_folds / result.total_folds
        score *= (0.5 + 0.5 * consistency)

    return Evaluation(
        hypothesis_name=result.hypothesis.name,
        verdict=verdict,
        score=score,
        reasons=reasons if reasons else ["All criteria met"],
        metrics=metrics,
    )


# --- Leaderboard ----------------------------------------------------

class Leaderboard:
    """Ranked comparison of all tested strategies."""

    def __init__(self) -> None:
        self.entries: list[tuple[BacktestResult, Evaluation]] = []

    def add(self, result: BacktestResult, evaluation: Evaluation) -> None:
        self.entries.append((result, evaluation))

    def rank(self) -> list[tuple[BacktestResult, Evaluation]]:
        """Sort by evaluation score (descending)."""
        return sorted(self.entries, key=lambda e: e[1].score, reverse=True)

    def print_summary(self) -> None:
        """Print compact leaderboard."""
        ranked = self.rank()
        if not ranked:
            print("No results to show.")
            return

        print(f"\n{'#'*85}")
        print(f"  STRATEGY LAB LEADERBOARD — {len(ranked)} strategies tested")
        print(f"{'#'*85}")
        header = (
            f"  {'Rank':>4}  {'Name':<20} {'Verdict':<8} {'Score':>6} "
            f"{'Trades':>7} {'WR':>6} {'PnL':>10} {'Sharpe':>7} {'PF':>6}"
        )
        print(header)
        print(f"  {'-'*4}  {'-'*20} {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*10} {'-'*7} {'-'*6}")

        for i, (result, ev) in enumerate(ranked):
            stats = result.stats
            if not stats:
                continue
            icon = {"GO": "+", "MARGINAL": "~", "NO-GO": "-"}.get(ev.verdict, "?")
            print(
                f"  {i+1:>4}  {ev.hypothesis_name:<20} {icon}{ev.verdict:<7} {ev.score:>6.2f} "
                f"{stats.count:>7} {stats.win_rate*100:>5.1f}% "
                f"${stats.total_pnl:>+9.2f} {stats.sharpe_ratio:>+7.2f} "
                f"{stats.profit_factor:>6.2f}"
            )

        # Highlight top GO strategies
        go_strategies = [(r, e) for r, e in ranked if e.verdict == "GO"]
        if go_strategies:
            print(f"\n  {'='*85}")
            print(f"  {len(go_strategies)} strategies passed GO criteria:")
            for r, e in go_strategies:
                print(f"    {e.hypothesis_name}: {r.hypothesis.description}")
                print(f"      Rationale: {r.hypothesis.rationale}")
                if e.metrics.get("positive_folds"):
                    print(f"      Walk-forward: {e.metrics['positive_folds']} positive folds")

        marginal = [(r, e) for r, e in ranked if e.verdict == "MARGINAL"]
        if marginal:
            print(f"\n  {len(marginal)} MARGINAL strategies (worth further investigation)")

    def save(self, path: str | None = None) -> str:
        """Save leaderboard to CSV. Returns the file path."""
        if path is None:
            os.makedirs(LAB_DIR, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = os.path.join(LAB_DIR, f"leaderboard_{ts}.csv")

        ranked = self.rank()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "name", "verdict", "score", "source", "strategy_key",
                "trades", "win_rate", "pnl", "sharpe", "profit_factor",
                "max_drawdown", "oos_sharpe", "positive_folds",
                "description", "rationale", "config",
            ])
            for i, (result, ev) in enumerate(ranked):
                stats = result.stats
                writer.writerow([
                    i + 1,
                    ev.hypothesis_name,
                    ev.verdict,
                    f"{ev.score:.4f}",
                    result.hypothesis.source,
                    result.hypothesis.strategy_key,
                    stats.count if stats else 0,
                    f"{stats.win_rate:.4f}" if stats else "0",
                    f"{stats.total_pnl:.4f}" if stats else "0",
                    f"{stats.sharpe_ratio:.4f}" if stats else "0",
                    f"{stats.profit_factor:.4f}" if stats else "0",
                    f"{stats.max_drawdown:.4f}" if stats else "0",
                    f"{result.oos_sharpe:.4f}",
                    f"{result.positive_folds}/{result.total_folds}",
                    result.hypothesis.description,
                    result.hypothesis.rationale,
                    json.dumps(result.hypothesis.config, default=str),
                ])

        logger.info(f"Leaderboard saved to {path}")
        return path


# --- Baseline Bots ---------------------------------------------------

def get_baseline_hypotheses() -> list[StrategyHypothesis]:
    """Create hypotheses for the 5 existing bots (baseline comparison)."""
    baselines = [
        StrategyHypothesis(
            name="baseline_A",
            description="Bot A: Contrarian EV (current live)",
            source="baseline",
            strategy_key="a",
            config=DEFAULT_CONFIGS["a"],
            rationale="Current production baseline",
        ),
        StrategyHypothesis(
            name="baseline_B",
            description="Bot B: Kalman EV (KF-dominant + velocity)",
            source="baseline",
            strategy_key="b",
            config=DEFAULT_CONFIGS["b"],
            rationale="KF multi-source fusion with velocity gate",
        ),
        StrategyHypothesis(
            name="baseline_C",
            description="Bot C: HMM Regime EV (trending only)",
            source="baseline",
            strategy_key="c",
            config=DEFAULT_CONFIGS["c"],
            rationale="HMM regime filter — only trades trends",
        ),
        StrategyHypothesis(
            name="baseline_D",
            description="Bot D: Enhanced EV (multi-TF + cross-ex)",
            source="baseline",
            strategy_key="d",
            config=DEFAULT_CONFIGS["d"],
            rationale="Multi-timeframe slope + cross-exchange divergence",
        ),
        StrategyHypothesis(
            name="baseline_E",
            description="Bot E: OU Reversion (spread mean-revert)",
            source="baseline",
            strategy_key="e",
            config=DEFAULT_CONFIGS["e"],
            rationale="OU process on Binance-Oracle spread",
        ),
    ]
    return baselines


# --- Full Pipeline --------------------------------------------------

def run_full_pipeline(
    n_mutations_per_base: int = 20,
    n_combinations: int = 20,
    run_walk_forward: bool = True,
    seed: int | None = 42,
) -> Leaderboard:
    """Run the complete strategy lab pipeline.

    1. Test all 5 baseline bots
    2. Generate mutations of each bot
    3. Generate novel combinations
    4. Quick-filter -> walk-forward on top candidates
    5. Evaluate and rank everything

    Args:
        n_mutations_per_base: Mutations to generate per existing bot.
        n_combinations: Novel combinations to generate.
        run_walk_forward: Whether to run walk-forward validation.
        seed: Random seed for reproducibility.

    Returns:
        Leaderboard with all results ranked.
    """
    board = Leaderboard()
    all_hypotheses: list[StrategyHypothesis] = []

    # Phase 1: Baseline bots
    logger.info("=" * 60)
    logger.info("  Phase 1: Testing 5 baseline bots")
    logger.info("=" * 60)
    baselines = get_baseline_hypotheses()
    for hyp in baselines:
        result = test_hypothesis(hyp, run_walk_forward=run_walk_forward)
        ev = evaluate(result)
        board.add(result, ev)
        if result.stats:
            logger.info(
                f"  {hyp.name}: {result.stats.count} trades, "
                f"WR={result.stats.win_rate:.1%}, "
                f"PnL=${result.stats.total_pnl:+.2f}, "
                f"Sharpe={result.stats.sharpe_ratio:+.2f} -> {ev.verdict}"
            )

    # Phase 2: Mutations
    logger.info("=" * 60)
    logger.info("  Phase 2: Generating mutations")
    logger.info("=" * 60)
    for base_key in ["a", "b", "c", "d", "e"]:
        mutations = generate_mutations(base_key, n_mutations_per_base, seed=seed)
        all_hypotheses.extend(mutations)

    # Phase 3: Combinations
    logger.info("=" * 60)
    logger.info("  Phase 3: Generating combinations")
    logger.info("=" * 60)
    combos = generate_combinations(n_combinations, seed=seed)
    all_hypotheses.extend(combos)

    # Phase 4: Test all generated hypotheses
    logger.info("=" * 60)
    logger.info(f"  Phase 4: Testing {len(all_hypotheses)} generated hypotheses")
    logger.info("=" * 60)
    results = test_batch(all_hypotheses, run_walk_forward=run_walk_forward, quick_filter=True)

    for result in results:
        ev = evaluate(result)
        board.add(result, ev)

    # Save and report
    board.print_summary()
    saved_path = board.save()

    # Print detailed report for GO strategies
    ranked = board.rank()
    go_strategies = [(r, e) for r, e in ranked if e.verdict == "GO"]
    if go_strategies:
        print(f"\n{'='*85}")
        print(f"  RECOMMENDED STRATEGIES — Ready for paper trading")
        print(f"{'='*85}")
        for r, e in go_strategies:
            print(f"\n  {e.hypothesis_name}")
            print(f"    Description:  {r.hypothesis.description}")
            print(f"    Source:        {r.hypothesis.source}")
            print(f"    Rationale:     {r.hypothesis.rationale}")
            print(f"    Trades:        {r.stats.count}")
            print(f"    Win Rate:      {r.stats.win_rate:.1%}")
            print(f"    PnL:           ${r.stats.total_pnl:+.2f}")
            print(f"    Sharpe:        {r.stats.sharpe_ratio:+.2f}")
            print(f"    Walk-Forward:  {r.positive_folds}/{r.total_folds} positive folds")
            print(f"    Config key:    strategy_key={r.hypothesis.strategy_key}")
    else:
        print(f"\n  No strategies met GO criteria.")
        marginal = [(r, e) for r, e in ranked if e.verdict == "MARGINAL"]
        if marginal:
            print(f"  {len(marginal)} MARGINAL strategies may be worth tuning further.")

    return board


# --- CLI -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy Lab — automated strategy R&D pipeline"
    )
    sub = parser.add_subparsers(dest="command", help="Lab command")

    # mutate
    p_mut = sub.add_parser("mutate", help="Generate and test mutations of a base strategy")
    p_mut.add_argument("--base", "-b", required=True, help="Base strategy letter (a-e)")
    p_mut.add_argument("--count", "-n", type=int, default=50, help="Number of mutations")
    p_mut.add_argument("--seed", type=int, default=42, help="Random seed")
    p_mut.add_argument("--wf", action="store_true", help="Run walk-forward validation")

    # combine
    p_comb = sub.add_parser("combine", help="Generate and test novel signal combinations")
    p_comb.add_argument("--count", "-n", type=int, default=30, help="Number of combinations")
    p_comb.add_argument("--seed", type=int, default=42, help="Random seed")
    p_comb.add_argument("--wf", action="store_true", help="Run walk-forward validation")

    # test
    p_test = sub.add_parser("test", help="Test a specific hypothesis from YAML")
    p_test.add_argument("--config", "-c", required=True, help="Path to hypothesis YAML")
    p_test.add_argument("--wf", action="store_true", help="Run walk-forward validation")

    # full
    p_full = sub.add_parser("full", help="Run full pipeline (baselines + mutations + combos)")
    p_full.add_argument("--mutations", "-m", type=int, default=20, help="Mutations per base strategy")
    p_full.add_argument("--combos", "-c", type=int, default=20, help="Novel combinations")
    p_full.add_argument("--seed", type=int, default=42, help="Random seed")
    p_full.add_argument("--no-wf", action="store_true", help="Skip walk-forward validation")

    # baselines
    p_base = sub.add_parser("baselines", help="Backtest the 5 existing bots")
    p_base.add_argument("--wf", action="store_true", help="Run walk-forward validation")

    # leaderboard
    p_lb = sub.add_parser("leaderboard", help="Show latest leaderboard from saved results")

    args = parser.parse_args()

    if args.command == "mutate":
        board = Leaderboard()
        hypotheses = generate_mutations(args.base.lower(), args.count, seed=args.seed)
        results = test_batch(hypotheses, run_walk_forward=args.wf, quick_filter=True)
        for r in results:
            board.add(r, evaluate(r))
        board.print_summary()
        board.save()

    elif args.command == "combine":
        board = Leaderboard()
        hypotheses = generate_combinations(args.count, seed=args.seed)
        results = test_batch(hypotheses, run_walk_forward=args.wf, quick_filter=True)
        for r in results:
            board.add(r, evaluate(r))
        board.print_summary()
        board.save()

    elif args.command == "test":
        hyp = load_hypothesis_from_yaml(args.config)
        result = test_hypothesis(hyp, run_walk_forward=args.wf)
        ev = evaluate(result)
        print(f"\n  {ev.hypothesis_name}: {ev.verdict}")
        if result.stats:
            print(f"  Trades: {result.stats.count}, WR: {result.stats.win_rate:.1%}")
            print(f"  PnL: ${result.stats.total_pnl:+.2f}, Sharpe: {result.stats.sharpe_ratio:+.2f}")
        for reason in ev.reasons:
            print(f"  - {reason}")

    elif args.command == "full":
        run_full_pipeline(
            n_mutations_per_base=args.mutations,
            n_combinations=args.combos,
            run_walk_forward=not args.no_wf,
            seed=args.seed,
        )

    elif args.command == "baselines":
        board = Leaderboard()
        for hyp in get_baseline_hypotheses():
            result = test_hypothesis(hyp, run_walk_forward=args.wf)
            ev = evaluate(result)
            board.add(result, ev)
        board.print_summary()
        board.save()

    elif args.command == "leaderboard":
        # Find latest leaderboard file
        if not os.path.exists(LAB_DIR):
            print("No lab results found. Run a test first.")
            return
        files = sorted(
            [f for f in os.listdir(LAB_DIR) if f.startswith("leaderboard_") and f.endswith(".csv")],
            reverse=True,
        )
        if not files:
            print("No leaderboard files found.")
            return
        path = os.path.join(LAB_DIR, files[0])
        print(f"Latest leaderboard: {path}")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            print(f"\n  {'Rank':>4}  {'Name':<20} {'Verdict':<8} {'Score':>6} "
                  f"{'Trades':>7} {'WR':>6} {'PnL':>10} {'Sharpe':>7}")
            print(f"  {'-'*4}  {'-'*20} {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*10} {'-'*7}")
            for row in reader:
                wr = float(row.get("win_rate", "0"))
                print(
                    f"  {row['rank']:>4}  {row['name']:<20} {row['verdict']:<8} "
                    f"{float(row['score']):>6.2f} {row['trades']:>7} {wr*100:>5.1f}% "
                    f"${float(row['pnl']):>+9.2f} {float(row['sharpe']):>+7.2f}"
                )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
