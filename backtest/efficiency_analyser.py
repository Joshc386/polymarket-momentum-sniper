#!/usr/bin/env python3
"""Efficiency Analyser — processes data collected by market_observer.py.

Reads the observation CSVs and computes:
1. Market efficiency score (how well mid-price predicts resolution)
2. Calibration curve (predicted vs actual UP probability)
3. Spread analysis over time
4. Oracle lag estimation
5. Recommended BacktestConfig parameters

Usage:
    python -m backtest.efficiency_analyser
    python -m backtest.efficiency_analyser --observations backtest/data/market_observations.csv
    python -m backtest.efficiency_analyser --raw backtest/data/market_samples_raw.csv
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ── Data Loading ──────────────────────────────────────────────────────

def load_observations(path: str) -> list[dict]:
    """Load the window-level summary CSV."""
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_raw_samples(path: str) -> list[dict]:
    """Load the per-sample raw CSV."""
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ── Core Metrics ──────────────────────────────────────────────────────

def compute_market_efficiency(observations: list[dict]) -> dict:
    """Compute the market efficiency score.

    Market efficiency = how well the market mid-price predicts the actual
    resolution. A perfectly efficient market would have:
    - When mid > 0.50: resolves UP proportional to the mid price
    - When mid < 0.50: resolves DOWN proportional to (1 - mid)

    Returns dict with efficiency scores at different time buckets.
    """
    results = {}

    # Analyse at each time bucket
    time_cols = {
        "4min": "yes_mid_at_4m",
        "3min": "yes_mid_at_3m",
        "2min": "yes_mid_at_2m",
        "1min": "yes_mid_at_1m",
        "30sec": "yes_mid_at_30s",
        "avg": "avg_yes_mid",
    }

    for label, col in time_cols.items():
        # Bucket observations by mid-price range
        buckets = defaultdict(lambda: {"total": 0, "up": 0})

        for obs in observations:
            mid_str = obs.get(col, "")
            if not mid_str:
                continue
            mid = float(mid_str)
            resolution = obs["resolution"]

            # Round mid to nearest 0.05 for bucketing
            bucket_key = round(mid * 20) / 20  # 0.05 increments
            buckets[bucket_key]["total"] += 1
            if resolution == "UP":
                buckets[bucket_key]["up"] += 1

        if not buckets:
            continue

        # Compute calibration error: |predicted_prob - actual_prob|
        total_cal_error = 0.0
        total_weight = 0
        bucket_details = []

        for mid_price in sorted(buckets.keys()):
            b = buckets[mid_price]
            if b["total"] < 3:  # Need minimum samples
                continue
            actual_up_rate = b["up"] / b["total"]
            cal_error = abs(mid_price - actual_up_rate)
            total_cal_error += cal_error * b["total"]
            total_weight += b["total"]
            bucket_details.append({
                "mid_price": mid_price,
                "n": b["total"],
                "actual_up_rate": actual_up_rate,
                "cal_error": cal_error,
            })

        avg_cal_error = total_cal_error / total_weight if total_weight > 0 else 0.5

        # Efficiency score: 1.0 = perfectly calibrated, 0.0 = no predictive power
        # Random market has ~0.25 average cal error for uniform mids
        # Perfect market has 0.0 cal error
        efficiency = 1.0 - min(avg_cal_error / 0.25, 1.0)

        results[label] = {
            "efficiency": efficiency,
            "avg_cal_error": avg_cal_error,
            "n_windows": total_weight,
            "buckets": bucket_details,
        }

    return results


def compute_directional_accuracy(observations: list[dict]) -> dict:
    """Simpler metric: when mid > 0.50, how often does it resolve UP?

    This directly measures whether the market "knows" the direction.
    """
    time_cols = {
        "4min": "yes_mid_at_4m",
        "3min": "yes_mid_at_3m",
        "2min": "yes_mid_at_2m",
        "1min": "yes_mid_at_1m",
        "30sec": "yes_mid_at_30s",
        "avg": "avg_yes_mid",
    }

    results = {}

    for label, col in time_cols.items():
        correct = 0
        total = 0
        strong_correct = 0  # mid > 0.55 or < 0.45
        strong_total = 0

        for obs in observations:
            mid_str = obs.get(col, "")
            if not mid_str:
                continue
            mid = float(mid_str)
            resolution = obs["resolution"]

            # Skip ambiguous (very close to 0.50)
            if abs(mid - 0.50) < 0.005:
                continue

            total += 1
            predicted_up = mid > 0.50
            actual_up = resolution == "UP"

            if predicted_up == actual_up:
                correct += 1

            # Strong signals
            if abs(mid - 0.50) > 0.05:
                strong_total += 1
                if predicted_up == actual_up:
                    strong_correct += 1

        results[label] = {
            "accuracy": correct / total if total > 0 else 0.5,
            "n": total,
            "strong_accuracy": strong_correct / strong_total if strong_total > 0 else 0.5,
            "strong_n": strong_total,
        }

    return results


def compute_spread_analysis(observations: list[dict]) -> dict:
    """Analyse spread patterns over time within windows."""
    time_cols = {
        "4min": "spread_at_4m",
        "3min": "spread_at_3m",
        "2min": "spread_at_2m",
        "1min": "spread_at_1m",
        "30sec": "spread_at_30s",
        "avg": "avg_spread",
    }

    results = {}
    for label, col in time_cols.items():
        spreads = []
        for obs in observations:
            s = obs.get(col, "")
            if s:
                spreads.append(float(s))
        if spreads:
            results[label] = {
                "mean": sum(spreads) / len(spreads),
                "min": min(spreads),
                "max": max(spreads),
                "median": sorted(spreads)[len(spreads) // 2],
                "n": len(spreads),
            }
    return results


def compute_resolution_stats(observations: list[dict]) -> dict:
    """Basic resolution statistics."""
    total = len(observations)
    up = sum(1 for o in observations if o["resolution"] == "UP")
    down = total - up

    return {
        "total_windows": total,
        "up": up,
        "down": down,
        "up_pct": up / total if total > 0 else 0.5,
    }


def estimate_oracle_lag(raw_samples: list[dict]) -> dict:
    """Estimate oracle lag from raw sample data.

    Looks at how oracle price compares to BTC price over time.
    If oracle consistently trails BTC, that's exploitable lag.
    """
    if not raw_samples:
        return {}

    # Group samples by window slug
    by_window = defaultdict(list)
    for s in raw_samples:
        by_window[s["slug"]].append(s)

    lag_estimates = []
    price_deviations = []

    for slug, samples in by_window.items():
        for s in samples:
            try:
                btc = float(s["btc_price"])
                oracle = float(s["oracle_price"])
                if btc > 0 and oracle > 0:
                    deviation_bps = abs(btc - oracle) / btc * 10000
                    price_deviations.append(deviation_bps)
            except (ValueError, KeyError):
                continue

    if not price_deviations:
        return {}

    price_deviations.sort()
    n = len(price_deviations)

    return {
        "mean_deviation_bps": sum(price_deviations) / n,
        "median_deviation_bps": price_deviations[n // 2],
        "p75_deviation_bps": price_deviations[int(n * 0.75)],
        "p90_deviation_bps": price_deviations[int(n * 0.90)],
        "p95_deviation_bps": price_deviations[int(n * 0.95)],
        "max_deviation_bps": price_deviations[-1],
        "n_samples": n,
    }


def compute_mid_price_distribution(raw_samples: list[dict]) -> dict:
    """Analyse the distribution of YES mid prices.

    A market stuck at 0.50 suggests low efficiency / no information.
    Wide distribution suggests the market IS incorporating information.
    """
    mids = []
    for s in raw_samples:
        try:
            mid = float(s["yes_mid"])
            if 0.01 < mid < 0.99:
                mids.append(mid)
        except (ValueError, KeyError):
            continue

    if not mids:
        return {}

    mids.sort()
    n = len(mids)
    mean = sum(mids) / n
    variance = sum((m - mean) ** 2 for m in mids) / n
    std = math.sqrt(variance)

    # What percentage of time is mid within 2 cents of 0.50?
    near_50 = sum(1 for m in mids if 0.48 <= m <= 0.52) / n

    # What percentage of time is mid > 0.55 or < 0.45?
    strong_signal = sum(1 for m in mids if m > 0.55 or m < 0.45) / n

    return {
        "mean": mean,
        "std": std,
        "min": mids[0],
        "max": mids[-1],
        "p10": mids[int(n * 0.10)],
        "p25": mids[int(n * 0.25)],
        "median": mids[n // 2],
        "p75": mids[int(n * 0.75)],
        "p90": mids[int(n * 0.90)],
        "pct_near_50": near_50,
        "pct_strong_signal": strong_signal,
        "n_samples": n,
    }


# ── Recommendation Engine ────────────────────────────────────────────

def recommend_config(
    efficiency_results: dict,
    directional_results: dict,
    spread_results: dict,
    resolution_stats: dict,
    oracle_lag: dict,
) -> dict:
    """Generate recommended BacktestConfig parameters from real data."""

    # Market efficiency parameter
    # Use the 1-minute mark as the primary reference (most relevant for trading)
    eff_1m = efficiency_results.get("1min", {}).get("efficiency", 0.35)
    eff_avg = efficiency_results.get("avg", {}).get("efficiency", 0.35)
    recommended_efficiency = round((eff_1m * 0.6 + eff_avg * 0.4), 2)

    # Spread parameter
    avg_spread = spread_results.get("avg", {}).get("mean", 0.06)
    recommended_spread = round(avg_spread, 3)

    # Directional accuracy → tells us if there's edge to exploit
    dir_1m = directional_results.get("1min", {}).get("accuracy", 0.5)
    dir_strong = directional_results.get("1min", {}).get("strong_accuracy", 0.5)

    # Oracle lag in bps → indicates potential edge magnitude
    oracle_deviation = oracle_lag.get("median_deviation_bps", 5.0)

    # Recommend max_adjustment based on how much the market moves from 0.50
    # If market barely moves, small adjustment is needed
    # If market moves a lot, larger adjustment captures real pricing
    recommended_max_adj = min(0.25, max(0.08, round(recommended_efficiency * 0.4, 2)))

    # Min edge: if market is efficient, need larger minimum edge to overcome
    if recommended_efficiency > 0.50:
        recommended_min_edge = 0.04
    elif recommended_efficiency > 0.30:
        recommended_min_edge = 0.02
    else:
        recommended_min_edge = 0.01

    # Kelly: scale down if market is very efficient (less certain edge)
    if recommended_efficiency > 0.50:
        recommended_kelly = 0.10
    elif recommended_efficiency > 0.35:
        recommended_kelly = 0.15
    else:
        recommended_kelly = 0.25

    return {
        "market_efficiency": recommended_efficiency,
        "market_base_spread": recommended_spread,
        "max_adjustment": recommended_max_adj,
        "min_edge": recommended_min_edge,
        "kelly_multiplier": recommended_kelly,
        "directional_accuracy_1m": dir_1m,
        "directional_accuracy_strong": dir_strong,
        "oracle_median_deviation_bps": oracle_deviation,
        "viability_assessment": _assess_viability(
            recommended_efficiency, dir_1m, avg_spread, oracle_deviation
        ),
    }


def _assess_viability(efficiency: float, accuracy: float, spread: float, oracle_dev: float) -> str:
    """Assess overall strategy viability."""
    score = 0

    # Low efficiency = more exploitable = good
    if efficiency < 0.25:
        score += 3
    elif efficiency < 0.40:
        score += 2
    elif efficiency < 0.55:
        score += 1

    # Market accuracy < 55% = can beat it
    if accuracy < 0.52:
        score += 2
    elif accuracy < 0.55:
        score += 1

    # Tight spreads = cheaper to trade
    if spread < 0.04:
        score += 2
    elif spread < 0.06:
        score += 1

    # Large oracle deviation = more exploitable lag
    if oracle_dev > 10:
        score += 2
    elif oracle_dev > 5:
        score += 1

    if score >= 7:
        return "STRONG — Market conditions are highly favourable for this strategy"
    elif score >= 5:
        return "GOOD — Reasonable edge exists, strategy should be profitable"
    elif score >= 3:
        return "MARGINAL — Small edge may exist, use conservative sizing"
    else:
        return "POOR — Market is too efficient, strategy unlikely to profit"


# ── Display ───────────────────────────────────────────────────────────

def print_report(
    observations: list[dict],
    raw_samples: list[dict],
    efficiency: dict,
    directional: dict,
    spreads: dict,
    resolution: dict,
    oracle_lag: dict,
    mid_dist: dict,
    recommendation: dict,
):
    print()
    print("=" * 75)
    print("  POLYMARKET EFFICIENCY ANALYSIS")
    print("=" * 75)
    print()

    # Resolution stats
    print("  -- Resolution Statistics --")
    print(f"  Total windows observed:  {resolution['total_windows']}")
    print(f"  UP resolutions:          {resolution['up']} ({resolution['up_pct']:.1%})")
    print(f"  DOWN resolutions:        {resolution['down']} ({1 - resolution['up_pct']:.1%})")
    print()

    # Mid-price distribution
    if mid_dist:
        print("  -- Mid-Price Distribution --")
        print(f"  Mean:    {mid_dist['mean']:.4f}    Std: {mid_dist['std']:.4f}")
        print(f"  Range:   [{mid_dist['min']:.4f}, {mid_dist['max']:.4f}]")
        print(f"  Median:  {mid_dist['median']:.4f}   "
              f"IQR: [{mid_dist['p25']:.4f}, {mid_dist['p75']:.4f}]")
        print(f"  % near 0.50 (±0.02):  {mid_dist['pct_near_50']:.1%}")
        print(f"  % strong signal (>0.55 or <0.45):  {mid_dist['pct_strong_signal']:.1%}")
        print()

    # Directional accuracy
    print("  -- Directional Accuracy (does mid-price predict resolution?) --")
    print(f"  {'Time':>8s}  {'Accuracy':>8s}  {'N':>6s}  {'Strong Acc':>10s}  {'Strong N':>8s}")
    for label in ["4min", "3min", "2min", "1min", "30sec", "avg"]:
        d = directional.get(label)
        if d:
            print(
                f"  {label:>8s}  "
                f"{d['accuracy']:>7.1%}  "
                f"{d['n']:>6d}  "
                f"{d['strong_accuracy']:>9.1%}  "
                f"{d['strong_n']:>8d}"
            )
    print()
    print("  (50% = random / no information.  >55% = market has real signal)")
    print()

    # Market efficiency
    print("  -- Market Efficiency Score --")
    print(f"  {'Time':>8s}  {'Efficiency':>10s}  {'Cal Error':>10s}  {'N':>6s}")
    for label in ["4min", "3min", "2min", "1min", "30sec", "avg"]:
        e = efficiency.get(label)
        if e:
            print(
                f"  {label:>8s}  "
                f"{e['efficiency']:>9.1%}  "
                f"{e['avg_cal_error']:>9.4f}  "
                f"{e['n_windows']:>6.0f}"
            )
    print()
    print("  (0% = random market.  100% = perfectly calibrated.)")
    print()

    # Calibration buckets for the 1-min mark
    eff_1m = efficiency.get("1min", {})
    if eff_1m and eff_1m.get("buckets"):
        print("  -- Calibration at 1-minute mark --")
        print(f"  {'Mid Price':>10s}  {'N':>5s}  {'Actual UP%':>10s}  {'Cal Error':>10s}")
        for b in eff_1m["buckets"]:
            marker = ""
            if b["cal_error"] < 0.05:
                marker = "  << well calibrated"
            elif b["cal_error"] > 0.15:
                marker = "  << miscalibrated"
            print(
                f"  {b['mid_price']:>9.2f}  "
                f"{b['n']:>5d}  "
                f"{b['actual_up_rate']:>9.1%}  "
                f"{b['cal_error']:>9.4f}"
                f"{marker}"
            )
        print()

    # Spread analysis
    if spreads:
        print("  -- Spread Analysis --")
        print(f"  {'Time':>8s}  {'Mean':>8s}  {'Median':>8s}  {'Min':>8s}  {'Max':>8s}")
        for label in ["4min", "3min", "2min", "1min", "30sec", "avg"]:
            s = spreads.get(label)
            if s:
                print(
                    f"  {label:>8s}  "
                    f"{s['mean']:>7.4f}  "
                    f"{s['median']:>7.4f}  "
                    f"{s['min']:>7.4f}  "
                    f"{s['max']:>7.4f}"
                )
        print()

    # Oracle lag
    if oracle_lag:
        print("  -- Oracle vs Exchange Deviation --")
        print(f"  Mean deviation:    {oracle_lag['mean_deviation_bps']:.1f} bps")
        print(f"  Median deviation:  {oracle_lag['median_deviation_bps']:.1f} bps")
        print(f"  P75 deviation:     {oracle_lag['p75_deviation_bps']:.1f} bps")
        print(f"  P90 deviation:     {oracle_lag['p90_deviation_bps']:.1f} bps")
        print(f"  P95 deviation:     {oracle_lag['p95_deviation_bps']:.1f} bps")
        print(f"  Max deviation:     {oracle_lag['max_deviation_bps']:.1f} bps")
        print(f"  Samples:           {oracle_lag['n_samples']}")
        print()

    # Recommendation
    print("  " + "=" * 71)
    print("  RECOMMENDED BACKTEST CONFIG")
    print("  " + "=" * 71)
    print()
    print(f"  market_efficiency    = {recommendation['market_efficiency']:.2f}")
    print(f"  market_base_spread   = {recommendation['market_base_spread']:.3f}")
    print(f"  max_adjustment       = {recommendation['max_adjustment']:.2f}")
    print(f"  min_edge             = {recommendation['min_edge']:.2f}")
    print(f"  kelly_multiplier     = {recommendation['kelly_multiplier']:.2f}")
    print()
    print(f"  Directional accuracy @ 1min:  {recommendation['directional_accuracy_1m']:.1%}")
    print(f"  Strong signal accuracy:       {recommendation['directional_accuracy_strong']:.1%}")
    print(f"  Oracle median deviation:      {recommendation['oracle_median_deviation_bps']:.1f} bps")
    print()
    print(f"  VIABILITY: {recommendation['viability_assessment']}")
    print()

    # Code snippet for easy copy-paste
    print("  -- Copy-paste for run_backtest.py --")
    print()
    print(f"    python -m backtest.run_backtest \\")
    print(f"        --market-efficiency {recommendation['market_efficiency']:.2f} \\")
    print(f"        --max-adjustment {recommendation['max_adjustment']:.2f} \\")
    print(f"        --min-edge {recommendation['min_edge']:.2f} \\")
    print(f"        --kelly {recommendation['kelly_multiplier']:.2f}")
    print()

    # Minimum data warning
    if resolution["total_windows"] < 50:
        print("  !!  WARNING: Only {0} windows observed. Need at least 50-100 for".format(
            resolution["total_windows"]
        ))
        print("     reliable efficiency estimates. Keep the observer running longer.")
        print()
    elif resolution["total_windows"] < 200:
        print("  !!  NOTE: {0} windows gives a reasonable estimate, but 500+".format(
            resolution["total_windows"]
        ))
        print("     windows (24+ hours) will be more reliable.")
        print()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse Polymarket market efficiency from observation data"
    )
    parser.add_argument(
        "--observations",
        default="backtest/data/market_observations.csv",
        help="Path to window-level observation CSV",
    )
    parser.add_argument(
        "--raw",
        default="backtest/data/market_samples_raw.csv",
        help="Path to raw per-sample CSV",
    )
    args = parser.parse_args()

    # Load data
    observations = load_observations(args.observations)
    raw_samples = load_raw_samples(args.raw)

    if not observations:
        print()
        print("  No observation data found!")
        print()
        print("  Run the market observer first to collect data:")
        print("    python -m backtest.market_observer")
        print()
        print("  Let it run for at least 1-2 hours (ideally 24-48 hours)")
        print("  Then re-run this analyser.")
        print()
        sys.exit(1)

    # Compute all metrics
    efficiency = compute_market_efficiency(observations)
    directional = compute_directional_accuracy(observations)
    spreads = compute_spread_analysis(observations)
    resolution = compute_resolution_stats(observations)
    oracle_lag = estimate_oracle_lag(raw_samples)
    mid_dist = compute_mid_price_distribution(raw_samples)

    recommendation = recommend_config(
        efficiency, directional, spreads, resolution, oracle_lag
    )

    # Print report
    print_report(
        observations, raw_samples,
        efficiency, directional, spreads,
        resolution, oracle_lag, mid_dist,
        recommendation,
    )


if __name__ == "__main__":
    main()
