"""Compute market efficiency from Polymarket historical trade data.

Reads the CSV output from fetch_polymarket_history.py and computes
the Brier Skill Score at each timepoint, which directly maps to the
market_efficiency parameter used in the backtest config.

Usage:
    python -m backtest.compute_efficiency [--data backtest/data/poly_history.csv]
    python -m backtest.compute_efficiency --data backtest/data/market_observations.csv --observer
"""

import argparse
import csv
import sys
from collections import defaultdict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_trade_data(csv_path: str) -> list:
    """Load data from fetch_polymarket_history output."""
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


def load_observer_data(csv_path: str) -> list:
    """Load data from market_observer output."""
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


def compute_efficiency_from_trades(rows: list):
    """Compute efficiency from trade-reconstructed data."""
    print(f"\n{'='*65}")
    print(f"  POLYMARKET MARKET EFFICIENCY ANALYSIS (Trade Data)")
    print(f"{'='*65}")

    # Filter to windows with actual trades
    good = [r for r in rows if int(r.get("window_trades", 0)) > 0]
    print(f"\n  Total windows: {len(rows)}")
    print(f"  Windows with trades: {len(good)}")

    up = sum(1 for r in good if r["resolution"] == "UP")
    down = sum(1 for r in good if r["resolution"] == "DOWN")
    print(f"  UP: {up} ({100*up/len(good):.1f}%)  DOWN: {down} ({100*down/len(good):.1f}%)")

    timepoints = ["4m", "3m", "2m", "1m", "30s"]

    print(f"\n  {'Time':>6s}  {'N':>5s}  {'Accuracy':>8s}  {'Brier':>7s}  {'Skill':>7s}  {'Efficiency':>10s}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*10}")

    efficiencies = {}

    for tp in timepoints:
        key = f"implied_up_{tp}"
        brier_market = 0
        brier_random = 0
        correct = 0
        n = 0

        for r in good:
            val = r.get(key, "")
            if not val:
                continue

            implied_up = float(val)
            actual_up = 1.0 if r["resolution"] == "UP" else 0.0

            brier_market += (implied_up - actual_up) ** 2
            brier_random += (0.5 - actual_up) ** 2

            if (implied_up > 0.5 and actual_up == 1.0) or (implied_up < 0.5 and actual_up == 0.0):
                correct += 1
            n += 1

        if n < 10:
            print(f"  {tp:>6s}  {n:>5d}  {'(insufficient data)':>40s}")
            continue

        brier_market /= n
        brier_random /= n
        skill = 1.0 - (brier_market / brier_random) if brier_random > 0 else 0
        efficiency = max(0, min(1, skill))
        accuracy = correct / n

        efficiencies[tp] = efficiency
        print(f"  {tp:>6s}  {n:>5d}  {accuracy:>7.1%}  {brier_market:>7.4f}  {skill:>7.4f}  {efficiency:>9.1%}")

    # Calibration analysis
    print(f"\n  -- Calibration (does implied price match actual resolution rate?) --")
    print(f"  {'Bin':>12s}  {'N':>5s}  {'Actual UP':>10s}  {'Expected':>9s}  {'Error':>7s}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*10}  {'-'*9}  {'-'*7}")

    # Use 2-minute mark for calibration (usually most data)
    best_tp = "2m"
    key = f"implied_up_{best_tp}"
    bins = [(0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 0.55),
            (0.55, 0.70), (0.70, 0.85), (0.85, 1.0)]

    for lo, hi in bins:
        matches = []
        for r in good:
            val = r.get(key, "")
            if not val:
                continue
            implied = float(val)
            if lo <= implied < hi:
                matches.append((implied, r["resolution"]))

        if len(matches) < 3:
            continue

        actual_up = sum(1 for _, res in matches if res == "UP") / len(matches)
        expected = (lo + hi) / 2
        cal_err = abs(actual_up - expected)
        print(f"  {lo:.2f}-{hi:.2f}  {len(matches):>5d}  {actual_up:>9.1%}  {expected:>8.1%}  {cal_err:>6.3f}")

    # Recommendation
    if efficiencies:
        early_eff = efficiencies.get("4m", efficiencies.get("3m", 0.35))
        late_eff = efficiencies.get("1m", efficiencies.get("30s", 0.35))
        avg_eff = sum(efficiencies.values()) / len(efficiencies)

        print(f"\n  {'='*60}")
        print(f"  RECOMMENDATION FOR BACKTEST CONFIG")
        print(f"  {'='*60}")
        print(f"  market_efficiency = {avg_eff:.2f}  (average across timepoints)")
        print(f"  Early window (4-3min): {early_eff:.2f}")
        print(f"  Late window  (1-0min): {late_eff:.2f}")
        print(f"")
        if early_eff < 0.20:
            print(f"  GOOD NEWS: Market is quite inefficient early in the window.")
            print(f"  The bot has a real edge in the first 1-2 minutes.")
        elif early_eff < 0.40:
            print(f"  MODERATE: Market captures some signal early but not all.")
            print(f"  Edge exists but is thin - need good execution.")
        else:
            print(f"  TOUGH: Market is fairly efficient even early on.")
            print(f"  Strategy may struggle to find consistent edge.")

        print(f"\n  To use in backtest:")
        print(f"  python -m backtest.run_backtest --market-efficiency {avg_eff:.2f}")


def compute_efficiency_from_observer(rows: list):
    """Compute efficiency from market_observer output."""
    print(f"\n{'='*65}")
    print(f"  POLYMARKET MARKET EFFICIENCY ANALYSIS (Observer Data)")
    print(f"{'='*65}")

    good = [r for r in rows if int(r.get("num_samples", 0)) >= 5]
    print(f"\n  Total windows: {len(rows)}")
    print(f"  Windows with 5+ samples: {len(good)}")

    up = sum(1 for r in good if r["resolution"] == "UP")
    down = sum(1 for r in good if r["resolution"] == "DOWN")
    print(f"  UP: {up} ({100*up/len(good):.1f}%)  DOWN: {down} ({100*down/len(good):.1f}%)")

    timepoints = ["4m", "3m", "2m", "1m", "30s"]

    print(f"\n  {'Time':>6s}  {'N':>5s}  {'Accuracy':>8s}  {'Brier':>7s}  {'Skill':>7s}  {'Efficiency':>10s}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*10}")

    for tp in timepoints:
        key = f"yes_mid_at_{tp}"
        brier_market = 0
        brier_random = 0
        correct = 0
        n = 0

        for r in good:
            val = r.get(key, "")
            if not val or float(val) == 0.0:
                continue

            mid = float(val)
            actual_up = 1.0 if r["resolution"] == "UP" else 0.0

            brier_market += (mid - actual_up) ** 2
            brier_random += (0.5 - actual_up) ** 2

            if (mid > 0.5 and actual_up == 1.0) or (mid < 0.5 and actual_up == 0.0):
                correct += 1
            n += 1

        if n < 10:
            print(f"  {tp:>6s}  {n:>5d}  {'(insufficient data)':>40s}")
            continue

        brier_market /= n
        brier_random /= n
        skill = 1.0 - (brier_market / brier_random) if brier_random > 0 else 0
        efficiency = max(0, min(1, skill))
        accuracy = correct / n

        print(f"  {tp:>6s}  {n:>5d}  {accuracy:>7.1%}  {brier_market:>7.4f}  {skill:>7.4f}  {efficiency:>9.1%}")


def main():
    parser = argparse.ArgumentParser(description="Compute Polymarket efficiency")
    parser.add_argument("--data", type=str, default="backtest/data/poly_history.csv")
    parser.add_argument("--observer", action="store_true", help="Use market_observer format")
    args = parser.parse_args()

    if args.observer:
        rows = load_observer_data(args.data)
        compute_efficiency_from_observer(rows)
    else:
        rows = load_trade_data(args.data)
        compute_efficiency_from_trades(rows)


if __name__ == "__main__":
    main()
