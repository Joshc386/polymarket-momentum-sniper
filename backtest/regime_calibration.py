"""Phase 2: Walk-forward regime calibration (fast version).

Pre-computes all features once over the full 2-year dataset, then
applies different threshold combinations as a fast filter step.
This makes the grid search O(N * combos) instead of O(N * combos * detector_cost).

Scoring approach:
- For each parameter combo, decide which 5-min windows are "tradeable"
- Score by how well the filter selects larger-move, directionally-biased windows
- Walk-forward: 3-month train / 1-month test / 1-month slide = ~20 folds
- Final rank by mean OOS test score with stability penalty
"""

import csv
import itertools
import logging
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy.mtf_regime_detector import (
    MTFRegimeDetector, Candle, load_candles_from_csv,
)
from strategy.regime_detector import Regime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class WindowFeatures:
    """Pre-computed features for a single 5-minute window."""
    timestamp_ms: int
    # 1m-level metrics
    trend_strength: float
    trend_direction: int
    vol_percentile: float
    choppiness: float
    atr: float
    # MTF metrics (for multiple alignment_weight values, store raw components)
    alignment_score: float   # Computed with alignment_weight=0.5 (middle)
    alignment_count: int
    htf_trend: int
    htf_strength: float      # Average HTF trend strength
    # Outcome
    price_change: float
    abs_move: float
    direction: int            # +1 up, -1 down


def aggregate_candles(candles_1m: list[Candle], tf_minutes: int) -> list[Candle]:
    """Aggregate 1m candles into higher timeframe candles."""
    if not candles_1m:
        return []
    tf_ms = tf_minutes * 60_000
    result: list[Candle] = []
    bucket: list[Candle] = []
    bucket_start_ms = (candles_1m[0].timestamp_ms // tf_ms) * tf_ms

    for c in candles_1m:
        candle_bucket = (c.timestamp_ms // tf_ms) * tf_ms
        if candle_bucket != bucket_start_ms and bucket:
            result.append(Candle(
                open=bucket[0].open,
                high=max(b.high for b in bucket),
                low=min(b.low for b in bucket),
                close=bucket[-1].close,
                volume=sum(b.volume for b in bucket),
                timestamp_ms=bucket_start_ms,
            ))
            bucket = []
            bucket_start_ms = candle_bucket
        bucket.append(c)
    if bucket:
        result.append(Candle(
            open=bucket[0].open,
            high=max(b.high for b in bucket),
            low=min(b.low for b in bucket),
            close=bucket[-1].close,
            volume=sum(b.volume for b in bucket),
            timestamp_ms=bucket_start_ms,
        ))
    return result


def precompute_features(
    candles_1m: list[Candle],
    candles_15m: list[Candle],
    candles_1h: list[Candle],
    candles_4h: list[Candle],
    candles_1d: list[Candle],
    window_minutes: int = 5,
) -> list[WindowFeatures]:
    """Pre-compute all features for every 5-minute window.

    Runs the MTF detector once with middle-ground params to extract
    raw feature values. The grid search then applies thresholds to
    these features without re-running the detector.
    """
    detector = MTFRegimeDetector(
        alignment_weight=0.5,  # Middle value; raw components stored separately
        stickiness=1,          # No stickiness during feature extraction
    )

    # Build HTF indexes
    def build_index(candles: list[Candle]) -> dict[int, int]:
        return {c.timestamp_ms: i for i, c in enumerate(candles)}

    idx_15m = build_index(candles_15m)
    idx_1h = build_index(candles_1h)
    idx_4h = build_index(candles_4h)
    idx_1d = build_index(candles_1d)

    def get_htf_slice(htf_candles, htf_index, current_ms, tf_ms, lookback=50):
        bucket_ms = (current_ms // tf_ms) * tf_ms
        while bucket_ms not in htf_index and bucket_ms > htf_candles[0].timestamp_ms:
            bucket_ms -= tf_ms
        if bucket_ms not in htf_index:
            return []
        end_idx = htf_index[bucket_ms] + 1
        start_idx = max(0, end_idx - lookback)
        return htf_candles[start_idx:end_idx]

    features: list[WindowFeatures] = []
    n = len(candles_1m)
    warmup = 200  # 1m candles before scoring

    total_windows = (n - warmup - window_minutes) // window_minutes
    report_every = max(1, total_windows // 10)

    for step, i in enumerate(range(warmup, n - window_minutes, window_minutes)):
        current_ms = candles_1m[i].timestamp_ms
        lookback = candles_1m[max(0, i - 200):i]
        if not lookback:
            continue

        # Update HTF trends
        htf_15m = get_htf_slice(candles_15m, idx_15m, current_ms, 15 * 60_000)
        htf_1h = get_htf_slice(candles_1h, idx_1h, current_ms, 60 * 60_000)
        htf_4h = get_htf_slice(candles_4h, idx_4h, current_ms, 240 * 60_000)
        htf_1d = get_htf_slice(candles_1d, idx_1d, current_ms, 1440 * 60_000)

        if htf_15m:
            detector.update_timeframe("15m", htf_15m)
        if htf_1h:
            detector.update_timeframe("1h", htf_1h)
        if htf_4h:
            detector.update_timeframe("4h", htf_4h)
        if htf_1d:
            detector.update_timeframe("1d", htf_1d)

        state = detector.detect(lookback)

        # 5-minute outcome
        window_close = candles_1m[min(i + window_minutes - 1, n - 1)].close
        window_open = candles_1m[i].open
        price_change = window_close - window_open

        # Compute raw HTF strength (average across timeframes)
        tf_strengths = []
        for tf_data in state.tf_trends.values():
            if isinstance(tf_data, dict):
                tf_strengths.append(tf_data.get("strength", 0))

        features.append(WindowFeatures(
            timestamp_ms=current_ms,
            trend_strength=state.trend_strength,
            trend_direction=1 if state.trend_strength > 0.1 else (-1 if state.trend_strength > 0.1 else 0),
            vol_percentile=state.volatility_pct,
            choppiness=state.choppiness,
            atr=0.0,  # Not needed for filtering
            alignment_score=state.alignment_score,
            alignment_count=state.alignment_count,
            htf_trend=state.htf_trend,
            htf_strength=sum(tf_strengths) / len(tf_strengths) if tf_strengths else 0,
            price_change=price_change,
            abs_move=abs(price_change),
            direction=1 if price_change > 0 else -1,
        ))

        if step % report_every == 0 and step > 0:
            ts = datetime.fromtimestamp(current_ms / 1000, tz=timezone.utc)
            logger.info("  Feature extraction: %d/%d (%.0f%%) - %s",
                       step, total_windows, step / total_windows * 100, ts.date())

    return features


def apply_filter(
    features: list[WindowFeatures],
    params: dict,
) -> tuple[list[WindowFeatures], list[WindowFeatures]]:
    """Apply a regime filter to pre-computed features.

    Returns (traded_windows, all_windows) where traded_windows is the
    subset that passes the filter.
    """
    trend_thresh = params["trend_threshold"]
    chop_thresh = params["chop_threshold"]
    high_vol_pct = params["high_vol_percentile"]
    low_vol_pct = params["low_vol_percentile"]
    align_weight = params["alignment_weight"]
    min_alignment = params["min_alignment"]
    min_htf_strength = params.get("min_htf_strength", 0.0)

    traded = []
    for f in features:
        # Block low vol
        if f.vol_percentile < low_vol_pct:
            continue

        # Block high vol
        if f.vol_percentile > high_vol_pct:
            continue

        # Compute effective trend (blend of 1m and HTF)
        abs_align = abs(f.alignment_score)
        effective_trend = (
            f.trend_strength * (1.0 - align_weight) +
            abs_align * align_weight
        )
        effective_chop = f.choppiness * (1.0 - abs_align * align_weight)

        # Is this a trending window?
        is_trending = effective_trend > trend_thresh and effective_chop < chop_thresh

        # Is HTF alignment strong enough?
        has_alignment = abs_align >= min_alignment

        # Trade if trending OR has strong alignment
        if is_trending or has_alignment:
            traded.append(f)

    return traded, features


def score_filter(
    traded: list[WindowFeatures],
    all_windows: list[WindowFeatures],
) -> dict:
    """Score how well a filter selects good windows.

    Returns dict with score and component metrics.
    """
    if not traded or not all_windows:
        return {"score": 0, "trade_frac": 0, "move_ratio": 1,
                "dir_accuracy": 0.5, "traded": 0, "total": len(all_windows)}

    avg_all = sum(f.abs_move for f in all_windows) / len(all_windows)
    avg_traded = sum(f.abs_move for f in traded) / len(traded)
    trade_frac = len(traded) / len(all_windows)
    move_ratio = avg_traded / avg_all if avg_all > 0 else 1.0

    # Directional accuracy: when HTF says up, does BTC go up?
    dir_total = 0
    dir_correct = 0
    for f in traded:
        if f.htf_trend != 0:
            dir_total += 1
            if f.htf_trend == f.direction:
                dir_correct += 1

    dir_accuracy = dir_correct / dir_total if dir_total > 0 else 0.5

    # Score: reward filters that capture larger moves with reasonable
    # frequency and directional accuracy
    # Penalize extremely low or high trade fractions
    freq_penalty = 1.0
    if trade_frac < 0.15:
        freq_penalty = trade_frac / 0.15  # Linear penalty below 15%
    elif trade_frac > 0.85:
        freq_penalty = (1.0 - trade_frac) / 0.15  # Penalty above 85%

    score = move_ratio * freq_penalty * (0.5 + dir_accuracy)

    return {
        "score": score,
        "trade_frac": trade_frac,
        "move_ratio": move_ratio,
        "dir_accuracy": dir_accuracy,
        "traded": len(traded),
        "total": len(all_windows),
    }


def generate_folds(
    features: list[WindowFeatures],
    train_days: int = 90,
    test_days: int = 30,
    slide_days: int = 30,
) -> list[dict]:
    """Generate walk-forward fold index ranges over the feature array."""
    day_ms = 86_400_000
    start_ms = features[0].timestamp_ms
    end_ms = features[-1].timestamp_ms

    # Build timestamp-to-index lookup
    ts_to_idx = {}
    for i, f in enumerate(features):
        ts_to_idx[f.timestamp_ms] = i

    folds = []
    current = start_ms

    while current + (train_days + test_days) * day_ms <= end_ms:
        train_start = current
        train_end = current + train_days * day_ms
        test_end = train_end + test_days * day_ms

        # Find nearest indices
        train_start_idx = None
        train_end_idx = None
        test_end_idx = None

        for i, f in enumerate(features):
            if train_start_idx is None and f.timestamp_ms >= train_start:
                train_start_idx = i
            if train_end_idx is None and f.timestamp_ms >= train_end:
                train_end_idx = i
            if test_end_idx is None and f.timestamp_ms >= test_end:
                test_end_idx = i
                break

        if train_start_idx is not None and train_end_idx is not None:
            if test_end_idx is None:
                test_end_idx = len(features)
            folds.append({
                "train_start": train_start_idx,
                "train_end": train_end_idx,
                "test_end": test_end_idx,
                "train_start_date": datetime.fromtimestamp(
                    features[train_start_idx].timestamp_ms / 1000, tz=timezone.utc
                ).date(),
                "test_end_date": datetime.fromtimestamp(
                    features[min(test_end_idx, len(features) - 1)].timestamp_ms / 1000,
                    tz=timezone.utc
                ).date(),
            })

        current += slide_days * day_ms

    return folds


def generate_param_grid() -> list[dict]:
    """Generate parameter combinations to test."""
    grid = {
        "trend_threshold": [0.2, 0.3, 0.4, 0.5],
        "chop_threshold": [0.5, 0.6, 0.7, 0.8],
        "high_vol_percentile": [0.75, 0.82, 0.90],
        "low_vol_percentile": [0.10, 0.15, 0.20],
        "alignment_weight": [0.25, 0.4, 0.55],
        "min_alignment": [0.15, 0.25, 0.40],
    }

    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def main() -> None:
    logger.info("Loading 1-minute candles...")
    candles_1m = load_candles_from_csv(str(DATA_DIR / "binance_klines_1m.csv"))
    logger.info("  %d candles loaded", len(candles_1m))

    logger.info("Aggregating higher timeframes...")
    candles_15m = aggregate_candles(candles_1m, 15)
    candles_1h = aggregate_candles(candles_1m, 60)
    candles_4h = aggregate_candles(candles_1m, 240)
    candles_1d = aggregate_candles(candles_1m, 1440)

    logger.info("Pre-computing features for all 5-minute windows...")
    features = precompute_features(
        candles_1m, candles_15m, candles_1h, candles_4h, candles_1d,
    )
    logger.info("  %d windows with features", len(features))

    # Free memory — we no longer need raw candles
    del candles_1m, candles_15m, candles_1h, candles_4h, candles_1d

    # Generate folds
    folds = generate_folds(features)
    logger.info("Generated %d walk-forward folds", len(folds))
    for i, fold in enumerate(folds):
        logger.info("  Fold %d: train %s -> test %s (%d train, %d test windows)",
                    i, fold["train_start_date"], fold["test_end_date"],
                    fold["train_end"] - fold["train_start"],
                    fold["test_end"] - fold["train_end"])

    # Generate parameter grid
    param_grid = generate_param_grid()
    logger.info("Parameter grid: %d combinations", len(param_grid))

    default_params = {
        "trend_threshold": 0.4,
        "chop_threshold": 0.6,
        "high_vol_percentile": 0.85,
        "low_vol_percentile": 0.15,
        "alignment_weight": 0.4,
        "min_alignment": 0.2,
    }

    # Run grid search
    results = []
    all_params = [default_params] + param_grid
    total = len(all_params)

    for combo_idx, params in enumerate(all_params):
        is_default = combo_idx == 0
        train_scores = []
        test_scores = []
        test_trade_fracs = []
        test_dir_accs = []
        test_details = []

        for fold in folds:
            train_features = features[fold["train_start"]:fold["train_end"]]
            test_features = features[fold["train_end"]:fold["test_end"]]

            # Score on train
            traded_train, _ = apply_filter(train_features, params)
            train_result = score_filter(traded_train, train_features)
            train_scores.append(train_result["score"])

            # Score on test (OOS)
            traded_test, _ = apply_filter(test_features, params)
            test_result = score_filter(traded_test, test_features)
            test_scores.append(test_result["score"])
            test_trade_fracs.append(test_result["trade_frac"])
            test_dir_accs.append(test_result["dir_accuracy"])
            test_details.append(test_result)

        train_mean = statistics.mean(train_scores) if train_scores else 0
        test_mean = statistics.mean(test_scores) if test_scores else 0
        test_std = statistics.stdev(test_scores) if len(test_scores) > 1 else 0
        avg_trade_frac = statistics.mean(test_trade_fracs) if test_trade_fracs else 0
        avg_dir_acc = statistics.mean(test_dir_accs) if test_dir_accs else 0

        # Stability-adjusted score: penalize high variance
        stability_score = test_mean - 0.5 * test_std

        results.append({
            "params": params,
            "train_score": train_mean,
            "test_score": test_mean,
            "test_std": test_std,
            "stability_score": stability_score,
            "avg_trade_frac": avg_trade_frac,
            "avg_dir_acc": avg_dir_acc,
            "test_details": test_details,
            "is_default": is_default,
        })

        if is_default or (combo_idx + 1) % 200 == 0:
            label = "DEFAULT" if is_default else f"Combo {combo_idx}"
            logger.info(
                "  %s: train=%.4f, test=%.4f (+/-%.4f), stable=%.4f, "
                "trade%%=%.1f%%, dir=%.1f%%",
                label, train_mean, test_mean, test_std, stability_score,
                avg_trade_frac * 100, avg_dir_acc * 100,
            )

    # Sort by stability-adjusted score
    results.sort(key=lambda r: r["stability_score"], reverse=True)

    # Print results
    print("\n" + "=" * 110)
    print("WALK-FORWARD REGIME CALIBRATION RESULTS")
    print(f"Dataset: {features[0].timestamp_ms} to {features[-1].timestamp_ms}")
    print(f"Folds: {len(folds)}, Param combos: {total}")
    print("=" * 110)

    # Default baseline
    default_result = next((r for r in results if r["is_default"]), None)
    if default_result:
        rank = results.index(default_result) + 1
        print(f"\nDEFAULT params rank: {rank}/{len(results)} "
              f"(test={default_result['test_score']:.4f}, "
              f"stable={default_result['stability_score']:.4f})")

    print(f"\n{'Rank':>4} {'Stable':>8} {'Test':>8} {'Train':>8} {'Std':>6} "
          f"{'Trade%':>7} {'Dir%':>5} | Key Parameters")
    print("-" * 110)

    for rank, result in enumerate(results[:25], 1):
        param_parts = []
        for k, v in result["params"].items():
            dv = default_params.get(k)
            if v != dv:
                param_parts.append(f"{k}={v}")
        param_str = ", ".join(param_parts) if param_parts else "(default)"
        marker = " *DEFAULT*" if result["is_default"] else ""

        print(f"{rank:>4} {result['stability_score']:>8.4f} "
              f"{result['test_score']:>8.4f} {result['train_score']:>8.4f} "
              f"{result['test_std']:>6.4f} {result['avg_trade_frac']*100:>6.1f}% "
              f"{result['avg_dir_acc']*100:>4.1f}% | {param_str}{marker}")

    # Best result details
    best = results[0]
    print("\n" + "=" * 110)
    print("BEST PARAMETER SET (by stability-adjusted OOS score)")
    print("=" * 110)
    for k, v in best["params"].items():
        dv = default_params.get(k)
        changed = " <-- CHANGED" if v != dv else ""
        print(f"  {k}: {v}{changed}")

    print(f"\n  Stability score: {best['stability_score']:.4f}")
    print(f"  Test score:      {best['test_score']:.4f} (+/- {best['test_std']:.4f})")
    print(f"  Train score:     {best['train_score']:.4f}")
    print(f"  Trade fraction:  {best['avg_trade_frac']*100:.1f}%")
    print(f"  Dir accuracy:    {best['avg_dir_acc']*100:.1f}%")

    print("\n  Per-fold OOS breakdown:")
    for i, detail in enumerate(best["test_details"]):
        print(f"    Fold {i}: score={detail['score']:.4f}, "
              f"traded={detail['traded']}/{detail['total']} "
              f"({detail['trade_frac']*100:.1f}%), "
              f"move_ratio={detail['move_ratio']:.3f}, "
              f"dir={detail['dir_accuracy']*100:.1f}%")

    # Most stable from top 50
    top_50 = results[:50]
    stable_sorted = sorted(top_50, key=lambda r: r["test_std"])
    print("\n" + "=" * 110)
    print("MOST STABLE PARAMS (lowest OOS variance, from top 50)")
    print("=" * 110)
    print(f"{'#':>3} {'Stable':>8} {'Test':>8} {'Std':>8} {'Trade%':>7} | Parameters")
    print("-" * 80)
    for i, result in enumerate(stable_sorted[:10], 1):
        param_parts = []
        for k, v in result["params"].items():
            dv = default_params.get(k)
            if v != dv:
                param_parts.append(f"{k}={v}")
        param_str = ", ".join(param_parts) if param_parts else "(default)"
        print(f"{i:>3} {result['stability_score']:>8.4f} "
              f"{result['test_score']:>8.4f} {result['test_std']:>8.4f} "
              f"{result['avg_trade_frac']*100:>6.1f}% | {param_str}")

    # Save results
    output_path = DATA_DIR / "regime_calibration_results.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "rank", "stability_score", "test_score", "train_score", "test_std",
            "avg_trade_frac", "avg_dir_acc",
        ] + list(default_params.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, 1):
            row = {
                "rank": rank,
                "stability_score": f"{result['stability_score']:.6f}",
                "test_score": f"{result['test_score']:.6f}",
                "train_score": f"{result['train_score']:.6f}",
                "test_std": f"{result['test_std']:.6f}",
                "avg_trade_frac": f"{result['avg_trade_frac']:.4f}",
                "avg_dir_acc": f"{result['avg_dir_acc']:.4f}",
            }
            row.update(result["params"])
            writer.writerow(row)

    logger.info("Results saved to %s (%d rows)", output_path, len(results))


if __name__ == "__main__":
    main()
