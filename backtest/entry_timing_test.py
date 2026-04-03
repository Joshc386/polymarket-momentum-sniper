"""Entry timing comparison: Fixed 4m vs Dynamic 4:30-1:00 window.

Compares two strategies:
1. Fixed: Enter at exactly 4m remaining (current bot behaviour)
2. Dynamic: Evaluate continuously from 4m to 1m, enter at peak confidence

Uses real Polymarket pricing + Binance signal data over 30 days.
"""

import csv
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data.binance_feed import Candle
from signals.oracle_lag import OracleLagSignal
from signals.momentum import MomentumSignal
from signals.combiner import SignalCombiner
from strategy.regime_detector import RegimeDetector


def load_candles(path: str) -> list[Candle]:
    candles = []
    with open(path) as f:
        for row in csv.DictReader(f):
            candles.append(Candle(
                open_time=int(row["open_time"]),
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=float(row["volume"]), is_closed=True,
            ))
    return candles


def load_poly(path: str) -> dict:
    windows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = int(row["window_ts"])
            pw = {"resolution": row["resolution"], "implied_up": {}, "trades_at": {}}
            for label in ["4m", "3m", "2m", "1m", "30s"]:
                val = row.get(f"implied_up_{label}", "")
                if val and val.strip():
                    pw["implied_up"][label] = float(val)
                trades = row.get(f"trades_{label}", "0")
                pw["trades_at"][label] = int(trades) if trades else 0
            windows[ts] = pw
    return windows


def get_entry_price(prediction: str, label: str, pw: dict, spread_cost: float):
    """Get entry price from poly data or estimate."""
    if label in pw["implied_up"] and pw["trades_at"].get(label, 0) >= 3:
        implied_up = pw["implied_up"][label]
        if prediction == "UP":
            return min(implied_up + spread_cost, 0.98), "YES", "poly"
        else:
            return min((1.0 - implied_up) + spread_cost, 0.98), "NO", "poly"
    else:
        entry_price = 0.50 + spread_cost
        side = "YES" if prediction == "UP" else "NO"
        return entry_price, side, "estimated"


def main():
    MIN_CONFIDENCE = 0.02
    SPREAD_COST = 0.005
    FEE = 0.02

    print("Loading data...")
    candles = load_candles("backtest/data/BTCUSDT_1m_1mo.csv")
    poly_windows = load_poly("backtest/data/poly_history_30d.csv")

    # Group candles by window
    candle_windows = {}
    for c in candles:
        ts_sec = c.open_time // 1000
        window_ts = ts_sec - (ts_sec % 300)
        if window_ts not in candle_windows:
            candle_windows[window_ts] = []
        candle_windows[window_ts].append(c)

    overlap_ts = sorted(set(candle_windows.keys()) & set(poly_windows.keys()))
    print(f"Overlapping windows: {len(overlap_ts)}")

    # Signal engine
    oracle_lag = OracleLagSignal(max_expected_lag=0.001)
    momentum = MomentumSignal(
        roc_weight=0.30, direction_weight=0.25, volume_weight=0.25,
        body_ratio_weight=0.10, rsi_weight=0.10, rsi_period=5, lookback_candles=10,
    )
    combiner = SignalCombiner(max_adjustment=0.20)
    regime_detector = RegimeDetector()
    history: deque = deque(maxlen=30)

    # Pre-fill
    first_ts = overlap_ts[0]
    pre = sorted([ts for ts in candle_windows if ts < first_ts], reverse=True)[:30]
    for ts in sorted(pre):
        for c in candle_windows[ts]:
            history.append(c)

    timing_map = {0: "4m", 1: "3m", 2: "2m", 3: "1m", 4: "30s"}
    secs_map = {"4m": 240, "3m": 180, "2m": 120, "1m": 60, "30s": 30}

    # Results storage
    results_fixed = []
    results_dynamic = []
    results_dynamic_first = []  # Enter at first signal above threshold
    results_tuned_ev = []       # Strategy 4: corrected fees + flattened thresholds
    results_aggressive_ev = []  # Strategy 5: max_edge=0.015 for early window access
    results_contrarian_ev = []  # Strategy 6: buy best-EV side (contrarian when market extreme)

    for idx, window_ts in enumerate(overlap_ts):
        wc = candle_windows[window_ts]
        pw = poly_windows[window_ts]

        if len(wc) < 2:
            for c in wc:
                history.append(c)
            continue

        oracle_open = wc[0].open
        regime_state = regime_detector.detect(history)
        regime_params = regime_detector.get_params(regime_state.regime)

        # Compute signals at each candle
        signals = {}
        for i, candle in enumerate(wc):
            label = timing_map.get(i)
            if label is None:
                continue
            oracle_val = oracle_lag.compute(
                exchange_price=candle.close, oracle_price=oracle_open,
                oracle_open_price=oracle_open,
            )
            mom_val = momentum.compute(candles=history, current_candle=candle)
            raw, prob_up = combiner.combine(
                oracle_lag_signal=oracle_val, momentum_signal=mom_val,
                liquidation_signal=0.0, seconds_remaining=secs_map[label],
                coinbase_direction=0.0, orderbook_signal=0.0, sentiment_signal=0.0,
                regime_weight_adjustments=regime_params.get("weight_adjustments"),
            )
            signals[i] = {
                "label": label, "prediction": "UP" if prob_up > 0.5 else "DOWN",
                "confidence": abs(prob_up - 0.5), "prob_up": prob_up,
            }

        market_resolution = pw["resolution"]

        def make_trade(sig, label):
            entry_price, side, source = get_entry_price(
                sig["prediction"], label, pw, SPREAD_COST
            )
            won = (side == "YES" and market_resolution == "UP") or \
                  (side == "NO" and market_resolution == "DOWN")
            # Fee is charged on winnings only, not on losses
            if won:
                gross_profit = 1.0 - entry_price
                pnl = gross_profit - (gross_profit * FEE)
            else:
                pnl = -entry_price
            return {
                "won": won, "pnl": pnl, "entry_price": entry_price,
                "confidence": sig["confidence"], "timing": label,
                "source": source, "side": side,
            }

        # Strategy 1: Fixed at 4m (candle 0)
        if 0 in signals and signals[0]["confidence"] >= MIN_CONFIDENCE:
            results_fixed.append(make_trade(signals[0], signals[0]["label"]))

        # Strategy 2: Dynamic — enter at BEST confidence across 4m-1m
        best_sig = None
        best_conf = -1
        for ci in [0, 1, 2, 3]:  # 4m, 3m, 2m, 1m
            if ci not in signals:
                continue
            if signals[ci]["confidence"] < MIN_CONFIDENCE:
                continue
            if signals[ci]["confidence"] > best_conf:
                best_conf = signals[ci]["confidence"]
                best_sig = signals[ci]
        if best_sig is not None:
            results_dynamic.append(make_trade(best_sig, best_sig["label"]))

        # Strategy 3: Dynamic — enter at FIRST signal above threshold
        for ci in [0, 1, 2, 3]:
            if ci not in signals:
                continue
            if signals[ci]["confidence"] >= MIN_CONFIDENCE:
                results_dynamic_first.append(
                    make_trade(signals[ci], signals[ci]["label"])
                )
                break

        # Strategy 4: Tuned EV — corrected fee + flattened thresholds
        # Uses the new parameters: min_edge=0.005, max_edge=0.04, fee as rate on winnings
        TUNED_MIN_EDGE = 0.005
        TUNED_MAX_EDGE = 0.04
        for ci in [0, 1, 2, 3]:  # 4m, 3m, 2m, 1m
            if ci not in signals:
                continue
            sig = signals[ci]
            if sig["confidence"] < MIN_CONFIDENCE:
                continue
            label = sig["label"]
            secs = secs_map[label]
            # Simulate EV with corrected fee formula
            entry_price, side, source = get_entry_price(
                sig["prediction"], label, pw, SPREAD_COST
            )
            profit_if_win = 1.0 - entry_price
            if side == "YES":
                ev = (sig["prob_up"] * (profit_if_win * (1.0 - FEE))) - ((1.0 - sig["prob_up"]) * entry_price)
            else:
                ev = ((1.0 - sig["prob_up"]) * (profit_if_win * (1.0 - FEE))) - (sig["prob_up"] * entry_price)
            required_edge = TUNED_MIN_EDGE + (TUNED_MAX_EDGE - TUNED_MIN_EDGE) * (secs / 300.0)
            if ev > required_edge:
                won = (side == "YES" and market_resolution == "UP") or \
                      (side == "NO" and market_resolution == "DOWN")
                if won:
                    pnl = profit_if_win - (profit_if_win * FEE)
                else:
                    pnl = -entry_price
                results_tuned_ev.append({
                    "won": won, "pnl": pnl, "entry_price": entry_price,
                    "confidence": sig["confidence"], "timing": label,
                    "source": source, "side": side, "ev": ev,
                })
                break

        # Strategy 5: Aggressive EV — max_edge=0.015 for early window trading
        AGG_MIN_EDGE = 0.005
        AGG_MAX_EDGE = 0.015
        for ci in [0, 1, 2, 3]:  # 4m, 3m, 2m, 1m
            if ci not in signals:
                continue
            sig = signals[ci]
            if sig["confidence"] < MIN_CONFIDENCE:
                continue
            label = sig["label"]
            secs = secs_map[label]
            entry_price, side, source = get_entry_price(
                sig["prediction"], label, pw, SPREAD_COST
            )
            profit_if_win = 1.0 - entry_price
            if side == "YES":
                ev = (sig["prob_up"] * (profit_if_win * (1.0 - FEE))) - ((1.0 - sig["prob_up"]) * entry_price)
            else:
                ev = ((1.0 - sig["prob_up"]) * (profit_if_win * (1.0 - FEE))) - (sig["prob_up"] * entry_price)
            required_edge = AGG_MIN_EDGE + (AGG_MAX_EDGE - AGG_MIN_EDGE) * (secs / 300.0)
            if ev > required_edge:
                won = (side == "YES" and market_resolution == "UP") or \
                      (side == "NO" and market_resolution == "DOWN")
                if won:
                    pnl = profit_if_win - (profit_if_win * FEE)
                else:
                    pnl = -entry_price
                results_aggressive_ev.append({
                    "won": won, "pnl": pnl, "entry_price": entry_price,
                    "confidence": sig["confidence"], "timing": label,
                    "source": source, "side": side, "ev": ev,
                })
                break

        # Strategy 6: Contrarian EV — buy whichever side has best EV (corrected fees)
        # This is the original approach but with correct fee math.
        # When market prices YES at 0.15, buying YES is cheap with huge upside.
        CONTR_MIN_EDGE = 0.003
        CONTR_MAX_EDGE = 0.02
        for ci in [0, 1, 2, 3]:  # 4m, 3m, 2m, 1m
            if ci not in signals:
                continue
            sig = signals[ci]
            if sig["confidence"] < MIN_CONFIDENCE:
                continue
            label = sig["label"]
            secs = secs_map[label]

            # Get market prices for BOTH sides
            prob_up = sig["prob_up"]
            if label in pw["implied_up"] and pw["trades_at"].get(label, 0) >= 3:
                implied_up = pw["implied_up"][label]
                yes_price = min(implied_up + SPREAD_COST, 0.98)
                no_price = min((1.0 - implied_up) + SPREAD_COST, 0.98)
                source = "poly"
            else:
                yes_price = 0.50 + SPREAD_COST
                no_price = 0.50 + SPREAD_COST
                source = "estimated"

            # EV for both sides
            profit_yes = 1.0 - yes_price
            ev_yes = (prob_up * (profit_yes * (1.0 - FEE))) - ((1.0 - prob_up) * yes_price)

            profit_no = 1.0 - no_price
            ev_no = ((1.0 - prob_up) * (profit_no * (1.0 - FEE))) - (prob_up * no_price)

            # Pick the better side
            if ev_yes >= ev_no:
                best_ev, side, entry_price, profit_if_win = ev_yes, "YES", yes_price, profit_yes
            else:
                best_ev, side, entry_price, profit_if_win = ev_no, "NO", no_price, profit_no

            required_edge = CONTR_MIN_EDGE + (CONTR_MAX_EDGE - CONTR_MIN_EDGE) * (secs / 300.0)
            if best_ev > required_edge:
                won = (side == "YES" and market_resolution == "UP") or \
                      (side == "NO" and market_resolution == "DOWN")
                if won:
                    pnl = profit_if_win - (profit_if_win * FEE)
                else:
                    pnl = -entry_price
                results_contrarian_ev.append({
                    "won": won, "pnl": pnl, "entry_price": entry_price,
                    "confidence": sig["confidence"], "timing": label,
                    "source": source, "side": side, "ev": best_ev,
                })
                break

        # Update history
        for c in wc:
            if c not in list(history):
                history.append(c)

        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx+1}/{len(overlap_ts)} windows...")

    # ── Print results ──
    all_strats = [
        ("FIXED @ 4m (current bot)", results_fixed),
        ("DYNAMIC best confidence 4m-1m", results_dynamic),
        ("DYNAMIC first signal 4m-1m", results_dynamic_first),
        ("TUNED EV (corrected fee + flat threshold)", results_tuned_ev),
        ("AGGRESSIVE EV (max_edge=0.015)", results_aggressive_ev),
        ("CONTRARIAN EV (best side, corrected fee)", results_contrarian_ev),
    ]

    print()
    print("=" * 75)
    print("  ENTRY TIMING COMPARISON — 30 Days, Real Polymarket Pricing")
    print("=" * 75)

    for name, trades in all_strats:
        if not trades:
            print(f"\n  {name}: No trades")
            continue

        n = len(trades)
        wins = sum(1 for t in trades if t["won"])
        wr = wins / n * 100
        total_pnl = sum(t["pnl"] for t in trades)
        avg_pnl = total_pnl / n
        avg_entry = sum(t["entry_price"] for t in trades) / n
        avg_conf = sum(t["confidence"] for t in trades) / n

        print(f"\n  {name}")
        print(f"  {'=' * 60}")
        print(f"  Trades: {n}  |  WR: {wr:.1f}%  |  Avg PnL: ${avg_pnl:+.4f}  |  Total: ${total_pnl:+.2f}")
        print(f"  Avg entry: ${avg_entry:.4f}  |  Avg confidence: {avg_conf:.4f}")

        # By timing
        timing_groups = defaultdict(list)
        for t in trades:
            timing_groups[t["timing"]].append(t)
        print(f"  Entry distribution:")
        for label in ["4m", "3m", "2m", "1m", "30s"]:
            if label in timing_groups:
                g = timing_groups[label]
                g_wr = sum(1 for t in g if t["won"]) / len(g) * 100
                g_pnl = sum(t["pnl"] for t in g)
                print(f"    {label}: {len(g):>5d} trades ({len(g)/n*100:>5.1f}%), "
                      f"WR={g_wr:.1f}%, PnL=${g_pnl:+.2f}")

        # By price source
        for source in ["poly", "estimated"]:
            g = [t for t in trades if t["source"] == source]
            if g:
                g_wr = sum(1 for t in g if t["won"]) / len(g) * 100
                g_pnl = sum(t["pnl"] for t in g)
                print(f"  Price source {source}: {len(g)} trades, "
                      f"WR={g_wr:.1f}%, PnL=${g_pnl:+.2f}")

    # Bankroll comparison
    print(f"\n  {'=' * 60}")
    print(f"  BANKROLL COMPARISON ($100 start, flat $1 bets)")
    print(f"  {'=' * 60}")

    for name, trades in all_strats:
        bankroll = 100.0
        peak = 100.0
        max_dd = 0.0
        for t in trades:
            bankroll += t["pnl"]
            peak = max(peak, bankroll)
            dd = (peak - bankroll) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        roi = bankroll - 100
        print(f"  {name:40s} -> ${bankroll:>8.2f} "
              f"(+${roi:.2f}, MaxDD: {max_dd*100:.1f}%)")

    # Confidence threshold sweep for dynamic strategy
    print(f"\n  {'=' * 60}")
    print(f"  CONFIDENCE THRESHOLD SWEEP (Dynamic best-confidence)")
    print(f"  {'=' * 60}")
    print(f"  {'Threshold':>10s}  {'N':>6s}  {'WR':>6s}  {'AvgPnL':>8s}  {'Total':>10s}")

    for thresh in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
        filtered = [t for t in results_dynamic if t["confidence"] >= thresh]
        if len(filtered) < 10:
            continue
        fw = sum(1 for t in filtered if t["won"])
        fp = sum(t["pnl"] for t in filtered)
        print(f"  {thresh:>9.0%}  {len(filtered):>6d}  {fw/len(filtered)*100:>5.1f}%  "
              f"${fp/len(filtered):>+7.4f}  ${fp:>+9.2f}")


if __name__ == "__main__":
    main()
