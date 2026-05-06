"""Deep dive analysis of backtest strategies.

Examines Signal Follow and Contrarian EV in detail:
- Rolling window WR (50, 100, 200 trade windows)
- Consecutive win/loss streaks
- PnL distribution and risk metrics
- Edge stability over time (weekly breakdown)
- Entry price vs WR relationship
- Signal strength vs WR (does stronger signal = better outcome?)
- Overlap analysis: when both strategies agree vs disagree
- Simulated bankroll curves with position sizing
"""

import csv
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Re-run the backtest but capture more detail
from backtest.backtest_real_pricing import (
    CandleLookup, MomentumSignal, OracleLagSignal, RegimeDetector,
    StrategyStats, TradeResult, REGIME_PARAMS,
    load_markets, load_snapshots,
    combine_signals, compute_ev, required_edge, compute_orderbook_imbalance,
    FEE_RATE, MIN_EDGE, MAX_EDGE, MIN_CONFIDENCE, MAX_ADJUSTMENT,
    EARLIEST_ENTRY, LATEST_ENTRY,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def strategy_both(
    snapshots_by_offset: dict[int, dict],
    btc_start: float,
    winner: str,
    start_time: str,
    candle_lookup: CandleLookup | None,
    momentum_signal: MomentumSignal,
    oracle_lag_signal: OracleLagSignal,
    regime_detector: RegimeDetector,
) -> dict | None:
    """Run both strategies on the same market, returning detailed data."""
    for offset_sec in sorted(snapshots_by_offset.keys()):
        snap = snapshots_by_offset[offset_sec]
        secs_remaining = 300 - offset_sec

        if secs_remaining > EARLIEST_ENTRY or secs_remaining < LATEST_ENTRY:
            continue

        yes_ask = snap.get("up_best_ask")
        no_ask = snap.get("down_best_ask")
        if not yes_ask or not no_ask or yes_ask <= 0 or no_ask <= 0:
            continue

        btc_price = snap.get("btc_price") or btc_start
        price_up = snap.get("price_up") or 0.5
        price_down = snap.get("price_down") or 0.5

        if not btc_price or btc_price <= 0:
            continue

        snapshot_time = snap.get("snapshot_time", "")
        if candle_lookup and snapshot_time:
            candles = candle_lookup.get_candles(snapshot_time, count=30)
        else:
            candles = []

        if candles and len(candles) >= 2:
            momentum = momentum_signal.compute(candles)
            regime_name, regime_conf, regime_params = regime_detector.detect(candles)
        else:
            momentum = 0.0
            regime_name = "ranging"
            regime_params = REGIME_PARAMS["ranging"]

        oracle = oracle_lag_signal.compute(btc_price, btc_start, btc_start)

        regime_adj = regime_params.get("weight_adjustments")
        raw_signal, prob_up = combine_signals(
            oracle, momentum, secs_remaining,
            regime_weight_adjustments=regime_adj,
        )

        if regime_name == "low_vol":
            continue

        confidence = abs(prob_up - 0.5)
        if confidence < MIN_CONFIDENCE:
            continue

        ev_yes, ev_no = compute_ev(prob_up, yes_ask, no_ask)
        regime_edge_mult = regime_params.get("edge_multiplier", 1.0)
        req_edge = required_edge(secs_remaining, regime_edge_mult)

        # Signal Follow decision
        if prob_up >= 0.5:
            sf_side, sf_price = "YES", yes_ask
            sf_ev = ev_yes
        else:
            sf_side, sf_price = "NO", no_ask
            sf_ev = ev_no

        sf_passes = sf_ev > req_edge

        # Contrarian EV decision
        if ev_yes >= ev_no:
            ce_side, ce_price, ce_ev = "YES", yes_ask, ev_yes
        else:
            ce_side, ce_price, ce_ev = "NO", no_ask, ev_no

        ce_passes = ce_ev > req_edge

        if not sf_passes and not ce_passes:
            continue

        # Determine outcomes
        actual_up = winner == "Up"

        def compute_pnl(side: str, price: float) -> tuple[bool, float]:
            won = (side == "YES" and actual_up) or (side == "NO" and not actual_up)
            if won:
                gross = 1.0 - price
                return True, gross - (gross * FEE_RATE)
            else:
                return False, -price

        result = {
            "market_id": snap.get("market_id", ""),
            "start_time": start_time,
            "secs_remaining": secs_remaining,
            "regime": regime_name,
            "raw_signal": raw_signal,
            "prob_up": prob_up,
            "confidence": confidence,
            "momentum": momentum,
            "oracle": oracle,
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "winner": winner,
            "req_edge": req_edge,
            # Signal Follow
            "sf_side": sf_side,
            "sf_price": sf_price,
            "sf_ev": sf_ev,
            "sf_passes": sf_passes,
            # Contrarian EV
            "ce_side": ce_side,
            "ce_price": ce_price,
            "ce_ev": ce_ev,
            "ce_passes": ce_passes,
            # Agreement
            "strategies_agree": sf_side == ce_side,
        }

        if sf_passes:
            sf_won, sf_pnl = compute_pnl(sf_side, sf_price)
            result["sf_won"] = sf_won
            result["sf_pnl"] = sf_pnl

        if ce_passes:
            ce_won, ce_pnl = compute_pnl(ce_side, ce_price)
            result["ce_won"] = ce_won
            result["ce_pnl"] = ce_pnl

        return result

    return None


def rolling_window_wr(trades: list[dict], key_won: str, window: int) -> list[float]:
    """Compute rolling WR over a window of trades."""
    if len(trades) < window:
        return []
    result = []
    for i in range(window, len(trades) + 1):
        chunk = trades[i - window:i]
        wins = sum(1 for t in chunk if t.get(key_won, False))
        result.append(wins / window)
    return result


def max_streak(trades: list[dict], key_won: str, target: bool) -> int:
    """Find longest consecutive win or loss streak."""
    max_s = 0
    current = 0
    for t in trades:
        if t.get(key_won) == target:
            current += 1
            max_s = max(max_s, current)
        else:
            current = 0
    return max_s


def sharpe_ratio(pnls: list[float]) -> float:
    """Annualized Sharpe ratio (assuming ~288 trades/day for 5-min markets)."""
    if len(pnls) < 2:
        return 0.0
    mean = sum(pnls) / len(pnls)
    variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.001
    # Annualize: 288 5-min windows per day * 365 days
    return (mean / std) * math.sqrt(288 * 365)


def main() -> None:
    markets_path = os.path.join(DATA_DIR, "polybacktest_markets.csv")
    snapshots_path = os.path.join(DATA_DIR, "polybacktest_snapshots.csv")
    klines_path = os.path.join(DATA_DIR, "binance_klines_1m.csv")

    logger.info("Loading data...")
    markets = load_markets(markets_path)
    snapshots = load_snapshots(snapshots_path)
    candle_lookup = CandleLookup(klines_path) if os.path.exists(klines_path) else None

    momentum_sig = MomentumSignal()
    oracle_sig = OracleLagSignal()
    regime_det = RegimeDetector()

    # Collect all results with full detail
    all_results = []
    sf_trades = []  # Signal Follow trades only
    ce_trades = []  # Contrarian EV trades only

    for i, market in enumerate(markets):
        if i % 1000 == 0 and i > 0:
            logger.info(f"Processing {i:,}/{len(markets):,}...")

        mid = market["market_id"]
        winner = market.get("winner")
        btc_start = market.get("btc_price_start", 0)
        if not winner or not btc_start:
            continue

        snaps = snapshots.get(mid, {})
        if not snaps:
            continue

        start_time = market.get("start_time", "")
        result = strategy_both(
            snaps, btc_start, winner, start_time,
            candle_lookup, momentum_sig, oracle_sig, regime_det,
        )
        if result:
            all_results.append(result)
            if result["sf_passes"]:
                sf_trades.append(result)
            if result["ce_passes"]:
                ce_trades.append(result)

    # Sort by time
    sf_trades.sort(key=lambda x: x.get("start_time", ""))
    ce_trades.sort(key=lambda x: x.get("start_time", ""))
    all_results.sort(key=lambda x: x.get("start_time", ""))

    logger.info(f"Total evaluations: {len(all_results)}")
    logger.info(f"Signal Follow trades: {len(sf_trades)}")
    logger.info(f"Contrarian EV trades: {len(ce_trades)}")

    # ─── 1. SIGNAL FOLLOW DEEP DIVE ──────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"  SIGNAL FOLLOW DEEP DIVE — {len(sf_trades)} trades")
    print(f"{'#'*70}")

    sf_wins = sum(1 for t in sf_trades if t.get("sf_won"))
    sf_wr = sf_wins / len(sf_trades) if sf_trades else 0
    sf_pnls = [t["sf_pnl"] for t in sf_trades if "sf_pnl" in t]

    print(f"\n  Win Rate: {sf_wr*100:.1f}% ({sf_wins}/{len(sf_trades)})")
    print(f"  Total PnL: ${sum(sf_pnls):+,.2f}")
    print(f"  Sharpe Ratio: {sharpe_ratio(sf_pnls):.2f}")

    # ── Rolling WR ──
    print(f"\n  Rolling Win Rate:")
    for window in [50, 100, 200]:
        rolling = rolling_window_wr(sf_trades, "sf_won", window)
        if rolling:
            min_wr = min(rolling)
            max_wr = max(rolling)
            avg_wr = sum(rolling) / len(rolling)
            # Check if it ever dipped below 50%
            below_50_pct = sum(1 for r in rolling if r < 0.5) / len(rolling) * 100
            print(f"    {window}-trade window: avg={avg_wr*100:.1f}%, "
                  f"min={min_wr*100:.1f}%, max={max_wr*100:.1f}%, "
                  f"below 50%: {below_50_pct:.1f}% of time")

    # ── Streaks ──
    print(f"\n  Streaks:")
    print(f"    Longest win streak:  {max_streak(sf_trades, 'sf_won', True)}")
    print(f"    Longest loss streak: {max_streak(sf_trades, 'sf_won', False)}")

    # ── Signal strength buckets ──
    print(f"\n  Signal Strength vs Win Rate:")
    strength_buckets: dict[str, list] = defaultdict(list)
    for t in sf_trades:
        conf = t["confidence"]
        if conf < 0.03:
            bucket = "0.02-0.03 (weak)"
        elif conf < 0.05:
            bucket = "0.03-0.05 (moderate)"
        elif conf < 0.08:
            bucket = "0.05-0.08 (strong)"
        elif conf < 0.12:
            bucket = "0.08-0.12 (very strong)"
        else:
            bucket = "0.12+     (extreme)"
        strength_buckets[bucket].append(t)

    for bucket in sorted(strength_buckets.keys()):
        trades = strength_buckets[bucket]
        wins = sum(1 for t in trades if t.get("sf_won"))
        wr = wins / len(trades) if trades else 0
        pnl = sum(t["sf_pnl"] for t in trades if "sf_pnl" in t)
        print(f"    {bucket}: {len(trades):>5} trades, {wr*100:>5.1f}% WR, ${pnl:>+9,.2f}")

    # ── Entry price buckets ──
    print(f"\n  Entry Price vs Win Rate:")
    price_buckets: dict[str, list] = defaultdict(list)
    for t in sf_trades:
        p = t["sf_price"]
        if p < 0.30:
            bucket = "<0.30"
        elif p < 0.40:
            bucket = "0.30-0.40"
        elif p < 0.50:
            bucket = "0.40-0.50"
        elif p < 0.60:
            bucket = "0.50-0.60"
        else:
            bucket = "0.60+"
        price_buckets[bucket].append(t)

    for bucket in sorted(price_buckets.keys()):
        trades = price_buckets[bucket]
        wins = sum(1 for t in trades if t.get("sf_won"))
        wr = wins / len(trades) if trades else 0
        pnl = sum(t["sf_pnl"] for t in trades if "sf_pnl" in t)
        avg_price = sum(t["sf_price"] for t in trades) / len(trades)
        be_wr = avg_price / (avg_price + (1.0 - avg_price) * (1.0 - FEE_RATE))
        print(f"    {bucket:<10}: {len(trades):>5} trades, {wr*100:>5.1f}% WR "
              f"(BE: {be_wr*100:.1f}%), ${pnl:>+9,.2f}")

    # ── Weekly breakdown ──
    print(f"\n  Weekly Performance:")
    weekly: dict[str, list] = defaultdict(list)
    for t in sf_trades:
        try:
            dt = datetime.fromisoformat(t["start_time"].replace("Z", "+00:00"))
            week_start = dt - timedelta(days=dt.weekday())
            week_key = week_start.strftime("%Y-%m-%d")
            weekly[week_key].append(t)
        except (ValueError, KeyError):
            pass

    for week in sorted(weekly.keys()):
        trades = weekly[week]
        wins = sum(1 for t in trades if t.get("sf_won"))
        wr = wins / len(trades) if trades else 0
        pnl = sum(t["sf_pnl"] for t in trades if "sf_pnl" in t)
        print(f"    w/c {week}: {len(trades):>4} trades, {wr*100:>5.1f}% WR, ${pnl:>+8,.2f}")

    # ── Momentum signal direction analysis ──
    print(f"\n  Momentum vs Oracle Signal Agreement:")
    both_agree = [t for t in sf_trades if (t["momentum"] > 0 and t["oracle"] > 0) or (t["momentum"] < 0 and t["oracle"] < 0)]
    disagree = [t for t in sf_trades if t not in both_agree]

    if both_agree:
        agree_wins = sum(1 for t in both_agree if t.get("sf_won"))
        print(f"    Both agree:    {len(both_agree):>5} trades, {agree_wins/len(both_agree)*100:.1f}% WR, "
              f"${sum(t['sf_pnl'] for t in both_agree if 'sf_pnl' in t):+,.2f}")
    if disagree:
        disagree_wins = sum(1 for t in disagree if t.get("sf_won"))
        print(f"    Disagree:      {len(disagree):>5} trades, {disagree_wins/len(disagree)*100:.1f}% WR, "
              f"${sum(t['sf_pnl'] for t in disagree if 'sf_pnl' in t):+,.2f}")

    # ── Momentum dominant vs Oracle dominant ──
    print(f"\n  Dominant Signal Layer:")
    mom_dominant = [t for t in sf_trades if abs(t["momentum"]) > abs(t["oracle"])]
    oracle_dominant = [t for t in sf_trades if abs(t["oracle"]) >= abs(t["momentum"])]

    if mom_dominant:
        wins = sum(1 for t in mom_dominant if t.get("sf_won"))
        print(f"    Momentum led:  {len(mom_dominant):>5} trades, {wins/len(mom_dominant)*100:.1f}% WR, "
              f"${sum(t['sf_pnl'] for t in mom_dominant if 'sf_pnl' in t):+,.2f}")
    if oracle_dominant:
        wins = sum(1 for t in oracle_dominant if t.get("sf_won"))
        print(f"    Oracle led:    {len(oracle_dominant):>5} trades, {wins/len(oracle_dominant)*100:.1f}% WR, "
              f"${sum(t['sf_pnl'] for t in oracle_dominant if 'sf_pnl' in t):+,.2f}")

    # ─── 2. STRATEGY OVERLAP ANALYSIS ────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"  STRATEGY OVERLAP ANALYSIS")
    print(f"{'#'*70}")

    both_pass = [r for r in all_results if r["sf_passes"] and r["ce_passes"]]
    sf_only = [r for r in all_results if r["sf_passes"] and not r["ce_passes"]]
    ce_only = [r for r in all_results if not r["sf_passes"] and r["ce_passes"]]

    print(f"\n  Both pass filter: {len(both_pass)}")
    print(f"  Signal Follow only: {len(sf_only)}")
    print(f"  Contrarian EV only: {len(ce_only)}")

    # When both pass — do they agree on side?
    if both_pass:
        agree = [r for r in both_pass if r["strategies_agree"]]
        disagree = [r for r in both_pass if not r["strategies_agree"]]

        print(f"\n  When both strategies fire:")
        print(f"    Agree on side: {len(agree)} ({len(agree)/len(both_pass)*100:.0f}%)")
        print(f"    Disagree:      {len(disagree)} ({len(disagree)/len(both_pass)*100:.0f}%)")

        if agree:
            sf_wins_agree = sum(1 for r in agree if r.get("sf_won"))
            sf_wr_agree = sf_wins_agree / len(agree)
            sf_pnl_agree = sum(r["sf_pnl"] for r in agree if "sf_pnl" in r)
            print(f"\n    When they AGREE (both pick same side):")
            print(f"      SF outcome: {sf_wr_agree*100:.1f}% WR, ${sf_pnl_agree:+,.2f}")

        if disagree:
            sf_wins_dis = sum(1 for r in disagree if r.get("sf_won"))
            ce_wins_dis = sum(1 for r in disagree if r.get("ce_won"))
            sf_wr_dis = sf_wins_dis / len(disagree)
            ce_wr_dis = ce_wins_dis / len(disagree)
            sf_pnl_dis = sum(r["sf_pnl"] for r in disagree if "sf_pnl" in r)
            ce_pnl_dis = sum(r["ce_pnl"] for r in disagree if "ce_pnl" in r)
            print(f"\n    When they DISAGREE (pick opposite sides):")
            print(f"      SF outcome: {sf_wr_dis*100:.1f}% WR, ${sf_pnl_dis:+,.2f}")
            print(f"      CE outcome: {ce_wr_dis*100:.1f}% WR, ${ce_pnl_dis:+,.2f}")

    # ─── 3. SIMULATED BANKROLL ────────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"  SIMULATED BANKROLL (Signal Follow, $2 flat bet)")
    print(f"{'#'*70}")

    bankroll = 100.0
    peak = 100.0
    max_dd_pct = 0.0
    equity_curve = [100.0]

    for t in sf_trades:
        if "sf_pnl" not in t:
            continue
        # Scale PnL to $2 bet
        pnl = t["sf_pnl"] * 2.0
        bankroll += pnl
        equity_curve.append(bankroll)
        if bankroll > peak:
            peak = bankroll
        dd_pct = (peak - bankroll) / peak * 100
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    print(f"\n  Starting bankroll: $100.00")
    print(f"  Final bankroll:    ${bankroll:,.2f}")
    print(f"  Return:            {(bankroll-100)/100*100:+.1f}%")
    print(f"  Peak bankroll:     ${peak:,.2f}")
    print(f"  Max drawdown:      {max_dd_pct:.1f}%")

    # Show equity curve as simple checkpoints
    print(f"\n  Equity curve checkpoints (every 100 trades):")
    for i in range(0, len(equity_curve), 100):
        idx = min(i, len(equity_curve) - 1)
        bar_len = int((equity_curve[idx] - 90) / 2)
        bar = "#" * max(0, bar_len)
        print(f"    Trade {i:>5}: ${equity_curve[idx]:>8,.2f}  {bar}")
    print(f"    Trade {len(equity_curve)-1:>5}: ${equity_curve[-1]:>8,.2f}")

    # ─── 4. WHAT THE LIVE BOT IS ACTUALLY DOING ──────────────────────
    print(f"\n{'#'*70}")
    print(f"  LIVE BOT COMPARISON")
    print(f"{'#'*70}")

    # The live bot uses Contrarian EV but with the full signal stack
    # Let's see: when CE picks the SAME side as SF, what's the WR?
    # That approximates what the live bot does (CE logic + good signals)

    ce_same_as_sf = [r for r in all_results if r["ce_passes"] and r["sf_passes"] and r["strategies_agree"]]
    ce_diff_from_sf = [r for r in all_results if r["ce_passes"] and r["sf_passes"] and not r["strategies_agree"]]
    ce_sf_missing = [r for r in all_results if r["ce_passes"] and not r["sf_passes"]]

    print(f"\n  When Contrarian EV fires, Signal Follow also fires and agrees: {len(ce_same_as_sf)}")
    if ce_same_as_sf:
        w = sum(1 for r in ce_same_as_sf if r.get("ce_won"))
        print(f"    CE WR: {w/len(ce_same_as_sf)*100:.1f}%, PnL: ${sum(r['ce_pnl'] for r in ce_same_as_sf if 'ce_pnl' in r):+,.2f}")

    print(f"\n  When CE fires but SF disagrees on side: {len(ce_diff_from_sf)}")
    if ce_diff_from_sf:
        w = sum(1 for r in ce_diff_from_sf if r.get("ce_won"))
        print(f"    CE WR: {w/len(ce_diff_from_sf)*100:.1f}%, PnL: ${sum(r['ce_pnl'] for r in ce_diff_from_sf if 'ce_pnl' in r):+,.2f}")

    print(f"\n  When CE fires but SF doesn't pass filter: {len(ce_sf_missing)}")
    if ce_sf_missing:
        w = sum(1 for r in ce_sf_missing if r.get("ce_won"))
        print(f"    CE WR: {w/len(ce_sf_missing)*100:.1f}%, PnL: ${sum(r['ce_pnl'] for r in ce_sf_missing if 'ce_pnl' in r):+,.2f}")

    # ─── 5. HYBRID STRATEGY ANALYSIS ─────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"  HYBRID STRATEGIES")
    print(f"{'#'*70}")

    # Hybrid 1: Use SF direction but CE entry price (buy cheap side ONLY when signal agrees)
    print(f"\n  Hybrid 1: Contrarian EV filtered by Signal Follow agreement")
    print(f"  (Only take CE trades when SF points the same direction)")
    if ce_same_as_sf:
        w = sum(1 for r in ce_same_as_sf if r.get("ce_won"))
        pnl = sum(r["ce_pnl"] for r in ce_same_as_sf if "ce_pnl" in r)
        print(f"    Trades: {len(ce_same_as_sf)}, WR: {w/len(ce_same_as_sf)*100:.1f}%, PnL: ${pnl:+,.2f}")
        if len(ce_same_as_sf) > 0:
            pf_win = sum(r["ce_pnl"] for r in ce_same_as_sf if r.get("ce_pnl", 0) > 0)
            pf_loss = abs(sum(r["ce_pnl"] for r in ce_same_as_sf if r.get("ce_pnl", 0) < 0))
            print(f"    Profit Factor: {pf_win/pf_loss:.3f}" if pf_loss > 0 else "    Profit Factor: inf")

    # Hybrid 2: Signal Follow but only in ranging regime
    print(f"\n  Hybrid 2: Signal Follow in ranging regime only")
    sf_ranging = [t for t in sf_trades if t.get("regime") == "ranging"]
    if sf_ranging:
        w = sum(1 for t in sf_ranging if t.get("sf_won"))
        pnl = sum(t["sf_pnl"] for t in sf_ranging if "sf_pnl" in t)
        print(f"    Trades: {len(sf_ranging)}, WR: {w/len(sf_ranging)*100:.1f}%, PnL: ${pnl:+,.2f}")

    # Hybrid 3: Signal Follow with strong signals only (confidence > 0.05)
    print(f"\n  Hybrid 3: Signal Follow with strong signals (confidence > 0.05)")
    sf_strong = [t for t in sf_trades if t["confidence"] > 0.05]
    if sf_strong:
        w = sum(1 for t in sf_strong if t.get("sf_won"))
        pnl = sum(t["sf_pnl"] for t in sf_strong if "sf_pnl" in t)
        print(f"    Trades: {len(sf_strong)}, WR: {w/len(sf_strong)*100:.1f}%, PnL: ${pnl:+,.2f}")

    # Hybrid 4: Signal Follow Mon-Thu only
    print(f"\n  Hybrid 4: Signal Follow weekdays only (Mon-Fri)")
    sf_weekday = []
    for t in sf_trades:
        try:
            dt = datetime.fromisoformat(t["start_time"].replace("Z", "+00:00"))
            if dt.weekday() < 5:
                sf_weekday.append(t)
        except (ValueError, KeyError):
            pass
    if sf_weekday:
        w = sum(1 for t in sf_weekday if t.get("sf_won"))
        pnl = sum(t["sf_pnl"] for t in sf_weekday if "sf_pnl" in t)
        print(f"    Trades: {len(sf_weekday)}, WR: {w/len(sf_weekday)*100:.1f}%, PnL: ${pnl:+,.2f}")

    # Hybrid 5: Signal Follow Mon-Thu only
    print(f"\n  Hybrid 5: Signal Follow Mon-Thu only (exclude Fri)")
    sf_monthu = []
    for t in sf_trades:
        try:
            dt = datetime.fromisoformat(t["start_time"].replace("Z", "+00:00"))
            if dt.weekday() < 4:
                sf_monthu.append(t)
        except (ValueError, KeyError):
            pass
    if sf_monthu:
        w = sum(1 for t in sf_monthu if t.get("sf_won"))
        pnl = sum(t["sf_pnl"] for t in sf_monthu if "sf_pnl" in t)
        print(f"    Trades: {len(sf_monthu)}, WR: {w/len(sf_monthu)*100:.1f}%, PnL: ${pnl:+,.2f}")

    # Hybrid 6: Signal Follow + strong signal + weekdays
    print(f"\n  Hybrid 6: Signal Follow + strong signal + Mon-Thu")
    sf_best = []
    for t in sf_trades:
        try:
            dt = datetime.fromisoformat(t["start_time"].replace("Z", "+00:00"))
            if dt.weekday() < 4 and t["confidence"] > 0.05:
                sf_best.append(t)
        except (ValueError, KeyError):
            pass
    if sf_best:
        w = sum(1 for t in sf_best if t.get("sf_won"))
        pnl = sum(t["sf_pnl"] for t in sf_best if "sf_pnl" in t)
        print(f"    Trades: {len(sf_best)}, WR: {w/len(sf_best)*100:.1f}%, PnL: ${pnl:+,.2f}")
        if len(sf_best) > 0:
            pf_win = sum(t["sf_pnl"] for t in sf_best if t.get("sf_pnl", 0) > 0)
            pf_loss = abs(sum(t["sf_pnl"] for t in sf_best if t.get("sf_pnl", 0) < 0))
            print(f"    Profit Factor: {pf_win/pf_loss:.3f}" if pf_loss > 0 else "    Profit Factor: inf")
            z = (w - len(sf_best) * 0.5) / math.sqrt(len(sf_best) * 0.25)
            print(f"    Z-score: {z:+.2f}")


if __name__ == "__main__":
    main()
