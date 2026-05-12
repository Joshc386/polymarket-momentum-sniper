"""Stop loss backtest — replay Bot G trades against tick data.

Tests two approaches:
- Approach C: Time-aware price thresholds (different stop levels by time remaining)
- Approach F: Hybrid (grace period + time-aware + dead zone)

For each trade, replays the tick-by-tick price path from entry to resolution
and checks whether a stop loss would have improved PnL.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class StopResult:
    """Result of applying a stop rule to a single trade."""
    trade_idx: int
    side: str
    entry_price: float
    size_usdc: float
    original_pnl: float
    resolution: str
    stop_triggered: bool = False
    stop_price: float = 0.0
    stop_secs_remaining: float = 0.0
    stop_pnl: float = 0.0


def get_stop_threshold_c(secs_remaining: float, params: dict) -> float:
    """Approach C: Time-aware stop thresholds.

    Returns the minimum price for our side below which we exit.
    Returns 0.0 if no stop applies (grace/dead zone).
    """
    thresholds = params["thresholds"]  # list of (min_secs, max_secs, stop_price)
    for min_s, max_s, stop_p in thresholds:
        if min_s <= secs_remaining < max_s:
            return stop_p
    return 0.0


def get_stop_threshold_f(
    secs_remaining: float,
    entry_price: float,
    params: dict,
) -> float:
    """Approach F: Hybrid — grace period + time-scaled + late stop.

    stop_price = entry_price × (base + time_scale × (secs_remaining / 300))
    But with a grace period after entry and a floor for late-window.
    """
    grace = params.get("grace_secs", 30)
    dead_zone = params.get("dead_zone_secs", 0)  # 0 = allow stops all the way
    late_floor = params.get("late_floor", 0.05)  # minimum stop price in last 60s
    base = params.get("base", 0.3)
    time_scale = params.get("time_scale", 0.2)

    if dead_zone > 0 and secs_remaining < dead_zone:
        return 0.0  # no stop in dead zone

    # Time-scaled threshold
    ratio = secs_remaining / 300.0
    threshold = entry_price * (base + time_scale * ratio)

    # Late-window floor
    if secs_remaining < 60 and late_floor > 0:
        threshold = max(threshold, late_floor)

    return threshold


def replay_trade(
    cur_ticks,
    tick_market_id: str,
    side: str,
    entry_price: float,
    entry_secs_remaining: float,
    size_usdc: float,
    original_pnl: float,
    resolution: str,
    approach: str,
    params: dict,
    trade_idx: int,
) -> StopResult:
    """Replay a single trade against tick data with stop logic."""

    result = StopResult(
        trade_idx=trade_idx,
        side=side,
        entry_price=entry_price,
        size_usdc=size_usdc,
        original_pnl=original_pnl,
        resolution=resolution,
    )

    # Entry is at seconds_into_window = 300 - entry_secs_remaining
    entry_sec_into = 300.0 - entry_secs_remaining
    grace_secs = params.get("grace_secs", 0)

    # Get ticks from entry onwards
    cur_ticks.execute('''
        SELECT seconds_into_window, price_up, price_down,
               up_best_bid, down_best_bid
        FROM snapshots
        WHERE market_id = ? AND seconds_into_window > ?
        ORDER BY seconds_into_window
    ''', (tick_market_id, entry_sec_into))

    ticks = cur_ticks.fetchall()
    if not ticks:
        return result

    for tick in ticks:
        sec_into, p_up, p_down, up_bid, down_bid = tick
        secs_remaining = 300.0 - sec_into

        # Skip grace period after entry
        time_since_entry = sec_into - entry_sec_into
        if grace_secs > 0 and time_since_entry < grace_secs:
            continue

        # Our side's current price (what we could sell at = best bid)
        if side == "YES":
            our_price = p_up  # mid price
            our_bid = up_bid if up_bid else p_up  # what we'd actually get
        else:
            our_price = p_down
            our_bid = down_bid if down_bid else p_down

        if not our_price or our_price <= 0:
            continue

        # Get stop threshold
        if approach == "C":
            threshold = get_stop_threshold_c(secs_remaining, params)
        elif approach == "F":
            threshold = get_stop_threshold_f(secs_remaining, entry_price, params)
        else:
            threshold = 0.0

        if threshold > 0 and our_price <= threshold:
            # Stop triggered — sell at best bid (realistic execution)
            sell_price = our_bid if our_bid > 0 else our_price
            num_shares = size_usdc / entry_price
            # PnL = (sell_price - entry_price) × num_shares
            result.stop_triggered = True
            result.stop_price = sell_price
            result.stop_secs_remaining = secs_remaining
            result.stop_pnl = (sell_price - entry_price) * num_shares
            return result

    return result


def run_backtest(approach: str, params: dict, label: str = ""):
    """Run full backtest with given stop parameters."""

    conn_trades = sqlite3.connect('data_runtime/bot_g_signal_aligned.db')
    conn_ticks = sqlite3.connect('backtest/data/tick_data.db')
    cur_t = conn_trades.cursor()
    cur_k = conn_ticks.cursor()

    # Build tick market lookup
    cur_k.execute('SELECT market_id, start_time FROM markets')
    tick_by_start = {}
    for mid, st in cur_k.fetchall():
        key = st.replace('Z', '').replace('+00:00', '')
        tick_by_start[key] = mid

    # Get all resolved trades
    cur_t.execute('''
        SELECT market_slug, side, entry_price, size_usdc, time_remaining_secs,
               pnl, resolution
        FROM trades
        WHERE resolution IS NOT NULL
        ORDER BY timestamp
    ''')
    trades = cur_t.fetchall()

    results = []
    matched = 0

    for idx, trade in enumerate(trades):
        slug, side, entry_price, size_usdc, secs_rem, pnl, resolution = trade

        # Find matching tick data
        unix_ts = int(slug.split('-')[-1])
        dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        key = dt.strftime('%Y-%m-%dT%H:%M:%S')

        tick_mid = tick_by_start.get(key)
        if not tick_mid:
            continue

        matched += 1
        result = replay_trade(
            cur_k, tick_mid, side, entry_price, secs_rem,
            size_usdc, pnl, resolution, approach, params, idx,
        )
        results.append(result)

    conn_trades.close()
    conn_ticks.close()

    return results, matched


def summarize(results_and_matched: tuple, label: str):
    results, matched = results_and_matched
    """Print summary statistics."""
    total = len(results)
    triggered = [r for r in results if r.stop_triggered]
    not_triggered = [r for r in results if not r.stop_triggered]

    original_pnl = sum(r.original_pnl for r in results)

    # New PnL: stopped trades get stop_pnl, others keep original
    new_pnl = sum(
        r.stop_pnl if r.stop_triggered else r.original_pnl
        for r in results
    )

    # How many stops helped vs hurt?
    helped = [r for r in triggered if r.stop_pnl > r.original_pnl]
    hurt = [r for r in triggered if r.stop_pnl <= r.original_pnl]

    # Saved = sum of (stop_pnl - original_pnl) for helped trades
    saved = sum(r.stop_pnl - r.original_pnl for r in helped)
    cost = sum(r.stop_pnl - r.original_pnl for r in hurt)

    # Avg time remaining when stop triggered
    avg_stop_time = (
        sum(r.stop_secs_remaining for r in triggered) / len(triggered)
        if triggered else 0
    )

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades analysed:    {total}")
    print(f"  Stops triggered:    {len(triggered)} ({len(triggered)/total*100:.1f}%)")
    print(f"  Stops helped:       {len(helped)} ({len(helped)/max(len(triggered),1)*100:.1f}% of triggered)")
    print(f"  Stops hurt:         {len(hurt)} ({len(hurt)/max(len(triggered),1)*100:.1f}% of triggered)")
    print(f"  Avg stop time rem:  {avg_stop_time:.0f}s")
    print(f"  --------------------")
    print(f"  Original PnL:       ${original_pnl:.2f}")
    print(f"  New PnL:            ${new_pnl:.2f}")
    print(f"  Difference:         ${new_pnl - original_pnl:+.2f}")
    print(f"  Money saved:        ${saved:.2f}")
    print(f"  Money lost (hurt):  ${cost:.2f}")
    print(f"  Net improvement:    ${saved + cost:+.2f}")

    if triggered:
        # Break down by side
        yes_triggered = [r for r in triggered if r.side == "YES"]
        no_triggered = [r for r in triggered if r.side == "NO"]

        yes_saved = sum(r.stop_pnl - r.original_pnl for r in yes_triggered)
        no_saved = sum(r.stop_pnl - r.original_pnl for r in no_triggered)

        print(f"  --------------------")
        print(f"  YES stops: {len(yes_triggered)}, net: ${yes_saved:+.2f}")
        print(f"  NO stops:  {len(no_triggered)}, net: ${no_saved:+.2f}")


if __name__ == "__main__":
    # ═══════════════════════════════════════════
    # APPROACH C: Time-aware fixed thresholds
    # ═══════════════════════════════════════════

    # Variant C1: Original thresholds (from discussion)
    summarize(run_backtest("C", {
        "thresholds": [
            (240, 300, 0.0),    # first 60s: no stop
            (180, 240, 0.20),   # 180-240s rem: stop if < $0.20
            (120, 180, 0.25),   # 120-180s: stop if < $0.25
            (60, 120, 0.30),    # 60-120s: stop if < $0.30
            (0, 60, 0.0),      # last 60s: no stop
        ],
    }), "C1: Original thresholds (high)")

    # Variant C2: Lower thresholds
    summarize(run_backtest("C", {
        "thresholds": [
            (240, 300, 0.0),
            (180, 240, 0.10),
            (120, 180, 0.15),
            (60, 120, 0.20),
            (0, 60, 0.0),
        ],
    }), "C2: Lower thresholds")

    # Variant C3: Very low thresholds
    summarize(run_backtest("C", {
        "thresholds": [
            (240, 300, 0.0),
            (180, 240, 0.05),
            (120, 180, 0.10),
            (60, 120, 0.15),
            (0, 60, 0.0),
        ],
    }), "C3: Very low thresholds")

    # Variant C4: With last-60s stop at $0.05
    summarize(run_backtest("C", {
        "thresholds": [
            (240, 300, 0.0),
            (180, 240, 0.10),
            (120, 180, 0.15),
            (60, 120, 0.20),
            (0, 60, 0.05),     # last 60s: stop at $0.05
        ],
    }), "C4: Lower thresholds + last-60s at $0.05")

    # Variant C5: Only last 60s stop at $0.05
    summarize(run_backtest("C", {
        "thresholds": [
            (60, 300, 0.0),     # no stop until last 60s
            (0, 60, 0.05),
        ],
    }), "C5: Only last-60s stop at $0.05")

    # Variant C6: Last 60s at $0.10
    summarize(run_backtest("C", {
        "thresholds": [
            (60, 300, 0.0),
            (0, 60, 0.10),
        ],
    }), "C6: Only last-60s stop at $0.10")

    # ═══════════════════════════════════════════
    # APPROACH F: Hybrid — grace + time-scaled + late floor
    # ═══════════════════════════════════════════

    # F1: Conservative — wide base, modest scaling
    summarize(run_backtest("F", {
        "grace_secs": 30,
        "dead_zone_secs": 0,
        "late_floor": 0.05,
        "base": 0.20,
        "time_scale": 0.15,
    }), "F1: base=0.20, scale=0.15, floor=$0.05, grace=30s")

    # F2: Moderate
    summarize(run_backtest("F", {
        "grace_secs": 30,
        "dead_zone_secs": 0,
        "late_floor": 0.05,
        "base": 0.25,
        "time_scale": 0.20,
    }), "F2: base=0.25, scale=0.20, floor=$0.05, grace=30s")

    # F3: Tighter
    summarize(run_backtest("F", {
        "grace_secs": 30,
        "dead_zone_secs": 0,
        "late_floor": 0.05,
        "base": 0.30,
        "time_scale": 0.25,
    }), "F3: base=0.30, scale=0.25, floor=$0.05, grace=30s")

    # F4: Very tight with higher late floor
    summarize(run_backtest("F", {
        "grace_secs": 30,
        "dead_zone_secs": 0,
        "late_floor": 0.10,
        "base": 0.25,
        "time_scale": 0.20,
    }), "F4: base=0.25, scale=0.20, floor=$0.10, grace=30s")

    # F5: Wide grace, late floor only
    summarize(run_backtest("F", {
        "grace_secs": 60,
        "dead_zone_secs": 0,
        "late_floor": 0.05,
        "base": 0.15,
        "time_scale": 0.10,
    }), "F5: base=0.15, scale=0.10, floor=$0.05, grace=60s")

    # F6: No dead zone, aggressive late floor
    summarize(run_backtest("F", {
        "grace_secs": 30,
        "dead_zone_secs": 0,
        "late_floor": 0.08,
        "base": 0.20,
        "time_scale": 0.15,
    }), "F6: base=0.20, scale=0.15, floor=$0.08, grace=30s")

    print("\n\nDone.")
