"""Stop loss backtest v2 — non-price-based stop mechanisms.

Tests 4 approaches against Bot G's historical trades + tick data:
1. Signal-flip stop: exit when combined signal reverses direction
2. BTC distance-from-open stop: exit when BTC moves too far against us
3. Early reversal stop: exit if price drops > X from entry within Y seconds
4. No-confirmation stop: exit if price hasn't improved after N seconds

Each approach tested across multiple parameter combinations.
"""

import sqlite3
from datetime import datetime, timezone


def load_data():
    """Load trades and build tick lookup."""
    conn_trades = sqlite3.connect('data_runtime/bot_g_signal_aligned.db')
    conn_ticks = sqlite3.connect('backtest/data/tick_data.db')
    cur_t = conn_trades.cursor()
    cur_k = conn_ticks.cursor()

    # Build tick market lookup
    cur_k.execute('SELECT market_id, start_time, btc_price_start FROM markets')
    tick_lookup = {}
    for mid, st, btc_open in cur_k.fetchall():
        key = st.replace('Z', '').replace('+00:00', '')
        tick_lookup[key] = (mid, btc_open)

    # Get all resolved trades with signal data
    cur_t.execute('''
        SELECT market_slug, side, entry_price, size_usdc, time_remaining_secs,
               pnl, resolution, combined_signal, btc_price_at_entry,
               oracle_price_at_open
        FROM trades
        WHERE resolution IS NOT NULL
        ORDER BY timestamp
    ''')
    trades = cur_t.fetchall()

    return conn_trades, conn_ticks, cur_t, cur_k, tick_lookup, trades


def sell_pnl(entry_price, sell_price, size_usdc):
    """Calculate PnL from selling at sell_price."""
    num_shares = size_usdc / entry_price
    return (sell_price - entry_price) * num_shares


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_result(label, matched, stops, helped, hurt, orig_pnl, new_pnl, saved, cost):
    pct = stops / matched * 100 if matched else 0
    h_pct = helped / stops * 100 if stops else 0
    diff = new_pnl - orig_pnl
    sign = "+" if diff >= 0 else ""
    print(f"  {label:<45} | {stops:>4} ({pct:>4.1f}%) | H:{helped:>4} ({h_pct:>4.0f}%) | {sign}${diff:>8.2f}")


# ═══════════════════════════════════════════════════
# APPROACH 1: Signal-flip stop
# Exit when combined signal crosses a threshold in
# the opposite direction to our trade.
# Uses the signal_log or interpolates from snapshots.
# Since we don't have per-tick signal values, we use
# BTC price movement as a proxy for signal direction.
# ═══════════════════════════════════════════════════

def backtest_signal_flip(cur_k, tick_lookup, trades, flip_threshold, min_hold_secs=30):
    """Exit when BTC moves enough to imply signal flip.

    We entered YES when signal > 0 (BTC above oracle/open).
    If BTC then moves flip_threshold below where it was at entry,
    the signal has likely flipped. Exit.

    For NO: if BTC moves flip_threshold ABOVE entry price.
    """
    orig_pnl = 0.0
    new_pnl = 0.0
    stops = 0
    helped = 0
    hurt = 0
    saved = 0.0
    cost = 0.0
    matched = 0

    for trade in trades:
        slug, side, entry_price, size_usdc, secs_rem, pnl, res, comb_sig, btc_entry, oracle_open = trade
        unix_ts = int(slug.split('-')[-1])
        dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        key = dt.strftime('%Y-%m-%dT%H:%M:%S')
        info = tick_lookup.get(key)
        if not info:
            orig_pnl += pnl
            new_pnl += pnl
            continue

        tick_mid, btc_open = info
        matched += 1
        orig_pnl += pnl

        entry_sec_into = 300.0 - secs_rem

        cur_k.execute('''
            SELECT seconds_into_window, btc_price, price_up, price_down,
                   up_best_bid, down_best_bid
            FROM snapshots
            WHERE market_id = ? AND seconds_into_window > ?
            ORDER BY seconds_into_window
        ''', (tick_mid, entry_sec_into))

        stopped = False
        for tick in cur_k.fetchall():
            sec_into, btc_now, p_up, p_down, up_bid, down_bid = tick
            if not btc_now or not btc_entry:
                continue

            time_held = sec_into - entry_sec_into
            if time_held < min_hold_secs:
                continue

            btc_move = btc_now - btc_entry

            # YES trade: signal flips if BTC drops significantly
            # NO trade: signal flips if BTC rises significantly
            if side == "YES" and btc_move < -flip_threshold:
                our_bid = up_bid if up_bid and up_bid > 0 else (p_up if p_up else 0)
                if our_bid and our_bid > 0:
                    sp = sell_pnl(entry_price, our_bid, size_usdc)
                    stops += 1
                    stopped = True
                    d = sp - pnl
                    if d > 0:
                        helped += 1
                        saved += d
                    else:
                        hurt += 1
                        cost += d
                    new_pnl += sp
                    break
            elif side == "NO" and btc_move > flip_threshold:
                our_bid = down_bid if down_bid and down_bid > 0 else (p_down if p_down else 0)
                if our_bid and our_bid > 0:
                    sp = sell_pnl(entry_price, our_bid, size_usdc)
                    stops += 1
                    stopped = True
                    d = sp - pnl
                    if d > 0:
                        helped += 1
                        saved += d
                    else:
                        hurt += 1
                        cost += d
                    new_pnl += sp
                    break

        if not stopped:
            new_pnl += pnl

    return matched, stops, helped, hurt, orig_pnl, new_pnl, saved, cost


# ═══════════════════════════════════════════════════
# APPROACH 2: BTC distance-from-open stop
# Exit when BTC has moved far enough from the window
# open price in the direction AGAINST our trade.
# ═══════════════════════════════════════════════════

def backtest_btc_distance(cur_k, tick_lookup, trades, distance_threshold, min_secs_remaining=0):
    """Exit when BTC is distance_threshold away from open, against us."""
    orig_pnl = 0.0
    new_pnl = 0.0
    stops = 0
    helped = 0
    hurt = 0
    saved = 0.0
    cost = 0.0
    matched = 0

    for trade in trades:
        slug, side, entry_price, size_usdc, secs_rem, pnl, res, comb_sig, btc_entry, oracle_open = trade
        unix_ts = int(slug.split('-')[-1])
        dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        key = dt.strftime('%Y-%m-%dT%H:%M:%S')
        info = tick_lookup.get(key)
        if not info:
            orig_pnl += pnl
            new_pnl += pnl
            continue

        tick_mid, btc_open = info
        matched += 1
        orig_pnl += pnl

        if not btc_open:
            new_pnl += pnl
            continue

        entry_sec_into = 300.0 - secs_rem

        cur_k.execute('''
            SELECT seconds_into_window, btc_price, price_up, price_down,
                   up_best_bid, down_best_bid
            FROM snapshots
            WHERE market_id = ? AND seconds_into_window > ?
            ORDER BY seconds_into_window
        ''', (tick_mid, entry_sec_into))

        stopped = False
        for tick in cur_k.fetchall():
            sec_into, btc_now, p_up, p_down, up_bid, down_bid = tick
            if not btc_now:
                continue

            secs_left = 300.0 - sec_into
            if secs_left < min_secs_remaining:
                break

            dist_from_open = btc_now - btc_open

            # YES: we lose if BTC goes DOWN from open
            # NO: we lose if BTC goes UP from open
            against_us = False
            if side == "YES" and dist_from_open < -distance_threshold:
                against_us = True
                our_bid = up_bid if up_bid and up_bid > 0 else (p_up if p_up else 0)
            elif side == "NO" and dist_from_open > distance_threshold:
                against_us = True
                our_bid = down_bid if down_bid and down_bid > 0 else (p_down if p_down else 0)

            if against_us and our_bid and our_bid > 0:
                sp = sell_pnl(entry_price, our_bid, size_usdc)
                stops += 1
                stopped = True
                d = sp - pnl
                if d > 0:
                    helped += 1
                    saved += d
                else:
                    hurt += 1
                    cost += d
                new_pnl += sp
                break

        if not stopped:
            new_pnl += pnl

    return matched, stops, helped, hurt, orig_pnl, new_pnl, saved, cost


# ═══════════════════════════════════════════════════
# APPROACH 3: Early reversal stop
# Exit if contract price drops > X from entry within
# the first Y seconds of holding.
# ═══════════════════════════════════════════════════

def backtest_early_reversal(cur_k, tick_lookup, trades, drop_threshold, max_hold_secs):
    """Exit if price drops > drop_threshold from entry within max_hold_secs."""
    orig_pnl = 0.0
    new_pnl = 0.0
    stops = 0
    helped = 0
    hurt = 0
    saved = 0.0
    cost = 0.0
    matched = 0

    for trade in trades:
        slug, side, entry_price, size_usdc, secs_rem, pnl, res, comb_sig, btc_entry, oracle_open = trade
        unix_ts = int(slug.split('-')[-1])
        dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        key = dt.strftime('%Y-%m-%dT%H:%M:%S')
        info = tick_lookup.get(key)
        if not info:
            orig_pnl += pnl
            new_pnl += pnl
            continue

        tick_mid, btc_open = info
        matched += 1
        orig_pnl += pnl

        entry_sec_into = 300.0 - secs_rem

        cur_k.execute('''
            SELECT seconds_into_window, price_up, price_down,
                   up_best_bid, down_best_bid
            FROM snapshots
            WHERE market_id = ? AND seconds_into_window > ?
              AND seconds_into_window <= ?
            ORDER BY seconds_into_window
        ''', (tick_mid, entry_sec_into, entry_sec_into + max_hold_secs))

        stopped = False
        for tick in cur_k.fetchall():
            sec_into, p_up, p_down, up_bid, down_bid = tick

            if side == "YES":
                our_price = p_up if p_up else 0
                our_bid = up_bid if up_bid and up_bid > 0 else our_price
            else:
                our_price = p_down if p_down else 0
                our_bid = down_bid if down_bid and down_bid > 0 else our_price

            if not our_price or our_price <= 0:
                continue

            drop = entry_price - our_price
            if drop >= drop_threshold:
                if our_bid and our_bid > 0:
                    sp = sell_pnl(entry_price, our_bid, size_usdc)
                    stops += 1
                    stopped = True
                    d = sp - pnl
                    if d > 0:
                        helped += 1
                        saved += d
                    else:
                        hurt += 1
                        cost += d
                    new_pnl += sp
                    break

        if not stopped:
            new_pnl += pnl

    return matched, stops, helped, hurt, orig_pnl, new_pnl, saved, cost


# ═══════════════════════════════════════════════════
# APPROACH 4: No-confirmation stop
# Exit if after N seconds of holding, price hasn't
# improved from entry (price <= entry_price).
# ═══════════════════════════════════════════════════

def backtest_no_confirmation(cur_k, tick_lookup, trades, wait_secs, require_drop=0.0):
    """Exit if price <= entry_price - require_drop after wait_secs.

    Args:
        wait_secs: How long to wait before checking.
        require_drop: If > 0, only exit if price dropped by at least this much.
                      If 0, exit if price hasn't improved at all.
    """
    orig_pnl = 0.0
    new_pnl = 0.0
    stops = 0
    helped = 0
    hurt = 0
    saved = 0.0
    cost = 0.0
    matched = 0

    for trade in trades:
        slug, side, entry_price, size_usdc, secs_rem, pnl, res, comb_sig, btc_entry, oracle_open = trade
        unix_ts = int(slug.split('-')[-1])
        dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        key = dt.strftime('%Y-%m-%dT%H:%M:%S')
        info = tick_lookup.get(key)
        if not info:
            orig_pnl += pnl
            new_pnl += pnl
            continue

        tick_mid, btc_open = info
        matched += 1
        orig_pnl += pnl

        entry_sec_into = 300.0 - secs_rem
        check_sec = entry_sec_into + wait_secs

        # Get the first tick at or after the check time
        cur_k.execute('''
            SELECT seconds_into_window, price_up, price_down,
                   up_best_bid, down_best_bid
            FROM snapshots
            WHERE market_id = ? AND seconds_into_window >= ?
            ORDER BY seconds_into_window
            LIMIT 1
        ''', (tick_mid, check_sec))

        tick = cur_k.fetchone()
        if not tick:
            new_pnl += pnl
            continue

        sec_into, p_up, p_down, up_bid, down_bid = tick

        if side == "YES":
            our_price = p_up if p_up else 0
            our_bid = up_bid if up_bid and up_bid > 0 else our_price
        else:
            our_price = p_down if p_down else 0
            our_bid = down_bid if down_bid and down_bid > 0 else our_price

        if not our_price or our_price <= 0 or not our_bid or our_bid <= 0:
            new_pnl += pnl
            continue

        threshold = entry_price - require_drop
        if our_price <= threshold:
            sp = sell_pnl(entry_price, our_bid, size_usdc)
            stops += 1
            d = sp - pnl
            if d > 0:
                helped += 1
                saved += d
            else:
                hurt += 1
                cost += d
            new_pnl += sp
        else:
            new_pnl += pnl

    return matched, stops, helped, hurt, orig_pnl, new_pnl, saved, cost


if __name__ == "__main__":
    conn_trades, conn_ticks, cur_t, cur_k, tick_lookup, trades = load_data()

    # ═══════════════════════════════════════════
    # 1. SIGNAL-FLIP STOP (BTC reversal proxy)
    # ═══════════════════════════════════════════
    print_header("APPROACH 1: Signal-flip (BTC reversal from entry)")
    print(f"  {'Parameters':<45} | {'Stops':>12} | {'Helped':>12} | {'PnL Delta':>12}")
    print(f"  {'-'*45}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for flip in [10, 15, 20, 25, 30, 40, 50]:
        for min_hold in [15, 30, 60]:
            r = backtest_signal_flip(cur_k, tick_lookup, trades, flip, min_hold)
            print_result(f"flip=${flip}, hold>{min_hold}s", *r)

    # ═══════════════════════════════════════════
    # 2. BTC DISTANCE-FROM-OPEN STOP
    # ═══════════════════════════════════════════
    print_header("APPROACH 2: BTC distance-from-open")
    print(f"  {'Parameters':<45} | {'Stops':>12} | {'Helped':>12} | {'PnL Delta':>12}")
    print(f"  {'-'*45}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for dist in [15, 20, 25, 30, 40, 50, 60]:
        for min_rem in [0, 30, 60, 120]:
            r = backtest_btc_distance(cur_k, tick_lookup, trades, dist, min_rem)
            print_result(f"dist=${dist}, min_rem={min_rem}s", *r)

    # ═══════════════════════════════════════════
    # 3. EARLY REVERSAL STOP
    # ═══════════════════════════════════════════
    print_header("APPROACH 3: Early reversal (price drop from entry)")
    print(f"  {'Parameters':<45} | {'Stops':>12} | {'Helped':>12} | {'PnL Delta':>12}")
    print(f"  {'-'*45}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for drop in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        for window in [30, 45, 60, 90, 120]:
            r = backtest_early_reversal(cur_k, tick_lookup, trades, drop, window)
            print_result(f"drop=${drop:.2f}, window={window}s", *r)

    # ═══════════════════════════════════════════
    # 4. NO-CONFIRMATION STOP
    # ═══════════════════════════════════════════
    print_header("APPROACH 4: No-confirmation (price hasn't improved)")
    print(f"  {'Parameters':<45} | {'Stops':>12} | {'Helped':>12} | {'PnL Delta':>12}")
    print(f"  {'-'*45}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for wait in [30, 45, 60, 90, 120]:
        for req_drop in [0.0, 0.02, 0.05, 0.08, 0.10]:
            r = backtest_no_confirmation(cur_k, tick_lookup, trades, wait, req_drop)
            print_result(f"wait={wait}s, req_drop=${req_drop:.2f}", *r)

    conn_trades.close()
    conn_ticks.close()
    print("\n\nDone.")
