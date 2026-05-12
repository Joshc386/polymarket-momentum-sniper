"""Stop loss backtest v2 (fast) — preloads tick data to avoid per-trade queries.

Tests 4 approaches against Bot G's historical trades + tick data:
1. Signal-flip stop: exit when BTC reverses from entry price
2. BTC distance-from-open stop: exit when BTC moves too far against us
3. Early reversal stop: exit if price drops > X from entry within Y seconds
4. No-confirmation stop: exit if price hasn't improved after N seconds
"""

import sqlite3
import sys
from datetime import datetime, timezone


def load_all_data():
    """Preload everything into memory for fast iteration."""
    print("Loading data...", flush=True)

    conn_trades = sqlite3.connect('data_runtime/bot_g_signal_aligned.db')
    conn_ticks = sqlite3.connect('backtest/data/tick_data.db')

    # Load markets
    cur = conn_ticks.cursor()
    cur.execute('SELECT market_id, start_time, btc_price_start FROM markets')
    market_lookup = {}  # start_time_key -> (market_id, btc_open)
    market_ids_needed = set()
    for mid, st, btc_open in cur.fetchall():
        key = st.replace('Z', '').replace('+00:00', '')
        market_lookup[key] = (str(mid), btc_open)

    # Load trades
    cur_t = conn_trades.cursor()
    cur_t.execute('''
        SELECT market_slug, side, entry_price, size_usdc, time_remaining_secs,
               pnl, resolution, combined_signal, btc_price_at_entry
        FROM trades WHERE resolution IS NOT NULL ORDER BY timestamp
    ''')
    trades_raw = cur_t.fetchall()

    # Match trades to markets and collect needed market_ids
    trades = []
    for t in trades_raw:
        slug = t[0]
        unix_ts = int(slug.split('-')[-1])
        dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        key = dt.strftime('%Y-%m-%dT%H:%M:%S')
        info = market_lookup.get(key)
        if info:
            mid, btc_open = info
            market_ids_needed.add(mid)
            trades.append({
                'side': t[1], 'entry_price': t[2], 'size_usdc': t[3],
                'secs_rem': t[4], 'pnl': t[5], 'resolution': t[6],
                'comb_sig': t[7], 'btc_entry': t[8],
                'market_id': mid, 'btc_open': btc_open,
            })

    print(f"  {len(trades)} trades matched to tick data", flush=True)

    # Preload ALL snapshots for matched markets
    # This is the key optimization — one big query instead of thousands
    print(f"  Loading snapshots for {len(market_ids_needed)} markets...", flush=True)

    placeholders = ','.join(['?'] * len(market_ids_needed))
    cur.execute(f'''
        SELECT market_id, seconds_into_window, btc_price,
               price_up, price_down, up_best_bid, down_best_bid
        FROM snapshots
        WHERE market_id IN ({placeholders})
        ORDER BY market_id, seconds_into_window
    ''', list(market_ids_needed))

    # Group by market_id
    ticks_by_market = {}
    count = 0
    for row in cur.fetchall():
        mid = str(row[0])
        if mid not in ticks_by_market:
            ticks_by_market[mid] = []
        ticks_by_market[mid].append(row[1:])  # (sec_into, btc, p_up, p_down, up_bid, dn_bid)
        count += 1

    print(f"  {count:,} snapshots loaded into memory", flush=True)

    conn_trades.close()
    conn_ticks.close()

    return trades, ticks_by_market


def sell_pnl(entry_price, sell_price, size_usdc):
    num_shares = size_usdc / entry_price
    return (sell_price - entry_price) * num_shares


def get_our_prices(side, tick):
    """Extract our side's mid price and best bid from a tick tuple."""
    sec_into, btc, p_up, p_down, up_bid, dn_bid = tick
    if side == "YES":
        price = p_up if p_up else 0
        bid = up_bid if up_bid and up_bid > 0 else price
    else:
        price = p_down if p_down else 0
        bid = dn_bid if dn_bid and dn_bid > 0 else price
    return price, bid


class StopBacktester:
    def __init__(self, trades, ticks_by_market):
        self.trades = trades
        self.ticks = ticks_by_market

    def _run(self, stop_fn):
        """Generic runner. stop_fn(trade, ticks_after_entry) -> sell_price or None."""
        orig_pnl = 0.0
        new_pnl = 0.0
        stops = 0
        helped = 0
        hurt = 0
        saved = 0.0
        cost = 0.0

        for trade in self.trades:
            pnl = trade['pnl']
            orig_pnl += pnl

            ticks = self.ticks.get(trade['market_id'])
            if not ticks:
                new_pnl += pnl
                continue

            entry_sec = 300.0 - trade['secs_rem']

            # Filter ticks to after entry
            sell_price = stop_fn(trade, ticks, entry_sec)

            if sell_price is not None:
                sp = sell_pnl(trade['entry_price'], sell_price, trade['size_usdc'])
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

        return len(self.trades), stops, helped, hurt, orig_pnl, new_pnl, saved, cost

    def signal_flip(self, flip_threshold, min_hold_secs=30):
        """BTC reversal from entry price as signal-flip proxy."""
        def stop_fn(trade, ticks, entry_sec):
            btc_entry = trade['btc_entry']
            side = trade['side']
            if not btc_entry:
                return None
            for tick in ticks:
                sec_into = tick[0]
                if sec_into <= entry_sec:
                    continue
                if sec_into - entry_sec < min_hold_secs:
                    continue
                btc_now = tick[1]
                if not btc_now:
                    continue
                btc_move = btc_now - btc_entry
                if side == "YES" and btc_move < -flip_threshold:
                    _, bid = get_our_prices(side, tick)
                    return bid if bid and bid > 0 else None
                elif side == "NO" and btc_move > flip_threshold:
                    _, bid = get_our_prices(side, tick)
                    return bid if bid and bid > 0 else None
            return None
        return self._run(stop_fn)

    def btc_distance(self, distance_threshold, min_secs_remaining=0):
        """BTC distance from window open price."""
        def stop_fn(trade, ticks, entry_sec):
            btc_open = trade['btc_open']
            side = trade['side']
            if not btc_open:
                return None
            for tick in ticks:
                sec_into = tick[0]
                if sec_into <= entry_sec:
                    continue
                secs_left = 300.0 - sec_into
                if secs_left < min_secs_remaining:
                    break
                btc_now = tick[1]
                if not btc_now:
                    continue
                dist = btc_now - btc_open
                if side == "YES" and dist < -distance_threshold:
                    _, bid = get_our_prices(side, tick)
                    return bid if bid and bid > 0 else None
                elif side == "NO" and dist > distance_threshold:
                    _, bid = get_our_prices(side, tick)
                    return bid if bid and bid > 0 else None
            return None
        return self._run(stop_fn)

    def early_reversal(self, drop_threshold, max_hold_secs):
        """Price drops > threshold from entry within max_hold_secs."""
        def stop_fn(trade, ticks, entry_sec):
            entry_price = trade['entry_price']
            side = trade['side']
            for tick in ticks:
                sec_into = tick[0]
                if sec_into <= entry_sec:
                    continue
                if sec_into > entry_sec + max_hold_secs:
                    break
                price, bid = get_our_prices(side, tick)
                if not price or price <= 0:
                    continue
                if entry_price - price >= drop_threshold:
                    return bid if bid and bid > 0 else None
            return None
        return self._run(stop_fn)

    def no_confirmation(self, wait_secs, require_drop=0.0):
        """Price hasn't improved after wait_secs (optionally requires a drop)."""
        def stop_fn(trade, ticks, entry_sec):
            entry_price = trade['entry_price']
            side = trade['side']
            check_sec = entry_sec + wait_secs
            # Find first tick at or after check time
            for tick in ticks:
                sec_into = tick[0]
                if sec_into < check_sec:
                    continue
                price, bid = get_our_prices(side, tick)
                if not price or price <= 0 or not bid or bid <= 0:
                    return None
                if price <= entry_price - require_drop:
                    return bid
                return None  # price improved, no stop
            return None
        return self._run(stop_fn)


def print_result(label, total, stops, helped, hurt, orig, new, saved, cost):
    pct = stops / total * 100 if total else 0
    h_pct = helped / stops * 100 if stops else 0
    diff = new - orig
    sign = "+" if diff >= 0 else ""
    marker = " <<<" if diff > 0 else ""
    print(f"  {label:<48} | {stops:>4} ({pct:>4.1f}%) | H:{helped:>4} ({h_pct:>4.0f}%) | {sign}${diff:>8.2f}{marker}")
    sys.stdout.flush()


if __name__ == "__main__":
    trades, ticks = load_all_data()
    bt = StopBacktester(trades, ticks)

    # ═══════════════════════════════════════════
    # 1. SIGNAL-FLIP STOP (BTC reversal proxy)
    # ═══════════════════════════════════════════
    print(f"\n{'='*85}")
    print(f"  APPROACH 1: Signal-flip (BTC reversal from entry price)")
    print(f"{'='*85}")
    print(f"  {'Parameters':<48} | {'Stops':>12} | {'Helped':>12} | {'PnL Delta':>12}")
    print(f"  {'-'*48}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for flip in [10, 15, 20, 25, 30, 40, 50, 60, 75]:
        for min_hold in [15, 30, 60]:
            r = bt.signal_flip(flip, min_hold)
            print_result(f"flip=${flip}, hold>{min_hold}s", *r)

    # ═══════════════════════════════════════════
    # 2. BTC DISTANCE-FROM-OPEN STOP
    # ═══════════════════════════════════════════
    print(f"\n{'='*85}")
    print(f"  APPROACH 2: BTC distance-from-open")
    print(f"{'='*85}")
    print(f"  {'Parameters':<48} | {'Stops':>12} | {'Helped':>12} | {'PnL Delta':>12}")
    print(f"  {'-'*48}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for dist in [15, 20, 25, 30, 40, 50, 60, 75, 100]:
        for min_rem in [0, 30, 60, 120]:
            r = bt.btc_distance(dist, min_rem)
            print_result(f"dist=${dist}, min_rem={min_rem}s", *r)

    # ═══════════════════════════════════════════
    # 3. EARLY REVERSAL STOP
    # ═══════════════════════════════════════════
    print(f"\n{'='*85}")
    print(f"  APPROACH 3: Early reversal (price drop from entry)")
    print(f"{'='*85}")
    print(f"  {'Parameters':<48} | {'Stops':>12} | {'Helped':>12} | {'PnL Delta':>12}")
    print(f"  {'-'*48}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for drop in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
        for window in [30, 45, 60, 90, 120, 180]:
            r = bt.early_reversal(drop, window)
            print_result(f"drop=${drop:.2f}, window={window}s", *r)

    # ═══════════════════════════════════════════
    # 4. NO-CONFIRMATION STOP
    # ═══════════════════════════════════════════
    print(f"\n{'='*85}")
    print(f"  APPROACH 4: No-confirmation (price hasn't improved)")
    print(f"{'='*85}")
    print(f"  {'Parameters':<48} | {'Stops':>12} | {'Helped':>12} | {'PnL Delta':>12}")
    print(f"  {'-'*48}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for wait in [30, 45, 60, 90, 120, 150]:
        for req_drop in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]:
            r = bt.no_confirmation(wait, req_drop)
            print_result(f"wait={wait}s, req_drop=${req_drop:.2f}", *r)

    print("\n\nDone.")
