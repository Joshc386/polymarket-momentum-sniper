"""Backtest latency arb with defensive exit logic.

Simulates Bot F on historical PolyBackTest Pro snapshots with:
  - Stop-loss: cut if bid drops below entry - max_loss_ticks
  - Time bail: exit at current bid if held > max_hold_secs
  - Trailing stop: once in profit, lock gains if bid drops from peak
  - Smart exit: sell when bid > entry + min_profit_ticks (existing)

Data: 5 snapshots per 5-min window at 30s, 60s, 120s, 180s, 240s.
Move detection: compare btc_price between consecutive snapshots.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field

# ── Configuration ──────────────────────────────────────────────────
# Move detection
THRESHOLD_BPS = 3.0          # Min Binance move to trigger
MIN_ORACLE_GAP_BPS = 1.0     # Oracle (proxy: CLOB mid) must be stale
MAX_ENTRY_PRICE = 0.55       # Only enter when odds are cheap

# Exit parameters to test
MIN_PROFIT_TICKS = 0.01      # Smart exit: sell when bid > entry + this
MAX_LOSS_TICKS = 0.03        # Stop-loss: cut when bid < entry - this
MAX_HOLD_SECS = 90.0         # Time bail: exit if held this long
TRAIL_TICKS = 0.02           # Trailing stop: lock gains if bid drops this much from peak

# Sizing
INITIAL_BANKROLL = 100.0
MAX_BET_USDC = 5.0
MIN_BET_USDC = 1.0
KELLY_MULT = 0.25
FEE_PCT = 0.02               # 2% on profit only


@dataclass
class Trade:
    """Single backtest trade."""
    market_id: int
    entry_time: str
    entry_offset: int
    side: str
    entry_price: float
    exit_price: float = 0.0
    exit_reason: str = ""
    exit_offset: int = 0
    num_shares: float = 0.0
    size_usdc: float = 0.0
    pnl: float = 0.0
    resolution: str = ""
    move_bps: float = 0.0
    peak_bid: float = 0.0


def run_backtest(
    min_profit_ticks: float = MIN_PROFIT_TICKS,
    max_loss_ticks: float = MAX_LOSS_TICKS,
    max_hold_secs: float = MAX_HOLD_SECS,
    trail_ticks: float = TRAIL_TICKS,
    label: str = "",
) -> list[Trade]:
    """Run the latency arb backtest with given exit parameters.

    Returns list of Trade objects.
    """
    snaps = pd.read_csv("backtest/data/polybacktest_snapshots.csv")
    mkts = pd.read_csv("backtest/data/polybacktest_markets.csv")

    # Build resolution lookup
    resolution_map = {}
    for _, m in mkts.iterrows():
        resolution_map[m.market_id] = m.winner  # "Up" or "Down"

    # Group snapshots by market, sorted by offset
    grouped = snaps.sort_values("offset_sec").groupby("market_id")

    trades: list[Trade] = []
    bankroll = INITIAL_BANKROLL

    for market_id, group in grouped:
        rows = group.reset_index(drop=True)
        if len(rows) < 2:
            continue

        resolution = resolution_map.get(market_id)
        if not resolution:
            continue

        # Try to detect a move between consecutive snapshots
        trade = None
        for i in range(len(rows) - 1):
            if trade is not None:
                break  # One trade per window max

            curr = rows.iloc[i]
            prev_price = rows.iloc[i].btc_price if i == 0 else rows.iloc[i - 1].btc_price
            curr_price = curr.btc_price

            # For first snapshot, compare to market start price from mkts
            if i == 0:
                mkt_row = mkts[mkts.market_id == market_id]
                if len(mkt_row) > 0:
                    prev_price = mkt_row.iloc[0].btc_price_start
                else:
                    continue

            # Calculate move in bps
            if prev_price <= 0:
                continue
            move_bps = ((curr_price - prev_price) / prev_price) * 10000.0

            if abs(move_bps) < THRESHOLD_BPS:
                continue

            # Determine direction
            direction = "up" if move_bps > 0 else "down"

            # Check if CLOB is stale (entry price is cheap)
            if direction == "up":
                entry_price = curr.up_best_ask
                side = "YES"
            else:
                entry_price = curr.down_best_ask
                side = "NO"

            if entry_price <= 0 or entry_price > MAX_ENTRY_PRICE:
                continue  # Odds already adjusted

            # Check oracle gap (use CLOB mid as proxy for oracle staleness)
            clob_mid = curr.price_up  # This IS the market-implied prob
            if direction == "up":
                # If Binance moved up, the UP price should be higher but isn't yet
                expected_prob = 0.5 + (abs(move_bps) / 100.0)  # rough
                gap = expected_prob - clob_mid
            else:
                expected_prob = 0.5 - (abs(move_bps) / 100.0)
                gap = clob_mid - expected_prob

            # Size the trade
            if bankroll < MIN_BET_USDC:
                continue

            est_win_prob = 0.70
            kelly_frac = max(0, est_win_prob - (1 - est_win_prob) / ((1.0 / entry_price) - 1))
            bet_size = min(
                bankroll * kelly_frac * KELLY_MULT,
                MAX_BET_USDC,
                bankroll * 0.5,
            )
            bet_size = max(MIN_BET_USDC, bet_size)
            if bet_size > bankroll:
                continue

            num_shares = bet_size / entry_price
            bankroll -= bet_size

            trade = Trade(
                market_id=market_id,
                entry_time=curr.snapshot_time,
                entry_offset=int(curr.offset_sec),
                side=side,
                entry_price=entry_price,
                num_shares=num_shares,
                size_usdc=bet_size,
                move_bps=abs(move_bps),
                resolution=resolution,
                peak_bid=entry_price,  # Start peak at entry
            )

        if trade is None:
            continue

        # ── Simulate exit logic using subsequent snapshots ──
        entry_idx = None
        for idx in range(len(rows)):
            if rows.iloc[idx].offset_sec == trade.entry_offset:
                entry_idx = idx
                break

        if entry_idx is None:
            # Shouldn't happen, but resolve at expiry
            _resolve_at_expiry(trade, resolution)
            bankroll += trade.size_usdc + trade.pnl
            trades.append(trade)
            continue

        exited = False
        for j in range(entry_idx + 1, len(rows)):
            snap = rows.iloc[j]
            elapsed_secs = snap.offset_sec - trade.entry_offset

            # Get current bid for our side
            if trade.side == "YES":
                current_bid = snap.up_best_bid
            else:
                current_bid = snap.down_best_bid

            if current_bid <= 0:
                continue

            # Track peak bid (for trailing stop)
            if current_bid > trade.peak_bid:
                trade.peak_bid = current_bid

            # ── 1. Stop-loss ──
            if max_loss_ticks > 0 and current_bid < trade.entry_price - max_loss_ticks:
                trade.exit_price = current_bid
                trade.exit_offset = int(snap.offset_sec)
                trade.exit_reason = "stop_loss"
                _compute_exit_pnl(trade)
                exited = True
                break

            # ── 2. Smart exit (profit take) ──
            if current_bid > trade.entry_price + min_profit_ticks:
                trade.exit_price = current_bid
                trade.exit_offset = int(snap.offset_sec)
                trade.exit_reason = "smart_exit"
                _compute_exit_pnl(trade)
                exited = True
                break

            # ── 3. Trailing stop ──
            if trail_ticks > 0 and trade.peak_bid > trade.entry_price:
                if current_bid < trade.peak_bid - trail_ticks:
                    trade.exit_price = current_bid
                    trade.exit_offset = int(snap.offset_sec)
                    trade.exit_reason = "trailing_stop"
                    _compute_exit_pnl(trade)
                    exited = True
                    break

            # ── 4. Time bail ──
            if max_hold_secs > 0 and elapsed_secs >= max_hold_secs:
                trade.exit_price = current_bid
                trade.exit_offset = int(snap.offset_sec)
                trade.exit_reason = "time_bail"
                _compute_exit_pnl(trade)
                exited = True
                break

        # If no early exit triggered, resolve at expiry
        if not exited:
            _resolve_at_expiry(trade, resolution)

        bankroll += trade.size_usdc + trade.pnl
        trades.append(trade)

    return trades


def _compute_exit_pnl(trade: Trade) -> None:
    """Compute PnL for early exit (sell shares at bid)."""
    gross = (trade.exit_price - trade.entry_price) * trade.num_shares
    fee = gross * FEE_PCT if gross > 0 else 0.0
    trade.pnl = gross - fee


def _resolve_at_expiry(trade: Trade, resolution: str) -> None:
    """Compute PnL for hold-to-resolution."""
    trade.exit_reason = "resolution"
    trade.resolution = resolution
    won = (
        (trade.side == "YES" and resolution == "Up")
        or (trade.side == "NO" and resolution == "Down")
    )
    if won:
        profit = (1.0 - trade.entry_price) * trade.num_shares
        fee = profit * FEE_PCT
        trade.pnl = profit - fee
    else:
        trade.pnl = -trade.entry_price * trade.num_shares


def print_results(trades: list[Trade], label: str = "") -> dict:
    """Print backtest summary and return stats dict."""
    if not trades:
        print(f"  {label}: No trades")
        return {}

    df = pd.DataFrame([t.__dict__ for t in trades])
    df["won"] = df.pnl > 0
    df["date"] = pd.to_datetime(df.entry_time).dt.date

    total_pnl = df.pnl.sum()
    n_trades = len(df)
    wr = df.won.mean() * 100
    avg_win = df[df.won].pnl.mean() if df.won.any() else 0
    avg_loss = df[~df.won].pnl.mean() if (~df.won).any() else 0
    n_days = df.date.nunique()

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  Trades: {n_trades} over {n_days} days ({n_trades/n_days:.1f}/day)")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Total PnL: ${total_pnl:+.2f}")
    print(f"  PnL/day: ${total_pnl/n_days:+.2f}")
    print(f"  Avg win:  ${avg_win:+.4f}")
    print(f"  Avg loss: ${avg_loss:+.4f}")

    if avg_loss != 0:
        print(f"  Win/Loss ratio: {abs(avg_win/avg_loss):.2f}x")

    # Bankroll curve
    bankroll = INITIAL_BANKROLL
    peak = bankroll
    max_dd = 0
    for _, t in df.iterrows():
        bankroll += t.pnl
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak
        if dd > max_dd:
            max_dd = dd

    print(f"  Final bankroll: ${bankroll:.2f}")
    print(f"  Max drawdown: {max_dd*100:.1f}%")

    # Exit reason breakdown
    print(f"\n  Exit reasons:")
    for reason, grp in df.groupby("exit_reason"):
        r_wr = grp.won.mean() * 100
        r_pnl = grp.pnl.sum()
        r_avg = grp.pnl.mean()
        print(f"    {reason:15s}: {len(grp):4d} trades, "
              f"WR={r_wr:.1f}%, PnL=${r_pnl:+.2f}, avg=${r_avg:+.4f}")

    # Daily breakdown (last 10 days)
    daily = df.groupby("date").agg(
        trades=("pnl", "count"),
        wins=("won", "sum"),
        pnl=("pnl", "sum"),
    )
    daily["wr"] = daily.wins / daily.trades * 100

    print(f"\n  Daily PnL (last 10 days):")
    for date, row in daily.tail(10).iterrows():
        print(f"    {date}: {int(row.trades):3d} trades, "
              f"WR={row.wr:.0f}%, PnL=${row.pnl:+.2f}")

    return {
        "label": label,
        "trades": n_trades,
        "wr": wr,
        "pnl": total_pnl,
        "pnl_per_day": total_pnl / n_days,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_dd": max_dd,
        "final_bankroll": bankroll,
    }


if __name__ == "__main__":
    results = []

    # ── Scenario 1: Current (smart exit only, no stop-loss) ──
    trades = run_backtest(
        min_profit_ticks=0.01,
        max_loss_ticks=0.0,      # No stop-loss
        max_hold_secs=0.0,       # No time bail
        trail_ticks=0.0,         # No trailing stop
        label="Current: Smart Exit Only",
    )
    results.append(print_results(trades, "Current: Smart Exit Only"))

    # ── Scenario 2: Hold to resolution (no early exits at all) ──
    trades = run_backtest(
        min_profit_ticks=99.0,   # Never triggers
        max_loss_ticks=0.0,
        max_hold_secs=0.0,
        trail_ticks=0.0,
        label="Hold to Resolution Only",
    )
    results.append(print_results(trades, "Hold to Resolution Only"))

    # ── Scenario 3: Defensive exits (proposed parameters) ──
    trades = run_backtest(
        min_profit_ticks=0.01,
        max_loss_ticks=0.03,
        max_hold_secs=90.0,
        trail_ticks=0.02,
        label="Defensive: SL=0.03, Bail=90s, Trail=0.02",
    )
    results.append(print_results(trades, "Defensive: SL=0.03, Bail=90s, Trail=0.02"))

    # ── Scenario 4: Tighter stop-loss ──
    trades = run_backtest(
        min_profit_ticks=0.01,
        max_loss_ticks=0.02,
        max_hold_secs=60.0,
        trail_ticks=0.01,
        label="Tight: SL=0.02, Bail=60s, Trail=0.01",
    )
    results.append(print_results(trades, "Tight: SL=0.02, Bail=60s, Trail=0.01"))

    # ── Scenario 5: Wider stop-loss ──
    trades = run_backtest(
        min_profit_ticks=0.01,
        max_loss_ticks=0.05,
        max_hold_secs=120.0,
        trail_ticks=0.03,
        label="Wide: SL=0.05, Bail=120s, Trail=0.03",
    )
    results.append(print_results(trades, "Wide: SL=0.05, Bail=120s, Trail=0.03"))

    # ── Scenario 6: Stop-loss only (no trail, no time bail) ──
    trades = run_backtest(
        min_profit_ticks=0.01,
        max_loss_ticks=0.03,
        max_hold_secs=0.0,
        trail_ticks=0.0,
        label="SL Only: SL=0.03",
    )
    results.append(print_results(trades, "SL Only: SL=0.03"))

    # ── Scenario 7: Time bail only ──
    trades = run_backtest(
        min_profit_ticks=0.01,
        max_loss_ticks=0.0,
        max_hold_secs=90.0,
        trail_ticks=0.0,
        label="Time Bail Only: 90s",
    )
    results.append(print_results(trades, "Time Bail Only: 90s"))

    # ── Summary comparison ──
    print(f"\n\n{'=' * 70}")
    print(f"  SCENARIO COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Scenario':<45s} {'Trades':>6s} {'WR':>6s} {'PnL':>10s} "
          f"{'$/day':>8s} {'MaxDD':>7s} {'Final$':>8s}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*10} {'-'*8} {'-'*7} {'-'*8}")
    for r in results:
        if not r:
            continue
        print(f"  {r['label']:<45s} {r['trades']:>6d} {r['wr']:>5.1f}% "
              f"${r['pnl']:>+8.2f} ${r['pnl_per_day']:>+6.2f} "
              f"{r['max_dd']*100:>5.1f}% ${r['final_bankroll']:>7.2f}")
