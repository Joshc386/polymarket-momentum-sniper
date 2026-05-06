"""Backtest latency arb with percentage-based unrealised loss cut.

Thesis: the trade exploits latency. If the CLOB reprices in our favour,
smart exit captures the gain. If it DOESN'T, the thesis has failed and
we cut the loss at a controlled percentage rather than holding to
resolution and losing the full entry price.

Tests multiple cut thresholds against real PolyBackTest Pro snapshots.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass

# ── Configuration ──────────────────────────────────────────────────
THRESHOLD_BPS = 3.0
MAX_ENTRY_PRICE = 0.55
MIN_PROFIT_TICKS = 0.01      # Smart exit threshold

INITIAL_BANKROLL = 100.0
MAX_BET_USDC = 5.0
MIN_BET_USDC = 1.0
KELLY_MULT = 0.25
FEE_PCT = 0.02


@dataclass
class Trade:
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
    unrealised_at_exit: float = 0.0


def run_backtest(
    loss_cut_pct: float = 0.0,
    max_hold_secs: float = 0.0,
    label: str = "",
) -> list[Trade]:
    """Run latency arb backtest with percentage-based loss cut.

    Args:
        loss_cut_pct: Cut if unrealised loss exceeds this % of position.
            E.g. 0.50 = cut if bid drops to 50% of entry price.
            0.0 = disabled (hold to resolution).
        max_hold_secs: Exit at current bid if held longer than this.
            0.0 = disabled.
        label: Display label for results.

    Returns:
        List of Trade objects.
    """
    snaps = pd.read_csv("backtest/data/polybacktest_snapshots.csv")
    mkts = pd.read_csv("backtest/data/polybacktest_markets.csv")

    resolution_map = {}
    for _, m in mkts.iterrows():
        resolution_map[m.market_id] = m.winner

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

        # ── Move detection ──
        trade = None
        mkt_row = mkts[mkts.market_id == market_id]
        if len(mkt_row) == 0:
            continue
        start_price = mkt_row.iloc[0].btc_price_start

        for i in range(len(rows)):
            if trade is not None:
                break

            curr = rows.iloc[i]
            prev_price = start_price if i == 0 else rows.iloc[i - 1].btc_price

            if prev_price <= 0:
                continue
            move_bps = ((curr.btc_price - prev_price) / prev_price) * 10000.0

            if abs(move_bps) < THRESHOLD_BPS:
                continue

            direction = "up" if move_bps > 0 else "down"

            if direction == "up":
                entry_price = curr.up_best_ask
                side = "YES"
            else:
                entry_price = curr.down_best_ask
                side = "NO"

            if entry_price <= 0 or entry_price > MAX_ENTRY_PRICE:
                continue

            if bankroll < MIN_BET_USDC:
                continue

            est_win_prob = 0.70
            kelly_frac = max(
                0,
                est_win_prob
                - (1 - est_win_prob) / ((1.0 / entry_price) - 1),
            )
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
            )

        if trade is None:
            continue

        # ── Simulate exit logic ──
        entry_idx = None
        for idx in range(len(rows)):
            if rows.iloc[idx].offset_sec == trade.entry_offset:
                entry_idx = idx
                break

        if entry_idx is None:
            _resolve_at_expiry(trade, resolution)
            bankroll += trade.size_usdc + trade.pnl
            trades.append(trade)
            continue

        exited = False
        for j in range(entry_idx + 1, len(rows)):
            snap = rows.iloc[j]
            elapsed_secs = snap.offset_sec - trade.entry_offset

            if trade.side == "YES":
                current_bid = snap.up_best_bid
            else:
                current_bid = snap.down_best_bid

            if current_bid <= 0:
                continue

            # Unrealised PnL as fraction of position
            unrealised_pct = (current_bid - trade.entry_price) / trade.entry_price

            # ── 1. Smart exit (profit take) ──
            if current_bid > trade.entry_price + MIN_PROFIT_TICKS:
                trade.exit_price = current_bid
                trade.exit_offset = int(snap.offset_sec)
                trade.exit_reason = "smart_exit"
                trade.unrealised_at_exit = unrealised_pct
                _compute_exit_pnl(trade)
                exited = True
                break

            # ── 2. Percentage loss cut ──
            if loss_cut_pct > 0 and unrealised_pct <= -loss_cut_pct:
                trade.exit_price = current_bid
                trade.exit_offset = int(snap.offset_sec)
                trade.exit_reason = f"loss_cut_{int(loss_cut_pct*100)}pct"
                trade.unrealised_at_exit = unrealised_pct
                _compute_exit_pnl(trade)
                exited = True
                break

            # ── 3. Time bail ──
            if max_hold_secs > 0 and elapsed_secs >= max_hold_secs:
                trade.exit_price = current_bid
                trade.exit_offset = int(snap.offset_sec)
                trade.exit_reason = "time_bail"
                trade.unrealised_at_exit = unrealised_pct
                _compute_exit_pnl(trade)
                exited = True
                break

        if not exited:
            _resolve_at_expiry(trade, resolution)

        bankroll += trade.size_usdc + trade.pnl
        trades.append(trade)

    return trades


def _compute_exit_pnl(trade: Trade) -> None:
    gross = (trade.exit_price - trade.entry_price) * trade.num_shares
    fee = gross * FEE_PCT if gross > 0 else 0.0
    trade.pnl = gross - fee


def _resolve_at_expiry(trade: Trade, resolution: str) -> None:
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

    # Bankroll curve
    bankroll = INITIAL_BANKROLL
    peak = bankroll
    max_dd = 0
    min_bankroll = bankroll
    for _, t in df.iterrows():
        bankroll += t.pnl
        if bankroll < min_bankroll:
            min_bankroll = bankroll
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

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
    print(f"  Final bankroll: ${bankroll:.2f}")
    print(f"  Min bankroll:   ${min_bankroll:.2f}")
    print(f"  Max drawdown:   {max_dd*100:.1f}%")

    # Exit reason breakdown
    print(f"\n  Exit reasons:")
    for reason in sorted(df.exit_reason.unique()):
        grp = df[df.exit_reason == reason]
        r_wr = grp.won.mean() * 100
        r_pnl = grp.pnl.sum()
        r_avg = grp.pnl.mean()
        print(f"    {reason:25s}: {len(grp):4d} trades, "
              f"WR={r_wr:.1f}%, PnL=${r_pnl:+.2f}, avg=${r_avg:+.4f}")

    # What happens to trades that WOULD have gone to resolution?
    # (i.e. trades that hit loss cut instead)
    loss_cuts = df[df.exit_reason.str.contains("loss_cut")]
    if len(loss_cuts) > 0:
        # How much did we save vs holding to resolution?
        saved_per_trade = abs(avg_loss) - abs(loss_cuts.pnl.mean()) if avg_loss != 0 else 0
        print(f"\n  Loss cut analysis:")
        print(f"    Avg loss cut PnL: ${loss_cuts.pnl.mean():+.4f}")
        print(f"    vs resolution loss avg: ${avg_loss:+.4f}")

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
        "min_bankroll": min_bankroll,
    }


if __name__ == "__main__":
    results = []

    # ── Baseline: smart exit only (current live behaviour) ──
    trades = run_backtest(loss_cut_pct=0.0, max_hold_secs=0.0)
    results.append(print_results(trades, "Baseline: Smart Exit Only"))

    # ── Hold to resolution (no smart exit, no cuts) ──
    # Simulated by setting min_profit_ticks very high in the function
    # Actually we need a separate flag... let's just note the baseline includes it.

    # ── Percentage loss cuts ──
    for pct in [0.25, 0.30, 0.40, 0.50, 0.60, 0.75]:
        trades = run_backtest(loss_cut_pct=pct, max_hold_secs=0.0)
        results.append(print_results(
            trades, f"Smart Exit + {int(pct*100)}% Loss Cut"
        ))

    # ── Loss cut + time bail combos ──
    for pct, secs in [(0.30, 60), (0.40, 90), (0.50, 90), (0.50, 120)]:
        trades = run_backtest(loss_cut_pct=pct, max_hold_secs=secs)
        results.append(print_results(
            trades, f"{int(pct*100)}% Cut + {secs}s Bail"
        ))

    # ── Summary comparison ──
    print(f"\n\n{'=' * 70}")
    print(f"  SCENARIO COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Scenario':<35s} {'Trades':>6s} {'WR':>6s} {'PnL':>10s} "
          f"{'$/day':>8s} {'AvgW':>8s} {'AvgL':>8s} {'MaxDD':>7s} {'Min$':>6s}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*10} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*7} {'-'*6}")
    for r in results:
        if not r:
            continue
        print(f"  {r['label']:<35s} {r['trades']:>6d} {r['wr']:>5.1f}% "
              f"${r['pnl']:>+8.2f} ${r['pnl_per_day']:>+6.2f} "
              f"${r['avg_win']:>+6.4f} ${r['avg_loss']:>+6.4f} "
              f"{r['max_dd']*100:>5.1f}% ${r['min_bankroll']:>5.0f}")
