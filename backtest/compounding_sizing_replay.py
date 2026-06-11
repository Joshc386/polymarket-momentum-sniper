"""Compounding wallet-proportional sizing replay.

Replays each bot's actual resolved trade ledger in chronological order,
scaling every bet by (current wallet / initial wallet) and clamping to a
1%-5% band of the live wallet. Because historical bets were already
clamped to $1-$5 on a $100 bankroll, the scale factor preserves the band
exactly; the clamp is applied anyway as the spec, not an assumption.

Read-only: reads data_runtime/<bot>.db, writes equity CSVs to
data_runtime/sizing_replay/. Never touches bot DBs.

Caveats (also printed with results):
- Recorded sizes contain small intra-session bankroll drift (paper
  bankroll moved with PnL between restarts), so scaling by wallet/100
  slightly double-counts compounding. Second-order.
- Recorded PnL embeds the fee model of its era (real fee model from
  Jun 3). Fees are linear in stake, so proportional scaling is exact.
- EARLY_EXIT trades are included: their PnL is exit-priced but still
  linear in stake.

Usage: python -m backtest.compounding_sizing_replay
"""

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path

INITIAL_BANKROLL = 100.0
FLOOR_PCT = 0.01
CEIL_PCT = 0.05
EXCHANGE_MIN_USDC = 1.0  # Polymarket minimum order

BOTS = {
    "Bot K": "data_runtime/bot_k_sm_confirmation.db",
    "Bot K2": "data_runtime/bot_k2_l1_floor.db",
}

OUT_DIR = Path("data_runtime/sizing_replay")


@dataclass
class TradeRow:
    timestamp: str
    size_usdc: float
    pnl: float
    entry_price: float
    ask_depth: float | None
    resolution: str


@dataclass
class ReplayResult:
    n_trades: int
    final_bankroll: float
    baseline_final: float          # flat sizing: 100 + sum(pnl)
    max_drawdown_pct: float        # compounded equity curve
    baseline_max_drawdown_pct: float
    min_bankroll: float
    max_bankroll: float
    n_below_exchange_min: int      # scaled bets < $1
    n_depth_exceeded: int          # scaled shares > recorded top-of-book ask depth
    n_depth_known: int             # trades with usable depth data
    equity: list[tuple[str, float, float, float]]  # (timestamp, scaled_size, scaled_pnl, bankroll_after)


def load_trades(db_path: str) -> list[TradeRow]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        "SELECT timestamp, size_usdc, pnl, entry_price, ob_ask_depth, COALESCE(resolution, '') "
        "FROM trades WHERE pnl IS NOT NULL ORDER BY timestamp"
    )
    rows = [TradeRow(*r) for r in cur.fetchall()]
    con.close()
    return rows


def replay(
    trades: list[TradeRow],
    initial: float = INITIAL_BANKROLL,
    cap_wallet: float | None = None,
) -> ReplayResult:
    """Replay with wallet-proportional sizing.

    cap_wallet: if set, the sizing wallet is min(bankroll, cap_wallet) —
    i.e. above this wallet the band freezes at [1%, 5%] of cap_wallet
    ($2-$10 at cap 200) instead of growing further.
    """
    bankroll = initial
    peak = initial
    max_dd = 0.0
    min_b, max_b = initial, initial
    below_min = 0
    depth_exceeded = 0
    depth_known = 0
    equity: list[tuple[str, float, float, float]] = []

    # Baseline (flat) drawdown tracked in parallel on the same trades
    flat = initial
    flat_peak = initial
    flat_max_dd = 0.0

    for t in trades:
        sizing_wallet = min(bankroll, cap_wallet) if cap_wallet else bankroll
        factor = sizing_wallet / initial
        scaled = t.size_usdc * factor
        scaled = max(FLOOR_PCT * sizing_wallet, min(CEIL_PCT * sizing_wallet, scaled))
        if scaled < EXCHANGE_MIN_USDC:
            below_min += 1

        scaled_pnl = t.pnl * (scaled / t.size_usdc) if t.size_usdc > 0 else 0.0
        bankroll += scaled_pnl

        if t.entry_price and t.entry_price > 0 and t.ask_depth and t.ask_depth > 0:
            depth_known += 1
            if scaled / t.entry_price > t.ask_depth:
                depth_exceeded += 1

        peak = max(peak, bankroll)
        max_dd = max(max_dd, (peak - bankroll) / peak)
        min_b = min(min_b, bankroll)
        max_b = max(max_b, bankroll)
        equity.append((t.timestamp, round(scaled, 2), round(scaled_pnl, 4), round(bankroll, 2)))

        flat += t.pnl
        flat_peak = max(flat_peak, flat)
        flat_max_dd = max(flat_max_dd, (flat_peak - flat) / flat_peak)

    return ReplayResult(
        n_trades=len(trades),
        final_bankroll=bankroll,
        baseline_final=initial + sum(t.pnl for t in trades),
        max_drawdown_pct=max_dd,
        baseline_max_drawdown_pct=flat_max_dd,
        min_bankroll=min_b,
        max_bankroll=max_b,
        n_below_exchange_min=below_min,
        n_depth_exceeded=depth_exceeded,
        n_depth_known=depth_known,
        equity=equity,
    )


def self_test() -> None:
    """Known-output regression check on a synthetic 3-trade sequence.

    Start $100. Trade 1: $5 bet, +$2 -> wallet 102.
    Trade 2: $5 bet scaled x1.02 = $5.10, pnl +$2 scaled to +$2.04 -> 104.04.
    Trade 3: $1 bet scaled x1.0404, pnl -$1 scaled to -$1.0404 -> 102.9996.
    """
    trades = [
        TradeRow("t1", 5.0, 2.0, 0.5, None, "UP"),
        TradeRow("t2", 5.0, 2.0, 0.5, None, "UP"),
        TradeRow("t3", 1.0, -1.0, 0.5, None, "DOWN"),
    ]
    r = replay(trades)
    assert abs(r.final_bankroll - 102.9996) < 1e-9, r.final_bankroll
    assert abs(r.baseline_final - 103.0) < 1e-9
    assert r.equity[1][1] == 5.10
    # Drawdown: peak 104.04 -> 102.9996
    assert abs(r.max_drawdown_pct - (104.04 - 102.9996) / 104.04) < 1e-12

    # Capped variant: wallet 300 with cap 200 -> factor 2.0, $5 bet -> $10
    capped = [
        TradeRow("t1", 5.0, 100.0, 0.5, None, "UP"),   # 100 -> 200 (factor 1)
        TradeRow("t2", 5.0, 50.0, 0.5, None, "UP"),    # factor 2 -> +100 -> 300
        TradeRow("t3", 5.0, 1.0, 0.5, None, "UP"),     # factor capped at 2 -> +2 -> 302
    ]
    rc = replay(capped, cap_wallet=200.0)
    assert abs(rc.final_bankroll - 302.0) < 1e-9, rc.final_bankroll
    assert rc.equity[2][1] == 10.0  # bet frozen at $10 above $200 wallet


def weekly_snapshots(equity: list[tuple[str, float, float, float]]) -> list[tuple[str, float]]:
    """Last bankroll value of each ISO week, for a compact time-series view."""
    out: dict[str, float] = {}
    for ts, _size, _pnl, bank in equity:
        out[ts[:10]] = bank  # last value per day
    # collapse days -> weekly (every 7th day label kept simple: just return daily lasts)
    days = sorted(out.items())
    return [days[i] for i in range(0, len(days), 7)] + ([days[-1]] if days else [])


def main() -> None:
    self_test()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for label, db in BOTS.items():
        trades = load_trades(db)
        r = replay(trades, cap_wallet=200.0)

        slug = label.lower().replace(" ", "_")
        csv_path = OUT_DIR / f"{slug}_capped200_equity.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "scaled_size_usdc", "scaled_pnl", "bankroll_after"])
            w.writerows(r.equity)

        print(f"\n=== {label} -- 1%-5% wallet sizing, band frozen at $2-$10 above $200 wallet ===")
        print(f"Resolved trades replayed: {r.n_trades}")
        print(f"Final wallet (compounded): ${r.final_bankroll:,.2f}  "
              f"(PnL {r.final_bankroll - INITIAL_BANKROLL:+,.2f})")
        print(f"Final wallet (flat, actual): ${r.baseline_final:,.2f}  "
              f"(PnL {r.baseline_final - INITIAL_BANKROLL:+,.2f})")
        print(f"Wallet range: ${r.min_bankroll:,.2f} .. ${r.max_bankroll:,.2f}")
        print(f"Max drawdown: {r.max_drawdown_pct:.1%} compounded "
              f"vs {r.baseline_max_drawdown_pct:.1%} flat")
        print(f"Scaled bets below $1 exchange min: {r.n_below_exchange_min}")
        if r.n_depth_known:
            print(f"Scaled size exceeded top-of-book ask depth: "
                  f"{r.n_depth_exceeded}/{r.n_depth_known} "
                  f"({100 * r.n_depth_exceeded / r.n_depth_known:.1f}% of trades with depth data)")
        print("Wallet over time (~weekly):")
        for day, bank in weekly_snapshots(r.equity):
            print(f"  {day}  ${bank:,.2f}")
        print(f"Equity curve written to {csv_path}")


if __name__ == "__main__":
    main()
