"""Size-neutral edge analysis for K2 paper trades.

Goal: separate "the bot's directional signal has real edge" from
"betting DOWN just happened to pay in late-May/June".

Everything is reported in PER-SHARE PnL (pnl / num_shares) so Kelly sizing is
stripped out and the numbers reflect what FLAT MIN-SIZE trading would realise.

Read-only. Reads data_runtime/bot_k2_l1_floor.db (is_paper=1).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data_runtime" / "bot_k2_l1_floor.db"


def rows() -> list[dict]:
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    out = [
        dict(r)
        for r in c.execute(
            "select side, entry_price, num_shares, pnl, resolution, regime, "
            "oracle_lag_signal, combined_signal, net_ev_per_share "
            "from trades where is_paper=1 and pnl is not null and num_shares>0"
        )
    ]
    c.close()
    return out


def ps(r: dict) -> float:
    """Per-share realised PnL = size-neutral edge (what flat min-size realises)."""
    return r["pnl"] / r["num_shares"]


def summarise(label: str, rs: list[dict]) -> None:
    if not rs:
        print(f"  {label:28s} n=0")
        return
    n = len(rs)
    wins = sum(1 for r in rs if r["pnl"] > 0)
    per_share = [ps(r) for r in rs]
    mean_ps = sum(per_share) / n
    # cents-per-share is easier to read
    print(
        f"  {label:28s} n={n:4d}  WR={wins/n:5.1%}  "
        f"per-share={mean_ps*100:+6.2f}c  total_ps={sum(per_share)*100:+8.1f}c"
    )


def main() -> None:
    rs = rows()
    print(f"DB: {DB.name}   resolved paper trades: {len(rs)}\n")

    print("=== HEADLINE (size-neutral, per-share) ===")
    summarise("ALL", rs)
    summarise("NO / short", [r for r in rs if r["side"] == "NO"])
    summarise("YES / long", [r for r in rs if r["side"] == "YES"])

    print("\n=== BY REGIME (does NO-edge survive every regime?) ===")
    for reg in sorted({r["regime"] for r in rs}):
        sub = [r for r in rs if r["regime"] == reg]
        print(f" regime={reg}")
        summarise("  NO", [r for r in sub if r["side"] == "NO"])
        summarise("  YES", [r for r in sub if r["side"] == "YES"])

    print("\n=== BY L1 / oracle_lag_signal ZONE (does edge concentrate where signal is strong?) ===")
    zones = [
        ("strong short  L1<=-0.4", lambda v: v <= -0.4),
        ("weak  -0.4<L1<0.4", lambda v: -0.4 < v < 0.4),
        ("strong long   L1>=0.4", lambda v: v >= 0.4),
    ]
    for name, pred in zones:
        summarise(name, [r for r in rs if pred(r["oracle_lag_signal"])])

    print("\n=== EDGE CALIBRATION (predicted net_ev_per_share -> realised per-share) ===")
    cal = sorted(
        (r for r in rs if r["net_ev_per_share"] is not None),
        key=lambda r: r["net_ev_per_share"],
    )
    if cal:
        q = len(cal) // 5
        for i in range(5):
            chunk = cal[i * q : (i + 1) * q] if i < 4 else cal[i * q :]
            lo = chunk[0]["net_ev_per_share"]
            hi = chunk[-1]["net_ev_per_share"]
            summarise(f"Q{i+1} pred_ev {lo:+.3f}..{hi:+.3f}", chunk)

    print("\n=== BASELINE: does SIDE-SELECTION beat a static short bias? ===")
    print("    (only cleanly-resolved UP/DOWN rows; early-exit stops excluded)")
    clean = [r for r in rs if r["resolution"] in ("UP", "DOWN")]
    bot_ps = sum(ps(r) for r in clean) / len(clean)

    def no_price(r: dict) -> float:
        return r["entry_price"] if r["side"] == "NO" else 1 - r["entry_price"]

    def always(side: str) -> float:
        tot = 0.0
        for r in clean:
            p = no_price(r) if side == "NO" else 1 - no_price(r)
            won = (side == "NO" and r["resolution"] == "DOWN") or (
                side == "YES" and r["resolution"] == "UP"
            )
            tot += (1 - p) if won else -p
        return tot / len(clean)

    print(f"  clean rows: {len(clean)}")
    print(f"  bot actual side          per-share = {bot_ps*100:+6.2f}c")
    print(f"  always-NO  (static short) per-share = {always('NO')*100:+6.2f}c")
    print(f"  always-YES (static long)  per-share = {always('YES')*100:+6.2f}c")


if __name__ == "__main__":
    main()
