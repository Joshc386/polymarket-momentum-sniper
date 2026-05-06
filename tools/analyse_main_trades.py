"""Quick analysis of main.py trade performance."""
import sqlite3
import pandas as pd

db = sqlite3.connect("data_runtime/trades.db")
df = pd.read_sql("SELECT * FROM trades WHERE pnl IS NOT NULL ORDER BY timestamp DESC", db)

print(f"Total trades: {len(df)}")
print(f"Date range: {df.timestamp.min()} to {df.timestamp.max()}")

df["ts"] = pd.to_datetime(df["timestamp"])
df["date"] = df["ts"].dt.date
df["hour"] = df["ts"].dt.hour
df["dow"] = df["ts"].dt.dayofweek
df["is_weekend"] = df["dow"] >= 5
df["won"] = df["pnl"] > 0

print("\n" + "=" * 70)
print("  DAILY P&L BREAKDOWN")
print("=" * 70)
daily = df.groupby("date").agg(
    trades=("pnl", "count"),
    wins=("won", "sum"),
    pnl=("pnl", "sum"),
).reset_index()
daily["wr"] = daily["wins"] / daily["trades"] * 100

for _, row in daily.iterrows():
    dow = pd.Timestamp(row["date"]).day_name()[:3]
    marker = " <<< WEEKEND" if pd.Timestamp(row["date"]).dayofweek >= 5 else ""
    print(f"  {row['date']} ({dow}): {int(row.trades):3d} trades, "
          f"WR={row.wr:.0f}%, PnL=${row.pnl:+.2f}{marker}")

print(f"\n  Total: ${daily.pnl.sum():.2f}")

# Cumulative PnL over time
daily["cum_pnl"] = daily["pnl"].cumsum()
print("\n  Cumulative PnL curve:")
for _, row in daily.iterrows():
    print(f"    {row['date']}: ${row['cum_pnl']:+.2f}")

print("\n" + "=" * 70)
print("  WEEKEND vs WEEKDAY")
print("=" * 70)
for label, subset in [("Weekday", df[~df.is_weekend]), ("Weekend", df[df.is_weekend])]:
    if len(subset) == 0:
        continue
    print(f"  {label}: {len(subset)} trades, WR={subset.won.mean()*100:.1f}%, "
          f"PnL=${subset.pnl.sum():.2f}, avg=${subset.pnl.mean():.4f}/trade")

print("\n" + "=" * 70)
print("  LAST 3 DAYS DETAIL")
print("=" * 70)
recent_dates = sorted(df["date"].unique())[-3:]
recent = df[df["date"].isin(recent_dates)].copy()
print(f"  Trades: {len(recent)}")
print(f"  WR: {recent.won.mean()*100:.1f}%")
print(f"  PnL: ${recent.pnl.sum():.2f}")
print(f"  Avg entry: ${recent.entry_price.mean():.3f}")
print(f"  Avg PnL/trade: ${recent.pnl.mean():.4f}")

for side in ["YES", "NO"]:
    s = recent[recent.side == side]
    if len(s) > 0:
        print(f"  {side}: {len(s)} trades, WR={s.won.mean()*100:.1f}%, PnL=${s.pnl.sum():.2f}")

if "regime" in recent.columns:
    print(f"\n  By regime (last 3 days):")
    for regime in sorted(recent["regime"].fillna("unknown").unique()):
        r = recent[recent["regime"] == regime]
        print(f"    {regime}: {len(r)} trades, WR={r.won.mean()*100:.1f}%, PnL=${r.pnl.sum():.2f}")

if "combined_signal" in recent.columns and "resolution" in recent.columns:
    recent["signal_dir"] = recent["combined_signal"].apply(lambda x: "UP" if x > 0 else "DOWN")
    recent["signal_correct"] = recent["signal_dir"] == recent["resolution"]
    print(f"\n  Signal accuracy (last 3 days): {recent.signal_correct.mean()*100:.1f}%")

    recent["traded_with_signal"] = (
        ((recent.side == "YES") & (recent.combined_signal > 0)) |
        ((recent.side == "NO") & (recent.combined_signal < 0))
    )
    print(f"  Traded WITH signal: {recent.traded_with_signal.mean()*100:.1f}%")

    with_sig = recent[recent.traded_with_signal]
    against_sig = recent[~recent.traded_with_signal]
    if len(with_sig) > 0:
        print(f"    With: {len(with_sig)} trades, WR={with_sig.won.mean()*100:.1f}%, PnL=${with_sig.pnl.sum():.2f}")
    if len(against_sig) > 0:
        print(f"    Against: {len(against_sig)} trades, WR={against_sig.won.mean()*100:.1f}%, PnL=${against_sig.pnl.sum():.2f}")

# Entry price vs outcome
print("\n" + "=" * 70)
print("  ENTRY PRICE ANALYSIS (last 3 days)")
print("=" * 70)
for lo, hi, label in [(0, 0.45, "<$0.45"), (0.45, 0.55, "$0.45-0.55"), (0.55, 1.0, ">$0.55")]:
    sub = recent[(recent.entry_price >= lo) & (recent.entry_price < hi)]
    if len(sub) < 3:
        continue
    print(f"  {label}: {len(sub)} trades, WR={sub.won.mean()*100:.1f}%, PnL=${sub.pnl.sum():.2f}")

# Profitable period vs losing period comparison
print("\n" + "=" * 70)
print("  PROFITABLE vs LOSING PERIODS")
print("=" * 70)
profitable = daily[daily["pnl"] > 0]
losing = daily[daily["pnl"] <= 0]
if len(profitable) > 0:
    print(f"  Profitable days: {len(profitable)}, avg PnL=${profitable.pnl.mean():.2f}, "
          f"avg WR={profitable.wr.mean():.0f}%")
if len(losing) > 0:
    print(f"  Losing days: {len(losing)}, avg PnL=${losing.pnl.mean():.2f}, "
          f"avg WR={losing.wr.mean():.0f}%")

db.close()
