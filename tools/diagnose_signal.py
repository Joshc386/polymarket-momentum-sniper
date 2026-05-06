"""Diagnose why the bot trades against its own signal in ranging markets."""
import sqlite3
import pandas as pd
import numpy as np

db = sqlite3.connect("data_runtime/trades.db")
df = pd.read_sql("SELECT * FROM trades WHERE pnl IS NOT NULL ORDER BY timestamp DESC", db)
db.close()

df["ts"] = pd.to_datetime(df["timestamp"])
df["date"] = df["ts"].dt.date
df["won"] = df["pnl"] > 0

# Last 3 days only
recent_dates = sorted(df["date"].unique())[-3:]
recent = df[df["date"].isin(recent_dates)].copy()

# Focus on ranging regime
ranging = recent[recent["regime"] == "ranging"].copy()

print("=" * 70)
print(f"  RANGING REGIME DIAGNOSIS ({len(ranging)} trades, last 3 days)")
print("=" * 70)

# Signal values
print(f"\n  Combined signal distribution:")
print(f"    Mean:   {ranging.combined_signal.mean():+.4f}")
print(f"    Median: {ranging.combined_signal.median():+.4f}")
print(f"    Std:    {ranging.combined_signal.std():.4f}")
print(f"    Min:    {ranging.combined_signal.min():+.4f}")
print(f"    Max:    {ranging.combined_signal.max():+.4f}")
print(f"    |signal| < 0.05: {(ranging.combined_signal.abs() < 0.05).sum()} ({(ranging.combined_signal.abs() < 0.05).mean()*100:.0f}%)")
print(f"    |signal| < 0.10: {(ranging.combined_signal.abs() < 0.10).sum()} ({(ranging.combined_signal.abs() < 0.10).mean()*100:.0f}%)")

print(f"\n  Estimated P(Up) distribution:")
print(f"    Mean:   {ranging.estimated_prob_up.mean():.4f}")
print(f"    Median: {ranging.estimated_prob_up.median():.4f}")
print(f"    Std:    {ranging.estimated_prob_up.std():.4f}")
print(f"    Min:    {ranging.estimated_prob_up.min():.4f}")
print(f"    Max:    {ranging.estimated_prob_up.max():.4f}")
pct_near_50 = ((ranging.estimated_prob_up > 0.48) & (ranging.estimated_prob_up < 0.52)).mean() * 100
print(f"    Within 0.48-0.52: {pct_near_50:.0f}%")

print(f"\n  Market implied prob:")
print(f"    Mean:   {ranging.market_implied_prob.mean():.4f}")
print(f"    Median: {ranging.market_implied_prob.median():.4f}")

# Signal direction vs side chosen vs resolution
ranging["signal_says_up"] = ranging["combined_signal"] > 0
ranging["bought_yes"] = ranging["side"] == "YES"
ranging["resolved_up"] = ranging["resolution"] == "UP"

# The critical question: when signal says UP, do we buy YES?
print(f"\n  Signal vs Side alignment:")
ranging["aligned"] = ranging["signal_says_up"] == ranging["bought_yes"]
print(f"    Signal aligned with side: {ranging.aligned.mean()*100:.1f}%")
print(f"    Signal says UP: {ranging.signal_says_up.sum()} times")
print(f"    Bought YES: {ranging.bought_yes.sum()} times")

# When signal says UP, what side did we buy?
sig_up = ranging[ranging.signal_says_up]
sig_down = ranging[~ranging.signal_says_up]
if len(sig_up) > 0:
    print(f"\n    When signal UP ({len(sig_up)} trades):")
    print(f"      Bought YES: {(sig_up.side=='YES').sum()} ({(sig_up.side=='YES').mean()*100:.0f}%)")
    print(f"      Bought NO:  {(sig_up.side=='NO').sum()} ({(sig_up.side=='NO').mean()*100:.0f}%)")
    print(f"      Resolved UP: {sig_up.resolved_up.sum()} ({sig_up.resolved_up.mean()*100:.0f}%)")
if len(sig_down) > 0:
    print(f"\n    When signal DOWN ({len(sig_down)} trades):")
    print(f"      Bought YES: {(sig_down.side=='YES').sum()} ({(sig_down.side=='YES').mean()*100:.0f}%)")
    print(f"      Bought NO:  {(sig_down.side=='NO').sum()} ({(sig_down.side=='NO').mean()*100:.0f}%)")
    print(f"      Resolved DOWN: {(~sig_down.resolved_up).sum()} ({(~sig_down.resolved_up).mean()*100:.0f}%)")

# WHY is entry logic picking the wrong side?
# The entry logic picks side based on: prob_edge_yes = est_prob_up - market_mid_yes
# If est_prob_up > market_mid → buy YES, else buy NO
# So if est_prob_up ≈ 0.50 and market_mid ≈ 0.50, tiny differences decide
print(f"\n  Probability edge analysis:")
ranging["prob_edge_yes"] = ranging["estimated_prob_up"] - ranging["market_implied_prob"]
ranging["entry_should_be_yes"] = ranging["prob_edge_yes"] > 0
print(f"    Prob edge YES mean: {ranging.prob_edge_yes.mean():+.4f}")
print(f"    Prob edge YES std:  {ranging.prob_edge_yes.std():.4f}")
print(f"    |prob edge| < 0.01: {(ranging.prob_edge_yes.abs() < 0.01).sum()} ({(ranging.prob_edge_yes.abs() < 0.01).mean()*100:.0f}%)")
print(f"    |prob edge| < 0.02: {(ranging.prob_edge_yes.abs() < 0.02).sum()} ({(ranging.prob_edge_yes.abs() < 0.02).mean()*100:.0f}%)")
print(f"    |prob edge| < 0.03: {(ranging.prob_edge_yes.abs() < 0.03).sum()} ({(ranging.prob_edge_yes.abs() < 0.03).mean()*100:.0f}%)")

# Individual signal layer values
print(f"\n  Individual signal layers (ranging):")
for col, label in [
    ("oracle_lag_signal", "L1 Oracle"),
    ("momentum_signal", "L2 Momentum"),
    ("liquidation_signal", "L3 Liquidation"),
    ("orderbook_signal", "L4 Orderbook"),
    ("sentiment_signal", "L5 Sentiment"),
]:
    if col in ranging.columns:
        vals = ranging[col].dropna()
        zero_pct = (vals == 0).mean() * 100
        print(f"    {label}: mean={vals.mean():+.4f}, "
              f"std={vals.std():.4f}, zeros={zero_pct:.0f}%")

# Signal weights
if "signal_weights" in ranging.columns:
    import json
    print(f"\n  Signal weights (first 5 trades):")
    for _, row in ranging.head(5).iterrows():
        try:
            w = json.loads(row["signal_weights"])
            print(f"    {w}")
        except Exception:
            print(f"    {row['signal_weights']}")

# The smoking gun: compare entry price to what signal suggests
print(f"\n" + "=" * 70)
print(f"  THE SMOKING GUN")
print(f"=" * 70)
aligned = ranging[ranging.aligned]
misaligned = ranging[~ranging.aligned]
print(f"\n  Trades aligned with signal: {len(aligned)}")
if len(aligned) > 0:
    print(f"    WR: {aligned.won.mean()*100:.1f}%, PnL: ${aligned.pnl.sum():.2f}")
    print(f"    Avg |prob_edge|: {aligned.prob_edge_yes.abs().mean():.4f}")
    print(f"    Avg est_prob_up: {aligned.estimated_prob_up.mean():.4f}")
print(f"\n  Trades AGAINST signal: {len(misaligned)}")
if len(misaligned) > 0:
    print(f"    WR: {misaligned.won.mean()*100:.1f}%, PnL: ${misaligned.pnl.sum():.2f}")
    print(f"    Avg |prob_edge|: {misaligned.prob_edge_yes.abs().mean():.4f}")
    print(f"    Avg est_prob_up: {misaligned.estimated_prob_up.mean():.4f}")
