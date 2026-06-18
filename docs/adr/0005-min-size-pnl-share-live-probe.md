# Flat min-size, both-sides, PnL/share-gated live probe (revises the J34 T1 probe)

**Status:** accepted (2026-06-18) — design grilled; to be built via /tdd under /gstack-guard

## Context

The first live K2 run performed terribly versus its paper shadow (22% WR, ~5×
fewer trades). The root cause was **size-starvation** (decisions_BTC J37), *not*
the strategy: on a depleted (~$46) wallet the wallet-proportional sizer skips any
trade whose 5-share exchange minimum exceeds the 4% Kelly ceiling, confining the
live bot to ≤$0.40 cheap losers. With the wallet topped up, we needed a way to
test the only question that matters next — **is the signal profitable in live
execution?** — without the sizing layer contaminating the answer.

A clean-data, size-neutral analysis of 1,872 *paper* K2 trades (per-share PnL, so
Kelly sizing is stripped out) established: the edge is **real but thin and
maker-dependent** (~+3–4c/share under realistic maker fills, surviving the
adverse-selection tax), it is **one-sided in WR terms but not in PnL terms**, and
crucially **49% WR yet net +PnL** — payoff asymmetry makes WR a misleading lens.
The bot's side-selection (+6.74c/sh) roughly doubled a static short bias (+3.38c),
NO was positive in every regime including trending_up, and L1 magnitude
monotonically predicted realised edge. (See progress.md Phase 30, bots.md.)

## Decision

Re-specify the J34 "T1 go-live probe" as a **flat min-size, both-sides,
PnL/share-gated** probe:

- **Flat min-size mode** — a config-toggled mode on `PositionSizer`
  (`strategy/sizing.py`): every entry sizes to the **5-share exchange minimum**
  (`max(5×price, $1)`), bypassing the 4% ceiling **and** the J37 floor>ceiling
  *skip*. It affects **sizing only** — every entry/EV/filter gate, the bankroll
  guard, and streak *halts* are preserved (streak size-reduction is a no-op at
  min size). The existing wallet-proportional sizing stays intact and remains the
  default for every other bot.
- **Both sides, evaluated per-side** — the probe trades NO and YES (YES is *not*
  unprofitable: −0.9c at ask was a fill-at-ask artifact; under the bot's actual
  maker fills YES is +2–3c/share). `side` is recorded on every trade, so the run
  is sliced per-side: **NO PnL/share is the primary go/no-go**; **YES PnL/share is
  an exploratory secondary read**. **Adaptive guard:** if live YES is persistently
  negative past the inconclusive floor (~50+ YES fills), flip to NO-only.
  Implemented as `entry_side_filter: both | NO | YES` (default `both`); the shadow
  mirrors live so the fill-gap is measured per-leg.
- **Success gate = PnL/share, not WR** — passes when live filled PnL/share > 0,
  read with a **day-clustered** t-stat (intraday trades share regime): t ≳ 1.5 =
  pass/continue & scale cautiously; t ≥ 2 = commit more capital. The shadow
  comparison is retained as the *diagnostic* separating execution gap from edge
  decay. **The WR gate is retired.**
- **Run to significance, not a calendar** — the thin edge against ~±50c/share
  variance needs ~150–600 fills for a ~2σ read; the 30-filled-trade floor is now
  only the "inconclusive / too-early" line, never a pass.

## Why

- **Why min-size:** a fixed min size removes the floor>ceiling rejection (J37), so
  the live bot trades the full price range like the shadow at minimal capital
  risk — isolating *edge × execution* from *sizing*. It is the cheapest fair test
  of the live edge.
- **Why PnL/share, not WR:** the clean-data analysis proved WR collapses the
  asymmetric-payoff structure (49% WR, +PnL). The probe exists to test
  *profitability*; PnL/share *is* profitability.
- **Why both sides, not NO-only:** YES is positive under maker fills, just thinner
  and more fill-fragile. Per-side tagging lets one run yield a clean NO read *and*
  a free live YES read (the highest-information question, since YES is the
  uncertain leg) — without forcing a choice or muddying the gate.
- **Why a t-stat, run-to-significance:** a raw "positive over 30 trades" sign is
  pure noise at this edge/variance ratio.

## Alternatives rejected

- **NO-only probe** — cleaner-looking, but discards the live YES data for free;
  per-side evaluation gets the clean NO read anyway.
- **Keep the WR gate** — would mark a *profitable* probe a failure on a WR dip.
- **Keep wallet-proportional sizing live** — re-triggers J37 starvation; the
  thing we are explicitly removing as a confound.
- **Fixed-calendar read-out** — too few fills for significance on a thin edge.

## Consequences

- Supersedes the J34 WR-based T1 gate; CONTEXT.md "T1 go-live probe" updated,
  decisions_BTC J38 records the reversal. Builds via /tdd (failing test first)
  under /gstack-guard; new sizer mode + `entry_side_filter` are config-gated and
  behaviour-neutral for all other bots.
- The probe's headline output is the **live-vs-modelled maker-fill / adverse-
  selection gap** — the venue-level go/no-go that any future strategy on this
  market also has to clear.
