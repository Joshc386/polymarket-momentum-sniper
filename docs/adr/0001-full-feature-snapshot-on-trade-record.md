# Full feature snapshot on every trade record

**Status:** accepted (2026-06-01)

## Decision

Each trade row in `<bot>.db / trades` captures a complete snapshot of everything
the bot saw at the instant of entry — every L1–L12 layer value and sub-component,
the full weight vector in force, regime, BTC price, market odds, `prob_edge`,
`net_ev_per_share`, and seconds into the window — stored as **flat columns**,
with `NULL` (never `0.0`) for any layer that did not genuinely compute that tick.
A **single shared feature-snapshot builder** populates both the per-trade record
and the per-tick `signal_diag` stream so the two can never drift apart.

## Why

Before this, the trade record kept only L1–L5 + combined signal, and the
sub-layer inputs (L1/L4 components, L6–L12) lived only in a per-tick
`signal_diag` stream — which had silently stopped writing for ~10 days because a
copy of the snapshot logic referenced an out-of-scope variable (`btc_ref_price`)
whose `NameError` was swallowed by best-effort logging. The result: no
self-contained, outcome-labelled record of why any trade was taken, and a
firehose that had to be time-aligned to trades to be useful. A complete
snapshot-at-entry gives one queryable row per trade (feature vector + outcome),
which is the prerequisite for honest trade analysis and any future RL.

## Considered and rejected

- **JSON blob column** instead of flat columns — rejected; the analysis stack is
  all `pd.read_sql` + column filters, and an opaque blob is easy to fill with
  silent garbage (the exact failure we're fixing).
- **Single store** (derive the per-trade dataset by joining `trade_placed=1`
  ticks to outcomes) — rejected; leaves training data in a per-tick DB needing
  post-processing. Both stores kept: `signal_diag` is the counterfactual /
  selection-bias dataset (every tick, including non-entries); `trades` is the
  clean labelled training set.
- **`0.0` for inactive layers** — rejected; indistinguishable from a genuine
  neutral reading and silently poisons feature-importance work.

## Consequences

- The `trades` table roughly triples in width (~20 new columns). Migrations are
  idempotent `ALTER TABLE ADD COLUMN`; rows predating the change keep `NULL` in
  the new columns (no backfill is possible — the vectors were never recorded).
- The legacy `edge` column continues to store `best_ev` (2% fee model); the new
  `prob_edge` and `net_ev_per_share` (Polymarket-doc fee `0.072·p·(1−p)`) are the
  correctly-named replacements. `best_ev` and the entry gate are left untouched
  because `best_ev > 0` gates entry and changing it would alter trade selection.
- New signal layers will require an explicit column migration — accepted as
  healthy discipline for a money-adjacent schema.
