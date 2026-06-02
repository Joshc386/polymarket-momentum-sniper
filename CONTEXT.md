# CONTEXT — Polymarket Momentum Sniper Bot

Shared vocabulary between the developer and the agent. Domain language only —
not implementation detail. See `CLAUDE.md` for workflow/standards and the
Obsidian vault (`5min BTC Bot/`) for the project's living history.

## Glossary

### Signal layers (L1–L12)
The ordered set of edge sources the bot blends into a directional view. Each
layer outputs a scalar in roughly [-1, +1] where positive = bullish (UP/YES),
negative = bearish (DOWN/NO).

- **L1 — oracle lag** (`oracle_lag_signal`). Legacy name. Since J9 it is BTC's
  *displacement from the window-open resolution line* (no live oracle). Has two
  sub-components: `l1_lag_component`, `l1_open_component`.
- **L2 — momentum** (`momentum_signal`). Dominant signal per walk-forward opt.
- **L3 — liquidation proximity** (`liquidation_signal`). Empirically weak/anti-
  predictive; under investigation.
- **L4 — orderbook** (`orderbook_signal`). Sub-components: `l4_imbalance`,
  `l4_flow`, `l4_mid_dev`, `l4_top_pressure`, `l4_thickness`.
- **L5 — sentiment** (`sentiment_signal`). Cross-exchange direction.
- **L6–L12** — additive/experimental layers: `l6_fade`, `l7_taker_ratio`,
  `l8_clob_flow`, `l9b_absorption`, `l10_exhaustion`, `l11_trade_size`,
  `l12_wallet_flow`. Not all are active in every bot.

### Feature snapshot
The full vector of signal state at one instant: every L1–L12 value, all
sub-components, the regime, the weights in force, BTC price, market odds,
seconds into the window, and the derived edge/EV. The canonical record of
"what the bot saw."

### Two recording stores (kept deliberately separate)
- **Per-tick signal-diagnostic stream** (`<bot>_signal_diag.db`, table
  `signal_ticks`). One row per tick (~2/sec), every tick whether or not a trade
  fired. Carries `would_enter`, `filter_blocked`, `trade_placed`. This is the
  **counterfactual / selection-bias dataset** — what the bot saw when it chose
  *not* to trade.
- **Enriched trade record** (`<bot>.db`, table `trades`). One row per actual
  entry, carrying the full feature snapshot **plus the outcome** (`resolution`,
  `pnl`). This is the **outcome-labelled training set** for analysis and future
  RL. Stored as flat columns (one per feature) for direct SQL/pandas querying.

Both stores are populated from a **single shared feature-snapshot builder** so
their feature definitions can never drift apart. (Drift between two copies of
the snapshot logic is exactly what caused the May-2026 silent logging bug.)

**Snapshot-at-entry principle:** the enriched trade record captures *everything
the bot saw at the instant of entry* — every L1–L12 value and sub-component, the
full weight vector in force (not just the core five), regime, BTC price, odds,
`prob_edge`, `net_ev_per_share`, seconds into the window. One row fully
reconstructs the decision. A value is recorded only when its layer genuinely
computed on that tick; otherwise it is `NULL` (never `0.0`) so "absent" is never
confused with "neutral". Active/absent is determined in the snapshot builder
(layer object present AND its data-presence guard satisfied), decoupled from the
`0.0` the combiner consumes.

### Edge vs EV (distinct — both shown in the TUI)
- **Edge** (`prob_edge`) = `|est_prob_up − market_prob_up|`. Model-vs-market
  probability disagreement. The **entry gate** ("the TRUE edge metric" — price-
  level-normalised). TUI "Edge".
- **EV** (`best_ev`) = `q·(1−p)·(1−fee) − (1−q)·p`. Dollar expected value per
  share. Mostly display, but `best_ev > 0` is also a gate condition. TUI "EV".
  Currently computed with the **2% winner fee** (`fee_adjustment: 0.02`).
- **Fee models disagree:** the live EV/PnL path uses 2%-on-winnings; backtests
  (and Polymarket docs) use `0.072·p·(1−p)` per share. The latter is canonical
  for analysis. `net_ev_per_share = (q − p) − 0.072·p·(1−p)`.
- **Recording note:** the `trades.edge` column historically stores `best_ev`
  (the EV), NOT `prob_edge`. Enrichment adds `prob_edge` and `net_ev_per_share`
  as new, correctly-named columns; the legacy `edge` column is left unchanged.

### Bots (recording-relevant)
- **Bot G** (`bot_g_signal_aligned`) — **DECOMMISSIONED / dead** (2026-06-01).
  No longer trading. Historical data retained.
- **Bot K** (`bot_k_sm_confirmation`) — maker variant, paper. **Go-live
  candidate; NOT frozen** (may still change before/at live deployment).
- **Bot K2** (`bot_k2_l1_floor`) — Bot K + directional L1 floor, paper A/B.

No bot is "frozen" for behaviour right now. The relevant constraint is narrower:
**a logging/instrumentation change must not alter any entry/sizing/filter
decision** — e.g. don't change the fee model inside `best_ev` because
`best_ev > 0` gates entry. Adding logging columns is behaviour-neutral.
