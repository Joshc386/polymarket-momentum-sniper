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
- **EV** (`best_ev`) = `(q − p) − fee_per_share(p)`. Dollar expected value per
  share. Mostly display, but `best_ev > 0` is also a gate condition. TUI "EV".
  Since 2026-06-03 computed with the **real Polymarket taker fee**
  `0.07·p·(1−p)` (charged at entry on every trade) — same as `net_ev_per_share`.
- **Fee model (unified 2026-06-03):** the live PnL resolver AND the entry-gate
  EV both use the canonical Polymarket taker fee `fee_per_share(p) = 0.07·p·(1−p)`
  per share (single helper in `feature_snapshot.py`, `FEE_RATE = 0.07`).
  `net_ev_per_share = (q − p) − 0.07·p·(1−p)`. The old 2%-on-winnings model was
  removed from both sites (see J17). Backtests still carry a separate
  `fee_adjustment` knob (low-fidelity harnesses, out of scope).
- **Recording note:** the `trades.edge` column historically stores `best_ev`
  (the EV), NOT `prob_edge`. Enrichment adds `prob_edge` and `net_ev_per_share`
  as new, correctly-named columns; the legacy `edge` column is left unchanged.

### Operational safety (kill switch — planned, see ADR-0002)
- **Kill switch** — an emergency stop that runs as a **separate OS process** from
  the trading bot, so it works even when the trading loop is hung. Cancels all
  resting orders, flattens open positions, and disables trading. Manual
  (`python -m tools.kill_switch`) or auto-fired by the watchdog.
- **Heartbeat** — `data_runtime/heartbeat.json`, written by the bot every loop
  iteration: `{ts, window_end_ts, token_ids}`. Liveness signal + the immutable
  `window_end_ts` the flatten guard reads.
- **Watchdog** — a separate, OS-supervised process that polls the heartbeat and
  auto-fires the kill switch when it goes stale (fail-safe: stale/unreadable =
  fire). The bot is *not* auto-restarted; the watchdog is.
- **HALT flag** — `data_runtime/HALT`, sticky. Presence = trading disabled. The
  bot checks it before every order and exits when set. **Cleared only by a
  human** — resuming live trading requires explicit acknowledgement.
- **Flatten** — force-exit open positions via an aggressive marketable sell that
  accepts partial fills, but only if **>60s to resolution** (else let the 5-min
  market resolve). 5-minute auto-resolution is the hard backstop.

### Execution policy vocabulary (maker fills — J26/J27)
The live entry is a **GTC round**: a maker limit posted at the touch (the bid),
cancelled after 10s if unfilled. Every validated entry happens in the **maker
regime** (>60s remaining); below the 60s entry floor the bot does not enter.

- **Miss** — a GTC round that times out unfilled. *Not* a dropped trade: the
  bot re-evaluates next tick and may post again.
- **Re-post loop** — the emergent live behaviour: cancel on timeout, post a new
  round at the then-current touch, repeat until fill or the 60s floor. Captures
  most misses eventually, as a maker, at the new touch.
- **Adverse selection** (maker) — a resting bid fills precisely when the side
  weakens, and systematically misses the runners. Empirical signature: missed
  rounds win far more often than filled ones (~0.62–0.92 vs ~0.52–0.53).
- **Chase / taker fallback** — crossing the spread at the repriced ask (paying
  the taker fee) to capture a miss instead of re-posting. +EV vs miss=drop, but
  only ~+16–28% vs the re-post loop — adoption deferred pending real fill
  telemetry (J27).
- **Hybrid (chase-at-floor)** — candidate policy: free maker re-post rounds,
  then one chase at the 60s floor for whatever is still unfilled. Computable
  from the same T1 telemetry; unmeasured.

### Bots (recording-relevant)
- **Bot G** (`bot_g_signal_aligned`) — **DECOMMISSIONED / dead** (2026-06-01).
  No longer trading. Historical data retained.
- **Bot K** (`bot_k_sm_confirmation`) — maker variant, paper. **Go-live
  candidate; NOT frozen** (may still change before/at live deployment).
- **Bot K2** (`bot_k2_l1_floor`) — Bot K + directional L1 floor, paper A/B.

No bot is "frozen" for behaviour right now. The relevant constraint is narrower:
**a logging/instrumentation change must not alter any entry/sizing/filter
decision** — adding logging columns is behaviour-neutral. (Deliberate,
user-approved behaviour changes — like the 2026-06-03 fee correction inside
`best_ev` — are fine; the rule is "no *accidental* behaviour change from a
logging edit", not "never change the gate".)
