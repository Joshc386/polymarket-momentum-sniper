# Bot K — Signal-Aligned Maker (fork of Bot G)

## Status
**Design doc — pending approval. No code written yet.**

## One-line summary

Take Bot G's exact signal/filter/sizing logic. Change *only* the execution
path: instead of FOK takes (paying 7.2% crypto fees), post limit orders
inside the spread to qualify for maker fees (0%) + 20% rebate. Run live
with very small bet sizes alongside Bot G; compare PnL to determine if
fee reduction beats opportunity cost from unfilled trades.

## Why this is worth building

Bot G's measured edge: **+$0.16/trade, 1,829 trades, +$292 PnL** (paper).

Bot G pays ~$0.09/trade in taker fees (7.2% × p × (1-p) on $5 bets). If the
maker variant could capture even half of trades as maker:
- Save ~$80 in fees
- Earn ~$20-30 in rebates
- Net improvement: ~$100-110, or **+35% on Bot G's existing edge**

A 35% boost to a working strategy is high-value, especially given Bot G's
edge is thin (2pp over break-even). Reducing costs may be the difference
between "marginally profitable" and "robustly profitable."

## Why live trading instead of paper

Paper trading would need to *simulate* maker fills — "would my limit at
$0.41 have filled?" — by observing trade flow in the next N seconds. This
is doable but error-prone and likely optimistic (we'd assume favourable
fills more often than reality).

Going live with small bet size ($0.50–$1 per trade) gives:
- **Real fill data** — no simulation accuracy issues
- **Real fee/rebate accounting** — actual cost structure
- **Real adverse selection** — if our quotes get picked off, we see it
- **Bounded downside** — at $1/trade, worst-case daily loss is ~$10-20

This is consistent with the project's "treat real money with caution"
principle: caution comes from sizing small until proven, not from
endlessly paper-trading a problem that paper can't simulate.

---

## Architecture

### Reuse from Bot G (no changes)
- `ContrarianEvStrategy` class (signals, filters, regime detection,
  sizing, risk management)
- `RiskManager` (loss caps, streak management)
- All signal layers (L1 oracle lag, L2 momentum, L3 liquidation, etc.)
- The new filter system (`yes_min_price: 0.40`, `skip_regimes: [trending_down]`)

### New components
1. **`MakerExecutionEngine`** — extends `LiveExecutionEngine` with
   post-only-style entry logic (described below)
2. **Order state tracker** — async task watching for fills/timeouts on
   open orders
3. **Bot config: `bot_k_signal_maker`** — enables the maker execution mode

### Configuration
```yaml
bot_k_signal_maker:
  strategy: contrarian_ev
  enabled: true
  db_path: ./data_runtime/bot_k_signal_maker.db
  weekend_flip: true

  # Same filters as Bot G
  filters:
    yes_min_price: 0.40
    skip_regimes: [trending_down]

  # Live execution mode
  execution:
    mode: live_maker          # vs "paper" or "live_taker"
    initial_bankroll: 20.0    # very small — limits worst-case loss
    bet_size_usdc: 1.0        # fixed small size per trade
    maker:
      entry_offset_ticks: 0   # 0 = join best bid; +1 = improve by 1 tick
      max_wait_secs: 30       # cancel if not filled
      fallback: skip          # what to do on timeout: "skip" or "taker"
      max_concurrent_orders: 1 # limit open orders to prevent runaway

  # Same signal config as Bot G
  signals: ...
  regime: ...
  entry: ...
  sizing: ...
  risk: ...
```

### Sizing
Override Bot G's Kelly sizing with **fixed small bet size** (`bet_size_usdc:
1.0`). The goal isn't to maximise EV — it's to validate the maker mechanism
with bounded loss. Once validated (1-2 weeks live), we can tune up sizing.

### Bankroll
Separate USDC funding from any other live trading (none currently). Start
with **$20** — bounds worst-case loss to "noticeable but not painful." If
bankroll drops below $10, auto-halt (existing risk manager handles this).

---

## Maker execution flow

### When the strategy fires (Bot G says "buy YES at $0.42 ask"):

1. **Check best bid for YES** — say it's $0.40
2. **Compute limit price**: `entry_price = best_bid + offset_ticks * tick_size`
   - `offset_ticks: 0` → post at $0.40 (join the queue)
   - `offset_ticks: 1` → post at $0.41 (improve, faster fill, still maker)
   - Validate `entry_price < best_ask` (otherwise it's a taker order)
3. **Submit GTD order** with expiry = `now + max_wait_secs`
   - GTD = Good-Til-Date: rests until filled or expiry
   - This is post-only behaviour by virtue of not crossing the spread
4. **Track order_id and start watching**:
   - WS subscription to user channel for fill events
   - Or polling order status every 1-2s
   - On fill → trade is in (record entry, etc.)
   - On timeout → cancel + apply fallback

### Fallback behaviour on timeout

`fallback: skip` (recommended for v1):
- Cancel the order
- Skip the trade entirely
- Log the missed opportunity for later analysis

`fallback: taker` (more aggressive):
- Cancel the maker order
- Submit FOK at current best ask
- Pay taker fees but capture the trade
- Use only if data shows unfilled trades have positive expectancy

We start with `skip` and decide later from data whether `taker` fallback
is worth it.

### Concurrent order limit

`max_concurrent_orders: 1` — only one open order at a time. If a new
signal fires while a maker order is pending, the new signal is dropped
(or queued with a timeout). Prevents runaway exposure.

### Inventory management

Within a single 5-min window, only one trade per window (existing Bot G
behaviour). The maker variant inherits this. So we can't accumulate
unintended inventory mid-window.

---

## Live trading prerequisites

Before any live order goes out, we need:

1. **Polymarket account** with USDC funded ($20 to start)
2. **API credentials configured** (already in `core/polymarket_client.py`)
3. **Wallet keys for signing** (EIP-712) — needs verification these work
4. **Account balance check** at startup — refuse to start if < bankroll
5. **Kill switch** — independent of bot process; can halt all open orders
6. **Telegram alerts on every trade** (entry, fill, cancel)
7. **Daily loss cap enforced** — hard halt if hit

---

## Phased rollout

### Phase 0 — Live trading infrastructure verification (1 day)
Before writing any maker logic, verify:
- Live order placement works end-to-end (place a tiny GTD, see it on the book, cancel it)
- Order cancellation works
- Fill notifications arrive (via WS user channel)
- Account balance reads correctly
- Kill switch tested

This is "can we even trade live with the existing infrastructure?" If something
is broken, fix that before moving on.

### Phase 1 — Build MakerExecutionEngine (2 days)
- Extend `LiveExecutionEngine` with the maker entry flow
- Order tracking (state machine: PENDING → FILLED / CANCELLED / TIMEOUT)
- Cancel-on-timeout logic
- Telegram notifications for each state transition
- Logging to `bot_k_signal_maker.db` (same schema as Bot G + maker-specific fields)

### Phase 2 — Wire into ContrarianEvStrategy (1 day)
- Add `execution_mode` config gate
- When `live_maker`, route trades through `MakerExecutionEngine`
- When `paper` (default), use `PaperExecutionEngine` as today
- Keeps Bot G unchanged

### Phase 3 — Tiny live deployment (1-2 weeks observation)
- Start bot with `$20` bankroll, `$1/trade`
- Run alongside Bot G (which stays in paper mode)
- **Monitor daily**: fill rate, fee savings, rebates earned, missed trades
- **Halt if**: bankroll drops below $10, or 2+ consecutive days of errors

### Phase 4 — Compare and decide
After 1-2 weeks of clean data:
- **Comparable PnL trajectory to Bot G?** → maker-mode works; scale up sizing
- **Significantly worse than Bot G?** → fee savings don't beat missed trades; abandon
- **Similar but with different volatility?** → look at risk-adjusted metrics

If we promote, scale up gradually: $1 → $5 → $20 over weeks, with risk
management active throughout.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Live order bug → unintended large trade | Low | Hard cap on `bet_size_usdc` in code (not just config); kill switch |
| Maker orders rarely fill | Medium-High | `fallback: taker` option; tune `entry_offset_ticks` |
| Adverse selection on entry | Medium | Signal decay check — re-evaluate signal at fill time, possibly skip if signal decayed |
| API key compromise | Low | Use minimal-permission key; never log it |
| Polygon network issues | Medium | Existing retry logic; halt on >3 consecutive RPC failures |
| Polymarket account ban for misbehaviour | Very Low | Stay well within rate limits; don't try anything fancy |
| Paper Bot G diverges from live Bot K | Expected | This IS the test — divergence tells us the comparison answer |

---

## Locked-in design choices (decided 2026-05-01)

| Question | Decision |
|---|---|
| Bot name | **Bot K** |
| Initial bankroll | **$20** |
| Bet size per trade | **$1** (worst-case ~$50/day) |
| `entry_offset_ticks` | **+1** (improve best bid by one tick, faster fill) |
| Fallback on timeout | **`skip`** (no taker fallback in v1) |
| Cancel on signal flip during wait | **Yes** (open question 6) |

---

## Recommended next step

You answer the 6 open questions, then I:
1. **Phase 0 first** — verify live trading infrastructure works (no maker
   logic yet, just basic order placement / cancellation tests with $0.10
   trades to prove the plumbing)
2. **Phase 0 result determines** whether we proceed with Phase 1+, or fix
   live infrastructure issues first

If Phase 0 reveals existing live infrastructure is broken or incomplete,
we deal with that before any maker logic. Don't want to discover live
order issues *while* trying to debug maker behaviour.
