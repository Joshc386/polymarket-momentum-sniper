# Bot H — Intra-Market Arbitrage Scanner

## Status
**Design doc — pending approval. No code written yet.**

## Thesis

When Polymarket orderbooks briefly show `yes_best_ask + no_best_ask < $1.00`, both
sides can be bought simultaneously for risk-free profit at resolution. One side
resolves to $1.00, the other to $0.00, so the trader nets `$1.00 - (yes_ask + no_ask) - fees`
per share regardless of outcome.

These windows exist because:
- Market makers haven't repriced after a fast move
- One side ran out of liquidity at the tight price
- Briefly inefficient gaps before professional arbitrageurs close them

**Reference paper:** Saguillo et al, "Unravelling the Probabilistic Forest:
Arbitrage in Prediction Markets" (AFT 2025) — claims $40M extracted historically
across 86M bets. Whether the edge is still available is the entire point of this bot.

---

## Why this strategy is fundamentally different from Bots A-G

| Dimension | Bots A-G (directional) | Bot H (arbitrage) |
|---|---|---|
| **Edge type** | Probabilistic (signals) | Mathematical (price discrepancy) |
| **Risk per trade** | Real | Zero (if both sides fill) |
| **Hold to resolution** | Coin flip | Guaranteed positive |
| **Speed required** | Seconds | Milliseconds |
| **Markets monitored** | One at a time (BTC 5-min) | All active markets simultaneously |
| **Execution failure** | Bad fill price | Stuck holding one side (real risk) |

---

## Phase 0 research findings

### Q1: Active market count — **ANSWERED**
- **6,500+ markets** across 1,500+ events currently active
- **3,000+ with meaningful volume** (>$1000)
- Cannot subscribe to all — must filter. See "Market universe" below.

### Q2: Fee structure — **ANSWERED** (critical finding)
Formula: `fee = shares × feeRate × p × (1 - p)`

| Category | Fee rate | Max fee @ p=0.50 | Fee @ p=0.46 |
|---|---|---|---|
| **Geopolitical / World Events** | **0%** | **$0.00** | **$0.00** |
| Sports | 3.0% | $0.0075/share | $0.00745 |
| Finance, Politics, Tech | 4.0% | $0.0100/share | $0.00994 |
| Economics, Culture, Weather | 5.0% | $0.0125/share | $0.01242 |
| Crypto | 7.2% | $0.0180/share | $0.01788 |

**Takeaway:**
- **Geopolitics = 0% fees. Any sum-of-asks < $1.00 is profitable there.**
- Makers pay zero fees + get 20-25% rebates. If we can be maker, free money.
- Crypto 5-min markets have the WORST fees (7.2%) — Bot F was swimming upstream.

### Q3: WebSocket limits — **PARTIALLY ANSWERED**
- Market channel supports multi-market subscription via `assets_ids` array
- No documented subscription limits
- Ping every 10s required
- Dynamic subscribe/unsubscribe without reconnecting
- **Verdict:** WS can handle many markets per connection. Need to empirically
  verify stability with 100+ subscriptions.

### Q4: Arb opportunity duration — **DEFERRED TO PHASE 1**
Not documented, must measure empirically.

### Q5: Typical depth — **DEFERRED TO PHASE 1**
Must measure empirically.

### Q6: Atomic order execution — **ANSWERED** (risk identified)
- **No true atomic execution.** Multi-order endpoint processes "in parallel"
  — one order can succeed while another fails.
- **Max 15 orders per batch request.**
- FOK (Fill-Or-Kill) order type exists — cancels entirely if can't fill
  immediately.
- **Partial-fill risk mitigation:** Submit both sides as FOK. If one fills
  and other cancels, immediately market-sell the filled side at the bid.
  Loss bounded by spread + fees (~$0.02-$0.05 per share).

### Q7: Edge survival in 2026 — **DEFERRED TO PHASE 1**
The paper's data spans a prior period. Phase 1 (read-only) tells us whether
opportunities still exist. Strong priors: liquid crypto arbs are gone,
long-tail geopolitics arbs probably survive longer.

### Additional findings

**Rate limits (generous):**
- `/book` (single): 1,500 req/10s
- `/books` (batch): 500 req/10s
- `POST /order`: 3,500 req/10s burst, 36,000/10min sustained
- `POST /orders` (batch): 1,000 req/10s burst
- Not a bottleneck for our scanner.

**Tick sizes:**
- Standard: $0.01
- Some markets: $0.001 or $0.0001 (more granular = more potential arb windows)

**Order types:**
- FOK — all-or-nothing immediate
- FAK — fill what's available, cancel rest
- GTC — rest on book
- GTD — rest until expiry
- No IOC. FOK is the tool for arb execution.

---

## Profitability math (updated with real fee structure)

Fee = `shares × feeRate × p × (1 - p)`. For an arbitrage trade buying BOTH
sides of the same market at similar prices, fees apply to both legs.

### Scenario: Buy YES + NO both at $0.46, 10 shares each, at resolution 1 side → $1, other → $0

Gross profit per share (before fees) = $1.00 - (yes_ask + no_ask) = $1.00 - $0.92 = $0.08

| Category | Fee rate | Fee per share (both legs) | Net profit/share | Net on 10 shares |
|---|---|---|---|---|
| **Geopolitics** | 0% | $0.0000 | **$0.0800** | **$0.80** |
| Sports | 3% | $0.0149 | $0.0651 | $0.65 |
| Politics/Finance | 4% | $0.0199 | $0.0601 | $0.60 |
| Economics etc | 5% | $0.0249 | $0.0551 | $0.55 |
| Crypto | 7.2% | $0.0358 | $0.0442 | $0.44 |

### Break-even threshold (where net profit = 0)

Worst case (crypto, 7.2%): need `yes_ask + no_ask ≤ $0.964` at p=0.50.
Best case (geopolitics): need `yes_ask + no_ask < $1.00`.

**Implication:** The bot's detection threshold should be **category-aware**.
A $0.97 sum is unprofitable in crypto but profitable in geopolitics.

### Maker rebates (upside we're not counting)

Makers get 20% (crypto) to 25% (other) rebates on the fees paid by takers
on their orders. If we can place resting limit orders rather than FOK,
we flip from fee-payer to fee-receiver. But the arb disappears if we wait
— tradeoff to revisit in Phase 2.

---

## Market universe strategy

With 6,500+ active markets, we can't practically subscribe to all of them.
Even if we could, many are too illiquid to trade or resolve too slowly for
capital efficiency.

### Proposed tiered approach for Phase 1

**Tier 1 (high priority — subscribe to all):**
- All **Geopolitics / World Events** markets (0% fees = best edge)
- Estimated: ~200-300 markets
- These are fee-free, so even a $0.99 sum is profitable

**Tier 2 (medium priority — subscribe to liquid ones):**
- Politics, Finance, Tech, Sports markets with volume > $10k
- Estimated: ~500-1000 markets
- Moderate fees but large opportunity count

**Tier 3 (skip in Phase 1):**
- Crypto (high fees, professional arbitrageurs saturated)
- Markets with near-zero volume (arbs there are untradeable)
- Markets resolving within 60 seconds (fill timing risk)

### Minimum quality filters (all tiers)
- Market is active (`active: true`, `closed: false`, `accepting_orders: true`)
- Time-to-resolution > 5 minutes
- Best ask + best bid exist on both YES and NO (two-sided book)
- Minimum daily volume > $500

### Refresh cadence
- Universe refresh (discover/drop markets): every 5 minutes
- Orderbook via WS: real-time

---

## Architecture overview

Reuse from existing codebase:
- `MarketDiscovery` (extend to find all markets, not just BTC 5-min)
- `polymarket_ws.py` (extend to multiple market subscriptions)
- `PaperExecutionEngine` (with arb-specific dual-order helper)
- `RiskManager` (loss caps, etc.)
- `Database` (per-bot SQLite)
- Multi-bot dashboard

New components:
1. **`MarketUniverse`** — discovers and maintains list of all active markets matching criteria
2. **`MultiMarketBookManager`** — maintains live orderbook state for N markets via WS
3. **`ArbDetector`** — checks every orderbook update for sum-of-asks < threshold
4. **`ArbExecutor`** — atomic dual-order placement (both sides or neither)
5. **`ArbStrategy`** — the bot itself, orchestrates the above

---

## Phased implementation plan

### Phase 0: Research & feasibility (NO CODE YET — answer Q1-Q7 first)

**Deliverable:** A short doc answering each unknown above. Sources:
- Polymarket docs (fees, WS limits, API docs)
- The arb paper itself (re-read sections on detection methodology)
- Their data on opportunity frequency
- Public analysis tools on Polymarket arbitrage activity

**Decision point:** Is this feasible from a Python/asyncio setup, or does it
need a low-level Rust/C++ system? If the latter, we stop here.

### Phase 1: Read-only scanner

Build the minimum viable detector:
- Discover all liquid markets (volume > threshold, time-to-resolution > threshold)
- Subscribe to orderbook WS for as many as feasible
- LOG every moment when sum-of-asks < $1.00
- Do NOT trade, just collect data

**Deliverable:** A SQLite log of detected opportunities with:
- Market slug, condition_id
- Timestamp
- yes_ask, no_ask, sum
- Theoretical profit (assuming we could fill)
- Depth on each side
- How long the opportunity persisted before sum returned to ≥ $1.00

**Run for 3-7 days. Then analyse:**
- How many opportunities per day?
- What's the average and distribution of theoretical profit?
- Which markets generate the most opportunities?
- What's the typical opportunity duration?

**Decision point:** If <10 opportunities/day or theoretical profit too small,
strategy is dead. Move on. If lots of opportunities, proceed to Phase 2.

### Phase 2: Paper trading

Add the execution simulation:
- When opportunity detected, simulate buying both sides at the asks
- Use realistic slippage model (how much depth was actually there?)
- Track simulated PnL with proper fees applied
- Add risk management: max bankroll exposure per arb, daily loss cap

**Deliverable:** A bot that runs alongside Bot G, paper-trading arbs.

**Run for 1-2 weeks. Then analyse:**
- Real PnL vs theoretical PnL (how much did slippage eat?)
- Win rate (any "wins" that turn into losses due to one-side fills?)
- Sharpe ratio
- Comparison to Bot G

**Decision point:** Profitable paper trading? Proceed to Phase 3.

### Phase 3: Live trading

Only after consistent paper profitability over 2+ weeks:
- Start with tiny size ($10-20 per arb)
- Atomic dual-order execution (must figure out how)
- Strict daily loss cap
- Kill switch
- Telegram alerts for every trade

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Edge already competed away** | Phase 1 (read-only) will reveal this in days |
| **Stuck holding one side after partial fill** | Phase 0: must determine if atomic execution is possible. If not, immediate sell of the filled side at any price (cap loss) |
| **Slippage eats profit** | Use depth-aware sizing. Don't take more than X% of best-ask depth |
| **Polymarket WS rate limits** | Phase 0: identify limits, plan subscription strategy |
| **Same markets used by Bot F/G** | Should be fine — Bot H is order-book-based, F/G are strategy-based. But coordinate on bankroll if shared |
| **Low frequency of arbs in liquid markets** | Phase 1 data tells us. May need to focus on illiquid markets where arbs persist longer |
| **Real fees higher than expected** | Fee buffer in detection threshold. Tighten if first paper trades are unprofitable |

---

## Multi-agent / parallelisation question

You raised the point that one bot can't effectively scan all markets. My take:

**For Phase 1 (read-only scanner):** Asyncio with multiple WS connections is enough.
Python can comfortably handle 50-100 concurrent WS streams. We don't need
multiple processes/agents.

**For Phase 2/3 (with execution):** Still single-process is fine, but with clear
async task separation. We don't want to introduce inter-process complexity
unless we hit a real bottleneck.

**True multi-agent (multiple processes) only if:**
- Polymarket WS limits force us to distribute across IPs
- Single Python process can't keep up with order book churn (unlikely for
  arbitrage scanning)
- We want geographic distribution for latency

This is an optimisation we can defer. Start single-process, scale only if needed.

---

## Resolved design decisions

1. **Bankroll:** Separate bankroll for Bot H. Measured independently from Bot F/G.

2. **Universe of markets:** Scan everything. Rationale: crypto markets are too
   liquid and too heavily arbitraged to find gaps. The edge more likely lives
   in lower-volume markets (sports, politics, long-tail events) where market
   makers are slower. We'll apply a minimum time-to-resolution filter only
   (markets that resolve in < 60 seconds are too risky — fill timing matters).

3. **Per-arb sizing:** Depth-based. Take no more than X% of best-ask depth
   on the thinner side, capped by a max bankroll fraction per arb.
   Exact parameters TBD in Phase 1 once we see typical depth distributions.

4. **Phase 1 pass threshold:** ≥20 opportunities/day with theoretical profit
   ≥$0.05/share at fillable depth (after fees). Below this → retire strategy.

5. **Database:** New arb-specific schema. Separate tables for:
   - `arb_opportunities` (every detected gap, whether we traded or not)
   - `arb_trades` (paper/live trades executed)
   - `market_snapshots` (periodic orderbook state for context)

---

## Recommended next step

**Phase 0 research first.** Specifically, I want to fetch:
1. Polymarket fee documentation
2. CLOB API documentation (especially atomic order capabilities)
3. WS API limits

Should I do that research now, or do you want to discuss/modify this plan first?
