# @polybacktest Research Findings

Source: https://x.com/polybacktest (April 2026)
Analysed: 2026-04-02

---

## Tweet 1 — Oracle Lag Reality Check
**Sample:** 2,194 five-minute markets, per-second Binance data

- Binance-to-Chainlink lag is real — 94.7% accuracy at 5s remaining
- But the actual exploitable gap is sub-second — bots already sit in it
- Last 5 seconds of NEW price movement alone: 52.3% (coin flip)
- High accuracy comes from moves that already happened minutes ago — already priced in
- When BTC moved >0.1% with 10s left: 100% accurate, zero misses

**Takeaway:** Oracle lag is real but the sub-second gap is saturated by bots. Our edge is in the 1-4 minute window where the move is developing but hasn't fully priced in.

---

## Tweet 2 — Order Book Imbalance is CONTRARIAN (Highest Alpha)
**Sample:** 1,023 markets, 3 million snapshots

The heavier the bid imbalance on one side, the MORE LIKELY that side LOSES:

| Imbalance Level | Follow WR | Fade WR |
|---|---|---|
| Mild (0.1-0.2) | 49.0% | 51.0% |
| Moderate (0.3-0.5) | 47.3% | 52.7% |
| Strong (0.7-1.0) | 42.3% | 57.7% |
| Extreme (1.5+) | 25.3% | **74.7%** |

After 4 minutes, heavy-bid side wins only 31.5%.

**Why:** Stale bids on the losing side are unfilled limits from when the price was different. Smart money hits the ask (lifts offers), it doesn't sit in the book. Heavy bid imbalance = trapped liquidity, not conviction.

Price itself is the best predictor: >70c = 74% WR, >90c = 94.4%.

**Potential Strategy 2:** Fade extreme order book imbalance. Buy the cheap opposite side when imbalance >0.7. Higher imbalance = stronger signal. Entry window minutes 2-4.

---

## Tweet 3 — High Win Rate != Profit (Efficient Market Proof)
**Sample:** 6,140 markets, real orderbook data

Win rates by entry time (following BTC direction):
- 0:30 — 55.8% WR, token costs $0.60
- 1:00 — 59.4%, costs $0.65
- 2:00 — 65.0%, costs $0.72
- 3:00 — 71.2%, costs $0.80
- 4:00 — 77.1%, costs $0.87
- 4:30 — 80.8%, costs $0.90

Filter for >0.2% BTC move at minute 4: **98.9% WR** (795/804 trades)

EV at every checkpoint: **NEGATIVE** (-$0.027 to -$0.034)

**Takeaway:** The market prices in information faster than you can act on the winning side. You can't profitably buy the expensive aligned token. Validates our contrarian cheap-token approach.

---

## Tweet 4 — 151 Hedge Fund Strategies Tested
**Sample:** 2,194 markets, per-second Binance data
**Reference paper:** SSRN-3247865 (151 Trading Strategies, 550+ formulas)

| Strategy | WR | Sample | Verdict |
|---|---|---|---|
| Mean Reversion | 49.7% | 2,194 | Dead on 5-min |
| Trend Following | 50.3% | 2,194 | Dead on 5-min |
| RSI Reversal | 48.9% | 2,194 | Dead — oversold is a trap |
| Fresh EMA Cross | **64.0%** | 52 | Promising but tiny sample |
| Volatility Filter | **52.6%** | 715 | Real edge, real sample |

Key findings:
- RSI 70-80 (mildly overbought) predicted DOWN at 59.8%
- RSI below 20 (deeply oversold) only bounced 43.1% — "oversold" means "still falling" on 5-min windows
- Fresh EMA crossovers (5/20 period) hit 64% but only 52 markets
- Volatility filter (trade momentum only when vol < median) gave 52.6% on 715 markets
- When ALL strategies agreed on direction: 51.1% — consensus barely beats noise

---

## Actionable Ideas for Future Implementation

### Priority 1: Order Book Fade Strategy (new Strategy 2)
- Fade extreme imbalance (>0.7 threshold) — 57.7-74.7% WR
- Buy cheap opposite token
- Completely independent signal from current strategy
- We already have order book data infrastructure
- Combine with volatility filter for extra edge

### Priority 2: Volatility Filter
- Only trade momentum when volatility < median
- 52.6% over 715 markets
- Could be added as a filter layer to current strategy OR to Strategy 2

### Priority 3: RSI Trap (Continuation not Reversal)
- Deeply oversold (RSI<20) = still falling, don't buy
- Mildly overbought (RSI 70-80) = decent DOWN signal at 59.8%
- Could be a filter: suppress contrarian UP entries when RSI < 20

### Priority 4: Fresh EMA Crossover
- 64% WR but only 52 markets — needs validation with more data
- If confirmed, could be a strong entry trigger

---

## Key Principles Confirmed
1. The market is efficient — high WR strategies have negative EV due to entry pricing
2. The edge is NOT in predicting direction with high confidence — it's in finding mispriced tokens
3. Order book shows trapped liquidity, not smart money direction
4. Most textbook signals (RSI, mean reversion, trend following) dissolve on 5-min windows
5. Sub-second oracle lag is real but already saturated by bots
