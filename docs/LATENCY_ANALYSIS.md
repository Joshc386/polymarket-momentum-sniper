# Latency Analysis & Reduction Plan

## Current measured latencies (your machine, today)

| Endpoint | RTT | Region | Notes |
|---|---|---|---|
| Polymarket Gamma API | **4ms** | Cloudflare (Toronto edge) | Excellent |
| Polymarket CLOB | **8ms** | Cloudflare (Toronto edge) | Excellent |
| Polymarket WS | **9ms** | Cloudflare (Toronto edge) | Excellent |
| Coinbase WS | **8ms** | Cloudflare | Excellent |
| Polygon RPC | **21ms** | US East | Good |
| **Binance WS (any endpoint)** | **234-284ms** | **Japan / Tokyo** | **Critical bottleneck** |

## The Binance problem

Every Binance WebSocket endpoint we tested resolves to AWS Tokyo:
- `stream.binance.com` → 18.176.200.216 (Tokyo) — 238ms
- `data-stream.binance.com` → 52.193.76.183 (Tokyo) — 240ms
- `fstream.binance.com` → 54.238.191.232 (Tokyo) — 236ms
- `stream.binance.us` → Ashburn USA — TIMEOUT (geo-blocked from UK)

This is by design — Binance routes international traffic through Asia for
regulatory/compliance reasons. We can't fix this from a UK home connection.

### What this means for Bot F

The strategy assumes: "Binance moves → CLOB hasn't repriced → we exploit gap."

Real timing:
1. Binance prints a trade at T=0
2. We receive it at T=235ms
3. We process and submit order at T=300-400ms
4. **Professional arbitrageurs (located near Binance/Polymarket) saw Binance at T=20-50ms**
5. They submitted to Polymarket by T=100-150ms
6. **By the time we react, the gap has closed**

We're competing in a race we structurally cannot win from this location with
this architecture.

---

## Three options for reducing Binance latency

### Option A: Switch primary signal to Coinbase (FREE, 30 min work)
- Coinbase WS: 8ms vs Binance: 234ms
- Coinbase is one of Chainlink Data Streams' sources
- Coinbase generally moves within 100-500ms of Binance for BTC
- Net effect: we'd see effective price moves ~100-500ms after they happen,
  but **226ms faster than Binance for us**
- Tradeoff: marginally less leading-edge data, but we already use the
  Binance+Coinbase average for the oracle

**My recommendation: do this regardless of other choices.** It's the
single biggest latency improvement available and it's free.

### Option B: VPS in Tokyo (~$5-10/mo, half-day work)
- Spin up a small AWS Lightsail / DigitalOcean droplet in Tokyo region
- Move Binance WS connection there only (or whole bot)
- Tokyo → AWS Tokyo Binance: ~5-15ms
- Move whole bot: also Tokyo → Polymarket = ~150ms (worse for Polymarket!)
- **Hybrid: Tokyo VPS forwards Binance data to UK over websocket** — adds
  latency back. Not worth it.

**Verdict: only worth it if Bot F's whole reason for being is Binance latency.**
And given Coinbase is "good enough", probably skip.

### Option C: Co-locate everything in US East (~$5-30/mo, 1-2 days work)
- Move whole bot to AWS us-east-1 (Polymarket's likely origin region)
- Polymarket: 1-5ms (vs 8ms now)
- Coinbase: 1-5ms (vs 8ms now)
- Binance: ~150ms (vs 234ms — Tokyo from US East)
- All exchanges: roughly best-case for Western trading
- Adds operational overhead (deployment, monitoring, secrets management)

**Verdict: worth it once a strategy is proven profitable on paper. Premature now.**

---

## Polymarket latency: already excellent

For Bot H (arbitrage), the chain is:
- Polymarket book change → us (9ms)
- Detect arb (<1ms in code)
- Submit FOK orders → Polymarket (8ms)
- **Round trip: ~25-50ms**

We're competitive here. Cloudflare's edge network is already putting us
near-optimal. Co-locating wouldn't help much.

---

## Application-level latency wins (Python optimisations)

These are general improvements for both bots. Most are 1-line changes.

### 1. Replace asyncio default loop with uvloop
- **Impact:** 2-4x faster async performance
- **Effort:** Add 2 lines of code
- **Risk:** None (mature library)

```python
import uvloop
uvloop.install()
```

### 2. Replace stdlib json with orjson
- **Impact:** 2-3x faster JSON parsing on WS messages
- **Effort:** Replace `import json` with `import orjson`
- **Risk:** Slightly different API, needs `.decode()` handling

### 3. Pre-sign order templates
- Polymarket EIP-712 signing involves several hashing operations
- Sign order templates ahead of time, fill in price/size at submission
- **Impact:** Saves ~5-20ms per order
- **Effort:** Moderate (refactor of order construction)
- **Risk:** Need to handle signature expiry correctly

### 4. Persistent HTTP connections for order submission
- Already using httpx, but verify connection pooling is enabled
- Use HTTP/2 if Polymarket supports it
- **Impact:** Saves ~10-30ms per order (no TCP handshake)
- **Effort:** Config change
- **Risk:** None

### 5. Reduce Python work between WS receipt and decision
- Instrument current code path for Bot F
- Remove any blocking I/O between WS message and order submission
- Use `asyncio.create_task` for non-critical work (logging, metrics)
- **Impact:** Variable, often 5-50ms savings
- **Effort:** Small per fix, requires profiling

### 6. WebSocket library upgrade
- Default `websockets` library is solid
- `aiohttp` WS is sometimes faster
- `websocket-client` (sync) can be wrapped
- **Impact:** Marginal (<1ms typically)
- **Effort:** Probably not worth it

---

## Measurement first

We don't actually know how long Bot F takes between data and action. Before
optimising, we need to measure.

**Proposed instrumentation (add to Bot F):**

1. Timestamp each Binance/Coinbase WS message on receipt
2. Timestamp when MoveDetector triggers
3. Timestamp when entry decision made
4. Timestamp when order submitted
5. Timestamp when order acknowledged
6. Log the full chain to CSV

After 24h of running, we'll know:
- Network latency (T_recv - T_exchange)
- Processing latency (T_decision - T_recv)
- Submission latency (T_ack - T_submit)
- **Total reaction time**

This tells us whether application-level optimisation is even worth doing,
or whether the entire latency budget is in the Binance round-trip.

---

## Recommendation summary

### Do immediately (free wins):
1. **Switch Bot F's primary trigger from Binance to Coinbase WS** (226ms savings)
2. **Add `uvloop`** (1-line change, broad benefit)
3. **Add `orjson`** (1-line change, JSON-heavy code)
4. **Instrument Bot F for end-to-end latency measurement**

### Do after Bot H Phase 1:
5. **Pre-signed order templates** (saves ~5-20ms per submission)
6. **Verify HTTP connection pooling** (saves 10-30ms per HTTP submission)

### Defer until a strategy is profitable on paper:
7. **VPS migration** (US East likely best — ~$10-30/mo)

### Skip:
- Tokyo VPS (Bot F doesn't need pure Binance access)
- Multi-region setup (premature)
- Custom WebSocket library (marginal)

---

## Why this matters specifically for Bot H

Bot H reacts to Polymarket book changes. Latency budget:
- Book update arrives: T=0 (Cloudflare edge to us = 9ms after generation)
- Detection: <1ms
- Submit FOK: 8ms one-way
- Polymarket processing: TBD (small)
- Total: **~25-50ms reaction time**

Pro arbitrageurs with co-location may be at 5-15ms. We're slower but
not catastrophically so. Most arbs that close in <50ms aren't economic
to chase anyway (gas/fees > profit).

**The arbs we can realistically capture: those persisting >100ms.**
Phase 1 will tell us what fraction of opportunities that is.
