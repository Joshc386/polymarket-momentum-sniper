# Polymarket Momentum Sniper Bot

An algorithmic trading bot for **Polymarket's 5-minute Bitcoin Up/Down markets**.
Every five minutes, Polymarket opens a binary market: *will BTC close this
window above or below its opening price?* This bot watches the underlying spot
market in real time, forms a directional view in the final minutes of each
window, and trades the side it believes is mispriced.

**Current status: paper trading.** Two bot variants (Bot K and Bot K2) run
side-by-side in an A/B test with a statistically validated paper edge. No live
capital is deployed. Go-live is gated behind an explicit readiness checklist
(see [Status](#current-status)).

> ⚠️ This is a personal research project that trades (on paper, for now) in a
> real-money venue. Nothing in this repository is financial advice. The risk
> framework, kill switch, and validation tooling exist precisely because the
> failure modes are real.

---

## Table of contents

- [How it works](#how-it-works)
- [Where the edge actually comes from](#where-the-edge-actually-comes-from)
- [Architecture](#architecture)
- [The signal engine (L1–L12)](#the-signal-engine-l1l12)
- [Entry gate and EV calculation](#entry-gate-and-ev-calculation)
- [Position sizing and risk management](#position-sizing-and-risk-management)
- [Execution model](#execution-model)
- [Safety: the independent kill switch](#safety-the-independent-kill-switch)
- [Data recording: two stores](#data-recording-two-stores)
- [R&D agents](#rd-agents)
- [Backtesting and validation](#backtesting-and-validation)
- [Running it](#running-it)
- [Testing](#testing)
- [Current status](#current-status)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)

---

## How it works

The lifecycle of one 5-minute window:

1. **Market discovery** finds the active `btc-updown-5m-*` market on Polymarket
   and tracks its open price (the *resolution line*).
2. **Data feeds** stream continuously and are shared across all bots: Binance
   spot trades and order book, Coinbase direction, cross-exchange sentiment,
   liquidation prints, funding/open-interest context, and the Polymarket CLOB
   order book itself.
3. From **4:30 remaining** the bot starts scanning. On every tick (~1/sec) the
   **signal combiner** blends the active signal layers into a single directional
   score using regime- and time-aware weights, producing `est_prob_up` — the
   bot's probability that the window resolves Up.
4. The **entry gate** compares that estimate against the market's own implied
   probability. If the expected value per share — after the real Polymarket
   taker fee — clears the required threshold, the bot picks a side (YES/UP or
   NO/DOWN).
5. **Risk management** (daily loss caps, drawdown breaker, regime sizing,
   per-trade caps) approves or vetoes, and **Kelly-adjacent sizing** decides the
   stake.
6. The **execution engine** places the order (GTC maker order at the touch for
   most entries; see [Execution model](#execution-model)). Paper mode simulates
   this; live mode uses `py-clob-client-v2` (CLOB V2).
7. The position is held to **resolution** (5-minute windows leave little room
   for exits). The result, with a full snapshot of everything the bot saw at
   entry, is written to the bot's trade database.

Two bots run simultaneously via `multi_runner.py`, sharing all data feeds but
owning independent signal processing, entry logic, risk state, and databases —
a clean A/B test.

## Where the edge actually comes from

An honest note, because it shapes everything else in the repo (decision J16 in
the project log):

The model's headline probability output (`est_prob_up`) is **calibrated but
compressed** — it almost never strays far from 0.50. Empirically, the bot's
edge is mechanically a **market-displacement fade**: when BTC has moved away
from the resolution line and the market's quoted odds have over- or under-shot,
the bot trades against the mispricing. The signal engine's job is less "predict
the future" and more "recognise when the current price is wrong."

That edge is validated, not assumed:

- `backtest/validate_edge.py` shows both live bots **beat the market on Brier
  score** (the model adds information beyond the order book), with
  day-clustered significance t ≈ 3.7 (Bot K) over 29 days.
- A maker-fill realism study (`backtest/maker_fill_realism.py`) re-priced the
  paper edge under realistic limit-order fills: the edge **survives with a
  haircut** (roughly a third off paper ROI), with adverse selection measured
  and reported rather than ignored.

## Architecture

```
                         ┌───────────────────────────────┐
   Binance (trades, OB,  │        SHARED DATA FEEDS      │  Polymarket CLOB
   liqs) · Coinbase ·    │  data/*.py — one connection   │  (orderbook, prices)
   Coinalyze · CoinGlass │  per source, fanned out       │
                         └──────────────┬────────────────┘
                                        │ ticks
                  ┌─────────────────────┼─────────────────────┐
                  ▼                     ▼                     ▼
          ┌──────────────┐      ┌──────────────┐       (one block per bot,
          │    BOT K     │      │    BOT K2    │        via multi_runner.py)
          │              │      │              │
          │ signals/     │      │  same, plus  │
          │  combiner ──▶│      │  directional │
          │ strategy/    │      │  L1 floor    │
          │  entry gate  │      │              │
          │  risk mgr    │      │              │
          │  sizing      │      │              │
          │ core/        │      │              │
          │  execution   │      │              │
          └──────┬───────┘      └──────┬───────┘
                 ▼                     ▼
        data_runtime/bot_k_*.db   data_runtime/bot_k2_*.db
        (trades + signal_diag)    (trades + signal_diag)

   tools/watchdog.py ──▶ heartbeat stale? ──▶ tools/kill_switch.py
   (separate process)                         (cancel, flatten, HALT)
```

## The signal engine (L1–L12)

Each layer outputs a scalar in roughly [−1, +1] (positive = bullish/UP). The
combiner blends them with weights that depend on **market regime** and **time
remaining in the window**. Full domain vocabulary lives in
[CONTEXT.md](CONTEXT.md).

| Layer | Name | What it measures | Status |
|---|---|---|---|
| L1 | Oracle lag (legacy name) | BTC's displacement from the window-open resolution line | Core |
| L2 | Momentum | ROC, direction, volume, candle body, RSI on spot | **Dominant** (per walk-forward optimisation) |
| L3 | Liquidation proximity | Distance to liquidation clusters, L/S crowding | Empirically weak; under investigation |
| L4 | Orderbook | CLOB imbalance, flow, weighted-mid deviation, pressure, thickness | Core |
| L5 | Sentiment | Cross-exchange direction agreement | Core |
| L6–L12 | Additive layers | Fade, taker ratio, CLOB flow, absorption, exhaustion, trade size, wallet flow | Mixed; several disabled by default |

Key engineering facts:

- **Weights are walk-forward optimised, not guessed.** Bot K's schedule comes
  from a grid search over 13,867 markets (7-day train / 5-day test folds,
  Sharpe-scored, 7/7 out-of-sample folds beat the default). Headline finding:
  momentum dominates (trade-weighted L2 ≈ 0.52), the L1 layer was historically
  over-weighted.
- **Regime detection** (`strategy/mtf_regime_detector.py`) classifies
  trending/ranging/high-vol/low-vol using parameters calibrated by walk-forward
  validation over 2 years of BTC data (21 folds). Regime switches the weight
  schedule and scales position size.
- **Bot K2 = Bot K + a directional L1 floor**: it refuses to bet *against* the
  resolution line when BTC is clearly displaced (deadband 0.1) — closing a
  measured leak where counter-line bets lost.

## Entry gate and EV calculation

The gate is expected-value-based, using Polymarket's real fee model:

```
fee_per_share = 0.07 · p · (1 − p)        # taker fee, crypto category
net_ev_per_share = p_win · (1 − price) − (1 − p_win) · price − fee
```

A trade fires only when `net_ev_per_share` clears the required edge (regime-
dependent multiplier over a 0.003 floor), signal confidence passes its minimum,
and the entry falls inside the validated timing window (first qualifying signal
between 4:30 and 1:00 remaining — a strategy worth ~+38% over fixed-time entry
in an 8,256-trade backtest).

## Position sizing and risk management

- **Quarter-Kelly sizing** (`strategy/sizing.py`): Kelly fraction from the
  estimated edge, multiplied by 0.25 as an estimation-error humility tax,
  clamped to absolute per-trade bounds.
- **Hard limits** (`strategy/risk_manager.py`): daily loss cap (20% of
  start-of-day bankroll, warning at 15%), drawdown circuit breaker (30% from
  peak), daily trade cap, minimum bankroll floor.
- **Regime-aware sizing**: e.g. 1.5× in ranging (the strategy's best regime),
  reduced in trending-up.
- **Losing-streak handling** is deliberately *disabled* on Bot K — its own
  trade history showed win rate at 5 consecutive losses was the *highest* of
  any streak length, so shrinking size there cost money. (Conventional wisdom,
  tested and rejected.)

All risk parameters live in [config.yaml](config.yaml), not in code.

## Execution model

Time-conditional maker/taker (`core/execution.py`):

- **> 60s remaining → GTC limit order** at the touch (maker, 0% fee). If
  unfilled after `gtc_timeout_sec` (10s), cancel and re-evaluate.
- **≤ 60s remaining → FOK** crossing the spread (taker, fee + slippage), for
  fill certainty when there's no time to rest an order.

In practice, 100% of validated entries occur in the maker regime — which is
why maker-fill realism (do you actually get filled, and are the fills adverse?)
was treated as the #1 go-live gate and studied explicitly (see
[Backtesting and validation](#backtesting-and-validation)).

Paper mode (`PaperExecutionEngine`) simulates fills optimistically (fill at the
ask, 100% certainty); the realism study quantifies exactly how optimistic.

## Safety: the independent kill switch

Built and validated before any live deployment (decisions J18–J23):

- The bot writes an **atomic heartbeat** every tick and checks a **HALT file**
  at the top of every loop and before every order.
- A **watchdog** (`tools/watchdog.py`) runs as a *separate OS process*
  (installable as a Windows Scheduled Task with auto-restart). If the heartbeat
  goes stale for 3 consecutive polls, it fires the kill switch. It never
  auto-restarts the bot — HALT is sticky until a human clears it.
- The **kill switch** (`tools/kill_switch.py`, also runnable manually via
  `python -m tools.kill_switch`) executes halt-first ordering: write HALT →
  cancel all open orders → discover positions (public Data API, with an
  on-chain ERC-1155 balance fallback — deliberately *not* the same code path
  the bot uses) → flatten each position with aggressive marketable sells →
  independently re-verify, logging a CRITICAL alert if anything remains.
- Every kill action is audit-logged as JSONL.

Fail-safe bias throughout: missing/corrupt heartbeat counts as stale; if
monitoring itself breaks, trading pauses rather than continues.

## Data recording: two stores

Deliberately separate (see CONTEXT.md for the full rationale):

1. **Per-tick signal diagnostics** (`data_runtime/<bot>_signal_diag.db`,
   table `signal_ticks`): one row per tick, *whether or not a trade fired*,
   including `would_enter` and what blocked it. This is the counterfactual /
   selection-bias dataset — what the bot saw when it chose **not** to trade.
2. **Enriched trade records** (`data_runtime/<bot>.db`, table `trades`): one
   row per actual entry carrying the **full feature snapshot at the instant of
   entry** (every layer value and sub-component, the weight vector in force,
   regime, prices, EV) plus the outcome. One row fully reconstructs the
   decision.

Both stores are populated from a single shared snapshot builder so feature
definitions cannot drift apart. Absent values are `NULL`, never `0.0`, so
"layer didn't compute" is never confused with "layer said neutral."

## R&D agents

A set of standalone CLI tools (`agents/`) for the research loop around the live
bots — all read-only with respect to bot databases:

| Agent | Purpose | CLI |
|---|---|---|
| Data pipeline | Fetch/unify/validate backtest data (Binance, Coinalyze, snapshots) | `python -m agents.data_pipeline {status\|fetch\|unify\|validate}` |
| Regime researcher | Regime classification over historical klines, regime↔PnL correlation | `python -m agents.regime_researcher {analyse\|correlate\|report}` |
| **Regime monitor** | **Daily drift watch on the live bots** — L1 distribution, regime mix, PnL by zone, and a drift alarm (PSI, expected-edge-at-risk) that catches regime shifts before they masquerade as signal degradation | `python -m agents.regime_monitor {profile\|pnl\|drift\|report}` |
| Signal lab | Test new signal hypotheses standalone and in combination | `python -m agents.signal_lab {test\|scan\|combine}` |
| Strategy prototyper | Full-pipeline backtests of strategy prototypes with go/no-go output | `python -m agents.strategy_prototyper {run\|compare\|promote\|list}` |
| Reconciliation | Quantify the live-vs-backtest gap | `python -m agents.reconciliation {compare\|gaps\|calibrate}` |
| Scheduler | Cron-style orchestration of the above | `python -m agents.scheduler` |

## Backtesting and validation

The `backtest/` directory is the project's evidence base. Highlights:

- **`validate_edge.py`** — the "is the edge real?" tool. Ground-truth analysis
  over resolved trades: calibration curves, model-vs-market Brier comparison,
  equity curves, and **day-clustered significance** (trade-level t-stats are
  inflated by within-day correlation; daily clustering is the honest metric).
- **`maker_fill_realism.py`** — re-prices the paper edge under realistic maker
  fills using the bots' actual logged trades plus the tick-level mid path.
  Quantifies fill rates, adverse selection (missed fills win 62–92% of the
  time — the winners run away from a resting bid), and the surviving edge.
- **`maker_execution_backtest.py`** — high-fidelity limit-order fill simulation
  over 42M order-book snapshots (offsets × patience grid).
- **`weight_optimiser.py`** — the walk-forward grid search behind Bot K's
  signal weights.
- **`regime_calibration.py`** — the walk-forward calibration behind the regime
  detector's parameters.
- Methodology rules enforced project-wide: walk-forward only (no lookahead),
  sample sizes always reported, day-clustered significance for anything that
  matters, and known-output regression tests for statistical code.

A 12 GB tick database (`backtest/data/tick_data.db`, not in git) holds ~42M
Polymarket order-book snapshots across 13,867 markets for fill-level studies.

## Running it

> Paper trading runs out of the box (no keys needed for market data). Live
> trading additionally requires Polymarket credentials and is intentionally
> gated behind the go-live checklist.

```bash
# 1. Environment (Python 3.13, venv recommended)
pip install -r requirements.txt

# 2. Configure
#    config.yaml        — strategy/risk/execution parameters (documented inline)
#    config_multi.yaml  — which bots run side-by-side
#    .env               — secrets, never committed. Names only:
#      POLYMARKET_PRIVATE_KEY, POLYMARKET_FUNDER_ADDRESS  (live only; empty on paper)
#      TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID               (optional notifications)
#      POLYGON_RPC_URL                                     (on-chain reads)

# 3. Run the A/B fleet (paper)
python multi_runner.py

# 4. Manual kill switch (separate terminal, works even if the bot hangs)
python -m tools.kill_switch

# 5. Watchdog (optional but recommended; auto-fires the kill switch)
python -m tools.watchdog
```

`main.py` is the legacy single-bot entry point; `multi_runner.py` is the
current one.

## Testing

```bash
python -m pytest tests/ -q
```

~555 tests. Project rules: trading-logic changes are TDD-only (failing test
first), and statistical code carries known-output regression tests (fixed
input → expected output) so refactors can't silently change results.

## Current status

| Item | State |
|---|---|
| Bot K (SM-confirmation) | **Paper, live A/B** — go-live candidate. Day-clustered t ≈ 3.7, beats market on Brier, ~+18–24% paper ROI (≈ −⅓ after maker-fill realism) |
| Bot K2 (K + directional L1 floor) | **Paper, live A/B** — evaluation gated until ~2 weeks of data; weaker under realistic fills |
| Bots A–G | Retired. Seven strategies tried; the K-line is the survivor. Bot G (the prior edge-holder) decommissioned 2026-06-01 |
| Kill switch + watchdog | Built, tested end-to-end (paper dry-run validated) |
| Security audit | Comprehensive scan clean (0 critical/high); dependency lockfile is the remaining pre-live gate |
| Go-live | **Not yet.** Two-threshold plan: T1 = bounded-loss live fill probe, T2 = scaled capital, each behind explicit checklists |

## Repository layout

```
core/          Bot runtime: config, execution engines (paper + live CLOB),
               market discovery, kill-switch I/O, event loop
strategy/      Decision layer: entry gate, risk manager, Kelly sizing,
               regime detectors, feature-snapshot builder
signals/       Signal layers L1–L12 + the combiner (weights & schedules)
data/          Market data feeds (Binance, Coinbase, Coinalyze, CoinGlass,
               Chainlink, Polymarket orderbook, liquidations, wallet flow)
agents/        R&D CLI agents (pipeline, regime monitor/researcher,
               signal lab, prototyper, reconciliation, scheduler)
backtest/      Backtests, optimisers, validation studies + their outputs
tools/         Kill switch, watchdog, diagnostics, latency tooling
tests/         pytest suite (~555 tests)
notifications/ Telegram alerts
multi_runner.py  Entry point — runs the bot fleet on shared feeds
config.yaml      All tunable parameters, documented inline
CONTEXT.md       Domain vocabulary (the ubiquitous-language doc)
docs/adr/        Architecture decision records
```

## Documentation

- **[CONTEXT.md](CONTEXT.md)** — the domain language: signal layers, edge vs
  EV, the two data stores, snapshot-at-entry principle.
- **[docs/adr/](docs/adr/)** — architecture decision records (e.g. the
  full-feature snapshot decision, kill-switch architecture).
- **[CLAUDE.md](CLAUDE.md)** — working agreements and agent roles for
  AI-assisted development on this repo.
- A private Obsidian vault holds the session-by-session project history,
  decision log (J-series), and retrospectives; key conclusions are mirrored
  into the files above.

---

*Built with discipline borrowed from people who blew up accounts so we don't
have to: validate out-of-sample, cluster your significance by day, treat paper
fills as a lie until proven otherwise, and keep a kill switch that doesn't
trust the thing it's killing.*
