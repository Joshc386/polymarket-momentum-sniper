# Polymarket Momentum Sniper Bot

## Project Overview

A trading bot targeting Polymarket's 5-minute Bitcoin Up/Down markets. Uses a three-layer signal engine (oracle lag, momentum, liquidation proximity) to generate directional predictions, with Kelly-adjacent position sizing and losing streak management. The signal engine is fully implemented and working. Paper trading is live.

## Tech Stack

- **Language:** Python
- **Signal Engine:** Three-layer system (oracle lag, momentum, liquidation proximity) — implemented and working
- **Execution:** py-clob-client for Polymarket integration
- **Testing:** pytest

## Project Status Notes

- Signal engine is built and working — do not refactor core signal logic without explicit approval
- Paper trading is currently live — all changes must be validated in paper trading before any live deployment
- Project structure is well organised with clear folders — maintain this standard for all new code
- This bot handles real money — every change must be treated with the caution that implies
- **Currently active bots:** Bot G (signal-aligned, paper). Bots A-F retired/disabled. Bot K (maker variant of Bot G) in design — first live deployment when ready

---

## Documentation Discipline (project-wide rule)

The project's living memory lives in the user's Obsidian vault at:

```
C:\Users\joshc\OneDrive\Desktop\Vault\Projects\5min BTC Bot\
```

This folder contains the canonical project notes (Obsidian wikilinks reference these, not anything in the repo). They must stay in sync with the codebase.

**Current notes in the vault:**
- `BTC5Min_index.md` — overview, current status, navigation
- `decisions.md` — architectural choices and rationale
- `progress.md` — phase-by-phase timeline of what's been built
- `bots.md` — bot-by-bot retrospective (failures + Bot G's success)
- `to-do.md` — running task list (also contains the reference table below)

**Rule:** When you make a code or config change that affects bot behaviour, architecture, or project state, update the relevant note(s) in the same session. Don't defer documentation. **Edit files at the vault path above, not anywhere in the repo.**

**What gets updated for what** (this table is also at the top of `to-do.md` for easy reference):

| Change type | Notes to update |
|---|---|
| Bot enabled/disabled | `BTC5Min_index.md`, `bots.md`, `progress.md` |
| Strategy code or config (behaviour change) | `bots.md`, plus `decisions.md` if architectural |
| New bot added | `BTC5Min_index.md`, `bots.md`, `to-do.md` |
| Strategy retired | `bots.md`, `BTC5Min_index.md`, `progress.md` |
| Data feed / execution engine / risk logic | `decisions.md`, `BTC5Min_index.md` if user-visible |
| Decision reversed | `decisions.md` Section H + original entry status |
| Backtest results | `progress.md`, `to-do.md` (mark done) |
| Task completed | `to-do.md` Completed section, dated |
| New task surfaces | `to-do.md` appropriate section |
| Major insight / lesson | `bots.md` "what's been learned", possibly new `lessons/*.md` in the vault |

**Trivial changes (typos, refactors, dependency updates) → no note updates.** Only touch notes when something user-meaningful changed.

**When unsure which note to touch → ask** rather than polluting docs with low-value updates.

The Completed section of `to-do.md` is the audit trail — every meaningful change should leave a footprint there, dated.

---

## Project-Specific Agents

### Signal Engine Agent

**Role:** Owns the three-layer signal system and the logic that combines data streams into trading decisions.

**Responsibilities:**
- Maintain and tune the three existing signal layers (oracle lag, momentum, liquidation proximity)
- Monitor signal accuracy during paper trading and identify drift or degradation
- Implement new signal layers if additional edge sources are identified
- Optimise signal weights and thresholds based on paper trading performance data
- Build signal diagnostics: per-layer accuracy tracking, false positive/negative rates, signal correlation analysis
- Ensure signals are generated within the latency budget — 5-minute markets leave no room for slow computation

**Constraints:**
- Never modify working signal logic without explicit approval — tune parameters, don't restructure
- All signal changes must be A/B tested in paper trading before replacing the current version
- New signal layers must demonstrate independent predictive value — no redundant signals that just add noise
- Log every signal decision with full context (all layer outputs, combined score, threshold used, timestamp)

**Key areas:** Signal generation, layer weighting, threshold logic, signal diagnostics

---

### Execution Agent

**Role:** Handles all interaction with Polymarket — order placement, position management, and trade lifecycle.

**Responsibilities:**
- Maintain the py-clob-client integration for reliable order execution
- Implement and monitor order placement, fill confirmation, and position tracking
- Handle execution edge cases: partial fills, rejected orders, API timeouts, market closure
- Implement the Kelly-adjacent position sizing logic and ensure it respects bankroll limits
- Build the losing streak management system — enforce cooldown periods and reduced sizing after consecutive losses
- Monitor execution latency and flag any degradation that could affect 5-minute market timing
- Implement emergency stop functionality — kill switch that closes all positions and halts trading

**Constraints:**
- Never place a live trade without the paper trading phase confirming the strategy works
- All orders must include maximum position size caps — no single trade should risk more than the defined threshold
- The kill switch must work independently of the main bot — it cannot rely on the same process
- All trades must be logged with full context: signal scores, position size calculation, execution price, fill status, timestamps

**Key areas:** py-clob-client integration, order management, position sizing, risk controls

---

### Risk Management Agent

**Role:** Enforces risk controls and protects the bankroll.

**Responsibilities:**
- Implement and enforce daily loss limits, per-trade risk limits, and maximum drawdown thresholds
- Monitor the losing streak management system and ensure cooldown rules are respected
- Track bankroll over time and flag concerning trends (drawdown approaching limits, win rate declining)
- Build risk dashboards: PnL curves, drawdown charts, Sharpe ratio tracking, win rate over rolling windows
- Validate that position sizing never exceeds Kelly fraction limits
- Implement circuit breakers: auto-pause trading if daily loss limit is hit, if API errors exceed threshold, or if signal quality degrades below minimum

**Constraints:**
- Risk limits are hard limits — no override without explicit human approval
- Circuit breakers must be fail-safe: if the monitoring system itself fails, trading should pause, not continue
- All risk parameters must be configurable via config files, not hardcoded
- Daily risk reports must be generated automatically whether or not any trades were placed

**Key areas:** Loss limits, drawdown tracking, circuit breakers, risk reporting

---

### Analytics Agent

**Role:** Analyses trading performance and provides insights for strategy improvement.

**Responsibilities:**
- Build performance tracking: PnL by hour, by day, by signal layer, by market condition
- Calculate key metrics: win rate, average win/loss size, profit factor, Sharpe ratio, max drawdown, expectancy
- Identify patterns in losing trades — are losses clustered by time of day, signal type, or market volatility?
- Compare paper trading performance against live trading performance to detect execution slippage
- Generate periodic performance reports (daily summary, weekly deep dive)
- Backtest proposed signal changes against historical data before they go into paper trading

**Constraints:**
- Analytics must use only completed trades — never include open positions in performance calculations
- All metrics must include sample size context — a 90% win rate on 10 trades means very little
- Backtesting must use walk-forward methodology — no lookahead bias
- Reports should highlight actionable insights, not just raw numbers

**Key areas:** Performance metrics, trade analysis, backtesting, reporting

---

### Data Pipeline Agent

**Role:** Automates data collection, normalisation, and freshness checks for the backtest pipeline.

**Responsibilities:**
- Monitor data file freshness and alert when files are stale (>48 hours)
- Fetch data from Binance (klines), Coinalyze (OI, L/S ratio, funding), and PolyBackTest Pro (snapshots)
- Merge all data sources into unified CSVs for backtesting
- Validate data integrity: check for gaps, duplicates, NaN values, timezone issues
- Manage date ranges automatically based on available market data

**Constraints:**
- Never overwrite existing data without a backup or append strategy
- All fetchers must respect API rate limits (Binance: 1200/min, Coinalyze: 40/min)
- Data validation must run before any backtest to prevent garbage-in-garbage-out

**Key areas:** `backtest/data/`, `backtest/fetch_*.py`, data validation

**CLI:** `python -m agents.data_pipeline {status|fetch|unify|validate}`

---

### Backtest Reconciliation Agent

**Role:** Compares backtest predictions against live bot actual trades to quantify the approximation gap.

**Responsibilities:**
- Match backtest snapshots to live trades by timestamp
- Compute divergence metrics: signal agreement rate, price deviation, sizing deviation, win rate delta
- Identify where backtest diverges most (time of day, volatility regime, signal layer)
- Suggest correction factors to bring backtest closer to live performance
- Track known limitations (oracle lag approximation, snapshot vs live orderbook, missing signals)

**Constraints:**
- Requires both backtest data and live trade data in `data_runtime/trades.db`
- Must clearly label all metrics with sample sizes
- Correction factors are suggestions only — never auto-apply to live bot

**Key areas:** `data_runtime/trades.db`, `backtest/data/`, signal comparison

**CLI:** `python -m agents.reconciliation {compare|gaps|calibrate}`

---

### Market Regime Researcher Agent

**Role:** Classifies market conditions and identifies which regimes favour which strategies.

**Responsibilities:**
- Run regime detection (ATR-based trending/ranging/volatile) over historical kline data
- Output regime classification timelines for use by other agents
- Cross-reference regime labels with strategy P&L to find regime-strategy correlations
- Analyse regime transitions and time-of-day patterns
- Generate recommended strategy weights per regime

**Constraints:**
- Reuses the exact `RegimeDetector` from `backtest_real_pricing.py` — no independent implementation
- Regime classifications are descriptive, not prescriptive — the Strategy Prototyper decides how to act on them
- Must produce machine-readable output (CSV) alongside human-readable reports

**Key areas:** `backtest/backtest_real_pricing.py` (RegimeDetector), `backtest/data/binance_klines_1m.csv`

**CLI:** `python -m agents.regime_researcher {analyse|correlate|report}`

---

### Signal Lab Agent

**Role:** Rapid prototyping and evaluation of new signal hypotheses before they enter the live signal engine.

**Responsibilities:**
- Test individual signal hypotheses against historical data
- Rank all hypotheses by standalone predictive power (accuracy, Sharpe, correlation)
- Test signal combinations with weight optimisation
- Compute inter-signal correlations to identify redundancy
- Auto-discover hypotheses from `signals/hypotheses/` directory

**Constraints:**
- Signal hypotheses must implement the `SignalHypothesis` protocol: `name`, `compute(row)`, `required_fields()`
- Testing uses historical snapshot data only — never live data
- A signal must show statistical significance (50+ samples, accuracy > 55%) before recommendation
- Weight optimisation uses grid search, not gradient descent — keep it simple and auditable

**Key areas:** `signals/hypotheses/`, `signals/combiner.py`, `backtest/data/polybacktest_snapshots.csv`

**CLI:** `python -m agents.signal_lab {test|scan|combine}`

---

### Strategy Prototyper Agent

**Role:** End-to-end strategy testing — takes a strategy definition, runs it through the full backtest pipeline with Kelly sizing and risk management, and produces a go/no-go recommendation.

**Responsibilities:**
- Run full backtests with `BacktestAccount` (Kelly sizing, daily loss caps, streak management)
- Side-by-side comparison of multiple strategies
- Auto-discover strategies from `strategies/` directory
- Generate deployable strategy module scaffolds via `promote` command
- Produce go/no-go recommendations based on win rate, Sharpe, profit factor

**Constraints:**
- Must use identical risk parameters to the live bot (from `config.yaml`)
- Reuses `BacktestAccount` from `backtest_real_pricing.py` — no independent sizing/risk implementation
- Strategy prototypes must implement the `StrategyPrototype` protocol: `name`, `description`, `should_enter()`, `entry_params()`
- Recommendations require 50+ trades for statistical significance

**Key areas:** `strategies/`, `backtest/backtest_real_pricing.py` (BacktestAccount), `config.yaml`

**CLI:** `python -m agents.strategy_prototyper {run|compare|promote|list}`

---

### Agent Scheduler

**Role:** Automated orchestration — runs agents on schedules and file-change triggers.

**Scheduled daily (default 06:00 UTC):**
- `data_pipeline status` + `validate` — freshness check and data integrity
- `regime_researcher analyse` — update regime timeline
- `reconciliation compare` — track live vs backtest divergence

**Triggered on file change:**
- New `.py` in `strategies/` → `strategy_prototyper compare` (all strategies)
- New `.py` in `signals/hypotheses/` → `signal_lab scan` (all hypotheses)
- Modified `.csv` in `backtest/data/` → `data_pipeline unify` (5-min cooldown)

**CLI:** `python -m agents.scheduler [--daily-hour 6] [--poll-interval 60] [--run-now]`

**Log file:** `data_runtime/scheduler.log`

---

## Agent Team Patterns For This Project

### Pattern 1: Signal tuning cycle
Spin up: Signal Engine Agent + Analytics Agent + QA Agent (global)
- Analytics reviews recent performance and identifies weak spots
- Signal Engine adjusts parameters based on findings
- QA writes tests for the updated logic

### Pattern 2: Going live preparation
Spin up: Execution Agent + Risk Management Agent + Security Agent (global)
- Execution verifies all order handling is production-ready
- Risk Management confirms all limits and circuit breakers are active
- Security audits API key handling, logging, and failover paths

### Pattern 3: Adding a new signal layer
Spin up: Signal Engine Agent + Analytics Agent + QA Agent (global) + Code Review Agent (global)
- Signal Engine implements the new layer
- Analytics backtests it against historical data
- QA writes tests
- Code Review validates the integration

### Pattern 4: Performance investigation
Spin up: Analytics Agent + Signal Engine Agent + Risk Management Agent
- Analytics identifies the problem (declining win rate, increased drawdown, etc.)
- Signal Engine checks for signal degradation
- Risk Management reviews whether limits were respected

### Pattern 5: Live monitoring setup
Spin up: Execution Agent + Risk Management Agent + Performance Agent (global)
- Execution builds real-time trade monitoring
- Risk Management builds alerting for limit breaches
- Performance reviews the monitoring code for efficiency under continuous operation

### Pattern 6: Strategy R&D cycle
Spin up: Data Pipeline Agent → Regime Researcher + Signal Lab → Strategy Prototyper → QA + Code Review (global)
- Data Pipeline ensures all data is fresh and validated
- Regime Researcher classifies conditions, Signal Lab evaluates hypotheses (parallel)
- Strategy Prototyper runs full backtests with Kelly sizing and risk management
- QA writes tests, Code Review validates the integration

### Pattern 7: Backtest calibration
Spin up: Reconciliation Agent + Analytics Agent + Signal Engine Agent
- Reconciliation identifies where backtest diverges from live
- Analytics provides performance context for the gap analysis
- Signal Engine reviews whether signal approximations can be improved

### Pattern 8: New signal evaluation
Spin up: Signal Lab + Regime Researcher + Strategy Prototyper
- Signal Lab tests the hypothesis standalone and in combination
- Regime Researcher checks if the signal works better in certain conditions
- Strategy Prototyper runs the full backtest to validate end-to-end

## Agent skills

### Issue tracker

Issues live in GitHub Issues at `Joshc386/polymarket-momentum-sniper`. Skills use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
