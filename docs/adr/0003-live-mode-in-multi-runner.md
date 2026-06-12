# Live execution mode inside multi_runner (T1 probe wiring)

**Status:** accepted (2026-06-12) — grilled, approved, building via /tdd

## Decision

Live trading for the T1 probe runs **inside `multi_runner.py`**, selected by a
per-bot config key (`execution_mode: live`, default `paper`). Bot K goes live;
Bot K2 stays paper; a new **`bot_k_shadow`** (config-only paper clone of K)
runs alongside so the paper-vs-live fill gap is measured per window, not
modelled. The old single-bot `main.py` live path is NOT used for T1.

Five sub-decisions:

1. **Async bridge — background task + entry-in-flight flag.** Strategy
   `on_tick` is sync; live `execute_trade` awaits a GTC round up to 10s.
   Entries spawn as fire-and-forget asyncio tasks; the strategy evaluates no
   new entries while one is in flight; a miss clears the flag and the next
   tick re-evaluates (the re-post loop made explicit). A done-callback logs
   task exceptions; `on_window_end` defensively cancels any in-flight task +
   resting order (unreachable in theory: 60s entry floor + 10s rounds).
2. **Real early exit — best-effort flatten.** Live `close_position_early`
   places an aggressive marketable SELL (cross best bid one tick, partials
   accepted, bounded price-stepping retries), **backstopped by resolution**:
   unfilled exit ⇒ position rides to settlement, loudly logged. The ledger
   records only actual fills at actual prices. (Replaces the simulation stub
   that would have made the $90 BTC distance stop fake on live capital.)
3. **Orphaned shares closed.** On GTC timeout-cancel, the final order state
   is queried once; any `size_matched > 0` becomes a real recorded trade
   (and a `filled` J27 telemetry round).
4. **Bankroll sync at startup + each window end.** Sizing only consults the
   bankroll at entries in fresh windows. Settlement lag under-reads the
   balance ⇒ sizes down (conservative) — accepted; every sync logged for
   reconciliation. Paper bots keep epoch-wallet mechanics (`bankroll_epoch`
   is paper-only).
5. **Guards.** Runner refuses to start when a live bot lacks CLOB auth (no
   silent paper fallback) or when more than one bot declares live (v1).
   Allowance checked once at startup. Rollback = flip the key to `paper` +
   restart; live rows carry `is_paper=False` in the same per-bot DB.

## Why

- **Why multi_runner, not main.py:** the kill-switch heartbeat, HALT check,
  and the J28 window-open fixes live in `multi_runner.py`. Running live via
  `main.py` would put real capital on a runner that never writes heartbeats —
  the watchdog would never arm. Disqualifying.
- **Why a background task, not inline await:** one runner iteration ticks all
  bots and writes the heartbeat. A 10s inline GTC monitor stalls K2's paper
  A/B, the dashboard, and the heartbeat — within 2× of the watchdog's 20s
  kill threshold. Shortening the GTC timeout instead would change fill
  economics and contaminate what T1 measures.
- **Why best-effort flatten, not passive SELL:** a passive exit is slow
  exactly when the stop matters (book running away). In a 5-minute binary,
  resolution is a natural backstop; pretending exits always fill (the old
  stub) was the worst of all options — ledger and wallet diverge on trade 1.
- **Why a shadow bot:** K-live vs K2-paper compares different fill models —
  it can't settle the J14 floor question. K-live vs K-shadow isolates pure
  execution reality; K-shadow vs K2 keeps the floor A/B like-for-like.

## Alternatives rejected

- `main.py` live (heartbeat/watchdog gap, J28 drift risk)
- Inline await / shortened GTC timeout (loop stall / fill-economics change)
- Passive maker exit (slow when it matters)
- No shadow (fill gap inferred from J26 models instead of measured)
- Multiple simultaneous live bots in v1 (one yaml typo = doubled exposure)

## Consequences

- T1 fleet: 3 bots — K (live), K2 (paper), K-shadow (paper).
- Strategy gains an in-flight state; entry decisions lag fill reality by ≤1
  round (~10s) — already true of live GTC, now explicit.
- The distance stop becomes best-effort: a stop that can't fill rides to
  resolution. This is the honest contract on a thin 5-minute book.
- Build is TDD-gated (trading logic); J27 telemetry records every round from
  day one.
