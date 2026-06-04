# Kill Switch — Build Spec (v1)

**Status:** plan approved 2026-06-03 (grilled via `/grill-with-docs`), not yet
built. Architectural rationale: [ADR-0002](adr/0002-independent-kill-switch.md).
Build under `/gstack-guard` with TDD. Money-adjacent / live-capital.

**Goal (success criteria):** if the trading process hangs or dies, all resting
orders are cancelled, open positions are flattened (subject to the >60s guard),
and trading stays disabled until a human re-enables it — **without depending on
the trading process being alive or healthy.**

---

## 1. Components

| Component | File | Process | Role |
|---|---|---|---|
| Kill action | `tools/kill_switch.py` | standalone (manual or called by watchdog) | HALT → cancel → flatten → verify, via its own `PolymarketClient` |
| Watchdog | `tools/watchdog.py` | standalone, OS-supervised | poll heartbeat; auto-fire kill action on stale |
| Heartbeat writer | in `multi_runner.py` | trading process | atomically write `heartbeat.json` each loop |
| HALT honouring | in `multi_runner.py` | trading process | check `HALT` before every order; exit on detection |
| OS supervisor | Windows Scheduled Task | OS | restart the watchdog if it crashes |

K and K2 share one `multi_runner` process and one Polymarket account → **one
heartbeat, one HALT flag, account-wide flatten.**

## 2. Files in `data_runtime/` (the signalling channel)

- **`heartbeat.json`** — written atomically (temp + `os.replace`) by the bot every
  loop iteration:
  ```json
  {"ts": 1780431948.84, "window_end_ts": 1780432200.0, "token_ids": ["0xYES…","0xNO…"]}
  ```
  `ts` = wall-clock of the write; `window_end_ts` = resolution time of the active
  market (immutable fact, used by the flatten guard); `token_ids` = tokens the
  active window trades (used to match discovered positions to `window_end_ts`).
- **`HALT`** — sticky flag file. Presence = trading disabled. Contents: a short
  record of who/when/why fired. **Removed only by a human.**
- **`kill_switch.log`** + **`kill_events`** (small table) — the kill switch's own
  authoritative record of every action it took.

## 3. Kill action — order of operations (`tools/kill_switch.py`)

Halt-first, so a bot that recovers mid-kill sees HALT immediately:

1. **Write `HALT`** (trigger = `manual`|`watchdog`, timestamp).
2. **Cancel all resting orders** — `poly_client.cancel_all_orders()`.
3. **Discover positions** (account ground truth):
   - primary: `poly_client.get_positions()` → **public Data API**
     `data-api.polymarket.com/positions?user=<funder>` *(spike-verified
     2026-06-04 against live public data; the old CLOB-based impl was a silent
     stub — see §8.1 + ADR-0002 amendment). Needs `POLYMARKET_FUNDER_ADDRESS`.*
   - fallback: on-chain CTF (ERC-1155) balances via Polygon RPC
   - advisory cross-check only: bot `trades` rows where `resolution IS NULL`
4. **Flatten each position** subject to the guard:
   - read `window_end_ts` from `heartbeat.json` (match by `token_ids`);
   - if `window_end_ts - now > 60s` → **sell**; else → skip (cancel+halt already
     done; let the 5-min market resolve);
   - if a position's timing is unknown (token not in heartbeat) → **bias to sell**.
   - **Sell mechanic:** aggressive marketable-limit SELL crossing the current bid,
     **accepts partial fills**, bounded retry loop: re-query remaining size,
     resubmit at progressively more aggressive prices, up to `N` attempts or a
     price floor (0.01). Not FOK.
5. **Re-verify flat** (re-query positions). If not flat after retries → **log
   LOUDLY**; rely on 5-min auto-resolution backstop.
6. Append everything to `kill_switch.log` / `kill_events`.

No confirmation prompt (emergency tool must act instantly).

## 4. Watchdog (`tools/watchdog.py`)

- Poll every **2s** (config).
- **Arm only after seeing ≥1 valid heartbeat** (avoids firing on a just-booting bot).
- Once armed, each poll computes `age = now - heartbeat.ts`. `age > 20s` (config)
  → increment stale counter, else reset. **Fire after 3 consecutive stale polls**
  (config) → ~26s worst-case detection.
- **Fail-safe:** once armed, a heartbeat that is missing / unreadable / unparseable
  counts as stale (pause beats run-blind).
- On fire: call the kill action (in-process import, same code path as manual).
- Loud logging on arm, each stale tick, and fire.

## 5. Bot-side changes (`multi_runner.py`)

- Write `heartbeat.json` atomically at the top of every loop iteration with the
  current `window_end_ts` + active `token_ids`.
- **Check `HALT` before placing any order** (entry *and* any sell) — not only at
  loop top. If present: stop all order placement, log, and **exit the process
  gracefully** (flush logs / DB).
- These are additive; no change to entry/sizing/signal logic.

## 6. Config (`kill_switch:` block) — DONE 2026-06-04

Added to `config.yaml` + loaded into `Config` (`ks_*` fields) and consumed by
the watchdog and `run_kill`. Paths stay as `core/kill_switch_io` constants
(`data_runtime/heartbeat.json`, `HALT`) rather than config — fixed locations.

```yaml
kill_switch:
  watchdog_poll_secs: 2       # watchdog heartbeat poll cadence
  staleness_secs: 20          # heartbeat age beyond this counts as stale
  stale_checks_to_fire: 3     # consecutive stale checks before firing (~26s)
  flatten_guard_secs: 60      # closer than this to resolution -> let it resolve
  flatten_max_retries: 4      # aggressive-sell attempts per position
```

## 7. Test plan (TDD; see ADR-0002 for paper-mode reality)

1. **Kill-action unit tests** (mock `PolymarketClient`): HALT-written-first;
   `cancel_all_orders` called; positions discovered from injected source; >60s
   guard sell-vs-skip; bounded retry continues on partial fills until flat or N;
   loud log + re-verify on failure.
2. **Watchdog-logic unit tests** (inject fake clock + fake heartbeat contents):
   arm-after-first-heartbeat; fire on 3 consecutive stale; fail-safe on
   missing/unreadable; no-fire when fresh.
3. **Integration — "fake hung bot"**: harness writes a heartbeat then stops; real
   watchdog runs against a **mock client**; assert end-to-end fire.
4. **Manual paper dry-run (gate before live):** induce staleness → watchdog fires
   → `HALT` written → bot stops + exits. (Cancel/flatten are no-ops in paper.)
5. **Q1 spike (read-only): DONE 2026-06-04.** Found the original
   `get_positions()` was a silent stub (CLOB SDK has no such method → always
   `[]`). Rewrote it onto the public Data API and verified the live contract
   (HTTP 200, rows carry `asset`/`size`/`conditionId`, parser confirmed) plus 9
   unit tests (`tests/test_polymarket_positions.py`). No orders placed; no live
   account configured (paper-only `.env`), so verified against live public data.

## 8. Build order (phased)

1. ✅ **Spike** `get_positions()` (§7.5) — **DONE 2026-06-04.** Was a silent stub;
   rewritten onto the public Data API + verified live. Next: step 2.
2. ✅ **Heartbeat writer + atomic write + `HALT` pre-order check & graceful exit
   — DONE 2026-06-04.** Shared I/O module `core/kill_switch_io.py`
   (atomic `write_heartbeat`/`write_halt`, fail-safe `read_heartbeat`,
   `halt_active`, `clear_halt`); paths hardcoded as constants (config block
   deferred to step 4/6). `multi_runner.py`: loop-top HALT check → graceful
   exit + Telegram notify, atomic heartbeat write each tick (`mkt.end_time` +
   tokens). Pre-order HALT guard added to `LiveExecutionEngine.execute_trade`
   (entries) **and** `close_position_early` (exits — defers flattening to the
   kill switch). Tests: `test_kill_switch_io.py` (11) + `test_execution_halt_guard.py`
   (3); suite **487 → 501 green**.
3. ✅ **Kill action `tools/kill_switch.py` — DONE 2026-06-04.** `run_kill(trigger,
   reason, poly, ...)` (importable so the watchdog uses the same path) +
   `python -m tools.kill_switch [manual|watchdog]`. Halt-first: write HALT →
   cancel_all → discover (`get_positions`) → flatten each subject to the >60s
   guard (heartbeat `window_end_ts` matched by `token_ids`; unknown token →
   bias-to-sell) → independent re-verify, loud CRITICAL log if not flat.
   Flatten = re-fetch book, cross best bid −1 tick, accept partials, retry to
   `MAX_RETRIES=4` / `PRICE_FLOOR=0.01`. Audit record = `data_runtime/kill_switch.log`
   JSONL (kill_events table deferred). Tests: `test_kill_switch_action.py` (8,
   mock client); suite **501 → 509 green**. CTF fallback (step 5) is a marked hook.
4. ✅ **Watchdog `tools/watchdog.py` — DONE 2026-06-04.** `Watchdog` class:
   `evaluate(now)` pure state machine (waiting → armed → fresh/stale → fire) +
   async `run(max_polls)` loop. Arms after ≥1 valid heartbeat; fires after
   `stale_checks_to_fire` consecutive polls with age > `staleness_secs`;
   fail-safe (missing/corrupt = stale once armed); fires `run_kill("watchdog")`
   once. Thresholds now **config-driven** — full `kill_switch:` block added to
   `config.yaml` + `Config` fields (§6); `run_kill` gained optional
   `guard_secs`/`max_retries` fed from config at the entry point. Tests:
   `test_watchdog.py` (8, fake clock + fake-hung-bot `run()`); suite
   **509 → 517 green**.
5. On-chain CTF fallback for discovery.
6. Windows Scheduled Task to supervise the watchdog (deployment doc).
7. Paper dry-run gate → then trust it for the bounded-loss live probe.

## 9. Known v1 limitations (documented, accepted)

- Orphaned `resolution IS NULL` rows in `trades` after a kill → **manual**
  reconciliation against `kill_events`.
- First real flatten is exercised live (paper can't).
- Watchdog supervised by the OS; nothing supervises the OS supervisor.
- Account-level discovery can't map a sell back to a specific bot trade row.
