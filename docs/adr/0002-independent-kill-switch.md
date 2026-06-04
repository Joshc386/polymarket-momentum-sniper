# Independent out-of-process kill switch

**Status:** accepted (2026-06-03) — plan approved, not yet built

## Decision

The bot gets an **emergency kill switch that runs as a separate OS process**,
independent of the trading process. It can be triggered manually
(`python -m tools.kill_switch`) or automatically by a **watchdog** process that
monitors a heartbeat. When it fires it: (1) writes a sticky `HALT` flag, then
(2) cancels all resting orders, then (3) flattens open positions, then
(4) re-verifies flat — all by talking to Polymarket directly with its **own**
`PolymarketClient` (same `.env` key), never by reaching into the bot's
in-memory state.

Signalling between processes is via **plain files in `data_runtime/`**:
- `heartbeat.json` — `{ts, window_end_ts, token_ids[]}`, written atomically by
  the bot every loop iteration.
- `HALT` — sticky flag; once written, trading stays disabled until a human
  deletes it. The bot checks it before placing *any* order and exits gracefully
  when it sees it.

Position discovery is **account-level ground truth**: `poly_client.get_positions()`
as primary, on-chain CTF (ERC-1155) balances via Polygon RPC as fallback, the
bot's `trades` DB as an advisory cross-check only.

> **Amendment 2026-06-04 (spike result).** The empirical spike (build step 1)
> found the original `get_positions()` was a **silent stub**: it called
> `ClobClient.get_positions()`, which does not exist in py-clob-client, so the
> `AttributeError` was swallowed and it always returned `[]`. Positions are not
> a CLOB concept. `get_positions()` was rewritten to query Polymarket's **public
> Data API** (`data-api.polymarket.com/positions?user=<funder>`) — account-wide,
> no auth required — and verified against live public data (HTTP 200, rows carry
> `asset`/`size`/`conditionId`, parser confirmed). The fallback (on-chain CTF)
> and advisory source (trades DB) are unchanged. **Going-live dependency:**
> discovery needs `POLYMARKET_FUNDER_ADDRESS` set (empty → returns `[]`); it is
> currently empty in the paper-only `.env`. Flatten is a **best-effort aggressive marketable sell
with bounded retries that accepts partial fills**, gated by a **>60s-to-resolution
guard** (closer than 60s → cancel + halt and let the 5-minute market resolve).
Residual exposure is always backstopped by **5-minute auto-resolution**.

## Why

The existing shutdown (`main.py:580-590`, `:1257-1260`; `multi_runner.py:260-264`)
is a SIGINT/SIGTERM handler **inside the same asyncio event loop as the trading
logic**. If that loop hangs — deadlock, a network call that never returns, an
exception storm — the handler never runs and there is no way to stop trading or
flatten exposure. For live capital on 5-minute markets that is an unacceptable
single point of failure. The Execution Agent constraint is explicit: *"the kill
switch must work independently of the main bot — it cannot rely on the same
process."* Only a separate process can act when the trading process is the thing
that is broken.

Files (not a socket/RPC/DB) are chosen for the signalling channel because they
are the most robust primitive available when one side may be wedged: no
connection to refuse, no lock to deadlock on, atomic via `os.replace`, and
trivially inspectable by a human mid-incident.

Account-level discovery (not the bot's memory) is required because a hung bot's
in-memory `pending_trade` is exactly the state you cannot trust at kill time.

## Considered and rejected

- **In-process kill switch / rely on the existing SIGINT handler** — rejected;
  it shares the event loop that may be hung, which is the whole failure mode.
- **Cross-process lock or socket/RPC channel** between bot and kill switch —
  rejected; adds a connection that can refuse/deadlock precisely when things are
  broken. Halt-first ordering + a check-before-every-order discipline on the bot
  closes the recovery race without locking, and the 5-min auto-resolution
  backstops anything that slips through.
- **FOK for the flatten sell** (as first scoped) — rejected; FOK is
  all-or-nothing, so on a thin book it kills and leaves you 100% exposed. Flatten
  needs *exit* certainty, not *price* certainty: an aggressive marketable sell
  that accepts partial fills and retries is correct. (FOK stays correct for
  *entry*.)
- **Always flatten, no time guard** — rejected; force-selling a position seconds
  from resolution crosses a thin spread for no benefit. >60s guard + let-resolve
  avoids pointless near-resolution unwinds.
- **Trust the bot's trades DB for open positions** — rejected as the primary
  source; a hung bot is when its DB is most likely mid-write/stale. Advisory
  cross-check only.
- **Auto-clearing halt / bot auto-resume** — rejected; a flapping hang could
  thrash in and out of live trading unsupervised. Sticky flag, manual clear,
  human reviews why it fired before re-enabling.
- **Auto-restart the bot under OS supervision** — rejected; only the *watchdog*
  is supervised (restart-on-crash). Restarting the bot would defeat the sticky
  halt. A dead bot → watchdog fires → correct.

## Consequences

- New surface: `tools/kill_switch.py` (kill action), `tools/watchdog.py`
  (heartbeat monitor + auto-trigger), heartbeat write + pre-order HALT check in
  `multi_runner.py`, and a Windows Scheduled Task supervising the watchdog.
- A kill event leaves orphaned `resolution IS NULL` rows in the bot's `trades`
  table (the bot was dead and could not record the forced exits). The kill
  switch writes its own authoritative record (`data_runtime/kill_switch.log` +
  `kill_events`); reconciling those rows is a **documented manual step** in v1.
- The first real flatten happens **live** (paper has no real account positions
  to cancel/flatten — those paths are no-ops in paper). Mock-client unit tests +
  the small probe size + the 5-min backstop bound that risk.
- Risk thresholds (staleness 20s, 3 checks, 2s poll, 60s guard, retry count) are
  **config-driven**, not hardcoded.
- Out of v1 (deliberately): remote/web trigger, TWAP/partial-unwind strategies,
  per-bot halt, notifications beyond a loud log line, auto-reconciliation of
  orphaned trade rows.
