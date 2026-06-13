# Migrate the execution layer to py-clob-client-v2 (CLOB V2)

**Status:** accepted (2026-06-13) — implemented via /tdd, live-smoke verified

## Context

Polymarket migrated its CLOB to **V2** (~2026-04-28), bumping the server's
expected EIP-712 **order** schema (Exchange domain version 1→2; order struct
fields changed). `py-clob-client` (v1) signs the old V1 order, so every
`post_order` was rejected with `400 'invalid order version, please use the
latest clob-client'`. We were already on the latest v1 (0.34.6), and the v1
repo was **archived / read-only (May 2026)** — no upstream fix. Auth and reads
were unaffected (that domain is unchanged), which is why the live smoke reached
the POST before failing. The official successor is **`py-clob-client-v2`**
(PyPI 1.0.1, Polymarket Engineering).

## Decision

Migrate the execution layer to `py-clob-client-v2`. The `PolymarketClient`
wrapper's **public interface is unchanged**; only the internal SDK calls move,
so `core/execution.py`, the kill switch, and the runner are unaffected except
where noted. Chosen over the newer unified `py-sdk` (recommended for greenfield)
because v2 is the minimal-delta path for our existing wrapper (anti-bloat).

Key mappings (`core/polymarket_client.py`):
- `ClobClient(host=, chain_id=, key=, signature_type=, funder=)` — proxy/Magic
  wallet (`signature_type=1`) supported; `create_or_derive_api_key()` +
  `set_api_creds()`.
- Order placement collapses to one `create_and_post_order(OrderArgs(...),
  order_type="GTC")` (this carries the new V2 signing — the actual fix);
  `side` → the `Side` IntEnum; order type passed as the plain string. The
  local shadowing `OrderType`/`OrderSide` enums (the v1 J30 bug) are deleted.
- `cancel_order(OrderPayload(orderID=...))` (v1's bare-string `cancel` removed);
  `get_orders` → `get_open_orders`; `get_balance_allowance` unchanged.

**Fee model — PnL-only (decision: keep the EV gate stable).** V2 sets taker
fees by the protocol at match time, exposed per token via `get_fee_rate_bps` /
`get_fee_exponent`; the per-share fee is `rate·(p·(1−p))^exponent` (verified
against v2 `fees.py`). The legacy `fee_per_share = 0.07·p·(1−p)` is exactly
this with rate=0.07, exponent=1. The **live** fee feeds **recorded PnL only**
(`LiveExecutionEngine.set_market_fee`, fetched once per window by the runner,
applied to live executors); the **entry EV gate and paper PnL keep 0.07**, so
trade selection and the K/K2 paper A/B are undisturbed and the fixed-output fee
regression test stays valid.

## Why

- **Why migrate, not patch v1:** v1 is archived and signs a schema the server
  rejects; hand-patching EIP-712 internals on an unmaintained lib that signs
  live-capital orders is fragile and unsupported. v2 is the official successor.
- **Why PnL-only fee:** raising the rate 7%→~10% in the EV gate would change
  which trades the bots take mid-A/B (a behaviour change in guarded
  signal-adjacent code). Recorded PnL is where the live fee is unambiguously
  more correct; the gate keeps the validated constant until a deliberate,
  A/B-tested change.
- **Why v2, not py-sdk:** our wrapper already mirrors the v1/v2 shape; the
  unified SDK is a larger rewrite for capability we don't need.

## Alternatives rejected

- Stay on v1 / monkeypatch its EIP-712 version (archived, unsupported, signing
  live capital on a guess).
- Migrate to the unified `py-sdk` (greenfield-oriented; larger delta).
- Apply the live fee to the EV gate too (changes trade selection mid-A/B).

## Consequences

- All bots remain paper; go-live unblocked on the SDK front.
- Live-smoke verified (2026-06-13): order POST succeeds; `get_balance` reports
  the funded **$100.31** (collateral tradeable under V2 — **no pUSD wrapping
  needed**); orderbook is now a **dict** (both `best_bid`/`best_ask` extractors
  already handle dict-or-object); the POST body lacks `size_matched`/`price`
  (fixed: `_execute_fok` now reads the authoritative order record, like
  `_execute_gtc` already did).
- Lockfile surface shrank (dropped `requests`/`urllib3`/`charset-normalizer`/
  `py-builder-signing-sdk`; `eth-abi` held at 5.2.0).
- Supersedes the "py-clob-client integration" choice; see decisions_BTC J32 and
  ADR-0003 (live-mode wiring) which this sits under. Remaining pre-flip
  checklist (watchdog task, kill-switch dry-run, /gstack-cso re-run, go-live
  grill) is unchanged.
