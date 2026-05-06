"""Fetch historical Chainlink BTC/USD oracle prices from Polygon.

Queries the same aggregator contract used by Polymarket for 5-minute
BTC Up/Down market resolution. Uses getRoundData() to retrieve every
price update with exact timestamps — capturing the real oracle update
cadence (irregular gaps, heartbeat delays) that drives the oracle lag
signal.

Usage:
    python -m backtest.fetch_chainlink_history

Output:
    backtest/data/chainlink_btc_history.csv

Columns:
    timestamp       Unix timestamp of the oracle update
    timestamp_utc   Human-readable ISO 8601
    price           BTC/USD price (8 decimal precision)
    round_id        Chainlink round ID (encodes phase + sequence)
"""

import csv
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "chainlink_btc_history.csv")

# Same contract and RPCs as the live bot (data/chainlink_oracle.py)
AGGREGATOR_PROXY = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
DECIMALS = 8

# Function selectors
LATEST_ROUND_DATA_SEL = "0xfeaf968c"  # latestRoundData()
GET_ROUND_DATA_SEL = "0x9a6fc8f5"     # getRoundData(uint80)

# Polygon RPCs — same as live bot, ordered by reliability
RPCS = [
    "https://polygon.drpc.org",
    "https://1rpc.io/matic",
    "https://polygon-bor-rpc.publicnode.com",
]

BATCH_SIZE = 50  # Rounds per batch JSON-RPC request


# ─── RPC Helpers ─────────────────────────────────────────────────────


def _encode_get_round_data(round_id: int) -> str:
    """Encode getRoundData(uint80) calldata."""
    return GET_ROUND_DATA_SEL + format(round_id, "064x")


def _decode_round_data(hex_result: str) -> tuple[int, float, int, int] | None:
    """Decode getRoundData response into (roundId, price, startedAt, updatedAt).

    Returns None if the response is empty or invalid.
    """
    if not hex_result or hex_result == "0x" or len(hex_result) < 322:
        return None

    data = hex_result[2:]  # Strip 0x prefix
    # 5 x 32-byte (64 hex char) fields:
    #   roundId, answer, startedAt, updatedAt, answeredInRound
    round_id = int(data[0:64], 16)
    answer = int(data[64:128], 16)
    started_at = int(data[128:192], 16)
    updated_at = int(data[192:256], 16)

    if answer == 0 or updated_at == 0:
        return None

    price = answer / (10 ** DECIMALS)
    return round_id, price, started_at, updated_at


def _call_rpc(
    client: httpx.Client,
    rpc_url: str,
    payload: dict | list,
    timeout: float = 30.0,
) -> dict | list | None:
    """Send a JSON-RPC request (single or batch). Returns parsed JSON."""
    for attempt in range(3):
        try:
            resp = client.post(rpc_url, json=payload, timeout=timeout)
            if resp.status_code == 429:
                wait = min(60, 5 * (attempt + 1))
                logger.warning(f"Rate limited on {rpc_url}, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug(f"RPC error on {rpc_url} (attempt {attempt+1}): {e}")
            time.sleep(1)
    return None


def call_single(
    client: httpx.Client,
    rpc_url: str,
    calldata: str,
) -> str | None:
    """Single eth_call, returns hex result or None."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": AGGREGATOR_PROXY, "data": calldata}, "latest"],
    }
    result = _call_rpc(client, rpc_url, payload)
    if result and "result" in result:
        return result["result"]
    return None


def call_batch(
    client: httpx.Client,
    rpc_url: str,
    round_ids: list[int],
) -> list[tuple[int, float, int, int]]:
    """Batch getRoundData calls. Returns list of (roundId, price, startedAt, updatedAt)."""
    batch = []
    for i, rid in enumerate(round_ids):
        batch.append({
            "jsonrpc": "2.0",
            "id": i,
            "method": "eth_call",
            "params": [
                {"to": AGGREGATOR_PROXY, "data": _encode_get_round_data(rid)},
                "latest",
            ],
        })

    raw = _call_rpc(client, rpc_url, batch, timeout=60.0)
    if not raw or not isinstance(raw, list):
        return []

    results = []
    for item in raw:
        hex_result = item.get("result", "")
        decoded = _decode_round_data(hex_result)
        if decoded:
            results.append(decoded)
    return results


# ─── Phase & Round Navigation ───────────────────────────────────────


def get_latest_round(client: httpx.Client, rpc_url: str) -> tuple[int, int, int] | None:
    """Get latest round. Returns (phaseId, aggregatorRoundId, updatedAt)."""
    hex_result = call_single(client, rpc_url, LATEST_ROUND_DATA_SEL)
    decoded = _decode_round_data(hex_result)
    if not decoded:
        return None

    round_id, price, started_at, updated_at = decoded
    phase_id = round_id >> 64
    agg_round = round_id & ((1 << 64) - 1)

    logger.info(
        f"Latest round: phase={phase_id}, round={agg_round}, "
        f"price=${price:,.2f}, updated={datetime.fromtimestamp(updated_at, tz=timezone.utc).isoformat()}"
    )
    return phase_id, agg_round, updated_at


def make_round_id(phase_id: int, agg_round: int) -> int:
    """Construct a composite Chainlink round ID."""
    return (phase_id << 64) | agg_round


def binary_search_round(
    client: httpx.Client,
    rpc_url: str,
    phase_id: int,
    lo: int,
    hi: int,
    target_ts: int,
) -> int:
    """Find the first round in the phase with updatedAt >= target_ts.

    Returns the aggregator round number (not composite ID).
    """
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        rid = make_round_id(phase_id, mid)
        hex_result = call_single(client, rpc_url, _encode_get_round_data(rid))
        decoded = _decode_round_data(hex_result)

        if not decoded:
            # Round doesn't exist — try lower
            hi = mid - 1
            continue

        _, _, _, updated_at = decoded
        if updated_at >= target_ts:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1

        time.sleep(0.05)

    return best


# ─── Main Fetch Logic ────────────────────────────────────────────────


def fetch_chainlink_history(
    start_ts: int,
    end_ts: int,
    output_path: str = OUTPUT_PATH,
) -> int:
    """Fetch all Chainlink BTC/USD oracle updates in a timestamp range.

    Args:
        start_ts: Start Unix timestamp.
        end_ts: End Unix timestamp.
        output_path: CSV output path.

    Returns:
        Total number of price updates fetched.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try each RPC until we find one that works
    rpc_url = RPCS[0]

    with httpx.Client(timeout=30.0) as client:
        # Step 1: Get latest round info
        latest = None
        for rpc in RPCS:
            latest = get_latest_round(client, rpc)
            if latest:
                rpc_url = rpc
                break

        if not latest:
            logger.error("Could not connect to any Polygon RPC")
            return 0

        phase_id, max_agg_round, latest_updated = latest

        # Step 2: Check if current phase covers our date range
        first_rid = make_round_id(phase_id, 1)
        hex_result = call_single(client, rpc_url, _encode_get_round_data(first_rid))
        first_decoded = _decode_round_data(hex_result)

        phases_to_fetch: list[tuple[int, int, int]] = []  # (phase_id, start_round, end_round)

        if first_decoded:
            _, _, _, first_updated = first_decoded
            logger.info(
                f"Phase {phase_id} started at "
                f"{datetime.fromtimestamp(first_updated, tz=timezone.utc).isoformat()}"
            )

            if first_updated > start_ts:
                # Need previous phase too
                logger.info(f"Phase {phase_id} doesn't cover full range, checking phase {phase_id - 1}")
                prev_phase = phase_id - 1
                if prev_phase > 0:
                    # Find last round of previous phase by probing
                    probe = 1
                    last_valid = 0
                    while probe < 10_000_000:
                        rid = make_round_id(prev_phase, probe)
                        hr = call_single(client, rpc_url, _encode_get_round_data(rid))
                        d = _decode_round_data(hr)
                        if d:
                            last_valid = probe
                            probe *= 2
                        else:
                            break
                        time.sleep(0.05)

                    # Binary search for exact last round
                    if last_valid > 0:
                        lo_p, hi_p = last_valid, probe
                        while lo_p < hi_p:
                            mid_p = (lo_p + hi_p + 1) // 2
                            rid = make_round_id(prev_phase, mid_p)
                            hr = call_single(client, rpc_url, _encode_get_round_data(rid))
                            if _decode_round_data(hr):
                                lo_p = mid_p
                            else:
                                hi_p = mid_p - 1
                            time.sleep(0.05)

                        prev_last = lo_p
                        prev_start = binary_search_round(
                            client, rpc_url, prev_phase, 1, prev_last, start_ts
                        )
                        phases_to_fetch.append((prev_phase, prev_start, prev_last))
                        logger.info(
                            f"Previous phase {prev_phase}: rounds {prev_start}-{prev_last}"
                        )

            # Current phase: from binary-searched start to latest
            current_start = binary_search_round(
                client, rpc_url, phase_id, 1, max_agg_round, start_ts
            )
            phases_to_fetch.append((phase_id, current_start, max_agg_round))
            logger.info(
                f"Current phase {phase_id}: rounds {current_start}-{max_agg_round}"
            )
        else:
            logger.error("Could not read first round of current phase")
            return 0

        # Step 3: Fetch all rounds via batch requests
        total_fetched = 0
        all_rows: list[tuple[int, str, float, int]] = []

        for pid, start_round, end_round in phases_to_fetch:
            total_rounds = end_round - start_round + 1
            logger.info(
                f"Fetching phase {pid}: {total_rounds:,} rounds "
                f"({start_round} → {end_round})"
            )

            current = start_round
            batch_attempt = BATCH_SIZE
            use_batch = True

            while current <= end_round:
                batch_end = min(current + batch_attempt - 1, end_round)
                round_ids = [make_round_id(pid, r) for r in range(current, batch_end + 1)]

                if use_batch and len(round_ids) > 1:
                    results = call_batch(client, rpc_url, round_ids)
                    if not results and len(round_ids) > 1:
                        # Batch might not be supported — try smaller or single
                        if batch_attempt > 10:
                            batch_attempt = 10
                            logger.info("Reducing batch size to 10")
                            continue
                        else:
                            use_batch = False
                            logger.info("Falling back to sequential calls")
                            continue
                else:
                    # Sequential fallback
                    results = []
                    for rid in round_ids:
                        hr = call_single(
                            client, rpc_url, _encode_get_round_data(rid)
                        )
                        d = _decode_round_data(hr)
                        if d:
                            results.append(d)
                        time.sleep(0.05)

                for round_id, price, started_at, updated_at in results:
                    if start_ts <= updated_at <= end_ts:
                        utc_str = datetime.fromtimestamp(
                            updated_at, tz=timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%SZ")
                        all_rows.append((updated_at, utc_str, price, round_id))
                        total_fetched += 1

                current = batch_end + 1

                # Progress logging
                done = current - start_round
                if done % 5000 < batch_attempt:
                    pct = done / max(1, total_rounds) * 100
                    logger.info(
                        f"  Phase {pid}: {done:,}/{total_rounds:,} ({pct:.1f}%) "
                        f"— {total_fetched:,} in range"
                    )

                time.sleep(0.1)

        # Step 4: Sort by timestamp and write CSV
        all_rows.sort(key=lambda r: r[0])

        # Deduplicate by timestamp (same update can appear in overlapping phases)
        seen_ts: set[int] = set()
        deduped: list[tuple[int, str, float, int]] = []
        for row in all_rows:
            if row[0] not in seen_ts:
                seen_ts.add(row[0])
                deduped.append(row)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "timestamp_utc", "price", "round_id"])
            for ts, utc, price, rid in deduped:
                writer.writerow([ts, utc, f"{price:.8f}", rid])

        logger.info(
            f"Done. {len(deduped):,} unique oracle updates saved to {output_path}"
        )
        if deduped:
            first_dt = deduped[0][1]
            last_dt = deduped[-1][1]
            logger.info(f"Range: {first_dt} → {last_dt}")

            # Update cadence stats
            if len(deduped) >= 2:
                gaps = [
                    deduped[i + 1][0] - deduped[i][0]
                    for i in range(len(deduped) - 1)
                ]
                avg_gap = sum(gaps) / len(gaps)
                min_gap = min(gaps)
                max_gap = max(gaps)
                logger.info(
                    f"Update cadence: avg={avg_gap:.1f}s, "
                    f"min={min_gap}s, max={max_gap}s"
                )

        return len(deduped)


def main() -> None:
    """Fetch oracle history covering the PolyBackTest Pro date range."""
    # Determine date range from markets CSV
    markets_path = os.path.join(DATA_DIR, "polybacktest_markets.csv")
    if not os.path.exists(markets_path):
        logger.error(f"Markets file not found: {markets_path}")
        logger.error("Run: python -m backtest.fetch_polybacktest_pro first")
        return

    earliest = None
    latest = None
    with open(markets_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["start_time", "end_time"]:
                t = row.get(key, "")
                if t:
                    try:
                        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
                        if earliest is None or dt < earliest:
                            earliest = dt
                        if latest is None or dt > latest:
                            latest = dt
                    except ValueError:
                        pass

    if not earliest or not latest:
        logger.error("Could not determine date range from markets CSV")
        return

    # Buffer: 30 min before (for signal warm-up) and 5 min after
    start_ts = int((earliest - timedelta(minutes=30)).timestamp())
    end_ts = int((latest + timedelta(minutes=5)).timestamp())

    days = (end_ts - start_ts) / 86400
    logger.info(f"Date range: {earliest.isoformat()} to {latest.isoformat()}")
    logger.info(f"With buffer: {days:.1f} days")

    fetch_chainlink_history(start_ts, end_ts)


if __name__ == "__main__":
    main()
