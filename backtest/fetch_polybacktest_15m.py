"""Fetch historical BTC 15-minute market data from PolyBackTest Pro API.

Mirrors fetch_polybacktest_pro.py but for 15m markets instead of 5m.
Only fetches resolved markets and their orderbook snapshots.

Usage:
    python -m backtest.fetch_polybacktest_15m

Output:
    backtest/data/polybacktest_markets_15m.csv   — market metadata + resolutions
    backtest/data/polybacktest_snapshots_15m.csv — orderbook at entry timepoints
"""

import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
logger = logging.getLogger(__name__)

API_BASE = "https://api.polybacktest.com"

API_KEY = os.environ.get("POLYBACKTEST_API_KEY", "")
if not API_KEY:
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line.startswith("POLYBACKTEST_API_KEY="):
                    API_KEY = _line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

COIN = "btc"
MARKET_TYPE = "15m"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WINDOW_SECS = 900  # 15 minutes

# Seconds INTO the 15-min window at which to sample snapshots.
# Primary target: 840s in (60s remaining = "last minute" like 5m strategy).
# Secondary: 780s (120s remaining) and 600s (300s remaining) for timing sweeps.
SNAPSHOT_OFFSETS_SEC = [600, 780, 840]

SNAPSHOT_FIELDS = [
    "market_id", "offset_sec", "seconds_remaining", "snapshot_time",
    "btc_price", "price_up", "price_down",
    "up_best_ask", "up_best_bid", "down_best_ask", "down_best_bid",
    "up_bid_depth_5", "up_ask_depth_5", "down_bid_depth_5", "down_ask_depth_5",
]


def api_get(endpoint: str, params: dict | None = None) -> dict | None:
    """Make authenticated GET request with backoff on 429."""
    url = f"{API_BASE}{endpoint}"
    headers = {"X-API-Key": API_KEY}
    time.sleep(0.3)  # Below documented rate limit

    for attempt in range(10):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                wait = max(retry_after + 5, 15)
                logger.warning(f"429 (attempt {attempt+1}) — waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return None
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"Request error (attempt {attempt+1}): {e}")
            time.sleep(5)

    logger.error(f"Failed after 10 attempts: {endpoint}")
    return None


def fetch_all_markets() -> list[dict]:
    """Fetch all resolved BTC 15m markets."""
    markets = []
    offset = 0
    batch_size = 100
    consecutive_failures = 0

    while True:
        data = api_get(
            "/v2/markets",
            params={
                "coin": COIN,
                "market_type": MARKET_TYPE,
                "limit": batch_size,
                "offset": offset,
                "resolved": "true",
            },
        )
        if not data or not data.get("markets"):
            consecutive_failures += 1
            if consecutive_failures >= 3:
                logger.warning(f"3 consecutive failures at offset {offset}, stopping")
                break
            time.sleep(5)
            continue

        consecutive_failures = 0
        batch = data["markets"]
        markets.extend(batch)
        offset += len(batch)
        total = data.get("total", "?")
        logger.info(f"  Markets: {len(markets)}/{total}")
        if len(batch) < batch_size:
            break

    return markets


def fetch_snapshots_for_market(market: dict) -> list[dict]:
    """Fetch orderbook snapshots at the configured offsets."""
    market_id = market["market_id"]
    start_time = datetime.fromisoformat(market["start_time"].replace("Z", "+00:00"))
    rows = []

    for offset_sec in SNAPSHOT_OFFSETS_SEC:
        ts = start_time.timestamp() + offset_sec
        iso_ts = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        data = api_get(
            f"/v2/markets/{market_id}/snapshot-at/{iso_ts}",
            params={"coin": COIN},
        )
        if not data or not data.get("snapshots"):
            continue

        snap = data["snapshots"][0]
        # Guard: API may return None when one side has no liquidity
        ob_up = snap.get("orderbook_up") or {}
        ob_down = snap.get("orderbook_down") or {}

        up_asks = ob_up.get("asks") or []
        up_bids = ob_up.get("bids") or []
        down_asks = ob_down.get("asks") or []
        down_bids = ob_down.get("bids") or []

        rows.append({
            "market_id": market_id,
            "offset_sec": offset_sec,
            "seconds_remaining": WINDOW_SECS - offset_sec,
            "snapshot_time": snap.get("time", iso_ts),
            "btc_price": snap.get("btc_price"),
            "price_up": snap.get("price_up"),
            "price_down": snap.get("price_down"),
            "up_best_ask": up_asks[0]["price"] if up_asks else None,
            "up_best_bid": up_bids[0]["price"] if up_bids else None,
            "down_best_ask": down_asks[0]["price"] if down_asks else None,
            "down_best_bid": down_bids[0]["price"] if down_bids else None,
            "up_bid_depth_5": sum(l["size"] for l in up_bids[:5]),
            "up_ask_depth_5": sum(l["size"] for l in up_asks[:5]),
            "down_bid_depth_5": sum(l["size"] for l in down_bids[:5]),
            "down_ask_depth_5": sum(l["size"] for l in down_asks[:5]),
        })

    return rows


def save_markets_csv(markets: list[dict], path: str) -> None:
    """Save market metadata to CSV."""
    fields = [
        "market_id", "slug", "start_time", "end_time",
        "btc_price_start", "btc_price_end", "winner",
        "final_volume", "final_liquidity",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for m in markets:
            writer.writerow(m)
    logger.info(f"Saved {len(markets)} markets to {path}")


def append_snapshots(
    snapshots: list[dict], path: str, write_header: bool = False
) -> None:
    """Append snapshot rows (supports incremental saves)."""
    mode = "w" if write_header else "a"
    # write_header alone should still create the file with header
    if not snapshots and not write_header:
        return
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_FIELDS)
        if write_header:
            writer.writeheader()
        if snapshots:
            writer.writerows(snapshots)


def load_processed_market_ids(snapshots_path: str) -> set[str]:
    """Return the set of market_ids already present in the snapshots CSV."""
    if not os.path.exists(snapshots_path):
        return set()
    seen: set[str] = set()
    with open(snapshots_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get("market_id")
            if mid:
                seen.add(str(mid))
    return seen


def load_markets_from_csv(path: str) -> list[dict]:
    """Load market metadata from a previously-saved CSV (for resume)."""
    markets: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            markets.append(row)
    return markets


def main() -> None:
    if not API_KEY:
        logger.error("POLYBACKTEST_API_KEY missing (env or .env)")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)
    markets_path = os.path.join(DATA_DIR, "polybacktest_markets_15m.csv")
    snapshots_path = os.path.join(DATA_DIR, "polybacktest_snapshots_15m.csv")

    logger.info(f"Fetching BTC 15-min markets from PolyBackTest Pro")
    logger.info(f"  Snapshot offsets: {SNAPSHOT_OFFSETS_SEC}")

    # 1. Markets metadata — reuse if present, else fetch
    if os.path.exists(markets_path):
        markets = load_markets_from_csv(markets_path)
        logger.info(f"\n[1/2] Reusing cached markets: {len(markets)} rows from {markets_path}")
    else:
        logger.info("\n[1/2] Fetching market metadata...")
        markets = fetch_all_markets()
        if not markets:
            logger.error("No markets returned")
            sys.exit(1)
        save_markets_csv(markets, markets_path)

    # 2. Snapshots — resume from where we left off
    already_done = load_processed_market_ids(snapshots_path)
    logger.info(f"\n[2/2] Fetching snapshots for {len(markets)} markets...")
    if already_done:
        logger.info(f"  Resume: {len(already_done)} markets already processed, "
                    f"will skip them")
    logger.info(f"  Est. time for remaining: "
                f"~{(len(markets)-len(already_done)) * len(SNAPSHOT_OFFSETS_SEC) * 0.4 / 60:.0f} min")

    # Only write header if file is brand new
    write_header = not os.path.exists(snapshots_path)
    if write_header:
        append_snapshots([], snapshots_path, write_header=True)

    total_snaps = 0
    processed_count = len(already_done)
    for i, market in enumerate(markets):
        mid = str(market.get("market_id", ""))
        if mid in already_done:
            continue
        try:
            snaps = fetch_snapshots_for_market(market)
        except Exception as e:
            logger.warning(f"  Error fetching snapshots for {mid}: {e}")
            continue
        if snaps:
            append_snapshots(snaps, snapshots_path, write_header=False)
            total_snaps += len(snaps)
        processed_count += 1

        if processed_count % 50 == 0:
            pct = processed_count / len(markets) * 100
            logger.info(
                f"  Progress: {processed_count}/{len(markets)} markets "
                f"({pct:.0f}%), +{total_snaps} snapshots this run"
            )

    logger.info(f"\nDone.")
    logger.info(f"  Markets: {markets_path}")
    logger.info(f"  Snapshots: {snapshots_path} (+{total_snaps} added this run)")


if __name__ == "__main__":
    main()
