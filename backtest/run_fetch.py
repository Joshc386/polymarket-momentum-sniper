"""Self-contained batch fetch — runs one batch at a time, resumable."""
import csv
import os
import sys
import time
from datetime import datetime, timezone

import requests

API_KEY = "pdm_CTfGOrxp1kcLpH44m9lkoDJScjGuiyvi"
API_BASE = "https://api.polybacktest.com"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OFFSETS = [30, 60, 120, 180, 240]
FIELDS = [
    "market_id", "offset_sec", "seconds_remaining", "snapshot_time",
    "btc_price", "price_up", "price_down",
    "up_best_ask", "up_best_bid", "down_best_ask", "down_best_bid",
    "up_bid_depth_5", "up_ask_depth_5", "down_bid_depth_5", "down_ask_depth_5",
]

# How many markets to fetch per run
BATCH_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def get(endpoint, params=None):
    headers = {"X-API-Key": API_KEY}
    for attempt in range(10):
        time.sleep(0.5)
        try:
            r = requests.get(f"{API_BASE}{endpoint}", headers=headers, params=params, timeout=30)
            if r.status_code == 429:
                w = max(int(r.headers.get("Retry-After", 10)) + 5, 15)
                print(f"  429, wait {w}s", flush=True)
                time.sleep(w)
                continue
            return r.json() if r.status_code == 200 else None
        except Exception as e:
            print(f"  err: {e}", flush=True)
            time.sleep(5)
    return None


def fetch_snaps(market_id, start_time_str):
    start = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
    rows = []
    for offset in OFFSETS:
        ts = datetime.fromtimestamp(start.timestamp() + offset, tz=timezone.utc).isoformat()
        data = get(f"/v2/markets/{market_id}/snapshot-at/{ts}", {"coin": "btc"})
        if not data or not data.get("snapshots"):
            continue
        s = data["snapshots"][0]
        ou, od = s.get("orderbook_up") or {}, s.get("orderbook_down") or {}
        ua, ub = ou.get("asks", []), ou.get("bids", [])
        da, db = od.get("asks", []), od.get("bids", [])
        rows.append({
            "market_id": market_id, "offset_sec": offset,
            "seconds_remaining": 300 - offset, "snapshot_time": s["time"],
            "btc_price": s.get("btc_price"),
            "price_up": s.get("price_up"), "price_down": s.get("price_down"),
            "up_best_ask": ua[0]["price"] if ua else "",
            "up_best_bid": ub[0]["price"] if ub else "",
            "down_best_ask": da[0]["price"] if da else "",
            "down_best_bid": db[0]["price"] if db else "",
            "up_bid_depth_5": sum(x["size"] for x in ub[:5]),
            "up_ask_depth_5": sum(x["size"] for x in ua[:5]),
            "down_bid_depth_5": sum(x["size"] for x in db[:5]),
            "down_ask_depth_5": sum(x["size"] for x in da[:5]),
        })
    return rows


def main():
    markets_path = os.path.join(DATA_DIR, "polybacktest_markets.csv")
    snaps_path = os.path.join(DATA_DIR, "polybacktest_snapshots.csv")

    with open(markets_path, "r", encoding="utf-8") as f:
        markets = list(csv.DictReader(f))

    done = set()
    if os.path.exists(snaps_path):
        with open(snaps_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(row["market_id"])

    remaining = [m for m in markets if m["market_id"] not in done]
    batch = remaining[:BATCH_SIZE]

    print(f"Total: {len(markets)} | Done: {len(done)} | Remaining: {len(remaining)} | This batch: {len(batch)}", flush=True)

    if not batch:
        print("Nothing to fetch!", flush=True)
        return

    all_rows = []
    t0 = time.time()

    for i, m in enumerate(batch, 1):
        rows = fetch_snaps(m["market_id"], m["start_time"])
        all_rows.extend(rows)
        if i % 10 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta_batch = (len(batch) - i) / rate / 60
            eta_total = (len(remaining) - i) / rate / 60
            print(f"  {i}/{len(batch)} ({rate:.2f}/s, batch ETA {eta_batch:.1f}m, total ETA {eta_total:.0f}m)", flush=True)

    # Save
    needs_header = not os.path.exists(snaps_path) or os.path.getsize(snaps_path) == 0
    with open(snaps_path, "w" if needs_header else "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if needs_header:
            w.writeheader()
        w.writerows(all_rows)

    new_done = len(done) + len(batch)
    print(f"Saved {len(all_rows)} snapshots. Progress: {new_done}/{len(markets)} ({new_done/len(markets)*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
