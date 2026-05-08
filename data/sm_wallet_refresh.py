"""SM Wallet Refresh — periodic Dune API refresh for L9 registry.

Executes Dune query Q18b (ID 7452414) to fetch the latest SM wallet
addresses that satisfy the qualification criteria. Overwrites the local
CSV and optionally triggers a registry reload.

Requires DUNE_API_KEY environment variable.
"""

import csv
import logging
import os
import time
from pathlib import Path

import httpx

from data.sm_wallets import DEFAULT_CSV_PATH, SMWalletRegistry

logger = logging.getLogger(__name__)

DUNE_QUERY_ID = 7452414
DUNE_API_BASE = "https://api.dune.com/api/v1"
POLL_INTERVAL_S = 5
MAX_POLL_ATTEMPTS = 60

CSV_COLUMNS = [
    "wallet",
    "markets_active",
    "direction_wr_pct",
    "total_trades",
    "total_volume",
    "net_pnl",
    "roi_pct",
    "avg_entry_price",
    "trade_wr_pct",
]


class DuneRefreshError(Exception):
    """Raised when a Dune API refresh fails."""


def _get_api_key() -> str:
    key = os.environ.get("DUNE_API_KEY", "").strip()
    if not key:
        raise DuneRefreshError("DUNE_API_KEY environment variable is not set")
    return key


def _headers(api_key: str) -> dict[str, str]:
    return {"X-Dune-API-Key": api_key}


def execute_query(
    query_id: int = DUNE_QUERY_ID,
    timeout_s: float = 30.0,
) -> str:
    """Trigger a Dune query execution.

    Args:
        query_id: Dune query ID to execute.
        timeout_s: HTTP request timeout.

    Returns:
        Execution ID string.
    """
    api_key = _get_api_key()
    url = f"{DUNE_API_BASE}/query/{query_id}/execute"

    resp = httpx.post(url, headers=_headers(api_key), timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    execution_id = data.get("execution_id")
    if not execution_id:
        raise DuneRefreshError(f"No execution_id in Dune response: {data}")

    logger.info("Dune query %d executing: %s", query_id, execution_id)
    return execution_id


def poll_execution(
    execution_id: str,
    poll_interval: float = POLL_INTERVAL_S,
    max_attempts: int = MAX_POLL_ATTEMPTS,
    timeout_s: float = 30.0,
) -> list[dict]:
    """Poll a Dune execution until complete, then return rows.

    Args:
        execution_id: Dune execution ID from execute_query().
        poll_interval: Seconds between status checks.
        max_attempts: Maximum number of polls before giving up.
        timeout_s: HTTP request timeout per poll.

    Returns:
        List of row dicts from the query result.
    """
    api_key = _get_api_key()
    url = f"{DUNE_API_BASE}/execution/{execution_id}/results"

    for attempt in range(1, max_attempts + 1):
        resp = httpx.get(url, headers=_headers(api_key), timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()

        state = data.get("state", "UNKNOWN")
        if state == "QUERY_STATE_COMPLETED":
            rows = data.get("result", {}).get("rows", [])
            logger.info(
                "Dune execution %s complete: %d rows returned",
                execution_id, len(rows),
            )
            return rows

        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
            raise DuneRefreshError(
                f"Dune execution {execution_id} ended with state: {state}"
            )

        logger.debug(
            "Dune execution %s: state=%s (attempt %d/%d)",
            execution_id, state, attempt, max_attempts,
        )
        time.sleep(poll_interval)

    raise DuneRefreshError(
        f"Dune execution {execution_id} did not complete within "
        f"{max_attempts * poll_interval}s"
    )


def write_csv(rows: list[dict], csv_path: Path = DEFAULT_CSV_PATH) -> int:
    """Write Dune result rows to the SM wallet CSV.

    Args:
        rows: List of row dicts from Dune API.
        csv_path: Destination CSV path.

    Returns:
        Number of rows written.
    """
    if not rows:
        logger.warning("No rows to write — CSV not updated")
        return 0

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info("SM wallet CSV written: %d rows → %s", len(rows), csv_path)
    return len(rows)


def refresh_wallets(
    registry: SMWalletRegistry | None = None,
    csv_path: Path = DEFAULT_CSV_PATH,
    query_id: int = DUNE_QUERY_ID,
) -> int:
    """Full refresh: execute Dune query, write CSV, reload registry.

    Args:
        registry: Optional registry to reload after CSV update.
        csv_path: Path for the CSV file.
        query_id: Dune query ID to execute.

    Returns:
        Number of wallets in the updated registry/CSV.

    Raises:
        DuneRefreshError: If any step fails. Existing CSV is preserved
            on failure (Dune returns no rows or API is down).
    """
    logger.info("Starting SM wallet refresh from Dune query %d", query_id)

    execution_id = execute_query(query_id=query_id)
    rows = poll_execution(execution_id)

    if not rows:
        raise DuneRefreshError(
            "Dune query returned 0 rows — refusing to overwrite existing CSV"
        )

    count = write_csv(rows, csv_path=csv_path)

    if registry is not None:
        registry.refresh(csv_path=csv_path)
        count = registry.wallet_count
        logger.info("Registry reloaded: %d wallets after filtering", count)

    return count
