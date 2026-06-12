"""Per-round GTC fill telemetry sink (decisions_BTC J27).

Every ``LiveExecutionEngine._execute_gtc`` call is one posting round of the
live re-post loop. Each round appends one JSONL record here so chase EV and
the chase-at-floor hybrid can be computed offline from real fills.

Telemetry is an observer: writes are best-effort and never raise, mirroring
the kill-switch audit log (``tools/kill_switch.py``).
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ROUNDS_LOG_PATH = Path("data_runtime") / "execution_rounds.log"


def log_round(record: dict, *, path: Path | None = None) -> None:
    """Append one JSONL round record. Best-effort — never raises."""
    if path is None:
        path = ROUNDS_LOG_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:  # telemetry must never break order flow
        logger.error(f"execution_rounds write failed: {e}")


def best_ask(book: object) -> float | None:
    """Extract the lowest ask price from an orderbook.

    Handles both the raw CLOB REST dict (``{"asks": [{"price","size"}]}``)
    and a py-clob-client ``OrderBookSummary`` object (``.asks`` of objects
    with ``.price``). Returns None when unavailable — never raises.
    """
    try:
        if book is None:
            return None
        asks = book.get("asks") if isinstance(book, dict) else getattr(book, "asks", None)
        if not asks:
            return None
        prices = []
        for level in asks:
            raw = level.get("price") if isinstance(level, dict) else getattr(level, "price", None)
            if raw is not None:
                prices.append(float(raw))
        return min(prices) if prices else None
    except Exception:
        return None
