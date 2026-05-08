"""SM Decision Logger — tracks L9 confirmation decisions for threshold monitoring.

Logs every SM confirmation decision with full context (flow fractions,
price, threshold, decision) to a SQLite database. This data is used to
evaluate whether the 60% agreement threshold is optimal.

The 60% threshold was set as a sensible default, NOT empirically
optimised. This logger provides the data to tune it in production.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from strategy.sm_confirmation import (
    PositionSide,
    SMConfirmationConfig,
    SMDecision,
    SMFlowState,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "sm_decisions.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sm_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    market_id TEXT,
    window_start TEXT,
    check_minute INTEGER,
    position_side TEXT NOT NULL,
    market_price REAL NOT NULL,
    sm_yes_volume REAL NOT NULL,
    sm_no_volume REAL NOT NULL,
    sm_total_volume REAL NOT NULL,
    sm_yes_fraction REAL NOT NULL,
    sm_num_wallets INTEGER NOT NULL,
    agreement_threshold REAL NOT NULL,
    decision TEXT NOT NULL,
    position_fraction REAL NOT NULL
)
"""


class SMDecisionLogger:
    """Logs SM confirmation decisions to SQLite for threshold analysis.

    Args:
        db_path: Path to SQLite database. Created if it doesn't exist.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(CREATE_TABLE_SQL)
        self._conn.commit()
        logger.info("SM decision logger initialised: %s", self._db_path)

    def log_decision(
        self,
        position_side: PositionSide,
        market_price: float,
        flow: SMFlowState,
        decision: SMDecision,
        config: SMConfirmationConfig,
        market_id: str | None = None,
        window_start: str | None = None,
        check_minute: int | None = None,
    ) -> None:
        """Record a single SM confirmation decision."""
        if position_side == PositionSide.YES:
            position_fraction = flow.yes_fraction
        else:
            position_fraction = flow.no_fraction

        self._conn.execute(
            """
            INSERT INTO sm_decisions (
                timestamp, market_id, window_start, check_minute,
                position_side, market_price,
                sm_yes_volume, sm_no_volume, sm_total_volume,
                sm_yes_fraction, sm_num_wallets,
                agreement_threshold, decision, position_fraction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                market_id,
                window_start,
                check_minute,
                position_side.value,
                market_price,
                flow.yes_volume,
                flow.no_volume,
                flow.total_volume,
                flow.yes_fraction,
                flow.num_wallets,
                config.agreement_threshold,
                decision.value,
                position_fraction,
            ),
        )
        self._conn.commit()

    def get_decision_counts(self) -> dict[str, int]:
        """Return counts of each decision type."""
        cursor = self._conn.execute(
            "SELECT decision, COUNT(*) FROM sm_decisions GROUP BY decision"
        )
        return dict(cursor.fetchall())

    def get_threshold_analysis(self) -> list[dict]:
        """Return per-decision stats for threshold tuning.

        Groups decisions by 5% position_fraction buckets and shows
        the distribution of HOLD/EXIT/IGNORE at each level.
        """
        cursor = self._conn.execute(
            """
            SELECT
                CAST(ROUND(position_fraction * 20) AS INTEGER) * 5 AS pct_bucket,
                decision,
                COUNT(*) AS count
            FROM sm_decisions
            WHERE decision != 'IGNORE'
            GROUP BY pct_bucket, decision
            ORDER BY pct_bucket
            """
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                "position_fraction_pct": row[0],
                "decision": row[1],
                "count": row[2],
            })
        return results

    def close(self) -> None:
        self._conn.close()
