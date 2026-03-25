import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    market_id TEXT NOT NULL,
    market_slug TEXT,
    side TEXT NOT NULL,                    -- 'YES' or 'NO'
    entry_price REAL NOT NULL,
    size_usdc REAL NOT NULL,
    num_shares REAL,
    oracle_lag_signal REAL,
    momentum_signal REAL,
    liquidation_signal REAL,
    combined_signal REAL,
    estimated_prob_up REAL,
    market_implied_prob REAL,
    edge REAL,
    time_remaining_secs REAL,
    resolution TEXT,                       -- 'UP', 'DOWN', or NULL if pending
    pnl REAL,
    is_paper INTEGER NOT NULL DEFAULT 1,   -- 1=paper, 0=live
    btc_price_at_entry REAL,
    oracle_price_at_entry REAL,
    oracle_price_at_open REAL,
    order_id TEXT,
    fill_price REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time TEXT NOT NULL,
    end_time TEXT,
    mode TEXT NOT NULL,                    -- 'paper' or 'live'
    starting_bankroll REAL,
    ending_bankroll REAL,
    total_trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    total_pnl REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS signal_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    estimated_prob_up REAL,
    market_implied_prob REAL,
    oracle_lag_signal REAL,
    momentum_signal REAL,
    liquidation_signal REAL,
    combined_signal REAL,
    time_remaining_secs REAL,
    trade_placed INTEGER DEFAULT 0,        -- 1 if trade was placed
    actual_resolution TEXT,                 -- filled post-resolution
    btc_price REAL,
    oracle_price REAL,
    market_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_market_id ON trades(market_id);
CREATE INDEX IF NOT EXISTS idx_signal_log_timestamp ON signal_log(timestamp);
"""


class Database:
    """SQLite trade log and signal validation store."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def connect(self):
        """Open connection and create tables if needed."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def insert_trade(self, **kwargs) -> int:
        """Insert a trade record. Returns row id."""
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join("?" for _ in kwargs)
        sql = f"INSERT INTO trades ({cols}) VALUES ({placeholders})"
        cursor = self.conn.execute(sql, list(kwargs.values()))
        self.conn.commit()
        return cursor.lastrowid

    def insert_signal(self, **kwargs) -> int:
        """Insert a signal log entry. Returns row id."""
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join("?" for _ in kwargs)
        sql = f"INSERT INTO signal_log ({cols}) VALUES ({placeholders})"
        cursor = self.conn.execute(sql, list(kwargs.values()))
        self.conn.commit()
        return cursor.lastrowid

    def get_session_stats(self, mode: str = "paper") -> dict:
        """Get aggregate stats for current session."""
        sql = """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl), 0) as total_pnl
            FROM trades
            WHERE is_paper = ?
        """
        is_paper = 1 if mode == "paper" else 0
        row = self.conn.execute(sql, (is_paper,)).fetchone()
        return dict(row) if row else {}
