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
    created_at TEXT DEFAULT (datetime('now')),
    -- Enriched snapshot fields (added for strategy analysis)
    orderbook_signal REAL,                 -- L4 orderbook imbalance signal
    sentiment_signal REAL,                 -- L5 cross-exchange sentiment signal
    coinbase_direction REAL,               -- Coinbase price direction confirmation
    regime TEXT,                           -- 'trending', 'ranging', 'volatile', 'low_volatility'
    regime_confidence REAL,                -- Regime classifier confidence 0-1
    risk_size_multiplier REAL,             -- Applied size multiplier (streak/regime/vol)
    consecutive_losses INTEGER,            -- Loss streak count at time of entry
    signal_weights TEXT,                   -- JSON: {"L1":w, "L2":w, "L3":w, "L4":w, "L5":w}
    ob_spread REAL,                        -- Orderbook spread at entry (ask - bid)
    ob_yes_depth REAL,                     -- YES side depth (top 5 levels)
    ob_ask_depth REAL,                     -- Ask side depth (top 5 levels)
    session_trade_num INTEGER,             -- Trade number within this session
    resolved_at TEXT,                      -- UTC timestamp when trade resolved
    -- Full feature snapshot at entry (2026-06-01). NULL = layer not computed
    -- this tick (never 0.0 for absent). See ADR-0001.
    l1_lag_component REAL,
    l1_open_component REAL,
    l4_imbalance REAL,
    l4_flow REAL,
    l4_mid_dev REAL,
    l4_top_pressure REAL,
    l4_thickness REAL,
    l6_fade REAL,
    l7_taker_ratio REAL,
    l8_clob_flow REAL,
    l9b_absorption REAL,
    l10_exhaustion REAL,
    l11_trade_size REAL,
    l12_wallet_flow REAL,
    prob_edge REAL,                        -- |model_prob - market_prob| (gate metric)
    net_ev_per_share REAL,                 -- (q-p) - 0.072*p*(1-p), Poly-doc fee
    required_edge REAL,                    -- dynamic gate threshold at entry
    secs_into_window REAL,                 -- 300 - time_remaining_secs
    schedule_override TEXT                 -- regime schedule in force
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
    market_id TEXT,
    -- Enriched fields (added for strategy analysis)
    orderbook_signal REAL,                 -- L4 signal value
    sentiment_signal REAL,                 -- L5 signal value
    coinbase_direction REAL,               -- Coinbase cross-confirmation
    regime TEXT,                           -- Current regime classification
    regime_confidence REAL,                -- Regime confidence
    entry_reason TEXT,                     -- Why trade was/wasn't taken
    signal_weights TEXT                    -- JSON of weight vector at this tick
);

CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_market_id ON trades(market_id);
CREATE INDEX IF NOT EXISTS idx_signal_log_timestamp ON signal_log(timestamp);
"""

# Columns to add to existing databases (migrations).
# Each tuple is (table, column_name, column_def).
_MIGRATIONS = [
    ("trades", "orderbook_signal", "REAL"),
    ("trades", "sentiment_signal", "REAL"),
    ("trades", "coinbase_direction", "REAL"),
    ("trades", "regime", "TEXT"),
    ("trades", "regime_confidence", "REAL"),
    ("trades", "risk_size_multiplier", "REAL"),
    ("trades", "consecutive_losses", "INTEGER"),
    ("trades", "signal_weights", "TEXT"),
    ("trades", "ob_spread", "REAL"),
    ("trades", "ob_yes_depth", "REAL"),
    ("trades", "ob_ask_depth", "REAL"),
    ("trades", "session_trade_num", "INTEGER"),
    ("trades", "resolved_at", "TEXT"),
    # Full feature snapshot at entry (2026-06-01) — see ADR-0001.
    ("trades", "l1_lag_component", "REAL"),
    ("trades", "l1_open_component", "REAL"),
    ("trades", "l4_imbalance", "REAL"),
    ("trades", "l4_flow", "REAL"),
    ("trades", "l4_mid_dev", "REAL"),
    ("trades", "l4_top_pressure", "REAL"),
    ("trades", "l4_thickness", "REAL"),
    ("trades", "l6_fade", "REAL"),
    ("trades", "l7_taker_ratio", "REAL"),
    ("trades", "l8_clob_flow", "REAL"),
    ("trades", "l9b_absorption", "REAL"),
    ("trades", "l10_exhaustion", "REAL"),
    ("trades", "l11_trade_size", "REAL"),
    ("trades", "l12_wallet_flow", "REAL"),
    ("trades", "prob_edge", "REAL"),
    ("trades", "net_ev_per_share", "REAL"),
    ("trades", "required_edge", "REAL"),
    ("trades", "secs_into_window", "REAL"),
    ("trades", "schedule_override", "TEXT"),
    # Sizing provenance (2026-06-12, ADR-0003): was the bet bumped up to the
    # exchange-minimum floor, and what raw quarter-Kelly USDC preceded it.
    ("trades", "size_bumped", "INTEGER"),
    ("trades", "kelly_raw_usdc", "REAL"),
    ("signal_log", "orderbook_signal", "REAL"),
    ("signal_log", "sentiment_signal", "REAL"),
    ("signal_log", "coinbase_direction", "REAL"),
    ("signal_log", "regime", "TEXT"),
    ("signal_log", "regime_confidence", "REAL"),
    ("signal_log", "entry_reason", "TEXT"),
    ("signal_log", "signal_weights", "TEXT"),
]


class Database:
    """SQLite trade log and signal validation store."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def connect(self):
        """Open connection, create tables if needed, and run migrations."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._run_migrations()
        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def _run_migrations(self) -> None:
        """Add new columns to existing tables (idempotent)."""
        for table, col_name, col_type in _MIGRATIONS:
            try:
                self.conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                )
                logger.info(f"Migration: added {table}.{col_name}")
            except sqlite3.OperationalError:
                pass  # Column already exists

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
