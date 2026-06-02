"""Tests for the enriched `trades` schema migration (2026-06-01).

The full feature snapshot adds ~20 columns to `trades`. Migration must be
idempotent and must upgrade a pre-enrichment DB in place (no backfill —
old rows keep NULL in the new columns). See
docs/adr/0001-full-feature-snapshot-on-trade-record.md.
"""

import sqlite3

from logging_db.database import Database

NEW_COLUMNS = [
    "l1_lag_component", "l1_open_component",
    "l4_imbalance", "l4_flow", "l4_mid_dev", "l4_top_pressure", "l4_thickness",
    "l6_fade", "l7_taker_ratio", "l8_clob_flow", "l9b_absorption",
    "l10_exhaustion", "l11_trade_size", "l12_wallet_flow",
    "prob_edge", "net_ev_per_share", "required_edge",
    "secs_into_window", "schedule_override",
]


def _columns(db_path: str, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    conn.close()
    return cols


def test_new_columns_added_to_fresh_db(tmp_path) -> None:
    db = Database(str(tmp_path / "fresh.db"))
    db.connect()
    db.close()
    cols = _columns(str(tmp_path / "fresh.db"), "trades")
    for c in NEW_COLUMNS:
        assert c in cols, f"missing column {c}"


def test_migration_upgrades_pre_enrichment_db(tmp_path) -> None:
    db_path = str(tmp_path / "old.db")
    # A pre-enrichment trades table: minimal, without the new columns.
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "timestamp TEXT, market_id TEXT, side TEXT, entry_price REAL, "
        "size_usdc REAL)"
    )
    conn.execute(
        "INSERT INTO trades (timestamp, market_id, side, entry_price, size_usdc) "
        "VALUES ('t', 'm', 'YES', 0.5, 1.0)"
    )
    conn.commit()
    conn.close()

    Database(db_path).connect()  # runs migrations in place
    cols = _columns(db_path, "trades")
    for c in NEW_COLUMNS:
        assert c in cols, f"migration did not add {c}"
    # Old row survives with NULL in a new column (no backfill).
    conn = sqlite3.connect(db_path)
    val = conn.execute("SELECT net_ev_per_share FROM trades").fetchone()[0]
    conn.close()
    assert val is None


def test_migration_is_idempotent(tmp_path) -> None:
    db_path = str(tmp_path / "idem.db")
    Database(db_path).connect()
    # Second connect must not raise (columns already exist).
    Database(db_path).connect()
    cols = _columns(db_path, "trades")
    for c in NEW_COLUMNS:
        assert c in cols
