"""Shared utilities for all R&D agents.

Provides consistent data loading, config access, and helper functions
so agents don't duplicate boilerplate.
"""

import csv
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "backtest" / "data"
RUNTIME_DIR = PROJECT_ROOT / "data_runtime"
DEFAULT_DB_PATH = RUNTIME_DIR / "trades.db"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class DateRange:
    """Consistent time window for data operations."""
    start: datetime
    end: datetime

    @property
    def start_ts(self) -> int:
        """Unix timestamp (seconds)."""
        return int(self.start.timestamp())

    @property
    def end_ts(self) -> int:
        """Unix timestamp (seconds)."""
        return int(self.end.timestamp())

    @property
    def start_ms(self) -> int:
        """Unix timestamp (milliseconds)."""
        return int(self.start.timestamp() * 1000)

    @property
    def end_ms(self) -> int:
        """Unix timestamp (milliseconds)."""
        return int(self.end.timestamp() * 1000)

    @property
    def days(self) -> float:
        return (self.end - self.start).total_seconds() / 86400

    def __str__(self) -> str:
        return f"{self.start.strftime('%Y-%m-%d')} to {self.end.strftime('%Y-%m-%d')} ({self.days:.1f} days)"


def ensure_data_dir(subdir: str = "") -> Path:
    """Create and return a data directory path.

    Args:
        subdir: Optional subdirectory under backtest/data/.

    Returns:
        Path to the directory (created if needed).
    """
    path = DATA_DIR / subdir if subdir else DATA_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config() -> dict[str, Any]:
    """Load config.yaml as a dict.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config.yaml doesn't exist.
    """
    import yaml

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_backtest_data(path: str | Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame with standard parsing.

    Args:
        path: Path to CSV file.

    Returns:
        DataFrame with parsed data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"Loaded {len(df):,} rows from {path.name}")
    return df


def load_trade_db(db_path: str | Path | None = None) -> pd.DataFrame:
    """Read all trades from the SQLite database.

    Args:
        db_path: Path to trades.db. Uses default if None.

    Returns:
        DataFrame with all trade records.

    Raises:
        FileNotFoundError: If the database doesn't exist.
    """
    db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"Trade database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp", conn)
        logger.info(f"Loaded {len(df):,} trades from {db_path.name}")
        return df
    finally:
        conn.close()


def load_signal_log(db_path: str | Path | None = None) -> pd.DataFrame:
    """Read all signal log entries from the SQLite database.

    Args:
        db_path: Path to trades.db. Uses default if None.

    Returns:
        DataFrame with all signal log records.

    Raises:
        FileNotFoundError: If the database doesn't exist.
    """
    db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"Trade database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query("SELECT * FROM signal_log ORDER BY timestamp", conn)
        logger.info(f"Loaded {len(df):,} signal log entries from {db_path.name}")
        return df
    finally:
        conn.close()


def get_market_date_range() -> DateRange | None:
    """Read the date range covered by polybacktest_markets.csv.

    Returns:
        DateRange or None if markets file doesn't exist.
    """
    markets_path = DATA_DIR / "polybacktest_markets.csv"
    if not markets_path.exists():
        return None

    earliest = None
    latest = None
    with open(markets_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["start_time", "end_time"]:
                t = row.get(key, "")
                if t:
                    dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
                    if earliest is None or dt < earliest:
                        earliest = dt
                    if latest is None or dt > latest:
                        latest = dt

    if earliest and latest:
        return DateRange(start=earliest, end=latest)
    return None


def file_age_hours(path: str | Path) -> float | None:
    """Return how many hours ago a file was last modified.

    Args:
        path: File path to check.

    Returns:
        Hours since last modification, or None if file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    age_secs = datetime.now(timezone.utc).timestamp() - mtime
    return age_secs / 3600


def format_table(headers: list[str], rows: list[list[str]], align: str = "left") -> str:
    """Format data as an aligned text table.

    Args:
        headers: Column headers.
        rows: List of row data (each row is a list of strings).
        align: 'left' or 'right' alignment.

    Returns:
        Formatted table string.
    """
    if not rows:
        return "  (no data)"

    all_rows = [headers] + rows
    col_widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(headers))]

    lines = []
    fmt = "  ".join(f"{{:<{w}}}" if align == "left" else f"{{:>{w}}}" for w in col_widths)
    lines.append(fmt.format(*headers))
    lines.append("  ".join("-" * w for w in col_widths))
    for row in rows:
        lines.append(fmt.format(*[str(v) for v in row]))

    return "\n".join(lines)
