"""Tests for SignalDiagnosticLogger.

Covers: disabled mode no-op, enabled mode writes rows, schema integrity,
sample_every_n throttle, graceful exception handling, close idempotency.
"""

import os
import sqlite3
import tempfile

import pytest

from strategy.signal_diagnostic_log import SignalDiagnosticLogger, SignalTick


@pytest.fixture
def tmp_db_path():
    """Temp DB path that's cleaned up after each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


class TestDisabledLogger:
    def test_no_op_when_disabled(self, tmp_db_path):
        log = SignalDiagnosticLogger(tmp_db_path, "test_bot", enabled=False)
        log.log(SignalTick(l1_oracle_lag=0.5, est_prob_up=0.6))
        log.close()
        # No DB file should be created when disabled
        assert not os.path.exists(tmp_db_path) or os.path.getsize(tmp_db_path) == 0


class TestEnabledLogger:
    def test_writes_one_row(self, tmp_db_path):
        log = SignalDiagnosticLogger(tmp_db_path, "test_bot", enabled=True)
        tick = SignalTick(
            l1_oracle_lag=0.123, l4_orderbook=0.456,
            est_prob_up=0.62, would_pick_side="YES",
        )
        log.log(tick)
        log.close()

        conn = sqlite3.connect(tmp_db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), l1_oracle_lag, l4_orderbook, "
                    "est_prob_up, would_pick_side FROM signal_ticks")
        row = cur.fetchone()
        assert row[0] == 1
        assert abs(row[1] - 0.123) < 1e-9
        assert abs(row[2] - 0.456) < 1e-9
        assert abs(row[3] - 0.62) < 1e-9
        assert row[4] == "YES"
        conn.close()

    def test_writes_many_rows(self, tmp_db_path):
        log = SignalDiagnosticLogger(tmp_db_path, "bot_x", enabled=True)
        for i in range(100):
            log.log(SignalTick(l1_oracle_lag=i * 0.01))
        log.close()

        conn = sqlite3.connect(tmp_db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM signal_ticks")
        assert cur.fetchone()[0] == 100
        conn.close()

    def test_auto_fills_bot_name(self, tmp_db_path):
        log = SignalDiagnosticLogger(tmp_db_path, "auto_bot", enabled=True)
        log.log(SignalTick())  # no bot_name in tick
        log.close()

        conn = sqlite3.connect(tmp_db_path)
        cur = conn.cursor()
        cur.execute("SELECT bot_name FROM signal_ticks LIMIT 1")
        assert cur.fetchone()[0] == "auto_bot"
        conn.close()

    def test_auto_fills_timestamp(self, tmp_db_path):
        log = SignalDiagnosticLogger(tmp_db_path, "bot", enabled=True)
        log.log(SignalTick())  # no timestamp/unix_time
        log.close()

        conn = sqlite3.connect(tmp_db_path)
        cur = conn.cursor()
        cur.execute("SELECT timestamp, unix_time FROM signal_ticks LIMIT 1")
        ts, ux = cur.fetchone()
        assert ts != ""
        assert ux > 0
        conn.close()


class TestSampling:
    def test_sample_every_n(self, tmp_db_path):
        log = SignalDiagnosticLogger(
            tmp_db_path, "bot", enabled=True, sample_every_n=5
        )
        for _ in range(20):
            log.log(SignalTick())
        log.close()

        conn = sqlite3.connect(tmp_db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM signal_ticks")
        # 20 calls, sample every 5 -> 4 rows
        assert cur.fetchone()[0] == 4
        conn.close()

    def test_sample_every_n_minimum_one(self, tmp_db_path):
        # sample_every_n=0 should be coerced to 1
        log = SignalDiagnosticLogger(
            tmp_db_path, "bot", enabled=True, sample_every_n=0
        )
        log.log(SignalTick())
        log.close()
        assert log.sample_every_n == 1


class TestSchemaIntegrity:
    def test_full_schema_columns_exist(self, tmp_db_path):
        log = SignalDiagnosticLogger(tmp_db_path, "bot", enabled=True)
        log.close()

        conn = sqlite3.connect(tmp_db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(signal_ticks)")
        cols = {r[1] for r in cur.fetchall()}
        # Check critical columns for the bug investigation
        expected = {
            "l1_oracle_lag", "l1_lag_component", "l1_open_component",
            "l4_orderbook", "l4_imbalance", "l4_flow", "l4_mid_dev",
            "l4_top_pressure", "l4_thickness",
            "combined_signal", "est_prob_up", "market_implied_prob",
            "would_pick_side", "would_enter", "trade_placed",
            "regime", "schedule_override",
        }
        missing = expected - cols
        assert not missing, f"Missing columns: {missing}"
        conn.close()

    def test_all_l_signals_loggable(self, tmp_db_path):
        """All L1-L12 signal columns should accept floats."""
        log = SignalDiagnosticLogger(tmp_db_path, "bot", enabled=True)
        log.log(SignalTick(
            l1_oracle_lag=0.1, l2_momentum=0.2, l3_liquidation=0.3,
            l4_orderbook=0.4, l5_sentiment=0.5,
            l6_fade=0.6, l7_taker_ratio=0.7, l8_clob_flow=0.8,
            l9b_absorption=0.9, l10_exhaustion=0.10, l11_trade_size=0.11,
            l12_wallet_flow=0.12,
        ))
        log.close()

        conn = sqlite3.connect(tmp_db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT l1_oracle_lag, l2_momentum, l3_liquidation, l4_orderbook, "
            "l5_sentiment, l6_fade, l7_taker_ratio, l8_clob_flow, "
            "l9b_absorption, l10_exhaustion, l11_trade_size, l12_wallet_flow "
            "FROM signal_ticks LIMIT 1"
        )
        row = cur.fetchone()
        assert row == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11, 0.12)
        conn.close()


class TestRobustness:
    def test_close_is_idempotent(self, tmp_db_path):
        log = SignalDiagnosticLogger(tmp_db_path, "bot", enabled=True)
        log.close()
        log.close()  # second close should not raise

    def test_log_after_close_is_silent(self, tmp_db_path):
        log = SignalDiagnosticLogger(tmp_db_path, "bot", enabled=True)
        log.close()
        # Should not raise even though connection is closed
        log.log(SignalTick())

    def test_disabled_logger_close_safe(self, tmp_db_path):
        log = SignalDiagnosticLogger(tmp_db_path, "bot", enabled=False)
        log.close()
