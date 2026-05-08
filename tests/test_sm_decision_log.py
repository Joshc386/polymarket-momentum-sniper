"""Tests for SM Decision Logger (threshold monitoring for L9)."""

from pathlib import Path

import pytest

from strategy.sm_confirmation import (
    PositionSide,
    SMConfirmationConfig,
    SMDecision,
    SMFlowState,
)
from strategy.sm_decision_log import SMDecisionLogger


@pytest.fixture
def db_logger(tmp_path: Path) -> SMDecisionLogger:
    logger = SMDecisionLogger(db_path=tmp_path / "test_decisions.db")
    yield logger
    logger.close()


@pytest.fixture
def default_config() -> SMConfirmationConfig:
    return SMConfirmationConfig()


class TestSMDecisionLogger:
    def test_creates_db_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        log = SMDecisionLogger(db_path=db_path)
        assert db_path.exists()
        log.close()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "dir" / "test.db"
        log = SMDecisionLogger(db_path=db_path)
        assert db_path.exists()
        log.close()

    def test_log_decision_stores_record(
        self, db_logger: SMDecisionLogger, default_config: SMConfirmationConfig,
    ) -> None:
        flow = SMFlowState(yes_volume=700.0, no_volume=300.0, num_wallets=5)
        db_logger.log_decision(
            position_side=PositionSide.YES,
            market_price=0.72,
            flow=flow,
            decision=SMDecision.HOLD,
            config=default_config,
            market_id="test_market_123",
            window_start="2026-05-08T12:00:00Z",
            check_minute=3,
        )
        counts = db_logger.get_decision_counts()
        assert counts["HOLD"] == 1

    def test_log_multiple_decisions(
        self, db_logger: SMDecisionLogger, default_config: SMConfirmationConfig,
    ) -> None:
        for decision in [SMDecision.HOLD, SMDecision.HOLD, SMDecision.EXIT, SMDecision.IGNORE]:
            flow = SMFlowState(yes_volume=500.0, no_volume=500.0, num_wallets=3)
            db_logger.log_decision(
                position_side=PositionSide.YES,
                market_price=0.72,
                flow=flow,
                decision=decision,
                config=default_config,
            )
        counts = db_logger.get_decision_counts()
        assert counts["HOLD"] == 2
        assert counts["EXIT"] == 1
        assert counts["IGNORE"] == 1

    def test_position_fraction_recorded_correctly(
        self, db_logger: SMDecisionLogger, default_config: SMConfirmationConfig,
    ) -> None:
        flow = SMFlowState(yes_volume=300.0, no_volume=700.0, num_wallets=5)
        db_logger.log_decision(
            position_side=PositionSide.NO,
            market_price=0.72,
            flow=flow,
            decision=SMDecision.HOLD,
            config=default_config,
        )
        cursor = db_logger._conn.execute(
            "SELECT position_fraction FROM sm_decisions"
        )
        fraction = cursor.fetchone()[0]
        assert fraction == pytest.approx(0.7)

    def test_optional_fields_nullable(
        self, db_logger: SMDecisionLogger, default_config: SMConfirmationConfig,
    ) -> None:
        flow = SMFlowState(yes_volume=500.0, no_volume=500.0, num_wallets=3)
        db_logger.log_decision(
            position_side=PositionSide.YES,
            market_price=0.72,
            flow=flow,
            decision=SMDecision.IGNORE,
            config=default_config,
        )
        cursor = db_logger._conn.execute(
            "SELECT market_id, window_start, check_minute FROM sm_decisions"
        )
        row = cursor.fetchone()
        assert row[0] is None
        assert row[1] is None
        assert row[2] is None

    def test_get_decision_counts_empty(self, db_logger: SMDecisionLogger) -> None:
        counts = db_logger.get_decision_counts()
        assert counts == {}

    def test_get_threshold_analysis(
        self, db_logger: SMDecisionLogger, default_config: SMConfirmationConfig,
    ) -> None:
        test_cases = [
            (SMFlowState(yes_volume=700.0, no_volume=300.0, num_wallets=5), SMDecision.HOLD),
            (SMFlowState(yes_volume=800.0, no_volume=200.0, num_wallets=5), SMDecision.HOLD),
            (SMFlowState(yes_volume=300.0, no_volume=700.0, num_wallets=5), SMDecision.EXIT),
        ]
        for flow, decision in test_cases:
            db_logger.log_decision(
                position_side=PositionSide.YES,
                market_price=0.72,
                flow=flow,
                decision=decision,
                config=default_config,
            )
        analysis = db_logger.get_threshold_analysis()
        assert len(analysis) > 0
        assert all("position_fraction_pct" in r for r in analysis)
        assert all("decision" in r for r in analysis)
        assert all("count" in r for r in analysis)
