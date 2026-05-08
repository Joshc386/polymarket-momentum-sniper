"""Tests for SM Wallet Refresh — Dune API integration (Phase 1 of L9)."""

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data.sm_wallet_refresh import (
    DUNE_QUERY_ID,
    DuneRefreshError,
    execute_query,
    poll_execution,
    refresh_wallets,
    write_csv,
)
from data.sm_wallets import SMWalletRegistry


SAMPLE_DUNE_ROWS = [
    {
        "wallet": "0xAAA1111111111111111111111111111111111111",
        "markets_active": 800,
        "direction_wr_pct": 78.5,
        "total_trades": 5000,
        "total_volume": 50000.0,
        "net_pnl": 2500.0,
        "roi_pct": 5.0,
        "avg_entry_price": 0.55,
        "trade_wr_pct": 58.0,
    },
    {
        "wallet": "0xBBB2222222222222222222222222222222222222",
        "markets_active": 600,
        "direction_wr_pct": 72.0,
        "total_trades": 3000,
        "total_volume": 30000.0,
        "net_pnl": 900.0,
        "roi_pct": 3.0,
        "avg_entry_price": 0.52,
        "trade_wr_pct": 55.0,
    },
]


class TestWriteCSV:
    def test_writes_rows_to_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "wallets.csv"
        count = write_csv(SAMPLE_DUNE_ROWS, csv_path=csv_path)
        assert count == 2
        assert csv_path.exists()

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["wallet"] == "0xAAA1111111111111111111111111111111111111"

    def test_empty_rows_returns_zero(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "wallets.csv"
        count = write_csv([], csv_path=csv_path)
        assert count == 0
        assert not csv_path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "nested" / "dir" / "wallets.csv"
        count = write_csv(SAMPLE_DUNE_ROWS, csv_path=csv_path)
        assert count == 2
        assert csv_path.exists()

    def test_ignores_extra_columns(self, tmp_path: Path) -> None:
        rows = [{**SAMPLE_DUNE_ROWS[0], "extra_col": "ignored"}]
        csv_path = tmp_path / "wallets.csv"
        write_csv(rows, csv_path=csv_path)

        with open(csv_path, newline="", encoding="utf-8") as f:
            header = f.readline().strip()
        assert "extra_col" not in header


class TestExecuteQuery:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DUNE_API_KEY", raising=False)
        with pytest.raises(DuneRefreshError, match="DUNE_API_KEY"):
            execute_query()

    @patch("data.sm_wallet_refresh.httpx.post")
    def test_returns_execution_id(
        self, mock_post: MagicMock, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DUNE_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"execution_id": "exec_123"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = execute_query(query_id=DUNE_QUERY_ID)
        assert result == "exec_123"
        mock_post.assert_called_once()

    @patch("data.sm_wallet_refresh.httpx.post")
    def test_no_execution_id_raises(
        self, mock_post: MagicMock, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DUNE_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        with pytest.raises(DuneRefreshError, match="No execution_id"):
            execute_query()


class TestPollExecution:
    @patch("data.sm_wallet_refresh.httpx.get")
    def test_returns_rows_on_completion(
        self, mock_get: MagicMock, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DUNE_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "state": "QUERY_STATE_COMPLETED",
            "result": {"rows": SAMPLE_DUNE_ROWS},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        rows = poll_execution("exec_123", poll_interval=0)
        assert len(rows) == 2

    @patch("data.sm_wallet_refresh.httpx.get")
    def test_raises_on_failure_state(
        self, mock_get: MagicMock, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DUNE_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"state": "QUERY_STATE_FAILED"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(DuneRefreshError, match="FAILED"):
            poll_execution("exec_123", poll_interval=0)

    @patch("data.sm_wallet_refresh.httpx.get")
    def test_times_out_after_max_attempts(
        self, mock_get: MagicMock, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DUNE_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"state": "QUERY_STATE_PENDING"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(DuneRefreshError, match="did not complete"):
            poll_execution("exec_123", poll_interval=0, max_attempts=3)


class TestRefreshWallets:
    @patch("data.sm_wallet_refresh.poll_execution")
    @patch("data.sm_wallet_refresh.execute_query")
    def test_full_refresh_writes_csv_and_reloads(
        self,
        mock_execute: MagicMock,
        mock_poll: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_execute.return_value = "exec_456"
        mock_poll.return_value = SAMPLE_DUNE_ROWS

        csv_path = tmp_path / "wallets.csv"
        registry = SMWalletRegistry(csv_path=csv_path, min_roi_pct=0.0, min_volume=0.0, min_trades=0)
        assert registry.wallet_count == 0

        count = refresh_wallets(registry=registry, csv_path=csv_path)
        assert count == 2
        assert registry.wallet_count == 2
        assert csv_path.exists()

    @patch("data.sm_wallet_refresh.poll_execution")
    @patch("data.sm_wallet_refresh.execute_query")
    def test_empty_result_raises_and_preserves_csv(
        self,
        mock_execute: MagicMock,
        mock_poll: MagicMock,
        tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "wallets.csv"
        write_csv(SAMPLE_DUNE_ROWS, csv_path=csv_path)

        mock_execute.return_value = "exec_789"
        mock_poll.return_value = []

        with pytest.raises(DuneRefreshError, match="0 rows"):
            refresh_wallets(csv_path=csv_path)

        with open(csv_path, newline="", encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == 2

    @patch("data.sm_wallet_refresh.poll_execution")
    @patch("data.sm_wallet_refresh.execute_query")
    def test_refresh_without_registry(
        self,
        mock_execute: MagicMock,
        mock_poll: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_execute.return_value = "exec_abc"
        mock_poll.return_value = SAMPLE_DUNE_ROWS

        csv_path = tmp_path / "wallets.csv"
        count = refresh_wallets(registry=None, csv_path=csv_path)
        assert count == 2
        assert csv_path.exists()
