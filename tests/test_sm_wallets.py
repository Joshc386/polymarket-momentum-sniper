"""Tests for SM Wallet Registry (Phase 1 of L9 signal layer)."""

import csv
import tempfile
from pathlib import Path

import pytest

from data.sm_wallets import SMWalletRegistry, SMWalletStats


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a sample SM wallet CSV for testing."""
    csv_path = tmp_path / "test_wallets.csv"
    rows = [
        {
            "wallet": "0xAAA1111111111111111111111111111111111111",
            "markets_active": "800",
            "direction_wr_pct": "78.5",
            "total_trades": "5000",
            "total_volume": "50000.00",
            "net_pnl": "2500.00",
            "roi_pct": "5.0",
            "avg_entry_price": "0.55",
            "trade_wr_pct": "58.0",
        },
        {
            "wallet": "0xBBB2222222222222222222222222222222222222",
            "markets_active": "600",
            "direction_wr_pct": "72.0",
            "total_trades": "3000",
            "total_volume": "30000.00",
            "net_pnl": "900.00",
            "roi_pct": "3.0",
            "avg_entry_price": "0.52",
            "trade_wr_pct": "55.0",
        },
        {
            "wallet": "0xCCC3333333333333333333333333333333333333",
            "markets_active": "500",
            "direction_wr_pct": "71.0",
            "total_trades": "1000",
            "total_volume": "5000.00",
            "net_pnl": "200.00",
            "roi_pct": "4.0",
            "avg_entry_price": "0.50",
            "trade_wr_pct": "52.0",
        },
        {
            "wallet": "0xDDD4444444444444444444444444444444444444",
            "markets_active": "550",
            "direction_wr_pct": "70.5",
            "total_trades": "2000",
            "total_volume": "15000.00",
            "net_pnl": "150.00",
            "roi_pct": "1.0",
            "avg_entry_price": "0.48",
            "trade_wr_pct": "50.0",
        },
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


class TestSMWalletRegistry:
    def test_loads_wallets_from_csv(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(csv_path=sample_csv)
        assert registry.wallet_count == 2

    def test_filters_low_roi(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(csv_path=sample_csv)
        low_roi_addr = "0xddd4444444444444444444444444444444444444"
        assert not registry.is_sm_wallet(low_roi_addr)

    def test_filters_low_volume(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(csv_path=sample_csv)
        low_vol_addr = "0xccc3333333333333333333333333333333333333"
        assert not registry.is_sm_wallet(low_vol_addr)

    def test_is_sm_wallet_positive(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(csv_path=sample_csv)
        assert registry.is_sm_wallet("0xAAA1111111111111111111111111111111111111")
        assert registry.is_sm_wallet("0xBBB2222222222222222222222222222222222222")

    def test_is_sm_wallet_negative(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(csv_path=sample_csv)
        assert not registry.is_sm_wallet("0x0000000000000000000000000000000000000000")

    def test_case_insensitive_lookup(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(csv_path=sample_csv)
        assert registry.is_sm_wallet("0xaaa1111111111111111111111111111111111111")
        assert registry.is_sm_wallet("0xAAA1111111111111111111111111111111111111")
        assert registry.is_sm_wallet("  0xAAA1111111111111111111111111111111111111  ")

    def test_get_stats_returns_data(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(csv_path=sample_csv)
        stats = registry.get_stats("0xAAA1111111111111111111111111111111111111")
        assert stats is not None
        assert stats.markets_active == 800
        assert stats.direction_wr_pct == 78.5
        assert stats.total_trades == 5000
        assert stats.total_volume == 50000.0
        assert stats.net_pnl == 2500.0
        assert stats.roi_pct == 5.0
        assert stats.avg_entry_price == 0.55
        assert stats.trade_wr_pct == 58.0

    def test_get_stats_returns_none_for_unknown(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(csv_path=sample_csv)
        assert registry.get_stats("0x0000000000000000000000000000000000000000") is None

    def test_get_all_addresses_returns_frozenset(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(csv_path=sample_csv)
        addrs = registry.get_all_addresses()
        assert isinstance(addrs, frozenset)
        assert len(addrs) == 2

    def test_custom_filter_thresholds(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(
            csv_path=sample_csv, min_roi_pct=4.0, min_volume=20_000.0,
        )
        assert registry.wallet_count == 1
        assert registry.is_sm_wallet("0xAAA1111111111111111111111111111111111111")

    def test_zero_filters_loads_all(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(
            csv_path=sample_csv, min_roi_pct=0.0, min_volume=0.0, min_trades=0,
        )
        assert registry.wallet_count == 4

    def test_filters_low_trades(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(
            csv_path=sample_csv, min_roi_pct=0.0, min_volume=0.0, min_trades=2000,
        )
        assert registry.wallet_count == 3
        assert not registry.is_sm_wallet("0xCCC3333333333333333333333333333333333333")

    def test_min_trades_default_filters_correctly(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(
            csv_path=sample_csv, min_roi_pct=0.0, min_volume=0.0,
        )
        assert not registry.is_sm_wallet("0xCCC3333333333333333333333333333333333333")

    def test_missing_csv_creates_empty_registry(self, tmp_path: Path) -> None:
        registry = SMWalletRegistry(csv_path=tmp_path / "nonexistent.csv")
        assert registry.wallet_count == 0

    def test_load_from_list(self) -> None:
        registry = SMWalletRegistry(csv_path=Path("/nonexistent"))
        registry.load_from_list([
            "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
        ])
        assert registry.wallet_count == 2
        assert registry.is_sm_wallet("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    def test_wallet_stats_is_frozen(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(csv_path=sample_csv)
        stats = registry.get_stats("0xAAA1111111111111111111111111111111111111")
        assert stats is not None
        with pytest.raises(AttributeError):
            stats.roi_pct = 99.0  # type: ignore[misc]


class TestRefresh:
    def test_refresh_reloads_data(self, sample_csv: Path, tmp_path: Path) -> None:
        registry = SMWalletRegistry(
            csv_path=sample_csv, min_roi_pct=0.0, min_volume=0.0, min_trades=0,
        )
        assert registry.wallet_count == 4

        new_csv = tmp_path / "updated.csv"
        rows = [
            {
                "wallet": "0xEEE5555555555555555555555555555555555555",
                "markets_active": "900",
                "direction_wr_pct": "80.0",
                "total_trades": "6000",
                "total_volume": "60000.00",
                "net_pnl": "3000.00",
                "roi_pct": "5.0",
                "avg_entry_price": "0.60",
                "trade_wr_pct": "60.0",
            },
        ]
        with open(new_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        count = registry.refresh(csv_path=new_csv)
        assert count == 1
        assert registry.wallet_count == 1
        assert registry.is_sm_wallet("0xEEE5555555555555555555555555555555555555")
        assert not registry.is_sm_wallet("0xAAA1111111111111111111111111111111111111")

    def test_refresh_clears_old_data(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(csv_path=sample_csv)
        assert registry.wallet_count == 2
        registry.refresh(csv_path=Path("/nonexistent.csv"))
        assert registry.wallet_count == 0

    def test_refresh_uses_default_path(self, sample_csv: Path) -> None:
        registry = SMWalletRegistry(csv_path=sample_csv)
        count = registry.refresh()
        assert count == registry.wallet_count


class TestRealCSV:
    """Tests against the actual sm_wallet_addresses.csv."""

    def test_real_csv_loads(self) -> None:
        registry = SMWalletRegistry()
        assert registry.wallet_count == 100

    def test_real_csv_all_positive_roi(self) -> None:
        registry = SMWalletRegistry()
        for addr in registry.get_all_addresses():
            stats = registry.get_stats(addr)
            assert stats is not None
            assert stats.roi_pct > 2.0
            assert stats.total_volume > 10_000.0

    def test_real_csv_known_top_wallet(self) -> None:
        registry = SMWalletRegistry()
        top = registry.get_stats("0x5285fa6269f2df68cceff60ef20f4064f02e99a1")
        assert top is not None
        assert top.roi_pct == 19.52
        assert top.net_pnl == 7363.17
