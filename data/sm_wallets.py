"""SM Wallet Registry — L9 signal layer for Bot K.

Loads smart-money wallet addresses from CSV and provides O(1)
set-based lookup. Wallets satisfy: >70% directional WR, 500+ markets,
ROI >2%, volume >$10k on Polymarket BTC 5-min binary options.

Source: Dune Analytics Q18b (query 7452414), run 2026-05-08.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = Path(__file__).parent / "sm_wallet_addresses.csv"


@dataclass(frozen=True)
class SMWalletStats:
    """Per-wallet statistics from Dune analysis."""

    address: str
    markets_active: int
    direction_wr_pct: float
    total_trades: int
    total_volume: float
    net_pnl: float
    roi_pct: float
    avg_entry_price: float
    trade_wr_pct: float


class SMWalletRegistry:
    """Registry of smart-money wallet addresses for on-chain monitoring.

    Args:
        csv_path: Path to CSV with wallet addresses and stats.
            Defaults to data/sm_wallet_addresses.csv.
        min_roi_pct: Minimum ROI % to include wallet. Default 2.0.
        min_volume: Minimum total volume to include wallet. Default 10000.
        min_trades: Minimum total trades to include wallet. Default 1500.
    """

    def __init__(
        self,
        csv_path: Path | None = None,
        min_roi_pct: float = 2.0,
        min_volume: float = 10_000.0,
        min_trades: int = 1_500,
    ) -> None:
        self._addresses: set[str] = set()
        self._stats: dict[str, SMWalletStats] = {}
        self._min_roi_pct = min_roi_pct
        self._min_volume = min_volume
        self._min_trades = min_trades
        self._csv_path: Path = csv_path or DEFAULT_CSV_PATH

        if self._csv_path.exists():
            self._load_csv(self._csv_path)
        else:
            logger.warning("SM wallet CSV not found at %s — registry empty", self._csv_path)

    def _load_csv(self, path: Path) -> None:
        """Load wallet addresses from CSV, applying filters."""
        loaded = 0
        filtered = 0
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                roi = float(row["roi_pct"])
                volume = float(row["total_volume"])
                trades = int(row["total_trades"])
                if (
                    roi < self._min_roi_pct
                    or volume < self._min_volume
                    or trades < self._min_trades
                ):
                    filtered += 1
                    continue

                addr = row["wallet"].strip().lower()
                stats = SMWalletStats(
                    address=addr,
                    markets_active=int(row["markets_active"]),
                    direction_wr_pct=float(row["direction_wr_pct"]),
                    total_trades=int(row["total_trades"]),
                    total_volume=volume,
                    net_pnl=float(row["net_pnl"]),
                    roi_pct=roi,
                    avg_entry_price=float(row["avg_entry_price"]),
                    trade_wr_pct=float(row["trade_wr_pct"]),
                )
                self._addresses.add(addr)
                self._stats[addr] = stats
                loaded += 1

        logger.info(
            "SM wallet registry: %d wallets loaded, %d filtered out "
            "(min_roi=%.1f%%, min_vol=$%.0f, min_trades=%d)",
            loaded, filtered, self._min_roi_pct, self._min_volume,
            self._min_trades,
        )

    def is_sm_wallet(self, address: str) -> bool:
        """Check if an address is a known SM wallet. O(1) lookup."""
        return address.strip().lower() in self._addresses

    def get_stats(self, address: str) -> SMWalletStats | None:
        """Get stats for a specific SM wallet, or None if not found."""
        return self._stats.get(address.strip().lower())

    def get_all_addresses(self) -> frozenset[str]:
        """Return all SM wallet addresses as a frozen set."""
        return frozenset(self._addresses)

    @property
    def wallet_count(self) -> int:
        """Number of SM wallets in the registry."""
        return len(self._addresses)

    def refresh(self, csv_path: Path | None = None) -> int:
        """Reload the registry from CSV, replacing all current data.

        Args:
            csv_path: Path to reload from. Defaults to the path used at init.

        Returns:
            Number of wallets loaded after filtering.
        """
        path = csv_path or self._csv_path
        old_count = self.wallet_count
        self._addresses.clear()
        self._stats.clear()

        if path.exists():
            self._load_csv(path)
        else:
            logger.warning("SM wallet CSV not found at %s — registry now empty", path)

        logger.info(
            "SM wallet registry refreshed: %d → %d wallets",
            old_count, self.wallet_count,
        )
        return self.wallet_count

    def load_from_list(self, addresses: list[str]) -> None:
        """Load addresses from a plain list (no stats). For testing."""
        for addr in addresses:
            normalised = addr.strip().lower()
            self._addresses.add(normalised)
        logger.info("SM wallet registry: added %d addresses from list", len(addresses))
