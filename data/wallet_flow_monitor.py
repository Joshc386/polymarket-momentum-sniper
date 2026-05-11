"""Wallet Flow Monitor — on-chain trade tracking for ALL wallets (L12).

Polls Polygon RPC via eth_getLogs for ERC-1155 TransferSingle events on
the Polymarket CTF contract. Unlike SMTradeMonitor (which filters for
known SM wallets only), this tracks every wallet that trades the current
market's tokens within a window.

Produces WalletFlowState consumed by signals.wallet_flow.WalletFlowSignal.

Reuses the same RPC pattern, constants, and log parsing as
data.sm_trade_monitor — separate class to avoid coupling L9 and L12.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

from data.sm_trade_monitor import (
    CTF_CONTRACT,
    DEFAULT_RPCS,
    TOKEN_DECIMALS,
    TRANSFER_SINGLE_TOPIC,
    _parse_log,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalletTrade:
    """A single on-chain trade by a specific wallet.

    Attributes:
        wallet: Lowercase hex address of the trader.
        direction: "BULL" (buying YES or selling NO) or "BEAR" (selling
            YES or buying NO).
        token: "YES" or "NO" — which token was transferred.
        value: Number of shares (USDC-scaled, already divided by 1e6).
    """

    wallet: str
    direction: str
    token: str
    value: float


@dataclass
class WalletFlowState:
    """Accumulated per-wallet flow for the current window.

    Attributes:
        wallet_volumes: {wallet_addr: {"bull": float, "bear": float}}.
            Total bullish and bearish volume per wallet.
        total_bull_volume: Sum of all bullish volume across wallets.
        total_bear_volume: Sum of all bearish volume across wallets.
        bull_wallets: Set of wallets with any bullish activity.
        bear_wallets: Set of wallets with any bearish activity.
        trades: List of individual WalletTrade records.
    """

    wallet_volumes: dict[str, dict[str, float]] = field(default_factory=dict)
    total_bull_volume: float = 0.0
    total_bear_volume: float = 0.0
    bull_wallets: set[str] = field(default_factory=set)
    bear_wallets: set[str] = field(default_factory=set)
    trades: list[WalletTrade] = field(default_factory=list)

    @property
    def num_wallets(self) -> int:
        """Total distinct wallets that traded."""
        return len(self.wallet_volumes)

    @property
    def total_volume(self) -> float:
        """Total volume across all wallets and directions."""
        return self.total_bull_volume + self.total_bear_volume


@dataclass
class WalletFlowMonitor:
    """Monitors on-chain trades for ALL wallets on the current market.

    Same RPC pattern as SMTradeMonitor but tracks every address, not
    just known SM wallets. Designed for L12 wallet flow signal.

    Args:
        rpcs: List of Polygon RPC URLs. Defaults to free public RPCs.
    """

    _rpcs: list[str] = field(default_factory=list)

    # Market state
    _yes_token_id: int = 0
    _no_token_id: int = 0
    _window_id: str = ""

    # Accumulated flow per wallet: {addr: {"bull": float, "bear": float}}
    _wallet_volumes: dict[str, dict[str, float]] = field(default_factory=dict)
    _trades: list[WalletTrade] = field(default_factory=list)
    _bull_wallets: set[str] = field(default_factory=set)
    _bear_wallets: set[str] = field(default_factory=set)
    _total_bull: float = 0.0
    _total_bear: float = 0.0

    # Block tracking
    _last_block: int = 0

    # Infra
    _http_client: httpx.AsyncClient | None = field(default=None, repr=False)
    _running: bool = False
    _last_fetch_time: float = 0.0

    def __post_init__(self) -> None:
        if not self._rpcs:
            rpc_env = os.environ.get("POLYGON_RPC_URL", "").strip()
            self._rpcs = (
                [rpc_env] + DEFAULT_RPCS if rpc_env else list(DEFAULT_RPCS)
            )

    def set_market(
        self, yes_token_id: str, no_token_id: str, window_id: str,
    ) -> None:
        """Set the current market's token IDs. Resets flow on new window.

        Args:
            yes_token_id: YES token ID (decimal string from MarketInfo).
            no_token_id: NO token ID (decimal string from MarketInfo).
            window_id: Window/market slug — triggers reset if changed.
        """
        if window_id != self._window_id:
            self.reset()
            self._window_id = window_id
            logger.debug(
                "Wallet flow monitor: new window %s, tokens YES=%s NO=%s",
                window_id, yes_token_id, no_token_id,
            )

        self._yes_token_id = int(yes_token_id)
        self._no_token_id = int(no_token_id)

    def reset(self) -> None:
        """Zero all accumulated flow state."""
        self._wallet_volumes.clear()
        self._trades.clear()
        self._bull_wallets.clear()
        self._bear_wallets.clear()
        self._total_bull = 0.0
        self._total_bear = 0.0
        self._last_block = 0

    def get_flow_state(self) -> WalletFlowState:
        """Return current accumulated flow state without fetching."""
        return WalletFlowState(
            wallet_volumes=dict(self._wallet_volumes),
            total_bull_volume=self._total_bull,
            total_bear_volume=self._total_bear,
            bull_wallets=set(self._bull_wallets),
            bear_wallets=set(self._bear_wallets),
            trades=list(self._trades),
        )

    def _classify_token(self, token_id: int) -> str | None:
        """Map a token ID to 'YES', 'NO', or None."""
        if token_id == self._yes_token_id:
            return "YES"
        if token_id == self._no_token_id:
            return "NO"
        return None

    def _process_logs(self, logs: list[dict]) -> None:
        """Parse and accumulate ALL trades from raw log entries."""
        for log in logs:
            parsed = _parse_log(log)
            if parsed is None:
                continue

            from_addr, to_addr, token_id, raw_value = parsed
            if raw_value == 0:
                continue

            token = self._classify_token(token_id)
            if token is None:
                continue

            value = raw_value / (10 ** TOKEN_DECIMALS)

            # Receiver is buying this token
            if to_addr and to_addr != "0x" + "0" * 40:
                if token == "YES":
                    direction = "BULL"
                else:
                    direction = "BEAR"
                self._record_trade(to_addr, direction, token, value)

            # Sender is selling this token
            if from_addr and from_addr != "0x" + "0" * 40:
                if token == "YES":
                    direction = "BEAR"
                else:
                    direction = "BULL"
                self._record_trade(from_addr, direction, token, value)

    def _record_trade(
        self, wallet: str, direction: str, token: str, value: float,
    ) -> None:
        """Record a single trade into accumulators."""
        # Per-wallet volume
        if wallet not in self._wallet_volumes:
            self._wallet_volumes[wallet] = {"bull": 0.0, "bear": 0.0}

        if direction == "BULL":
            self._wallet_volumes[wallet]["bull"] += value
            self._total_bull += value
            self._bull_wallets.add(wallet)
        else:
            self._wallet_volumes[wallet]["bear"] += value
            self._total_bear += value
            self._bear_wallets.add(wallet)

        self._trades.append(WalletTrade(
            wallet=wallet,
            direction=direction,
            token=token,
            value=value,
        ))

    async def _rpc_call(self, payload: dict) -> dict | None:
        """Make a JSON-RPC call with fallback across RPCs."""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=5.0)

        for rpc_url in self._rpcs:
            try:
                resp = await self._http_client.post(rpc_url, json=payload)
                data = resp.json()
                if "error" in data:
                    logger.debug(
                        "Wallet flow RPC error via %s: %s",
                        rpc_url, data["error"],
                    )
                    continue
                return data
            except Exception as e:
                logger.debug("Wallet flow RPC failed via %s: %s", rpc_url, e)
                continue

        logger.warning("Wallet flow monitor: all RPC endpoints failed")
        return None

    async def _get_block_number(self) -> int | None:
        """Fetch the current block number."""
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_blockNumber",
            "params": [],
            "id": 1,
        }
        data = await self._rpc_call(payload)
        if data and "result" in data:
            return int(data["result"], 16)
        return None

    async def _fetch_logs(self, from_block: int, to_block: int) -> list[dict]:
        """Fetch TransferSingle logs from the CTF contract."""
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getLogs",
            "params": [
                {
                    "address": CTF_CONTRACT,
                    "topics": [TRANSFER_SINGLE_TOPIC],
                    "fromBlock": hex(from_block),
                    "toBlock": hex(to_block),
                }
            ],
            "id": 2,
        }
        data = await self._rpc_call(payload)
        if data and "result" in data:
            return data["result"]
        return []

    async def fetch_once(self) -> WalletFlowState:
        """Single poll: fetch new trades since last block, accumulate.

        Returns:
            Current accumulated WalletFlowState.
        """
        current_block = await self._get_block_number()
        if current_block is None:
            return self.get_flow_state()

        # First call for this window — start from current block
        if self._last_block == 0:
            self._last_block = current_block
            return self.get_flow_state()

        if current_block <= self._last_block:
            return self.get_flow_state()

        # Cap block range to avoid huge queries (max ~500 blocks = ~16 min)
        from_block = self._last_block + 1
        if current_block - from_block > 500:
            from_block = current_block - 500

        logs = await self._fetch_logs(from_block, current_block)
        self._process_logs(logs)
        self._last_block = current_block
        self._last_fetch_time = time.time()

        return self.get_flow_state()

    async def start(self, poll_interval: float = 3.0) -> None:
        """Start background polling loop.

        Args:
            poll_interval: Seconds between polls. Default 3.0.
        """
        self._http_client = httpx.AsyncClient(timeout=5.0)
        self._running = True
        logger.info(
            "Wallet flow monitor started (poll every %.1fs)", poll_interval,
        )

        while self._running:
            try:
                await self.fetch_once()
            except Exception as e:
                logger.error(
                    "Wallet flow monitor poll error: %s", e, exc_info=True,
                )
            await asyncio.sleep(poll_interval)

    async def stop(self) -> None:
        """Stop polling and close the HTTP client."""
        self._running = False
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("Wallet flow monitor stopped")
