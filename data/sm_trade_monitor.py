"""SM Trade Monitor — on-chain SM wallet trade tracking for L9.

Polls Polygon RPC via eth_getLogs for ERC-1155 TransferSingle events on
the Polymarket CTF Exchange contract. Filters for SM wallet addresses
and accumulates per-window flow (YES/NO volume split, distinct wallets).

Produces SMFlowState consumed by strategy.sm_confirmation.check_sm_confirmation().

Follows the same RPC call pattern as data.chainlink_oracle (httpx, fallback RPCs).
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from data.sm_wallets import SMWalletRegistry
from strategy.sm_confirmation import SMFlowState

logger = logging.getLogger(__name__)

# Polymarket Conditional Token Framework (CTF) contract on Polygon
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# ERC-1155 TransferSingle(address,address,address,uint256,uint256)
# keccak256 of the event signature
TRANSFER_SINGLE_TOPIC = (
    "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"
)

# CTF token amounts are in 1e6 scale (USDC-denominated shares)
TOKEN_DECIMALS = 6

# Free Polygon RPCs — same list as chainlink_oracle.py
DEFAULT_RPCS = [
    "https://polygon.drpc.org",
    "https://1rpc.io/matic",
    "https://polygon-bor-rpc.publicnode.com",
]


def _addr_from_topic(topic: str) -> str:
    """Extract a lowercase address from a 32-byte hex-encoded log topic."""
    return "0x" + topic[-40:].lower()


def _parse_log(log: dict) -> tuple[str, str, int, int] | None:
    """Parse a TransferSingle log entry.

    Args:
        log: Raw log dict from eth_getLogs response.

    Returns:
        (from_addr, to_addr, token_id, value) or None if unparseable.
    """
    topics = log.get("topics", [])
    data = log.get("data", "")

    if len(topics) < 4 or topics[0].lower() != TRANSFER_SINGLE_TOPIC:
        return None

    from_addr = _addr_from_topic(topics[2])
    to_addr = _addr_from_topic(topics[3])

    # data contains ABI-encoded (uint256 id, uint256 value)
    data_hex = data[2:] if data.startswith("0x") else data
    if len(data_hex) < 128:
        return None

    token_id = int(data_hex[0:64], 16)
    value = int(data_hex[64:128], 16)

    return from_addr, to_addr, token_id, value


@dataclass
class SMTradeMonitor:
    """Monitors on-chain SM wallet trades on the Polymarket CTF contract.

    Polls Polygon RPC for TransferSingle events, filters by SM wallet
    addresses, and accumulates per-window flow state.

    Args:
        sm_registry: SM wallet registry for address filtering.
        rpcs: List of Polygon RPC URLs. Defaults to free public RPCs.
    """

    _sm_registry: SMWalletRegistry
    _rpcs: list[str] = field(default_factory=list)

    # Market state
    _yes_token_id: int = 0
    _no_token_id: int = 0
    _window_id: str = ""

    # Accumulated flow
    _yes_volume: float = 0.0
    _no_volume: float = 0.0
    _sm_wallets_seen: set[str] = field(default_factory=set)

    # Block tracking
    _last_block: int = 0

    # Infra
    _http_client: httpx.AsyncClient | None = field(default=None, repr=False)
    _running: bool = False
    _last_fetch_time: float = 0.0

    def __post_init__(self) -> None:
        if not self._rpcs:
            rpc_env = os.environ.get("POLYGON_RPC_URL", "").strip()
            self._rpcs = [rpc_env] + DEFAULT_RPCS if rpc_env else list(DEFAULT_RPCS)
        # Ensure mutable default
        if not isinstance(self._sm_wallets_seen, set):
            self._sm_wallets_seen = set()

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
            logger.info(
                "SM monitor: new window %s, tokens YES=%s NO=%s",
                window_id, yes_token_id, no_token_id,
            )

        self._yes_token_id = int(yes_token_id)
        self._no_token_id = int(no_token_id)

    def reset(self) -> None:
        """Zero all accumulated flow state."""
        self._yes_volume = 0.0
        self._no_volume = 0.0
        self._sm_wallets_seen.clear()
        self._last_block = 0

    def get_flow_state(self) -> SMFlowState:
        """Return current accumulated flow state without fetching."""
        return SMFlowState(
            yes_volume=self._yes_volume,
            no_volume=self._no_volume,
            num_wallets=len(self._sm_wallets_seen),
        )

    def _classify_token(self, token_id: int) -> str | None:
        """Map a token ID to 'YES', 'NO', or None."""
        if token_id == self._yes_token_id:
            return "YES"
        if token_id == self._no_token_id:
            return "NO"
        return None

    def _process_logs(self, logs: list[dict]) -> None:
        """Parse and accumulate SM wallet trades from raw log entries."""
        sm_addresses = self._sm_registry.get_all_addresses()

        for log in logs:
            parsed = _parse_log(log)
            if parsed is None:
                continue

            from_addr, to_addr, token_id, raw_value = parsed
            if raw_value == 0:
                continue

            side = self._classify_token(token_id)
            if side is None:
                continue

            value = raw_value / (10 ** TOKEN_DECIMALS)
            from_is_sm = from_addr in sm_addresses
            to_is_sm = to_addr in sm_addresses

            if to_is_sm:
                # SM wallet is receiving (buying) this token
                if side == "YES":
                    self._yes_volume += value
                else:
                    self._no_volume += value
                self._sm_wallets_seen.add(to_addr)

            if from_is_sm:
                # SM wallet is sending (selling) this token
                # Selling YES is bearish → add to NO volume
                # Selling NO is bullish → add to YES volume
                if side == "YES":
                    self._no_volume += value
                else:
                    self._yes_volume += value
                self._sm_wallets_seen.add(from_addr)

    async def _rpc_call(self, payload: dict) -> dict | None:
        """Make a JSON-RPC call with fallback across RPCs.

        Returns:
            Parsed JSON response, or None if all RPCs failed.
        """
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=5.0)

        for rpc_url in self._rpcs:
            try:
                resp = await self._http_client.post(rpc_url, json=payload)
                data = resp.json()
                if "error" in data:
                    logger.debug(
                        "SM monitor RPC error via %s: %s",
                        urlsplit(rpc_url).hostname or "<rpc>", data["error"],
                    )
                    continue
                return data
            except Exception as e:
                logger.debug("SM monitor RPC failed via %s: %s", urlsplit(rpc_url).hostname or "<rpc>", e)
                continue

        logger.warning("SM monitor: all RPC endpoints failed")
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

    async def fetch_once(self) -> SMFlowState:
        """Single poll: fetch new SM trades since last block, accumulate.

        Returns:
            Current accumulated SMFlowState.
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
        logger.info("SM trade monitor started (poll every %.1fs)", poll_interval)

        while self._running:
            try:
                await self.fetch_once()
            except Exception as e:
                logger.error("SM monitor poll error: %s", e, exc_info=True)
            await asyncio.sleep(poll_interval)

    async def stop(self) -> None:
        """Stop polling and close the HTTP client."""
        self._running = False
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("SM trade monitor stopped")
