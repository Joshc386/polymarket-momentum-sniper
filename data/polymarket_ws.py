"""Polymarket CLOB WebSocket feed — real-time orderbook streaming.

Connects to wss://ws-subscriptions-clob.polymarket.com/ws/market and
streams live orderbook updates for the current YES/NO token pair.

Replaces REST polling (2s interval) with sub-second push updates.
Produces the same OrderbookSummary dataclass used by the rest of the system.

Usage:
    ws_feed = PolymarketWebSocket()
    task = asyncio.create_task(ws_feed.start())
    ws_feed.subscribe(yes_token_id, no_token_id)
    ...
    summary = ws_feed.last_summary  # Always up-to-date
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import websockets

from core import fast_json as json
from websockets.exceptions import ConnectionClosed

from data.orderbook import OrderLevel, OrderbookSummary

logger = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
PING_INTERVAL = 10  # Server requires PING every 10 seconds
STALE_TIMEOUT = 30  # Force reconnect if no events for this many seconds


@dataclass
class CLOBTradeFlow:
    """Snapshot of recent Polymarket CLOB trade flow.

    Tracks buy/sell volume on YES and NO tokens from last_trade_price
    events within a rolling window. This is real order flow data —
    actual shares being bought and sold on the market.
    """

    # YES token flow
    yes_buy_volume: float = 0.0    # Total size of BUY trades on YES
    yes_sell_volume: float = 0.0   # Total size of SELL trades on YES
    yes_trade_count: int = 0       # Number of YES trades in window

    # NO token flow
    no_buy_volume: float = 0.0
    no_sell_volume: float = 0.0
    no_trade_count: int = 0

    # Derived
    total_volume: float = 0.0      # All volume across both tokens
    net_yes_flow: float = 0.0      # yes_buy - yes_sell (positive = bullish)
    imbalance: float = 0.0         # [-1, 1] — positive = net buying YES

    # Last trade
    last_trade_side: str = ""      # "BUY" or "SELL"
    last_trade_token: str = ""     # "YES" or "NO"
    last_trade_price: float = 0.0
    last_trade_size: float = 0.0
    last_trade_time: float = 0.0

    # Individual trades for size analysis (L11).
    # Each tuple: (timestamp, token, side, price, size).
    # Populated from the rolling window — same trades used to compute
    # the aggregate fields above.
    recent_trades: list = field(default_factory=list)


class _TradeFlowTracker:
    """Accumulates CLOB trade events in a rolling window."""

    def __init__(self, window_secs: float = 60.0) -> None:
        self._window_secs = window_secs
        # Each entry: (timestamp, token, side, price, size)
        self._trades: deque[tuple[float, str, str, float, float]] = deque()

    def reset(self) -> None:
        """Clear all tracked trades (e.g. on market change)."""
        self._trades.clear()

    def add_trade(
        self, token: str, side: str, price: float, size: float, timestamp: float
    ) -> None:
        """Record a trade from a last_trade_price WS event.

        Args:
            token: "YES" or "NO".
            side: "BUY" or "SELL".
            price: Trade price (0-1 range).
            size: Number of shares traded.
            timestamp: Unix timestamp.
        """
        self._trades.append((timestamp, token, side, price, size))

    def snapshot(self) -> CLOBTradeFlow:
        """Build a CLOBTradeFlow snapshot from the rolling window."""
        now = time.time()
        cutoff = now - self._window_secs

        # Prune old trades
        while self._trades and self._trades[0][0] < cutoff:
            self._trades.popleft()

        flow = CLOBTradeFlow()

        for ts, token, side, price, size in self._trades:
            if token == "YES":
                flow.yes_trade_count += 1
                if side == "BUY":
                    flow.yes_buy_volume += size
                else:
                    flow.yes_sell_volume += size
            else:
                flow.no_trade_count += 1
                if side == "BUY":
                    flow.no_buy_volume += size
                else:
                    flow.no_sell_volume += size

        flow.total_volume = (
            flow.yes_buy_volume + flow.yes_sell_volume
            + flow.no_buy_volume + flow.no_sell_volume
        )
        flow.net_yes_flow = flow.yes_buy_volume - flow.yes_sell_volume

        # Imbalance: how much of the total flow is net-buying YES
        # Range [-1, 1]: +1 = all buying YES, -1 = all selling YES
        total_directional = (
            flow.yes_buy_volume + flow.no_sell_volume   # Bullish actions
            + flow.yes_sell_volume + flow.no_buy_volume  # Bearish actions
        )
        if total_directional > 0:
            bullish = flow.yes_buy_volume + flow.no_sell_volume
            bearish = flow.yes_sell_volume + flow.no_buy_volume
            flow.imbalance = (bullish - bearish) / total_directional

        # Last trade
        if self._trades:
            ts, token, side, price, size = self._trades[-1]
            flow.last_trade_side = side
            flow.last_trade_token = token
            flow.last_trade_price = price
            flow.last_trade_size = size
            flow.last_trade_time = ts

        # Expose individual trades for L11 trade size analysis
        flow.recent_trades = list(self._trades)

        return flow


@dataclass
class _BookState:
    """Internal mutable orderbook state for one token (YES or NO)."""

    bids: dict[str, float] = field(default_factory=dict)  # price_str -> size
    asks: dict[str, float] = field(default_factory=dict)
    best_bid: float = 0.0
    best_ask: float = 0.0
    last_trade_price: float | None = None

    def apply_book_snapshot(self, raw_bids: list[dict], raw_asks: list[dict]) -> None:
        """Replace entire book with a full snapshot."""
        self.bids.clear()
        self.asks.clear()
        for b in raw_bids:
            price_str = str(b["price"])
            size = float(b["size"])
            if size > 0:
                self.bids[price_str] = size
        for a in raw_asks:
            price_str = str(a["price"])
            size = float(a["size"])
            if size > 0:
                self.asks[price_str] = size
        self._update_bba()

    def apply_price_change(self, price: str, size: str, side: str) -> None:
        """Apply an incremental price_change update.

        A size of "0" means remove that level.
        """
        sz = float(size)
        book = self.bids if side == "BUY" else self.asks
        if sz <= 0:
            book.pop(price, None)
        else:
            book[price] = sz
        self._update_bba()

    def apply_bba(self, best_bid: str | None, best_ask: str | None) -> None:
        """Apply a best_bid_ask event (quick BBA update)."""
        if best_bid is not None:
            self.best_bid = float(best_bid)
        if best_ask is not None:
            self.best_ask = float(best_ask)

    def _update_bba(self) -> None:
        """Recompute best bid/ask from the full book."""
        if self.bids:
            self.best_bid = max(float(p) for p in self.bids)
        else:
            self.best_bid = 0.0
        if self.asks:
            self.best_ask = min(float(p) for p in self.asks)
        else:
            self.best_ask = 0.0

    def sorted_bid_levels(self) -> list[OrderLevel]:
        """Return bid levels sorted highest-first."""
        levels = [
            OrderLevel(price=float(p), size=s) for p, s in self.bids.items()
        ]
        levels.sort(key=lambda x: x.price, reverse=True)
        return levels

    def sorted_ask_levels(self) -> list[OrderLevel]:
        """Return ask levels sorted lowest-first."""
        levels = [
            OrderLevel(price=float(p), size=s) for p, s in self.asks.items()
        ]
        levels.sort(key=lambda x: x.price)
        return levels


class PolymarketWebSocket:
    """Real-time orderbook feed via Polymarket CLOB WebSocket.

    Thread-safe: updates internal state from the WS task, read by the
    main loop via .last_summary or .build_summary().

    Attributes:
        last_summary: Most recent OrderbookSummary (or None before first book event).
        is_connected: Whether the websocket is currently connected.
        updates_received: Total number of WS messages processed.
        last_update_time: Timestamp of most recent message.
    """

    def __init__(self) -> None:
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._yes_state = _BookState()
        self._no_state = _BookState()
        self._yes_token_id: str = ""
        self._no_token_id: str = ""
        self._running: bool = False
        self._subscribe_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._reconnect_delay: float = 1.0

        # Public state
        self.last_summary: OrderbookSummary | None = None
        self.is_connected: bool = False
        self.updates_received: int = 0
        self.last_update_time: float = 0.0

        # Order flow deltas (matching OrderbookManager interface)
        self._prev_yes_bid_total: float = 0.0
        self._prev_yes_ask_total: float = 0.0
        self._prev_no_bid_total: float = 0.0
        self._prev_no_ask_total: float = 0.0
        self.yes_bid_delta: float = 0.0
        self.yes_ask_delta: float = 0.0
        self.no_bid_delta: float = 0.0
        self.no_ask_delta: float = 0.0

        # CLOB trade flow tracking (from last_trade_price events)
        self._trade_flow = _TradeFlowTracker(window_secs=60.0)
        self.trade_flow: CLOBTradeFlow = CLOBTradeFlow()

    @property
    def is_fresh(self) -> bool:
        """Whether the WS data is fresh enough to use.

        Returns False if disconnected, no data yet, or no updates
        received within STALE_TIMEOUT seconds.
        """
        if not self.is_connected or self.last_summary is None:
            return False
        if self.last_update_time <= 0:
            return False
        return (time.time() - self.last_update_time) < STALE_TIMEOUT

    def subscribe(self, yes_token_id: str, no_token_id: str) -> None:
        """Queue a subscription change (called from main loop).

        Unsubscribes from old tokens and subscribes to new ones.
        Safe to call before start() — the subscription is queued.
        """
        if yes_token_id == self._yes_token_id and no_token_id == self._no_token_id:
            return  # Already subscribed
        self._subscribe_queue.put_nowait((yes_token_id, no_token_id))

    async def start(self) -> None:
        """Connect and run the websocket event loop forever.

        Reconnects automatically on disconnect.
        """
        self._running = True
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"WS connection error: {e}")

            self.is_connected = False
            if not self._running:
                break

            logger.info(
                f"WS reconnecting in {self._reconnect_delay:.0f}s..."
            )
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)

    async def stop(self) -> None:
        """Gracefully disconnect."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        self.is_connected = False

    def close(self) -> None:
        """Synchronous cleanup (for shutdown sequences)."""
        self._running = False

    async def _connect_and_listen(self) -> None:
        """Single connection lifecycle: connect, subscribe, listen."""
        logger.info(f"Connecting to Polymarket WS: {WS_URL}")

        async with websockets.connect(
            WS_URL,
            ping_interval=None,  # We handle pings manually
            close_timeout=5,
            max_size=10 * 1024 * 1024,  # 10MB max message
        ) as ws:
            self._ws = ws
            self.is_connected = True
            self._reconnect_delay = 1.0  # Reset on successful connect
            logger.info("Polymarket WS connected")

            # Subscribe to any queued tokens
            if self._yes_token_id and self._no_token_id:
                await self._send_subscribe(
                    ws, self._yes_token_id, self._no_token_id
                )

            # Run ping + listen concurrently
            ping_task = asyncio.create_task(self._ping_loop(ws))
            sub_task = asyncio.create_task(self._subscription_handler(ws))

            try:
                async for raw_msg in ws:
                    if not self._running:
                        break
                    try:
                        self._handle_message(raw_msg)
                    except Exception as e:
                        logger.debug(f"WS message parse error: {e}")
            except ConnectionClosed as e:
                logger.info(f"WS disconnected: {e}")
            finally:
                ping_task.cancel()
                sub_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass
                try:
                    await sub_task
                except asyncio.CancelledError:
                    pass

    async def _ping_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Send PING every 10 seconds and check for staleness.

        If no market events received for STALE_TIMEOUT seconds, force
        a reconnect by closing the websocket. This handles cases where
        the TCP connection stays alive but the server stops sending data
        (e.g. after a market transition or server-side issue).
        """
        while True:
            try:
                await ws.send("PING")

                # Check for staleness: if subscribed but no updates
                if (
                    self._yes_token_id
                    and self.last_update_time > 0
                    and (time.time() - self.last_update_time) > STALE_TIMEOUT
                ):
                    logger.warning(
                        f"WS stale — no events for {STALE_TIMEOUT}s, "
                        f"forcing reconnect"
                    )
                    await ws.close()
                    break

                await asyncio.sleep(PING_INTERVAL)
            except (ConnectionClosed, asyncio.CancelledError):
                break

    async def _subscription_handler(
        self, ws: websockets.WebSocketClientProtocol
    ) -> None:
        """Process subscription change requests from the main loop."""
        while True:
            try:
                yes_id, no_id = await self._subscribe_queue.get()
            except asyncio.CancelledError:
                break

            # Unsubscribe old tokens
            if self._yes_token_id or self._no_token_id:
                old_ids = [
                    t for t in [self._yes_token_id, self._no_token_id] if t
                ]
                if old_ids:
                    try:
                        unsub_msg = json.dumps({
                            "assets_ids": old_ids,
                            "operation": "unsubscribe",
                        })
                        await ws.send(unsub_msg)
                        logger.debug(f"Unsubscribed from {len(old_ids)} tokens")
                    except ConnectionClosed:
                        break

            # Reset book state and trade flow for new tokens
            self._yes_state = _BookState()
            self._no_state = _BookState()
            self._trade_flow.reset()
            self.trade_flow = CLOBTradeFlow()

            # Subscribe new tokens
            await self._send_subscribe(ws, yes_id, no_id)

    async def _send_subscribe(
        self,
        ws: websockets.WebSocketClientProtocol,
        yes_token_id: str,
        no_token_id: str,
    ) -> None:
        """Send subscription message for both tokens."""
        self._yes_token_id = yes_token_id
        self._no_token_id = no_token_id

        asset_ids = [t for t in [yes_token_id, no_token_id] if t]
        if not asset_ids:
            return

        sub_msg = json.dumps({
            "assets_ids": asset_ids,
            "type": "market",
            "custom_feature_enabled": True,
        })
        await ws.send(sub_msg)
        logger.info(
            f"Subscribed to {len(asset_ids)} tokens "
            f"(YES={yes_token_id[:12]}..., NO={no_token_id[:12]}...)"
        )

    def _handle_message(self, raw: str) -> None:
        """Route an incoming WS message to the appropriate handler."""
        if raw == "PONG":
            return

        data = json.loads(raw)

        # Messages can be a list of events
        events = data if isinstance(data, list) else [data]

        for event in events:
            event_type = event.get("event_type", "")
            asset_id = event.get("asset_id", "")

            # Determine which book this event applies to
            if asset_id == self._yes_token_id:
                state = self._yes_state
            elif asset_id == self._no_token_id:
                state = self._no_state
            else:
                # Events like price_change have nested data
                if event_type == "price_change":
                    self._handle_price_change(event)
                    continue
                # Unknown asset — skip
                continue

            if event_type == "book":
                state.apply_book_snapshot(
                    event.get("bids", []),
                    event.get("asks", []),
                )
                self.updates_received += 1
                self.last_update_time = time.time()
                self._rebuild_summary()

            elif event_type == "best_bid_ask":
                state.apply_bba(
                    event.get("best_bid"),
                    event.get("best_ask"),
                )
                self.updates_received += 1
                self.last_update_time = time.time()
                self._rebuild_summary()

            elif event_type == "last_trade_price":
                price = event.get("price")
                if price is not None:
                    state.last_trade_price = float(price)

                # Track trade flow for volume signal
                token = "YES" if asset_id == self._yes_token_id else "NO"
                trade_side = event.get("side", "")
                trade_size = float(event.get("size", 0))
                trade_price = float(price) if price else 0.0
                if trade_side and trade_size > 0:
                    self._trade_flow.add_trade(
                        token=token,
                        side=trade_side,
                        price=trade_price,
                        size=trade_size,
                        timestamp=time.time(),
                    )
                    self.trade_flow = self._trade_flow.snapshot()

                self.updates_received += 1
                self.last_update_time = time.time()

    def _handle_price_change(self, event: dict) -> None:
        """Handle price_change events with nested price_changes array."""
        changes = event.get("price_changes", [])
        for change in changes:
            asset_id = change.get("asset_id", "")
            if asset_id == self._yes_token_id:
                state = self._yes_state
            elif asset_id == self._no_token_id:
                state = self._no_state
            else:
                continue

            state.apply_price_change(
                price=str(change.get("price", "0")),
                size=str(change.get("size", "0")),
                side=change.get("side", "BUY"),
            )

        self.updates_received += 1
        self.last_update_time = time.time()
        self._rebuild_summary()

    def _rebuild_summary(self) -> None:
        """Rebuild OrderbookSummary from current YES/NO book state."""
        yes = self._yes_state
        no = self._no_state

        yes_bid_levels = yes.sorted_bid_levels()
        yes_ask_levels = yes.sorted_ask_levels()
        no_bid_levels = no.sorted_bid_levels()
        no_ask_levels = no.sorted_ask_levels()

        yes_mid = (
            (yes.best_bid + yes.best_ask) / 2
            if yes.best_bid > 0 and yes.best_ask > 0
            else 0.5
        )
        no_mid = (
            (no.best_bid + no.best_ask) / 2
            if no.best_bid > 0 and no.best_ask > 0
            else 0.5
        )
        spread = (
            yes.best_ask - yes.best_bid
            if yes.best_ask > 0 and yes.best_bid > 0
            else 0.0
        )

        yes_bid_total = sum(l.size for l in yes_bid_levels)
        yes_ask_total = sum(l.size for l in yes_ask_levels)
        no_bid_total = sum(l.size for l in no_bid_levels)
        no_ask_total = sum(l.size for l in no_ask_levels)

        # Compute order flow deltas
        self.yes_bid_delta = yes_bid_total - self._prev_yes_bid_total
        self.yes_ask_delta = yes_ask_total - self._prev_yes_ask_total
        self.no_bid_delta = no_bid_total - self._prev_no_bid_total
        self.no_ask_delta = no_ask_total - self._prev_no_ask_total

        self._prev_yes_bid_total = yes_bid_total
        self._prev_yes_ask_total = yes_ask_total
        self._prev_no_bid_total = no_bid_total
        self._prev_no_ask_total = no_ask_total

        # Size-weighted mid (same logic as OrderbookManager)
        sw_mid = self._compute_size_weighted_mid(
            yes_bid_levels, yes_ask_levels, yes_mid
        )

        self.last_summary = OrderbookSummary(
            yes_best_bid=yes.best_bid,
            yes_best_ask=yes.best_ask,
            yes_midpoint=yes_mid,
            no_best_bid=no.best_bid,
            no_best_ask=no.best_ask,
            no_midpoint=no_mid,
            yes_bid_depth=sum(l.size for l in yes_bid_levels[:5]),
            yes_ask_depth=sum(l.size for l in yes_ask_levels[:5]),
            spread=spread,
            last_trade_price_yes=yes.last_trade_price,
            yes_bid_levels=yes_bid_levels,
            yes_ask_levels=yes_ask_levels,
            no_bid_levels=no_bid_levels,
            no_ask_levels=no_ask_levels,
            yes_bid_total_size=yes_bid_total,
            yes_ask_total_size=yes_ask_total,
            no_bid_total_size=no_bid_total,
            no_ask_total_size=no_ask_total,
            yes_size_weighted_mid=sw_mid,
            num_yes_bid_levels=len(yes_bid_levels),
            num_yes_ask_levels=len(yes_ask_levels),
        )

    @staticmethod
    def _compute_size_weighted_mid(
        bid_levels: list[OrderLevel],
        ask_levels: list[OrderLevel],
        simple_mid: float,
    ) -> float:
        """Compute size-weighted midpoint using top levels.

        Identical logic to OrderbookManager._compute_size_weighted_mid.
        """
        if not bid_levels or not ask_levels:
            return simple_mid

        top_bids = bid_levels[:3]
        top_asks = ask_levels[:3]

        bid_vwap = sum(l.price * l.size for l in top_bids)
        bid_vol = sum(l.size for l in top_bids)
        ask_vwap = sum(l.price * l.size for l in top_asks)
        ask_vol = sum(l.size for l in top_asks)

        if bid_vol <= 0 or ask_vol <= 0:
            return simple_mid

        total_vol = bid_vol + ask_vol
        bid_weight = bid_vol / total_vol
        ask_weight = ask_vol / total_vol

        weighted_bid_price = bid_vwap / bid_vol
        weighted_ask_price = ask_vwap / ask_vol

        return weighted_ask_price * bid_weight + weighted_bid_price * ask_weight
