"""Execution engines for paper and live trading.

Both engines share the same interface (execute_trade / resolve_pending_trade)
so main.py can swap between them based on config.mode.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from logging_db.database import Database

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """A completed (or pending) trade."""
    market_id: str
    market_slug: str
    side: str               # "YES" or "NO"
    entry_price: float
    size_usdc: float
    num_shares: float
    oracle_lag_signal: float
    momentum_signal: float
    liquidation_signal: float
    combined_signal: float
    estimated_prob_up: float
    market_implied_prob: float
    edge: float
    time_remaining_secs: float
    btc_price_at_entry: float
    oracle_price_at_entry: float
    oracle_price_at_open: float
    is_paper: bool
    resolution: str | None = None  # "UP" or "DOWN" or None
    pnl: float | None = None
    db_id: int | None = None
    order_id: str | None = None


# ── Shared trade parameters (used by both engines) ────────────────────

_TRADE_FIELDS = [
    "side", "price", "size_usdc", "market_id", "market_slug",
    "oracle_lag_signal", "momentum_signal", "liquidation_signal",
    "combined_signal", "estimated_prob_up", "market_implied_prob",
    "edge", "time_remaining_secs", "btc_price", "oracle_price",
    "oracle_open_price",
]


def _log_to_db(db, trade, is_paper: bool) -> int:
    """Insert a trade into the database. Returns row ID."""
    now_str = datetime.now(timezone.utc).isoformat()
    return db.insert_trade(
        timestamp=now_str,
        market_id=trade.market_id,
        market_slug=trade.market_slug,
        side=trade.side,
        entry_price=trade.entry_price,
        size_usdc=trade.size_usdc,
        num_shares=trade.num_shares,
        oracle_lag_signal=trade.oracle_lag_signal,
        momentum_signal=trade.momentum_signal,
        liquidation_signal=trade.liquidation_signal,
        combined_signal=trade.combined_signal,
        estimated_prob_up=trade.estimated_prob_up,
        market_implied_prob=trade.market_implied_prob,
        edge=trade.edge,
        time_remaining_secs=trade.time_remaining_secs,
        is_paper=1 if is_paper else 0,
        btc_price_at_entry=trade.btc_price_at_entry,
        oracle_price_at_entry=trade.oracle_price_at_entry,
        oracle_price_at_open=trade.oracle_price_at_open,
        fill_price=trade.entry_price,
        order_id=trade.order_id,
    )


def _resolve_trade(trade: TradeRecord, resolution: str) -> bool:
    """Compute P&L for a trade. Returns True if won."""
    trade.resolution = resolution
    won = (trade.side == "YES" and resolution == "UP") or \
          (trade.side == "NO" and resolution == "DOWN")

    if won:
        payout = trade.num_shares * 1.0
        gross_profit = payout - trade.size_usdc
        fee = gross_profit * 0.02 if gross_profit > 0 else 0
        trade.pnl = gross_profit - fee
    else:
        trade.pnl = -trade.size_usdc

    return won


# ══════════════════════════════════════════════════════════════════════
#  PAPER EXECUTION ENGINE
# ══════════════════════════════════════════════════════════════════════

class PaperExecutionEngine:
    """Simulates order execution for paper trading.

    Fills at current best ask + configurable slippage.
    Tracks virtual bankroll and logs all trades to SQLite.
    """

    def __init__(
        self,
        db: Database,
        initial_bankroll: float = 100.0,
        slippage: float = 0.005,
    ):
        self.db = db
        self.bankroll = initial_bankroll
        self.slippage = slippage
        self.pending_trade: TradeRecord | None = None
        self.session_trades: list[TradeRecord] = []
        self.session_wins = 0
        self.session_losses = 0
        self.session_pnl = 0.0
        self.is_paper = True

    def execute_paper_trade(
        self,
        side: str,
        price: float,
        size_usdc: float,
        market_id: str,
        market_slug: str,
        oracle_lag_signal: float,
        momentum_signal: float,
        liquidation_signal: float,
        combined_signal: float,
        estimated_prob_up: float,
        market_implied_prob: float,
        edge: float,
        time_remaining_secs: float,
        btc_price: float,
        oracle_price: float,
        oracle_open_price: float,
    ) -> TradeRecord | None:
        """Simulate placing a trade. Returns TradeRecord if 'filled'."""
        fill_price = min(price + self.slippage, 0.99)
        num_shares = size_usdc / fill_price
        self.bankroll -= size_usdc

        trade = TradeRecord(
            market_id=market_id, market_slug=market_slug, side=side,
            entry_price=fill_price, size_usdc=size_usdc, num_shares=num_shares,
            oracle_lag_signal=oracle_lag_signal, momentum_signal=momentum_signal,
            liquidation_signal=liquidation_signal, combined_signal=combined_signal,
            estimated_prob_up=estimated_prob_up, market_implied_prob=market_implied_prob,
            edge=edge, time_remaining_secs=time_remaining_secs,
            btc_price_at_entry=btc_price, oracle_price_at_entry=oracle_price,
            oracle_price_at_open=oracle_open_price, is_paper=True,
        )

        trade.db_id = _log_to_db(self.db, trade, is_paper=True)
        self.pending_trade = trade

        logger.info(
            f"[PAPER] {side} @ ${fill_price:.4f} | "
            f"Size: ${size_usdc:.2f} | Shares: {num_shares:.2f} | "
            f"Edge: {edge:+.4f}"
        )
        return trade

    # Alias so main.py can call executor.execute_trade() uniformly
    execute_trade = execute_paper_trade

    def resolve_pending_trade(self, resolution: str) -> TradeRecord | None:
        """Resolve a pending trade."""
        trade = self.pending_trade
        if not trade:
            return None

        won = _resolve_trade(trade, resolution)

        if won:
            self.session_wins += 1
        else:
            self.session_losses += 1

        self.bankroll += trade.size_usdc + trade.pnl
        self.session_pnl += trade.pnl
        self.session_trades.append(trade)

        if trade.db_id and self.db.conn:
            self.db.conn.execute(
                "UPDATE trades SET resolution = ?, pnl = ? WHERE id = ?",
                (resolution, trade.pnl, trade.db_id),
            )
            self.db.conn.commit()

        self.pending_trade = None

        logger.info(
            f"[PAPER] Resolved {resolution} | {trade.side} | "
            f"{'WIN' if won else 'LOSS'} | P&L: ${trade.pnl:+.2f} | "
            f"Bankroll: ${self.bankroll:.2f}"
        )
        return trade

    @property
    def win_rate(self) -> float:
        total = self.session_wins + self.session_losses
        return self.session_wins / total if total > 0 else 0.0

    @property
    def total_trades(self) -> int:
        return self.session_wins + self.session_losses


# ══════════════════════════════════════════════════════════════════════
#  LIVE EXECUTION ENGINE
# ══════════════════════════════════════════════════════════════════════

class LiveExecutionEngine:
    """Real order execution on Polymarket CLOB.

    Order strategy:
    - >60s remaining: GTC limit at best bid/ask. Cancel and re-evaluate if
      not filled within gtc_timeout_sec.
    - <=60s remaining: FOK limit slightly above best ask (accept slippage
      for fill certainty).
    - Never market orders (orderbook can be thin).

    Tracks real bankroll via on-chain balance and logs to SQLite.
    """

    def __init__(
        self,
        db: Database,
        poly_client,              # PolymarketClient instance
        initial_bankroll: float = 100.0,
        fok_slippage: float = 0.005,
        gtc_timeout_sec: int = 10,
    ):
        self.db = db
        self.poly = poly_client
        self.bankroll = initial_bankroll
        self.fok_slippage = fok_slippage
        self.gtc_timeout_sec = gtc_timeout_sec
        self.pending_trade: TradeRecord | None = None
        self.pending_order_id: str | None = None
        self.session_trades: list[TradeRecord] = []
        self.session_wins = 0
        self.session_losses = 0
        self.session_pnl = 0.0
        self.is_paper = False

    async def execute_trade(
        self,
        side: str,
        price: float,
        size_usdc: float,
        market_id: str,
        market_slug: str,
        oracle_lag_signal: float,
        momentum_signal: float,
        liquidation_signal: float,
        combined_signal: float,
        estimated_prob_up: float,
        market_implied_prob: float,
        edge: float,
        time_remaining_secs: float,
        btc_price: float,
        oracle_price: float,
        oracle_open_price: float,
        yes_token_id: str = "",
        no_token_id: str = "",
    ) -> TradeRecord | None:
        """Place a real order on Polymarket.

        Chooses between GTC and FOK based on time remaining.
        Handles order monitoring and cancellation.
        """
        # Determine which token to buy
        token_id = yes_token_id if side == "YES" else no_token_id
        if not token_id:
            logger.error(f"No token ID for side {side}")
            return None

        # Choose order type based on time remaining
        if time_remaining_secs > 60:
            fill_result = await self._execute_gtc(
                token_id, price, size_usdc, time_remaining_secs
            )
        else:
            fill_result = await self._execute_fok(
                token_id, price, size_usdc
            )

        if not fill_result:
            logger.warning(f"[LIVE] Order NOT filled: {side} @ ${price:.4f}")
            return None

        fill_price, num_shares, order_id = fill_result
        actual_cost = fill_price * num_shares

        # Deduct from tracked bankroll
        self.bankroll -= actual_cost

        trade = TradeRecord(
            market_id=market_id, market_slug=market_slug, side=side,
            entry_price=fill_price, size_usdc=actual_cost, num_shares=num_shares,
            oracle_lag_signal=oracle_lag_signal, momentum_signal=momentum_signal,
            liquidation_signal=liquidation_signal, combined_signal=combined_signal,
            estimated_prob_up=estimated_prob_up, market_implied_prob=market_implied_prob,
            edge=edge, time_remaining_secs=time_remaining_secs,
            btc_price_at_entry=btc_price, oracle_price_at_entry=oracle_price,
            oracle_price_at_open=oracle_open_price, is_paper=False,
            order_id=order_id,
        )

        trade.db_id = _log_to_db(self.db, trade, is_paper=False)
        self.pending_trade = trade
        self.pending_order_id = order_id

        logger.info(
            f"[LIVE] {side} @ ${fill_price:.4f} | "
            f"Size: ${actual_cost:.2f} | Shares: {num_shares:.2f} | "
            f"Edge: {edge:+.4f} | Order: {order_id[:16] if order_id else 'N/A'}..."
        )
        return trade

    async def _execute_gtc(
        self, token_id: str, price: float, size_usdc: float,
        time_remaining: float,
    ) -> tuple[float, float, str] | None:
        """Place a GTC limit order and wait for fill.

        Cancel and return None if not filled within gtc_timeout_sec.
        Returns (fill_price, num_shares, order_id) or None.
        """
        num_shares = size_usdc / price

        resp = await self.poly.place_order(
            token_id=token_id,
            side="BUY",
            price=price,
            size=num_shares,
            order_type="GTC",
        )

        if not resp:
            return None

        order_id = resp.get("orderID", resp.get("id", ""))
        if not order_id:
            logger.warning("GTC order placed but no order ID returned")
            return None

        # Monitor for fill
        deadline = time.time() + self.gtc_timeout_sec
        while time.time() < deadline:
            await asyncio.sleep(1.0)

            status = await self.poly.get_order_status(order_id)
            if not status:
                continue

            order_status = status.get("status", "").upper()

            if order_status in ("MATCHED", "FILLED"):
                fill_px = float(status.get("price", price))
                fill_sz = float(status.get("size_matched", num_shares))
                logger.info(f"GTC order filled: {fill_sz:.2f} @ ${fill_px:.4f}")
                return (fill_px, fill_sz, order_id)

            if order_status in ("CANCELED", "CANCELLED", "EXPIRED"):
                logger.info(f"GTC order {order_status}")
                return None

        # Timeout — cancel the unfilled order
        logger.info(f"GTC order not filled in {self.gtc_timeout_sec}s, cancelling")
        await self.poly.cancel_order(order_id)
        return None

    async def _execute_fok(
        self, token_id: str, price: float, size_usdc: float,
    ) -> tuple[float, float, str] | None:
        """Place a FOK (fill-or-kill) order with slight slippage.

        Returns (fill_price, num_shares, order_id) or None.
        """
        # Accept slippage for fill certainty
        fok_price = min(price + self.fok_slippage, 0.99)
        num_shares = size_usdc / fok_price

        resp = await self.poly.place_order(
            token_id=token_id,
            side="BUY",
            price=fok_price,
            size=num_shares,
            order_type="FOK",
        )

        if not resp:
            return None

        order_id = resp.get("orderID", resp.get("id", ""))

        # FOK is instant — check if it was filled
        status_str = resp.get("status", "").upper()
        if status_str in ("MATCHED", "FILLED"):
            fill_px = float(resp.get("price", fok_price))
            fill_sz = float(resp.get("size_matched", num_shares))
            return (fill_px, fill_sz, order_id)

        # If not immediately filled, FOK was rejected
        # Try to check status once more
        await asyncio.sleep(0.5)
        status = await self.poly.get_order_status(order_id)
        if status:
            s = status.get("status", "").upper()
            if s in ("MATCHED", "FILLED"):
                fill_px = float(status.get("price", fok_price))
                fill_sz = float(status.get("size_matched", num_shares))
                return (fill_px, fill_sz, order_id)

        logger.info(f"FOK order not filled at ${fok_price:.4f}")
        return None

    def resolve_pending_trade(self, resolution: str) -> TradeRecord | None:
        """Resolve a pending trade (market settled)."""
        trade = self.pending_trade
        if not trade:
            return None

        won = _resolve_trade(trade, resolution)

        if won:
            self.session_wins += 1
        else:
            self.session_losses += 1

        # For live: actual balance updated on-chain.
        # We track locally for dashboard display.
        self.bankroll += trade.size_usdc + trade.pnl
        self.session_pnl += trade.pnl
        self.session_trades.append(trade)

        if trade.db_id and self.db.conn:
            self.db.conn.execute(
                "UPDATE trades SET resolution = ?, pnl = ? WHERE id = ?",
                (resolution, trade.pnl, trade.db_id),
            )
            self.db.conn.commit()

        self.pending_trade = None
        self.pending_order_id = None

        logger.info(
            f"[LIVE] Resolved {resolution} | {trade.side} | "
            f"{'WIN' if won else 'LOSS'} | P&L: ${trade.pnl:+.2f} | "
            f"Bankroll: ${self.bankroll:.2f}"
        )
        return trade

    async def cancel_pending_order(self) -> bool:
        """Cancel any pending order (used during shutdown)."""
        if self.pending_order_id:
            result = await self.poly.cancel_order(self.pending_order_id)
            if result:
                logger.info(f"Cancelled pending order on shutdown")
            return result
        return False

    async def sync_bankroll(self):
        """Sync bankroll with on-chain USDC balance."""
        balance = await self.poly.get_balance()
        if balance > 0:
            self.bankroll = balance
            logger.info(f"Bankroll synced: ${balance:.2f} USDC")

    @property
    def win_rate(self) -> float:
        total = self.session_wins + self.session_losses
        return self.session_wins / total if total > 0 else 0.0

    @property
    def total_trades(self) -> int:
        return self.session_wins + self.session_losses
