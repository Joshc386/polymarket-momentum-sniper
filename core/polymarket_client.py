"""Polymarket CLOB client wrapper.

Handles authentication, order placement, cancellation, balance queries,
and token allowance management. Wraps py-clob-client with rate limiting
and error handling.
"""

import asyncio
import logging
from enum import Enum

from core.config import Config
from core.rate_limiter import TokenBucketRateLimiter, public_limiter, trading_limiter

logger = logging.getLogger(__name__)

# Polymarket CLOB constants
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon


class OrderType(Enum):
    GTC = "GTC"   # Good-til-cancelled
    FOK = "FOK"   # Fill-or-kill
    GTD = "GTD"   # Good-til-date


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class PolymarketClient:
    """Wrapper around py-clob-client for Polymarket CLOB interaction.

    Handles authentication, API credential derivation, order management,
    and provides a clean interface for both paper and live trading.
    """

    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self._authenticated = False
        self._public_limiter = public_limiter()
        self._trading_limiter = trading_limiter()

    async def connect(self):
        """Initialize and authenticate the CLOB client."""
        if not self.config.polymarket_private_key:
            logger.warning(
                "No POLYMARKET_PRIVATE_KEY set -- running in read-only mode. "
                "Set it in .env for trading."
            )
            self._authenticated = False
            return

        try:
            from py_clob_client.client import ClobClient

            self.client = ClobClient(
                CLOB_HOST,
                key=self.config.polymarket_private_key,
                chain_id=CHAIN_ID,
                signature_type=self.config.polymarket_signature_type,
                funder=self.config.polymarket_funder_address or None,
            )
            creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(creds)
            self._authenticated = True
            logger.info("Polymarket CLOB client authenticated")
        except ImportError:
            logger.error("py-clob-client not installed. Run: pip install py-clob-client")
            self._authenticated = False
        except Exception as e:
            logger.error(f"Polymarket auth failed: {e}")
            self._authenticated = False

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    def get_server_time(self) -> str | None:
        """Get Polymarket server time for clock sync."""
        if not self.client:
            return None
        try:
            return self.client.get_server_time()
        except Exception as e:
            logger.error(f"Failed to get server time: {e}")
            return None

    def get_orderbook(self, token_id: str) -> dict | None:
        """Fetch orderbook for a given token ID."""
        if not self.client:
            return None
        try:
            return self.client.get_order_book(token_id)
        except Exception as e:
            logger.error(f"Failed to get orderbook for {token_id}: {e}")
            return None

    # ── Order Management ──────────────────────────────────────────────

    async def place_order(
        self,
        token_id: str,
        side: str,          # "BUY" or "SELL"
        price: float,
        size: float,         # Number of shares (not USDC)
        order_type: str = "GTC",
    ) -> dict | None:
        """Place an order on the CLOB.

        Args:
            token_id: The token to trade (YES or NO token ID).
            side: "BUY" or "SELL".
            price: Limit price (0.01 - 0.99).
            size: Number of shares.
            order_type: "GTC", "FOK", or "GTD".

        Returns:
            Order response dict with order_id, or None on failure.
        """
        if not self._authenticated or not self.client:
            logger.error("Cannot place order: not authenticated")
            return None

        await self._trading_limiter.acquire()

        try:
            from py_clob_client.order_builder.constants import BUY, SELL

            order_side = BUY if side.upper() == "BUY" else SELL

            # Build the order
            order_args = {
                "token_id": token_id,
                "price": round(price, 2),
                "size": round(size, 2),
                "side": order_side,
            }

            if order_type == "FOK":
                order_args["order_type"] = "FOK"

            signed_order = self.client.create_order(order_args)
            resp = self.client.post_order(signed_order, order_type=order_type)

            if resp and isinstance(resp, dict):
                order_id = resp.get("orderID", resp.get("id", ""))
                logger.info(
                    f"Order placed: {side} {size:.2f} @ ${price:.4f} "
                    f"[{order_type}] ID={order_id[:16]}..."
                )
                return resp
            else:
                logger.warning(f"Order response unexpected: {resp}")
                return resp

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Returns:
            True if successfully cancelled, False otherwise.
        """
        if not self._authenticated or not self.client:
            return False

        await self._trading_limiter.acquire()

        try:
            resp = self.client.cancel(order_id)
            logger.info(f"Order cancelled: {order_id[:16]}...")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id[:16]}: {e}")
            return False

    async def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        if not self._authenticated or not self.client:
            return False

        try:
            resp = self.client.cancel_all()
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel all orders: {e}")
            return False

    async def get_order_status(self, order_id: str) -> dict | None:
        """Check the status of an order."""
        if not self._authenticated or not self.client:
            return None

        await self._public_limiter.acquire()

        try:
            return self.client.get_order(order_id)
        except Exception as e:
            logger.debug(f"Failed to get order status: {e}")
            return None

    async def get_open_orders(self) -> list:
        """Get all open orders."""
        if not self._authenticated or not self.client:
            return []

        await self._public_limiter.acquire()

        try:
            resp = self.client.get_orders()
            if isinstance(resp, list):
                return resp
            return resp.get("orders", []) if isinstance(resp, dict) else []
        except Exception as e:
            logger.debug(f"Failed to get open orders: {e}")
            return []

    # ── Balance & Allowance ───────────────────────────────────────────

    async def get_balance(self) -> float:
        """Get USDC balance on Polygon. Returns balance in USDC."""
        if not self._authenticated or not self.client:
            return 0.0

        await self._public_limiter.acquire()

        try:
            # py-clob-client doesn't have a direct balance method;
            # use the Polygon RPC or check via the API
            balance_resp = self.client.get_balance_allowance(
                asset_type="COLLATERAL"  # USDC
            )
            if isinstance(balance_resp, dict):
                balance = float(balance_resp.get("balance", 0))
                return balance / 1e6  # USDC has 6 decimals
            return 0.0
        except Exception as e:
            logger.debug(f"Failed to get balance: {e}")
            return 0.0

    async def check_and_set_allowance(self) -> bool:
        """Check and set token allowances for trading.

        For EOA wallets, we need to approve USDC.e and CTF contracts.
        For Magic/email wallets (signature_type=1), this is handled automatically.

        Returns:
            True if allowances are set, False if failed.
        """
        if not self._authenticated or not self.client:
            return False

        if self.config.polymarket_signature_type == 1:
            # Magic wallets handle allowances automatically
            logger.debug("Magic wallet — allowances handled automatically")
            return True

        try:
            # For EOA wallets, check and set allowances
            # The py-clob-client handles this via update_balance_allowance
            self.client.update_balance_allowance(
                asset_type="COLLATERAL"
            )
            logger.info("Token allowances set for trading")
            return True
        except Exception as e:
            logger.error(f"Failed to set allowances: {e}")
            return False

    # ── Position Tracking ─────────────────────────────────────────────

    async def get_positions(self) -> list:
        """Get current positions/balances for conditional tokens."""
        if not self._authenticated or not self.client:
            return []

        await self._public_limiter.acquire()

        try:
            resp = self.client.get_positions()
            if isinstance(resp, list):
                return resp
            return []
        except Exception as e:
            logger.debug(f"Failed to get positions: {e}")
            return []
