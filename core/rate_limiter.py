import asyncio
import time


class TokenBucketRateLimiter:
    """Token bucket rate limiter for async code.

    Tokens refill at a steady rate. Each request consumes one token.
    If no tokens available, callers wait until one is refilled.
    """

    def __init__(self, rate: float, capacity: int | None = None):
        """
        Args:
            rate: Tokens per second (e.g., 100/60 for 100 req/min).
            capacity: Max burst size. Defaults to rate * 1 second worth.
        """
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(int(rate), 1)
        self.tokens = float(self.capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self._last_refill = now

    async def acquire(self):
        """Wait until a token is available, then consume it."""
        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                # Calculate wait time for next token
                wait = (1.0 - self.tokens) / self.rate
            await asyncio.sleep(wait)


# Pre-configured limiters matching Polymarket rate limits
def public_limiter() -> TokenBucketRateLimiter:
    """100 requests/min for public API."""
    return TokenBucketRateLimiter(rate=100 / 60, capacity=10)


def trading_limiter() -> TokenBucketRateLimiter:
    """60 orders/min for trading API."""
    return TokenBucketRateLimiter(rate=60 / 60, capacity=5)
