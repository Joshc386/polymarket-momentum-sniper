from dataclasses import dataclass


@dataclass
class OracleLagSignal:
    """Layer 1: Compares real-time exchange price vs Chainlink oracle price.

    Positive signal = exchange price ahead of oracle (suggests Up).
    Negative signal = exchange price behind oracle (suggests Down).

    Also compares exchange price vs window opening oracle price, since
    that is the actual resolution comparison.
    """

    max_expected_lag: float = 0.001  # 0.1%

    def compute(
        self,
        exchange_price: float,
        oracle_price: float,
        oracle_open_price: float,
    ) -> float:
        """Compute oracle lag signal in [-1.0, 1.0].

        Args:
            exchange_price: Current real-time BTC price from Binance.
            oracle_price: Current Chainlink oracle price.
            oracle_open_price: Oracle price at start of 5-min window (resolution ref).

        Returns:
            Signal between -1.0 (strong bearish) and 1.0 (strong bullish).
        """
        # Reset diagnostic sub-components — written below if computed.
        self.last_lag_component = 0.0
        self.last_open_component = 0.0

        if oracle_price <= 0 or exchange_price <= 0:
            return 0.0

        # Primary: exchange vs current oracle (detects where price is heading)
        lag_delta = (exchange_price - oracle_price) / oracle_price
        lag_signal = self._clamp(lag_delta / self.max_expected_lag)
        self.last_lag_component = lag_signal

        # Secondary: exchange vs window open oracle (actual resolution comparison)
        if oracle_open_price > 0:
            open_delta = (exchange_price - oracle_open_price) / oracle_open_price
            open_signal = self._clamp(open_delta / self.max_expected_lag)
            self.last_open_component = open_signal
            # Blend: 60% lag signal, 40% open comparison
            return self._clamp(0.6 * lag_signal + 0.4 * open_signal)

        return lag_signal

    @staticmethod
    def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))
