from dataclasses import dataclass


@dataclass
class OracleLagSignal:
    """Layer 1: how far has BTC moved from the window-open oracle price.

    Positive signal = BTC has moved up since window open (suggests Up resolution).
    Negative signal = BTC has moved down since window open (suggests Down).

    History (2026-05-14):
      Originally this layer blended two components:
        60% on (exchange_price - oracle_NOW) / oracle_NOW  -- the "lag" component
        40% on (exchange_price - oracle_OPEN) / oracle_OPEN -- the "open" component

      The 60% lag component was REMOVED on 2026-05-14 after the signal
      diagnostic logger captured 27,886 ticks showing the lag component
      stuck near +0.21 with std 0.036 (100% positive). Investigation:

        oracle_NOW = (binance + coinbase) / 2
        exchange_price = binance (typically)
        => exchange - oracle_NOW = (binance - coinbase) / 2

      So the "lag" component was measuring half the inter-exchange spread.
      Since Binance was persistently ~$33 above Coinbase during the data
      window, the lag component injected a +0.21 constant offset every
      tick. This shifted est_prob_up systematically above 0.5, breaking
      the entry threshold's symmetric pass/fail behaviour and causing
      Bot G and Bot K to take 100 percent YES trades from May 11 onwards.

      The "open" component (40 percent weight historically) is now used at
      100 percent weight: it measures the actual signal we want, namely
      "how much has BTC moved from the resolution reference price."

      See decisions_BTC J9 for the full investigation.
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
            oracle_price: Current Chainlink oracle price (no longer used
                for the signal; retained in signature for backwards-
                compatible call sites).
            oracle_open_price: Oracle price at start of 5-min window
                (resolution ref). This is the only price the signal
                actually uses now.

        Returns:
            Signal between -1.0 (strong bearish) and 1.0 (strong bullish).
            Returns 0.0 if oracle_open_price is unavailable (signal is
            meaningless without a window-open reference).
        """
        # Reset diagnostic sub-components — written below if computed.
        # `last_lag_component` is kept at 0 because the lag component is no
        # longer used; the field is retained so SignalDiagnosticLogger and
        # other consumers don't break.
        self.last_lag_component = 0.0
        self.last_open_component = 0.0

        if exchange_price <= 0:
            return 0.0

        if oracle_open_price <= 0:
            # No window-open reference -> signal is undefined. Return 0
            # rather than fall back to the (broken) lag component.
            return 0.0

        open_delta = (exchange_price - oracle_open_price) / oracle_open_price
        open_signal = self._clamp(open_delta / self.max_expected_lag)
        self.last_open_component = open_signal
        return open_signal

    @staticmethod
    def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))
