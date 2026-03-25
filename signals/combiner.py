"""Signal combiner — merges all signal layers into a single P(Up) estimate.

Phase 8 enhancements:
- Layer 4: Polymarket orderbook signal (depth, imbalance, flow)
- Coinbase cross-exchange confirmation as confidence modifier
- Configurable weight schedule with 4 signal layers
"""

from dataclasses import dataclass


# Dynamic weight schedule based on time remaining in 5-min window
# (time_remaining_min, oracle_w, momentum_w, liquidation_w, orderbook_w)
WEIGHT_SCHEDULE = [
    (300, 0.30, 0.35, 0.20, 0.15),   # 5:00 - 3:00  (early: oracle + momentum lead)
    (180, 0.25, 0.35, 0.15, 0.25),   # 3:00 - 1:30  (orderbook builds signal)
    (90,  0.20, 0.35, 0.10, 0.35),   # 1:30 - 0:30  (orderbook strongest here)
    (30,  0.15, 0.40, 0.05, 0.40),   # 0:30 - 0:05  (orderbook + momentum dominate)
]


@dataclass
class SignalCombiner:
    """Combines signal layers into a single estimated probability of UP.

    4 signal layers:
    - L1 Oracle Lag: Exchange vs oracle price divergence
    - L2 Momentum: 1-min candle trend analysis
    - L3 Liquidation: CoinGlass liquidation level proximity
    - L4 Orderbook: Polymarket CLOB depth, imbalance, and order flow

    Weights shift dynamically over the 5-min window:
    - Early: more weight on oracle lag + momentum
    - Late: more weight on orderbook + momentum (book activity picks up)

    Cross-exchange confirmation from Coinbase acts as a confidence
    multiplier — when both exchanges agree on direction, the combined
    signal is amplified; when they disagree, it's dampened.
    """

    max_adjustment: float = 0.15
    # Coinbase cross-confirmation parameters
    confirmation_boost: float = 1.20   # Multiply signal by this when confirmed
    disagreement_dampen: float = 0.70  # Multiply signal by this when disagreement

    def get_weights(self, seconds_remaining: float) -> tuple[float, float, float, float]:
        """Get (oracle, momentum, liquidation, orderbook) weights for current time."""
        for threshold, w1, w2, w3, w4 in WEIGHT_SCHEDULE:
            if seconds_remaining >= threshold:
                return w1, w2, w3, w4
        # Under 30s: use last row
        return WEIGHT_SCHEDULE[-1][1], WEIGHT_SCHEDULE[-1][2], WEIGHT_SCHEDULE[-1][3], WEIGHT_SCHEDULE[-1][4]

    def combine(
        self,
        oracle_lag_signal: float,
        momentum_signal: float,
        liquidation_signal: float,
        seconds_remaining: float,
        coinbase_direction: float = 0.0,
        orderbook_signal: float = 0.0,
    ) -> tuple[float, float]:
        """Combine signals into estimated probability of UP.

        Args:
            oracle_lag_signal: Layer 1 output [-1, 1]
            momentum_signal: Layer 2 output [-1, 1]
            liquidation_signal: Layer 3 output [-1, 1] (0 if unavailable)
            seconds_remaining: Seconds left in the 5-min window
            coinbase_direction: Coinbase price direction [-1, 1] (0 if unavailable)
            orderbook_signal: Layer 4 output [-1, 1] (0 if unavailable)

        Returns:
            (raw_signal, estimated_prob_up)
        """
        w1, w2, w3, w4 = self.get_weights(seconds_remaining)

        # Redistribute weights for unavailable signals
        unavailable_weight = 0.0

        if liquidation_signal == 0.0 and w3 > 0:
            unavailable_weight += w3
            w3 = 0.0

        if orderbook_signal == 0.0 and w4 > 0:
            unavailable_weight += w4
            w4 = 0.0

        # Redistribute to available signals proportionally
        if unavailable_weight > 0:
            available_total = w1 + w2 + w3 + w4
            if available_total > 0:
                scale = (available_total + unavailable_weight) / available_total
                w1 *= scale
                w2 *= scale
                w3 *= scale
                w4 *= scale

        raw_signal = (
            w1 * oracle_lag_signal
            + w2 * momentum_signal
            + w3 * liquidation_signal
            + w4 * orderbook_signal
        )

        # Cross-exchange confirmation from Coinbase
        if coinbase_direction != 0.0 and raw_signal != 0.0:
            # Same sign = confirmation, opposite sign = disagreement
            agreement = raw_signal * coinbase_direction  # Positive = agree
            if agreement > 0:
                # Both point same way — boost confidence
                # Scale boost by strength of Coinbase signal
                boost = 1.0 + (self.confirmation_boost - 1.0) * min(abs(coinbase_direction), 1.0)
                raw_signal *= boost
            else:
                # Disagreement — dampen
                dampen = 1.0 - (1.0 - self.disagreement_dampen) * min(abs(coinbase_direction), 1.0)
                raw_signal *= dampen

        # Clamp raw signal
        raw_signal = max(-1.0, min(1.0, raw_signal))

        estimated_prob_up = 0.5 + (raw_signal * self.max_adjustment)
        estimated_prob_up = max(0.05, min(0.95, estimated_prob_up))

        return raw_signal, estimated_prob_up
