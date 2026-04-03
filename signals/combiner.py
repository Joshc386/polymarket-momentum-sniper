"""Signal combiner — merges all signal layers into a single P(Up) estimate.

5 signal layers:
- L1 Oracle Lag: Exchange vs oracle price divergence
- L2 Momentum: 1-min candle trend analysis
- L3 Liquidation: CoinGlass liquidation level proximity
- L4 Orderbook: Polymarket CLOB depth, imbalance, and order flow
- L5 Sentiment: Coinalyze cross-exchange OI, L/S ratios, funding rates

Coinbase cross-exchange confirmation as confidence modifier.
"""

from dataclasses import dataclass


# Dynamic weight schedule based on time remaining in 5-min window
# (time_remaining_sec, oracle_w, momentum_w, liquidation_w, orderbook_w, sentiment_w)
WEIGHT_SCHEDULE = [
    (300, 0.25, 0.30, 0.15, 0.15, 0.15),  # 5:00-3:00 (early: balanced, sentiment matters)
    (180, 0.20, 0.30, 0.12, 0.23, 0.15),  # 3:00-1:30 (orderbook building)
    (90,  0.15, 0.30, 0.08, 0.32, 0.15),  # 1:30-0:30 (orderbook strongest)
    (30,  0.10, 0.35, 0.05, 0.35, 0.15),  # 0:30-0:05 (orderbook + momentum dominate)
]


@dataclass
class SignalCombiner:
    """Combines 5 signal layers into a single estimated probability of UP.

    Weights shift dynamically over the 5-min window:
    - Early: balanced across all layers, sentiment has steady influence
    - Late: orderbook + momentum dominate as resolution approaches

    Cross-exchange confirmation from Coinbase acts as a confidence
    multiplier — when both exchanges agree on direction, the combined
    signal is amplified; when they disagree, it's dampened.
    """

    max_adjustment: float = 0.20
    confirmation_boost: float = 1.20
    disagreement_dampen: float = 0.70

    def get_weights(
        self, seconds_remaining: float
    ) -> tuple[float, float, float, float, float]:
        """Get (oracle, momentum, liquidation, orderbook, sentiment) weights."""
        for threshold, w1, w2, w3, w4, w5 in WEIGHT_SCHEDULE:
            if seconds_remaining >= threshold:
                return w1, w2, w3, w4, w5
        last = WEIGHT_SCHEDULE[-1]
        return last[1], last[2], last[3], last[4], last[5]

    def combine(
        self,
        oracle_lag_signal: float,
        momentum_signal: float,
        liquidation_signal: float,
        seconds_remaining: float,
        coinbase_direction: float = 0.0,
        orderbook_signal: float = 0.0,
        sentiment_signal: float = 0.0,
        regime_weight_adjustments: dict = None,
    ) -> tuple[float, float]:
        """Combine signals into estimated probability of UP.

        Args:
            oracle_lag_signal: Layer 1 output [-1, 1]
            momentum_signal: Layer 2 output [-1, 1]
            liquidation_signal: Layer 3 output [-1, 1] (0 if unavailable)
            seconds_remaining: Seconds left in the 5-min window
            coinbase_direction: Coinbase price direction [-1, 1] (0 if unavailable)
            orderbook_signal: Layer 4 output [-1, 1] (0 if unavailable)
            sentiment_signal: Layer 5 output [-1, 1] (0 if unavailable)
            regime_weight_adjustments: Dict with keys oracle, momentum,
                liquidation, orderbook, sentiment — additive adjustments.

        Returns:
            (raw_signal, estimated_prob_up)
        """
        w1, w2, w3, w4, w5 = self.get_weights(seconds_remaining)

        # Apply regime-based weight adjustments
        if regime_weight_adjustments:
            w1 += regime_weight_adjustments.get("oracle", 0.0)
            w2 += regime_weight_adjustments.get("momentum", 0.0)
            w3 += regime_weight_adjustments.get("liquidation", 0.0)
            w4 += regime_weight_adjustments.get("orderbook", 0.0)
            w5 += regime_weight_adjustments.get("sentiment", 0.0)
            # Clamp to non-negative and re-normalize to sum to 1.0
            w1 = max(0.0, w1)
            w2 = max(0.0, w2)
            w3 = max(0.0, w3)
            w4 = max(0.0, w4)
            w5 = max(0.0, w5)
            total = w1 + w2 + w3 + w4 + w5
            if total > 0:
                w1 /= total
                w2 /= total
                w3 /= total
                w4 /= total
                w5 /= total

        # Redistribute weights for unavailable signals
        unavailable_weight = 0.0
        signals = [
            (oracle_lag_signal, w1),
            (momentum_signal, w2),
            (liquidation_signal, w3),
            (orderbook_signal, w4),
            (sentiment_signal, w5),
        ]

        active_weights = []
        for sig_val, w in signals:
            if sig_val == 0.0 and w > 0:
                unavailable_weight += w
                active_weights.append(0.0)
            else:
                active_weights.append(w)

        w1, w2, w3, w4, w5 = active_weights

        # Redistribute to available signals proportionally
        if unavailable_weight > 0:
            available_total = w1 + w2 + w3 + w4 + w5
            if available_total > 0:
                scale = (available_total + unavailable_weight) / available_total
                w1 *= scale
                w2 *= scale
                w3 *= scale
                w4 *= scale
                w5 *= scale

        raw_signal = (
            w1 * oracle_lag_signal
            + w2 * momentum_signal
            + w3 * liquidation_signal
            + w4 * orderbook_signal
            + w5 * sentiment_signal
        )

        # Cross-exchange confirmation from Coinbase
        if coinbase_direction != 0.0 and raw_signal != 0.0:
            agreement = raw_signal * coinbase_direction
            if agreement > 0:
                boost = 1.0 + (self.confirmation_boost - 1.0) * min(abs(coinbase_direction), 1.0)
                raw_signal *= boost
            else:
                dampen = 1.0 - (1.0 - self.disagreement_dampen) * min(abs(coinbase_direction), 1.0)
                raw_signal *= dampen

        raw_signal = max(-1.0, min(1.0, raw_signal))

        estimated_prob_up = 0.5 + (raw_signal * self.max_adjustment)
        estimated_prob_up = max(0.05, min(0.95, estimated_prob_up))

        return raw_signal, estimated_prob_up
