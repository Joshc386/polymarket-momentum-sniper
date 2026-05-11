"""Signal combiner — merges all signal layers into a single P(Up) estimate.

5 core signal layers:
- L1 Oracle Lag: Exchange vs oracle price divergence
- L2 Momentum: 1-min candle trend analysis
- L3 Liquidation: CoinGlass liquidation level proximity
- L4 Orderbook: Polymarket CLOB depth, imbalance, and order flow
- L5 Sentiment: Coinalyze cross-exchange OI, L/S ratios, funding rates

Additive modifiers (applied after core blend, weights configurable):
- L6 Orderbook Fade: Contrarian signal on extreme CLOB imbalances (default 10%)
- L7 Taker Ratio: Binance taker buy/sell volume pressure (default 8%)
- L8 CLOB Trade Flow: Polymarket executed trade direction (default 12%)

Coinbase cross-exchange confirmation as confidence modifier.
"""

from dataclasses import dataclass


# Dynamic weight schedule based on time remaining in 5-min window
# (time_remaining_sec, oracle_w, momentum_w, liquidation_w, orderbook_w, sentiment_w)
WEIGHT_SCHEDULE_DEFAULT = [
    (300, 0.25, 0.30, 0.15, 0.15, 0.15),  # 5:00-3:00 (early: balanced, sentiment matters)
    (180, 0.20, 0.30, 0.12, 0.23, 0.15),  # 3:00-1:30 (orderbook building)
    (90,  0.15, 0.30, 0.08, 0.32, 0.15),  # 1:30-0:30 (orderbook strongest)
    (30,  0.10, 0.35, 0.05, 0.35, 0.15),  # 0:30-0:05 (orderbook + momentum dominate)
]

# KF-dominant schedule — L1 (Kalman Filter) is the primary driver.
# Used by Bot B where the KF's multi-source fusion and velocity tracking
# should lead decisions, not just contribute 10-25% to a blend.
WEIGHT_SCHEDULE_KF_DOMINANT = [
    (300, 0.45, 0.20, 0.10, 0.10, 0.15),  # Early: KF leads, sentiment steady
    (180, 0.45, 0.20, 0.08, 0.15, 0.12),  # Mid: KF leads, orderbook growing
    (90,  0.40, 0.20, 0.05, 0.25, 0.10),  # Late: KF + orderbook
    (30,  0.35, 0.25, 0.05, 0.25, 0.10),  # Final: KF + orderbook + momentum
]

# Ranging regime schedule — orderflow-dominant.
# In ranging markets, L1 (oracle lag) and L5 (sentiment) are noise.
# L4 (orderbook) is 63% accurate and L2 (momentum) is 60% accurate.
# Suppress the noise layers and let flow dominate the decision.
WEIGHT_SCHEDULE_RANGING = [
    (300, 0.05, 0.30, 0.05, 0.55, 0.05),  # Early: book dominates, momentum secondary
    (180, 0.05, 0.25, 0.05, 0.60, 0.05),  # Mid: book strongest as depth builds
    (90,  0.05, 0.25, 0.05, 0.60, 0.05),  # Late: same — ranging doesn't shift over time
    (30,  0.05, 0.30, 0.05, 0.55, 0.05),  # Final: slight momentum boost for late moves
]

# Named schedules for config lookup
WEIGHT_SCHEDULES = {
    "default": WEIGHT_SCHEDULE_DEFAULT,
    "kf_dominant": WEIGHT_SCHEDULE_KF_DOMINANT,
    "ranging": WEIGHT_SCHEDULE_RANGING,
}


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
    weight_schedule_name: str = "default"

    # Additive modifier weights for L6/L7/L8/L9b
    # These are applied after the core L1-L5 blend.
    fade_weight: float = 0.10       # L6: orderbook fade (contrarian on extreme imbalance)
    taker_ratio_weight: float = 0.08  # L7: Binance taker buy/sell ratio
    clob_flow_weight: float = 0.12  # L8: Polymarket executed trade flow
    absorption_weight: float = 0.0  # L9b: absorption detection (disabled by default)
    exhaustion_weight: float = 0.0  # L10: level exhaustion (disabled by default)
    trade_size_weight: float = 0.0  # L11: trade size conviction (disabled by default)
    wallet_flow_weight: float = 0.0  # L12: on-chain wallet flow (disabled by default)

    def __post_init__(self) -> None:
        self._schedule = WEIGHT_SCHEDULES.get(
            self.weight_schedule_name, WEIGHT_SCHEDULE_DEFAULT
        )

    def get_weights(
        self,
        seconds_remaining: float,
        schedule_override: str = "",
    ) -> tuple[float, float, float, float, float]:
        """Get (oracle, momentum, liquidation, orderbook, sentiment) weights.

        Args:
            seconds_remaining: Seconds left in the 5-min window.
            schedule_override: If set, use this named schedule instead of
                the instance default. Used for regime-specific weighting
                (e.g. 'ranging' schedule in ranging markets).
        """
        schedule = self._schedule
        if schedule_override:
            schedule = WEIGHT_SCHEDULES.get(schedule_override, self._schedule)
        for threshold, w1, w2, w3, w4, w5 in schedule:
            if seconds_remaining >= threshold:
                return w1, w2, w3, w4, w5
        last = schedule[-1]
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
        flip_signal: bool = False,
        fade_signal: float = 0.0,
        taker_ratio_signal: float = 0.0,
        clob_flow_signal: float = 0.0,
        absorption_signal: float = 0.0,
        exhaustion_signal: float = 0.0,
        trade_size_signal: float = 0.0,
        wallet_flow_signal: float = 0.0,
        schedule_override: str = "",
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
            flip_signal: If True, negate the final raw signal before
                converting to probability. Used for weekend mode where
                the contrarian strategy inverts.
            fade_signal: Layer 6 orderbook fade [-1, 1] (0 if not triggered).
                Applied as additive modifier after core blend.
            taker_ratio_signal: Layer 7 taker buy ratio [-1, 1] (0 if unavailable).
                Applied as additive modifier after core blend.
            clob_flow_signal: Layer 8 CLOB trade flow [-1, 1] (0 if unavailable).
                Real Polymarket trade volume direction. Applied as additive
                modifier after core blend.
            absorption_signal: Layer 9b absorption [-1, 1] (0 if unavailable).
                Hidden liquidity detection from depth resilience vs trade
                flow pressure. Applied as additive modifier after core blend.
            exhaustion_signal: Layer 10 level exhaustion [-1, 1] (0 if
                unavailable). Gap + wall depletion detection. Applied as
                additive modifier after core blend.
            trade_size_signal: Layer 11 trade size conviction [-1, 1] (0
                if unavailable). Conviction asymmetry from trade size
                distribution. Applied as additive modifier after core blend.
            wallet_flow_signal: Layer 12 on-chain wallet flow [-1, 1] (0
                if unavailable). Wallet-level conviction from Polygon
                blockchain data. Applied as additive modifier after core.
            schedule_override: Named weight schedule to use instead of
                default. Pass 'ranging' for orderflow-dominant weighting.

        Returns:
            (raw_signal, estimated_prob_up)
        """
        w1, w2, w3, w4, w5 = self.get_weights(seconds_remaining, schedule_override)

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

        # L6: Orderbook fade — additive modifier (sparse but high-conviction)
        if fade_signal != 0.0 and self.fade_weight > 0:
            raw_signal += self.fade_weight * fade_signal

        # L7: Taker ratio — additive modifier (Binance trade flow)
        if taker_ratio_signal != 0.0 and self.taker_ratio_weight > 0:
            raw_signal += self.taker_ratio_weight * taker_ratio_signal

        # L8: CLOB trade flow — additive modifier (Polymarket executed trades)
        if clob_flow_signal != 0.0 and self.clob_flow_weight > 0:
            raw_signal += self.clob_flow_weight * clob_flow_signal

        # L9b: Absorption — hidden liquidity detection
        if absorption_signal != 0.0 and self.absorption_weight > 0:
            raw_signal += self.absorption_weight * absorption_signal

        # L10: Level exhaustion — gap + wall depletion
        if exhaustion_signal != 0.0 and self.exhaustion_weight > 0:
            raw_signal += self.exhaustion_weight * exhaustion_signal

        # L11: Trade size conviction — size distribution asymmetry
        if trade_size_signal != 0.0 and self.trade_size_weight > 0:
            raw_signal += self.trade_size_weight * trade_size_signal

        # L12: On-chain wallet flow — wallet-level conviction
        if wallet_flow_signal != 0.0 and self.wallet_flow_weight > 0:
            raw_signal += self.wallet_flow_weight * wallet_flow_signal

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

        # Weekend flip: negate the signal so the bot takes the opposite
        # side. Weekend markets are more efficient — the cheap side is
        # cheap for a reason, so we follow the market instead of fading it.
        if flip_signal:
            raw_signal = -raw_signal

        estimated_prob_up = 0.5 + (raw_signal * self.max_adjustment)
        estimated_prob_up = max(0.05, min(0.95, estimated_prob_up))

        return raw_signal, estimated_prob_up
