import logging
from dataclasses import dataclass

from strategy.feature_snapshot import fee_per_share

logger = logging.getLogger(__name__)


@dataclass
class EntryDecision:
    """Result of the entry logic evaluation."""
    should_enter: bool = False
    side: str = ""          # "YES" or "NO"
    price: float = 0.0      # Price we'd pay for the share
    ev_yes: float = 0.0
    ev_no: float = 0.0
    best_ev: float = 0.0
    prob_edge: float = 0.0  # |model_prob - market_prob| — the real edge metric
    required_edge: float = 0.0
    signal_confidence: float = 0.0
    reason: str = ""


class EntryLogic:
    """Dynamic entry logic — enters at first strong signal within the window.

    Key findings from entry timing backtest (8,256 trades, 30 days):
    - "First signal above threshold" strategy: 67.5% WR, $1,045 total PnL
    - Beats fixed 4m entry ($755) by capturing 2,573 extra trades
    - Scanning 4:30→1:00 catches windows where signal is weak early but
      strengthens as more price data arrives
    - Earlier entry = better prices (market less efficient)
    - Variable timing prevents market makers from front-running a fixed pattern

    Entry window:
    - Opens at earliest_entry_secs (4:30 default) — start scanning
    - Enters as soon as confidence + edge thresholds are met
    - Closes at latest_entry_secs (1:00 default) — stop trying
    - One trade per window maximum
    """

    def __init__(
        self,
        min_edge: float = 0.015,
        max_edge: float = 0.08,
        fee_adjustment: float = 0.02,
        min_confidence: float = 0.02,
        preferred_entry_secs: int = 180,
        latest_entry_secs: int = 60,
        earliest_entry_secs: int = 270,
    ):
        self.min_edge = min_edge
        self.max_edge = max_edge
        # DEPRECATED (2026-06-03): the gate now uses the real per-share fee
        # fee_per_share(p) = 0.07*p*(1-p) in the EV calc, not this flat 2%
        # haircut. Kept only so existing config/callers don't break.
        self.fee_adjustment = fee_adjustment
        self.min_confidence = min_confidence
        self.preferred_entry_secs = preferred_entry_secs
        self.latest_entry_secs = latest_entry_secs
        self.earliest_entry_secs = earliest_entry_secs

    def evaluate(
        self,
        est_prob_up: float,
        yes_best_ask: float,
        no_best_ask: float,
        yes_best_bid: float,
        no_best_bid: float,
        seconds_remaining: float,
        has_position: bool,
        regime_edge_multiplier: float = 1.0,
        signal_aligned: bool = False,
    ) -> EntryDecision:
        """Evaluate whether to enter a position.

        Args:
            est_prob_up: Model's estimated probability of UP [0, 1].
            yes_best_ask: Best ask price for YES token.
            no_best_ask: Best ask price for NO token.
            yes_best_bid: Best bid price for YES token.
            no_best_bid: Best bid price for NO token.
            seconds_remaining: Seconds left in the 5-min window.
            has_position: Whether we already have a position this window.
            regime_edge_multiplier: Multiplier on required edge from regime
                detector (>1.0 = need more edge, <1.0 = accept less).
            signal_aligned: When True, pick side by signal direction
                (est_prob_up > 0.5 -> YES) instead of fading the market.
                Use in ranging regimes where the contrarian logic inverts
                the signal and destroys edge.

        Returns:
            EntryDecision with trade details if we should enter.
        """
        decision = EntryDecision()

        if has_position:
            decision.reason = "Already have position this window"
            return decision

        # Hard cutoff: too early — window hasn't opened yet
        if seconds_remaining > self.earliest_entry_secs:
            decision.reason = f"Window not open ({seconds_remaining:.0f}s > {self.earliest_entry_secs}s)"
            return decision

        # Hard cutoff: too close to expiry — market is fully efficient
        if seconds_remaining < self.latest_entry_secs:
            decision.reason = f"Too late ({seconds_remaining:.0f}s < {self.latest_entry_secs}s cutoff)"
            return decision

        if yes_best_ask <= 0 or no_best_ask <= 0:
            decision.reason = "No orderbook data"
            return decision

        # Signal confidence check (most important filter from backtest)
        signal_confidence = abs(est_prob_up - 0.5)
        decision.signal_confidence = signal_confidence

        if signal_confidence < self.min_confidence:
            decision.reason = f"Low confidence ({signal_confidence:.3f} < {self.min_confidence})"
            return decision

        # Dynamic threshold: higher edge required early, lower late
        # But now capped by the entry window (preferred_entry_secs to latest_entry_secs)
        time_remaining_pct = seconds_remaining / 300.0
        required_edge = self.min_edge + (self.max_edge - self.min_edge) * time_remaining_pct
        # Regime adjustment — trending reduces threshold, choppy/volatile increases it
        required_edge *= regime_edge_multiplier
        decision.required_edge = required_edge

        # ── Market-implied probability from orderbook midpoint ──
        # The midpoint is a better estimate than the ask alone because
        # the ask includes the spread.
        market_mid_yes = (yes_best_bid + yes_best_ask) / 2.0
        market_prob_up = max(0.05, min(0.95, market_mid_yes))

        # ── Probability edge: how much our model disagrees with the market ──
        # This is the TRUE edge metric. Raw dollar-EV is misleading because
        # cheap tokens ($0.25) produce huge EV even with negligible conviction.
        # Probability edge normalises across all price levels.
        prob_edge_yes = est_prob_up - market_prob_up         # >0 -> model is more bullish
        prob_edge_no = (1.0 - est_prob_up) - (1.0 - market_prob_up)  # = -prob_edge_yes

        # Pick the side to trade.
        # Default (contrarian): disagree with the market the most — fade
        #   overbought/oversold odds. Works in trending regimes.
        # Signal-aligned: follow the model's directional call — if est_prob_up
        #   > 0.5, buy YES. Fixes the ranging regime problem where contrarian
        #   logic inverts a correct signal (55.9% accurate -> 43.5% side hit).
        if signal_aligned:
            # Follow the signal direction, not the market disagreement
            if est_prob_up >= 0.5:
                side = "YES"
                price = yes_best_ask
            else:
                side = "NO"
                price = no_best_ask
            # Edge is still the magnitude of model-market disagreement
            # (used for threshold gating, not side selection)
            prob_edge = abs(prob_edge_yes)
        else:
            # Contrarian: pick the side where we disagree with market most
            if prob_edge_yes >= 0:
                side = "YES"
                prob_edge = prob_edge_yes
                price = yes_best_ask
            else:
                side = "NO"
                prob_edge = abs(prob_edge_no)
                price = no_best_ask

        decision.prob_edge = prob_edge

        # Per-share EV with the REAL Polymarket taker fee: 0.07*p*(1-p),
        # charged at entry on every trade. EV = (q - p) - fee_per_share(p),
        # the same formula as feature_snapshot.net_ev_per_share. This is the
        # `best_ev > 0` entry gate below, so it must reflect true fees — the
        # old 2%-of-profit-on-winners haircut (self.fee_adjustment) let
        # marginal near-50/50 trades through that lose money at live fees.
        yes_price = yes_best_ask
        ev_yes = (est_prob_up - yes_price) - fee_per_share(yes_price)

        no_price = no_best_ask
        ev_no = ((1.0 - est_prob_up) - no_price) - fee_per_share(no_price)

        decision.ev_yes = ev_yes
        decision.ev_no = ev_no
        best_ev = ev_yes if side == "YES" else ev_no
        decision.best_ev = best_ev

        # ── Entry gate: probability edge must exceed dynamic threshold ──
        # required_edge is now in probability units (e.g. 0.04 = 4% disagreement)
        if prob_edge > required_edge and best_ev > 0:
            decision.should_enter = True
            decision.side = side
            decision.price = price
            decision.reason = (
                f"Edge {prob_edge:.3f} > {required_edge:.3f} "
                f"(EV ${best_ev:.4f}) | "
                f"conf={signal_confidence:.3f} | {seconds_remaining:.0f}s left"
            )
        else:
            if best_ev <= 0:
                decision.reason = (
                    f"Negative EV ({best_ev:.4f}) | "
                    f"edge={prob_edge:.3f}"
                )
            else:
                decision.reason = (
                    f"Edge {prob_edge:.3f} < threshold {required_edge:.3f}"
                )

        return decision
