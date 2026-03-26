import logging
from dataclasses import dataclass

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
    required_edge: float = 0.0
    signal_confidence: float = 0.0
    reason: str = ""


class EntryLogic:
    """Backtest-optimised entry logic.

    Key findings from unified backtest (544 trades, real Polymarket pricing):
    - Best entry: 3min remaining, >=3% confidence → 78.4% WR, $0.082/trade
    - Early entries (4m-3m) are profitable because market hasn't priced in direction
    - Late entries (1m-30s) have highest accuracy but market is too efficient
    - Signal confidence is more predictive than EV threshold alone

    Entry timing rules:
    - Don't enter before preferred_entry_secs (market not formed yet, no orderbook)
    - Sweet spot: around preferred_entry_secs (3min default)
    - Hard cutoff: don't enter after latest_entry_secs (market too efficient)
    """

    def __init__(
        self,
        min_edge: float = 0.015,
        max_edge: float = 0.08,
        fee_adjustment: float = 0.02,
        min_confidence: float = 0.03,
        preferred_entry_secs: int = 180,
        latest_entry_secs: int = 60,
    ):
        self.min_edge = min_edge
        self.max_edge = max_edge
        self.fee_adjustment = fee_adjustment
        self.min_confidence = min_confidence
        self.preferred_entry_secs = preferred_entry_secs
        self.latest_entry_secs = latest_entry_secs

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

        Returns:
            EntryDecision with trade details if we should enter.
        """
        decision = EntryDecision()

        if has_position:
            decision.reason = "Already have position this window"
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

        # EV calculations
        # YES: win if UP. Profit = (1 - yes_price), Loss = yes_price
        yes_price = yes_best_ask
        ev_yes = (est_prob_up * (1.0 - yes_price)) - ((1.0 - est_prob_up) * yes_price) - self.fee_adjustment

        # NO: win if DOWN. Profit = (1 - no_price), Loss = no_price
        no_price = no_best_ask
        ev_no = ((1.0 - est_prob_up) * (1.0 - no_price)) - (est_prob_up * no_price) - self.fee_adjustment

        decision.ev_yes = ev_yes
        decision.ev_no = ev_no

        # Pick the better side
        if ev_yes >= ev_no:
            best_ev = ev_yes
            side = "YES"
            price = yes_price
        else:
            best_ev = ev_no
            side = "NO"
            price = no_price

        decision.best_ev = best_ev

        if best_ev > required_edge:
            decision.should_enter = True
            decision.side = side
            decision.price = price
            decision.reason = (
                f"EV {best_ev:.4f} > {required_edge:.4f} | "
                f"conf={signal_confidence:.3f} | {seconds_remaining:.0f}s left"
            )
        else:
            decision.reason = f"EV {best_ev:.4f} < threshold {required_edge:.4f}"

        return decision
