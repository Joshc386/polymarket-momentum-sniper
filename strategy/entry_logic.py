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
    reason: str = ""


class EntryLogic:
    """Dynamic entry threshold — continuously evaluates whether to enter.

    Edge required decreases as time remaining in the window decreases:
    early in the window we demand more edge, late we accept less.
    """

    def __init__(
        self,
        min_edge: float = 0.02,
        max_edge: float = 0.10,
        fee_adjustment: float = 0.01,
    ):
        self.min_edge = min_edge
        self.max_edge = max_edge
        self.fee_adjustment = fee_adjustment

    def evaluate(
        self,
        est_prob_up: float,
        yes_best_ask: float,
        no_best_ask: float,
        yes_best_bid: float,
        no_best_bid: float,
        seconds_remaining: float,
        has_position: bool,
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

        Returns:
            EntryDecision with trade details if we should enter.
        """
        decision = EntryDecision()

        if has_position:
            decision.reason = "Already have position this window"
            return decision

        if seconds_remaining < 5:
            decision.reason = "Too close to expiry (<5s)"
            return decision

        if yes_best_ask <= 0 or no_best_ask <= 0:
            decision.reason = "No orderbook data"
            return decision

        # Dynamic threshold: higher edge required early, lower late
        time_remaining_pct = seconds_remaining / 300.0
        required_edge = self.min_edge + (self.max_edge - self.min_edge) * time_remaining_pct
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
            decision.reason = f"EV {best_ev:.4f} > threshold {required_edge:.4f}"
        else:
            decision.reason = f"EV {best_ev:.4f} < threshold {required_edge:.4f}"

        return decision
