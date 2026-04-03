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

        # EV calculations
        # Polymarket fee is charged on WINNINGS only (not on losses).
        # fee_adjustment is a rate (0.02 = 2%), applied to profit if we win.
        yes_price = yes_best_ask
        profit_yes = 1.0 - yes_price
        ev_yes = (est_prob_up * (profit_yes * (1.0 - self.fee_adjustment))) - ((1.0 - est_prob_up) * yes_price)

        no_price = no_best_ask
        profit_no = 1.0 - no_price
        ev_no = ((1.0 - est_prob_up) * (profit_no * (1.0 - self.fee_adjustment))) - (est_prob_up * no_price)

        decision.ev_yes = ev_yes
        decision.ev_no = ev_no

        # Pick whichever side has better EV — this buys the cheap contrarian
        # side when the market has overreacted.  Early paper trading showed
        # 39% WR but +315% ROI with this approach because wins on cheap tokens
        # pay $0.85 while losses cost $0.15.  The corrected fee formula (rate
        # on winnings only, not flat penalty) now gives accurate EV for both sides.
        if ev_yes >= ev_no:
            side = "YES"
            best_ev = ev_yes
            price = yes_price
        else:
            side = "NO"
            best_ev = ev_no
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
