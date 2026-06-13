import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


EXCHANGE_MIN_USDC = 1.0  # Polymarket minimum order notional (backstop)


@dataclass
class SizingDecision:
    """Outcome of a sizing call. ``compute`` returns ``.size``; callers that
    need to tag trades (bumped/skipped) use ``decide`` and read the flags."""
    size: float        # final bet USDC (0.0 => no trade)
    raw: float         # pre-clamp quarter-Kelly USDC
    floor: float       # minimum risk for this trade
    ceiling: float     # maximum risk for this trade
    bumped: bool = False   # raw was below the floor -> lifted up to it
    skipped: bool = False  # no tradeable size (floor > ceiling, or sub-floor)


class PositionSizer:
    """Kelly-adjacent position sizing (quarter-Kelly by default).

    Two modes:
    - legacy (default): absolute min/max bounds in USDC.
    - wallet_proportional (revised 2026-06-12):
        * ceiling = ceiling_pct (4%) of the sizing wallet = min(bankroll,
          wallet_cap_usdc=$200). $4 @ $100, $8 @ $200, flat $8 above.
        * floor below the cap = the dynamic exchange minimum
          max(min_order_shares x price, $1) — price-driven, not wallet-
          driven; Kelly results below it are BUMPED up and the trade is
          tagged. At/above the cap the floor is the flat floor_at_cap_usdc
          ($5).
        * if floor > ceiling (high price on a small wallet) the trade is
          SKIPPED — never breach the max-risk cap to take a low-edge trade.
    """

    def __init__(
        self,
        kelly_multiplier: float = 0.25,
        min_bet_usdc: float = 1.0,
        max_bet_usdc: float = 5.0,
        wallet_proportional: bool = False,
        floor_pct: float = 0.01,        # legacy/deprecated (unused in revised scheme)
        ceiling_pct: float = 0.04,
        wallet_cap_usdc: float = 200.0,
        min_order_shares: float = 5.0,
        floor_at_cap_usdc: float = 5.0,
    ):
        self.kelly_multiplier = kelly_multiplier
        self.min_bet_usdc = min_bet_usdc
        self.max_bet_usdc = max_bet_usdc
        self.wallet_proportional = wallet_proportional
        self.floor_pct = floor_pct
        self.ceiling_pct = ceiling_pct
        self.wallet_cap_usdc = wallet_cap_usdc
        self.min_order_shares = min_order_shares
        self.floor_at_cap_usdc = floor_at_cap_usdc

    def compute(
        self,
        est_prob: float,
        share_price: float,
        bankroll: float,
        size_multiplier: float = 1.0,
    ) -> float:
        """Bet size in USDC (0.0 if no trade). Thin wrapper over decide()."""
        return self.decide(est_prob, share_price, bankroll, size_multiplier).size

    def decide(
        self,
        est_prob: float,
        share_price: float,
        bankroll: float,
        size_multiplier: float = 1.0,
    ) -> SizingDecision:
        """Full sizing decision with bump/skip flags for trade tagging.

        Args:
            est_prob: Estimated probability of winning (for the chosen side).
            share_price: Price per share we'd pay.
            bankroll: Current bankroll in USDC.
            size_multiplier: External multiplier (risk manager, e.g. 0.5).
        """
        if share_price <= 0 or share_price >= 1.0 or est_prob <= 0:
            return SizingDecision(0.0, 0.0, 0.0, 0.0, skipped=True)

        # Payout odds: if we pay `share_price`, we win `1 - share_price` net.
        payout_odds = (1.0 - share_price) / share_price
        # Kelly fraction: (p*b - q)/b, p=prob win, b=payout odds, q=1-p.
        kelly_fraction = (est_prob * payout_odds - (1.0 - est_prob)) / payout_odds
        if kelly_fraction <= 0:
            return SizingDecision(0.0, 0.0, 0.0, 0.0, skipped=True)

        adjusted_fraction = kelly_fraction * self.kelly_multiplier * size_multiplier

        if self.wallet_proportional:
            # Bets are proportional to the capped sizing wallet; above the
            # cap they stop growing entirely.
            sizing_wallet = min(bankroll, self.wallet_cap_usdc)
            raw = adjusted_fraction * sizing_wallet
            ceiling = self.ceiling_pct * sizing_wallet
            if bankroll >= self.wallet_cap_usdc:
                floor = self.floor_at_cap_usdc          # flat $5 at/above cap
            else:
                # Dynamic exchange minimum: 5 shares x price, $1 notional floor.
                floor = max(self.min_order_shares * share_price, EXCHANGE_MIN_USDC)
        else:
            raw = adjusted_fraction * bankroll
            floor = self.min_bet_usdc
            ceiling = self.max_bet_usdc

        # The exchange floor cannot fit under the risk cap — skip rather than
        # breach the max (only at high prices on a small wallet).
        if floor > ceiling:
            return SizingDecision(0.0, raw, floor, ceiling, skipped=True)

        bumped = raw < floor
        bet_size = max(floor, min(ceiling, raw))

        # Never bet more than the bankroll; if that drops below the floor the
        # wallet is too small for a compliant order — skip.
        if bet_size > bankroll:
            bet_size = bankroll
        if bet_size < floor:
            return SizingDecision(0.0, raw, floor, ceiling, skipped=True)

        return SizingDecision(round(bet_size, 2), raw, floor, ceiling, bumped=bumped)
