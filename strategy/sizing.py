import logging

logger = logging.getLogger(__name__)


EXCHANGE_MIN_USDC = 1.0  # Polymarket minimum order


class PositionSizer:
    """Kelly-adjacent position sizing.

    Uses quarter-Kelly to be conservative. Two clamp modes:
    - legacy (default): absolute min/max bounds in USDC.
    - wallet_proportional: floor = max($1 exchange min, floor_pct of the
      sizing wallet), ceiling = ceiling_pct of the sizing wallet, where
      sizing wallet = min(bankroll, wallet_cap_usdc). Above the cap the
      band freezes (e.g. $2-$10 at the default 1%/5%/$200).
    """

    def __init__(
        self,
        kelly_multiplier: float = 0.25,
        min_bet_usdc: float = 1.0,
        max_bet_usdc: float = 5.0,
        wallet_proportional: bool = False,
        floor_pct: float = 0.01,
        ceiling_pct: float = 0.05,
        wallet_cap_usdc: float = 200.0,
    ):
        self.kelly_multiplier = kelly_multiplier
        self.min_bet_usdc = min_bet_usdc
        self.max_bet_usdc = max_bet_usdc
        self.wallet_proportional = wallet_proportional
        self.floor_pct = floor_pct
        self.ceiling_pct = ceiling_pct
        self.wallet_cap_usdc = wallet_cap_usdc

    def compute(
        self,
        est_prob: float,
        share_price: float,
        bankroll: float,
        size_multiplier: float = 1.0,
    ) -> float:
        """Compute bet size in USDC.

        Args:
            est_prob: Estimated probability of winning (for the chosen side).
            share_price: Price per share we'd pay.
            bankroll: Current bankroll in USDC.
            size_multiplier: External multiplier (from risk manager, e.g., 0.5 for streak reduction).

        Returns:
            Bet size in USDC, or 0 if below minimum.
        """
        if share_price <= 0 or share_price >= 1.0 or est_prob <= 0:
            return 0.0

        # Payout odds: if we pay `share_price`, we win `1 - share_price` net
        payout_odds = (1.0 - share_price) / share_price

        # Kelly fraction: (p * b - q) / b where p=prob of win, b=payout odds, q=1-p
        kelly_fraction = (est_prob * payout_odds - (1.0 - est_prob)) / payout_odds

        if kelly_fraction <= 0:
            return 0.0

        adjusted_fraction = kelly_fraction * self.kelly_multiplier * size_multiplier

        if self.wallet_proportional:
            # The whole computation is proportional to the capped sizing
            # wallet (validated in backtest/compounding_sizing_replay.py):
            # above the cap, bets stop growing entirely.
            sizing_wallet = min(bankroll, self.wallet_cap_usdc)
            bet_size = adjusted_fraction * sizing_wallet
            floor = max(EXCHANGE_MIN_USDC, self.floor_pct * sizing_wallet)
            ceiling = self.ceiling_pct * sizing_wallet
        else:
            bet_size = adjusted_fraction * bankroll
            floor = self.min_bet_usdc
            ceiling = self.max_bet_usdc
        bet_size = max(floor, min(ceiling, bet_size))

        # Don't bet more than bankroll allows
        if bet_size > bankroll:
            bet_size = bankroll

        # Below minimum after all adjustments
        if bet_size < floor:
            return 0.0

        return round(bet_size, 2)
