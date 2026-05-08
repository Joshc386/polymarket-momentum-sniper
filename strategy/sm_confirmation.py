"""SM Confirmation Logic — L9 signal layer for Bot K.

Pure decision function: given position side, market price, and aggregate
SM wallet flow, return HOLD / EXIT / IGNORE. No side effects.

Decision logic (from Dune Q21-Q23, backtest-validated on 24,278 windows):
- Price < $0.65: IGNORE (SM accuracy = coin flip in this zone)
- Price > $0.80: IGNORE (market already decided, SM adds nothing)
- SM flow agrees with position (60%+ dollars on same side): HOLD
- SM flow disagrees with position: EXIT (saves losses 75.2% of the time)
- Insufficient SM volume: IGNORE (not enough data to act on)
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SMDecision(Enum):
    """Possible outcomes from the SM confirmation check."""

    HOLD = "HOLD"
    EXIT = "EXIT"
    IGNORE = "IGNORE"


class PositionSide(Enum):
    """Side of the current position."""

    YES = "YES"
    NO = "NO"


@dataclass(frozen=True)
class SMFlowState:
    """Aggregate SM wallet flow for the current window.

    Args:
        yes_volume: Total SM dollar volume on YES side.
        no_volume: Total SM dollar volume on NO side.
        num_wallets: Number of distinct SM wallets that traded.
    """

    yes_volume: float
    no_volume: float
    num_wallets: int

    @property
    def total_volume(self) -> float:
        return self.yes_volume + self.no_volume

    @property
    def yes_fraction(self) -> float:
        if self.total_volume == 0:
            return 0.0
        return self.yes_volume / self.total_volume

    @property
    def no_fraction(self) -> float:
        if self.total_volume == 0:
            return 0.0
        return self.no_volume / self.total_volume


@dataclass(frozen=True)
class SMConfirmationConfig:
    """Configuration for SM confirmation logic.

    Args:
        price_floor: Below this price, SM signal is ignored. Default $0.65.
        price_ceiling: Above this price, SM signal is ignored. Default $0.80.
        agreement_threshold: Fraction of SM dollars on one side to count
            as agreement/disagreement. Default 0.6 (60%).
        min_sm_volume: Minimum total SM dollar volume to act on. Default $100.
        min_sm_wallets: Minimum distinct SM wallets to act on. Default 2.
    """

    price_floor: float = 0.65
    price_ceiling: float = 0.80
    agreement_threshold: float = 0.60
    min_sm_volume: float = 100.0
    min_sm_wallets: int = 2


def check_sm_confirmation(
    position_side: PositionSide,
    market_price: float,
    flow: SMFlowState,
    config: SMConfirmationConfig | None = None,
) -> SMDecision:
    """Evaluate SM wallet flow against the current position.

    Args:
        position_side: Which side (YES/NO) the bot currently holds.
        market_price: Current market price ($0.00 - $1.00).
        flow: Aggregate SM flow state for this window.
        config: Configuration parameters. Uses defaults if None.

    Returns:
        SMDecision: HOLD, EXIT, or IGNORE.
    """
    cfg = config or SMConfirmationConfig()

    if market_price < cfg.price_floor:
        logger.debug(
            "SM IGNORE: price $%.2f below floor $%.2f",
            market_price, cfg.price_floor,
        )
        return SMDecision.IGNORE

    if market_price > cfg.price_ceiling:
        logger.debug(
            "SM IGNORE: price $%.2f above ceiling $%.2f",
            market_price, cfg.price_ceiling,
        )
        return SMDecision.IGNORE

    if flow.total_volume < cfg.min_sm_volume:
        logger.debug(
            "SM IGNORE: SM volume $%.2f below minimum $%.2f",
            flow.total_volume, cfg.min_sm_volume,
        )
        return SMDecision.IGNORE

    if flow.num_wallets < cfg.min_sm_wallets:
        logger.debug(
            "SM IGNORE: %d SM wallets below minimum %d",
            flow.num_wallets, cfg.min_sm_wallets,
        )
        return SMDecision.IGNORE

    if position_side == PositionSide.YES:
        position_fraction = flow.yes_fraction
    else:
        position_fraction = flow.no_fraction

    if position_fraction >= cfg.agreement_threshold:
        logger.info(
            "SM HOLD: %.0f%% of SM flow agrees with %s position (price $%.2f)",
            position_fraction * 100, position_side.value, market_price,
        )
        return SMDecision.HOLD

    opposite_fraction = 1.0 - position_fraction
    if opposite_fraction >= cfg.agreement_threshold:
        logger.info(
            "SM EXIT: %.0f%% of SM flow disagrees with %s position (price $%.2f)",
            opposite_fraction * 100, position_side.value, market_price,
        )
        return SMDecision.EXIT

    logger.debug(
        "SM IGNORE: SM flow split (%.0f%% agree) — no clear signal",
        position_fraction * 100,
    )
    return SMDecision.IGNORE
