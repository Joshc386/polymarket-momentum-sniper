"""Strategy registry — maps strategy names to factory classes.

The multi-bot runner looks up strategies by name from config and
instantiates them via this registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.bot_strategy import BotStrategy

from strategies.contrarian_ev import ContrarianEvStrategy
from strategies.kalman_ev import KalmanEvStrategy
from strategies.hmm_regime_ev import HmmRegimeEvStrategy
from strategies.enhanced_ev import EnhancedEvStrategy
from strategies.ou_reversion_ev import OuReversionEvStrategy
from strategies.latency_arb import LatencyArbStrategy


STRATEGY_REGISTRY: dict[str, type] = {
    "contrarian_ev": ContrarianEvStrategy,
    "kalman_ev": KalmanEvStrategy,
    "hmm_regime_ev": HmmRegimeEvStrategy,
    "enhanced_ev": EnhancedEvStrategy,
    "ou_reversion_ev": OuReversionEvStrategy,
    "latency_arb": LatencyArbStrategy,
}


def create_bot(name: str, strategy_name: str, cfg: dict) -> "BotStrategy":
    """Instantiate a bot strategy by name.

    Args:
        name: Display name for the bot instance.
        strategy_name: Key in STRATEGY_REGISTRY.
        cfg: Per-bot config dict from config_multi.yaml.

    Returns:
        Initialized BotStrategy instance.

    Raises:
        KeyError: If strategy_name not found in registry.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise KeyError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available: {available}"
        )
    cls = STRATEGY_REGISTRY[strategy_name]
    return cls(name=name, cfg=cfg)
