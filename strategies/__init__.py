"""Strategy prototypes for evaluation by the Strategy Prototyper agent.

Each strategy implements the StrategyPrototype protocol:
    name: str
    description: str
    should_enter(signals, regime, account) -> (enter, direction, confidence)
    entry_params() -> dict
"""
