"""Signal hypotheses for evaluation by the Signal Lab agent.

Each hypothesis implements the SignalHypothesis protocol:
    name: str
    def compute(row: dict) -> float  # [-1, 1], 0.0 = no signal
    def required_fields() -> list[str]  # CSV columns needed
"""
