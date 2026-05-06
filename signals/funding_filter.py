"""Funding rate regime filter.

Uses cross-exchange funding rate as a risk/regime modifier rather than
a directional signal. When funding is extreme, liquidation cascade risk
is elevated and outcomes are less predictable — reduce sizing and raise
edge thresholds.

Research basis: Funding rate >0.05% per 8h signals overcrowded positioning.
Historically precedes cascades and increased volatility. Not useful for
predicting 5-min direction, but valuable as a regime filter.

Thresholds (from research):
  0.00-0.01%: Normal — no adjustment
  0.01-0.05%: Mild bias — slight caution
  0.05-0.10%: Overcrowded — reduce sizing, raise thresholds
  >0.10%: Market stress — high cascade risk, maximum caution
"""

from dataclasses import dataclass


@dataclass
class FundingRegimeFilter:
    """Modifies sizing and edge thresholds based on funding rate extremity.

    Does NOT produce a directional signal. Instead returns multipliers
    that other components use to adjust their behaviour.

    Args:
        mild_threshold: Funding rate (%) above which mild caution applies.
        high_threshold: Funding rate (%) above which heavy caution applies.
        stress_threshold: Funding rate (%) indicating market stress.
        mild_size_factor: Sizing multiplier in mild regime.
        high_size_factor: Sizing multiplier in high regime.
        stress_size_factor: Sizing multiplier in stress regime.
        mild_edge_multiplier: Edge threshold multiplier in mild regime.
        high_edge_multiplier: Edge threshold multiplier in high regime.
        stress_edge_multiplier: Edge threshold multiplier in stress regime.
    """

    mild_threshold: float = 0.01
    high_threshold: float = 0.05
    stress_threshold: float = 0.10
    mild_size_factor: float = 0.85
    high_size_factor: float = 0.60
    stress_size_factor: float = 0.30
    mild_edge_multiplier: float = 1.1
    high_edge_multiplier: float = 1.4
    stress_edge_multiplier: float = 2.0

    def evaluate(
        self,
        funding_rate: float,
        oi_change_pct: float = 0.0,
    ) -> dict:
        """Evaluate funding regime and return adjustment factors.

        Args:
            funding_rate: Current average funding rate across exchanges (%).
                Positive = longs pay shorts, negative = shorts pay longs.
            oi_change_pct: OI change percentage (amplifies signal when
                OI is rising during extreme funding — more leverage building).

        Returns:
            Dict with:
                regime: str — "normal", "mild", "high", "stress"
                size_factor: float — multiply position size by this (≤1.0)
                edge_multiplier: float — multiply required edge by this (≥1.0)
                cascade_risk: float — estimated cascade probability [0, 1]
        """
        abs_funding = abs(funding_rate)

        # Determine base regime
        if abs_funding >= self.stress_threshold:
            regime = "stress"
            size_factor = self.stress_size_factor
            edge_mult = self.stress_edge_multiplier
            cascade_risk = 0.80
        elif abs_funding >= self.high_threshold:
            regime = "high"
            size_factor = self.high_size_factor
            edge_mult = self.high_edge_multiplier
            cascade_risk = 0.50
        elif abs_funding >= self.mild_threshold:
            regime = "mild"
            size_factor = self.mild_size_factor
            edge_mult = self.mild_edge_multiplier
            cascade_risk = 0.20
        else:
            regime = "normal"
            size_factor = 1.0
            edge_mult = 1.0
            cascade_risk = 0.05

        # Amplify caution if OI is expanding during extreme funding
        # (more leverage building on top of already extreme positioning)
        if oi_change_pct > 2.0 and abs_funding >= self.mild_threshold:
            size_factor *= 0.85
            edge_mult *= 1.15
            cascade_risk = min(1.0, cascade_risk * 1.3)

        return {
            "regime": regime,
            "size_factor": max(0.1, size_factor),
            "edge_multiplier": edge_mult,
            "cascade_risk": cascade_risk,
            "funding_rate": funding_rate,
            "abs_funding": abs_funding,
        }
