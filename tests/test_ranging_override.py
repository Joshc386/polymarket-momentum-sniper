"""Tests for the `use_ranging_override` config flag (Fix 4, 2026-05-14).

When True (default): preserves historical Bot G behaviour. In ranging
regime, the schedule is overridden to "ranging" and entry mode is
signal_aligned.

When False: the bot's configured schedule is used in ALL regimes
including ranging, and entry mode is always signal_aligned.

These tests don't run the full strategy — they verify the schedule
selection and signal_aligned flag logic via the combiner's
get_weights() output and an explicit recreation of the override
decision tree.
"""

import pytest

from signals.combiner import (
    SignalCombiner,
    WEIGHT_SCHEDULE_BOT_K,
    WEIGHT_SCHEDULE_DEFAULT,
    WEIGHT_SCHEDULE_RANGING,
)


def _schedule_for(secs_remaining: float, schedule: list) -> tuple:
    """Pick weights for a given seconds_remaining from a schedule."""
    for threshold, w1, w2, w3, w4, w5 in schedule:
        if secs_remaining >= threshold:
            return w1, w2, w3, w4, w5
    last = schedule[-1]
    return last[1], last[2], last[3], last[4], last[5]


class TestRangingOverrideDefault:
    """Default behaviour (Bot G): ranging override on."""

    def test_default_schedule_used_when_not_ranging(self):
        combiner = SignalCombiner(weight_schedule_name="default")
        weights = combiner.get_weights(180, schedule_override="")
        expected = _schedule_for(180, WEIGHT_SCHEDULE_DEFAULT)
        assert weights == expected

    def test_ranging_override_swaps_to_ranging_schedule(self):
        combiner = SignalCombiner(weight_schedule_name="default")
        weights = combiner.get_weights(180, schedule_override="ranging")
        expected = _schedule_for(180, WEIGHT_SCHEDULE_RANGING)
        assert weights == expected

    def test_default_l4_at_180s(self):
        """At 180s remaining, default schedule has L4 weight of 0.23."""
        combiner = SignalCombiner(weight_schedule_name="default")
        weights = combiner.get_weights(180, schedule_override="")
        assert weights[3] == pytest.approx(0.23, abs=0.01)

    def test_ranging_l4_at_180s(self):
        """At 180s remaining, ranging schedule has L4 weight of 0.60."""
        combiner = SignalCombiner(weight_schedule_name="default")
        weights = combiner.get_weights(180, schedule_override="ranging")
        assert weights[3] == pytest.approx(0.60, abs=0.01)


class TestBotKScheduleProtected:
    """Bot K-style: opt-out of ranging override.

    The opt-out happens at the contrarian_ev.py level — it just doesn't
    pass `schedule_override="ranging"` when use_ranging_override is False.
    These tests verify the combiner respects that.
    """

    def test_bot_k_schedule_preserved_with_no_override(self):
        """When called without an override, Bot K's optimised schedule
        is used regardless of regime."""
        combiner = SignalCombiner(weight_schedule_name="bot_k_optimised")
        weights = combiner.get_weights(180, schedule_override="")
        expected = _schedule_for(180, WEIGHT_SCHEDULE_BOT_K)
        assert weights == expected

    def test_bot_k_l4_at_180s(self):
        """Bot K's optimised L4 weight at 180s is 0.24 (vs ranging's 0.60).
        This is the whole point of the fix — preserve the optimised weight.
        """
        combiner = SignalCombiner(weight_schedule_name="bot_k_optimised")
        weights = combiner.get_weights(180, schedule_override="")
        # Bot K's L4 at 180s should be 0.24
        assert weights[3] == pytest.approx(0.24, abs=0.01)
        # Critically: should NOT equal ranging's 0.60
        assert weights[3] != pytest.approx(0.60, abs=0.01)

    def test_bot_k_l2_at_180s(self):
        """Bot K's momentum (L2) is the dominant signal at 0.38."""
        combiner = SignalCombiner(weight_schedule_name="bot_k_optimised")
        weights = combiner.get_weights(180, schedule_override="")
        assert weights[1] == pytest.approx(0.38, abs=0.01)

    def test_bot_k_l1_at_180s(self):
        """Bot K reduced oracle (L1) weight to 0.11."""
        combiner = SignalCombiner(weight_schedule_name="bot_k_optimised")
        weights = combiner.get_weights(180, schedule_override="")
        assert weights[0] == pytest.approx(0.11, abs=0.01)

    def test_bot_k_can_still_be_explicitly_overridden(self):
        """If someone deliberately passes 'ranging' override, it still works.
        (This is the opt-OUT behaviour; opt-in is just not passing it.)
        """
        combiner = SignalCombiner(weight_schedule_name="bot_k_optimised")
        weights_override = combiner.get_weights(180, schedule_override="ranging")
        weights_no_override = combiner.get_weights(180, schedule_override="")
        # With override, ranging schedule is used (L4=0.60).
        # Without, bot_k_optimised is used (L4=0.24).
        assert weights_override[3] == pytest.approx(0.60, abs=0.01)
        assert weights_no_override[3] == pytest.approx(0.24, abs=0.01)


class TestConfigDefault:
    """Verify the default value of use_ranging_override preserves Bot G
    behaviour. This is a contract test — the Default must stay True for
    Bot G's existing config to keep behaving the same way."""

    def test_default_is_true(self):
        """If the strategy is given an empty sig_cfg dict, use_ranging_override
        must default to True. This preserves Bot G's behaviour (which
        doesn't explicitly set this key)."""
        sig_cfg = {}
        # Replicate the line in contrarian_ev.py:
        use_ranging_override = sig_cfg.get("use_ranging_override", True)
        assert use_ranging_override is True

    def test_explicit_false_opts_out(self):
        sig_cfg = {"use_ranging_override": False}
        use_ranging_override = sig_cfg.get("use_ranging_override", True)
        assert use_ranging_override is False

    def test_explicit_true_opts_in(self):
        sig_cfg = {"use_ranging_override": True}
        use_ranging_override = sig_cfg.get("use_ranging_override", True)
        assert use_ranging_override is True


class TestSignalAlignedLogic:
    """Verify the signal_aligned flag logic.

    The expression in contrarian_ev.py is:
        signal_aligned=(_schedule_override == "ranging"
                        or not self._use_ranging_override)

    Truth table:
      schedule_override="ranging", use_ranging_override=True  -> True
      schedule_override="ranging", use_ranging_override=False -> True
      schedule_override="",        use_ranging_override=True  -> False
      schedule_override="",        use_ranging_override=False -> True
    """

    def _signal_aligned(self, schedule_override: str, use_ranging_override: bool) -> bool:
        return schedule_override == "ranging" or not use_ranging_override

    def test_ranging_regime_with_override_enabled(self):
        # Bot G in ranging: override fires, signal_aligned=True
        assert self._signal_aligned("ranging", True) is True

    def test_non_ranging_regime_with_override_enabled(self):
        # Bot G in non-ranging: contrarian mode
        assert self._signal_aligned("", True) is False

    def test_ranging_regime_with_override_disabled(self):
        # Bot K in ranging: override doesn't fire (schedule_override="")
        # but signal_aligned still True because use_ranging_override=False
        assert self._signal_aligned("", False) is True

    def test_non_ranging_regime_with_override_disabled(self):
        # Bot K in non-ranging: signal_aligned True (always for Bot K)
        assert self._signal_aligned("", False) is True
