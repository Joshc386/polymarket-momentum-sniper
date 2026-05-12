import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """Current risk management state."""
    can_trade: bool = True
    size_multiplier: float = 1.0
    reason: str = ""
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    daily_pnl: float = 0.0
    paused_until: float = 0.0     # Unix timestamp
    daily_trades: int = 0


class RiskManager:
    """Enforces loss caps, streak handling, drawdown breaker, trade caps,
    and volatility filters.

    Backward compatible — Bot G passes old params and gets old behavior.
    New features (drawdown breaker, daily trade cap, disable streak
    reduction) only activate when explicitly configured.

    Supports injectable time sources so the backtest can use simulated
    candle timestamps instead of wall-clock time.
    """

    def __init__(
        self,
        daily_loss_cap_pct: float = 0.20,
        daily_loss_warn_pct: float = 0.15,
        min_bankroll: float = 10.0,
        streak_reduce_at: int = 3,
        streak_reduce_factor: float = 0.75,
        streak_heavy_reduce_at: int = 5,
        streak_heavy_reduce_factor: float = 0.5,
        streak_pause_at: int = 7,
        streak_pause_minutes: int = 30,
        streak_reset_wins: int = 3,
        low_volatility_threshold: float = 0.0001,
        high_volatility_threshold: float = 0.02,
        high_volatility_size_factor: float = 0.5,
        # New features — disabled by default for backward compatibility
        streak_reduction_enabled: bool = True,
        drawdown_pct: float = 0.0,
        daily_trade_cap: int = 0,
        btc_distance_stop: float = 0.0,
        btc_distance_min_secs_remaining: float = 60.0,
        regime_size_overrides: dict | None = None,
        time_func: callable = None,
        date_func: callable = None,
    ):
        self.daily_loss_cap_pct = daily_loss_cap_pct
        self.daily_loss_warn_pct = daily_loss_warn_pct
        self.min_bankroll = min_bankroll
        self.streak_reduce_at = streak_reduce_at
        self.streak_reduce_factor = streak_reduce_factor
        self.streak_heavy_reduce_at = streak_heavy_reduce_at
        self.streak_heavy_reduce_factor = streak_heavy_reduce_factor
        self.streak_pause_at = streak_pause_at
        self.streak_pause_minutes = streak_pause_minutes
        self.streak_reset_wins = streak_reset_wins
        self.low_vol_threshold = low_volatility_threshold
        self.high_vol_threshold = high_volatility_threshold
        self.high_vol_size_factor = high_volatility_size_factor

        # New features
        self.streak_reduction_enabled = streak_reduction_enabled
        self.drawdown_pct = drawdown_pct
        self.daily_trade_cap = daily_trade_cap
        self.btc_distance_stop = btc_distance_stop
        self.btc_distance_min_secs_remaining = btc_distance_min_secs_remaining
        self.regime_size_overrides = regime_size_overrides or {}

        # Time sources — injectable for backtesting
        self._now = time_func or time.time
        self._today = date_func or (lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        # State
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.daily_pnl = 0.0
        self.daily_start_bankroll = 0.0
        self.daily_trades = 0
        self.last_daily_reset: str = ""
        self.paused_until = 0.0
        self._streak_was_paused = False

        # Drawdown breaker — tracks peak bankroll across the session
        self.peak_bankroll = 0.0
        self._drawdown_paused = False

    def set_daily_start(self, bankroll: float):
        """Set the starting bankroll for daily loss cap tracking.

        Also resets streak counters on new UTC day — fresh day, fresh start.
        """
        today = self._today()
        if today != self.last_daily_reset:
            if self.last_daily_reset:  # Don't log on first call
                logger.info(
                    f"New day {today}: reset daily PnL and streak counters "
                    f"(prev: {self.consecutive_losses}L/{self.consecutive_wins}W, "
                    f"{self.daily_trades} trades)"
                )
            self.daily_pnl = 0.0
            self.daily_start_bankroll = bankroll
            self.daily_trades = 0
            self.consecutive_losses = 0
            self.consecutive_wins = 0
            self._streak_was_paused = False
            self.paused_until = 0.0
            self.last_daily_reset = today

        # Update peak bankroll for drawdown tracking
        if bankroll > self.peak_bankroll:
            self.peak_bankroll = bankroll

    def record_result(self, pnl: float):
        """Record a trade result and update streak tracking."""
        self.daily_pnl += pnl
        self.daily_trades += 1

        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            # Reset streak pause after enough consecutive wins
            if self.consecutive_wins >= self.streak_reset_wins and self._streak_was_paused:
                self._streak_was_paused = False
                logger.info(f"Streak reset after {self.consecutive_wins} consecutive wins")
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

            # Check if we need to pause
            if self.consecutive_losses >= self.streak_pause_at:
                self.paused_until = self._now() + self.streak_pause_minutes * 60
                self._streak_was_paused = True
                logger.warning(
                    f"{self.consecutive_losses} consecutive losses — "
                    f"pausing for {self.streak_pause_minutes}m"
                )

    def evaluate(
        self,
        bankroll: float,
        recent_volatility: float = 0.0,
        current_regime: str = "",
    ) -> RiskState:
        """Check all risk rules and return current state.

        Args:
            bankroll: Current bankroll in USDC.
            recent_volatility: BTC price change (absolute fraction) over last 15 minutes.
            current_regime: Current market regime label (e.g. "ranging", "trending_up").

        Returns:
            RiskState with trading permission and size multiplier.
        """
        self.set_daily_start(bankroll)
        state = RiskState(
            consecutive_losses=self.consecutive_losses,
            consecutive_wins=self.consecutive_wins,
            daily_pnl=self.daily_pnl,
            paused_until=self.paused_until,
            daily_trades=self.daily_trades,
        )

        # Check minimum bankroll
        if bankroll < self.min_bankroll:
            state.can_trade = False
            state.reason = f"Bankroll ${bankroll:.2f} below minimum ${self.min_bankroll:.2f}"
            return state

        # Drawdown breaker — pause if bankroll dropped too far from peak
        if self.drawdown_pct > 0 and self.peak_bankroll > 0:
            drawdown = (self.peak_bankroll - bankroll) / self.peak_bankroll
            if drawdown >= self.drawdown_pct:
                state.can_trade = False
                state.reason = (
                    f"Drawdown breaker: {drawdown:.1%} from peak "
                    f"${self.peak_bankroll:.2f} (limit: {self.drawdown_pct:.0%})"
                )
                self._drawdown_paused = True
                return state
            elif self._drawdown_paused:
                # Recovery: bankroll recovered above drawdown threshold
                self._drawdown_paused = False
                logger.info(
                    f"Drawdown breaker released: bankroll ${bankroll:.2f} "
                    f"recovered (peak ${self.peak_bankroll:.2f}, "
                    f"dd {drawdown:.1%} < {self.drawdown_pct:.0%})"
                )

        # Daily trade cap
        if self.daily_trade_cap > 0 and self.daily_trades >= self.daily_trade_cap:
            state.can_trade = False
            state.reason = (
                f"Daily trade cap: {self.daily_trades}/{self.daily_trade_cap}"
            )
            return state

        # Streak pause expiry — reset BEFORE daily cap check so counters
        # aren't left in a broken half-reset state when both conditions fire.
        if self._streak_was_paused and self.paused_until > 0 and self._now() >= self.paused_until:
            logger.info("Streak pause served — resetting to full trading")
            self.consecutive_losses = 0
            self.consecutive_wins = 0
            self._streak_was_paused = False
            self.paused_until = 0.0
            state.consecutive_losses = 0
            state.consecutive_wins = 0
            state.paused_until = 0.0

        # Check daily loss cap (hard stop at 20%)
        if self.daily_start_bankroll > 0:
            loss_cap = self.daily_start_bankroll * self.daily_loss_cap_pct
            if self.daily_pnl < -loss_cap:
                state.can_trade = False
                state.reason = f"Daily loss cap hit: ${self.daily_pnl:.2f} (cap: -${loss_cap:.2f})"
                return state

        # Check streak pause (still active — hasn't expired yet)
        if self.paused_until > 0 and self._now() < self.paused_until:
            remaining = int(self.paused_until - self._now())
            state.can_trade = False
            state.reason = f"Streak pause -- {remaining}s remaining"
            return state

        # Size multiplier — start at 1.0 and apply adjustments
        multiplier = 1.0

        # Graduated size multiplier from streak (only if enabled)
        if self.streak_reduction_enabled:
            if self.consecutive_losses >= self.streak_heavy_reduce_at:
                multiplier = self.streak_heavy_reduce_factor  # 0.5x at 5+ losses
            elif self.consecutive_losses >= self.streak_reduce_at:
                multiplier = self.streak_reduce_factor  # 0.75x at 3-4 losses

        # Soft daily loss warning (reduce size at 15%, hard stop at 20%)
        _daily_warn_active = False
        if self.daily_start_bankroll > 0:
            warn_cap = self.daily_start_bankroll * self.daily_loss_warn_pct
            if self.daily_pnl < -warn_cap:
                multiplier *= 0.5
                _daily_warn_active = True

        # Regime sizing — apply size overrides per regime
        _regime_sizing_active = False
        regime_mult = 1.0
        if current_regime and self.regime_size_overrides:
            regime_mult = self.regime_size_overrides.get(current_regime, 1.0)
            if regime_mult != 1.0:
                multiplier *= regime_mult
                _regime_sizing_active = True

        # Volatility filter
        if recent_volatility > 0:
            if recent_volatility < self.low_vol_threshold:
                state.can_trade = False
                state.reason = f"Low volatility ({recent_volatility:.5f} < {self.low_vol_threshold})"
                return state
            elif recent_volatility > self.high_vol_threshold:
                multiplier *= self.high_vol_size_factor

        # Floor: never go below 0.25x (allow tiny bets to recover)
        multiplier = max(0.25, multiplier)

        state.size_multiplier = multiplier
        state.can_trade = True
        if multiplier < 1.0 or multiplier > 1.0:
            reasons = []
            if self.streak_reduction_enabled:
                if self.consecutive_losses >= self.streak_heavy_reduce_at:
                    reasons.append(f"{self.consecutive_losses} losses (heavy)")
                elif self.consecutive_losses >= self.streak_reduce_at:
                    reasons.append(f"{self.consecutive_losses} losses")
            if _daily_warn_active:
                reasons.append("daily loss warning")
            if _regime_sizing_active:
                reasons.append(f"regime:{current_regime} ({regime_mult:.2f}x)")
            if recent_volatility > self.high_vol_threshold:
                reasons.append("high vol")
            if reasons:
                state.reason = f"Size adj ({multiplier:.2f}x): {', '.join(reasons)}"
            else:
                state.reason = "OK"
        else:
            state.reason = "OK"

        return state

    def should_stop_btc_distance(
        self,
        side: str,
        btc_price_now: float,
        btc_open_price: float,
        secs_remaining: float,
    ) -> bool:
        """Check if BTC has moved too far against our position vs window open.

        The backtest showed that when BTC moves $100+ against us relative
        to window open, exiting early saves money 91% of the time.

        Args:
            side: Our position side ("YES" or "NO").
            btc_price_now: Current BTC price.
            btc_open_price: BTC price at window open (oracle open).
            secs_remaining: Seconds remaining in window.

        Returns:
            True if the stop should trigger (exit position).
        """
        if self.btc_distance_stop <= 0:
            return False

        if btc_open_price <= 0 or btc_price_now <= 0:
            return False

        if secs_remaining < self.btc_distance_min_secs_remaining:
            return False

        distance = btc_price_now - btc_open_price

        # YES position loses when BTC drops → distance is negative
        if side == "YES" and distance < -self.btc_distance_stop:
            return True

        # NO position loses when BTC rises → distance is positive
        if side == "NO" and distance > self.btc_distance_stop:
            return True

        return False

    def compute_recent_volatility(self, candles) -> float:
        """Compute BTC price change over recent candles as proxy for 15-min volatility."""
        if len(candles) < 2:
            return 0.005  # Default mid-range if not enough data
        # Use last 15 candles (15 minutes of 1-min data)
        recent = list(candles)[-15:]
        if not recent:
            return 0.005
        high = max(c.high for c in recent)
        low = min(c.low for c in recent)
        mid = (high + low) / 2
        if mid <= 0:
            return 0.005
        return (high - low) / mid
