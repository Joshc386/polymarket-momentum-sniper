"""Replay engine — REALISTIC simulation of live trading conditions.

Key differences from a naive backtest:
1. Oracle lag is NOISY and VARIABLE — not a clean N-candle delay. The oracle
   updates unpredictably with measurement noise, simulating Chainlink's real
   aggregator behaviour (multiple node reports, heartbeat delays, deviation
   thresholds).
2. Oracle open price has measurement error — the exact price the oracle
   records at the 5-min boundary is slightly stale/noisy.
3. Market prices are SEMI-EFFICIENT — Polymarket orderbooks aren't random
   noise around 50%. Other traders are watching the same data. The market
   mid reflects partial information about the true direction, making it
   harder to find edge.
4. Slippage is variable — thin books mean worse fills on larger sizes.
5. Fill probability < 100% — GTC orders may not fill, FOK may get rejected.
6. No future knowledge — signals only see data available at decision time.
"""

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from data.binance_feed import Candle
from signals.oracle_lag import OracleLagSignal
from signals.momentum import MomentumSignal
from signals.orderbook_signal import OrderbookSignal
from signals.combiner import SignalCombiner
from strategy.entry_logic import EntryLogic, EntryDecision
from strategy.sizing import PositionSizer
from strategy.risk_manager import RiskManager
from logging_db.database import Database

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """All tuneable backtest parameters."""
    # Signal
    oracle_max_expected_lag: float = 0.001
    max_adjustment: float = 0.15
    momentum_roc_weight: float = 0.30
    momentum_direction_weight: float = 0.25
    momentum_volume_weight: float = 0.25
    momentum_body_ratio_weight: float = 0.10
    momentum_rsi_weight: float = 0.10
    momentum_rsi_period: int = 5
    momentum_lookback_candles: int = 10

    # Entry
    min_edge: float = 0.02
    max_edge: float = 0.10
    fee_adjustment: float = 0.01

    # Sizing
    kelly_multiplier: float = 0.25
    min_bet_usdc: float = 1.0
    max_bet_usdc: float = 5.0
    initial_bankroll: float = 100.0

    # Risk
    daily_loss_cap_pct: float = 0.20
    min_bankroll: float = 10.0
    streak_reduce_at: int = 3
    streak_reduce_factor: float = 0.5
    streak_pause_at: int = 7
    streak_pause_minutes: int = 30
    streak_reset_wins: int = 3
    low_volatility_threshold: float = 0.0001
    high_volatility_threshold: float = 0.02
    high_volatility_size_factor: float = 0.5

    # ── Realistic simulation parameters ──

    # Oracle simulation
    oracle_lag_mean_sec: float = 45.0     # Average oracle lag in seconds
    oracle_lag_std_sec: float = 30.0      # Std dev (lag varies 15-75s typically)
    oracle_noise_bps: float = 3.0         # Oracle measurement noise in basis points
    oracle_update_prob: float = 0.4       # Probability oracle updates on any given candle
    oracle_open_noise_bps: float = 5.0    # Noise on the window-open price snapshot

    # Market efficiency (how much the orderbook reflects true direction)
    market_efficiency: float = 0.35       # 0.0 = random, 1.0 = perfectly efficient
    market_base_spread: float = 0.06      # Base spread (wider = harder to profit)
    market_spread_noise: float = 0.02     # Spread variability
    market_mid_noise: float = 0.03        # Random noise on top of informed mid

    # Orderbook simulation (for L4 signal in backtest)
    ob_base_depth: float = 500.0          # Base total depth per side (shares)
    ob_depth_noise: float = 150.0         # Depth variability
    ob_num_levels: int = 8                # Number of price levels per side
    ob_imbalance_signal_strength: float = 0.4  # How much true direction leaks into book
    ob_flow_noise: float = 100.0          # Noise in depth changes between ticks

    # Execution realism
    slippage_base: float = 0.005          # Base slippage
    slippage_size_factor: float = 0.002   # Additional slippage per $1 bet
    fill_rate_gtc: float = 0.85           # GTC fill probability
    fill_rate_fok: float = 0.70           # FOK fill probability

    seed: int = 42


@dataclass
class WindowResult:
    """Result of a single 5-minute window."""
    window_start_time: int
    window_end_time: int
    open_price: float
    close_price: float
    actual_direction: str
    trade_placed: bool = False
    trade_side: str = ""
    trade_price: float = 0.0
    trade_size: float = 0.0
    trade_edge: float = 0.0
    trade_pnl: float = 0.0
    entry_time_remaining: float = 0.0
    oracle_lag_signal: float = 0.0
    momentum_signal: float = 0.0
    orderbook_signal_val: float = 0.0
    combined_signal: float = 0.0
    est_prob_up: float = 0.5
    won: bool = False
    fill_failed: bool = False


@dataclass
class BacktestResult:
    """Aggregate results from a full backtest run."""
    windows: list[WindowResult] = field(default_factory=list)
    total_windows: int = 0
    trades_placed: int = 0
    trades_attempted: int = 0
    fill_failures: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    final_bankroll: float = 0.0
    initial_bankroll: float = 100.0
    max_bankroll: float = 0.0
    min_bankroll: float = 0.0
    max_drawdown: float = 0.0
    daily_pnl: dict = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades_placed if self.trades_placed > 0 else 0.0

    @property
    def roi(self) -> float:
        return (self.final_bankroll - self.initial_bankroll) / self.initial_bankroll

    @property
    def avg_pnl_per_trade(self) -> float:
        return self.total_pnl / self.trades_placed if self.trades_placed > 0 else 0.0

    @property
    def skip_rate(self) -> float:
        return 1.0 - (self.trades_placed / self.total_windows) if self.total_windows > 0 else 0.0

    @property
    def fill_rate(self) -> float:
        return self.trades_placed / self.trades_attempted if self.trades_attempted > 0 else 0.0


class OracleSimulator:
    """Simulates realistic Chainlink oracle behaviour.

    Chainlink oracles don't update every block. They update when:
    - A heartbeat interval passes (e.g., 60s)
    - Price deviates beyond a threshold (e.g., 0.5%)
    - Multiple node operators submit and aggregation occurs

    This creates variable, noisy lag — NOT a clean fixed delay.
    """

    def __init__(self, rng: random.Random, cfg: BacktestConfig):
        self.rng = rng
        self.cfg = cfg
        self.current_oracle_price: float = 0.0
        self.last_update_candle: int = -999
        self._pending_price: float = 0.0

    def initialize(self, price: float):
        self.current_oracle_price = price
        self._pending_price = price

    def tick(self, candle_idx: int, exchange_price: float) -> float:
        """Called each candle. May or may not update the oracle price.

        Returns the current oracle price (which may be stale).
        """
        self._pending_price = exchange_price

        # Decide if oracle updates this tick
        candles_since_update = candle_idx - self.last_update_candle

        # Higher chance of update if:
        # - Long time since last update
        # - Large price deviation from current oracle
        deviation = abs(exchange_price - self.current_oracle_price) / self.current_oracle_price \
            if self.current_oracle_price > 0 else 0

        base_prob = self.cfg.oracle_update_prob
        # Increase probability if stale or large deviation
        time_boost = min(0.3, candles_since_update * 0.1)
        deviation_boost = min(0.4, deviation * 100)  # 1% deviation -> +0.4
        update_prob = min(0.95, base_prob + time_boost + deviation_boost)

        if self.rng.random() < update_prob:
            # Oracle updates — but to a NOISY version of a LAGGED price
            # Simulate variable lag: oracle sees a price from some seconds ago
            lag_secs = max(5, self.rng.gauss(
                self.cfg.oracle_lag_mean_sec,
                self.cfg.oracle_lag_std_sec,
            ))
            # Approximate: we only have candle-level data, so lag in fractional candles
            lag_fraction = lag_secs / 60.0

            # Interpolate: the "lagged" price is between previous and current
            # (rough approximation since we don't have sub-minute data)
            if lag_fraction >= 1.0:
                # More than 1 candle lag — use previous oracle price with noise
                lagged_price = self.current_oracle_price
            else:
                # Partial candle lag — blend current and oracle
                lagged_price = (
                    exchange_price * (1.0 - lag_fraction) +
                    self.current_oracle_price * lag_fraction
                )

            # Add measurement noise
            noise_pct = self.rng.gauss(0, self.cfg.oracle_noise_bps / 10000)
            self.current_oracle_price = lagged_price * (1.0 + noise_pct)
            self.last_update_candle = candle_idx

        return self.current_oracle_price

    def snapshot_window_open(self, exchange_price: float) -> float:
        """Snapshot oracle price at window open — with measurement noise.

        This is the resolution reference price. In reality the oracle
        captures a slightly stale/noisy price at the boundary.
        """
        noise_pct = self.rng.gauss(0, self.cfg.oracle_open_noise_bps / 10000)
        return exchange_price * (1.0 + noise_pct)


class MarketSimulator:
    """Simulates a semi-efficient Polymarket orderbook.

    In reality, Polymarket markets reflect information from other traders
    watching the same BTC price feeds. The market is NOT dumb noise around
    50% — it partially prices in the likely direction.

    Market efficiency parameter controls how much true direction leaks:
    - 0.0 = market is pure noise (unrealistically easy)
    - 0.35 = market captures ~35% of the true signal (realistic)
    - 1.0 = market is perfectly efficient (impossible to beat)

    Also simulates full orderbook depth for L4 signal backtesting.
    """

    def __init__(self, rng: random.Random, cfg: BacktestConfig):
        self.rng = rng
        self.cfg = cfg
        # Track previous depth for delta computation
        self._prev_yes_bid_total = cfg.ob_base_depth
        self._prev_yes_ask_total = cfg.ob_base_depth
        self._prev_no_bid_total = cfg.ob_base_depth
        self._prev_no_ask_total = cfg.ob_base_depth

    def get_orderbook(
        self,
        exchange_price: float,
        window_open_price: float,
        seconds_remaining: float,
    ) -> tuple[float, float, float, float, float]:
        """Generate realistic orderbook prices.

        Args:
            exchange_price: Current BTC price.
            window_open_price: BTC price at start of window.
            seconds_remaining: Time left in window.

        Returns:
            (yes_best_bid, yes_best_ask, no_best_bid, no_best_ask, market_mid)
        """
        true_signal, market_mid, spread = self._compute_mid_and_spread(
            exchange_price, window_open_price, seconds_remaining
        )

        half_spread = spread / 2
        yes_best_bid = max(0.01, market_mid - half_spread)
        yes_best_ask = min(0.99, market_mid + half_spread)
        no_best_bid = max(0.01, (1.0 - market_mid) - half_spread)
        no_best_ask = min(0.99, (1.0 - market_mid) + half_spread)

        return yes_best_bid, yes_best_ask, no_best_bid, no_best_ask, market_mid

    def get_orderbook_full(
        self,
        exchange_price: float,
        window_open_price: float,
        seconds_remaining: float,
    ) -> dict:
        """Generate full orderbook with depth data for L4 signal.

        Returns dict with all fields needed by OrderbookSignal.compute().
        """
        true_signal, market_mid, spread = self._compute_mid_and_spread(
            exchange_price, window_open_price, seconds_remaining
        )

        half_spread = spread / 2
        yes_best_bid = max(0.01, market_mid - half_spread)
        yes_best_ask = min(0.99, market_mid + half_spread)
        no_best_bid = max(0.01, (1.0 - market_mid) - half_spread)
        no_best_ask = min(0.99, (1.0 - market_mid) + half_spread)

        # Simulate depth with directional bias
        # true_signal > 0 means UP is likely → informed traders buy YES, sell NO
        # This creates: more YES bid depth, less YES ask depth
        imbalance_strength = self.cfg.ob_imbalance_signal_strength

        # Time-dependent: book becomes more informed as window progresses
        time_pct = 1.0 - (seconds_remaining / 300.0)
        effective_imbalance = imbalance_strength * (0.3 + 0.7 * time_pct)

        # Base depth with directional bias
        base = self.cfg.ob_base_depth
        noise = self.cfg.ob_depth_noise

        # true_signal is in [-0.3, 0.3], scale to depth bias
        direction_bias = true_signal * effective_imbalance * base * 3.0

        yes_bid_total = max(50, base + direction_bias + self.rng.gauss(0, noise))
        yes_ask_total = max(50, base - direction_bias + self.rng.gauss(0, noise))
        no_bid_total = max(50, base - direction_bias + self.rng.gauss(0, noise))
        no_ask_total = max(50, base + direction_bias + self.rng.gauss(0, noise))

        # Top 5 depth is a fraction of total
        top5_fraction = 0.4 + self.rng.gauss(0, 0.05)
        yes_bid_top5 = yes_bid_total * max(0.2, min(0.7, top5_fraction))
        yes_ask_top5 = yes_ask_total * max(0.2, min(0.7, top5_fraction))

        # Number of levels (more levels when more depth)
        num_bid_levels = max(3, min(15, int(self.cfg.ob_num_levels + self.rng.gauss(0, 2))))
        num_ask_levels = max(3, min(15, int(self.cfg.ob_num_levels + self.rng.gauss(0, 2))))

        # Size-weighted mid: biased slightly toward the dominant side
        sw_mid_offset = (yes_bid_total - yes_ask_total) / (yes_bid_total + yes_ask_total) * 0.01
        sw_mid = market_mid + sw_mid_offset + self.rng.gauss(0, 0.003)

        # Deltas (flow)
        flow_noise = self.cfg.ob_flow_noise
        # True direction creates a flow bias: buying adds to bids, removes from asks
        flow_bias = true_signal * effective_imbalance * 100

        yes_bid_delta = (yes_bid_total - self._prev_yes_bid_total) + flow_bias + self.rng.gauss(0, flow_noise)
        yes_ask_delta = (yes_ask_total - self._prev_yes_ask_total) - flow_bias + self.rng.gauss(0, flow_noise)
        no_bid_delta = (no_bid_total - self._prev_no_bid_total) - flow_bias + self.rng.gauss(0, flow_noise)
        no_ask_delta = (no_ask_total - self._prev_no_ask_total) + flow_bias + self.rng.gauss(0, flow_noise)

        self._prev_yes_bid_total = yes_bid_total
        self._prev_yes_ask_total = yes_ask_total
        self._prev_no_bid_total = no_bid_total
        self._prev_no_ask_total = no_ask_total

        return {
            "yes_best_bid": yes_best_bid,
            "yes_best_ask": yes_best_ask,
            "no_best_bid": no_best_bid,
            "no_best_ask": no_best_ask,
            "market_mid": market_mid,
            "yes_bid_total": yes_bid_total,
            "yes_ask_total": yes_ask_total,
            "no_bid_total": no_bid_total,
            "no_ask_total": no_ask_total,
            "yes_bid_top5": yes_bid_top5,
            "yes_ask_top5": yes_ask_top5,
            "size_weighted_mid": sw_mid,
            "num_bid_levels": num_bid_levels,
            "num_ask_levels": num_ask_levels,
            "yes_bid_delta": yes_bid_delta,
            "yes_ask_delta": yes_ask_delta,
            "no_bid_delta": no_bid_delta,
            "no_ask_delta": no_ask_delta,
        }

    def _compute_mid_and_spread(
        self,
        exchange_price: float,
        window_open_price: float,
        seconds_remaining: float,
    ) -> tuple[float, float, float]:
        """Core mid-price and spread computation.

        Returns (true_signal, market_mid, spread).
        """
        # True signal: which way is price actually going?
        if window_open_price > 0:
            true_direction = (exchange_price - window_open_price) / window_open_price
        else:
            true_direction = 0.0

        # Convert to probability-like signal (sigmoid-ish)
        true_signal = max(-0.3, min(0.3, true_direction * 300))

        # Market captures a fraction of the true signal
        time_pct = 1.0 - (seconds_remaining / 300.0)
        effective_efficiency = self.cfg.market_efficiency * (0.5 + 0.5 * time_pct)

        informed_mid = 0.50 + true_signal * effective_efficiency

        # Add noise
        noise = self.rng.gauss(0, self.cfg.market_mid_noise)
        market_mid = max(0.15, min(0.85, informed_mid + noise))

        # Spread
        base_spread = self.cfg.market_base_spread
        spread_noise = self.rng.gauss(0, self.cfg.market_spread_noise)
        time_tightening = 0.8 + 0.2 * (seconds_remaining / 300.0)
        spread = max(0.02, (base_spread + spread_noise) * time_tightening)

        return true_signal, market_mid, spread


class ReplayEngine:
    """Replays historical candles through the full strategy stack
    with realistic market simulation."""

    def __init__(self, config: BacktestConfig, db: Database | None = None):
        self.cfg = config
        self.db = db
        self.rng = random.Random(config.seed)

        # Signal engine
        self.oracle_lag_signal = OracleLagSignal(max_expected_lag=config.oracle_max_expected_lag)
        self.momentum_signal = MomentumSignal(
            roc_weight=config.momentum_roc_weight,
            direction_weight=config.momentum_direction_weight,
            volume_weight=config.momentum_volume_weight,
            body_ratio_weight=config.momentum_body_ratio_weight,
            rsi_weight=config.momentum_rsi_weight,
            rsi_period=config.momentum_rsi_period,
            lookback_candles=config.momentum_lookback_candles,
        )
        self.orderbook_signal = OrderbookSignal()
        self.combiner = SignalCombiner(max_adjustment=config.max_adjustment)

        # Strategy
        self.entry_logic = EntryLogic(
            min_edge=config.min_edge,
            max_edge=config.max_edge,
            fee_adjustment=config.fee_adjustment,
        )
        self.sizer = PositionSizer(
            kelly_multiplier=config.kelly_multiplier,
            min_bet_usdc=config.min_bet_usdc,
            max_bet_usdc=config.max_bet_usdc,
        )
        # Simulated time for backtest — updated as candles are processed
        self._sim_time_ms: int = 0

        self.risk_mgr = RiskManager(
            daily_loss_cap_pct=config.daily_loss_cap_pct,
            min_bankroll=config.min_bankroll,
            streak_reduce_at=config.streak_reduce_at,
            streak_reduce_factor=config.streak_reduce_factor,
            streak_pause_at=config.streak_pause_at,
            streak_pause_minutes=config.streak_pause_minutes,
            streak_reset_wins=config.streak_reset_wins,
            low_volatility_threshold=config.low_volatility_threshold,
            high_volatility_threshold=config.high_volatility_threshold,
            high_volatility_size_factor=config.high_volatility_size_factor,
            time_func=lambda: self._sim_time_ms / 1000.0,
            date_func=lambda: datetime.fromtimestamp(
                self._sim_time_ms / 1000.0, tz=timezone.utc
            ).strftime("%Y-%m-%d"),
        )

        # Realistic simulators
        self.oracle_sim = OracleSimulator(self.rng, config)
        self.market_sim = MarketSimulator(self.rng, config)

    def run(self, candles: list[dict]) -> BacktestResult:
        """Run the full backtest over historical candles."""
        result = BacktestResult(initial_bankroll=self.cfg.initial_bankroll)
        bankroll = self.cfg.initial_bankroll
        result.max_bankroll = bankroll
        result.min_bankroll = bankroll

        candle_objects = self._to_candle_objects(candles)
        windows = self._chunk_into_windows(candle_objects)

        candle_history: deque = deque(maxlen=30)
        global_candle_idx = 0

        logger.info(f"Backtesting {len(windows)} windows ({len(candle_objects)} candles) [REALISTIC MODE]")

        # Initialize oracle with first price
        if candle_objects:
            self.oracle_sim.initialize(candle_objects[0].open)

        peak_bankroll = bankroll

        for window in windows:
            if len(window) < 2:
                global_candle_idx += len(window)
                continue

            result.total_windows += 1

            # Set simulated time to window start
            self._sim_time_ms = window[0].open_time

            # ── Window open: snapshot oracle price (with noise) ──
            # This is what the oracle records as the "start" price for resolution
            oracle_open_price = self.oracle_sim.snapshot_window_open(window[0].open)

            # Actual resolution (determined by ORACLE at end, not exact exchange price)
            # Oracle end price also has noise — simulated at window close
            window_close_exchange = window[-1].close
            oracle_close_price = self.oracle_sim.snapshot_window_open(window_close_exchange)
            actual_direction = "UP" if oracle_close_price >= oracle_open_price else "DOWN"

            window_result = WindowResult(
                window_start_time=window[0].open_time,
                window_end_time=window[-1].open_time,
                open_price=window[0].open,
                close_price=window[-1].close,
                actual_direction=actual_direction,
            )

            # Risk check
            recent_vol = self.risk_mgr.compute_recent_volatility(candle_history)
            risk_state = self.risk_mgr.evaluate(bankroll, recent_vol)

            trade_placed = False

            # Reset L4 history for each new window
            self.orderbook_signal.reset()

            # ── Process each candle in the window ──
            for i, candle in enumerate(window):
                global_candle_idx += 1
                self._sim_time_ms = candle.open_time

                # Update oracle (may or may not actually update)
                oracle_price = self.oracle_sim.tick(global_candle_idx, candle.close)

                secs_remaining = (len(window) - i - 1) * 60
                if secs_remaining < 5:
                    if candle.is_closed:
                        candle_history.append(candle)
                    continue

                # ── Compute L1 & L2 signals (only using available data) ──
                oracle_lag_val = self.oracle_lag_signal.compute(
                    exchange_price=candle.close,
                    oracle_price=oracle_price,
                    oracle_open_price=oracle_open_price,
                )

                momentum_val = self.momentum_signal.compute(
                    candles=candle_history,
                    current_candle=candle,
                )

                # ── Get full orderbook with depth data ──
                ob_full = self.market_sim.get_orderbook_full(
                    exchange_price=candle.close,
                    window_open_price=window[0].open,
                    seconds_remaining=secs_remaining,
                )

                ybb = ob_full["yes_best_bid"]
                yba = ob_full["yes_best_ask"]
                nbb = ob_full["no_best_bid"]
                nba = ob_full["no_best_ask"]
                market_mid = ob_full["market_mid"]

                # ── Compute L4 orderbook signal ──
                ob_signal_val = self.orderbook_signal.compute(
                    yes_bid_total=ob_full["yes_bid_total"],
                    yes_ask_total=ob_full["yes_ask_total"],
                    no_bid_total=ob_full["no_bid_total"],
                    no_ask_total=ob_full["no_ask_total"],
                    yes_best_bid=ybb,
                    yes_best_ask=yba,
                    yes_bid_depth_top5=ob_full["yes_bid_top5"],
                    yes_ask_depth_top5=ob_full["yes_ask_top5"],
                    size_weighted_mid=ob_full["size_weighted_mid"],
                    simple_mid=market_mid,
                    yes_bid_delta=ob_full["yes_bid_delta"],
                    yes_ask_delta=ob_full["yes_ask_delta"],
                    no_bid_delta=ob_full["no_bid_delta"],
                    no_ask_delta=ob_full["no_ask_delta"],
                    num_bid_levels=ob_full["num_bid_levels"],
                    num_ask_levels=ob_full["num_ask_levels"],
                )

                # ── Combine all signals ──
                raw_signal, est_prob_up = self.combiner.combine(
                    oracle_lag_signal=oracle_lag_val,
                    momentum_signal=momentum_val,
                    liquidation_signal=0.0,
                    seconds_remaining=secs_remaining,
                    coinbase_direction=0.0,
                    orderbook_signal=ob_signal_val,
                )

                # ── Entry evaluation ──
                if risk_state.can_trade and not trade_placed:
                    decision = self.entry_logic.evaluate(
                        est_prob_up=est_prob_up,
                        yes_best_ask=yba,
                        no_best_ask=nba,
                        yes_best_bid=ybb,
                        no_best_bid=nbb,
                        seconds_remaining=secs_remaining,
                        has_position=trade_placed,
                    )

                    if decision.should_enter:
                        win_prob = est_prob_up if decision.side == "YES" else (1.0 - est_prob_up)
                        bet_size = self.sizer.compute(
                            est_prob=win_prob,
                            share_price=decision.price,
                            bankroll=bankroll,
                            size_multiplier=risk_state.size_multiplier,
                        )

                        if bet_size > 0:
                            result.trades_attempted += 1

                            # ── Simulate fill probability ──
                            if secs_remaining > 60:
                                fill_prob = self.cfg.fill_rate_gtc
                            else:
                                fill_prob = self.cfg.fill_rate_fok

                            if self.rng.random() > fill_prob:
                                # Order not filled
                                result.fill_failures += 1
                                window_result.fill_failed = True
                                if candle.is_closed:
                                    candle_history.append(candle)
                                continue

                            trade_placed = True

                            # ── Realistic slippage ──
                            size_slip = self.cfg.slippage_size_factor * bet_size
                            total_slip = self.cfg.slippage_base + size_slip
                            # Random component
                            total_slip *= (0.5 + self.rng.random())
                            fill_price = min(decision.price + total_slip, 0.99)
                            num_shares = bet_size / fill_price

                            # ── Resolve ──
                            won = (decision.side == "YES" and actual_direction == "UP") or \
                                  (decision.side == "NO" and actual_direction == "DOWN")

                            if won:
                                payout = num_shares * 1.0
                                gross_profit = payout - bet_size
                                fee = gross_profit * 0.02 if gross_profit > 0 else 0
                                pnl = gross_profit - fee
                            else:
                                pnl = -bet_size

                            bankroll += pnl
                            self.risk_mgr.record_result(pnl)

                            window_result.trade_placed = True
                            window_result.trade_side = decision.side
                            window_result.trade_price = fill_price
                            window_result.trade_size = bet_size
                            window_result.trade_edge = decision.best_ev
                            window_result.trade_pnl = pnl
                            window_result.entry_time_remaining = secs_remaining
                            window_result.oracle_lag_signal = oracle_lag_val
                            window_result.momentum_signal = momentum_val
                            window_result.orderbook_signal_val = ob_signal_val
                            window_result.combined_signal = raw_signal
                            window_result.est_prob_up = est_prob_up
                            window_result.won = won

                            if won:
                                result.wins += 1
                            else:
                                result.losses += 1
                            result.trades_placed += 1
                            result.total_pnl += pnl

                            if self.db:
                                ts = datetime.fromtimestamp(
                                    candle.open_time / 1000, tz=timezone.utc
                                ).isoformat()
                                self.db.insert_trade(
                                    timestamp=ts,
                                    market_id=f"backtest-{window[0].open_time}",
                                    market_slug=f"bt-{window[0].open_time}",
                                    side=decision.side,
                                    entry_price=fill_price,
                                    size_usdc=bet_size,
                                    num_shares=num_shares,
                                    oracle_lag_signal=oracle_lag_val,
                                    momentum_signal=momentum_val,
                                    liquidation_signal=0.0,
                                    combined_signal=raw_signal,
                                    estimated_prob_up=est_prob_up,
                                    market_implied_prob=market_mid,
                                    edge=decision.best_ev,
                                    time_remaining_secs=secs_remaining,
                                    resolution=actual_direction,
                                    pnl=pnl,
                                    is_paper=2,
                                    btc_price_at_entry=candle.close,
                                    oracle_price_at_entry=oracle_price,
                                    oracle_price_at_open=oracle_open_price,
                                    fill_price=fill_price,
                                )

                            break  # One entry per window

                if candle.is_closed:
                    candle_history.append(candle)

            # Track bankroll
            result.max_bankroll = max(result.max_bankroll, bankroll)
            result.min_bankroll = min(result.min_bankroll, bankroll)
            peak_bankroll = max(peak_bankroll, bankroll)
            drawdown = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            result.max_drawdown = max(result.max_drawdown, drawdown)

            day_str = datetime.fromtimestamp(
                window[0].open_time / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d")
            if window_result.trade_placed:
                result.daily_pnl[day_str] = result.daily_pnl.get(day_str, 0) + window_result.trade_pnl

            result.windows.append(window_result)

            if result.total_windows % 2000 == 0:
                logger.info(
                    f"  Window {result.total_windows}: "
                    f"{result.trades_placed}/{result.trades_attempted} filled, "
                    f"WR={result.win_rate:.1%}, "
                    f"P&L=${result.total_pnl:+.2f}, "
                    f"Bankroll=${bankroll:.2f}"
                )

        result.final_bankroll = bankroll
        return result

    def _to_candle_objects(self, raw_candles: list[dict]) -> list[Candle]:
        return [
            Candle(
                open_time=c["open_time"],
                open=c["open"],
                high=c["high"],
                low=c["low"],
                close=c["close"],
                volume=c["volume"],
                is_closed=True,
            )
            for c in raw_candles
        ]

    def _chunk_into_windows(self, candles: list[Candle]) -> list[list[Candle]]:
        windows = []
        current_window = []

        for candle in candles:
            window_id = candle.open_time // 300000
            if current_window:
                prev_window_id = current_window[0].open_time // 300000
                if window_id != prev_window_id:
                    if len(current_window) >= 2:
                        windows.append(current_window)
                    current_window = []
            current_window.append(candle)

        if len(current_window) >= 2:
            windows.append(current_window)

        return windows
