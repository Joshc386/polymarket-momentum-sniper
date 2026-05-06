"""Kalman Filter signal — multi-source price fusion for oracle prediction.

Fuses Binance, Coinbase, and Chainlink oracle prices to estimate the
true underlying BTC price and predict where the oracle will land before
it updates. Replaces the simple oracle lag (L1) signal with a proper
state-space estimator.

State model:
    x = [price, velocity]  (2×1)

    Prediction:
        x̂ₖ|ₖ₋₁ = F · x̂ₖ₋₁|ₖ₋₁
        Pₖ|ₖ₋₁ = F · Pₖ₋₁|ₖ₋₁ · Fᵀ + Q

    Update (per measurement source):
        yₖ = zₖ - H · x̂ₖ|ₖ₋₁           (innovation)
        Sₖ = H · Pₖ|ₖ₋₁ · Hᵀ + R       (innovation covariance)
        Kₖ = Pₖ|ₖ₋₁ · Hᵀ · Sₖ⁻¹        (Kalman gain)
        x̂ₖ|ₖ = x̂ₖ|ₖ₋₁ + Kₖ · yₖ
        Pₖ|ₖ = (I - Kₖ · H) · Pₖ|ₖ₋₁

    Three measurement sources (different R per source):
        - Binance:  fastest, noisiest (R_binance ≈ 1e-4)
        - Coinbase: independent exchange (R_coinbase ≈ 2e-4)
        - Oracle:   authoritative but lagged (R_oracle ≈ 5e-4)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KalmanOutput:
    """Output from a Kalman Filter update step."""
    predicted_price: float = 0.0    # State estimate before measurement update
    filtered_price: float = 0.0     # State estimate after update
    velocity: float = 0.0           # Price rate of change ($/sec)
    prediction_residual: float = 0.0  # (filtered - oracle) / oracle
    noise_ratio: float = 0.0       # P[0,0] / baseline — estimation uncertainty
    breakout_flag: bool = False     # True if |velocity| exceeds threshold
    signal: float = 0.0            # [-1, 1] oracle lag signal (drop-in for L1)


class KalmanFilterSignal:
    """Multi-source Kalman Filter for oracle price prediction.

    Fuses price data from multiple exchanges to estimate the true
    BTC price and its velocity. The velocity estimate provides a
    leading indicator of where the oracle price will move next.

    Args:
        process_noise_scale: Base process noise variance for Q matrix.
        measurement_noise_binance: R for Binance measurements.
        measurement_noise_coinbase: R for Coinbase measurements.
        measurement_noise_oracle: R for Chainlink oracle measurements.
        adaptive_window: Number of recent innovations to track for
            adaptive Q scaling.
        breakout_velocity_threshold: Velocity ($/sec) above which a
            breakout flag is set.
        max_expected_lag: Maximum expected oracle lag as fraction of
            price (for signal normalisation).
        velocity_confidence_boost: Signal multiplier when velocity
            confirms the lag direction.
    """

    def __init__(
        self,
        process_noise_scale: float = 1e-5,
        measurement_noise_binance: float = 1e-4,
        measurement_noise_coinbase: float = 2e-4,
        measurement_noise_oracle: float = 5e-4,
        adaptive_window: int = 30,
        breakout_velocity_threshold: float = 0.5,
        max_expected_lag: float = 0.001,
        velocity_confidence_boost: float = 1.15,
    ) -> None:
        self._process_noise_scale = process_noise_scale
        self._R_binance = measurement_noise_binance
        self._R_coinbase = measurement_noise_coinbase
        self._R_oracle = measurement_noise_oracle
        self._adaptive_window = adaptive_window
        self._breakout_threshold = breakout_velocity_threshold
        self._max_expected_lag = max_expected_lag
        self._velocity_boost = velocity_confidence_boost

        # State: [price, velocity]
        self._x = np.zeros(2)       # State estimate
        self._P = np.eye(2) * 1e-2  # Covariance estimate
        self._H = np.array([[1.0, 0.0]])  # Measurement matrix

        # Adaptive Q tracking
        self._innovations: deque[float] = deque(maxlen=adaptive_window)
        self._base_innovation_var = process_noise_scale * 100

        # Timing
        self._last_update_time: float = 0.0
        self._initialized = False

        # Latest output
        self.output = KalmanOutput()

    def reset(self) -> None:
        """Reset filter state (call on new session)."""
        self._x = np.zeros(2)
        self._P = np.eye(2) * 1e-2
        self._innovations.clear()
        self._last_update_time = 0.0
        self._initialized = False
        self.output = KalmanOutput()

    def update(
        self,
        binance_price: float,
        coinbase_price: float,
        oracle_price: float,
        coinbase_connected: bool = True,
        timestamp: float | None = None,
    ) -> KalmanOutput:
        """Run one predict-update cycle with available measurements.

        Args:
            binance_price: Current Binance BTC/USDT price.
            coinbase_price: Current Coinbase BTC-USD price (0 if offline).
            oracle_price: Current Chainlink oracle price.
            coinbase_connected: Whether Coinbase feed is active.
            timestamp: Unix timestamp (defaults to time.time()).

        Returns:
            KalmanOutput with filtered price, velocity, signal, etc.
        """
        now = timestamp or time.time()

        # First call: initialise state from Binance price
        if not self._initialized:
            if binance_price > 0:
                self._x[0] = binance_price
                self._x[1] = 0.0
                self._last_update_time = now
                self._initialized = True
                self.output = KalmanOutput(
                    predicted_price=binance_price,
                    filtered_price=binance_price,
                )
            return self.output

        dt = now - self._last_update_time
        if dt <= 0:
            return self.output
        self._last_update_time = now

        # ── Predict step ──
        # F = [[1, dt], [0, 1]]
        F = np.array([[1.0, dt], [0.0, 1.0]])
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + self._build_Q(dt)

        predicted_price = x_pred[0]

        # ── Update step (sequential per measurement source) ──
        # Process each available measurement independently
        x_upd = x_pred.copy()
        P_upd = P_pred.copy()

        # Binance (always available if price > 0)
        if binance_price > 0:
            x_upd, P_upd, innov = self._measurement_update(
                x_upd, P_upd, binance_price, self._R_binance
            )
            self._innovations.append(innov)

        # Coinbase (only if connected and price valid)
        if coinbase_connected and coinbase_price > 0:
            x_upd, P_upd, _ = self._measurement_update(
                x_upd, P_upd, coinbase_price, self._R_coinbase
            )

        # Oracle (always available if price > 0, but lagged)
        if oracle_price > 0:
            x_upd, P_upd, _ = self._measurement_update(
                x_upd, P_upd, oracle_price, self._R_oracle
            )

        # ── Store updated state ──
        self._x = x_upd
        self._P = P_upd

        filtered_price = self._x[0]
        velocity = self._x[1]

        # ── Compute signal output ──
        # How far ahead the filtered price is vs oracle
        if oracle_price > 0:
            residual = (filtered_price - oracle_price) / oracle_price
        else:
            residual = 0.0

        # Normalise to [-1, 1]
        signal = np.clip(residual / self._max_expected_lag, -1.0, 1.0)

        # Boost when velocity confirms direction
        if (np.sign(residual) == np.sign(velocity)
                and abs(velocity) > self._breakout_threshold * 0.1):
            signal = np.clip(
                signal * self._velocity_boost, -1.0, 1.0
            )

        # Noise ratio: estimation uncertainty relative to baseline
        noise_ratio = self._P[0, 0] / max(self._base_innovation_var, 1e-10)

        # Breakout detection: velocity exceeds threshold
        breakout = abs(velocity) > self._breakout_threshold

        self.output = KalmanOutput(
            predicted_price=predicted_price,
            filtered_price=filtered_price,
            velocity=velocity,
            prediction_residual=residual,
            noise_ratio=noise_ratio,
            breakout_flag=breakout,
            signal=float(signal),
        )
        return self.output

    def _measurement_update(
        self,
        x: np.ndarray,
        P: np.ndarray,
        z: float,
        R: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Single measurement update step.

        Args:
            x: Prior state estimate [price, velocity].
            P: Prior covariance (2×2).
            z: Measured price.
            R: Measurement noise variance.

        Returns:
            (updated_x, updated_P, innovation)
        """
        H = self._H
        y = z - (H @ x)[0]                          # Innovation
        S = (H @ P @ H.T)[0, 0] + R                 # Innovation covariance
        K = (P @ H.T) / S                            # Kalman gain (2×1)
        x_new = x + K.flatten() * y
        P_new = (np.eye(2) - K @ H) @ P
        return x_new, P_new, y

    def _build_Q(self, dt: float) -> np.ndarray:
        """Build adaptive process noise matrix Q.

        Q is scaled by recent innovation variance. This lets the filter
        track faster during volatile periods and smooth more during calm.

        The continuous-time process noise model assumes random acceleration:
            Q = q * [[dt³/3, dt²/2],
                      [dt²/2, dt    ]]

        where q is the spectral density, adapted from recent innovations.
        """
        # Adaptive scaling
        if len(self._innovations) >= 5:
            recent_var = np.var(self._innovations)
            scale = max(1.0, recent_var / max(self._base_innovation_var, 1e-10))
        else:
            scale = 1.0

        q = self._process_noise_scale * scale

        Q = q * np.array([
            [dt**3 / 3, dt**2 / 2],
            [dt**2 / 2, dt],
        ])
        return Q

    def compute(
        self,
        binance_price: float,
        coinbase_price: float,
        oracle_price: float,
        coinbase_connected: bool = True,
    ) -> float:
        """Convenience method: update and return just the signal value.

        Drop-in replacement for OracleLagSignal.compute() output range.

        Returns:
            Signal in [-1, 1]. Positive = exchange leading oracle upward.
        """
        output = self.update(
            binance_price=binance_price,
            coinbase_price=coinbase_price,
            oracle_price=oracle_price,
            coinbase_connected=coinbase_connected,
        )
        return output.signal
