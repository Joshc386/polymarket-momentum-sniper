"""Health monitor for 24/7 operation.

Tracks data feed health, detects stale data, and triggers alerts
when components go unhealthy. Used by main loop to decide whether
to pause trading until feeds recover.
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FeedHealth:
    """Health status of a single data feed."""
    name: str
    last_update: float = 0.0
    stale_threshold_sec: float = 10.0
    consecutive_failures: int = 0
    max_failures_before_alert: int = 3
    is_critical: bool = True  # If True, trading pauses when unhealthy

    @property
    def is_healthy(self) -> bool:
        if self.last_update <= 0:
            return False
        age = time.time() - self.last_update
        return age < self.stale_threshold_sec

    @property
    def age_seconds(self) -> float:
        if self.last_update <= 0:
            return float("inf")
        return time.time() - self.last_update

    @property
    def status_str(self) -> str:
        if self.last_update <= 0:
            return "NO DATA"
        age = self.age_seconds
        if age < self.stale_threshold_sec:
            return f"OK ({age:.0f}s)"
        return f"STALE ({age:.0f}s)"


class HealthMonitor:
    """Monitors all data feeds and bot health.

    Call update_feed() whenever a feed produces data.
    Call check_health() periodically to get overall status.
    """

    def __init__(self):
        self.feeds: dict[str, FeedHealth] = {
            "binance": FeedHealth(
                name="Binance",
                stale_threshold_sec=10.0,
                is_critical=True,
            ),
            "oracle": FeedHealth(
                name="Chainlink Oracle",
                stale_threshold_sec=30.0,  # Oracle polls less frequently
                is_critical=True,
            ),
            "coinbase": FeedHealth(
                name="Coinbase",
                stale_threshold_sec=15.0,
                is_critical=False,  # Non-critical — just confirmation
            ),
            "orderbook": FeedHealth(
                name="Polymarket Orderbook",
                stale_threshold_sec=20.0,
                is_critical=True,
            ),
            "market_discovery": FeedHealth(
                name="Market Discovery",
                stale_threshold_sec=60.0,
                is_critical=True,
            ),
            "coinglass": FeedHealth(
                name="CoinGlass",
                stale_threshold_sec=120.0,
                is_critical=False,
            ),
        }

        self._last_alert_time: dict[str, float] = {}
        self._alert_cooldown = 300  # 5 minutes between repeated alerts
        self.bot_start_time = time.time()
        self.total_restarts = 0

    def update_feed(self, feed_name: str, success: bool = True):
        """Record a successful or failed update from a feed."""
        feed = self.feeds.get(feed_name)
        if not feed:
            return
        if success:
            feed.last_update = time.time()
            feed.consecutive_failures = 0
        else:
            feed.consecutive_failures += 1

    def check_health(self) -> tuple[bool, list[str]]:
        """Check overall health.

        Returns:
            (can_trade, issues) — can_trade is False if any critical feed is unhealthy.
            issues is a list of human-readable problem descriptions.
        """
        can_trade = True
        issues = []

        for key, feed in self.feeds.items():
            if not feed.is_healthy:
                issue = f"{feed.name}: {feed.status_str}"
                issues.append(issue)
                if feed.is_critical:
                    can_trade = False

        return can_trade, issues

    def get_alerts(self) -> list[str]:
        """Get alerts for feeds that just became unhealthy (with cooldown).

        Returns list of alert messages to send via Telegram.
        """
        alerts = []
        now = time.time()

        for key, feed in self.feeds.items():
            if not feed.is_healthy and feed.consecutive_failures >= feed.max_failures_before_alert:
                last_alert = self._last_alert_time.get(key, 0)
                if now - last_alert > self._alert_cooldown:
                    alerts.append(
                        f"Feed unhealthy: {feed.name} — {feed.status_str} "
                        f"(failures: {feed.consecutive_failures})"
                    )
                    self._last_alert_time[key] = now

        return alerts

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.bot_start_time

    @property
    def uptime_str(self) -> str:
        secs = self.uptime_seconds
        hours = int(secs // 3600)
        mins = int((secs % 3600) // 60)
        return f"{hours}h {mins}m"

    def summary_str(self) -> str:
        """One-line health summary for dashboard."""
        healthy = sum(1 for f in self.feeds.values() if f.is_healthy)
        total = len(self.feeds)
        return f"{healthy}/{total} feeds OK | Uptime: {self.uptime_str}"
