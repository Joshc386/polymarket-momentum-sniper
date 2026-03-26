"""Event calendar — pauses trading around scheduled macro events.

FOMC, CPI, NFP, and other major events break technical patterns.
Trading through them with a momentum/technical strategy is gambling.
"""

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


# Major scheduled events that move BTC significantly.
# Format: (month, day, hour_utc, name)
# Times are approximate — most are announced at 14:00 or 18:00 UTC.
EVENTS_2025 = [
    # FOMC Decisions (announcement ~18:00 UTC)
    (1, 29, 18, "FOMC Decision"),
    (3, 19, 18, "FOMC Decision"),
    (5, 7, 18, "FOMC Decision"),
    (6, 18, 18, "FOMC Decision"),
    (7, 30, 18, "FOMC Decision"),
    (9, 17, 18, "FOMC Decision"),
    (10, 29, 18, "FOMC Decision"),
    (12, 10, 18, "FOMC Decision"),
    # CPI Releases (8:30 AM ET = 12:30 UTC)
    (1, 15, 13, "CPI Release"),
    (2, 12, 13, "CPI Release"),
    (3, 12, 12, "CPI Release"),
    (4, 10, 12, "CPI Release"),
    (5, 13, 12, "CPI Release"),
    (6, 11, 12, "CPI Release"),
    (7, 15, 12, "CPI Release"),
    (8, 12, 12, "CPI Release"),
    (9, 10, 12, "CPI Release"),
    (10, 14, 12, "CPI Release"),
    (11, 12, 13, "CPI Release"),
    (12, 10, 13, "CPI Release"),
    # NFP / Jobs Report (8:30 AM ET = 12:30 UTC, first Friday)
    (1, 10, 13, "NFP Jobs Report"),
    (2, 7, 13, "NFP Jobs Report"),
    (3, 7, 13, "NFP Jobs Report"),
    (4, 4, 12, "NFP Jobs Report"),
    (5, 2, 12, "NFP Jobs Report"),
    (6, 6, 12, "NFP Jobs Report"),
    (7, 3, 12, "NFP Jobs Report"),
    (8, 1, 12, "NFP Jobs Report"),
    (9, 5, 12, "NFP Jobs Report"),
    (10, 3, 12, "NFP Jobs Report"),
    (11, 7, 13, "NFP Jobs Report"),
    (12, 5, 13, "NFP Jobs Report"),
]

EVENTS_2026 = [
    # FOMC Decisions
    (1, 28, 18, "FOMC Decision"),
    (3, 18, 18, "FOMC Decision"),
    (4, 29, 18, "FOMC Decision"),
    (6, 17, 18, "FOMC Decision"),
    (7, 29, 18, "FOMC Decision"),
    (9, 16, 18, "FOMC Decision"),
    (10, 28, 18, "FOMC Decision"),
    (12, 9, 18, "FOMC Decision"),
    # CPI Releases (estimated, BLS publishes schedule in advance)
    (1, 14, 13, "CPI Release"),
    (2, 11, 13, "CPI Release"),
    (3, 11, 12, "CPI Release"),
    (4, 14, 12, "CPI Release"),
    (5, 12, 12, "CPI Release"),
    (6, 10, 12, "CPI Release"),
    (7, 14, 12, "CPI Release"),
    (8, 12, 12, "CPI Release"),
    (9, 15, 12, "CPI Release"),
    (10, 13, 12, "CPI Release"),
    (11, 10, 13, "CPI Release"),
    (12, 9, 13, "CPI Release"),
    # NFP Jobs Reports (estimated first Fridays)
    (1, 9, 13, "NFP Jobs Report"),
    (2, 6, 13, "NFP Jobs Report"),
    (3, 6, 13, "NFP Jobs Report"),
    (4, 3, 12, "NFP Jobs Report"),
    (5, 1, 12, "NFP Jobs Report"),
    (6, 5, 12, "NFP Jobs Report"),
    (7, 2, 12, "NFP Jobs Report"),
    (8, 7, 12, "NFP Jobs Report"),
    (9, 4, 12, "NFP Jobs Report"),
    (10, 2, 12, "NFP Jobs Report"),
    (11, 6, 13, "NFP Jobs Report"),
    (12, 4, 13, "NFP Jobs Report"),
]


class EventCalendar:
    """Tracks scheduled macro events and blocks trading around them."""

    def __init__(self, buffer_minutes: int = 15):
        self.buffer = timedelta(minutes=buffer_minutes)
        self._events = self._build_event_list()

    def _build_event_list(self) -> list:
        """Build sorted list of (datetime, name) tuples."""
        events = []
        for month, day, hour, name in EVENTS_2025:
            try:
                dt = datetime(2025, month, day, hour, 0, tzinfo=timezone.utc)
                events.append((dt, name))
            except ValueError:
                pass  # Skip invalid dates

        for month, day, hour, name in EVENTS_2026:
            try:
                dt = datetime(2026, month, day, hour, 0, tzinfo=timezone.utc)
                events.append((dt, name))
            except ValueError:
                pass

        events.sort(key=lambda x: x[0])
        return events

    def is_event_window(self, timestamp: float = None) -> tuple:
        """Check if we're within the buffer zone of a scheduled event.

        Args:
            timestamp: Unix timestamp to check (default: now).

        Returns:
            (is_blocked, event_name or None, minutes_until_event or None)
        """
        if timestamp is None:
            now = datetime.now(timezone.utc)
        else:
            now = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        for event_dt, event_name in self._events:
            diff = event_dt - now
            # Check if we're within buffer before or after the event
            if abs(diff) <= self.buffer:
                minutes_until = diff.total_seconds() / 60
                return True, event_name, minutes_until

        return False, None, None

    def next_event(self, timestamp: float = None) -> tuple:
        """Get the next upcoming event.

        Returns:
            (event_name, datetime, minutes_until) or (None, None, None)
        """
        if timestamp is None:
            now = datetime.now(timezone.utc)
        else:
            now = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        for event_dt, event_name in self._events:
            if event_dt > now:
                minutes_until = (event_dt - now).total_seconds() / 60
                return event_name, event_dt, minutes_until

        return None, None, None

    def status_str(self, timestamp: float = None) -> str:
        """Human-readable status for dashboard display."""
        blocked, name, minutes = self.is_event_window(timestamp)
        if blocked:
            if minutes > 0:
                return f"BLOCKED: {name} in {minutes:.0f}m"
            else:
                return f"BLOCKED: {name} was {abs(minutes):.0f}m ago"

        next_name, next_dt, next_min = self.next_event(timestamp)
        if next_name and next_min < 120:
            return f"Next: {next_name} in {next_min:.0f}m"
        elif next_name:
            hours = next_min / 60
            return f"Next: {next_name} in {hours:.1f}h"
        return "No upcoming events"
