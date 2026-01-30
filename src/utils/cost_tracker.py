"""
MLB Video Pipeline - Cost Tracker

Track API usage and costs across services:
- OpenAI (GPT-4, GPT-3.5, etc.)
- ElevenLabs (TTS)

Features:
- Thread-safe singleton pattern
- Persistent storage (JSON)
- Daily/monthly cost summaries
- Budget alerts
- Detailed usage breakdown

Usage:
    from src.utils.cost_tracker import CostTracker

    tracker = CostTracker()
    tracker.log_openai_call(input_tokens=500, output_tokens=200, model="gpt-4o")
    tracker.log_elevenlabs_call(characters=450)
    print(tracker.get_daily_total())  # $0.45
"""

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict

from .logger import get_cost_logger


# =============================================================================
# Pricing Configuration (Updated January 2025)
# =============================================================================
# IMPORTANT: Update these prices regularly from official vendor pages

OPENAI_PRICING = {
    # Per 1 million tokens
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
}

ELEVENLABS_PRICING = {
    # Per 1,000 characters
    "starter": 0.30 / 1000,      # ~30 cents per 1000 chars
    "creator": 0.30 / 1000,
    "pro": 0.24 / 1000,
    "scale": 0.18 / 1000,
    "standard": 0.30 / 1000,     # Default tier
}

# Default budget limits
DEFAULT_DAILY_LIMIT = 10.0      # $10/day
DEFAULT_MONTHLY_LIMIT = 100.0   # $100/month
DEFAULT_ALERT_THRESHOLD = 0.8  # Alert at 80% of limit


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class APICall:
    """Record of a single API call."""
    timestamp: str
    service: str
    model: str
    cost_usd: float
    usage: Dict[str, Any]
    details: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DailyStats:
    """Daily usage statistics."""
    date: str
    total_cost: float = 0.0
    openai_cost: float = 0.0
    elevenlabs_cost: float = 0.0
    openai_tokens: int = 0
    elevenlabs_chars: int = 0
    call_count: int = 0
    calls: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Cost Tracker
# =============================================================================

class CostTracker:
    """
    Thread-safe singleton for tracking API costs.

    Features:
    - Automatic daily/monthly reset
    - Persistent storage
    - Budget alerts
    - Detailed summaries
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        daily_limit: float = DEFAULT_DAILY_LIMIT,
        monthly_limit: float = DEFAULT_MONTHLY_LIMIT,
        alert_callback: Optional[Callable[[str, float, float], None]] = None
    ):
        """
        Initialize the cost tracker.

        Args:
            storage_path: Path to store cost data (default: logs/cost_data.json)
            daily_limit: Daily budget limit in USD
            monthly_limit: Monthly budget limit in USD
            alert_callback: Function to call when approaching limits
                            Signature: (alert_type, current, limit) -> None
        """
        if self._initialized:
            return

        self.logger = get_cost_logger()
        self.storage_path = storage_path or Path("logs/cost_data.json")
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.alert_callback = alert_callback
        self.alert_threshold = DEFAULT_ALERT_THRESHOLD

        # Current period tracking
        self._current_date: str = ""
        self._current_month: str = ""
        self._daily_stats: DailyStats = DailyStats(date="")
        self._monthly_cost: float = 0.0
        self._monthly_breakdown: Dict[str, float] = {}

        # Alert tracking (avoid duplicate alerts)
        self._daily_alert_sent = False
        self._monthly_alert_sent = False

        # Load existing data
        self._load_data()

        self._initialized = True

    # =========================================================================
    # Public API
    # =========================================================================

    def log_openai_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4o"
    ) -> float:
        """
        Log an OpenAI API call and calculate cost.

        Args:
            input_tokens: Number of input (prompt) tokens
            output_tokens: Number of output (completion) tokens
            model: Model name (gpt-4o, gpt-4o-mini, gpt-3.5-turbo, etc.)

        Returns:
            Cost of this call in USD

        Raises:
            ValueError: If model is not recognized
        """
        with self._lock:
            self._check_and_reset_periods()

            pricing = OPENAI_PRICING.get(model)
            if not pricing:
                raise ValueError(
                    f"Unknown OpenAI model: {model}. "
                    f"Known models: {list(OPENAI_PRICING.keys())}"
                )

            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            output_cost = (output_tokens / 1_000_000) * pricing["output"]
            total_cost = input_cost + output_cost

            # Record the call
            call = APICall(
                timestamp=datetime.now(timezone.utc).isoformat(),
                service="openai",
                model=model,
                cost_usd=round(total_cost, 6),
                usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                details=f"Input: {input_tokens}, Output: {output_tokens} tokens"
            )

            self._record_call(call)
            self._daily_stats.openai_cost += total_cost
            self._daily_stats.openai_tokens += input_tokens + output_tokens

            # Log to file
            self.logger.info(
                "OpenAI API call",
                extra={
                    "service": "openai",
                    "model": model,
                    "cost_usd": round(total_cost, 6),
                    "usage": call.usage,
                    "details": call.details
                }
            )

            self._check_alerts()
            self._save_data()

            return total_cost

    def log_elevenlabs_call(
        self,
        characters: int,
        model: str = "standard"
    ) -> float:
        """
        Log an ElevenLabs TTS API call and calculate cost.

        Args:
            characters: Number of characters synthesized
            model: Pricing tier (starter, creator, pro, scale, standard)

        Returns:
            Cost of this call in USD

        Raises:
            ValueError: If model/tier is not recognized
        """
        with self._lock:
            self._check_and_reset_periods()

            price_per_char = ELEVENLABS_PRICING.get(model)
            if price_per_char is None:
                raise ValueError(
                    f"Unknown ElevenLabs tier: {model}. "
                    f"Known tiers: {list(ELEVENLABS_PRICING.keys())}"
                )

            total_cost = characters * price_per_char

            call = APICall(
                timestamp=datetime.now(timezone.utc).isoformat(),
                service="elevenlabs",
                model=model,
                cost_usd=round(total_cost, 6),
                usage={"characters": characters},
                details=f"{characters} characters synthesized"
            )

            self._record_call(call)
            self._daily_stats.elevenlabs_cost += total_cost
            self._daily_stats.elevenlabs_chars += characters

            self.logger.info(
                "ElevenLabs API call",
                extra={
                    "service": "elevenlabs",
                    "model": model,
                    "cost_usd": round(total_cost, 6),
                    "usage": call.usage,
                    "details": call.details
                }
            )

            self._check_alerts()
            self._save_data()

            return total_cost

    def get_daily_total(self) -> float:
        """Get total cost for today in USD."""
        with self._lock:
            self._check_and_reset_periods()
            return round(self._daily_stats.total_cost, 4)

    def get_monthly_total(self) -> float:
        """Get total cost for this month in USD."""
        with self._lock:
            self._check_and_reset_periods()
            return round(self._monthly_cost, 4)

    def get_daily_summary(self) -> Dict[str, Any]:
        """
        Get detailed daily usage summary.

        Returns:
            Dictionary with cost breakdown, usage stats, and remaining budget
        """
        with self._lock:
            self._check_and_reset_periods()
            stats = self._daily_stats

            return {
                "date": stats.date,
                "total_cost": round(stats.total_cost, 4),
                "breakdown": {
                    "openai": round(stats.openai_cost, 4),
                    "elevenlabs": round(stats.elevenlabs_cost, 4),
                },
                "usage": {
                    "openai_tokens": stats.openai_tokens,
                    "elevenlabs_characters": stats.elevenlabs_chars,
                    "api_calls": stats.call_count,
                },
                "budget": {
                    "daily_limit": self.daily_limit,
                    "remaining": round(self.daily_limit - stats.total_cost, 4),
                    "percent_used": round((stats.total_cost / self.daily_limit) * 100, 1),
                }
            }

    def get_monthly_summary(self) -> Dict[str, Any]:
        """
        Get detailed monthly usage summary.

        Returns:
            Dictionary with cost breakdown and remaining budget
        """
        with self._lock:
            self._check_and_reset_periods()

            return {
                "month": self._current_month,
                "total_cost": round(self._monthly_cost, 4),
                "breakdown": dict(self._monthly_breakdown),
                "budget": {
                    "monthly_limit": self.monthly_limit,
                    "remaining": round(self.monthly_limit - self._monthly_cost, 4),
                    "percent_used": round(
                        (self._monthly_cost / self.monthly_limit) * 100, 1
                    ),
                }
            }

    def get_remaining_budget(self) -> Dict[str, float]:
        """Get remaining budget for daily and monthly limits."""
        with self._lock:
            self._check_and_reset_periods()
            return {
                "daily": round(self.daily_limit - self._daily_stats.total_cost, 4),
                "monthly": round(self.monthly_limit - self._monthly_cost, 4),
            }

    def is_within_budget(self, estimated_cost: float = 0.0) -> bool:
        """
        Check if we're within budget (with optional estimated cost).

        Args:
            estimated_cost: Additional cost to consider

        Returns:
            True if within both daily and monthly limits
        """
        with self._lock:
            self._check_and_reset_periods()
            daily_ok = (self._daily_stats.total_cost + estimated_cost) <= self.daily_limit
            monthly_ok = (self._monthly_cost + estimated_cost) <= self.monthly_limit
            return daily_ok and monthly_ok

    def set_limits(self, daily: Optional[float] = None, monthly: Optional[float] = None):
        """Update budget limits."""
        with self._lock:
            if daily is not None:
                self.daily_limit = daily
            if monthly is not None:
                self.monthly_limit = monthly

    def format_summary(self) -> str:
        """
        Get a formatted string summary of costs.

        Returns:
            Human-readable cost summary
        """
        daily = self.get_daily_summary()
        monthly = self.get_monthly_summary()

        lines = [
            f"=== Cost Summary ({daily['date']}) ===",
            f"",
            f"TODAY:",
            f"  Total: ${daily['total_cost']:.4f}",
            f"  OpenAI: ${daily['breakdown']['openai']:.4f} ({daily['usage']['openai_tokens']:,} tokens)",
            f"  ElevenLabs: ${daily['breakdown']['elevenlabs']:.4f} ({daily['usage']['elevenlabs_characters']:,} chars)",
            f"  Remaining: ${daily['budget']['remaining']:.4f} ({100 - daily['budget']['percent_used']:.1f}%)",
            f"",
            f"THIS MONTH ({monthly['month']}):",
            f"  Total: ${monthly['total_cost']:.4f}",
            f"  Remaining: ${monthly['budget']['remaining']:.4f} ({100 - monthly['budget']['percent_used']:.1f}%)",
        ]
        return "\n".join(lines)

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _check_and_reset_periods(self):
        """Reset daily/monthly totals if the period has changed."""
        now = datetime.now(timezone.utc)
        current_date = now.strftime("%Y-%m-%d")
        current_month = now.strftime("%Y-%m")

        # Check for month change
        if current_month != self._current_month:
            self._monthly_cost = 0.0
            self._monthly_breakdown = {}
            self._current_month = current_month
            self._monthly_alert_sent = False

        # Check for day change
        if current_date != self._current_date:
            self._daily_stats = DailyStats(date=current_date)
            self._current_date = current_date
            self._daily_alert_sent = False

    def _record_call(self, call: APICall):
        """Record an API call in the current stats."""
        self._daily_stats.total_cost += call.cost_usd
        self._daily_stats.call_count += 1
        self._daily_stats.calls.append(call.to_dict())

        self._monthly_cost += call.cost_usd

        # Update monthly breakdown by service
        service = call.service
        if service not in self._monthly_breakdown:
            self._monthly_breakdown[service] = 0.0
        self._monthly_breakdown[service] += call.cost_usd

    def _check_alerts(self):
        """Check if budget alerts should be triggered."""
        # Daily alert
        daily_percent = self._daily_stats.total_cost / self.daily_limit
        if daily_percent >= self.alert_threshold and not self._daily_alert_sent:
            self._send_alert("daily", self._daily_stats.total_cost, self.daily_limit)
            self._daily_alert_sent = True

        # Monthly alert
        monthly_percent = self._monthly_cost / self.monthly_limit
        if monthly_percent >= self.alert_threshold and not self._monthly_alert_sent:
            self._send_alert("monthly", self._monthly_cost, self.monthly_limit)
            self._monthly_alert_sent = True

    def _send_alert(self, alert_type: str, current: float, limit: float):
        """Send a budget alert."""
        percent = (current / limit) * 100
        message = (
            f"BUDGET ALERT: {alert_type.upper()} spending at {percent:.1f}% "
            f"(${current:.2f} / ${limit:.2f})"
        )

        # Log the alert
        self.logger.warning(
            message,
            extra={
                "service": "cost_tracker",
                "model": "alert",
                "cost_usd": current,
                "usage": {"type": alert_type, "limit": limit},
                "details": message
            }
        )

        # Use warnings module for visibility
        warnings.warn(message, UserWarning)

        # Call custom callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert_type, current, limit)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

    def _save_data(self):
        """Persist current stats to storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "current_date": self._current_date,
                "current_month": self._current_month,
                "daily_stats": self._daily_stats.to_dict(),
                "monthly_cost": self._monthly_cost,
                "monthly_breakdown": self._monthly_breakdown,
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save cost data: {e}")

    def _load_data(self):
        """Load existing stats from storage."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path) as f:
                    data = json.load(f)

                # Only load if data is from today/this month
                now = datetime.now(timezone.utc)
                current_date = now.strftime("%Y-%m-%d")
                current_month = now.strftime("%Y-%m")

                stored_date = data.get("current_date", "")
                stored_month = data.get("current_month", "")

                # Load monthly data if same month
                if stored_month == current_month:
                    self._monthly_cost = data.get("monthly_cost", 0.0)
                    self._monthly_breakdown = data.get("monthly_breakdown", {})
                    self._current_month = current_month

                # Load daily data if same day
                if stored_date == current_date:
                    daily_data = data.get("daily_stats", {})
                    self._daily_stats = DailyStats(
                        date=daily_data.get("date", current_date),
                        total_cost=daily_data.get("total_cost", 0.0),
                        openai_cost=daily_data.get("openai_cost", 0.0),
                        elevenlabs_cost=daily_data.get("elevenlabs_cost", 0.0),
                        openai_tokens=daily_data.get("openai_tokens", 0),
                        elevenlabs_chars=daily_data.get("elevenlabs_chars", 0),
                        call_count=daily_data.get("call_count", 0),
                        calls=daily_data.get("calls", [])
                    )
                    self._current_date = current_date
                else:
                    self._daily_stats = DailyStats(date=current_date)
                    self._current_date = current_date

        except Exception as e:
            # Start fresh if loading fails
            now = datetime.now(timezone.utc)
            self._current_date = now.strftime("%Y-%m-%d")
            self._current_month = now.strftime("%Y-%m")
            self._daily_stats = DailyStats(date=self._current_date)
            self._monthly_cost = 0.0
            self._monthly_breakdown = {}


# =============================================================================
# Convenience Functions
# =============================================================================

def get_cost_tracker() -> CostTracker:
    """Get the singleton cost tracker instance."""
    return CostTracker()


def estimate_openai_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o"
) -> float:
    """
    Estimate cost for an OpenAI call without logging it.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name

    Returns:
        Estimated cost in USD
    """
    pricing = OPENAI_PRICING.get(model)
    if not pricing:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def estimate_elevenlabs_cost(characters: int, model: str = "standard") -> float:
    """
    Estimate cost for an ElevenLabs call without logging it.

    Args:
        characters: Number of characters
        model: Pricing tier

    Returns:
        Estimated cost in USD
    """
    price_per_char = ELEVENLABS_PRICING.get(model, 0.0)
    return characters * price_per_char
