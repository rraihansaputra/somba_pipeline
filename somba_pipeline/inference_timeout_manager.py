"""
Inference timeout manager for motion-triggered inference optimization.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class InferenceDecision:
    """Decision about whether to run inference."""
    should_infer: bool
    reason: str
    confidence: float = 1.0


class InferenceTimeoutManager:
    """Manages timeout-based inference fallback and adaptive inference strategies."""

    def __init__(
        self,
        camera_uuid: str,
        timeout_seconds: float = 30.0,
        min_interval_seconds: float = 1.0,
        adaptive_timeout: bool = True,
    ):
        self.camera_uuid = camera_uuid
        self.timeout_seconds = timeout_seconds
        self.min_interval_seconds = min_interval_seconds
        self.adaptive_timeout = adaptive_timeout

        # State tracking
        self.last_inference_time: Optional[datetime] = None
        self.last_motion_time: Optional[datetime] = None
        self.inference_count = 0
        self.motion_count = 0

        # Adaptive parameters
        self.current_timeout = timeout_seconds
        self.min_timeout = 5.0  # Minimum timeout
        self.max_timeout = 120.0  # Maximum timeout

        # Performance tracking
        self.recent_inference_intervals = []
        self.recent_motion_rates = []

        logger.info(f"InferenceTimeoutManager initialized for {camera_uuid}")
        logger.info(f"Base timeout: {timeout_seconds}s, min interval: {min_interval_seconds}s")

    def should_trigger_inference(self, motion_detected: bool) -> InferenceDecision:
        """
        Decide whether to trigger inference based on motion and timeout.

        Args:
            motion_detected: Whether motion was detected in the current frame

        Returns:
            InferenceDecision with boolean decision and reason
        """
        current_time = datetime.now()
        time_since_last = (current_time - self.last_inference_time).total_seconds() if self.last_inference_time else float('inf')

        # Always infer if motion detected (highest priority)
        if motion_detected:
            self.last_motion_time = current_time
            self.motion_count += 1
            self._update_last_inference_time(current_time)
            self._update_adaptive_timeout(True)

            logger.debug(f"Timeout mgr {self.camera_uuid}: Trigger INFER (motion_detected), time_since_last={time_since_last:.1f}s")
            return InferenceDecision(
                should_infer=True,
                reason="motion_detected",
                confidence=1.0
            )

        # Check if we're in the minimum interval cooldown, but allow timeout to override
        # Only apply cooldown if we haven't reached timeout AND we're not in a timeout-triggered inference window
        timeout_triggered_flag = getattr(self, '_timeout_triggered', False)

        if (
            self.last_inference_time is not None
            and time_since_last < self.min_interval_seconds
            and time_since_last < self.current_timeout  # Only apply cooldown if we haven't reached timeout
            and not timeout_triggered_flag  # Don't apply cooldown if timeout just triggered
        ):
            logger.debug(f"Timeout mgr {self.camera_uuid}: SKIP (min_interval_cooldown), time_since_last={time_since_last:.1f}s < {self.min_interval_seconds}s")
            return InferenceDecision(
                should_infer=False,
                reason="min_interval_cooldown",
                confidence=1.0
            )

        # Clear timeout triggered flag after first check
        if timeout_triggered_flag:
            self._timeout_triggered = False

        # Check timeout-based inference (fallback)
        if (
            self.last_inference_time is not None
            and time_since_last >= self.current_timeout
        ):
            self._update_last_inference_time(current_time)
            self._update_adaptive_timeout(False)

            # Set timeout triggered flag to prevent immediate cooldown
            self._timeout_triggered = True

            logger.debug(f"Timeout mgr {self.camera_uuid}: Trigger INFER (timeout_{self.current_timeout:.1f}s), time_since_last={time_since_last:.1f}s")
            return InferenceDecision(
                should_infer=True,
                reason=f"timeout_{self.current_timeout:.1f}s",
                confidence=0.8
            )

        # No inference needed
        logger.debug(f"Timeout mgr {self.camera_uuid}: SKIP (no_motion_no_timeout), time_since_last={time_since_last:.1f}s < {self.current_timeout}s")
        return InferenceDecision(
            should_infer=False,
            reason="no_motion_no_timeout",
            confidence=1.0
        )

    def _update_last_inference_time(self, current_time: datetime):
        """Update last inference time and track performance metrics."""
        if self.last_inference_time is not None:
            interval = (current_time - self.last_inference_time).total_seconds()
            self.recent_inference_intervals.append(interval)

            # Keep only recent intervals (last 20)
            if len(self.recent_inference_intervals) > 20:
                self.recent_inference_intervals.pop(0)

        self.last_inference_time = current_time
        self.inference_count += 1

    def _update_adaptive_timeout(self, motion_triggered: bool):
        """Update timeout based on recent activity patterns."""
        if not self.adaptive_timeout:
            return

        current_time = datetime.now()

        # Calculate recent motion rate
        if self.last_motion_time is not None:
            time_since_motion = (current_time - self.last_motion_time).total_seconds()
            self.recent_motion_rates.append(1.0 if time_since_motion < 60.0 else 0.0)

            # Keep only recent rates (last 30)
            if len(self.recent_motion_rates) > 30:
                self.recent_motion_rates.pop(0)

        # Adaptive timeout logic
        if len(self.recent_motion_rates) >= 10:
            recent_motion_rate = sum(self.recent_motion_rates[-10:]) / 10.0

            if recent_motion_rate > 0.7:  # High activity area
                self.current_timeout = max(self.min_timeout, self.current_timeout * 0.9)
            elif recent_motion_rate < 0.1:  # Low activity area
                self.current_timeout = min(self.max_timeout, self.current_timeout * 1.1)

            logger.debug(
                f"Adaptive timeout for {self.camera_uuid}: "
                f"{self.current_timeout:.1f}s (motion rate: {recent_motion_rate:.2f})"
            )

    def get_stats(self) -> dict:
        """Get timeout manager statistics."""
        current_time = datetime.now()

        stats = {
            "inference_count": self.inference_count,
            "motion_count": self.motion_count,
            "current_timeout": self.current_timeout,
            "base_timeout": self.timeout_seconds,
            "min_interval": self.min_interval_seconds,
            "adaptive_enabled": self.adaptive_timeout,
        }

        # Time since last inference
        if self.last_inference_time is not None:
            stats["time_since_last_inference"] = (current_time - self.last_inference_time).total_seconds()

        # Time since last motion
        if self.last_motion_time is not None:
            stats["time_since_last_motion"] = (current_time - self.last_motion_time).total_seconds()

        # Average inference interval
        if self.recent_inference_intervals:
            stats["avg_inference_interval"] = sum(self.recent_inference_intervals) / len(self.recent_inference_intervals)

        # Recent motion rate
        if self.recent_motion_rates:
            stats["recent_motion_rate"] = sum(self.recent_motion_rates) / len(self.recent_motion_rates)

        return stats

    def reset_stats(self):
        """Reset timeout manager statistics."""
        self.inference_count = 0
        self.motion_count = 0
        self.recent_inference_intervals.clear()
        self.recent_motion_rates.clear()
        self.current_timeout = self.timeout_seconds

    def force_inference(self) -> InferenceDecision:
        """Force inference to run (for testing or manual triggers)."""
        current_time = datetime.now()
        self._update_last_inference_time(current_time)

        return InferenceDecision(
            should_infer=True,
            reason="forced",
            confidence=1.0
        )

    def set_timeout(self, timeout_seconds: float):
        """Update the timeout value."""
        self.timeout_seconds = max(self.min_timeout, min(self.max_timeout, timeout_seconds))
        self.current_timeout = self.timeout_seconds

        logger.info(f"Timeout updated for {self.camera_uuid}: {self.timeout_seconds}s")

    def get_remaining_timeout(self) -> float:
        """Get remaining time until timeout-based inference."""
        if self.last_inference_time is None:
            return 0.0

        current_time = datetime.now()
        elapsed = (current_time - self.last_inference_time).total_seconds()
        remaining = max(0.0, self.current_timeout - elapsed)

        return remaining
