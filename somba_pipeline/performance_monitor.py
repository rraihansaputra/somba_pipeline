"""
Performance monitoring for adaptive inference optimization.
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single time window."""
    timestamp: datetime
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    inference_rate: float = 0.0
    frame_rate: float = 0.0
    motion_rate: float = 0.0
    latency_ms: float = 0.0
    queue_depth: int = 0


class PerformanceMonitor:
    """Monitors system performance and provides adaptive recommendations."""
    
    def __init__(
        self,
        camera_uuid: str,
        window_size: int = 60,  # 1 minute window
        sample_interval: float = 1.0,  # 1 second samples
    ):
        self.camera_uuid = camera_uuid
        self.window_size = window_size
        self.sample_interval = sample_interval
        
        # Performance history (circular buffer)
        self.metrics_history = deque(maxlen=window_size)
        
        # Current metrics
        self.current_metrics = PerformanceMetrics(timestamp=datetime.now())
        
        # Aggregated statistics
        self.stats = {
            "avg_cpu": 0.0,
            "avg_memory": 0.0,
            "avg_inference_rate": 0.0,
            "avg_frame_rate": 0.0,
            "avg_latency": 0.0,
            "max_cpu": 0.0,
            "max_memory": 0.0,
            "max_latency": 0.0,
            "cpu_trend": 0.0,  # Positive = increasing, negative = decreasing
            "memory_trend": 0.0,
            "latency_trend": 0.0,
        }
        
        # Thresholds for adaptive behavior
        self.thresholds = {
            "cpu_high": 80.0,      # CPU usage percentage
            "cpu_critical": 90.0,  # CPU usage percentage
            "memory_high": 80.0,   # Memory usage percentage
            "memory_critical": 90.0,  # Memory usage percentage
            "latency_high": 100.0,  # Latency in milliseconds
            "latency_critical": 200.0,  # Latency in milliseconds
        }
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        logger.info(f"PerformanceMonitor initialized for {camera_uuid}")
    
    def start_monitoring(self):
        """Start performance monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Performance monitoring started for {self.camera_uuid}")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info(f"Performance monitoring stopped for {self.camera_uuid}")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                self._collect_metrics()
                self._update_stats()
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(self.sample_interval)
    
    def _collect_metrics(self):
        """Collect current performance metrics."""
        import psutil
        import os
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Process-specific metrics
            process = psutil.Process()
            process_cpu = process.cpu_percent()
            process_memory = process.memory_percent()
            
            # Create new metrics snapshot
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_percent,
                memory_usage=memory_percent,
                inference_rate=self.current_metrics.inference_rate,
                frame_rate=self.current_metrics.frame_rate,
                motion_rate=self.current_metrics.motion_rate,
                latency_ms=self.current_metrics.latency_ms,
                queue_depth=self.current_metrics.queue_depth,
            )
            
            self.metrics_history.append(metrics)
            self.current_metrics = metrics
            
        except Exception as e:
            logger.debug(f"Error collecting metrics: {e}")
    
    def _update_stats(self):
        """Update aggregated statistics."""
        if len(self.metrics_history) < 2:
            return
        
        # Calculate averages
        recent_metrics = list(self.metrics_history)[-30:]  # Last 30 samples
        
        self.stats["avg_cpu"] = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        self.stats["avg_memory"] = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        self.stats["avg_inference_rate"] = sum(m.inference_rate for m in recent_metrics) / len(recent_metrics)
        self.stats["avg_frame_rate"] = sum(m.frame_rate for m in recent_metrics) / len(recent_metrics)
        self.stats["avg_latency"] = sum(m.latency_ms for m in recent_metrics) / len(recent_metrics)
        
        # Calculate maximums
        self.stats["max_cpu"] = max(m.cpu_usage for m in recent_metrics)
        self.stats["max_memory"] = max(m.memory_usage for m in recent_metrics)
        self.stats["max_latency"] = max(m.latency_ms for m in recent_metrics)
        
        # Calculate trends (simple linear regression)
        if len(recent_metrics) >= 10:
            self._calculate_trends(recent_metrics[-10:])
    
    def _calculate_trends(self, recent_metrics: List[PerformanceMetrics]):
        """Calculate trend indicators using simple linear regression."""
        if len(recent_metrics) < 2:
            return
        
        n = len(recent_metrics)
        x_values = list(range(n))
        
        # CPU trend
        y_cpu = [m.cpu_usage for m in recent_metrics]
        self.stats["cpu_trend"] = self._simple_linear_regression(x_values, y_cpu)
        
        # Memory trend
        y_memory = [m.memory_usage for m in recent_metrics]
        self.stats["memory_trend"] = self._simple_linear_regression(x_values, y_memory)
        
        # Latency trend
        y_latency = [m.latency_ms for m in recent_metrics]
        self.stats["latency_trend"] = self._simple_linear_regression(x_values, y_latency)
    
    def _simple_linear_regression(self, x: List[int], y: List[float]) -> float:
        """Calculate slope using simple linear regression."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        return slope
    
    def update_inference_metrics(self, inference_rate: float, frame_rate: float, motion_rate: float):
        """Update inference-related metrics."""
        self.current_metrics.inference_rate = inference_rate
        self.current_metrics.frame_rate = frame_rate
        self.current_metrics.motion_rate = motion_rate
    
    def update_latency(self, latency_ms: float):
        """Update latency metrics."""
        self.current_metrics.latency_ms = latency_ms
    
    def update_queue_depth(self, queue_depth: int):
        """Update queue depth metrics."""
        self.current_metrics.queue_depth = queue_depth
    
    def is_overloaded(self) -> bool:
        """Check if system is overloaded based on current metrics."""
        return (
            self.current_metrics.cpu_usage > self.thresholds["cpu_critical"] or
            self.current_metrics.memory_usage > self.thresholds["memory_critical"] or
            self.current_metrics.latency_ms > self.thresholds["latency_critical"]
        )
    
    def is_stressed(self) -> bool:
        """Check if system is under stress (high but not critical)."""
        return (
            self.current_metrics.cpu_usage > self.thresholds["cpu_high"] or
            self.current_metrics.memory_usage > self.thresholds["memory_high"] or
            self.current_metrics.latency_ms > self.thresholds["latency_high"]
        )
    
    def get_load_level(self) -> str:
        """Get current load level."""
        if self.is_overloaded():
            return "critical"
        elif self.is_stressed():
            return "high"
        elif self.current_metrics.cpu_usage > 50 or self.current_metrics.memory_usage > 50:
            return "medium"
        else:
            return "low"
    
    def get_adaptive_recommendation(self) -> Dict[str, Any]:
        """Get adaptive recommendations based on current performance."""
        load_level = self.get_load_level()
        
        recommendation = {
            "load_level": load_level,
            "current_metrics": {
                "cpu": self.current_metrics.cpu_usage,
                "memory": self.current_metrics.memory_usage,
                "latency": self.current_metrics.latency_ms,
            },
            "recommendations": [],
        }
        
        if load_level == "critical":
            recommendation["recommendations"].extend([
                "reduce_inference_rate",
                "increase_motion_threshold",
                "skip_non_critical_frames",
            ])
        elif load_level == "high":
            recommendation["recommendations"].extend([
                "moderate_inference_rate",
                "consider_motion_threshold_adjustment",
            ])
        elif load_level == "low":
            recommendation["recommendations"].extend([
                "normal_operation",
                "can_increase_sensitivity",
            ])
        
        # Trend-based recommendations
        if self.stats["cpu_trend"] > 2.0:  # CPU rapidly increasing
            recommendation["recommendations"].append("monitor_cpu_trend")
        
        if self.stats["memory_trend"] > 1.0:  # Memory rapidly increasing
            recommendation["recommendations"].append("monitor_memory_trend")
        
        return recommendation
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "current_metrics": {
                "cpu_usage": self.current_metrics.cpu_usage,
                "memory_usage": self.current_metrics.memory_usage,
                "inference_rate": self.current_metrics.inference_rate,
                "frame_rate": self.current_metrics.frame_rate,
                "motion_rate": self.current_metrics.motion_rate,
                "latency_ms": self.current_metrics.latency_ms,
                "queue_depth": self.current_metrics.queue_depth,
            },
            "aggregated_stats": self.stats.copy(),
            "thresholds": self.thresholds.copy(),
            "load_level": self.get_load_level(),
            "monitoring_active": self.monitoring,
            "history_size": len(self.metrics_history),
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.metrics_history.clear()
        self.current_metrics = PerformanceMetrics(timestamp=datetime.now())
        
        for key in self.stats:
            self.stats[key] = 0.0
    
    def set_threshold(self, threshold_name: str, value: float):
        """Update a performance threshold."""
        if threshold_name in self.thresholds:
            self.thresholds[threshold_name] = value
            logger.info(f"Updated threshold {threshold_name} to {value} for {self.camera_uuid}")
    
    def get_history(self, duration_seconds: int = 60) -> List[Dict[str, Any]]:
        """Get performance history for the specified duration."""
        cutoff_time = datetime.now() - timedelta(seconds=duration_seconds)
        
        history = []
        for metrics in self.metrics_history:
            if metrics.timestamp >= cutoff_time:
                history.append({
                    "timestamp": metrics.timestamp.isoformat(),
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "inference_rate": metrics.inference_rate,
                    "frame_rate": metrics.frame_rate,
                    "motion_rate": metrics.motion_rate,
                    "latency_ms": metrics.latency_ms,
                    "queue_depth": metrics.queue_depth,
                })
        
        return history