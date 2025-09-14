"""
Production worker implementing the full Phase 2 specifications.
Wraps InferencePipeline with RabbitMQ, Prometheus, motion detection, and zones.
"""
import os
os.environ["OPENVINO_FORCE"]="0"                # <— force-reinsert OV even if the model removed it
os.environ["OPENVINO_DEVICE_TYPE"]="GPU"
os.environ["OPENVINO_PRECISION"]="FP16"
# os.environ["OPENVINO_PRECISION"]="INT8"
os.environ["OPENVINO_NUM_STREAMS"]="8"
os.environ["OPENVINO_CACHE_DIR"]="/tmp/ov_cache"
os.environ["OV_PATCH_VERBOSE"]="1"
os.environ["ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING"]="True"
os.environ["ENABLE_WORKFLOWS_PROFILING"]="False"

import somba_pipeline.ov_ep_patch as ov_ep_patch
ov_ep_patch.enable_openvino_gpu()

import asyncio
import json
import logging
import signal
import threading
import time
import ulid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import asdict
# import os
import hashlib

import aio_pika
import cv2
import numpy as np
from aiohttp import web
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Import InferencePipeline and related classes
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline, SinkMode
from inference.core.interfaces.camera.entities import (
    VideoFrame,
    StatusUpdate,
    UpdateSeverity,
)

from .schemas import (
    ShardConfig,
    DetectionEvent,
    StatusEvent,
    ErrorEvent,
    ModelInfo,
    FrameInfo,
    ZonesConfigInfo,
    CameraConfig,
    DetectionLatency,
    ZonesStats,
    ZoneStats,
)
from .zone_attribution import MultiCameraZoneAttributor
from .debug_vis import DebugManager
from .overlays import render_debug_overlay
from .scaling import resolve_proc_size
from .zones import ZoneMaskBuilder
from .zero_copy_motion_detector import MotionData
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    format="%(asctime)s %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ProductionWorker:
    """
    Production worker implementing the full technical specification.
    Wraps InferencePipeline with RabbitMQ, Prometheus, motion detection, and zones.
    """

    def __init__(self, config: ShardConfig, config_path: Optional[str] = None):
        self.config = config
        self.config_path = config_path
        self.config_hash = self._calculate_config_hash() if config_path else None
        self.pipeline: Optional[InferencePipeline] = None  # InferencePipeline instance

        # State tracking
        self.running = True
        self.ready = False
        self.draining = False
        self.camera_states: Dict[str, str] = {}  # camera_uuid -> state
        self.last_frame_times: Dict[str, datetime] = {}  # camera_uuid -> timestamp
        self.frame_counts: Dict[str, int] = {}  # camera_uuid -> frame_count

        # Config watching
        self.config_watcher_task: Optional[asyncio.Task] = None

        # Lease awareness (Phase 4)
        self.lease_id: Optional[str] = None
        self.worker_id: Optional[str] = None
        self.lease_manager = None  # Will be set by manager
        self.config_sync = None  # Will be set by manager
        self.lease_heartbeat_task: Optional[asyncio.Task] = None

        # Get frame dimensions (would normally come from first frame)
        self.frame_width = 1920  # Default, should be updated from actual frames
        self.frame_height = 1080

        # Initialize zone attribution
        camera_configs = {
            source["camera_uuid"]: self.config.cameras.get(
                source["camera_uuid"], CameraConfig(camera_uuid=source["camera_uuid"])
            )
            for source in self.config.sources
        }
        self.zone_attributor = MultiCameraZoneAttributor(
            camera_configs, self.frame_width, self.frame_height
        )

        # Initialize optimized motion detection components
        from .zero_copy_motion_detector import ZeroCopyMotionDetector
        from .inference_timeout_manager import InferenceTimeoutManager
        from .performance_monitor import PerformanceMonitor

        # Optimized motion detection components
        self.optimized_motion_detectors: Dict[str, ZeroCopyMotionDetector] = {}
        self.timeout_managers: Dict[str, InferenceTimeoutManager] = {}
        self.performance_monitors: Dict[str, PerformanceMonitor] = {}

        # Optimization statistics
        self.inference_decisions = {
            "motion_triggered": 0,
            "timeout_triggered": 0,
            "skipped_no_motion": 0,
            "skipped_cooldown": 0,
        }

        for camera_uuid, camera_config in camera_configs.items():
            if camera_config.motion_gating.enabled:
                # Initialize optimized zero-copy motion detector
                self.optimized_motion_detectors[camera_uuid] = ZeroCopyMotionDetector(
                    camera_uuid=camera_uuid,
                    motion_config=camera_config.motion_gating,
                    zones=camera_config.zones,
                    frame_width=self.frame_width,
                    frame_height=self.frame_height,
                )

                # Initialize timeout manager
                timeout_seconds = getattr(
                    camera_config.motion_gating,
                    'max_inference_interval_seconds',
                    30.0
                )
                self.timeout_managers[camera_uuid] = InferenceTimeoutManager(
                    camera_uuid=camera_uuid,
                    timeout_seconds=timeout_seconds,
                    min_interval_seconds=1.0,
                    adaptive_timeout=True,
                )

                # Initialize performance monitor
                self.performance_monitors[camera_uuid] = PerformanceMonitor(
                    camera_uuid=camera_uuid,
                    window_size=60,
                    sample_interval=1.0,
                )
                self.performance_monitors[camera_uuid].start_monitoring()

        # Initialize metrics
        self._init_metrics()


        # setup debug manager
        self.debug_mgr = DebugManager()

        # Per-camera metadata for debug
        self._camera_meta: Dict[str, dict] = {}
        self._motion_state: Dict[str, dict] = {}  # {cam: {"on": bool, "mask": np.ndarray, "contours": [...], "zone_hits": [...]} }
        self._motion_debug: Dict[str, dict] = {}  # {cam: {"mask": np.ndarray, "contours": [...], "active": bool, "cooldown": int, "gate_reason": str, "min_area_px": int}}
        self._debug_rate: Dict[str, float] = {}   # {cam: last_publish_ts}
        self._native_res: Dict[str, Tuple[int, int]] = {}  # store native resolution per camera
        self._last_infer_state: Dict[str, str] = {}  # {cam: "INFER"|"SKIP"|"COOLDOWN"}
        self._last_detections: Dict[str, List[dict]] = {}  # {cam: [last detections]}
        self._debug_target_fps: int = int(os.getenv("DEBUG_FPS", "8"))  # throttle; 8 fps is plenty
        # FPS tracking (EMA per camera)
        self._fps_last_ts: Dict[str, float] = {}
        self._fps_ema: Dict[str, float] = {}

        # When True, we have installed a motion-gating wrapper over the
        # pipeline's on_video_frame callable to avoid wasted inference.
        # Used to guard metrics updates and skip duplicate gating in sinks.
        self._use_motion_gating_wrapper: bool = False

        # Event queues
        self.detection_queue = asyncio.Queue(maxsize=1000)
        self.status_queue = asyncio.Queue(maxsize=100)
        # Frame batch queue and concurrency controls for async processing
        self.frame_batch_queue: Optional[asyncio.Queue] = None
        self._frame_concurrency: int = int(os.getenv("WORKER_FRAME_CONCURRENCY", "4"))
        self._frame_semaphore: Optional[asyncio.Semaphore] = None

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Threading
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.event_thread: Optional[threading.Thread] = None

        logger.info(
            f"Worker initialized: runner={config.runner_id}, shard={config.shard_id}"
        )
        logger.info(f"Processing {len(config.sources)} cameras")

        # Start config watching if config path is provided
        if self.config_path:
            logger.info(f"Configuration hot-reload enabled for: {self.config_path}")

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        # Gauges
        # Camera-level gauges (include tenant/site)
        self.stream_up = Gauge(
            "stream_up", "Stream connectivity status", ["tenant_id", "site_id", "camera_uuid"]
        )
        self.last_frame_age = Gauge(
            "last_frame_age_seconds", "Age of last frame", ["tenant_id", "site_id", "camera_uuid"]
        )
        self.stream_fps = Gauge(
            "stream_fps", "Stream FPS (actual)", ["tenant_id", "site_id", "camera_uuid"]
        )
        self.pipeline_fps = Gauge(
            "pipeline_fps", "Pipeline FPS", ["runner_id", "shard_id"]
        )

        # Histograms
        self.inference_latency = Histogram(
            "inference_latency_seconds", "Inference latency", ["tenant_id", "site_id", "camera_uuid"]
        )
        self.e2e_latency = Histogram(
            "e2e_latency_seconds", "End-to-end latency", ["tenant_id", "site_id", "camera_uuid"]
        )

        # Counters
        self.stream_errors = Counter(
            "stream_errors_total", "Stream errors", ["tenant_id", "site_id", "camera_uuid", "code"]
        )
        self.detections_published = Counter(
            "detections_published_total", "Frames with any published detections", ["tenant_id", "site_id", "camera_uuid"]
        )

        # Zone-specific metrics from Phase 2 spec
        self.frames_total = Counter(
            "zones_frames_total", "Total frames processed", ["tenant_id", "site_id", "camera_uuid"]
        )
        self.frames_skipped_motion_total = Counter(
            "zones_frames_skipped_motion_total", "Frames skipped by motion", ["tenant_id", "site_id", "camera_uuid"]
        )
        self.detections_raw_total = Counter(
            "zones_detections_raw_total", "Raw detections before filtering", ["tenant_id", "site_id", "camera_uuid"]
        )
        self.detections_published_zone = Counter(
            "zones_detections_published_total",
            "Published detections",
            ["tenant_id", "site_id", "camera_uuid", "zone_id", "label"],
        )
        self.detections_dropped_total = Counter(
            "zones_detections_dropped_total",
            "Dropped detections",
            ["tenant_id", "site_id", "camera_uuid", "zone_id", "reason"],
        )
        self.zones_config_hash = Gauge(
            "zones_config_hash", "Zones config hash", ["tenant_id", "site_id", "camera_uuid"]
        )

        # Decision accounting: reasons why frames were inferred or skipped
        self.inference_decision_total = Counter(
            "inference_decision_total",
            "Count of inference gating decisions by reason",
            ["tenant_id", "site_id", "camera_uuid", "reason"],
        )

        # Motion activity and skip ratios per camera with tenant/site labels
        self.motion_rate_gauge = Gauge(
            "motion_rate",
            "Fraction of frames with motion (0..1)",
            ["tenant_id", "site_id", "camera_uuid"],
        )
        self.motion_skip_ratio_gauge = Gauge(
            "motion_skip_ratio",
            "Fraction of frames skipped by motion gating (0..1)",
            ["tenant_id", "site_id", "camera_uuid"],
        )

        logger.info("Prometheus metrics initialized")

    def _update_motion_rate_metrics(self, camera_uuid: str) -> None:
        """Update motion rate and skip ratio gauges for a camera."""
        try:
            detector = self.optimized_motion_detectors.get(camera_uuid)
            if not detector:
                return
            stats = detector.get_stats() or {}
            motion_rate = float(stats.get("motion_rate", 0.0) or 0.0)
            skip_ratio = float(stats.get("skip_rate", 0.0) or 0.0)
            tenant_id = self._get_tenant_id(camera_uuid)
            site_id = self._get_site_id(camera_uuid)
            self.motion_rate_gauge.labels(
                tenant_id=tenant_id, site_id=site_id, camera_uuid=camera_uuid
            ).set(motion_rate)
            self.motion_skip_ratio_gauge.labels(
                tenant_id=tenant_id, site_id=site_id, camera_uuid=camera_uuid
            ).set(skip_ratio)
        except Exception:
            # Metrics should never break the pipeline
            pass

    def _cam_labels(self, camera_uuid: str) -> Dict[str, str]:
        return {
            "tenant_id": self._get_tenant_id(camera_uuid),
            "site_id": self._get_site_id(camera_uuid),
            "camera_uuid": camera_uuid,
        }

    def _calculate_config_hash(self) -> str:
        """Calculate SHA256 hash of the config file."""
        if not self.config_path:
            return ""

        try:
            with open(self.config_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating config hash: {e}")
            return ""

    async def _watch_config_changes(self):
        """Watch for configuration changes and reload when detected."""
        if not self.config_path:
            return

        logger.info(f"Starting configuration watcher for: {self.config_path}")

        while self.running:
            try:
                # Check if file exists
                if not Path(self.config_path).exists():
                    logger.warning(f"Config file not found: {self.config_path}")
                    await asyncio.sleep(5)
                    continue

                # Calculate current hash
                current_hash = self._calculate_config_hash()

                # Compare with stored hash
                if current_hash and current_hash != self.config_hash:
                    logger.info(f"Configuration change detected, reloading...")
                    await self._reload_configuration()
                    self.config_hash = current_hash

                # Wait before next check
                await asyncio.sleep(2)  # Check every 2 seconds

            except Exception as e:
                logger.error(f"Error in config watcher: {e}")
                await asyncio.sleep(5)

    async def _reload_configuration(self):
        """Reload configuration from file."""
        try:
            # Load new configuration
            new_config = ShardConfig.from_json_file(self.config_path)

            # Check for zone changes specifically
            old_cameras = {cam_id: cam.dict() for cam_id, cam in self.config.cameras.items()}
            new_cameras = {cam_id: cam.dict() for cam_id, cam in new_config.cameras.items()}

            zones_changed = False
            for cam_id in old_cameras:
                if cam_id in new_cameras:
                    if old_cameras[cam_id].get('zones') != new_cameras[cam_id].get('zones'):
                        zones_changed = True
                        logger.info(f"Zone configuration changed for camera: {cam_id}")
                        break

            # Update configuration
            self.config = new_config

            # Reload zone attributor if zones changed
            if zones_changed:
                await self._reload_zones()

            logger.info("Configuration reloaded successfully")

        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")

    async def _reload_zones(self):
        """Reload zone configurations for all cameras."""
        try:
            logger.info("Reloading zone configurations...")

            # Get camera configurations
            camera_configs = {
                source["camera_uuid"]: self.config.cameras.get(
                    source["camera_uuid"], CameraConfig(camera_uuid=source["camera_uuid"])
                )
                for source in self.config.sources
            }

            # Recreate zone attributor
            self.zone_attributor = MultiCameraZoneAttributor(
                camera_configs, self.frame_width, self.frame_height
            )

            # Update optimized motion detectors with new zones
            for camera_uuid, camera_config in camera_configs.items():
                detector = self.optimized_motion_detectors.get(camera_uuid)
                if detector and camera_config.motion_gating.enabled:
                    detector.update_zones(camera_config.zones)

            # Update config hash metrics
            for camera_uuid in camera_configs:
                config_hash = self.zone_attributor.get_zones_config_hash(camera_uuid)
                self.zones_config_hash.labels(**self._cam_labels(camera_uuid)).set(config_hash)

            logger.info("Zone configurations reloaded successfully")

        except Exception as e:
            logger.error(f"Error reloading zones: {e}")

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.draining = True
        self.running = False

    # Lease-aware methods (Phase 4)
    def set_lease_info(self, lease_id: str, worker_id: str):
        """Set lease information for this worker."""
        self.lease_id = lease_id
        self.worker_id = worker_id
        logger.info(f"Worker {worker_id} set lease ID: {lease_id}")

    async def start_lease_heartbeat(self):
        """Start lease heartbeat task."""
        if not self.lease_id or not self.lease_manager:
            return

        self.lease_heartbeat_task = asyncio.create_task(self._lease_heartbeat_loop())
        logger.info(f"Started lease heartbeat for worker {self.worker_id}")

    async def stop_lease_heartbeat(self):
        """Stop lease heartbeat task."""
        if self.lease_heartbeat_task:
            self.lease_heartbeat_task.cancel()
            try:
                await self.lease_heartbeat_task
            except asyncio.CancelledError:
                pass
            self.lease_heartbeat_task = None
            logger.info(f"Stopped lease heartbeat for worker {self.worker_id}")

    async def _lease_heartbeat_loop(self):
        """Background task to send lease heartbeats."""
        while self.running and self.lease_id:
            try:
                # Collect processing stats
                stats = {
                    "processed_frames": sum(self.frame_counts.values()),
                    "active_cameras": len(self.camera_states),
                    "camera_states": self.camera_states.copy(),
                    "last_frame_times": {
                        cam: ts.isoformat() for cam, ts in self.last_frame_times.items()
                    },
                }

                # Send heartbeat
                success = await self.lease_manager.send_heartbeat(
                    self.lease_id,
                    stats,
                    "ONLINE" if self.ready else "OFFLINE",  # Temporary: comment out CameraStatus until defined
                )

                if success:
                    logger.debug(f"Lease heartbeat sent for {self.lease_id}")
                else:
                    logger.warning(
                        f"Failed to send lease heartbeat for {self.lease_id}"
                    )

                await asyncio.sleep(15)  # Heartbeat every 15 seconds

            except asyncio.CancelledError:
                logger.debug("Lease heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in lease heartbeat loop: {e}")
                await asyncio.sleep(10)

    async def handle_configuration_update(self, new_config: CameraConfig):
        """Handle configuration updates from manager."""
        if not new_config:
            return

        camera_uuid = new_config.camera_uuid
        logger.info(f"Updating configuration for camera {camera_uuid}")

        try:
            # Update zone attribution
            self.zone_attributor.update_camera_config(camera_uuid, new_config)

            # Update motion detection (optimized only)
            if camera_uuid in self.optimized_motion_detectors:
                self.optimized_motion_detectors[camera_uuid].update_zones(new_config.zones)

            # Update metrics
            config_hash = self.zone_attributor.get_zones_config_hash(camera_uuid)
            self.zones_config_hash.labels(**self._cam_labels(camera_uuid)).set(config_hash)

            logger.info(f"Configuration updated successfully for camera {camera_uuid}")

        except Exception as e:
            logger.error(f"Error updating configuration for camera {camera_uuid}: {e}")

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics for reporting."""
        return {
            "worker_id": self.worker_id,
            "lease_id": self.lease_id,
            "running": self.running,
            "ready": self.ready,
            "draining": self.draining,
            "camera_count": len(self.camera_states),
            "processed_frames": sum(self.frame_counts.values()),
            "camera_states": self.camera_states.copy(),
            "last_activity": max(
                (ts.isoformat() for ts in self.last_frame_times.values()),
                default="never",
            )
            if self.last_frame_times
            else "never",
        }

    async def handle_lease_loss(self):
        """Handle loss of lease."""
        logger.warning(f"Worker {self.worker_id} lost lease {self.lease_id}")

        # Stop processing
        self.draining = True
        self.running = False

        # Stop lease heartbeat
        await self.stop_lease_heartbeat()

        # Shutdown pipeline
        if self.pipeline:
            await self._shutdown_pipeline()

        logger.info(f"Worker {self.worker_id} shut down due to lease loss")

    def _make_contours(self, mask: Any) -> List[List[tuple[int,int]]]:
        try:
            m = np.asarray(mask)
        except Exception:
            return []
        if m.ndim == 3:
            m = m[..., 0]
        if m.dtype != np.uint8:
            m_min, m_max = float(m.min()), float(m.max())
            if 0.0 <= m_min and m_max <= 1.0:
                m = (m * 255.0).astype(np.uint8)
            else:
                m = np.clip(m, 0, 255).astype(np.uint8)
        m = (m > 0).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys: List[List[tuple[int,int]]] = []
        for c in contours:
            if len(c) < 3:
                continue
            pts = c.reshape(-1, 2)
            polys.append([(int(x), int(y)) for x, y in pts])
        return polys

    def _should_publish_debug(self, cam: str) -> bool:
        """Simple rate limiter while a sink exists."""
        if not self.debug_mgr.has_sink(cam):
            return False
        target_dt = 1.0 / max(1, self._debug_target_fps)
        now = time.time()
        last = self._debug_rate.get(cam, 0.0)
        if (now - last) >= target_dt:
            self._debug_rate[cam] = now
            return True
        return False

    def _update_motion_state_from_motion_data(self, camera_uuid: str, video_frame: VideoFrame, motion_data: MotionData) -> None:
        """
        Update per-camera debug state from MotionData produced by ZeroCopyMotionDetector.
        Keeps overlay and debug rendering consistent regardless of where detection runs.
        """
        # Camera meta (width/height/fps) used by overlay renderer
        h, w = video_frame.image.shape[:2]
        fps_val = float(getattr(video_frame, "fps", 0) or 0)
        try:
            cam_cfg_obj = self.config.cameras.get(camera_uuid)
            cam_cfg = cam_cfg_obj.dict() if cam_cfg_obj else {}
        except Exception:
            cam_cfg = {}

        self._camera_meta[camera_uuid] = {
            "width": w,
            "height": h,
            "fps": int(fps_val or 10),
            "zones": cam_cfg.get("zones", []),
        }

        # Contours scaled back to original resolution if needed
        contours: List[np.ndarray] = []
        detector = self.optimized_motion_detectors.get(camera_uuid)
        if motion_data.significant_contours:
            scale = 1.0
            if detector and getattr(detector.config, "downscale", 1.0) < 1.0:
                scale = 1.0 / float(detector.config.downscale)
            for contour in motion_data.significant_contours:
                if scale != 1.0:
                    scaled = contour.astype(np.float32) * scale
                    contours.append(scaled.astype(np.int32))
                else:
                    contours.append(contour)

        # Motion state used by overlay and drawer
        self._motion_state[camera_uuid] = {
            "on": bool(motion_data.has_motion),
            "mask": motion_data.final_motion_mask,
            "contours": contours,
            "zone_hits": ["motion"] if motion_data.has_motion else [],
        }

        # Rich debug block shown in overlay
        self._motion_debug[camera_uuid] = {
            "mask": motion_data.final_motion_mask,
            "contours": motion_data.debug.get("all_contours", contours),
            "active": bool(motion_data.has_motion),
            "cooldown": motion_data.debug.get("cooldown_frames_left", 0),
            "gate_reason": motion_data.debug.get("threshold_status", "no_motion"),
            "min_area_px": int(motion_data.debug.get("min_area_px", 0)),
            "motion_area": motion_data.motion_area,
            "raw_motion_pixels": motion_data.debug.get("raw_motion_pixels", 0),
            "filtered_motion_area": motion_data.debug.get("filtered_motion_area", 0),
            "raw_motion_percent": motion_data.debug.get("raw_motion_percent", 0.0),
            "filtered_motion_percent": motion_data.debug.get("filtered_motion_percent", 0.0),
            "threshold_percent": motion_data.debug.get("threshold_percent", 0.0),
            "roi_area": motion_data.debug.get("roi_area", 0),
            "total_contours": motion_data.debug.get("total_contours", 0),
            "significant_contours": motion_data.debug.get("significant_contours", 0),
            "below_threshold_contours": motion_data.debug.get("below_threshold_contours", 0),
            "noise_floor": motion_data.debug.get("noise_floor", 0),
        }

    def _build_motion_gated_wrapper(self, original_on_video_frame: Callable[[List[VideoFrame]], List[Any]]) -> Callable[[List[VideoFrame]], List[Any]]:
        """
        Create a wrapper around the pipeline's on_video_frame that applies motion-first gating.

        - For each frame, run ZeroCopyMotionDetector and InferenceTimeoutManager.
        - If a frame should be skipped, return None in its position.
        - For frames to infer, call the original on_video_frame with the subset and re-align results.
        - Update per-camera debug state, decision counters, and metrics here (pre-inference hot path).
        """
        def gated_on_video_frame(video_frames: List[VideoFrame]) -> List[Any]:
            # Ensure list semantics as required by InferencePipeline
            frames: List[VideoFrame] = list(video_frames)
            results: List[Optional[Any]] = [None] * len(frames)

            frames_to_infer: List[VideoFrame] = []
            map_infer_index: List[int] = []

            for idx, vf in enumerate(frames):
                camera_uuid = self._get_camera_uuid(vf.source_id)
                now_ts = datetime.now(timezone.utc)

                # Track latest activity; keep counters aligned with wrapper usage
                self.last_frame_times[camera_uuid] = now_ts
                self.frame_counts[camera_uuid] = self.frame_counts.get(camera_uuid, 0) + 1
                self.frames_total.labels(**self._cam_labels(camera_uuid)).inc()
                # Update actual ingest FPS EMA and gauge
                ema_fps = self._tick_fps(camera_uuid)

                # Optimized path: zero-copy motion detector
                if camera_uuid in self.optimized_motion_detectors:
                    motion_detector = self.optimized_motion_detectors[camera_uuid]
                    timeout_manager = self.timeout_managers[camera_uuid]
                    perf_monitor = self.performance_monitors[camera_uuid]

                    motion_start = time.time()
                    motion_data = motion_detector.process_frame(vf.image)
                    _ = time.time() - motion_start  # reserved for future metrics

                    # Maintain debug and overlay state for this camera
                    self._update_motion_state_from_motion_data(camera_uuid, vf, motion_data)

                    decision = timeout_manager.should_trigger_inference(motion_data.has_motion)

                    # Update motion rate/skip ratio gauges
                    self._update_motion_rate_metrics(camera_uuid)

                    # Decision accounting for analysis
                    if decision.should_infer:
                        if decision.reason == "motion_detected":
                            self.inference_decisions["motion_triggered"] += 1
                        elif "timeout" in decision.reason:
                            self.inference_decisions["timeout_triggered"] += 1
                        self._last_infer_state[camera_uuid] = "INFER"
                    else:
                        if decision.reason == "min_interval_cooldown":
                            self.inference_decisions["skipped_cooldown"] += 1
                        else:
                            self.inference_decisions["skipped_no_motion"] += 1
                        self._last_infer_state[camera_uuid] = "SKIP"

                    # Prometheus counter by decision reason
                    try:
                        self.inference_decision_total.labels(
                            **self._cam_labels(camera_uuid), reason=str(decision.reason)
                        ).inc()
                    except Exception:
                        pass

                    # Perf monitor updates (lightweight rates only)
                    perf_monitor.update_inference_metrics(
                        inference_rate=1.0 if decision.should_infer else 0.0,
                        frame_rate=float(ema_fps or 0.0),
                        motion_rate=1.0 if motion_data.has_motion else 0.0,
                    )

                    if decision.should_infer:
                        frames_to_infer.append(vf)
                        map_infer_index.append(idx)
                    else:
                        # Count skipped frames and schedule a throttled debug frame snapshot
                        self.frames_skipped_motion_total.labels(**self._cam_labels(camera_uuid)).inc()
                        try:
                            self._schedule_debug_render(
                                cam=camera_uuid,
                                frame_bgr=vf.image,
                                predictions=None,
                                skipped_by_motion=True,
                                fps=self._calculate_fps(camera_uuid),
                            )
                        except Exception:
                            pass
                    continue

                # Fallback path: we do not gate here; let existing sink logic handle legacy motion detector
                frames_to_infer.append(vf)
                map_infer_index.append(idx)

            # Run original on_video_frame only for frames chosen for inference
            if frames_to_infer:
                try:
                    inferred = original_on_video_frame(frames_to_infer)
                except Exception:
                    # Preserve stability: if original fails, keep None for those positions
                    inferred = [None] * len(frames_to_infer)
                # Re-align back to original indices
                for local_i, global_i in enumerate(map_infer_index):
                    # Defensive: length check
                    if local_i < len(inferred):
                        results[global_i] = inferred[local_i]
            return results  # positions with None mean “skipped”

        return gated_on_video_frame

    def _build_debug_detections(self, predictions: Optional[Dict], frame_width: int, frame_height: int) -> List[dict]:
        """Convert raw predictions to standardized detection format for debug overlay."""
        dets = []
        if not predictions:
            return dets

        # Handle supervision Detections object
        if hasattr(predictions, "xyxy") and hasattr(predictions, "confidence"):
            if predictions.xyxy is not None and len(predictions.xyxy) > 0:
                for i in range(len(predictions.xyxy)):
                    x1, y1, x2, y2 = predictions.xyxy[i]
                    bbox = [float(x1), float(y1), float(x2), float(y2)]

                    label = "unknown"
                    if predictions.data and "class_name" in predictions.data:
                        label = str(predictions.data["class_name"][i])

                    score = float(predictions.confidence[i]) if predictions.confidence is not None else 0.0

                    track_id = None
                    if predictions.data and "tracker_id" in predictions.data:
                        track_id = int(predictions.data["tracker_id"][i])

                    dets.append({
                        "bbox": bbox,
                        "label": label,
                        "score": score,
                        "track_id": track_id
                    })
        elif isinstance(predictions, list):
            # Handle list of dictionaries
            for p in predictions:
                if not isinstance(p, dict):
                    continue

                # Try different bbox formats
                bbox = p.get("bbox") or p.get("xyxy")
                if bbox is None and all(k in p for k in ("x1","y1","x2","y2")):
                    bbox = [p["x1"], p["y1"], p["x2"], p["y2"]]
                if bbox is None and all(k in p for k in ("x","y","width","height")):
                    x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
                    bbox = [x - w/2, y - h/2, x + w/2, y + h/2]
                if bbox is None and all(k in p for k in ("cx","cy","w","h")):
                    cx, cy, w, h = float(p["cx"]), float(p["cy"]), float(p["w"]), float(p["h"])
                    if max(cx, cy, w, h) <= 1.5:  # normalized coords
                        cx, cy, w, h = cx*frame_width, cy*frame_height, w*frame_width, h*frame_height
                    bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

                if not bbox:
                    continue

                # Convert to pixel coordinates and clip
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, min(frame_width-1, x1))
                x2 = max(0, min(frame_width-1, x2))
                y1 = max(0, min(frame_height-1, y1))
                y2 = max(0, min(frame_height-1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": p.get("label") or p.get("class") or "obj",
                    "score": float(p.get("score") or p.get("confidence") or 0.0),
                    "track_id": p.get("track_id")
                })
        return dets

    def _publish_debug(
        self, cam: str, frame_bgr: np.ndarray, *,
        predictions: Optional[Dict] = None,
        skipped_by_motion: bool = False,
        fps: Optional[float] = None,
    ) -> None:
        """Unified, non-blocking call into DebugManager with full debug context."""
        if not self._should_publish_debug(cam):
            return

        # Remember native resolution for proper scaling of zones
        self._remember_native_res(cam, frame_bgr)

        # Build camera configuration dict for overlay
        cam_cfg = {}
        try:
            cam_cfg_obj = self.config.cameras.get(cam)
            if cam_cfg_obj:
                cam_cfg = cam_cfg_obj.dict()
        except Exception:
            cam_cfg = {}

        # Build detections list
        height, width = frame_bgr.shape[:2]
        dets = self._build_debug_detections(predictions, width, height)

        # Update last detections if we ran inference
        if not skipped_by_motion and dets:
            self._last_detections[cam] = dets

        # Use last detections for ghosting on skipped frames
        dets_for_render = dets
        if skipped_by_motion and not dets:
            dets_for_render = self._last_detections.get(cam, [])

        # Render overlay using the new Python renderer
        annotated = render_debug_overlay(
            frame_bgr.copy(),
            camera_cfg=cam_cfg,
            detections=dets_for_render,
            native_resolution=self._native_res.get(cam),
            show_fps=fps,
            timestamp=time.time(),
            motion_debug=self._motion_debug.get(cam),
            inference_state=self._last_infer_state.get(cam),
        )
        # Publish the annotated frame via the debug manager
        self.debug_mgr.publish_sink(cam, annotated)

    async def _render_overlay_async(
        self,
        cam: str,
        frame_bgr: np.ndarray,
        *,
        predictions: Optional[Dict] = None,
        skipped_by_motion: bool = False,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """Render overlay in thread executor to avoid blocking event loop."""
        # Prepare data needed for rendering
        self._remember_native_res(cam, frame_bgr)
        cam_cfg_obj = self.config.cameras.get(cam)
        cam_cfg = cam_cfg_obj.dict() if cam_cfg_obj else {}
        h, w = frame_bgr.shape[:2]
        dets = self._build_debug_detections(predictions, w, h)
        if not skipped_by_motion and dets:
            self._last_detections[cam] = dets
        dets_for_render = dets if (not skipped_by_motion or dets) else self._last_detections.get(cam, [])
        loop = asyncio.get_running_loop()
        annotated = await loop.run_in_executor(
            None,
            lambda: render_debug_overlay(
                frame_bgr.copy(),
                camera_cfg=cam_cfg,
                detections=dets_for_render,
                native_resolution=self._native_res.get(cam),
                show_fps=fps,
                timestamp=time.time(),
                motion_debug=self._motion_debug.get(cam),
                inference_state=self._last_infer_state.get(cam),
            ),
        )
        return annotated

    async def _publish_debug_async(
        self, cam: str, frame_bgr: np.ndarray, *,
        predictions: Optional[Dict] = None,
        skipped_by_motion: bool = False,
        fps: Optional[float] = None,
    ) -> None:
        if not self._should_publish_debug(cam):
            return
        try:
            annotated = await self._render_overlay_async(
                cam,
                frame_bgr,
                predictions=predictions,
                skipped_by_motion=skipped_by_motion,
                fps=fps,
            )
            self.debug_mgr.publish_sink(cam, annotated)
        except Exception:
            pass

    def _schedule_debug_render(
        self, cam: str, frame_bgr: np.ndarray, *,
        predictions: Optional[Dict] = None,
        skipped_by_motion: bool = False,
        fps: Optional[float] = None,
    ) -> None:
        """Schedule debug rendering on the event loop without blocking sink thread."""
        if not self.event_loop:
            return
        def _submit():
            asyncio.create_task(
                self._publish_debug_async(
                    cam,
                    frame_bgr,
                    predictions=predictions,
                    skipped_by_motion=skipped_by_motion,
                    fps=fps,
                )
            )
        self.event_loop.call_soon_threadsafe(_submit)

    def _remember_native_res(self, cam: str, frame_bgr: np.ndarray) -> None:
        if cam not in self._native_res and frame_bgr is not None:
            h, w = frame_bgr.shape[:2]
            self._native_res[cam] = (w, h)

    async def _on_prediction_single_frame(
        self, predictions: Optional[Dict], video_frame: VideoFrame
    ):
        """
        Handle predictions for a single frame from inference pipeline.
        Integrates motion detection, zone attribution, and event publishing.
        """
        try:
            camera_uuid = self._get_camera_uuid(video_frame.source_id)
            frame_timestamp = datetime.now(timezone.utc)
            # logger.info(f"Prediction callback triggered: {camera_uuid=} frame_id={video_frame.frame_id} {frame_timestamp=}")

            # When motion gating wrapper is active, the hot-path already updates
            # counters and last activity. Avoid double counting here.
            if not self._use_motion_gating_wrapper:
                self.last_frame_times[camera_uuid] = frame_timestamp
                self.frame_counts[camera_uuid] = self.frame_counts.get(camera_uuid, 0) + 1
                self.frames_total.labels(**self._cam_labels(camera_uuid)).inc()
                # Update per-camera FPS EMA and gauge
                self._tick_fps(camera_uuid)

            # Apply motion detection if enabled for this camera
            skipped_by_motion = False

            # If wrapper is active and prediction is None, this frame was skipped
            # pre-inference. Debug frame already published from wrapper.
            if self._use_motion_gating_wrapper and predictions is None:
                self._last_infer_state[camera_uuid] = "SKIP"
                return

            if not self._use_motion_gating_wrapper and camera_uuid in self.optimized_motion_detectors:
                # Use optimized zero-copy motion detector
                motion_detector = self.optimized_motion_detectors[camera_uuid]
                timeout_manager = self.timeout_managers[camera_uuid]
                perf_monitor = self.performance_monitors[camera_uuid]

                # Zero-copy motion detection
                motion_start_time = time.time()
                motion_data = motion_detector.process_frame(video_frame.image)
                motion_processing_time = time.time() - motion_start_time

                # Make inference decision
                inference_decision = timeout_manager.should_trigger_inference(motion_data.has_motion)

                # Track decision statistics
                if inference_decision.should_infer:
                    if inference_decision.reason == "motion_detected":
                        self.inference_decisions["motion_triggered"] += 1
                    elif "timeout" in inference_decision.reason:
                        self.inference_decisions["timeout_triggered"] += 1
                else:
                    self.inference_decisions["skipped_no_motion"] += 1

                # Update performance monitor
                perf_monitor.update_inference_metrics(
                    inference_rate=1.0 if inference_decision.should_infer else 0.0,
                    frame_rate=float(self._fps_ema.get(camera_uuid, 0.0)),
                    motion_rate=1.0 if motion_data.has_motion else 0.0,
                )

                # Log decision (elevated to INFO for debugging)
                logger.debug(
                    f"Worker inference decision for {camera_uuid}: "
                    f"{inference_decision.reason} (has_motion: {motion_data.has_motion}, "
                    f"motion_area: {motion_data.motion_area})"
                )

                # Update motion rate/skip ratio gauges
                self._update_motion_rate_metrics(camera_uuid)

                # Track decision statistics (existing)
                if inference_decision.should_infer:
                    if inference_decision.reason == "motion_detected":
                        self.inference_decisions["motion_triggered"] += 1
                    elif "timeout" in inference_decision.reason:
                        self.inference_decisions["timeout_triggered"] += 1
                else:
                    self.inference_decisions["skipped_no_motion"] += 1

                # Skip processing if no inference should run
                if not inference_decision.should_infer:
                    skipped_by_motion = True
                    self.frames_skipped_motion_total.labels(**self._cam_labels(camera_uuid)).inc()
                    self._last_infer_state[camera_uuid] = "SKIP"

                    # Render and publish debug frame (throttled) without blocking event loop
                    await self._publish_debug_async(
                        cam=camera_uuid,
                        frame_bgr=video_frame.image,
                        predictions=None,
                        skipped_by_motion=True,
                        fps=self._calculate_fps(camera_uuid),
                    )
                    # IMPORTANT: skip processing since inference result is not needed
                    # BUT we still need to update the motion detector state for the next frame
                    return

            elif not self._use_motion_gating_wrapper:
                # No wrapper and no optimized detector configured; proceed without gating
                pass

            # Update camera metadata for optimized detector (if not already done)
            if camera_uuid in self.optimized_motion_detectors and camera_uuid not in self._camera_meta:
                # 1) Update per-camera meta (so /debug/start can derive width/height/fps)
                h, w = video_frame.image.shape[:2]
                fps_val = float(getattr(video_frame, "fps", 0) or 0)
                ts_ms = int(getattr(video_frame, "ts_ms", int(time.time() * 1000)))

                # Build camera configuration dict for overlay (include zones)
                cam_cfg = {}
                try:
                    cam_cfg_obj = self.config.cameras.get(camera_uuid)
                    if cam_cfg_obj:
                        cam_cfg = cam_cfg_obj.dict()
                except Exception:
                    cam_cfg = {}

                self._camera_meta[camera_uuid] = {
                    "width": w,
                    "height": h,
                    "fps": int(fps_val or 10),
                    "zones": cam_cfg.get("zones", [])
                }

            # Build motion state for overlays (optimized detector)
            # Only rebuild here when not using the motion-gating wrapper.
            # When the wrapper is active, it already updates _motion_state and _motion_debug.
            if camera_uuid in self.optimized_motion_detectors and not self._use_motion_gating_wrapper:
                # For optimized detector, use motion_data directly
                motion_on = motion_data.has_motion
                raw_mask = motion_data.final_motion_mask
                mask = raw_mask

                # Convert contours to format expected by overlay
                contours = []
                if motion_data.significant_contours:
                    for contour in motion_data.significant_contours:
                        # Scale contours back to original frame size if needed
                        if self.optimized_motion_detectors[camera_uuid].config.downscale < 1.0:
                            scale = 1.0 / self.optimized_motion_detectors[camera_uuid].config.downscale
                            scaled_contour = contour.astype(np.float32) * scale
                            contours.append(scaled_contour.astype(np.int32))
                        else:
                            contours.append(contour)

                raw_zone_hits = ["motion"] if motion_on else []

            def _as_zone_hits(v) -> list[str]:
                # mirror drawer’s logic, but lighter
                if v is None or v is False:
                    return []
                if v is True:
                    return ["true"]
                if isinstance(v, (str, bytes, bytearray)):
                    try:
                        return [v.decode("utf-8")] if isinstance(v, (bytes, bytearray)) else [v]
                    except Exception:
                        return [str(v)]
                if isinstance(v, dict):
                    return [str(k) for k in v.keys()]
                try:
                    from collections.abc import Iterable
                    if isinstance(v, Iterable) and not isinstance(v, (str, bytes, bytearray)):
                        return [str(x) for x in v]
                except Exception:
                    pass
                return [str(v)]


                zone_hits = _as_zone_hits(raw_zone_hits)
                contours = None
                if mask is not None:
                    contours = self._make_contours(mask)

                    self._motion_state[camera_uuid] = {
                        "on": motion_on,
                        "mask": mask,             # keep original; draw will resize if needed
                        "contours": contours,
                        "zone_hits": zone_hits,
                    }
                    # Update motion debug state based on detector type
                    # For optimized detector, use enhanced motion_data debug info
                    self._motion_debug[camera_uuid] = {
                        "mask": mask,
                        "contours": motion_data.debug.get("all_contours", contours),  # Use ALL contours for display
                        "active": motion_on,
                        "cooldown": motion_data.debug.get("cooldown_frames_left", 0),
                        "gate_reason": motion_data.debug.get("threshold_status", "no_motion"),
                        "min_area_px": int(motion_data.debug.get("min_area_px", 0)),
                        "motion_area": motion_data.motion_area,
                        # Enhanced debug metrics
                        "raw_motion_pixels": motion_data.debug.get("raw_motion_pixels", 0),
                        "filtered_motion_area": motion_data.debug.get("filtered_motion_area", 0),
                        "raw_motion_percent": motion_data.debug.get("raw_motion_percent", 0.0),
                        "filtered_motion_percent": motion_data.debug.get("filtered_motion_percent", 0.0),
                        "threshold_percent": motion_data.debug.get("threshold_percent", 0.0),
                        "roi_area": motion_data.debug.get("roi_area", 0),
                        "total_contours": motion_data.debug.get("total_contours", 0),
                        "significant_contours": motion_data.debug.get("significant_contours", 0),
                        "below_threshold_contours": motion_data.debug.get("below_threshold_contours", 0),
                        "noise_floor": motion_data.debug.get("noise_floor", 0),
                    }

                    # No fallback skip path; gating handled above for optimized detectors

                    # Log enhanced motion detection metrics
                    logger.debug(
                        f"Motion detected {camera_uuid}: "
                        f"raw={motion_data.debug.get('raw_motion_pixels', 0)}px ({motion_data.debug.get('raw_motion_percent', 0):.1f}%), "
                        f"filtered={motion_data.motion_area}px ({motion_data.debug.get('filtered_motion_percent', 0):.1f}%), "
                        f"threshold={motion_data.debug.get('min_area_px', 0)}px ({motion_data.debug.get('threshold_percent', 0):.1f}%), "
                        f"contours={motion_data.debug.get('total_contours', 0)} total, "
                        f"{motion_data.debug.get('significant_contours', 0)} significant"
                    )

            # Process detections through zone attribution
            raw_detections = predictions.get("predictions", []) if predictions else []
            # Do not assume raw_detections is a list; counting handled after zone processing

            # Apply zone attribution and filtering
            published_objects, zone_stats = (
                self.zone_attributor.process_camera_detections(
                    camera_uuid, raw_detections
                )
            )

            # Update zone metrics
            for obj in published_objects:
                self.detections_published_zone.labels(
                    **self._cam_labels(camera_uuid),
                    zone_id=str(obj.primary_zone_id),
                    label=obj.label,
                ).inc()

            # Update raw detections counter robustly using zone_stats
            try:
                total_raw = 0
                for zs in zone_stats.values():
                    total_raw += int(zs.get("objects", 0)) + int(zs.get("dropped", 0))
                if total_raw:
                    self.detections_raw_total.labels(**self._cam_labels(camera_uuid)).inc(total_raw)
            except Exception:
                pass

            # Update dropped metrics
            for zone_id_str, stats in zone_stats.items():
                if stats["dropped"] > 0:
                    self.detections_dropped_total.labels(
                        **self._cam_labels(camera_uuid), zone_id=zone_id_str, reason="zone_filter"
                    ).inc(stats["dropped"])

            # Create detection event
            detection_event = DetectionEvent(
                schema_version=2,
                event_id=str(ulid.new()),
                ts_ns=int(frame_timestamp.timestamp() * 1_000_000_000),
                tenant_id=self._get_tenant_id(camera_uuid),
                site_id=self._get_site_id(camera_uuid),
                camera_uuid=camera_uuid,
                model=ModelInfo(id="coco/11", adapter="roboflow-inference-pipeline"),
                frame=FrameInfo(
                    w=self.frame_width,
                    h=self.frame_height,
                    seq=video_frame.frame_id,
                    fps=self._calculate_fps(camera_uuid),
                    skipped_by_motion=skipped_by_motion,
                ),
                zones_config=ZonesConfigInfo(
                    zone_version=self.zone_attributor.get_zones_config_hash(
                        camera_uuid
                    ),
                    zone_test=self.config.cameras.get(
                        camera_uuid, CameraConfig(camera_uuid=camera_uuid)
                    ).zone_test,
                    iou_threshold=self.config.cameras.get(
                        camera_uuid, CameraConfig(camera_uuid=camera_uuid)
                    ).iou_threshold,
                ),
                objects=published_objects,
            )

            # Queue for publishing
            await self.detection_queue.put(detection_event)

            # Debug visual stream (non-blocking, fire-and-forget)
            self._last_infer_state[camera_uuid] = "INFER"
            await self._publish_debug_async(
                cam=camera_uuid,
                frame_bgr=video_frame.image,
                predictions=raw_detections,
                skipped_by_motion=False,
                fps=self._calculate_fps(camera_uuid),
            )

            # Update published metrics (fix reference)
            # Count frames with any published detections
            self.detections_published.labels(**self._cam_labels(camera_uuid)).inc(int(len(published_objects) > 0))

            # Update inference latency
            if "time" in predictions:
                self.inference_latency.labels(**self._cam_labels(camera_uuid)).observe(
                    predictions["time"]
                )

            # Update e2e latency
            e2e_latency = self._calculate_e2e_latency(video_frame)
            self.e2e_latency.labels(**self._cam_labels(camera_uuid)).observe(e2e_latency)

            logger.debug(
                f"Processed prediction for {camera_uuid}: "
                f"{len(published_objects)} objects published"
            )

        except Exception as e:
            logger.error(
                f"Error processing prediction for {camera_uuid}: {e}", exc_info=True
            )
            self.stream_errors.labels(
                **self._cam_labels(camera_uuid), code="PREDICTION_ERROR"
            ).inc()

    async def _on_prediction(self, predictions: Optional[Dict], video_frames):
        """
        Handle predictions from inference pipeline.
        When multiple sources are used, video_frames is a list.
        When single source is used, video_frames is a single VideoFrame.
        """
        # Handle both single frame and list of frames
        if isinstance(video_frames, list):
            # Multiple frames - process each one
            # The predictions dict contains results for all frames
            for idx, video_frame in enumerate(video_frames):
                if video_frame is None:
                    continue
                # Extract predictions for this specific frame/source
                frame_predictions = None
                if predictions and isinstance(predictions, list):
                    # If predictions is also a list, match by aligned index
                    if idx < len(predictions):
                        frame_predictions = predictions[idx]
                elif predictions:
                    # If predictions is a single dict, use it for all frames
                    frame_predictions = predictions

                await self._on_prediction_single_frame(frame_predictions, video_frame)
        else:
            # Single frame
            await self._on_prediction_single_frame(predictions, video_frames)

    def _on_prediction_sync(self, predictions: Optional[Dict], video_frame: VideoFrame):
        """
        Synchronous version of prediction handler for testing.
        Forwards to async version via event loop.
        """
        logger.info(
            f"=== SYNC _on_prediction_sync called: frame_id={video_frame.frame_id} ==="
        )

        if not self.event_loop:
            logger.error("Event loop not available for sync prediction handler!")
            return

        try:
            # Submit to event loop and optionally wait for result
            future = asyncio.run_coroutine_threadsafe(
                self._on_prediction(predictions, video_frame), self.event_loop
            )
            logger.debug(f"Submitted prediction to event loop via sync handler")
            # Don't wait - let it run async
        except Exception as e:
            logger.error(f"Error in sync prediction handler: {e}", exc_info=True)

    def _on_status_update(self, update: StatusUpdate):
        """Handle status updates from the inference pipeline."""
        try:
            # Filter out excessive debug events - only log meaningful status changes
            is_frame_event = update.event_type and update.event_type.lower() in [
                "frame_captured",
                "frame_consumed",
                "frame_dropped",
            ]

            if not is_frame_event or update.severity != UpdateSeverity.DEBUG:
                logger.debug(
                    f"Status update received - event_type: {update.event_type}, "
                    f"payload: {update.payload}, severity: {update.severity}, "
                    f"context: {getattr(update, 'context', 'N/A')}"
                )

            # Extract camera UUID from update
            camera_uuid = self._extract_camera_from_update(update)

            # Skip processing if this is not a camera-specific update (batch updates, empty payloads, etc.)
            if camera_uuid is None:
                return

            # Map pipeline state to our state names
            state = self._map_pipeline_state(update)

            # Update local state
            old_state = self.camera_states.get(camera_uuid)
            self.camera_states[camera_uuid] = state

            # Update metrics
            self.stream_up.labels(**self._cam_labels(camera_uuid)).set(
                1 if state == "STREAMING" else 0
            )

            # Handle errors
            if update.severity == UpdateSeverity.ERROR:
                self._handle_error_update(update, camera_uuid)

            # Create status event (edge-triggered) - but don't spam with frame events
            if state != old_state and not (
                is_frame_event and old_state == "STREAMING" and state == "STREAMING"
            ):
                status_event = StatusEvent(
                    type="stream.status",
                    state=state,
                    camera_uuid=camera_uuid,
                    runner_id=self.config.runner_id,
                    shard_id=self.config.shard_id,
                    fps=self._calculate_fps(camera_uuid)
                    if state == "STREAMING"
                    else None,
                    ts=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                )

                # Queue for publishing
                if self.event_loop:
                    asyncio.run_coroutine_threadsafe(
                        self.status_queue.put(status_event), self.event_loop
                    )

                logger.info(f"Status change for {camera_uuid}: {old_state} -> {state}")

        except Exception as e:
            logger.error(f"Error processing status update: {e}")
            logger.debug(
                f"Update object: {update.__dict__ if hasattr(update, '__dict__') else str(update)}"
            )

    def _handle_error_update(self, update: StatusUpdate, camera_uuid: str):
        """Handle error status updates."""
        error_code = update.payload.get("error_type", "UNKNOWN")

        self.stream_errors.labels(**self._cam_labels(camera_uuid), code=error_code).inc()

        error_event = ErrorEvent(
            type="stream.error",
            camera_uuid=camera_uuid,
            runner_id=self.config.runner_id,
            code=error_code,
            detail=update.payload.get("message", str(update.payload)),
            retry_in_ms=8000,  # Default retry
            ts=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )

        if self.event_loop:
            asyncio.run_coroutine_threadsafe(
                self.status_queue.put(error_event), self.event_loop
            )

    async def _publish_detection_events(self):
        """Continuously publish detection events to RabbitMQ."""
        connection = None
        channel = None

        while self.running:
            try:
                # Connect to RabbitMQ if needed
                if not connection or connection.is_closed:
                    connection = await aio_pika.connect_robust(
                        f"amqp://guest:guest@{self.config.amqp['host']}/"
                    )
                    channel = await connection.channel()
                    await channel.set_qos(prefetch_count=10)
                    # Ensure detections exchange exists (topic, durable)
                    try:
                        await channel.declare_exchange(
                            self.config.amqp["ex_detect"],
                            aio_pika.ExchangeType.TOPIC,
                            durable=True,
                        )
                    except Exception:
                        pass

                # Get detection event from queue
                try:
                    event = await asyncio.wait_for(
                        self.detection_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Publish to RabbitMQ
                routing_key = (
                    f"detections.{event.tenant_id}.{event.site_id}.{event.camera_uuid}"
                )

                message = aio_pika.Message(
                    body=event.model_dump_json().encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                )

                exchange = await channel.get_exchange(self.config.amqp["ex_detect"])
                await exchange.publish(message, routing_key=routing_key)

                logger.debug(f"Published detection for {event.camera_uuid}")

            except Exception as e:
                logger.error(f"Error publishing detection: {e}")
                await asyncio.sleep(1)

        # Cleanup
        if connection and not connection.is_closed:
            await connection.close()

    async def _publish_status_events(self):
        """Continuously publish status events to RabbitMQ."""
        connection = None
        channel = None

        while self.running:
            try:
                # Connect to RabbitMQ if needed
                if not connection or connection.is_closed:
                    connection = await aio_pika.connect_robust(
                        f"amqp://guest:guest@{self.config.amqp['host']}/"
                    )
                    channel = await connection.channel()
                    # Ensure status exchange exists (topic, durable)
                    try:
                        await channel.declare_exchange(
                            self.config.amqp["ex_status"],
                            aio_pika.ExchangeType.TOPIC,
                            durable=True,
                        )
                    except Exception:
                        pass

                # Get status event from queue
                try:
                    event = await asyncio.wait_for(self.status_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Determine routing key
                tenant_id = self._get_tenant_id(event.camera_uuid)
                site_id = self._get_site_id(event.camera_uuid)

                if event.type == "stream.error":
                    routing_key = (
                        f"stream.error.{tenant_id}.{site_id}.{event.camera_uuid}"
                    )
                else:
                    routing_key = (
                        f"stream.status.{tenant_id}.{site_id}.{event.camera_uuid}"
                    )

                message = aio_pika.Message(
                    body=event.model_dump_json().encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                )

                exchange = await channel.get_exchange(self.config.amqp["ex_status"])
                await exchange.publish(message, routing_key=routing_key)

                # Log differently based on event type
                if hasattr(event, "state") and event.state:
                    logger.debug(
                        f"Published status for {event.camera_uuid}: {event.state}"
                    )
                else:
                    logger.debug(
                        f"Published event for {event.camera_uuid}: {event.type}"
                    )

            except Exception as e:
                logger.error(f"Error publishing status: {e}")
                await asyncio.sleep(1)

        # Cleanup
        if connection and not connection.is_closed:
            await connection.close()

    async def _periodic_status_summary(self):
        """Send periodic status summaries every 5 seconds."""
        while self.running:
            await asyncio.sleep(self.config.telemetry.get("report_interval_seconds", 5))

            for camera_uuid in [src["camera_uuid"] for src in self.config.sources]:
                try:
                    last_frame = self.last_frame_times.get(camera_uuid)
                    frame_age = None
                    if last_frame:
                        frame_age = (
                            datetime.now(timezone.utc) - last_frame
                        ).total_seconds()
                        self.last_frame_age.labels(**self._cam_labels(camera_uuid)).set(
                            frame_age
                        )

                    # Get zone statistics
                    zone_attributor = self.zone_attributor.attributors.get(camera_uuid)
                    motion_detector = self.optimized_motion_detectors.get(camera_uuid)

                    zones_stats = None
                    if zone_attributor:
                        attr_stats = zone_attributor.get_stats()
                        per_zone_stats = {}

                        # Build per-zone stats
                        for zone_id_str in attr_stats["per_zone_assignments"]:
                            per_zone_stats[zone_id_str] = ZoneStats(
                                objects=attr_stats["per_zone_assignments"].get(
                                    zone_id_str, 0
                                ),
                                dropped=attr_stats["per_zone_drops"].get(
                                    zone_id_str, 0
                                ),
                            )

                        # Get motion stats for this camera
                        motion_stats = motion_detector.get_stats() if motion_detector else {}

                        zones_stats = ZonesStats(
                            frames_skipped_motion=motion_stats.get(
                                "frames_skipped_no_motion", 0
                            ),
                            frames_processed=motion_stats.get("frames_processed", 0),
                            objects_published=attr_stats.get("detections_processed", 0)
                            - attr_stats.get("detections_dropped_filters", 0),
                            objects_dropped_by_filters=attr_stats.get(
                                "detections_dropped_filters", 0
                            ),
                            per_zone=per_zone_stats,
                        )

                    # Create summary event
                    summary = StatusEvent(
                        type="stream.status",
                        state=self.camera_states.get(camera_uuid, "UNKNOWN"),
                        camera_uuid=camera_uuid,
                        runner_id=self.config.runner_id,
                        shard_id=self.config.shard_id,
                        fps=self._calculate_fps(camera_uuid),
                        last_frame_ts=last_frame.isoformat().replace("+00:00", "Z")
                        if last_frame
                        else None,
                        last_frame_age_s=frame_age,
                        zones_stats=zones_stats,
                        ts=datetime.now(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                    )

                    await self.status_queue.put(summary)

                except Exception as e:
                    logger.error(
                        f"Error creating status summary for {camera_uuid}: {e}"
                    )

    async def _health_handler(self, request):
        """Health check endpoint."""
        if self.pipeline and not self.draining:
            return web.Response(text="OK", status=200)
        return web.Response(text="NOT_READY", status=503)

    async def _ready_handler(self, request):
        """Readiness check endpoint."""
        if not self.ready:
            return web.Response(text="NOT_READY", status=503)

        # Check readiness quorum
        total_cameras = len(self.config.sources)
        streaming_cameras = sum(
            1 for state in self.camera_states.values() if state == "STREAMING"
        )

        readiness_pct = (
            (streaming_cameras / total_cameras * 100) if total_cameras > 0 else 0
        )
        required_pct = 80  # Default readiness quorum

        if readiness_pct >= required_pct:
            return web.Response(
                text=json.dumps(
                    {"ready": True, "cameras": f"{streaming_cameras}/{total_cameras}"}
                ),
                status=200,
                content_type="application/json",
            )

        return web.Response(
            text=json.dumps(
                {"ready": False, "cameras": f"{streaming_cameras}/{total_cameras}"}
            ),
            status=503,
            content_type="application/json",
        )

    async def _drain_handler(self, request):
        """Initiate graceful drain."""
        logger.info("Drain requested via HTTP endpoint")
        self.draining = True

        # Schedule shutdown after grace period
        asyncio.create_task(self._delayed_shutdown())

        return web.Response(text="DRAINING", status=202)

    async def _terminate_handler(self, request):
        """Immediate termination."""
        logger.info("Immediate termination requested")
        self.running = False
        self.draining = True
        return web.Response(text="TERMINATING", status=202)

    async def _delayed_shutdown(self):
        """Shutdown after grace period."""
        grace_period = 30  # seconds
        await asyncio.sleep(grace_period)
        self.running = False

    async def _run_health_server(self):
        """Run the health check HTTP server."""
        app = web.Application()
        app["worker"] = self
        app["get_camera_meta"] = lambda cam: self._camera_meta.get(cam, {})
        app.router.add_get("/healthz", self._health_handler)
        app.router.add_get("/ready", self._ready_handler)
        app.router.add_post("/drain", self._drain_handler)
        app.router.add_post("/terminate", self._terminate_handler)

        ## debug
        async def debug_start(request: web.Request) -> web.Response:
            cam = request.match_info["camera_uuid"]
            qs = request.rel_url.query

            # Host/port to receive the RTP stream (often localhost + go2rtc/ffplay port)
            host = qs.get("host", "127.0.0.1")
            port = int(qs.get("port", "5002"))
            ttl  = int(qs.get("ttl", "30"))

            # Derive camera geometry/fps from known state (fallbacks are safe)
            try:
                meta = request.app["get_camera_meta"](cam) or {}
            except Exception:
                meta = {}
            width  = int(meta.get("width", 1280))
            height = int(meta.get("height", 720))
            fps    = int(meta.get("fps", 10))

            url = request.app["worker"].debug_mgr.start(
                cam, width=width, height=height, fps=fps, host=host, port=port, ttl_s=ttl
            )
            return web.json_response({
                "camera_uuid": cam,
                "url": url,
                "ffplay": f"ffplay -fflags nobuffer -flags low_delay -i {url}",
                "width": width, "height": height, "fps": fps, "ttl_s": ttl
            })

        async def debug_stop(request: web.Request) -> web.Response:
            cam = request.match_info["camera_uuid"]
            ok = request.app["worker"].debug_mgr.stop(cam)
            return web.json_response({"camera_uuid": cam, "stopped": ok})

        app.add_routes([
            web.post("/debug/{camera_uuid}/start", debug_start),
            web.get("/debug/{camera_uuid}/start", debug_start),
            web.post("/debug/{camera_uuid}/stop", debug_stop),
            web.get("/debug/{camera_uuid}/stop", debug_stop),
        ])

        runner = web.AppRunner(app)
        try:
            await runner.setup()
            # Configurable bind address/port via env
            host = os.getenv("WORKER_HTTP_HOST", "0.0.0.0")
            port = int(os.getenv("WORKER_HTTP_PORT", "8080"))
            # Bind on all interfaces to avoid IPv4/IPv6 localhost issues
            site = web.TCPSite(runner, host, port)
            await site.start()
            logger.info(f"Health server started on http://{host}:{port}")
        except Exception as e:
            logger.error(f"Failed to start health server: {e}")
            # Try fallback port
            try:
                host = os.getenv("WORKER_HTTP_HOST", "0.0.0.0")
                port_fallback = int(os.getenv("WORKER_HTTP_FALLBACK_PORT", "8081"))
                site = web.TCPSite(runner, host, port_fallback)
                await site.start()
                logger.info(f"Health server started on http://{host}:{port_fallback} (fallback)")
            except Exception as ee:
                logger.error(f"Failed to start health server on fallback :8081: {ee}")
                return  # Give up starting the health server

        # Keep running until shutdown
        while self.running:
            await asyncio.sleep(1)

        await runner.cleanup()

    async def _run_event_loop(self):
        """Run the async event loop for publishing."""
        self.event_loop = asyncio.get_running_loop()

        # Start async tasks
        tasks = [
            self._publish_detection_events(),
            self._publish_status_events(),
            self._periodic_status_summary(),
            self._run_health_server(),
            self._run_debug_server(),
        ]

        # Init frame batch processing primitives and workers
        self.frame_batch_queue = asyncio.Queue(maxsize=16)
        self._frame_semaphore = asyncio.Semaphore(self._frame_concurrency)
        frame_workers = int(os.getenv("WORKER_FRAME_WORKERS", "2"))
        for _ in range(max(1, frame_workers)):
            tasks.append(self._frame_worker())

        # Add config watcher if enabled
        if self.config_path:
            self.config_watcher_task = asyncio.create_task(self._watch_config_changes())
            tasks.append(self.config_watcher_task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_debug_server(self):
        pass

    def _enqueue_batch_from_sink(self, predictions, video_frames) -> None:
        """Enqueue batch processing from sink thread without blocking it."""
        if not self.event_loop or not self.frame_batch_queue:
            # Fallback to previous behaviour if loop not ready
            try:
                asyncio.run_coroutine_threadsafe(
                    self._on_prediction(predictions, video_frames), self.event_loop
                )
            except Exception:
                pass
            return
        def _put():
            try:
                self.frame_batch_queue.put_nowait((predictions, video_frames))
            except asyncio.QueueFull:
                # Drop oldest by getting one and putting new to keep freshness
                try:
                    _ = self.frame_batch_queue.get_nowait()
                except Exception:
                    pass
                try:
                    self.frame_batch_queue.put_nowait((predictions, video_frames))
                except Exception:
                    pass
        self.event_loop.call_soon_threadsafe(_put)

    async def _frame_worker(self):
        """Consume batches and process frames with bounded concurrency."""
        while self.running:
            try:
                item = await asyncio.wait_for(self.frame_batch_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                if item is None:
                    continue
                predictions, video_frames = item
                await self._process_batch_item(predictions, video_frames)
            except Exception:
                pass
            finally:
                try:
                    self.frame_batch_queue.task_done()
                except Exception:
                    pass

    async def _process_batch_item(self, predictions, video_frames) -> None:
        # Normalize to lists and align by index; skip None frames
        frames_list: List[VideoFrame] = (
            list(video_frames)
            if isinstance(video_frames, list)
            else ([video_frames] if video_frames is not None else [])
        )
        # predictions could be list or single dict
        if isinstance(predictions, list):
            preds_list = predictions
        else:
            preds_list = [predictions] * len(frames_list)
        tasks = []
        for idx, vf in enumerate(frames_list):
            if vf is None:
                continue
            pred = preds_list[idx] if idx < len(preds_list) else None
            tasks.append(self._limited_process_frame(pred, vf))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _limited_process_frame(self, predictions, video_frame: VideoFrame) -> None:
        if self._frame_semaphore is None:
            await self._on_prediction_single_frame(predictions, video_frame)
            return
        async with self._frame_semaphore:
            await self._on_prediction_single_frame(predictions, video_frame)

    def _get_camera_uuid(self, source_id: int) -> str:
        """Map source_id to camera_uuid."""
        if source_id < len(self.config.sources):
            return self.config.sources[source_id]["camera_uuid"]
        return f"unknown_{source_id}"

    def _get_tenant_id(self, camera_uuid: str) -> str:
        """Get tenant_id for camera."""
        for source in self.config.sources:
            if source["camera_uuid"] == camera_uuid:
                return source.get("tenant_id", "unknown")
        return "unknown"

    def _get_site_id(self, camera_uuid: str) -> str:
        """Get site_id for camera."""
        for source in self.config.sources:
            if source["camera_uuid"] == camera_uuid:
                return source.get("site_id", "unknown")
        return "unknown"

    def _calculate_fps(self, camera_uuid: str) -> float:
        """Calculate current FPS for camera."""
        # Return EMA-estimated ingest/processed FPS if available; fallback to pipeline watchdog FPS
        ema = self._fps_ema.get(camera_uuid)
        if isinstance(ema, (int, float)) and ema > 0:
            return float(ema)
        return float(self.config.max_fps)

    def _tick_fps(self, camera_uuid: str, now_ts: Optional[float] = None) -> float:
        """Update per-camera FPS EMA and gauge; return current EMA."""
        try:
            t = now_ts if now_ts is not None else time.time()
            last = self._fps_last_ts.get(camera_uuid)
            if last is not None:
                dt = max(1e-6, t - last)
                inst_fps = 1.0 / dt
                prev = self._fps_ema.get(camera_uuid, inst_fps)
                # EMA smoothing
                alpha = 0.2
                ema = (1 - alpha) * prev + alpha * inst_fps
                self._fps_ema[camera_uuid] = ema
                # Update gauge
            self.stream_fps.labels(**self._cam_labels(camera_uuid)).set(ema)
            self._fps_last_ts[camera_uuid] = t
            return float(self._fps_ema.get(camera_uuid, 0.0))
        except Exception:
            return 0.0

    def _calculate_e2e_latency(self, frame: VideoFrame) -> float:
        """Calculate end-to-end latency.

        Handles both naive and timezone-aware datetime objects from InferencePipeline.
        Assumes naive timestamps are in UTC (standard for video processing).
        """
        if hasattr(frame, "frame_timestamp") and frame.frame_timestamp:
            now = datetime.now(timezone.utc)

            if isinstance(frame.frame_timestamp, datetime):
                frame_ts = frame.frame_timestamp

                # Convert naive datetime to UTC-aware if needed
                # InferencePipeline typically provides naive timestamps
                if frame_ts.tzinfo is None:
                    frame_ts = frame_ts.replace(tzinfo=timezone.utc)

                return (now - frame_ts).total_seconds()

        return 0.1  # Default latency

    def _extract_camera_from_update(self, update: StatusUpdate) -> str:
        """Extract camera UUID from status update."""
        # Handle empty payload - this is likely a pipeline-wide event, not camera-specific
        if not update.payload:
            return None  # Signal this is not a camera-specific update

        # Check for multi-camera batch updates (contains data for all cameras)
        if "sources_id" in update.payload or "frames_ids" in update.payload:
            # This is a batch update for multiple cameras, not a single camera status change
            return (
                None  # Signal this should not trigger individual camera status changes
            )

        # Try to extract camera info from the update context or payload
        if hasattr(update, "context") and update.context:
            # Parse context string to extract camera info
            context_parts = update.context.split("/")
            for part in context_parts:
                if part.startswith("camera_"):
                    return part
                # Try to match against known camera UUIDs
                for source in self.config.sources:
                    if source["camera_uuid"] in part:
                        return source["camera_uuid"]

        # Fallback: try payload - check various possible keys
        payload_keys = ["camera_uuid", "camera_id", "source", "video_reference"]
        for key in payload_keys:
            if key in update.payload:
                value = update.payload[key]
                # If it's a source_id (integer), map it to camera_uuid
                if key == "source_id" or (
                    isinstance(value, int) and 0 <= value < len(self.config.sources)
                ):
                    return self._get_camera_uuid(value)
                # If it's a URL, try to match it
                elif isinstance(value, str):
                    for source in self.config.sources:
                        if source["url"] in value or source["camera_uuid"] in value:
                            return source["camera_uuid"]
                    # If it looks like a camera UUID directly
                    if value.startswith("cam-"):
                        return value

        # Try to infer from video reference URL in context
        if hasattr(update, "context") and update.context:
            for source in self.config.sources:
                if source["url"] in update.context:
                    return source["camera_uuid"]

        # Last resort: try to infer from source_id in payload (with better error handling)
        if "source_id" in update.payload:
            try:
                source_id = int(update.payload["source_id"])
                if 0 <= source_id < len(self.config.sources):
                    return self._get_camera_uuid(source_id)
            except (ValueError, TypeError):
                pass

        # Default fallback - log warning and return first camera
        logger.debug(f"Could not extract camera UUID from update: {update.payload}")
        return (
            self.config.sources[0]["camera_uuid"] if self.config.sources else "unknown"
        )

    def _map_pipeline_state(self, update: StatusUpdate) -> str:
        """Map InferencePipeline state to our state names."""
        # Map based on update event type and payload
        event_type = update.event_type.lower() if update.event_type else "unknown"

        # Enhanced state mapping for InferencePipeline events
        state_mapping = {
            # Source/stream events
            "source_connected": "STREAMING",  # When source successfully connects
            "source_running": "STREAMING",
            "source_disconnected": "DISCONNECTED",
            "source_error": "ERROR",
            "source_paused": "PAUSED",
            "source_ended": "DISCONNECTED",
            # Pipeline events
            "pipeline_started": "STREAMING",
            "pipeline_stopped": "DISCONNECTED",
            "pipeline_terminated": "DISCONNECTED",
            # Inference events
            "inference_started": "STREAMING",
            "inference_error": "ERROR",
            "inference_stopped": "DISCONNECTED",
            # Video frame events - these indicate successful streaming
            "frame_captured": "STREAMING",  # Camera is capturing frames
            "frame_consumed": "STREAMING",  # Pipeline is processing frames
            "frame_received": "STREAMING",
            "frame_dropped": "STREAMING",  # Even dropped frames mean stream is active
            "video_consumption_started": "STREAMING",
            "video_consumption_finished": "DISCONNECTED",
            # Connection events
            "connecting": "CONNECTING",
            "connected": "STREAMING",
            "reconnecting": "CONNECTING",
            "disconnected": "DISCONNECTED",
            # Error events
            "error": "ERROR",
            "timeout": "ERROR",
            "connection_error": "ERROR",
        }

        # First check payload for explicit state info
        if "state" in update.payload:
            payload_state = str(update.payload["state"]).upper()
            if payload_state in ["RUNNING", "STREAMING", "ACTIVE", "CONNECTED"]:
                return "STREAMING"
            elif payload_state in [
                "STOPPED",
                "ENDED",
                "TERMINATED",
                "DISCONNECTED",
                "FINISHED",
            ]:
                return "DISCONNECTED"
            elif payload_state in [
                "CONNECTING",
                "INITIALISING",
                "STARTING",
                "INITIALIZING",
            ]:
                return "CONNECTING"
            elif payload_state in ["PAUSED", "SUSPENDED"]:
                return "PAUSED"
            elif payload_state in ["ERROR", "FAILED", "TIMEOUT"]:
                return "ERROR"

        # Check for specific indicators in payload
        if "error" in update.payload or update.severity == UpdateSeverity.ERROR:
            return "ERROR"

        # Check if it's a connection/reconnection attempt
        if (
            "retry" in str(update.payload).lower()
            or "reconnect" in str(update.payload).lower()
        ):
            return "CONNECTING"

        # Use event type mapping
        mapped_state = state_mapping.get(event_type, "UNKNOWN")

        # Log unmapped event types for debugging (but not the frame events we now handle)
        if mapped_state == "UNKNOWN" and event_type not in [
            "unknown",
            "frame_captured",
            "frame_consumed",
            "frame_dropped",
        ]:
            logger.debug(
                f"Unmapped event type: {event_type}, payload: {update.payload}"
            )

        return mapped_state

    def start(self):
        """Start the worker."""
        logger.info("Starting Production Worker...")

        try:
            # Start Prometheus metrics server
            start_http_server(9108, addr="127.0.0.1")
            logger.info("Prometheus metrics server started on :9108")

            # Start event loop in thread first
            self.event_thread = threading.Thread(
                target=lambda: asyncio.run(self._run_event_loop()), daemon=True
            )
            self.event_thread.start()

            # Wait for event loop to be ready
            while not self.event_loop:
                time.sleep(0.1)

            # Initialize real InferencePipeline
            logger.info("Initializing InferencePipeline...")
            self.pipeline = self._create_inference_pipeline()
            logger.info("InferencePipeline created successfully")

            # Note: Motion detection is now handled in _on_prediction method
            # since InferencePipeline handles inference internally

            # Start the pipeline with background threading (CRITICAL FIX)
            logger.info("Starting InferencePipeline with background threading...")
            self.pipeline.start(use_main_thread=False)
            logger.info("InferencePipeline started - main thread NOT blocked")

            # Validate pipeline threads are running
            logger.info(
                f"Inference thread active: {self.pipeline._inference_thread and self.pipeline._inference_thread.is_alive()}"
            )
            logger.info(
                f"Dispatching thread active: {self.pipeline._dispatching_thread and self.pipeline._dispatching_thread.is_alive()}"
            )

            # Check if prediction callback is properly registered
            logger.info(
                f"Pipeline callback registered: {self.pipeline._on_prediction is not None}"
            )
            if hasattr(self.pipeline, "_on_prediction"):
                logger.info(
                    f"Callback function details: {self.pipeline._on_prediction}"
                )

            # Mark as ready after initial startup period
            logger.info("Waiting for cameras to stabilize...")
            time.sleep(15)  # Give more time for cameras to connect
            self.ready = True
            logger.info("Worker is ready - entering main loop")

            # Keep running until shutdown
            while self.running:
                time.sleep(1)

                # Update pipeline FPS metric
                if hasattr(self.pipeline, "_watchdog") and self.pipeline._watchdog:
                    try:
                        # Get FPS from pipeline watchdog if available
                        fps = getattr(
                            self.pipeline._watchdog, "fps", self.config.max_fps
                        )
                        self.pipeline_fps.labels(
                            runner_id=self.config.runner_id,
                            shard_id=self.config.shard_id,
                        ).set(fps)
                    except:
                        # Fallback to configured FPS
                        self.pipeline_fps.labels(
                            runner_id=self.config.runner_id,
                            shard_id=self.config.shard_id,
                        ).set(self.config.max_fps)

            # Graceful shutdown
            logger.info("Shutting down worker...")

            if self.pipeline:
                logger.info("Terminating InferencePipeline...")
                self.pipeline.terminate()
                self.pipeline.join()

            if self.event_thread and self.event_thread.is_alive():
                self.event_thread.join(timeout=5)

            logger.info("Worker shutdown complete")

        except Exception as e:
            logger.error(f"Fatal error in worker: {e}")
            raise

    def _create_inference_pipeline(self) -> InferencePipeline:
        """Create and configure the real InferencePipeline."""
        logger.info("Initializing InferencePipeline...")

        # Collect video URLs from sources
        video_urls = [source["url"] for source in self.config.sources]
        logger.info(f"Video sources: {video_urls}")

        # Initialize InferencePipeline with multiple sources and status handlers
        # pipeline = InferencePipeline.init(
        # model_id="yolov8n-640",  # Use a standard model for testing

        # Define sync wrapper for async prediction callback
        def sync_prediction_wrapper(predictions, video_frames):
            """Synchronous wrapper for async prediction callback."""
            # Handle both single frame and list of frames
            if isinstance(video_frames, list):
                logger.debug(
                    f"=== SYNC WRAPPER called with {len(video_frames)} frames ==="
                )
                for vf in video_frames:
                    if vf is None:
                        continue
                    try:
                        logger.debug(
                            f"  Frame: source_id={vf.source_id}, frame_id={vf.frame_id}"
                        )
                    except Exception:
                        pass
            else:
                try:
                    logger.debug(
                        f"=== SYNC WRAPPER called with single frame: source_id={video_frames.source_id}, frame_id={video_frames.frame_id} ==="
                    )
                except Exception:
                    logger.debug("=== SYNC WRAPPER called with single frame ===")

            logger.debug(f"Predictions type: {type(predictions)}")

            # Check if we have an event loop
            if not self.event_loop:
                logger.error("Event loop not available for prediction callback!")
                return

            # Enqueue batch to async workers without blocking sink
            self._enqueue_batch_from_sink(predictions, video_frames)

        logger.info("Creating InferencePipeline with workflow...")
        logger.info(f"Registering prediction callback: {sync_prediction_wrapper}")

        # TODO accept configuration parameters for workflow id and workspace name and api key
        pipeline = InferencePipeline.init_with_workflow(
            api_key="M64AKKSKeEZdY6LizsYO",
            workspace_name="xstar",
            # workflow_id="detect-count-and-visualize",
            workflow_id="detect-count-and-visualize-rf-detr-base",
            video_reference=video_urls,
            on_prediction=sync_prediction_wrapper,
            status_update_handlers=[self._on_status_update],
            max_fps=self.config.max_fps,
            workflows_thread_pool_workers=2,
            sink_mode=SinkMode.BATCH,
            batch_collection_timeout=float(os.getenv("BATCH_COLLECTION_TIMEOUT", "0.2")),
        )

        logger.info(f"InferencePipeline initialized with {len(video_urls)} sources")

        # Log pipeline configuration
        logger.info(f"Pipeline on_prediction callback: {pipeline._on_prediction}")
        logger.info(f"Pipeline has watchdog: {hasattr(pipeline, '_watchdog')}")
        logger.info(
            f"Pipeline has inference thread: {hasattr(pipeline, '_inference_thread')}"
        )

        # Install motion-first gating wrapper over pipeline's on_video_frame.
        # This avoids wasted inference when no motion or during cooldown.
        try:
            original_on_video_frame = getattr(pipeline, "_on_video_frame", None)
            if callable(original_on_video_frame):
                pipeline._on_video_frame = self._build_motion_gated_wrapper(original_on_video_frame)
                self._use_motion_gating_wrapper = True
                logger.info("Motion-gating wrapper installed on InferencePipeline.")
            else:
                logger.warning("Pipeline on_video_frame not found. Skipping motion gating wrapper installation.")
        except Exception as e:
            logger.error(f"Failed to install motion gating wrapper: {e}")

        # Set up status monitoring after pipeline creation
        self._setup_status_monitoring(pipeline)

        return pipeline

    def _setup_status_monitoring(self, pipeline: InferencePipeline):
        """Set up status monitoring for the pipeline."""
        # The InferencePipeline now automatically sends status updates via our handler
        logger.info("Status monitoring connected via pipeline status_update_handlers")

        # Initialize camera states as CONNECTING
        for source in self.config.sources:
            camera_uuid = source["camera_uuid"]
            self.camera_states[camera_uuid] = "CONNECTING"
            self.stream_up.labels(**self._cam_labels(camera_uuid)).set(0)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics and performance metrics."""
        stats = {
            "inference_decisions": self.inference_decisions.copy(),
            "total_frames": sum(self.frame_counts.values()),
            "cameras_optimized": len(self.optimized_motion_detectors),
            "cameras_fallback": 0,
        }

        # Calculate optimization metrics
        total_decisions = sum(self.inference_decisions.values())
        if total_decisions > 0:
            stats["inference_reduction_rate"] = (
                self.inference_decisions["skipped_no_motion"] / total_decisions
            )
            stats["motion_trigger_rate"] = (
                self.inference_decisions["motion_triggered"] / total_decisions
            )
            stats["timeout_trigger_rate"] = (
                self.inference_decisions["timeout_triggered"] / total_decisions
            )

        # Add motion detector stats
        stats["motion_detectors"] = {}
        for camera_uuid, detector in self.optimized_motion_detectors.items():
            stats["motion_detectors"][camera_uuid] = detector.get_stats()

        # Add timeout manager stats
        stats["timeout_managers"] = {}
        for camera_uuid, manager in self.timeout_managers.items():
            stats["timeout_managers"][camera_uuid] = manager.get_stats()

        # Add performance monitor stats
        stats["performance_monitors"] = {}
        for camera_uuid, monitor in self.performance_monitors.items():
            stats["performance_monitors"][camera_uuid] = monitor.get_stats()

        return stats


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m somba_pipeline.worker <config.json>")
        sys.exit(1)

    # Load configuration
    config_path = sys.argv[1]
    if not Path(config_path).exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        config = ShardConfig.from_json_file(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Start worker
    worker = ProductionWorker(config, config_path)
    worker.start()


if __name__ == "__main__":
    main()
