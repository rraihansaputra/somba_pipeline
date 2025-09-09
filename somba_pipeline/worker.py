"""
Production worker implementing the full Phase 2 specifications.
Wraps InferencePipeline with RabbitMQ, Prometheus, motion detection, and zones.
"""

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
import os
import hashlib

import aio_pika
import cv2
import numpy as np
from aiohttp import web
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Import InferencePipeline and related classes
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
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

        # Initialize individual motion detectors
        from .motion_detection import MotionDetector

        self.motion_detectors: Dict[str, MotionDetector] = {}

        for camera_uuid, camera_config in camera_configs.items():
            if camera_config.motion_gating.enabled:
                self.motion_detectors[camera_uuid] = MotionDetector(
                    camera_uuid=camera_uuid,
                    motion_config=camera_config.motion_gating,
                    zones=camera_config.zones,
                    frame_width=self.frame_width,
                    frame_height=self.frame_height,
                )

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

        # Event queues
        self.detection_queue = asyncio.Queue(maxsize=1000)
        self.status_queue = asyncio.Queue(maxsize=100)

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
        self.stream_up = Gauge(
            "stream_up", "Stream connectivity status", ["camera_uuid"]
        )
        self.last_frame_age = Gauge(
            "last_frame_age_seconds", "Age of last frame", ["camera_uuid"]
        )
        self.stream_fps = Gauge("stream_fps", "Stream FPS", ["camera_uuid"])
        self.pipeline_fps = Gauge(
            "pipeline_fps", "Pipeline FPS", ["runner_id", "shard_id"]
        )

        # Histograms
        self.inference_latency = Histogram(
            "inference_latency_seconds", "Inference latency", ["camera_uuid"]
        )
        self.e2e_latency = Histogram(
            "e2e_latency_seconds", "End-to-end latency", ["camera_uuid"]
        )

        # Counters
        self.stream_errors = Counter(
            "stream_errors_total", "Stream errors", ["camera_uuid", "code"]
        )
        self.detections_published = Counter(
            "detections_published_total", "Detections published", ["camera_uuid"]
        )

        # Zone-specific metrics from Phase 2 spec
        self.frames_total = Counter(
            "zones_frames_total", "Total frames processed", ["camera"]
        )
        self.frames_skipped_motion_total = Counter(
            "zones_frames_skipped_motion_total", "Frames skipped by motion", ["camera"]
        )
        self.detections_raw_total = Counter(
            "zones_detections_raw_total", "Raw detections before filtering", ["camera"]
        )
        self.detections_published_zone = Counter(
            "zones_detections_published_total",
            "Published detections",
            ["camera", "zone_id", "label"],
        )
        self.detections_dropped_total = Counter(
            "zones_detections_dropped_total",
            "Dropped detections",
            ["camera", "zone_id", "reason"],
        )
        self.zones_config_hash = Gauge(
            "zones_config_hash", "Zones config hash", ["camera"]
        )

        logger.info("Prometheus metrics initialized")

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
            
            # Reload motion detectors
            from .motion_detection import MotionDetector
            
            self.motion_detectors.clear()
            
            for camera_uuid, camera_config in camera_configs.items():
                if camera_config.motion_gating.enabled:
                    self.motion_detectors[camera_uuid] = MotionDetector(
                        camera_uuid=camera_uuid,
                        motion_config=camera_config.motion_gating,
                        zones=camera_config.zones,
                        frame_width=self.frame_width,
                        frame_height=self.frame_height,
                    )
            
            # Update config hash metrics
            for camera_uuid in camera_configs:
                config_hash = self.zone_attributor.get_zones_config_hash(camera_uuid)
                self.zones_config_hash.labels(camera=camera_uuid).set(config_hash)
            
            logger.info("Zone configurations reloaded successfully")
            
        except Exception as e:
            logger.error(f"Error reloading zones: {e}")

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.draining = True
        self.running = False

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

            # Update frame tracking
            self.last_frame_times[camera_uuid] = frame_timestamp
            self.frame_counts[camera_uuid] = self.frame_counts.get(camera_uuid, 0) + 1

            # Update metrics
            self.frames_total.labels(camera=camera_uuid).inc()

            # Apply motion detection if enabled for this camera
            skipped_by_motion = False

            if camera_uuid in self.motion_detectors:
                motion_detector = self.motion_detectors[camera_uuid]
                motion_result = motion_detector.detect_motion(video_frame.image)

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

                # 2) Build motion state for overlays
                motion_on = bool(motion_result.motion_detected)
                raw_mask = getattr(motion_result, "mask", None)
                mask = None if isinstance(raw_mask, (bool, np.bool_, int, float)) else raw_mask
                raw_zone_hits = getattr(motion_result, "zone_hits", None)

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
                self._motion_debug[camera_uuid] = {
                    "mask": mask,
                    "contours": contours,
                    "active": motion_on,
                    "cooldown": int(getattr(motion_result, "cooldown_frames_left", 0)),
                    "gate_reason": str(getattr(motion_result, "gate_reason", "")),
                    "min_area_px": int(getattr(motion_result, "min_area_px", 0)),
                }

                if not motion_result.motion_detected:
                    skipped_by_motion = True
                    self.frames_skipped_motion_total.labels(camera=camera_uuid).inc()
                    self._last_infer_state[camera_uuid] = "SKIP"

                    # Render and publish debug frame (throttled) - always show zones even when no motion
                    self._publish_debug(
                        cam=camera_uuid,
                        frame_bgr=video_frame.image,
                        predictions=None,
                        skipped_by_motion=True,
                        fps=self._calculate_fps(camera_uuid),
                    )
                    # IMPORTANT: still skip inference
                    return

                logger.debug(
                    f"Motion detected {camera_uuid}: "
                    f"{motion_result.pixels_changed}px in {motion_result.motion_in_include_area}px include area"
                )

            # Process detections through zone attribution
            raw_detections = predictions.get("predictions", []) if predictions else []
            if len(raw_detections.class_id):
                logger.info(f"{raw_detections=}")
            self.detections_raw_total.labels(camera=camera_uuid).inc(
                len(raw_detections)
            )

            # Apply zone attribution and filtering
            published_objects, zone_stats = (
                self.zone_attributor.process_camera_detections(
                    camera_uuid, raw_detections
                )
            )

            # Update zone metrics
            for obj in published_objects:
                self.detections_published_zone.labels(
                    camera=camera_uuid,
                    zone_id=str(obj.primary_zone_id),
                    label=obj.label,
                ).inc()

            # Update dropped metrics
            for zone_id_str, stats in zone_stats.items():
                if stats["dropped"] > 0:
                    self.detections_dropped_total.labels(
                        camera=camera_uuid, zone_id=zone_id_str, reason="zone_filter"
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
            self._publish_debug(
                cam=camera_uuid,
                frame_bgr=video_frame.image,
                predictions=raw_detections,
                skipped_by_motion=False,
                fps=self._calculate_fps(camera_uuid),
            )

            # Update published metrics (fix reference)
            self.detections_published.labels(camera_uuid=camera_uuid).inc()

            # Update inference latency
            if "time" in predictions:
                self.inference_latency.labels(camera_uuid=camera_uuid).observe(
                    predictions["time"]
                )

            # Update e2e latency
            e2e_latency = self._calculate_e2e_latency(video_frame)
            self.e2e_latency.labels(camera_uuid=camera_uuid).observe(e2e_latency)

            logger.debug(
                f"Processed prediction for {camera_uuid}: "
                f"{len(published_objects)} objects published"
            )

        except Exception as e:
            logger.error(
                f"Error processing prediction for {camera_uuid}: {e}", exc_info=True
            )
            self.stream_errors.labels(
                camera_uuid=camera_uuid, code="PREDICTION_ERROR"
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
            for video_frame in video_frames:
                # Extract predictions for this specific frame/source
                frame_predictions = None
                if predictions and isinstance(predictions, list):
                    # If predictions is also a list, match by index
                    source_idx = video_frame.source_id
                    if source_idx < len(predictions):
                        frame_predictions = predictions[source_idx]
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
            self.stream_up.labels(camera_uuid=camera_uuid).set(
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

        self.stream_errors.labels(camera_uuid=camera_uuid, code=error_code).inc()

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
                        self.last_frame_age.labels(camera_uuid=camera_uuid).set(
                            frame_age
                        )

                    # Get zone statistics
                    zone_attributor = self.zone_attributor.attributors.get(camera_uuid)
                    motion_detector = self.motion_detectors.get(camera_uuid)

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
                        motion_stats = (
                            motion_detector.get_stats() if motion_detector else {}
                        )

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

            # Derive camera geometry/fps from your known state (fallbacks are safe)
            # meta = await request.app["get_camera_meta"](cam)  # you may implement this; see below
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
            web.post("/debug/{camera_uuid}/stop", debug_stop),
        ])

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, "127.0.0.1", 8080)
        await site.start()

        logger.info("Health server started on http://127.0.0.1:8080")

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
        
        # Add config watcher if enabled
        if self.config_path:
            self.config_watcher_task = asyncio.create_task(self._watch_config_changes())
            tasks.append(self.config_watcher_task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_debug_server(self):
        pass

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
        # Simple FPS calculation based on frame count and time
        # In practice, would use a sliding window
        return float(self.config.max_fps)

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
                    logger.debug(
                        f"  Frame: source_id={vf.source_id}, frame_id={vf.frame_id}"
                    )
            else:
                logger.debug(
                    f"=== SYNC WRAPPER called with single frame: source_id={video_frames.source_id}, frame_id={video_frames.frame_id} ==="
                )

            logger.debug(f"Predictions type: {type(predictions)}")

            # Check if we have an event loop
            if not self.event_loop:
                logger.error("Event loop not available for prediction callback!")
                return

            # Submit async work to event loop
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._on_prediction(predictions, video_frames), self.event_loop
                )
                logger.debug(
                    f"Submitted async prediction to event loop, future: {future}"
                )
                # Don't wait for the future - let it run async
            except Exception as e:
                logger.error(
                    f"Failed to submit prediction to event loop: {e}", exc_info=True
                )

        logger.info("Creating InferencePipeline with workflow...")
        logger.info(f"Registering prediction callback: {sync_prediction_wrapper}")

        # TODO accept configuration parameters for workflow id and workspace name and api key
        pipeline = InferencePipeline.init_with_workflow(
            api_key="M64AKKSKeEZdY6LizsYO",
            workspace_name="xstar",
            workflow_id="detect-count-and-visualize",
            video_reference=video_urls,
            on_prediction=sync_prediction_wrapper,
            status_update_handlers=[self._on_status_update],
            max_fps=self.config.max_fps,
        )

        logger.info(f"InferencePipeline initialized with {len(video_urls)} sources")

        # Log pipeline configuration
        logger.info(f"Pipeline on_prediction callback: {pipeline._on_prediction}")
        logger.info(f"Pipeline has watchdog: {hasattr(pipeline, '_watchdog')}")
        logger.info(
            f"Pipeline has inference thread: {hasattr(pipeline, '_inference_thread')}"
        )

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
            self.stream_up.labels(camera_uuid=camera_uuid).set(0)


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
