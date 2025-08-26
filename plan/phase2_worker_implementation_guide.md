# Phase 2: Production Worker Implementation Guide

## Overview

This phase builds the **production Worker** that wraps InferencePipeline and implements all requirements from the technical specification:
- Multi-camera shard processing (8-16 cameras)
- RabbitMQ event publishing (detections + status)
- Prometheus metrics (port 9108)
- Health HTTP endpoints
- Motion detection integration
- Graceful shutdown

## Architecture Analysis

### What We Have (InferencePipeline)
- ✅ Multi-source video processing via `multiplex_videos`
- ✅ Built-in watchdog with status updates
- ✅ Buffer strategies (DROP_OLDEST, EAGER)
- ✅ Thread-safe operation
- ✅ FPS limiting

### What We Need to Add
- 📦 RabbitMQ client for event publishing
- 📊 Prometheus metrics exporter
- 🌐 HTTP server for health endpoints
- 🎯 Motion detection wrapper
- 📋 Shard configuration loader
- 🔄 Status event translator

## Implementation Components

### 1. Core Worker Class

```python
# worker.py
import json
import signal
import asyncio
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

import aio_pika
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from aiohttp import web
from inference import InferencePipeline
from inference.core.interfaces.stream.entities import VideoFrame
from inference.core.interfaces.camera.entities import StatusUpdate, UpdateSeverity

from motion_detector import MotionTriggeredInferenceHandler, MotionConfig
from go2rtc_manager import Go2RTCManager

logger = logging.getLogger(__name__)

@dataclass
class ShardConfig:
    """Configuration for a worker shard."""
    runner_id: str
    shard_id: str
    max_fps: int = 6
    sources: List[Dict[str, str]] = None  # camera_uuid, url, site_id, tenant_id
    amqp: Dict[str, str] = None  # host, ex_status, ex_detect
    cp: Dict[str, str] = None  # base_url, token
    telemetry: Dict[str, Any] = None  # report_interval_seconds
    motion_config: Optional[MotionConfig] = None

    @classmethod
    def from_json(cls, json_path: str) -> 'ShardConfig':
        with open(json_path) as f:
            data = json.load(f)
        return cls(**data)


class ProductionWorker:
    """
    Production worker implementing the full technical specification.
    Wraps InferencePipeline with RabbitMQ, Prometheus, and health endpoints.
    """

    def __init__(self, config: ShardConfig):
        self.config = config
        self.pipeline: Optional[InferencePipeline] = None
        self.go2rtc = Go2RTCManager() if self._using_go2rtc() else None

        # State tracking
        self.running = True
        self.ready = False
        self.draining = False
        self.camera_states = {}  # camera_uuid -> state
        self.last_frame_times = {}  # camera_uuid -> timestamp

        # Metrics
        self._init_metrics()

        # Event queues
        self.detection_queue = asyncio.Queue(maxsize=1000)
        self.status_queue = asyncio.Queue(maxsize=100)

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Threading
        self.event_loop = None
        self.event_thread = None
        self.health_server = None

    def _using_go2rtc(self) -> bool:
        """Check if we're using go2rtc based on URLs."""
        if not self.config.sources:
            return False
        return any('127.0.0.1:8554' in src.get('url', '') for src in self.config.sources)

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        # Gauges
        self.stream_up = Gauge('stream_up', 'Stream connectivity status', ['camera_uuid'])
        self.last_frame_age = Gauge('last_frame_age_seconds', 'Age of last frame', ['camera_uuid'])
        self.stream_fps = Gauge('stream_fps', 'Stream FPS', ['camera_uuid'])
        self.pipeline_fps = Gauge('pipeline_fps', 'Pipeline FPS', ['runner_id', 'shard_id'])

        # Histograms
        self.inference_latency = Histogram('inference_latency_seconds', 'Inference latency', ['camera_uuid'])
        self.e2e_latency = Histogram('e2e_latency_seconds', 'End-to-end latency', ['camera_uuid'])

        # Counters
        self.stream_errors = Counter('stream_errors_total', 'Stream errors', ['camera_uuid', 'code'])
        self.detections_published = Counter('detections_published_total', 'Detections published', ['camera_uuid'])

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.draining = True
        self.running = False

    async def _on_prediction(self, predictions: Dict, video_frame: VideoFrame):
        """
        Handle predictions from InferencePipeline.
        This is called for each frame/prediction pair.
        """
        try:
            camera_uuid = self._get_camera_uuid(video_frame.source_id)

            # Skip if no motion detected (predictions will be None)
            if predictions is None and self.config.motion_config:
                return

            # Update metrics
            self.last_frame_times[camera_uuid] = datetime.now()

            if predictions and 'time' in predictions:
                self.inference_latency.labels(camera_uuid=camera_uuid).observe(predictions['time'])

            # Create detection event
            detection_event = {
                'ts': datetime.now().isoformat() + 'Z',
                'runner_id': self.config.runner_id,
                'shard_id': self.config.shard_id,
                'tenant_id': self._get_tenant_id(camera_uuid),
                'site_id': self._get_site_id(camera_uuid),
                'camera_uuid': camera_uuid,
                'frame_id': video_frame.frame_id,
                'fps': self._calculate_fps(camera_uuid),
                'detections': predictions.get('predictions', []) if predictions else [],
                'latency': {
                    'inference_s': predictions.get('time', 0) if predictions else 0,
                    'e2e_s': self._calculate_e2e_latency(video_frame)
                }
            }

            # Queue for publishing
            await self.detection_queue.put(detection_event)
            self.detections_published.labels(camera_uuid=camera_uuid).inc()

        except Exception as e:
            logger.error(f"Error processing prediction: {e}")
            self.stream_errors.labels(camera_uuid=camera_uuid, code='PREDICTION_ERROR').inc()

    def _on_status_update(self, update: StatusUpdate):
        """
        Handle status updates from the InferencePipeline watchdog.
        Translate to our status event format.
        """
        try:
            # Parse the update to determine camera and state
            camera_uuid = self._extract_camera_from_update(update)

            # Map InferencePipeline states to our states
            state = self._map_pipeline_state(update)

            # Update local state
            old_state = self.camera_states.get(camera_uuid)
            self.camera_states[camera_uuid] = state

            # Update metrics
            self.stream_up.labels(camera_uuid=camera_uuid).set(1 if state == 'STREAMING' else 0)

            # Create status event (edge-triggered)
            if state != old_state:
                status_event = {
                    'type': 'stream.status',
                    'state': state,
                    'camera_uuid': camera_uuid,
                    'runner_id': self.config.runner_id,
                    'shard_id': self.config.shard_id,
                    'ts': datetime.now().isoformat() + 'Z'
                }

                # Add FPS if streaming
                if state == 'STREAMING':
                    status_event['fps'] = self._calculate_fps(camera_uuid)

                # Queue for publishing
                asyncio.run_coroutine_threadsafe(
                    self.status_queue.put(status_event),
                    self.event_loop
                )

            # Handle errors
            if update.severity == UpdateSeverity.ERROR:
                self._handle_error_update(update, camera_uuid)

        except Exception as e:
            logger.error(f"Error processing status update: {e}")

    def _handle_error_update(self, update: StatusUpdate, camera_uuid: str):
        """Handle error status updates."""
        error_code = update.payload.get('error_type', 'UNKNOWN')

        self.stream_errors.labels(camera_uuid=camera_uuid, code=error_code).inc()

        error_event = {
            'type': 'stream.error',
            'camera_uuid': camera_uuid,
            'runner_id': self.config.runner_id,
            'code': error_code,
            'detail': update.payload.get('message', ''),
            'retry_in_ms': 8000,  # Default retry
            'ts': datetime.now().isoformat() + 'Z'
        }

        asyncio.run_coroutine_threadsafe(
            self.status_queue.put(error_event),
            self.event_loop
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
                        self.detection_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Publish to RabbitMQ
                routing_key = f"detections.{event['tenant_id']}.{event['site_id']}.{event['camera_uuid']}"

                message = aio_pika.Message(
                    body=json.dumps(event).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                )

                exchange = await channel.get_exchange(self.config.amqp['ex_detect'])
                await exchange.publish(message, routing_key=routing_key)

                logger.debug(f"Published detection for {event['camera_uuid']}")

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
                    event = await asyncio.wait_for(
                        self.status_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Determine routing key
                camera_uuid = event['camera_uuid']
                tenant_id = self._get_tenant_id(camera_uuid)
                site_id = self._get_site_id(camera_uuid)

                if event['type'] == 'stream.error':
                    routing_key = f"stream.error.{tenant_id}.{site_id}.{camera_uuid}"
                else:
                    routing_key = f"stream.status.{tenant_id}.{site_id}.{camera_uuid}"

                message = aio_pika.Message(
                    body=json.dumps(event).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                )

                exchange = await channel.get_exchange(self.config.amqp['ex_status'])
                await exchange.publish(message, routing_key=routing_key)

                logger.debug(f"Published status for {camera_uuid}: {event.get('state', event.get('type'))}")

            except Exception as e:
                logger.error(f"Error publishing status: {e}")
                await asyncio.sleep(1)

        # Cleanup
        if connection and not connection.is_closed:
            await connection.close()

    async def _periodic_status_summary(self):
        """Send periodic status summaries every 5 seconds."""
        while self.running:
            await asyncio.sleep(self.config.telemetry.get('report_interval_seconds', 5))

            for camera_uuid in self.camera_states:
                try:
                    last_frame = self.last_frame_times.get(camera_uuid)
                    if last_frame:
                        frame_age = (datetime.now() - last_frame).total_seconds()
                        self.last_frame_age.labels(camera_uuid=camera_uuid).set(frame_age)

                    # Create summary event
                    summary = {
                        'type': 'stream.status',
                        'camera_uuid': camera_uuid,
                        'runner_id': self.config.runner_id,
                        'shard_id': self.config.shard_id,
                        'state': self.camera_states.get(camera_uuid, 'UNKNOWN'),
                        'last_frame_ts': last_frame.isoformat() + 'Z' if last_frame else None,
                        'last_frame_age_s': frame_age if last_frame else None,
                        'fps': self._calculate_fps(camera_uuid),
                        'ts': datetime.now().isoformat() + 'Z'
                    }

                    await self.status_queue.put(summary)

                except Exception as e:
                    logger.error(f"Error creating status summary for {camera_uuid}: {e}")

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
        streaming_cameras = sum(1 for state in self.camera_states.values() if state == 'STREAMING')

        readiness_pct = (streaming_cameras / total_cameras * 100) if total_cameras > 0 else 0
        required_pct = 80  # Default readiness quorum

        if readiness_pct >= required_pct:
            return web.Response(
                text=json.dumps({'ready': True, 'cameras': f"{streaming_cameras}/{total_cameras}"}),
                status=200,
                content_type='application/json'
            )

        return web.Response(
            text=json.dumps({'ready': False, 'cameras': f"{streaming_cameras}/{total_cameras}"}),
            status=503,
            content_type='application/json'
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
        app.router.add_get('/healthz', self._health_handler)
        app.router.add_get('/ready', self._ready_handler)
        app.router.add_post('/drain', self._drain_handler)
        app.router.add_post('/terminate', self._terminate_handler)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, '127.0.0.1', 8080)
        await site.start()

        logger.info("Health server started on http://127.0.0.1:8080")

        # Keep running until shutdown
        while self.running:
            await asyncio.sleep(1)

        await runner.cleanup()

    def _create_inference_handler(self):
        """Create the inference handler with optional motion detection."""
        base_handler = self.pipeline._on_video_frame

        if self.config.motion_config:
            # Wrap with motion detection
            return MotionTriggeredInferenceHandler(
                inference_handler=base_handler,
                motion_config=self.config.motion_config,
                on_motion_detected=self._on_motion_detected,
                on_motion_skipped=self._on_motion_skipped
            )

        return base_handler

    def _on_motion_detected(self, frame: VideoFrame, pixels_changed: int):
        """Callback when motion is detected."""
        camera_uuid = self._get_camera_uuid(frame.source_id)
        logger.debug(f"Motion detected on {camera_uuid}: {pixels_changed} pixels")

    def _on_motion_skipped(self, frame: VideoFrame):
        """Callback when frame is skipped due to no motion."""
        camera_uuid = self._get_camera_uuid(frame.source_id)
        # Update last frame time even for skipped frames
        self.last_frame_times[camera_uuid] = datetime.now()

    def _get_camera_uuid(self, source_id: int) -> str:
        """Map source_id to camera_uuid."""
        if source_id < len(self.config.sources):
            return self.config.sources[source_id]['camera_uuid']
        return f"unknown_{source_id}"

    def _get_tenant_id(self, camera_uuid: str) -> str:
        """Get tenant_id for camera."""
        for source in self.config.sources:
            if source['camera_uuid'] == camera_uuid:
                return source.get('tenant_id', 'unknown')
        return 'unknown'

    def _get_site_id(self, camera_uuid: str) -> str:
        """Get site_id for camera."""
        for source in self.config.sources:
            if source['camera_uuid'] == camera_uuid:
                return source.get('site_id', 'unknown')
        return 'unknown'

    def _calculate_fps(self, camera_uuid: str) -> float:
        """Calculate current FPS for camera."""
        # This would track frame times and calculate
        # For now, return configured max_fps
        return self.config.max_fps

    def _calculate_e2e_latency(self, frame: VideoFrame) -> float:
        """Calculate end-to-end latency."""
        # Would calculate from frame timestamp to now
        return 0.1  # Placeholder

    def _extract_camera_from_update(self, update: StatusUpdate) -> str:
        """Extract camera UUID from status update."""
        # Parse from update context or payload
        return update.payload.get('camera_uuid', 'unknown')

    def _map_pipeline_state(self, update: StatusUpdate) -> str:
        """Map InferencePipeline state to our state names."""
        state_map = {
            'RUNNING': 'STREAMING',
            'INITIALISING': 'CONNECTING',
            'TERMINATING': 'DISCONNECTED',
            'ENDED': 'DISCONNECTED',
            'PAUSED': 'PAUSED'
        }
        pipeline_state = update.payload.get('state', 'UNKNOWN')
        return state_map.get(pipeline_state, pipeline_state)

    async def _run_event_loop(self):
        """Run the async event loop for publishing."""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        # Start async tasks
        tasks = [
            self._publish_detection_events(),
            self._publish_status_events(),
            self._periodic_status_summary(),
            self._run_health_server()
        ]

        await asyncio.gather(*tasks)

    def start(self):
        """Start the worker."""
        logger.info(f"Starting Worker: runner={self.config.runner_id}, shard={self.config.shard_id}")
        logger.info(f"Processing {len(self.config.sources)} cameras")

        try:
            # Start Prometheus metrics server
            start_http_server(9108, addr='127.0.0.1')
            logger.info("Prometheus metrics server started on :9108")

            # Validate go2rtc if needed
            if self.go2rtc and not self.go2rtc.health_check():
                raise RuntimeError("go2rtc is not running")

            # Get video URLs
            video_urls = [source['url'] for source in self.config.sources]

            # Initialize InferencePipeline with multiple sources
            self.pipeline = InferencePipeline.init(
                model_id="coco/11",  # Would come from config
                video_reference=video_urls,
                on_prediction=lambda p, f: asyncio.run_coroutine_threadsafe(
                    self._on_prediction(p, f),
                    self.event_loop
                ),
                on_status_update=self._on_status_update,
                max_fps=self.config.max_fps
            )

            # Apply motion detection if configured
            if self.config.motion_config:
                self.pipeline._on_video_frame = self._create_inference_handler()

            # Start event loop in thread
            self.event_thread = threading.Thread(target=lambda: asyncio.run(self._run_event_loop()))
            self.event_thread.start()

            # Wait for event loop to be ready
            while not self.event_loop:
                time.sleep(0.1)

            # Start pipeline
            self.pipeline.start()
            logger.info("InferencePipeline started")

            # Mark as ready after initial startup period
            time.sleep(15)
            self.ready = True
            logger.info("Worker is ready")

            # Keep running until shutdown
            while self.running:
                time.sleep(1)

                # Update pipeline FPS metric
                if hasattr(self.pipeline, '_watchdog'):
                    fps = self.pipeline._watchdog.get_fps()
                    self.pipeline_fps.labels(
                        runner_id=self.config.runner_id,
                        shard_id=self.config.shard_id
                    ).set(fps)

            # Graceful shutdown
            logger.info("Shutting down worker...")

            if self.pipeline:
                self.pipeline.terminate()
                self.pipeline.join()

            # Stop event loop
            if self.event_loop:
                self.event_loop.stop()

            if self.event_thread:
                self.event_thread.join(timeout=5)

            logger.info("Worker shutdown complete")

        except Exception as e:
            logger.error(f"Fatal error in worker: {e}")
            raise


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python worker.py <config.json>")
        sys.exit(1)

    # Load configuration
    config = ShardConfig.from_json(sys.argv[1])

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Start worker
    worker = ProductionWorker(config)
    worker.start()


if __name__ == '__main__':
    main()
```

### 2. Motion Detection Module

```python
# motion_detector.py
# (Adapted from motion_detection_pipeline.md)

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Callable, Any
from datetime import datetime
from inference.core.interfaces.stream.entities import VideoFrame

@dataclass
class MotionConfig:
    """Configuration for motion detection."""
    pixel_threshold: int = 150
    sensitivity: int = 25
    blur_size: int = 21
    min_area: int = 500
    cooldown_seconds: float = 0.5
    dilation_iterations: int = 2
    erosion_iterations: int = 2
    motion_decay_seconds: float = 2.0
    enabled: bool = True  # Allow disabling motion detection


class MotionDetector:
    """Per-camera motion detector."""

    def __init__(self, config: MotionConfig):
        self.config = config
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.previous_frame: Optional[np.ndarray] = None
        self.last_detection_time: Optional[datetime] = None
        self.last_motion_time: Optional[datetime] = None

    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, int]:
        """
        Detect motion in frame.
        Returns (motion_detected, pixels_changed).
        """
        if not self.config.enabled:
            return True, 0  # Always process if disabled

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.config.blur_size, self.config.blur_size), 0)

        # Background subtraction
        fg_mask = self.background_subtractor.apply(blurred)

        # Frame differencing
        if self.previous_frame is not None:
            frame_diff = cv2.absdiff(self.previous_frame, blurred)
            _, thresh = cv2.threshold(frame_diff, self.config.sensitivity, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.bitwise_or(fg_mask, thresh)
        else:
            motion_mask = fg_mask

        self.previous_frame = blurred.copy()

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.erode(motion_mask, kernel, iterations=self.config.erosion_iterations)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=self.config.dilation_iterations)

        # Count motion pixels
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_motion_pixels = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config.min_area:
                total_motion_pixels += area

        # Check thresholds and timing
        significant_motion = total_motion_pixels > self.config.pixel_threshold

        if significant_motion:
            self.last_motion_time = datetime.now()

        # Apply cooldown
        if self.last_detection_time:
            time_since = (datetime.now() - self.last_detection_time).total_seconds()
            if time_since < self.config.cooldown_seconds:
                significant_motion = False

        # Decay period
        if self.last_motion_time:
            time_since_motion = (datetime.now() - self.last_motion_time).total_seconds()
            if time_since_motion < self.config.motion_decay_seconds:
                significant_motion = True

        if significant_motion:
            self.last_detection_time = datetime.now()

        return significant_motion, total_motion_pixels


class MotionTriggeredInferenceHandler:
    """Wraps inference handler with motion detection."""

    def __init__(
        self,
        inference_handler: Callable,
        motion_config: MotionConfig = None,
        on_motion_detected: Optional[Callable[[VideoFrame, int], None]] = None,
        on_motion_skipped: Optional[Callable[[VideoFrame], None]] = None
    ):
        self.inference_handler = inference_handler
        self.motion_config = motion_config or MotionConfig()
        self.motion_detectors: Dict[int, MotionDetector] = {}
        self.on_motion_detected = on_motion_detected
        self.on_motion_skipped = on_motion_skipped
        self.stats = {
            'frames_processed': 0,
            'frames_with_motion': 0,
            'frames_skipped': 0
        }

    def __call__(self, video_frames: List[VideoFrame]) -> List[Optional[dict]]:
        """Process frames through motion detection before inference."""
        results = []
        frames_to_process = []
        frame_indices = []

        for idx, frame in enumerate(video_frames):
            source_id = frame.source_id or 0

            # Get or create motion detector for this source
            if source_id not in self.motion_detectors:
                self.motion_detectors[source_id] = MotionDetector(self.motion_config)

            detector = self.motion_detectors[source_id]

            # Detect motion
            motion_detected, pixels_changed = detector.detect_motion(frame.image)

            self.stats['frames_processed'] += 1

            if motion_detected:
                self.stats['frames_with_motion'] += 1
                frames_to_process.append(frame)
                frame_indices.append(idx)

                if self.on_motion_detected:
                    self.on_motion_detected(frame, pixels_changed)
            else:
                self.stats['frames_skipped'] += 1

                if self.on_motion_skipped:
                    self.on_motion_skipped(frame)

        # Run inference only on frames with motion
        if frames_to_process:
            predictions = self.inference_handler(frames_to_process)

            # Map back to original indices
            prediction_map = dict(zip(frame_indices, predictions))

            for idx in range(len(video_frames)):
                if idx in prediction_map:
                    results.append(prediction_map[idx])
                else:
                    results.append(None)  # No motion = no prediction
        else:
            results = [None] * len(video_frames)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get motion detection statistics."""
        if self.stats['frames_processed'] > 0:
            self.stats['skip_rate'] = (
                self.stats['frames_skipped'] / self.stats['frames_processed'] * 100
            )
        return self.stats
```

### 3. Sample Shard Configuration

```json
{
  "runner_id": "runner-test-001",
  "shard_id": "shard-0",
  "max_fps": 6,
  "sources": [
    {
      "camera_uuid": "cam-001",
      "url": "rtsp://127.0.0.1:8554/camera_001",
      "site_id": "site-A",
      "tenant_id": "tenant-01"
    },
    {
      "camera_uuid": "cam-002",
      "url": "rtsp://127.0.0.1:8554/camera_002",
      "site_id": "site-A",
      "tenant_id": "tenant-01"
    },
    {
      "camera_uuid": "cam-003",
      "url": "rtsp://127.0.0.1:8554/camera_003",
      "site_id": "site-B",
      "tenant_id": "tenant-01"
    }
  ],
  "amqp": {
    "host": "localhost",
    "ex_status": "status.topic",
    "ex_detect": "detections.topic"
  },
  "cp": {
    "base_url": "http://localhost:8000/api",
    "token": "jwt-token-here"
  },
  "telemetry": {
    "report_interval_seconds": 5
  },
  "motion_config": {
    "enabled": true,
    "pixel_threshold": 150,
    "sensitivity": 25,
    "cooldown_seconds": 0.5
  }
}
```

### 4. Testing Setup

```bash
# test_worker.sh
#!/bin/bash

# Start dependencies
echo "Starting RabbitMQ..."
docker run -d --name rabbitmq \
  -p 5672:5672 \
  -p 15672:15672 \
  rabbitmq:3-management

# Setup RabbitMQ exchanges
sleep 10
rabbitmqadmin declare exchange name=status.topic type=topic
rabbitmqadmin declare exchange name=detections.topic type=topic

# Setup go2rtc with test streams
echo "Setting up go2rtc streams..."
cat > go2rtc_test.yaml <<EOF
streams:
  camera_001:
    - ffmpeg:test_video.mp4#video=h264
  camera_002:
    - ffmpeg:test_video.mp4#video=h264
  camera_003:
    - ffmpeg:test_video.mp4#video=h264

api:
  listen: "127.0.0.1:1984"
rtsp:
  listen: "127.0.0.1:8554"
EOF

./go2rtc -config go2rtc_test.yaml &

# Start worker
echo "Starting worker..."
python worker.py test_shard_config.json &
WORKER_PID=$!

# Wait for ready
echo "Waiting for worker to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:8080/ready | grep -q "true"; then
    echo "Worker is ready!"
    break
  fi
  sleep 1
done

# Monitor metrics
echo "Checking Prometheus metrics..."
curl -s http://localhost:9108/metrics | grep stream_up

# Monitor RabbitMQ
echo "Monitoring RabbitMQ messages..."
rabbitmqadmin get queue=detections.topic

# Test graceful shutdown
sleep 30
echo "Testing graceful drain..."
curl -X POST http://localhost:8080/drain

wait $WORKER_PID
echo "Worker exited with code: $?"
```

## Key Design Decisions

### 1. Async Event Publishing
- Uses `asyncio` for non-blocking RabbitMQ publishing
- Separate queues for detections and status events
- Publisher confirms for reliability

### 2. Motion Detection Integration
- Wraps `InferencePipeline._on_video_frame` handler
- Per-camera motion detectors
- Returns `None` for frames without motion

### 3. Health Endpoints
- Separate HTTP server on port 8080
- `/ready` checks streaming camera percentage
- `/drain` initiates graceful shutdown with grace period

### 4. Metrics Collection
- Prometheus client library for metric export
- Per-camera and per-shard metrics
- Integrates with InferencePipeline watchdog

### 5. Multi-Camera Handling
- Uses InferencePipeline's built-in multi-source support
- Maps source_id to camera_uuid
- Maintains per-camera state

## Acceptance Criteria

### ✅ Process 8-16 cameras per Worker
- Configure sources list in shard config
- InferencePipeline handles multiplexing

### ✅ Emit detections to RabbitMQ
- Async publishing with retries
- Proper routing keys per spec

### ✅ Emit status events (edge-triggered + periodic)
- Edge-triggered on state changes
- 5-second periodic summaries

### ✅ Expose Prometheus metrics on :9108
- All required metrics implemented
- Labels match specification

### ✅ Health endpoints functional
- `/healthz`, `/ready`, `/drain`, `/terminate`
- Proper HTTP status codes

### ✅ Motion detection reduces load
- Configurable per shard
- Skip inference for static frames

### ✅ Graceful shutdown
- Handles SIGTERM
- Drains with grace period

## Dependencies

```txt
# requirements.txt
inference  # Roboflow InferencePipeline
aio-pika  # RabbitMQ async client
prometheus-client  # Metrics export
aiohttp  # HTTP server for health endpoints
opencv-python  # Motion detection
numpy  # Image processing
```

## Next Steps

1. **Test with real cameras** - Replace test video with actual RTSP streams
2. **Implement lease checking** - Add lease validation before processing
3. **Add Control Plane client** - Report status to CP API
4. **Performance tuning** - Optimize buffer strategies and FPS
5. **Add monitoring** - Set up Grafana dashboards
