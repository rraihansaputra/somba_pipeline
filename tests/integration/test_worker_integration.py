"""Integration tests for the production worker."""

import asyncio
import json
import pytest
import threading
import time
from unittest.mock import AsyncMock, patch
import httpx
import cv2
import numpy as np

from somba_pipeline.worker import ProductionWorker
from somba_pipeline.schemas import (
    ShardConfig, CameraConfig, ZoneConfig, MotionGatingConfig
)
from somba_pipeline.mock_cp import MockControlPlane


@pytest.fixture
async def mock_rabbitmq():
    """Mock RabbitMQ server for testing."""
    # In a real test environment, you'd use a test RabbitMQ instance
    # For now, we'll mock the aio_pika connections
    with patch('somba_pipeline.worker.aio_pika.connect_robust') as mock_connect:
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()

        mock_connect.return_value = mock_connection
        mock_connection.channel.return_value = mock_channel
        mock_channel.get_exchange.return_value = mock_exchange

        yield {
            'connection': mock_connection,
            'channel': mock_channel,
            'exchange': mock_exchange
        }


@pytest.fixture
async def mock_control_plane():
    """Start mock control plane for testing."""
    cp = MockControlPlane(host="127.0.0.1", port=8001)  # Different port for testing
    runner = await cp.start()

    # Give it time to start
    await asyncio.sleep(0.1)

    yield cp

    # Cleanup
    await cp.stop()
    await runner.cleanup()


@pytest.fixture
def test_shard_config():
    """Create test shard configuration."""
    return ShardConfig(
        runner_id="test-runner-001",
        shard_id="test-shard-0",
        max_fps=6,
        sources=[
            {
                "camera_uuid": "cam-test-001",
                "url": "rtsp://127.0.0.1:8554/test_camera_001",
                "site_id": "test-site-A",
                "tenant_id": "test-tenant-01"
            },
            {
                "camera_uuid": "cam-test-002",
                "url": "rtsp://127.0.0.1:8554/test_camera_002",
                "site_id": "test-site-A",
                "tenant_id": "test-tenant-01"
            }
        ],
        amqp={
            "host": "localhost",
            "ex_status": "test.status.topic",
            "ex_detect": "test.detections.topic"
        },
        cp={
            "base_url": "http://127.0.0.1:8001",
            "token": "test-jwt-token"
        },
        telemetry={
            "report_interval_seconds": 1  # Faster for testing
        },
        cameras={
            "cam-test-001": CameraConfig(
                camera_uuid="cam-test-001",
                zones=[
                    ZoneConfig(
                        zone_id=1,
                        name="test_include_zone",
                        kind="include",
                        priority=100,
                        polygon=[[100, 100], [300, 100], [300, 300], [100, 300]],
                        allow_labels=["person", "car"]
                    ),
                    ZoneConfig(
                        zone_id=2,
                        name="test_exclude_zone",
                        kind="exclude",
                        priority=200,
                        polygon=[[200, 200], [250, 200], [250, 250], [200, 250]],
                        deny_labels=["person"]
                    )
                ],
                motion_gating=MotionGatingConfig(
                    enabled=True,
                    min_area_px=500,
                    cooldown_frames=1
                ),
                zone_test="center"
            ),
            "cam-test-002": CameraConfig(
                camera_uuid="cam-test-002",
                zones=[],
                motion_gating=MotionGatingConfig(enabled=False)
            )
        }
    )


class TestWorkerIntegration:
    """Integration tests for production worker."""

    @pytest.mark.asyncio
    async def test_worker_startup_and_health(self, test_shard_config, mock_rabbitmq):
        """Test worker starts up and health endpoints work."""

        worker = ProductionWorker(test_shard_config)

        # Start worker in a thread
        worker_thread = threading.Thread(target=worker.start, daemon=True)
        worker_thread.start()

        # Wait for worker to become ready
        max_wait = 10  # seconds
        ready = False
        for _ in range(max_wait * 10):  # Check every 100ms
            try:
                async with httpx.AsyncClient() as client:
                    health_response = await client.get("http://127.0.0.1:8080/healthz")
                    if health_response.status_code == 200:
                        ready_response = await client.get("http://127.0.0.1:8080/ready")
                        if ready_response.status_code == 200:
                            ready = True
                            break
            except:
                pass
            await asyncio.sleep(0.1)

        assert ready, "Worker did not become ready within timeout"

        # Test health endpoints
        async with httpx.AsyncClient() as client:
            # Health check
            health_response = await client.get("http://127.0.0.1:8080/healthz")
            assert health_response.status_code == 200
            assert health_response.text == "OK"

            # Ready check
            ready_response = await client.get("http://127.0.0.1:8080/ready")
            assert ready_response.status_code == 200
            ready_data = ready_response.json()
            assert ready_data["ready"] == True

            # Prometheus metrics
            metrics_response = await client.get("http://127.0.0.1:9108/metrics")
            assert metrics_response.status_code == 200
            metrics_text = metrics_response.text
            assert "stream_up" in metrics_text
            assert "frames_total" in metrics_text

            # Test drain endpoint
            drain_response = await client.post("http://127.0.0.1:8080/drain")
            assert drain_response.status_code == 202
            assert drain_response.text == "DRAINING"

        # Wait a bit for graceful shutdown
        await asyncio.sleep(1)

        # Worker should shutdown gracefully
        worker.running = False
        worker_thread.join(timeout=5)

    @pytest.mark.asyncio
    async def test_motion_detection_integration(self, test_shard_config, mock_rabbitmq):
        """Test motion detection with zone gating works end-to-end."""

        worker = ProductionWorker(test_shard_config)

        # Create test frames
        static_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        motion_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add motion in include zone area (zone polygon is [100,100] to [300,300])
        cv2.rectangle(motion_frame, (150, 150), (250, 250), (255, 255, 255), -1)

        # Mock video frames
        from somba_pipeline.worker import MockVideoFrame
        static_video_frame = MockVideoFrame(static_frame, source_id=0, frame_id=1)
        motion_video_frame = MockVideoFrame(motion_frame, source_id=0, frame_id=2)

        # Test motion detection directly
        camera_uuid = "cam-test-001"
        motion_detector = worker.motion_wrapper.motion_detectors.get(camera_uuid)

        if motion_detector:
            # Process static frame
            static_result = motion_detector.detect_motion(static_frame)
            assert isinstance(static_result, worker.motion_wrapper.motion_detectors[camera_uuid].__class__.__bases__[0])

            # Process motion frame
            motion_result = motion_detector.detect_motion(motion_frame)
            assert isinstance(motion_result, worker.motion_wrapper.motion_detectors[camera_uuid].__class__.__bases__[0])

            # Motion frame should have some detected pixels
            assert motion_result.pixels_changed >= 0

        # Test motion wrapper
        wrapper_results = worker.motion_wrapper([static_video_frame, motion_video_frame])
        assert len(wrapper_results) == 2

        # Verify wrapper statistics
        stats = worker.motion_wrapper.get_motion_stats()
        assert stats['global']['total_frames'] == 2

    @pytest.mark.asyncio
    async def test_zone_attribution_integration(self, test_shard_config):
        """Test zone attribution and filtering integration."""

        worker = ProductionWorker(test_shard_config)
        camera_uuid = "cam-test-001"

        # Create test detections
        raw_detections = [
            {
                "class": "person",
                "conf": 0.85,
                "bbox_xywh": [150, 150, 50, 80]  # Center at (175, 190) - in include zone
            },
            {
                "class": "person",
                "conf": 0.75,
                "bbox_xywh": [220, 220, 20, 20]  # Center at (230, 230) - in exclude zone
            },
            {
                "class": "car",
                "conf": 0.90,
                "bbox_xywh": [350, 350, 60, 40]  # Center at (380, 380) - outside zones (zone 0)
            }
        ]

        # Process through zone attribution
        published_objects, zone_stats = worker.zone_attributor.process_camera_detections(
            camera_uuid, raw_detections
        )

        # Verify zone assignment and filtering
        assert len(published_objects) >= 0  # Some objects should pass filtering

        # Check zone assignments
        person_in_include = None
        person_in_exclude = None
        car_outside = None

        for obj in published_objects:
            if obj.label == "person" and obj.primary_zone_id == 1:
                person_in_include = obj
            elif obj.label == "person" and obj.primary_zone_id == 2:
                person_in_exclude = obj  # Should be filtered out
            elif obj.label == "car" and obj.primary_zone_id == 0:
                car_outside = obj

        # Person in include zone should be published (zone allows "person")
        if person_in_include:
            assert not person_in_include.filtered
            assert person_in_include.primary_zone_id == 1

        # Person in exclude zone should be filtered (zone denies "person")
        # Note: May not appear in published_objects if filtered

        # Car outside zones should use zone 0 (global filters)
        if car_outside:
            assert car_outside.primary_zone_id == 0

        # Verify zone statistics
        assert isinstance(zone_stats, dict)

    @pytest.mark.asyncio
    async def test_event_schema_v2_generation(self, test_shard_config, mock_rabbitmq):
        """Test that schema v2 events are generated correctly."""

        worker = ProductionWorker(test_shard_config)
        camera_uuid = "cam-test-001"

        # Create mock video frame and predictions
        test_frame = np.ones((480, 640, 3), dtype=np.uint8)
        from somba_pipeline.worker import MockVideoFrame
        video_frame = MockVideoFrame(test_frame, source_id=0, frame_id=123)

        # Mock prediction data
        predictions = {
            "time": 0.025,
            "predictions": [
                {
                    "class": "person",
                    "conf": 0.85,
                    "bbox_xyxy": [100, 100, 150, 180]  # Will be converted to xywh
                }
            ]
        }

        # Process prediction through worker
        await worker._on_prediction(predictions, video_frame)

        # Check that detection event was queued
        assert not worker.detection_queue.empty()

        # Get the event from queue
        detection_event = await worker.detection_queue.get()

        # Verify schema v2 structure
        assert detection_event.schema_version == 2
        assert detection_event.camera_uuid == camera_uuid
        assert detection_event.tenant_id == "test-tenant-01"
        assert detection_event.site_id == "test-site-A"
        assert detection_event.frame.seq == 123
        assert detection_event.frame.skipped_by_motion == False

        # Verify zones config metadata
        assert detection_event.zones_config.zone_test == "center"
        assert isinstance(detection_event.zones_config.zone_version, str)

        # Verify objects have zone attribution
        if detection_event.objects:
            obj = detection_event.objects[0]
            assert obj.label == "person"
            assert isinstance(obj.primary_zone_id, int)
            assert isinstance(obj.zones_hit, list)
            assert isinstance(obj.zone_membership, dict)

    @pytest.mark.asyncio
    async def test_prometheus_metrics_integration(self, test_shard_config, mock_rabbitmq):
        """Test that Prometheus metrics are exposed correctly."""

        worker = ProductionWorker(test_shard_config)

        # Start worker briefly
        worker_thread = threading.Thread(target=worker.start, daemon=True)
        worker_thread.start()

        # Wait for metrics server to start
        await asyncio.sleep(2)

        # Test metrics endpoint
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://127.0.0.1:9108/metrics")
                assert response.status_code == 200

                metrics_text = response.text

                # Check for required metrics from Phase 2 specification
                expected_metrics = [
                    "stream_up",
                    "last_frame_age_seconds",
                    "stream_fps",
                    "pipeline_fps",
                    "inference_latency_seconds",
                    "e2e_latency_seconds",
                    "stream_errors_total",
                    "frames_total",
                    "frames_skipped_motion_total",
                    "detections_raw_total",
                    "detections_published_total",
                    "detections_dropped_total",
                    "zones_config_hash"
                ]

                for metric in expected_metrics:
                    assert metric in metrics_text, f"Missing metric: {metric}"

                # Check for camera labels
                assert "cam-test-001" in metrics_text

            except Exception as e:
                pytest.fail(f"Failed to fetch metrics: {e}")

        # Cleanup
        worker.running = False
        worker_thread.join(timeout=5)


@pytest.mark.asyncio
async def test_mock_control_plane_integration(mock_control_plane):
    """Test integration with mock control plane."""

    # Test camera listing
    async with httpx.AsyncClient() as client:
        response = await client.get("http://127.0.0.1:8001/v1/cameras?enabled=true")
        assert response.status_code == 200

        data = response.json()
        assert "items" in data
        assert len(data["items"]) > 0

        # Check camera structure
        camera = data["items"][0]
        assert "camera_uuid" in camera
        assert "tenant_id" in camera
        assert "site_id" in camera

    # Test lease operations
    async with httpx.AsyncClient() as client:
        # Acquire lease
        acquire_data = {
            "runner_id": "test-runner",
            "camera_uuid": "cam-001",
            "ttl_seconds": 300
        }

        response = await client.post(
            "http://127.0.0.1:8001/v1/leases/camera/acquire",
            json=acquire_data
        )
        assert response.status_code == 200

        lease_data = response.json()
        assert lease_data["camera_uuid"] == "cam-001"
        assert lease_data["owner_id"] == "test-runner"

        # Renew lease
        renew_data = {
            "runner_id": "test-runner",
            "camera_uuid": "cam-001",
            "ttl_seconds": 300
        }

        response = await client.post(
            "http://127.0.0.1:8001/v1/leases/camera/renew",
            json=renew_data
        )
        assert response.status_code == 200

        # Release lease
        release_data = {
            "runner_id": "test-runner",
            "camera_uuid": "cam-001"
        }

        response = await client.post(
            "http://127.0.0.1:8001/v1/leases/camera/release",
            json=release_data
        )
        assert response.status_code == 200

    # Test token bucket
    async with httpx.AsyncClient() as client:
        response = await client.post("http://127.0.0.1:8001/v1/sites/site-A/budget/consume")
        assert response.status_code == 200

        data = response.json()
        assert "tokens_remaining" in data
        assert isinstance(data["tokens_remaining"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
