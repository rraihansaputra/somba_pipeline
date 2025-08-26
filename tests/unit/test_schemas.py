"""Unit tests for event schemas and configuration models."""

import json
import pytest
from datetime import datetime, timezone
from typing import Dict, Any

from somba_pipeline.schemas import (
    ZoneConfig, MotionGatingConfig, CameraConfig, DetectionEvent,
    StatusEvent, ErrorEvent, DetectedObject, ZoneMembership,
    ModelInfo, FrameInfo, ZonesConfigInfo, ShardConfig
)


class TestZoneConfig:
    """Test zone configuration validation."""

    def test_valid_zone_config(self):
        """Test valid zone configuration."""
        zone = ZoneConfig(
            zone_id=1,
            name="test_zone",
            kind="include",
            priority=100,
            polygon=[[100, 100], [200, 100], [200, 200]]
        )

        assert zone.zone_id == 1
        assert zone.name == "test_zone"
        assert zone.kind == "include"
        assert zone.priority == 100
        assert len(zone.polygon) == 3

    def test_invalid_zone_id(self):
        """Test that zone_id < 1 is rejected."""
        with pytest.raises(ValueError):
            ZoneConfig(
                zone_id=0,  # Invalid - must be >= 1
                name="test",
                kind="include",
                priority=100,
                polygon=[[0, 0], [100, 0], [100, 100]]
            )

    def test_invalid_kind(self):
        """Test that invalid zone kind is rejected."""
        with pytest.raises(ValueError):
            ZoneConfig(
                zone_id=1,
                name="test",
                kind="invalid_kind",  # Must be include or exclude
                priority=100,
                polygon=[[0, 0], [100, 0], [100, 100]]
            )

    def test_minimum_polygon_vertices(self):
        """Test that polygon must have at least 3 vertices."""
        with pytest.raises(ValueError):
            ZoneConfig(
                zone_id=1,
                name="test",
                kind="include",
                priority=100,
                polygon=[[0, 0], [100, 0]]  # Only 2 vertices - invalid
            )

    def test_optional_label_filters(self):
        """Test optional label filter fields."""
        zone = ZoneConfig(
            zone_id=1,
            name="test",
            kind="include",
            priority=100,
            polygon=[[0, 0], [100, 0], [100, 100]],
            allow_labels=["person", "car"],
            deny_labels=["truck"],
            min_score=0.7
        )

        assert zone.allow_labels == ["person", "car"]
        assert zone.deny_labels == ["truck"]
        assert zone.min_score == 0.7


class TestMotionGatingConfig:
    """Test motion gating configuration."""

    def test_default_values(self):
        """Test default motion gating configuration."""
        config = MotionGatingConfig()

        assert config.enabled == True
        assert config.downscale == 0.5
        assert config.dilation_px == 6
        assert config.min_area_px == 1500
        assert config.cooldown_frames == 2
        assert config.noise_floor == 12

    def test_custom_values(self):
        """Test custom motion gating configuration."""
        config = MotionGatingConfig(
            enabled=False,
            downscale=0.3,
            dilation_px=10,
            min_area_px=2000,
            cooldown_frames=5,
            noise_floor=20
        )

        assert config.enabled == False
        assert config.downscale == 0.3
        assert config.dilation_px == 10
        assert config.min_area_px == 2000
        assert config.cooldown_frames == 5
        assert config.noise_floor == 20

    def test_validation_constraints(self):
        """Test validation constraints."""
        # downscale must be > 0 and <= 1
        with pytest.raises(ValueError):
            MotionGatingConfig(downscale=0.0)

        with pytest.raises(ValueError):
            MotionGatingConfig(downscale=1.5)

        # Negative values should be rejected
        with pytest.raises(ValueError):
            MotionGatingConfig(dilation_px=-1)


class TestCameraConfig:
    """Test camera configuration with zones."""

    def test_zones_hash_generation(self):
        """Test that zones hash is generated correctly."""
        config = CameraConfig(
            camera_uuid="test_camera",
            zones=[
                ZoneConfig(
                    zone_id=1,
                    name="zone1",
                    kind="include",
                    priority=100,
                    polygon=[[0, 0], [100, 0], [100, 100]]
                )
            ]
        )

        hash1 = config.get_zones_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex digest length

        # Same config should produce same hash
        config2 = CameraConfig(
            camera_uuid="test_camera",
            zones=[
                ZoneConfig(
                    zone_id=1,
                    name="zone1",
                    kind="include",
                    priority=100,
                    polygon=[[0, 0], [100, 0], [100, 100]]
                )
            ]
        )
        hash2 = config2.get_zones_hash()
        assert hash1 == hash2

        # Different zones should produce different hash
        config3 = CameraConfig(
            camera_uuid="test_camera",
            zones=[
                ZoneConfig(
                    zone_id=2,  # Different zone_id
                    name="zone1",
                    kind="include",
                    priority=100,
                    polygon=[[0, 0], [100, 0], [100, 100]]
                )
            ]
        )
        hash3 = config3.get_zones_hash()
        assert hash1 != hash3

    def test_zone_test_validation(self):
        """Test zone_test field validation."""
        # Valid values
        config1 = CameraConfig(camera_uuid="test", zone_test="center")
        assert config1.zone_test == "center"

        config2 = CameraConfig(camera_uuid="test", zone_test="center+iou")
        assert config2.zone_test == "center+iou"

        # Invalid values should be rejected
        with pytest.raises(ValueError):
            CameraConfig(camera_uuid="test", zone_test="invalid_test")


class TestDetectionEvent:
    """Test detection event schema v2."""

    def test_valid_detection_event(self):
        """Test valid detection event creation."""
        event = DetectionEvent(
            schema_version=2,
            event_id="01H5J9K2M3N4P5Q6R7S8T9V0WX",
            ts_ns=int(datetime.now(timezone.utc).timestamp() * 1_000_000_000),
            tenant_id="tenant-01",
            site_id="site-A",
            camera_uuid="cam-001",
            model=ModelInfo(id="coco/11", adapter="roboflow-inference-pipeline"),
            frame=FrameInfo(w=1920, h=1080, seq=12345, fps=15.0, skipped_by_motion=False),
            zones_config=ZonesConfigInfo(
                zone_version="abc123...",
                zone_test="center",
                iou_threshold=0.10
            ),
            objects=[
                DetectedObject(
                    label="person",
                    score=0.92,
                    bbox_xywh=[100, 100, 50, 150],
                    primary_zone_id=1,
                    zones_hit=[1],
                    zone_membership={
                        "1": ZoneMembership(center_in=True, iou=0.85)
                    },
                    filtered=False
                )
            ]
        )

        assert event.schema_version == 2
        assert len(event.objects) == 1
        assert event.objects[0].label == "person"
        assert event.objects[0].primary_zone_id == 1

    def test_detection_event_serialization(self):
        """Test that detection event serializes to valid JSON."""
        event = DetectionEvent(
            schema_version=2,
            event_id="test_event",
            ts_ns=1234567890000000000,
            tenant_id="test_tenant",
            site_id="test_site",
            camera_uuid="test_camera",
            model=ModelInfo(id="test_model", adapter="test_adapter"),
            frame=FrameInfo(w=640, h=480, seq=1, fps=30.0),
            zones_config=ZonesConfigInfo(zone_version="test_hash", zone_test="center"),
            objects=[]
        )

        json_str = event.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["schema_version"] == 2
        assert parsed["event_id"] == "test_event"
        assert parsed["zones_config"]["zone_test"] == "center"

    def test_detected_object_with_zone_membership(self):
        """Test detected object with complete zone membership details."""
        obj = DetectedObject(
            label="car",
            score=0.78,
            bbox_xywh=[200, 300, 80, 40],
            primary_zone_id=2,
            zones_hit=[2, 1],  # Sorted by priority desc
            zone_membership={
                "1": ZoneMembership(center_in=True, iou=0.25),
                "2": ZoneMembership(center_in=True, iou=0.90)
            },
            filtered=False,
            filter_reason=None
        )

        assert obj.primary_zone_id == 2
        assert obj.zones_hit == [2, 1]
        assert len(obj.zone_membership) == 2
        assert obj.zone_membership["2"].iou == 0.90
        assert not obj.filtered


class TestStatusEvent:
    """Test status event schemas."""

    def test_basic_status_event(self):
        """Test basic status event creation."""
        event = StatusEvent(
            type="stream.status",
            state="STREAMING",
            camera_uuid="cam-001",
            runner_id="runner-001",
            shard_id="shard-0",
            fps=15.2,
            ts=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        )

        assert event.type == "stream.status"
        assert event.state == "STREAMING"
        assert event.fps == 15.2

    def test_status_event_with_zones_stats(self):
        """Test status event with zone statistics."""
        from somba_pipeline.schemas import ZonesStats, ZoneStats

        zones_stats = ZonesStats(
            frames_skipped_motion=125,
            frames_processed=500,
            objects_published=89,
            objects_dropped_by_filters=12,
            per_zone={
                "0": ZoneStats(objects=45, dropped=5),
                "1": ZoneStats(objects=44, dropped=7)
            }
        )

        event = StatusEvent(
            type="stream.status",
            camera_uuid="cam-001",
            runner_id="runner-001",
            shard_id="shard-0",
            zones_stats=zones_stats,
            ts=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        )

        assert event.zones_stats.frames_skipped_motion == 125
        assert event.zones_stats.objects_published == 89
        assert len(event.zones_stats.per_zone) == 2

    def test_error_event(self):
        """Test error event creation."""
        event = ErrorEvent(
            type="stream.error",
            camera_uuid="cam-001",
            runner_id="runner-001",
            code="RTSP_AUTH_FAILED",
            detail="401 Unauthorized",
            retry_in_ms=8000,
            ts=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        )

        assert event.type == "stream.error"
        assert event.code == "RTSP_AUTH_FAILED"
        assert event.retry_in_ms == 8000


class TestShardConfig:
    """Test shard configuration loading and validation."""

    def test_shard_config_creation(self):
        """Test shard configuration creation."""
        config = ShardConfig(
            runner_id="test-runner",
            shard_id="shard-0",
            max_fps=6,
            sources=[
                {"camera_uuid": "cam-001", "url": "rtsp://test", "site_id": "site-A", "tenant_id": "t1"}
            ],
            amqp={"host": "localhost", "ex_status": "status", "ex_detect": "detect"},
            cp={"base_url": "http://test", "token": "jwt"},
            telemetry={"report_interval_seconds": 5},
            cameras={}
        )

        assert config.runner_id == "test-runner"
        assert config.max_fps == 6
        assert len(config.sources) == 1
        assert config.sources[0]["camera_uuid"] == "cam-001"

    def test_shard_config_with_camera_configs(self):
        """Test shard configuration with per-camera configurations."""
        camera_config = CameraConfig(
            camera_uuid="cam-001",
            zones=[
                ZoneConfig(
                    zone_id=1, name="test", kind="include",
                    priority=100, polygon=[[0, 0], [100, 0], [100, 100]]
                )
            ]
        )

        config = ShardConfig(
            runner_id="test-runner",
            shard_id="shard-0",
            sources=[{"camera_uuid": "cam-001", "url": "test"}],
            amqp={"host": "test"},
            cp={"base_url": "test"},
            telemetry={"report_interval_seconds": 5},
            cameras={"cam-001": camera_config}
        )

        assert "cam-001" in config.cameras
        assert len(config.cameras["cam-001"].zones) == 1

    def test_json_serialization_roundtrip(self):
        """Test that configuration can be serialized and deserialized."""
        original_config = ShardConfig(
            runner_id="test",
            shard_id="test",
            sources=[{"camera_uuid": "test"}],
            amqp={"host": "test"},
            cp={"base_url": "test"},
            telemetry={"report_interval_seconds": 5}
        )

        # Serialize to dict
        config_dict = original_config.model_dump()

        # Should be able to create new instance from dict
        restored_config = ShardConfig(**config_dict)

        assert restored_config.runner_id == original_config.runner_id
        assert restored_config.shard_id == original_config.shard_id


class TestSchemaValidation:
    """Test schema validation edge cases."""

    def test_bbox_validation(self):
        """Test bounding box validation."""
        # Valid bbox
        obj = DetectedObject(
            label="test",
            score=0.5,
            bbox_xywh=[10, 20, 30, 40],  # [x, y, w, h]
            primary_zone_id=0,
            zones_hit=[0]
        )
        assert len(obj.bbox_xywh) == 4

        # Invalid bbox (wrong length)
        with pytest.raises(ValueError):
            DetectedObject(
                label="test",
                score=0.5,
                bbox_xywh=[10, 20, 30],  # Missing height
                primary_zone_id=0,
                zones_hit=[0]
            )

    def test_score_validation(self):
        """Test score range validation."""
        # Valid scores
        DetectedObject(label="test", score=0.0, bbox_xywh=[0, 0, 1, 1], primary_zone_id=0, zones_hit=[0])
        DetectedObject(label="test", score=1.0, bbox_xywh=[0, 0, 1, 1], primary_zone_id=0, zones_hit=[0])
        DetectedObject(label="test", score=0.5, bbox_xywh=[0, 0, 1, 1], primary_zone_id=0, zones_hit=[0])

        # Invalid scores
        with pytest.raises(ValueError):
            DetectedObject(label="test", score=-0.1, bbox_xywh=[0, 0, 1, 1], primary_zone_id=0, zones_hit=[0])

        with pytest.raises(ValueError):
            DetectedObject(label="test", score=1.1, bbox_xywh=[0, 0, 1, 1], primary_zone_id=0, zones_hit=[0])

    def test_timestamp_format(self):
        """Test timestamp format in events."""
        now = datetime.now(timezone.utc)
        ts_str = now.isoformat().replace('+00:00', 'Z')

        event = StatusEvent(
            type="stream.status",
            camera_uuid="test",
            runner_id="test",
            shard_id="test",
            ts=ts_str
        )

        assert event.ts.endswith('Z')
        assert 'T' in event.ts  # ISO format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
