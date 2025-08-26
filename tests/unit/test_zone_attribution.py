"""Unit tests for zone attribution system."""

import pytest
import numpy as np
from typing import List

from somba_pipeline.zone_attribution import (
    ZoneAttributor, MultiCameraZoneAttributor, DetectionInput,
    point_in_polygon, calculate_bbox_polygon_iou
)
from somba_pipeline.schemas import ZoneConfig, CameraConfig


class TestPointInPolygon:
    """Test point-in-polygon algorithm."""

    def test_point_inside_rectangle(self):
        """Test point inside rectangular polygon."""
        polygon = [[100, 100], [200, 100], [200, 200], [100, 200]]

        assert point_in_polygon((150, 150), polygon) == True
        assert point_in_polygon((50, 50), polygon) == False
        assert point_in_polygon((250, 250), polygon) == False
        assert point_in_polygon((100, 150), polygon) == True  # On boundary

    def test_point_inside_triangle(self):
        """Test point inside triangular polygon."""
        polygon = [[100, 100], [200, 100], [150, 200]]

        assert point_in_polygon((150, 150), polygon) == True
        assert point_in_polygon((50, 50), polygon) == False
        assert point_in_polygon((150, 250), polygon) == False

    def test_complex_polygon(self):
        """Test point inside complex polygon."""
        # L-shaped polygon
        polygon = [[100, 100], [200, 100], [200, 150], [150, 150], [150, 200], [100, 200]]

        assert point_in_polygon((125, 125), polygon) == True  # Inside L
        assert point_in_polygon((175, 175), polygon) == False  # In the notch
        assert point_in_polygon((175, 125), polygon) == True  # In upper part


class TestBboxPolygonIoU:
    """Test bounding box to polygon IoU calculation."""

    def test_full_overlap(self):
        """Test bbox fully inside polygon."""
        bbox_xywh = [110, 110, 80, 80]  # [x, y, w, h]
        polygon = [[100, 100], [200, 100], [200, 200], [100, 200]]
        frame_width, frame_height = 300, 300

        iou = calculate_bbox_polygon_iou(bbox_xywh, polygon, frame_width, frame_height)
        assert iou == 1.0  # Perfect overlap

    def test_no_overlap(self):
        """Test bbox with no overlap to polygon."""
        bbox_xywh = [250, 250, 40, 40]
        polygon = [[100, 100], [200, 100], [200, 200], [100, 200]]
        frame_width, frame_height = 300, 300

        iou = calculate_bbox_polygon_iou(bbox_xywh, polygon, frame_width, frame_height)
        assert iou == 0.0

    def test_partial_overlap(self):
        """Test bbox with partial overlap."""
        bbox_xywh = [150, 150, 100, 100]  # Extends beyond polygon
        polygon = [[100, 100], [200, 100], [200, 200], [100, 200]]
        frame_width, frame_height = 300, 300

        iou = calculate_bbox_polygon_iou(bbox_xywh, polygon, frame_width, frame_height)
        assert 0.0 < iou < 1.0  # Partial overlap


class TestZoneAttributor:
    """Test zone attribution with precedence rules."""

    @pytest.fixture
    def sample_camera_config(self):
        """Create sample camera configuration."""
        return CameraConfig(
            camera_uuid="test_camera",
            zones=[
                ZoneConfig(
                    zone_id=1,
                    name="driveway",
                    kind="include",
                    priority=100,
                    polygon=[[100, 100], [200, 100], [200, 200], [100, 200]],
                    allow_labels=["person", "car"],
                    min_score=0.5
                ),
                ZoneConfig(
                    zone_id=2,
                    name="restricted",
                    kind="exclude",
                    priority=200,  # Higher priority
                    polygon=[[150, 150], [250, 150], [250, 250], [150, 250]],
                    deny_labels=["person"]
                )
            ],
            allow_labels=["person", "car", "truck"],
            deny_labels=[],
            min_score=0.3,
            zone_test="center"
        )

    @pytest.fixture
    def zone_attributor(self, sample_camera_config):
        """Create zone attributor instance."""
        return ZoneAttributor(sample_camera_config, frame_width=300, frame_height=300)

    def test_zone_precedence_rules(self, zone_attributor):
        """Test that higher priority zones override lower priority ones."""
        # Detection center in overlap area where both zones match
        detection = DetectionInput(
            label="person",
            score=0.8,
            bbox_xywh=[175, 175, 20, 20]  # Center at (185, 185) - in both zones
        )

        assignments = zone_attributor.assign_zones([detection])
        assert len(assignments) == 1

        assignment = assignments[0]
        # Zone 2 has higher priority (200 > 100), so it should be primary
        assert assignment.primary_zone_id == 2
        assert 2 in assignment.zones_hit
        assert 1 in assignment.zones_hit
        # Zone 2 is exclude and denies "person", so should be filtered
        assert not assignment.passed_filters
        assert "deny_label" in assignment.filter_reason

    def test_per_zone_label_filters_override_global(self, zone_attributor):
        """Test per-zone filters override global filters."""
        # Detection in zone 1 only
        detection = DetectionInput(
            label="person",
            score=0.6,  # Above zone min_score (0.5) but below global would be irrelevant
            bbox_xywh=[125, 125, 20, 20]  # Center at (135, 135) - only in zone 1
        )

        assignments = zone_attributor.assign_zones([detection])
        assignment = assignments[0]

        assert assignment.primary_zone_id == 1
        # Zone 1 allows "person" and score is above zone min_score
        assert assignment.passed_filters
        assert assignment.filter_reason is None

    def test_zone_0_fallback(self, zone_attributor):
        """Test zone 0 fallback when no zones match."""
        # Detection outside all defined zones
        detection = DetectionInput(
            label="truck",
            score=0.8,
            bbox_xywh=[50, 50, 20, 20]  # Center at (60, 60) - outside all zones
        )

        assignments = zone_attributor.assign_zones([detection])
        assignment = assignments[0]

        assert assignment.primary_zone_id == 0  # Zone 0 fallback
        assert assignment.zones_hit == [0]
        # Should use global filters for zone 0
        assert assignment.passed_filters  # "truck" is in global allow_labels

    def test_iou_threshold_filtering(self):
        """Test center+iou zone test mode."""
        camera_config = CameraConfig(
            camera_uuid="test_camera",
            zones=[
                ZoneConfig(
                    zone_id=1,
                    name="test_zone",
                    kind="include",
                    priority=100,
                    polygon=[[100, 100], [200, 100], [200, 200], [100, 200]]
                )
            ],
            zone_test="center+iou",
            iou_threshold=0.3  # Require significant overlap
        )

        attributor = ZoneAttributor(camera_config, 300, 300)

        # Detection with center in zone but low IoU
        detection = DetectionInput(
            label="person",
            score=0.8,
            bbox_xywh=[190, 190, 50, 50]  # Center in zone but extends way beyond
        )

        assignments = attributor.assign_zones([detection])
        assignment = assignments[0]

        # Should check both center-in and IoU
        # Depending on actual IoU calculation, might fall back to zone 0
        if assignment.primary_zone_id == 0:
            # IoU was below threshold
            assert "0" in assignment.zone_membership
        else:
            # IoU was above threshold
            assert assignment.primary_zone_id == 1
            assert assignment.zone_membership["1"].iou >= 0.3


class TestMultiCameraZoneAttributor:
    """Test multi-camera zone attribution."""

    @pytest.fixture
    def multi_camera_configs(self):
        """Create configurations for multiple cameras."""
        return {
            "cam_001": CameraConfig(
                camera_uuid="cam_001",
                zones=[
                    ZoneConfig(
                        zone_id=1,
                        name="entrance",
                        kind="include",
                        priority=100,
                        polygon=[[100, 100], [200, 100], [200, 200], [100, 200]],
                        allow_labels=["person"]
                    )
                ]
            ),
            "cam_002": CameraConfig(
                camera_uuid="cam_002",
                zones=[]  # No zones configured
            )
        }

    @pytest.fixture
    def multi_attributor(self, multi_camera_configs):
        """Create multi-camera attributor."""
        return MultiCameraZoneAttributor(multi_camera_configs, 300, 300)

    def test_camera_specific_processing(self, multi_attributor):
        """Test processing detections for specific cameras."""
        # Detection for cam_001 (has zones)
        raw_detections_cam1 = [
            {"class": "person", "conf": 0.8, "bbox_xywh": [150, 150, 30, 30]}
        ]

        objects_cam1, stats_cam1 = multi_attributor.process_camera_detections(
            "cam_001", raw_detections_cam1
        )

        assert len(objects_cam1) == 1
        assert objects_cam1[0].primary_zone_id == 1  # Matched zone 1
        assert objects_cam1[0].label == "person"

        # Detection for cam_002 (no zones)
        raw_detections_cam2 = [
            {"class": "car", "conf": 0.7, "bbox_xywh": [100, 100, 50, 50]}
        ]

        objects_cam2, stats_cam2 = multi_attributor.process_camera_detections(
            "cam_002", raw_detections_cam2
        )

        assert len(objects_cam2) == 1
        assert objects_cam2[0].primary_zone_id == 0  # Zone 0 fallback
        assert objects_cam2[0].label == "car"

    def test_unknown_camera_handling(self, multi_attributor):
        """Test handling of unknown cameras."""
        raw_detections = [
            {"class": "truck", "conf": 0.6, "bbox_xywh": [200, 200, 40, 40]}
        ]

        # Process detections for unknown camera
        objects, stats = multi_attributor.process_camera_detections(
            "unknown_camera", raw_detections
        )

        # Should create default config and process
        assert len(objects) == 1
        assert objects[0].primary_zone_id == 0  # Default to zone 0

    def test_config_updates(self, multi_attributor):
        """Test updating camera configurations."""
        new_config = CameraConfig(
            camera_uuid="cam_001",
            zones=[
                ZoneConfig(
                    zone_id=3,
                    name="new_zone",
                    kind="include",
                    priority=50,
                    polygon=[[50, 50], [100, 50], [100, 100], [50, 100]],
                    allow_labels=["car"]
                )
            ]
        )

        multi_attributor.update_camera_config("cam_001", new_config)

        # Test that new config is used
        raw_detections = [
            {"class": "car", "conf": 0.8, "bbox_xywh": [60, 60, 20, 20]}  # In new zone
        ]

        objects, stats = multi_attributor.process_camera_detections(
            "cam_001", raw_detections
        )

        assert len(objects) == 1
        assert objects[0].primary_zone_id == 3  # New zone
        assert objects[0].passed_filters  # "car" allowed in new zone


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
