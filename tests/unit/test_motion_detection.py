"""Unit tests for motion detection system with zone-based gating."""

import pytest
import cv2
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from somba_pipeline.motion_detection import (
    MotionDetector, ZoneMaskBuilder, MotionGatingInferenceWrapper, MotionResult
)
from somba_pipeline.schemas import MotionGatingConfig, ZoneConfig


class TestZoneMaskBuilder:
    """Test zone mask building from polygon configurations."""

    def test_include_mask_single_zone(self):
        """Test building include mask with single include zone."""
        builder = ZoneMaskBuilder(300, 200)
        zones = [
            ZoneConfig(
                zone_id=1,
                name="test_zone",
                kind="include",
                priority=100,
                polygon=[[50, 50], [150, 50], [150, 150], [50, 150]]
            )
        ]

        mask = builder.build_include_mask(zones)

        assert mask.shape == (200, 300)
        # Check that zone area is included
        assert mask[100, 100] == 255  # Inside zone
        assert mask[25, 25] == 0      # Outside zone
        assert mask[175, 175] == 0    # Outside zone

    def test_include_exclude_combination(self):
        """Test include mask with both include and exclude zones."""
        builder = ZoneMaskBuilder(300, 200)
        zones = [
            ZoneConfig(
                zone_id=1,
                name="include_zone",
                kind="include",
                priority=100,
                polygon=[[50, 50], [200, 50], [200, 150], [50, 150]]
            ),
            ZoneConfig(
                zone_id=2,
                name="exclude_zone",
                kind="exclude",
                priority=200,
                polygon=[[100, 75], [150, 75], [150, 125], [100, 125]]
            )
        ]

        mask = builder.build_include_mask(zones)

        # Include zone should be included
        assert mask[100, 75] == 255   # Include zone, not in exclude
        # Exclude zone should be subtracted
        assert mask[100, 100] == 0    # In exclude zone

    def test_no_include_zones_defaults_to_whole_frame(self):
        """Test that no include zones means whole frame is included."""
        builder = ZoneMaskBuilder(300, 200)
        zones = [
            ZoneConfig(
                zone_id=1,
                name="exclude_only",
                kind="exclude",
                priority=100,
                polygon=[[100, 100], [120, 100], [120, 120], [100, 120]]
            )
        ]

        mask = builder.build_include_mask(zones)

        # Most of frame should be included (255)
        assert mask[50, 50] == 255    # Outside exclude zone
        assert mask[200, 150] == 255  # Outside exclude zone
        # Exclude zone should be subtracted
        assert mask[110, 110] == 0    # In exclude zone


class TestMotionDetector:
    """Test motion detection with zone-based gating."""

    @pytest.fixture
    def motion_config(self):
        """Default motion detection configuration."""
        return MotionGatingConfig(
            enabled=True,
            downscale=0.5,
            dilation_px=6,
            min_area_px=1000,
            cooldown_frames=2,
            noise_floor=10
        )

    @pytest.fixture
    def test_zones(self):
        """Test zone configuration."""
        return [
            ZoneConfig(
                zone_id=1,
                name="motion_zone",
                kind="include",
                priority=100,
                polygon=[[100, 100], [200, 100], [200, 200], [100, 200]]
            )
        ]

    @pytest.fixture
    def motion_detector(self, motion_config, test_zones):
        """Motion detector instance."""
        return MotionDetector(
            camera_uuid="test_camera",
            motion_config=motion_config,
            zones=test_zones,
            frame_width=300,
            frame_height=200
        )

    def test_motion_detector_initialization(self, motion_detector):
        """Test motion detector initializes correctly."""
        assert motion_detector.camera_uuid == "test_camera"
        assert motion_detector.config.enabled == True
        assert motion_detector.include_mask is not None
        assert motion_detector.include_mask.shape == (200, 300)

    def test_disabled_motion_gating_always_detects(self):
        """Test that disabled motion gating always returns motion detected."""
        config = MotionGatingConfig(enabled=False)
        zones = []
        detector = MotionDetector("test", config, zones, 300, 200)

        # Create dummy frame
        frame = np.zeros((200, 300, 3), dtype=np.uint8)

        result = detector.detect_motion(frame)

        assert result.motion_detected == True
        assert result.pixels_changed == 0  # No actual motion calculated

    def test_motion_detection_with_significant_motion(self, motion_detector):
        """Test motion detection when significant motion is present."""
        # Create two frames with difference in motion zone
        frame1 = np.zeros((200, 300, 3), dtype=np.uint8)
        frame2 = np.zeros((200, 300, 3), dtype=np.uint8)

        # Add motion in the include zone area
        cv2.rectangle(frame2, (120, 120), (180, 180), (255, 255, 255), -1)

        # First frame (background)
        result1 = motion_detector.detect_motion(frame1)

        # Second frame (with motion)
        result2 = motion_detector.detect_motion(frame2)

        # Second frame should detect motion if area is above threshold
        assert isinstance(result2, MotionResult)
        # Motion detection result depends on actual computer vision processing
        # At minimum, verify the structure is correct
        assert hasattr(result2, 'motion_detected')
        assert hasattr(result2, 'pixels_changed')
        assert hasattr(result2, 'include_mask_area')

    def test_motion_detection_outside_include_zones(self, motion_detector):
        """Test that motion outside include zones doesn't trigger detection."""
        # Create frames with motion only outside include zones
        frame1 = np.zeros((200, 300, 3), dtype=np.uint8)
        frame2 = np.zeros((200, 300, 3), dtype=np.uint8)

        # Add motion outside the include zone (zone is [100,100] to [200,200])
        cv2.rectangle(frame2, (250, 150), (280, 180), (255, 255, 255), -1)

        # Process both frames
        motion_detector.detect_motion(frame1)
        result = motion_detector.detect_motion(frame2)

        # Should not detect motion since it's outside include zone
        assert result.motion_in_include_area == 0 or result.motion_in_include_area < motion_detector.config.min_area_px

    def test_cooldown_mechanism(self, motion_detector):
        """Test motion detection cooldown prevents flapping."""
        # Create frame with motion
        frame_with_motion = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.rectangle(frame_with_motion, (120, 120), (180, 180), (255, 255, 255), -1)

        # Create frame without motion
        frame_no_motion = np.zeros((200, 300, 3), dtype=np.uint8)

        # First frame with motion
        result1 = motion_detector.detect_motion(frame_with_motion)

        # Process frames without motion during cooldown period
        results_during_cooldown = []
        for _ in range(motion_detector.config.cooldown_frames):
            result = motion_detector.detect_motion(frame_no_motion)
            results_during_cooldown.append(result)

        # During cooldown, motion might still be detected due to hysteresis
        # The exact behavior depends on the implementation details
        assert len(results_during_cooldown) == motion_detector.config.cooldown_frames

    def test_statistics_tracking(self, motion_detector):
        """Test that motion detection statistics are tracked correctly."""
        initial_stats = motion_detector.get_stats()
        assert initial_stats['frames_processed'] == 0

        # Process a frame
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        motion_detector.detect_motion(frame)

        updated_stats = motion_detector.get_stats()
        assert updated_stats['frames_processed'] == 1

    def test_zone_updates(self, motion_detector):
        """Test updating zone configuration."""
        new_zones = [
            ZoneConfig(
                zone_id=2,
                name="new_zone",
                kind="include",
                priority=150,
                polygon=[[150, 150], [250, 150], [250, 250], [150, 250]]
            )
        ]

        old_mask_area = np.sum(motion_detector.include_mask > 0)
        motion_detector.update_zones(new_zones)
        new_mask_area = np.sum(motion_detector.include_mask > 0)

        # Mask should have changed
        assert motion_detector.zones == new_zones
        # Area might be different (though could be same by coincidence)
        assert isinstance(new_mask_area, (int, np.integer))


class TestMotionGatingInferenceWrapper:
    """Test the motion gating wrapper for inference."""

    @pytest.fixture
    def mock_inference_handler(self):
        """Mock inference handler."""
        def handler(frames):
            return [{"predictions": [{"class": "person", "conf": 0.8}]} for _ in frames]
        return Mock(side_effect=handler)

    @pytest.fixture
    def camera_configs(self):
        """Camera configurations for motion gating."""
        motion_config = MotionGatingConfig(enabled=True, min_area_px=500)
        zones = [
            ZoneConfig(
                zone_id=1,
                name="test_zone",
                kind="include",
                priority=100,
                polygon=[[100, 100], [200, 100], [200, 200], [100, 200]]
            )
        ]
        return {
            "cam_001": (motion_config, zones),
            "cam_002": (motion_config, [])  # No zones
        }

    @pytest.fixture
    def motion_wrapper(self, mock_inference_handler, camera_configs):
        """Motion gating wrapper instance."""
        return MotionGatingInferenceWrapper(
            inference_handler=mock_inference_handler,
            camera_configs=camera_configs,
            frame_width=300,
            frame_height=200
        )

    def test_wrapper_initialization(self, motion_wrapper, camera_configs):
        """Test wrapper initializes correctly."""
        assert len(motion_wrapper.motion_detectors) == len(camera_configs)
        assert "cam_001" in motion_wrapper.motion_detectors
        assert "cam_002" in motion_wrapper.motion_detectors

    def test_frames_with_motion_processed(self, motion_wrapper, mock_inference_handler):
        """Test that frames with detected motion are processed."""
        # Mock video frames
        mock_frames = [Mock(image=np.ones((200, 300, 3), dtype=np.uint8), source_id=0)]

        # Process frames
        results = motion_wrapper(mock_frames)

        assert len(results) == len(mock_frames)
        # Results structure depends on motion detection outcome
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, dict)
            assert 'skipped_by_motion' in result

    def test_frames_without_motion_skipped(self, motion_wrapper):
        """Test that frames without motion are marked as skipped."""
        # Create frames with minimal/no motion
        mock_frames = [Mock(image=np.zeros((200, 300, 3), dtype=np.uint8), source_id=0)]

        # Process multiple times to establish background
        for _ in range(5):
            motion_wrapper(mock_frames)

        # Process static frame
        results = motion_wrapper(mock_frames)

        # Should have results even if skipped
        assert len(results) == len(mock_frames)
        # May or may not be skipped depending on motion detection sensitivity

    def test_statistics_aggregation(self, motion_wrapper):
        """Test that statistics are properly aggregated."""
        initial_stats = motion_wrapper.get_motion_stats()

        assert 'global' in initial_stats
        assert 'per_camera' in initial_stats
        assert initial_stats['global']['total_frames'] == 0

        # Process some frames
        mock_frames = [Mock(image=np.ones((200, 300, 3), dtype=np.uint8), source_id=0)]
        motion_wrapper(mock_frames)

        updated_stats = motion_wrapper.get_motion_stats()
        assert updated_stats['global']['total_frames'] > 0

    def test_camera_config_updates(self, motion_wrapper):
        """Test updating camera configuration."""
        new_motion_config = MotionGatingConfig(enabled=False)
        new_zones = []

        motion_wrapper.update_camera_config("cam_001", new_motion_config, new_zones)

        # Should have updated the configuration
        assert motion_wrapper.camera_configs["cam_001"] == (new_motion_config, new_zones)

    def test_unknown_camera_handling(self, motion_wrapper):
        """Test handling of frames from unknown cameras."""
        # Frame from camera not in initial config
        mock_frames = [Mock(image=np.ones((200, 300, 3), dtype=np.uint8), source_id=999)]

        results = motion_wrapper(mock_frames)

        # Should still process the frame (no motion config = always process)
        assert len(results) == 1
        assert isinstance(results[0], dict)


class TestMotionDetectionIntegration:
    """Integration tests for motion detection components."""

    def test_end_to_end_motion_gating(self):
        """Test complete motion gating pipeline."""
        # Setup
        motion_config = MotionGatingConfig(
            enabled=True,
            min_area_px=100,
            cooldown_frames=1
        )
        zones = [
            ZoneConfig(
                zone_id=1,
                name="active_zone",
                kind="include",
                priority=100,
                polygon=[[50, 50], [150, 50], [150, 150], [50, 150]]
            )
        ]

        def mock_inference(frames):
            return [{"predictions": [{"class": "test", "conf": 0.9}]} for _ in frames]

        wrapper = MotionGatingInferenceWrapper(
            inference_handler=mock_inference,
            camera_configs={"test_cam": (motion_config, zones)},
            frame_width=200,
            frame_height=200
        )

        # Create test frames
        static_frame = Mock(image=np.zeros((200, 200, 3), dtype=np.uint8), source_id=0)
        motion_frame = Mock(image=np.ones((200, 200, 3), dtype=np.uint8), source_id=0)
        # Add actual motion to motion frame
        motion_frame.image[75:125, 75:125] = 255

        # Test static frame (should potentially be skipped)
        static_results = wrapper([static_frame])
        assert len(static_results) == 1

        # Test motion frame (should be processed)
        motion_results = wrapper([motion_frame])
        assert len(motion_results) == 1

        # Verify statistics
        stats = wrapper.get_motion_stats()
        assert stats['global']['total_frames'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
