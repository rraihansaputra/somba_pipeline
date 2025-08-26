"""
Motion detection system with zone-based gating according to Phase 2 specifications.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timedelta
import logging

from .schemas import MotionGatingConfig, ZoneConfig

logger = logging.getLogger(__name__)


@dataclass
class MotionResult:
    """Result of motion detection."""

    motion_detected: bool
    pixels_changed: int
    contour_count: int
    include_mask_area: int
    motion_in_include_area: int


class ZoneMaskBuilder:
    """Builds include/exclude masks from zone configurations."""

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def build_include_mask(self, zones: List[ZoneConfig]) -> np.ndarray:
        """Build IncludeMask = union(include zones) - union(exclude zones)."""
        # Start with all zeros
        include_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        exclude_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)

        # Process include zones
        for zone in zones:
            if zone.kind == "include":
                # Convert polygon to mask
                polygon_array = np.array(zone.polygon, dtype=np.int32)
                cv2.fillPoly(include_mask, [polygon_array], 255)

        # Process exclude zones
        for zone in zones:
            if zone.kind == "exclude":
                polygon_array = np.array(zone.polygon, dtype=np.int32)
                cv2.fillPoly(exclude_mask, [polygon_array], 255)

        # If no include zones defined, include whole frame
        if not any(zone.kind == "include" for zone in zones):
            include_mask.fill(255)

        # Subtract exclude zones from include zones
        final_mask = cv2.subtract(include_mask, exclude_mask)

        return final_mask


class MotionDetector:
    """Per-camera motion detector with zone-based gating."""

    def __init__(
        self,
        camera_uuid: str,
        motion_config: MotionGatingConfig,
        zones: List[ZoneConfig],
        frame_width: int,
        frame_height: int,
    ):
        self.camera_uuid = camera_uuid
        self.config = motion_config
        self.zones = zones
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=16, history=500
        )

        # Motion state
        self.previous_gray: Optional[np.ndarray] = None
        self.last_motion_time: Optional[datetime] = None
        self.cooldown_counter = 0

        # Build zone masks
        self.mask_builder = ZoneMaskBuilder(frame_width, frame_height)
        self.include_mask = self.mask_builder.build_include_mask(zones)

        # Statistics
        self.stats = {
            "frames_processed": 0,
            "frames_with_motion": 0,
            "frames_skipped_cooldown": 0,
            "frames_skipped_no_motion": 0,
        }

        logger.info(f"Motion detector initialized for {camera_uuid}")
        logger.debug(f"Include mask area: {np.sum(self.include_mask > 0)} pixels")

    def detect_motion(self, frame: np.ndarray) -> MotionResult:
        """
        Detect motion in frame according to Phase 2 specifications.

        Returns MotionResult with motion_detected=True if inference should run.
        """
        self.stats["frames_processed"] += 1

        if not self.config.enabled:
            # Motion gating disabled - always run inference
            return MotionResult(
                motion_detected=True,
                pixels_changed=0,
                contour_count=0,
                include_mask_area=np.sum(self.include_mask > 0),
                motion_in_include_area=0,
            )

        # Downscale frame for performance
        if self.config.downscale < 1.0:
            height, width = frame.shape[:2]
            new_height = int(height * self.config.downscale)
            new_width = int(width * self.config.downscale)
            frame_small = cv2.resize(frame, (new_width, new_height))
            include_mask_small = cv2.resize(self.include_mask, (new_width, new_height))
        else:
            frame_small = frame
            include_mask_small = self.include_mask

        # Convert to grayscale
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # Background subtraction
        fg_mask = self.bg_subtractor.apply(gray)

        # Frame differencing if we have previous frame
        if self.previous_gray is not None:
            frame_diff = cv2.absdiff(self.previous_gray, gray)
            # Use adaptive threshold based on local statistics
            mean_val = cv2.mean(frame_diff)[0]
            threshold = max(25, min(50, mean_val * 2))  # Dynamic threshold
            _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.bitwise_or(fg_mask, thresh)
        else:
            motion_mask = fg_mask

        self.previous_gray = gray.copy()

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        # Dilate motion mask by dilation_px
        if self.config.dilation_px > 0:
            dilation_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.dilation_px * 2 + 1, self.config.dilation_px * 2 + 1),
            )
            motion_mask = cv2.dilate(motion_mask, dilation_kernel, iterations=1)

        # Find contours and filter by size
        contours, _ = cv2.findContours(
            motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by noise floor
        significant_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config.noise_floor:
                significant_contours.append(contour)

        # Create final motion mask from significant contours
        final_motion_mask = np.zeros_like(motion_mask)
        if significant_contours:
            cv2.fillPoly(final_motion_mask, significant_contours, 255)

        # Intersect motion mask with include mask
        motion_in_include = cv2.bitwise_and(final_motion_mask, include_mask_small)
        motion_area = np.sum(motion_in_include > 0)

        # Check if motion area exceeds threshold
        significant_motion = motion_area >= self.config.min_area_px

        # Apply cooldown logic
        if significant_motion:
            self.last_motion_time = datetime.now()
            self.cooldown_counter = 0
            self.stats["frames_with_motion"] += 1
        else:
            if self.cooldown_counter < self.config.cooldown_frames:
                self.cooldown_counter += 1
                # Still in cooldown - check if we recently had motion
                if (
                    self.last_motion_time
                    and datetime.now() - self.last_motion_time < timedelta(seconds=2)
                ):
                    significant_motion = True
                    self.stats["frames_skipped_cooldown"] += 1
                else:
                    self.stats["frames_skipped_no_motion"] += 1
            else:
                self.stats["frames_skipped_no_motion"] += 1

        return MotionResult(
            motion_detected=significant_motion,
            pixels_changed=motion_area,
            contour_count=len(significant_contours),
            include_mask_area=np.sum(include_mask_small > 0),
            motion_in_include_area=motion_area,
        )

    def update_zones(self, zones: List[ZoneConfig]) -> None:
        """Update zone configuration and rebuild masks."""
        self.zones = zones
        self.include_mask = self.mask_builder.build_include_mask(zones)
        logger.info(
            f"Updated zones for {self.camera_uuid}, new include mask area: {np.sum(self.include_mask > 0)}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get motion detection statistics."""
        stats = self.stats.copy()
        if stats["frames_processed"] > 0:
            stats["motion_rate_pct"] = (
                stats["frames_with_motion"] / stats["frames_processed"] * 100
            )
            stats["skip_rate_pct"] = (
                (stats["frames_skipped_cooldown"] + stats["frames_skipped_no_motion"])
                / stats["frames_processed"]
                * 100
            )
        return stats


class MotionGatingInferenceWrapper:
    """Wraps inference handler with motion-based gating."""

    def __init__(
        self,
        inference_handler: Any,
        camera_configs: Dict[str, Tuple[MotionGatingConfig, List[ZoneConfig]]],
        frame_width: int,
        frame_height: int,
    ):
        self.inference_handler = inference_handler
        self.camera_configs = camera_configs
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Create motion detectors per camera
        self.motion_detectors: Dict[str, MotionDetector] = {}
        for camera_uuid, (motion_config, zones) in camera_configs.items():
            self.motion_detectors[camera_uuid] = MotionDetector(
                camera_uuid=camera_uuid,
                motion_config=motion_config,
                zones=zones,
                frame_width=frame_width,
                frame_height=frame_height,
            )

        # Global statistics
        self.global_stats = {
            "total_frames": 0,
            "total_skipped": 0,
            "total_processed": 0,
        }

    def __call__(self, video_frames: List[Any]) -> List[Optional[Dict]]:
        """Process video frames with motion gating."""
        results = []
        frames_to_process = []
        frame_camera_map = []  # Maps frame index to (original_index, camera_uuid)

        # Motion detection phase
        for idx, frame in enumerate(video_frames):
            camera_uuid = self._extract_camera_uuid(frame, idx)

            if camera_uuid not in self.motion_detectors:
                # No motion config for this camera - always process
                frames_to_process.append(frame)
                frame_camera_map.append((idx, camera_uuid))
                continue

            detector = self.motion_detectors[camera_uuid]
            motion_result = detector.detect_motion(frame.image)

            self.global_stats["total_frames"] += 1

            if motion_result.motion_detected:
                frames_to_process.append(frame)
                frame_camera_map.append((idx, camera_uuid))
                self.global_stats["total_processed"] += 1

                logger.debug(
                    f"Motion detected {camera_uuid}: "
                    f"{motion_result.pixels_changed}px in {motion_result.motion_in_include_area}px include area"
                )
            else:
                self.global_stats["total_skipped"] += 1
                logger.debug(
                    f"Motion skipped {camera_uuid}: insufficient motion in include zones"
                )

        # Run inference only on frames with detected motion
        if frames_to_process:
            predictions = self.inference_handler(frames_to_process)

            # Map predictions back to original frame indices
            prediction_idx = 0
            for idx in range(len(video_frames)):
                # Find if this original index has a prediction
                original_idx = None
                for map_idx, (orig_idx, _) in enumerate(frame_camera_map):
                    if orig_idx == idx:
                        original_idx = map_idx
                        break

                if original_idx is not None:
                    # This frame was processed
                    if prediction_idx < len(predictions):
                        result = predictions[prediction_idx]
                        # Mark that frame was processed (not skipped by motion)
                        if isinstance(result, dict):
                            result["skipped_by_motion"] = False
                        results.append(result)
                        prediction_idx += 1
                    else:
                        results.append(None)
                else:
                    # This frame was skipped due to no motion
                    results.append(
                        {"skipped_by_motion": True, "predictions": [], "time": 0.0}
                    )
        else:
            # No frames had motion - return skipped results
            results = [
                {"skipped_by_motion": True, "predictions": [], "time": 0.0}
                for _ in video_frames
            ]

        return results

    def _extract_camera_uuid(self, frame: Any, frame_idx: int) -> str:
        """Extract camera UUID from frame or frame index."""
        # This would depend on how frames are structured
        # For now, use source_id or frame index
        if hasattr(frame, "source_id") and frame.source_id is not None:
            source_id = frame.source_id
        else:
            source_id = frame_idx

        # Map source_id to camera_uuid based on shard config
        camera_uuids = list(self.camera_configs.keys())
        if source_id < len(camera_uuids):
            return camera_uuids[source_id]
        return f"unknown_camera_{source_id}"

    def get_motion_stats(self) -> Dict[str, Any]:
        """Get aggregated motion statistics."""
        stats = {"global": self.global_stats.copy(), "per_camera": {}}

        for camera_uuid, detector in self.motion_detectors.items():
            stats["per_camera"][camera_uuid] = detector.get_stats()

        # Calculate global skip rate
        if stats["global"]["total_frames"] > 0:
            stats["global"]["skip_rate_pct"] = (
                stats["global"]["total_skipped"] / stats["global"]["total_frames"] * 100
            )

        return stats

    def update_camera_config(
        self,
        camera_uuid: str,
        motion_config: MotionGatingConfig,
        zones: List[ZoneConfig],
    ) -> None:
        """Update configuration for a specific camera."""
        if camera_uuid in self.motion_detectors:
            self.motion_detectors[camera_uuid].update_zones(zones)
        else:
            self.motion_detectors[camera_uuid] = MotionDetector(
                camera_uuid=camera_uuid,
                motion_config=motion_config,
                zones=zones,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
            )

        self.camera_configs[camera_uuid] = (motion_config, zones)
        logger.info(f"Updated motion config for camera {camera_uuid}")
