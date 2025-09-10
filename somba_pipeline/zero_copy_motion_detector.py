"""
Zero-copy motion detection implementation for optimized performance.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timedelta
import logging

from .schemas import MotionGatingConfig, ZoneConfig
from .motion_detection import MotionResult, ZoneMaskBuilder

logger = logging.getLogger(__name__)


@dataclass
class MotionData:
    """Result of zero-copy motion detection."""

    has_motion: bool
    motion_area: int
    contour_count: int
    include_mask_area: int
    significant_contours: List[np.ndarray]
    final_motion_mask: np.ndarray
    debug: dict


class ZeroCopyMotionDetector:
    """Zero-copy motion detector with pre-allocated memory buffers."""

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

        # Flag to track if buffers have been allocated
        self.buffers_initialized = False
        self.actual_frame_width = None
        self.actual_frame_height = None

        # Calculate downscaled dimensions (will be recalculated on first frame)
        if self.config.downscale < 1.0:
            self.process_width = int(frame_width * self.config.downscale)
            self.process_height = int(frame_height * self.config.downscale)
        else:
            self.process_width = frame_width
            self.process_height = frame_height

        # Don't allocate buffers yet - wait for first frame to get actual size
        # self._allocate_memory_buffers()

        # SIMPLIFIED: Disable background subtractor - use frame-to-frame differencing only
        # Background subtractors can get stuck and add unnecessary complexity
        self.use_background_subtractor = False  # Can be made configurable later
        self.bg_subtractor = None

        # Motion state
        self.previous_gray: Optional[np.ndarray] = None
        self.last_motion_time: Optional[datetime] = None
        self.last_inference_time: Optional[datetime] = None
        self.cooldown_counter = 0

        # Warm-up period to allow background subtractor to learn
        self.warmup_frames = 100
        self.frame_count = 0

        # Statistics
        self.stats = {
            "frames_processed": 0,
            "frames_with_motion": 0,
            "frames_skipped_cooldown": 0,
            "frames_skipped_no_motion": 0,
            "frames_skipped_interval": 0,
        }

        # Zone mask builder - will build masks when we know actual frame size
        self.mask_builder = None
        self.include_mask = None
        self.include_mask_small = None

        logger.info(f"Zero-copy motion detector created for {camera_uuid}")
        logger.info(f"Expected dimensions: {frame_width}x{frame_height}")
        logger.info(f"Zone count: {len(zones)}")
        if zones:
            logger.info(f"Zone types: {[zone.kind for zone in zones]}")
        else:
            logger.info("No zones defined - will use full frame for motion detection")

    def _allocate_memory_buffers(self, actual_width=None, actual_height=None):
        """Pre-allocate all memory buffers based on actual frame size."""
        # Use actual frame dimensions if provided, otherwise use configured dimensions
        if actual_width and actual_height:
            self.actual_frame_width = actual_width
            self.actual_frame_height = actual_height
            logger.info(f"{self.camera_uuid} - Allocating buffers for actual size: {actual_width}x{actual_height}")
        else:
            self.actual_frame_width = self.frame_width
            self.actual_frame_height = self.frame_height

        # Recalculate process dimensions based on actual frame size
        if self.config.downscale < 1.0:
            self.process_width = int(self.actual_frame_width * self.config.downscale)
            self.process_height = int(self.actual_frame_height * self.config.downscale)
        else:
            self.process_width = self.actual_frame_width
            self.process_height = self.actual_frame_height

        # Grayscale buffer for original frame
        self.gray_buffer = np.empty((self.actual_frame_height, self.actual_frame_width), dtype=np.uint8)

        # Downscaled buffers
        self.gray_small_buffer = np.empty((self.process_height, self.process_width), dtype=np.uint8)
        self.include_mask_small_buffer = np.empty((self.process_height, self.process_width), dtype=np.uint8)

        # Processing buffers
        self.gray_masked_buffer = np.empty((self.process_height, self.process_width), dtype=np.uint8)
        self.prev_masked_buffer = np.empty((self.process_height, self.process_width), dtype=np.uint8)
        self.frame_diff_buffer = np.empty((self.process_height, self.process_width), dtype=np.uint8)
        self.thresh_buffer = np.empty((self.process_height, self.process_width), dtype=np.uint8)
        self.fg_mask_buffer = np.empty((self.process_height, self.process_width), dtype=np.uint8)
        self.motion_mask_buffer = np.empty((self.process_height, self.process_width), dtype=np.uint8)

        # Morphology kernel (reusable)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Dilation kernel (reusable)
        if self.config.dilation_px > 0:
            kernel_size = self.config.dilation_px * 2 + 1
            self.dilation_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
        else:
            self.dilation_kernel = None

    def process_frame(self, frame: np.ndarray) -> MotionData:
        """
        Process frame using zero-copy operations.

        Args:
            frame: Input frame (BGR format)

        Returns:
            MotionData with motion detection results
        """
        # Initialize buffers on first frame with actual dimensions
        if not self.buffers_initialized:
            h, w = frame.shape[:2]
            logger.info(f"{self.camera_uuid} - First frame received: {w}x{h}")

            # Allocate buffers with actual frame size
            self._allocate_memory_buffers(w, h)

            # Build zone masks with actual frame size
            self.mask_builder = ZoneMaskBuilder(w, h)
            self.include_mask = self.mask_builder.build_include_mask(self.zones)

            # Check include mask coverage
            include_area = np.sum(self.include_mask > 0)
            total_area = self.include_mask.shape[0] * self.include_mask.shape[1]
            logger.info(f"{self.camera_uuid} - Include mask: {include_area}/{total_area} pixels ({100*include_area/total_area:.1f}%)")

            # Create downscaled include mask if needed
            if self.config.downscale < 1.0:
                self.include_mask_small = cv2.resize(
                    self.include_mask,
                    (self.process_width, self.process_height),
                    interpolation=cv2.INTER_AREA
                )
            else:
                self.include_mask_small = self.include_mask

            self.buffers_initialized = True
            logger.info(f"{self.camera_uuid} - Buffers initialized for {w}x{h}, process size: {self.process_width}x{self.process_height}")

        self.stats["frames_processed"] += 1
        self.frame_count += 1

        if not self.config.enabled:
            # Motion gating disabled - always trigger inference
            return MotionData(
                has_motion=True,
                motion_area=0,
                contour_count=0,
                include_mask_area=np.sum(self.include_mask > 0),
                significant_contours=[],
                final_motion_mask=np.zeros((self.process_height, self.process_width), dtype=np.uint8),
                debug={"disabled": True}
            )

        # During warm-up period, always trigger inference to allow background learning
        # if self.frame_count <= self.warmup_frames:
        if False:
            logger.info(f"Warm-up period {self.camera_uuid}: frame {self.frame_count}/{self.warmup_frames} - forcing inference")

            # Convert to grayscale (don't use dst parameter - may not work correctly)
            gray_temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            np.copyto(self.gray_buffer, gray_temp)

            # Zero-copy downscaling if needed
            if self.config.downscale < 1.0:
                gray_small_temp = cv2.resize(
                    self.gray_buffer,
                    (self.process_width, self.process_height),
                    interpolation=cv2.INTER_AREA
                )
                np.copyto(self.gray_small_buffer, gray_small_temp)
                gray_small = self.gray_small_buffer
            else:
                gray_small = self.gray_buffer

            # CRITICAL: Initialize previous frame properly
            if self.previous_gray is None:
                self.previous_gray = gray_small.copy()
                logger.info(f"{self.camera_uuid} - Initialized previous_gray during warmup")
            else:
                # Update previous frame for next comparison
                self.previous_gray = gray_small.copy()

            return MotionData(
                has_motion=True,
                motion_area=0,
                contour_count=0,
                include_mask_area=np.sum(self.include_mask > 0),
                significant_contours=[],
                final_motion_mask=np.zeros((self.process_height, self.process_width), dtype=np.uint8),
                debug={"warmup": True}
            )

        # DIAGNOSTIC: Check input frame before conversion
        if self.stats["frames_processed"] % 30 == 0:
            logger.debug(f"{self.camera_uuid} - Input frame check:")
            logger.debug(f"  - Frame shape: {frame.shape}, dtype: {frame.dtype}")
            logger.debug(f"  - Frame mean (BGR): {np.mean(frame):.1f}, max: {np.max(frame)}, min: {np.min(frame)}")

        # CRITICAL FIX: Don't use dst parameter - it might not work correctly with pre-allocated buffers
        # Convert to grayscale normally first
        gray_temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Verify conversion worked
        if np.mean(gray_temp) == 0:
            logger.warning(f"{self.camera_uuid} - Grayscale conversion resulted in black frame! Input mean: {np.mean(frame):.1f}")

        # Copy to buffer (ensures proper memory layout)
        np.copyto(self.gray_buffer, gray_temp)

        # Zero-copy downscaling if needed
        if self.config.downscale < 1.0:
            # Use normal resize first, then copy to buffer
            gray_small_temp = cv2.resize(
                self.gray_buffer,
                (self.process_width, self.process_height),
                interpolation=cv2.INTER_AREA
            )
            np.copyto(self.gray_small_buffer, gray_small_temp)
            gray_small = self.gray_small_buffer
            # CRITICAL FIX: Use the actual downscaled mask, not the empty buffer!
            include_mask_small = self.include_mask_small
        else:
            gray_small = self.gray_buffer
            include_mask_small = self.include_mask

        # Ensure mask is binary uint8 0/255
        if include_mask_small.dtype != np.uint8:
            include_mask_small = include_mask_small.astype(np.uint8)
        if include_mask_small.max() == 1:
            include_mask_small = include_mask_small * 255

        # Debug ROI mask info (only log occasionally to reduce spam)
        roi_area = np.sum(include_mask_small > 0)
        if self.stats["frames_processed"] % 100 == 0:  # Log every 100 frames
            total_area = include_mask_small.shape[0] * include_mask_small.shape[1]
            logger.debug(f"{self.camera_uuid} - ROI mask: {roi_area}/{total_area} pixels ({100*roi_area/total_area:.1f}%)")

        # Zero-copy motion detection pipeline
        motion_data = self._detect_motion_zero_copy(gray_small, include_mask_small)

        # Log motion detection details at DEBUG level for debugging
        logger.debug(
            f"Motion debug {self.camera_uuid}: "
            f"area={motion_data.motion_area} >= min={motion_data.debug.get('min_area_px', 0)}? "
            f"{motion_data.has_motion}, contours={motion_data.contour_count}, "
            f"cooldown_left={motion_data.debug.get('cooldown_frames_left', 0)}, "
            f"reason={'motion' if motion_data.motion_area > 0 else 'no_motion'}"
        )

        return motion_data

    def _detect_motion_zero_copy(self, gray: np.ndarray, roi_mask: np.ndarray) -> MotionData:
        """Core zero-copy motion detection logic."""

        # DIAGNOSTIC: Check inputs before processing
        if self.stats["frames_processed"] % 30 == 0:  # Log every 30 frames
            logger.debug(f"{self.camera_uuid} - Motion detector state check:")
            logger.debug(f"  - Gray shape: {gray.shape}, dtype: {gray.dtype}, mean: {np.mean(gray):.1f}")
            logger.debug(f"  - ROI mask shape: {roi_mask.shape}, dtype: {roi_mask.dtype}, pixels: {np.sum(roi_mask > 0)}")
            logger.debug(f"  - Previous gray exists: {self.previous_gray is not None}")
            if self.previous_gray is not None:
                logger.debug(f"  - Previous gray shape: {self.previous_gray.shape}, mean: {np.mean(self.previous_gray):.1f}")

        # Mask frames BEFORE differencing (zero-copy)
        cv2.bitwise_and(gray, gray, mask=roi_mask, dst=self.gray_masked_buffer)
        gray_masked = self.gray_masked_buffer

        if self.previous_gray is not None:
            cv2.bitwise_and(self.previous_gray, self.previous_gray, mask=roi_mask, dst=self.prev_masked_buffer)
            prev_masked = self.prev_masked_buffer
            cv2.absdiff(prev_masked, gray_masked, dst=self.frame_diff_buffer)

            # DIAGNOSTIC: Check if frames are actually different
            diff_mean = np.mean(self.frame_diff_buffer)
            diff_max = np.max(self.frame_diff_buffer)
            if self.stats["frames_processed"] % 30 == 0:
                logger.debug(f"  - Frame diff mean: {diff_mean:.2f}, max: {diff_max}")
                # Check if current and previous are identical (stuck state)
                if np.array_equal(gray_masked, prev_masked):
                    logger.warning(f"{self.camera_uuid} - IDENTICAL FRAMES DETECTED! Motion detector may be stuck.")
        else:
            # First frame: no diff
            self.frame_diff_buffer.fill(0)
            logger.info(f"{self.camera_uuid} - First frame, initializing previous_gray")

        frame_diff = self.frame_diff_buffer

        # Adaptive threshold using ONLY ROI pixels with better sensitivity
        mean_val = cv2.mean(frame_diff, mask=roi_mask)[0]
        # Lower threshold for better motion sensitivity (since we're not using BG subtractor)
        dynamic_thresh = max(5.0, mean_val * 1.5)  # More sensitive threshold
        cv2.threshold(frame_diff, dynamic_thresh, 255, cv2.THRESH_BINARY, dst=self.thresh_buffer)

        # Apply ROI mask to threshold result
        cv2.bitwise_and(self.thresh_buffer, roi_mask, dst=self.thresh_buffer)

        # SIMPLIFIED: Skip background subtraction, use only frame differencing
        # This is more reliable and doesn't get stuck
        thresh_pixels = np.sum(self.thresh_buffer > 0)

        # DIAGNOSTIC: Log detection state regularly
        if self.stats["frames_processed"] % 10 == 0:  # More frequent logging for debugging
            logger.debug(f"{self.camera_uuid} - Motion detection state:")
            logger.debug(f"  - Mean diff: {mean_val:.2f}, Threshold: {dynamic_thresh:.2f}")
            logger.debug(f"  - Diff pixels: {thresh_pixels}")
            logger.debug(f"  - ROI pixels: {np.sum(roi_mask > 0)}")

        # Use threshold buffer directly as motion mask (simpler, more reliable)
        np.copyto(self.motion_mask_buffer, self.thresh_buffer)

        # Morphological operations to clean up noise (zero-copy) with appropriate kernel
        noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Very small for noise removal
        cv2.morphologyEx(self.motion_mask_buffer, cv2.MORPH_OPEN, noise_kernel, dst=self.motion_mask_buffer)

        # Use larger kernel for closing to connect nearby motion areas
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.morphologyEx(self.motion_mask_buffer, cv2.MORPH_CLOSE, connect_kernel, dst=self.motion_mask_buffer)

        # Dilate motion mask if needed (zero-copy)
        if self.dilation_kernel is not None:
            cv2.dilate(self.motion_mask_buffer, self.dilation_kernel, dst=self.motion_mask_buffer)

        # Find contours and filter by size
        contours, _ = cv2.findContours(
            self.motion_mask_buffer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Keep ALL contours for debug display, but filter for significance
        all_contours = list(contours)  # Keep all for debug
        significant_contours = []
        below_threshold_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config.noise_floor:
                significant_contours.append(contour)
            else:
                below_threshold_contours.append(contour)

        # Create final motion mask from significant contours (zero-copy)
        final_motion_mask = np.zeros_like(self.motion_mask_buffer)
        if significant_contours:
            cv2.fillPoly(final_motion_mask, significant_contours, 255)

        # Calculate statistics - include ALL motion pixels (before filtering)
        raw_motion_pixels = np.sum(self.motion_mask_buffer > 0)  # All detected motion
        filtered_motion_area = np.sum(final_motion_mask > 0)  # After noise filtering
        roi_area = np.sum(roi_mask > 0)

        # Calculate percentages
        raw_motion_percent = (raw_motion_pixels / max(1, roi_area)) * 100.0
        filtered_motion_percent = (filtered_motion_area / max(1, roi_area)) * 100.0

        # Resolve min_area_px
        if hasattr(self.config, 'min_area_mode') and self.config.min_area_mode == "roi_percent":
            min_area_px = int((self.config.min_area_roi_percent / 100.0) * max(1, roi_area))
        else:
            min_area_px = int(self.config.min_area_px)

        # Check if motion area exceeds threshold (use filtered area)
        significant_motion = filtered_motion_area >= min_area_px

        # Track whether we're above or below threshold
        threshold_status = "above" if significant_motion else "below"

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

        # Check maximum inference interval (timeout fallback)
        current_time = datetime.now()
        timeout_triggered = False
        if (
            hasattr(self.config, 'max_inference_interval_seconds')
            and self.config.max_inference_interval_seconds > 0
            and self.last_inference_time is not None
            and (current_time - self.last_inference_time).total_seconds() >= self.config.max_inference_interval_seconds
        ):
            # Force inference due to maximum interval
            significant_motion = True
            timeout_triggered = True
            self.stats["frames_skipped_interval"] = self.stats.get("frames_skipped_interval", 0) + 1

            logger.debug(
                f"Force inference {self.camera_uuid}: "
                f"{(current_time - self.last_inference_time).total_seconds():.1f}s since last inference"
            )

        # Update last inference time if we're running inference
        if significant_motion:
            self.last_inference_time = current_time

        # CRITICAL: Update state AFTER computation
        # Must copy the ORIGINAL gray frame, not the masked version!
        self.previous_gray = gray.copy()  # Full frame copy for next comparison

        # DIAGNOSTIC: Verify previous frame was updated
        if self.stats["frames_processed"] % 30 == 0:
            logger.debug(f"{self.camera_uuid} - Previous frame updated. New mean: {np.mean(self.previous_gray):.1f}")

        # Build debug info with enhanced metrics
        debug = {
            # Motion pixel counts
            "raw_motion_pixels": raw_motion_pixels,  # All detected motion pixels
            "filtered_motion_area": filtered_motion_area,  # After noise filtering
            "roi_area": roi_area,

            # Motion percentages
            "raw_motion_percent": raw_motion_percent,  # % of ROI with any motion
            "filtered_motion_percent": filtered_motion_percent,  # % after filtering

            # Threshold info
            "min_area_px": min_area_px,
            "threshold_status": threshold_status,  # "above" or "below"
            "threshold_percent": (min_area_px / max(1, roi_area)) * 100.0,

            # Contour info
            "significant_contours": len(significant_contours),
            "below_threshold_contours": len(below_threshold_contours),
            "total_contours": len(all_contours),
            "all_contours": all_contours,  # Keep ALL contours for display

            # Processing params
            "mean_diff": mean_val,
            "dynamic_threshold": dynamic_thresh,
            "noise_floor": self.config.noise_floor,

            # State info
            "cooldown_frames_left": max(0, self.config.cooldown_frames - self.cooldown_counter),
            "timeout_triggered": timeout_triggered,
            "time_since_last_inference": (current_time - self.last_inference_time).total_seconds() if self.last_inference_time else 0.0,
        }

        return MotionData(
            has_motion=significant_motion,
            motion_area=filtered_motion_area,  # Use filtered area
            contour_count=len(significant_contours),
            include_mask_area=roi_area,
            significant_contours=significant_contours,  # Only significant ones trigger inference
            final_motion_mask=final_motion_mask,
            debug=debug
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get motion detection statistics."""
        total_processed = self.stats["frames_processed"]
        if total_processed == 0:
            return self.stats

        return {
            **self.stats,
            "motion_rate": self.stats["frames_with_motion"] / total_processed,
            "skip_rate": (self.stats["frames_skipped_cooldown"] +
                         self.stats["frames_skipped_no_motion"] +
                         self.stats["frames_skipped_interval"]) / total_processed,
        }

    def reset_stats(self):
        """Reset motion detection statistics."""
        for key in self.stats:
            self.stats[key] = 0

    def should_infer(self) -> bool:
        """Check if inference should be triggered based on motion and timeout."""
        current_time = datetime.now()

        # Check if we're in cooldown period after recent motion
        if (
            self.last_motion_time
            and (current_time - self.last_motion_time).total_seconds() < 2.0
        ):
            return True

        # Check maximum inference interval
        if (
            hasattr(self.config, 'max_inference_interval_seconds')
            and self.config.max_inference_interval_seconds > 0
            and self.last_inference_time is not None
            and (current_time - self.last_inference_time).total_seconds() >= self.config.max_inference_interval_seconds
        ):
            return True

        return False
