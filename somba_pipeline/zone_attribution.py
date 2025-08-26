"""
Zone attribution system implementing precedence rules and per-zone label filtering.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass
import logging

from .schemas import ZoneConfig, DetectedObject, ZoneMembership, CameraConfig

logger = logging.getLogger(__name__)


@dataclass
class DetectionInput:
    """Input detection before zone processing."""

    label: str
    score: float
    bbox_xywh: List[float]
    segmentation: Optional[List[List[float]]] = None


@dataclass
class ZoneAssignment:
    """Result of zone assignment for a detection."""

    primary_zone_id: int
    zones_hit: List[int]  # Sorted by priority desc
    zone_membership: Dict[str, ZoneMembership]
    passed_filters: bool
    filter_reason: Optional[str] = None


def point_in_polygon(point: Tuple[float, float], polygon: List[List[int]]) -> bool:
    """Test if point is inside polygon using ray casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def calculate_bbox_polygon_iou(
    bbox_xywh: List[float],
    polygon: List[List[int]],
    frame_width: int,
    frame_height: int,
) -> float:
    """Calculate IoU between bounding box and polygon mask."""
    x, y, w, h = bbox_xywh

    # Create bbox mask
    bbox_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    # Clamp to frame bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_width, x2), min(frame_height, y2)
    bbox_mask[y1:y2, x1:x2] = 255

    # Create polygon mask
    polygon_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    polygon_array = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(polygon_mask, [polygon_array], 255)

    # Calculate intersection and union
    intersection = cv2.bitwise_and(bbox_mask, polygon_mask)
    union = cv2.bitwise_or(bbox_mask, polygon_mask)

    intersection_area = np.sum(intersection > 0)
    union_area = np.sum(union > 0)

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


class ZoneAttributor:
    """Handles zone attribution and filtering for detections."""

    def __init__(
        self, camera_config: CameraConfig, frame_width: int, frame_height: int
    ):
        self.camera_config = camera_config
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Sort zones by priority (descending)
        self.zones_by_priority = sorted(
            camera_config.zones, key=lambda z: z.priority, reverse=True
        )

        # Build zone lookup for quick access
        self.zone_lookup = {zone.zone_id: zone for zone in camera_config.zones}

        # Statistics
        self.stats = {
            "detections_processed": 0,
            "detections_assigned_zone_0": 0,
            "detections_dropped_filters": 0,
            "per_zone_assignments": {},
            "per_zone_drops": {},
        }

        logger.info(
            f"Zone attributor initialized with {len(camera_config.zones)} zones"
        )

    def assign_zones(self, detections: List[DetectionInput]) -> List[ZoneAssignment]:
        """Assign zones to detections and apply filters."""
        results = []

        for detection in detections:
            assignment = self._assign_single_detection(detection)
            results.append(assignment)

            # Update statistics
            self.stats["detections_processed"] += 1
            if assignment.primary_zone_id == 0:
                self.stats["detections_assigned_zone_0"] += 1
            if not assignment.passed_filters:
                self.stats["detections_dropped_filters"] += 1
                zone_id_str = str(assignment.primary_zone_id)
                self.stats["per_zone_drops"][zone_id_str] = (
                    self.stats["per_zone_drops"].get(zone_id_str, 0) + 1
                )
            else:
                zone_id_str = str(assignment.primary_zone_id)
                self.stats["per_zone_assignments"][zone_id_str] = (
                    self.stats["per_zone_assignments"].get(zone_id_str, 0) + 1
                )

        return results

    def _assign_single_detection(self, detection: DetectionInput) -> ZoneAssignment:
        """Assign zones to a single detection."""
        # Calculate center point
        x, y, w, h = detection.bbox_xywh
        center_x = x + w / 2
        center_y = y + h / 2
        center_point = (center_x, center_y)

        # Find candidate zones
        candidate_zones = []
        zone_membership = {}

        for zone in self.zones_by_priority:
            # Test center-in-polygon
            center_in = point_in_polygon(center_point, zone.polygon)

            # Calculate IoU if needed
            iou = 0.0
            if center_in or self.camera_config.zone_test == "center+iou":
                iou = calculate_bbox_polygon_iou(
                    detection.bbox_xywh,
                    zone.polygon,
                    self.frame_width,
                    self.frame_height,
                )

            # Store membership details for auditing
            zone_membership[str(zone.zone_id)] = ZoneMembership(
                center_in=center_in, iou=iou
            )

            # Determine if this zone qualifies
            qualifies = center_in
            if self.camera_config.zone_test == "center+iou":
                qualifies = center_in and iou >= self.camera_config.iou_threshold

            if qualifies:
                candidate_zones.append(zone)

        # Determine primary zone (highest priority among candidates)
        if candidate_zones:
            primary_zone = candidate_zones[0]  # Already sorted by priority desc
            primary_zone_id = primary_zone.zone_id
            zones_hit = [zone.zone_id for zone in candidate_zones]
        else:
            # No zones matched - assign to zone 0 (whole frame)
            primary_zone = None
            primary_zone_id = 0
            zones_hit = [0]
            # Add zone 0 membership
            zone_membership["0"] = ZoneMembership(center_in=True, iou=1.0)

        # Apply label filtering
        passed_filters, filter_reason = self._apply_label_filters(
            detection, primary_zone
        )

        return ZoneAssignment(
            primary_zone_id=primary_zone_id,
            zones_hit=zones_hit,
            zone_membership=zone_membership,
            passed_filters=passed_filters,
            filter_reason=filter_reason,
        )

    def _apply_label_filters(
        self, detection: DetectionInput, primary_zone: Optional[ZoneConfig]
    ) -> Tuple[bool, Optional[str]]:
        """Apply label filtering rules with zone precedence."""

        # Determine which filters to use
        if primary_zone and (
            primary_zone.allow_labels is not None
            or primary_zone.deny_labels is not None
            or primary_zone.min_score is not None
        ):
            # Use zone-specific filters
            allow_labels = primary_zone.allow_labels
            deny_labels = primary_zone.deny_labels or []
            min_score = primary_zone.min_score or 0.0
            filter_source = f"zone_{primary_zone.zone_id}"
        else:
            # Use camera/global filters
            allow_labels = self.camera_config.allow_labels
            deny_labels = self.camera_config.deny_labels or []
            min_score = self.camera_config.min_score
            filter_source = "global"

        logger.debug(
            f"Applying {filter_source} filters to {detection.label} "
            f"(score={detection.score:.3f})"
        )

        # Apply score threshold first
        if detection.score < min_score:
            return False, f"min_score_{filter_source}"

        # Apply deny list (takes precedence)
        if detection.label in deny_labels:
            return False, f"deny_label_{filter_source}"

        # Apply allow list (if specified)
        if allow_labels is not None and detection.label not in allow_labels:
            return False, f"allow_label_{filter_source}"

        return True, None

    def create_detected_object(
        self, detection: DetectionInput, assignment: ZoneAssignment
    ) -> DetectedObject:
        """Create DetectedObject with zone attribution."""
        return DetectedObject(
            label=detection.label,
            score=detection.score,
            bbox_xywh=detection.bbox_xywh,
            segmentation=detection.segmentation,
            primary_zone_id=assignment.primary_zone_id,
            zones_hit=assignment.zones_hit,
            zone_membership=assignment.zone_membership,
            filtered=not assignment.passed_filters,
            filter_reason=assignment.filter_reason,
        )

    def process_detections(
        self, raw_detections: Any
    ) -> Tuple[List[DetectedObject], Dict[str, Any]]:
        """
        Process raw detections through zone attribution and filtering.

        Handles both dictionary format and supervision Detections object.

        Returns:
            Tuple of (published_objects, zone_stats)
        """
        # Convert raw detections to input format
        detection_inputs = []

        # Check if this is a supervision Detections object
        if hasattr(raw_detections, "xyxy") and hasattr(raw_detections, "confidence"):
            # Handle supervision Detections object
            if raw_detections.xyxy is not None and len(raw_detections.xyxy) > 0:
                for i in range(len(raw_detections.xyxy)):
                    # Get bounding box in xyxy format
                    x1, y1, x2, y2 = raw_detections.xyxy[i]
                    bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                    # Get label from class_name in data
                    label = "unknown"
                    if raw_detections.data and "class_name" in raw_detections.data:
                        label = str(raw_detections.data["class_name"][i])

                    # Get confidence score
                    score = (
                        float(raw_detections.confidence[i])
                        if raw_detections.confidence is not None
                        else 0.0
                    )

                    detection_inputs.append(
                        DetectionInput(
                            label=label,
                            score=score,
                            bbox_xywh=bbox_xywh,
                            segmentation=None,
                        )
                    )
        elif isinstance(raw_detections, list):
            # Handle list of dictionaries format
            for det in raw_detections:
                # Handle different detection formats
                if isinstance(det, dict):
                    if "class" in det:
                        label = det["class"]
                        score = det.get("conf", det.get("confidence", 0.0))
                        # Convert bbox format if needed
                        if "bbox_xyxy" in det:
                            x1, y1, x2, y2 = det["bbox_xyxy"]
                            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
                        else:
                            bbox_xywh = det.get(
                                "bbox_xywh", det.get("bbox", [0, 0, 0, 0])
                            )
                    else:
                        label = det.get("label", "unknown")
                        score = det.get("score", 0.0)
                        bbox_xywh = det.get("bbox_xywh", [0, 0, 0, 0])

                    detection_inputs.append(
                        DetectionInput(
                            label=label,
                            score=score,
                            bbox_xywh=bbox_xywh,
                            segmentation=det.get("segmentation"),
                        )
                    )

        # Assign zones
        assignments = self.assign_zones(detection_inputs)

        # Create detected objects
        all_objects = []
        published_objects = []

        for detection, assignment in zip(detection_inputs, assignments):
            detected_obj = self.create_detected_object(detection, assignment)
            all_objects.append(detected_obj)

            if assignment.passed_filters:
                published_objects.append(detected_obj)

        # Generate zone statistics
        zone_stats = self._generate_zone_stats(all_objects)

        logger.debug(
            f"Processed {len(detection_inputs)} detections: "
            f"{len(published_objects)} published, "
            f"{len(all_objects) - len(published_objects)} filtered"
        )

        return published_objects, zone_stats

    def _generate_zone_stats(self, objects: List[DetectedObject]) -> Dict[str, Any]:
        """Generate zone statistics for status events."""
        zone_stats = {}

        # Count objects and drops per zone
        for obj in objects:
            zone_id_str = str(obj.primary_zone_id)
            if zone_id_str not in zone_stats:
                zone_stats[zone_id_str] = {"objects": 0, "dropped": 0}

            if obj.filtered:
                zone_stats[zone_id_str]["dropped"] += 1
            else:
                zone_stats[zone_id_str]["objects"] += 1

        return zone_stats

    def update_camera_config(self, camera_config: CameraConfig) -> None:
        """Update camera configuration."""
        self.camera_config = camera_config
        self.zones_by_priority = sorted(
            camera_config.zones, key=lambda z: z.priority, reverse=True
        )
        self.zone_lookup = {zone.zone_id: zone for zone in camera_config.zones}
        logger.info(f"Updated camera config with {len(camera_config.zones)} zones")

    def get_stats(self) -> Dict[str, Any]:
        """Get zone attribution statistics."""
        stats = self.stats.copy()

        if stats["detections_processed"] > 0:
            stats["zone_0_rate_pct"] = (
                stats["detections_assigned_zone_0"]
                / stats["detections_processed"]
                * 100
            )
            stats["filter_drop_rate_pct"] = (
                stats["detections_dropped_filters"]
                / stats["detections_processed"]
                * 100
            )

        return stats


class MultiCameraZoneAttributor:
    """Manages zone attributors for multiple cameras."""

    def __init__(
        self,
        camera_configs: Dict[str, CameraConfig],
        frame_width: int,
        frame_height: int,
    ):
        self.camera_configs = camera_configs
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Create attributors per camera
        self.attributors: Dict[str, ZoneAttributor] = {}
        for camera_uuid, config in camera_configs.items():
            self.attributors[camera_uuid] = ZoneAttributor(
                config, frame_width, frame_height
            )

        logger.info(
            f"Multi-camera zone attributor initialized for {len(camera_configs)} cameras"
        )

    def process_camera_detections(
        self, camera_uuid: str, raw_detections: Any
    ) -> Tuple[List[DetectedObject], Dict[str, Any]]:
        """Process detections for a specific camera."""
        if camera_uuid not in self.attributors:
            # No zone config for this camera - create default
            logger.warning(f"No zone config for camera {camera_uuid}, creating default")
            default_config = CameraConfig(camera_uuid=camera_uuid)
            self.attributors[camera_uuid] = ZoneAttributor(
                default_config, self.frame_width, self.frame_height
            )

        return self.attributors[camera_uuid].process_detections(raw_detections)

    def update_camera_config(self, camera_uuid: str, config: CameraConfig) -> None:
        """Update configuration for a specific camera."""
        self.camera_configs[camera_uuid] = config

        if camera_uuid in self.attributors:
            self.attributors[camera_uuid].update_camera_config(config)
        else:
            self.attributors[camera_uuid] = ZoneAttributor(
                config, self.frame_width, self.frame_height
            )

        logger.info(f"Updated zone config for camera {camera_uuid}")

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all cameras."""
        return {
            camera_uuid: attributor.get_stats()
            for camera_uuid, attributor in self.attributors.items()
        }

    def get_zones_config_hash(self, camera_uuid: str) -> str:
        """Get zones configuration hash for a camera."""
        if camera_uuid in self.camera_configs:
            return self.camera_configs[camera_uuid].get_zones_hash()
        return "no_zones"
