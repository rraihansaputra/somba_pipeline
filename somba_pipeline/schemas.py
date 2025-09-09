"""
Event schemas v2 for detections and status events with zone support.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field
import hashlib
import json


class ZoneConfig(BaseModel):
    """Zone configuration."""

    zone_id: int = Field(
        ..., ge=1, description="Zone ID (>= 1, 0 is reserved for whole frame)"
    )
    name: str = Field(..., description="Zone name")
    kind: str = Field(..., pattern="^(include|exclude)$", description="Zone kind")
    priority: int = Field(..., description="Zone priority (higher wins)")
    polygon: List[List[int]] = Field(
        ..., min_items=3, description="Polygon vertices [[x,y], ...]"
    )
    allow_labels: Optional[List[str]] = Field(None, description="Allowed labels")
    deny_labels: Optional[List[str]] = Field(None, description="Denied labels")
    min_score: Optional[float] = Field(
        None, ge=0, le=1, description="Minimum score threshold"
    )


class MotionGatingConfig(BaseModel):
    """Motion gating configuration."""

    enabled: bool = Field(True, description="Enable motion gating")
    downscale: float = Field(
        0.5, gt=0, le=1, description="Downscale factor for motion detection"
    )
    dilation_px: int = Field(6, ge=0, description="Dilation pixels")
    min_area_px: int = Field(
        1500, ge=0, description="Minimum area intersection with IncludeMask"
    )
    cooldown_frames: int = Field(2, ge=0, description="Cooldown frames for hysteresis")
    noise_floor: int = Field(12, ge=0, description="Ignore tiny contours")

    # New: maximum time between inference (even when no motion)
    max_inference_interval_seconds: float = Field(
        10, ge=0, description="Maximum seconds between inference (0 = no max interval)"
    )

    # Advanced settings
    roi_native: bool = Field(True, description="Compute motion entirely inside include zones")
    adaptive_threshold_factor: float = Field(0.7, description="Adaptive threshold factor for ROI-native mode")
    min_area_mode: Literal["px", "roi_percent"] = Field("px", description="Minimum area mode")
    min_area_roi_percent: float = Field(0.5, description="Minimum area as percentage of ROI")


class CameraConfig(BaseModel):
    """Per-camera configuration."""

    camera_uuid: str
    zones: List[ZoneConfig] = Field(default_factory=list)
    motion_gating: MotionGatingConfig = Field(default_factory=MotionGatingConfig)
    allow_labels: Optional[List[str]] = Field(None, description="Global allow labels")
    deny_labels: Optional[List[str]] = Field(None, description="Global deny labels")
    min_score: float = Field(0.30, ge=0, le=1, description="Global min score")
    zone_test: str = Field("center", pattern="^(center|center\\+iou)$")
    iou_threshold: float = Field(
        0.10, ge=0, le=1, description="IoU threshold for center+iou test"
    )

    def get_zones_hash(self) -> str:
        """Generate hash of zones configuration for drift detection."""
        zones_data = [zone.model_dump() for zone in self.zones]
        zones_json = json.dumps(zones_data, sort_keys=True)
        return hashlib.sha256(zones_json.encode()).hexdigest()


class ModelInfo(BaseModel):
    """Model information."""

    id: str = Field(..., description="Model ID")
    adapter: str = Field(..., description="Model adapter")


class FrameInfo(BaseModel):
    """Frame information."""

    w: int = Field(..., description="Frame width")
    h: int = Field(..., description="Frame height")
    seq: int = Field(..., description="Frame sequence number")
    fps: float = Field(..., description="Current FPS")
    skipped_by_motion: bool = Field(False, description="Frame skipped by motion gating")


class ZonesConfigInfo(BaseModel):
    """Zones configuration metadata."""

    zone_version: str = Field(..., description="SHA256 hash of zones configuration")
    zone_test: str = Field(..., description="Zone test method")
    iou_threshold: Optional[float] = Field(
        None, description="IoU threshold if using center+iou"
    )


class ZoneMembership(BaseModel):
    """Zone membership details for auditing."""

    center_in: bool = Field(..., description="Object center is in zone")
    iou: float = Field(..., description="IoU between object bbox and zone polygon")


class DetectedObject(BaseModel):
    """Detected object with zone attribution."""

    label: str = Field(..., description="Object label")
    score: float = Field(..., ge=0, le=1, description="Detection confidence score")
    bbox_xywh: List[float] = Field(
        ..., min_items=4, max_items=4, description="Bounding box [x,y,w,h]"
    )
    segmentation: Optional[List[List[float]]] = Field(
        None, description="Segmentation polygons"
    )
    primary_zone_id: int = Field(..., description="Primary zone ID (0 for whole frame)")
    zones_hit: List[int] = Field(
        ..., description="All zones hit, sorted by priority desc"
    )
    zone_membership: Optional[Dict[str, ZoneMembership]] = Field(
        None, description="Zone membership details"
    )
    filtered: bool = Field(False, description="Would be filtered by zone rules")
    filter_reason: Optional[str] = Field(
        None, description="Filter reason if applicable"
    )


class DetectionLatency(BaseModel):
    """Detection latency information."""

    inference_s: float = Field(..., description="Inference time in seconds")
    e2e_s: float = Field(..., description="End-to-end latency in seconds")


class DetectionEvent(BaseModel):
    """Detection event schema v2."""

    schema_version: int = Field(2, description="Schema version")
    event_id: str = Field(..., description="Event ID (ULID)")
    ts_ns: int = Field(..., description="Timestamp in nanoseconds")
    tenant_id: str = Field(..., description="Tenant ID")
    site_id: str = Field(..., description="Site ID")
    camera_uuid: str = Field(..., description="Camera UUID")
    model: ModelInfo = Field(..., description="Model information")
    frame: FrameInfo = Field(..., description="Frame information")
    zones_config: ZonesConfigInfo = Field(..., description="Zones configuration")
    objects: List[DetectedObject] = Field(..., description="Detected objects")


class ZoneStats(BaseModel):
    """Per-zone statistics."""

    objects: int = Field(..., description="Objects detected in zone")
    dropped: int = Field(..., description="Objects dropped by filters")


class ZonesStats(BaseModel):
    """Zones statistics for status events."""

    frames_skipped_motion: int = Field(
        ..., description="Frames skipped by motion gating"
    )
    frames_processed: int = Field(..., description="Frames processed")
    objects_published: int = Field(..., description="Objects published")
    objects_dropped_by_filters: int = Field(
        ..., description="Objects dropped by filters"
    )
    per_zone: Dict[str, ZoneStats] = Field(..., description="Per-zone statistics")


class StatusEvent(BaseModel):
    """Status event with optional zone statistics."""

    type: str = Field(..., description="Event type")
    state: Optional[str] = Field(None, description="Stream state")
    camera_uuid: str = Field(..., description="Camera UUID")
    runner_id: str = Field(..., description="Runner ID")
    shard_id: str = Field(..., description="Shard ID")
    fps: Optional[float] = Field(None, description="Stream FPS")
    last_frame_ts: Optional[str] = Field(None, description="Last frame timestamp")
    last_frame_age_s: Optional[float] = Field(
        None, description="Last frame age in seconds"
    )
    zones_stats: Optional[ZonesStats] = Field(None, description="Zone statistics")
    ts: str = Field(..., description="Event timestamp")


class ErrorEvent(BaseModel):
    """Stream error event."""

    type: str = Field("stream.error", description="Event type")
    camera_uuid: str = Field(..., description="Camera UUID")
    runner_id: str = Field(..., description="Runner ID")
    code: str = Field(..., description="Error code")
    detail: str = Field(..., description="Error detail")
    retry_in_ms: int = Field(..., description="Retry delay in milliseconds")
    ts: str = Field(..., description="Event timestamp")


class ShardConfig(BaseModel):
    """Worker shard configuration."""

    runner_id: str = Field(..., description="Runner ID")
    shard_id: str = Field(..., description="Shard ID")
    max_fps: int = Field(6, ge=1, description="Maximum FPS per camera")
    sources: List[Dict[str, str]] = Field(..., description="Camera sources")
    amqp: Dict[str, str] = Field(..., description="AMQP configuration")
    cp: Dict[str, str] = Field(..., description="Control plane configuration")
    telemetry: Dict[str, Any] = Field(..., description="Telemetry configuration")
    cameras: Dict[str, CameraConfig] = Field(
        default_factory=dict, description="Per-camera configs"
    )

    @classmethod
    def from_json_file(cls, file_path: str) -> "ShardConfig":
        """Load shard config from JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)
