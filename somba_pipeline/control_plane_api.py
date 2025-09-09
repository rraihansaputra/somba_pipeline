"""
Control Plane API Interface Specifications for Phase 3 Integration.

This module defines the exact API contracts that the external Django REST Framework
control plane must implement for seamless integration with the Somba Pipeline.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class LeaseStatus(str, Enum):
    """Lease status enumeration."""

    ACTIVE = "active"
    EXPIRED = "expired"
    RELEASED = "released"
    REVOKED = "revoked"
    PENDING = "pending"


class RunnerStatus(str, Enum):
    """Runner status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"


class CameraStatus(str, Enum):
    """Camera status enumeration."""

    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"


# ============================================================================
# Core Data Models
# ============================================================================


class ZoneConfig(BaseModel):
    """Zone configuration matching existing schemas.py:ZoneConfig"""

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
    """Motion gating configuration matching existing schemas.py:MotionGatingConfig"""

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
    max_inference_interval_seconds: float = Field(
        10, ge=0, description="Maximum seconds between inference"
    )
    roi_native: bool = Field(
        True, description="Compute motion entirely inside include zones"
    )
    adaptive_threshold_factor: float = Field(
        0.7, description="Adaptive threshold factor for ROI-native mode"
    )
    min_area_mode: str = Field(
        "px", pattern="^(px|roi_percent)$", description="Minimum area mode"
    )
    min_area_roi_percent: float = Field(
        0.5, description="Minimum area as percentage of ROI"
    )


class CameraConfig(BaseModel):
    """Camera configuration matching existing schemas.py:CameraConfig"""

    camera_uuid: str = Field(..., description="Camera UUID")
    rtsp_url: str = Field(..., description="RTSP stream URL")
    tenant_id: str = Field(..., description="Tenant ID")
    site_id: str = Field(..., description="Site ID")
    zones: List[ZoneConfig] = Field(
        default_factory=list, description="Zone configurations"
    )
    motion_gating: MotionGatingConfig = Field(
        default_factory=MotionGatingConfig, description="Motion gating config"
    )
    allow_labels: Optional[List[str]] = Field(None, description="Global allow labels")
    deny_labels: Optional[List[str]] = Field(None, description="Global deny labels")
    min_score: float = Field(0.30, ge=0, le=1, description="Global min score")
    zone_test: str = Field(
        "center", pattern="^(center|center\\+iou)$", description="Zone test method"
    )
    iou_threshold: float = Field(
        0.10, ge=0, le=1, description="IoU threshold for center+iou test"
    )
    status: CameraStatus = Field(CameraStatus.OFFLINE, description="Camera status")

    @validator("camera_uuid")
    def validate_camera_uuid(cls, v):
        if not v or not v.startswith("cam-"):
            raise ValueError('camera_uuid must start with "cam-"')
        return v


class Lease(BaseModel):
    """Lease model for distributed camera ownership."""

    lease_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique lease identifier"
    )
    camera_uuid: str = Field(..., description="Camera UUID")
    runner_id: str = Field(..., description="Runner ID holding the lease")
    shard_id: str = Field(..., description="Shard ID holding the lease")
    tenant_id: str = Field(..., description="Tenant ID")
    site_id: str = Field(..., description="Site ID")

    # Timing
    acquired_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Lease acquisition timestamp",
    )
    expires_at: datetime = Field(..., description="Lease expiration timestamp")
    renewed_at: Optional[datetime] = Field(None, description="Last renewal timestamp")
    last_heartbeat: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last heartbeat timestamp",
    )

    # State
    status: LeaseStatus = Field(LeaseStatus.PENDING, description="Lease status")
    processing_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Processing statistics"
    )

    # Configuration versioning
    config_version: str = Field(..., description="Configuration version hash")
    zone_version: str = Field(..., description="Zone configuration version hash")

    @validator("expires_at")
    def validate_expiration(cls, v, values):
        if "acquired_at" in values and v <= values["acquired_at"]:
            raise ValueError("expires_at must be after acquired_at")
        return v


class RunnerInfo(BaseModel):
    """Runner information for registration and monitoring."""

    runner_id: str = Field(..., description="Unique runner identifier")
    tenant_id: str = Field(..., description="Tenant ID")
    site_id: str = Field(..., description="Site ID")

    # Registration
    registered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Registration timestamp",
    )
    last_heartbeat: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last heartbeat timestamp",
    )

    # Deployment
    deployment_group: str = Field(..., description="Deployment group")
    deployment_version: str = Field(..., description="Deployment version")
    host: str = Field(..., description="Host address")
    port: int = Field(..., ge=1, le=65535, description="Port number")
    metrics_port: int = Field(9108, ge=1, le=65535, description="Metrics port")

    # Status
    status: RunnerStatus = Field(RunnerStatus.ACTIVE, description="Runner status")
    capacity: int = Field(
        16, ge=1, le=100, description="Maximum cameras this runner can handle"
    )
    current_load: int = Field(0, ge=0, description="Current number of cameras assigned")

    # Capabilities
    capabilities: List[str] = Field(
        default_factory=lambda: ["rtsp", "motion_detection", "zone_filtering"],
        description="Runner capabilities",
    )

    # Metrics
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Runner metrics")


class LeaseEvent(BaseModel):
    """Lease event for audit trail and notifications."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Event ID"
    )
    lease_id: str = Field(..., description="Associated lease ID")
    event_type: str = Field(..., description="Event type")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp",
    )
    details: Dict[str, Any] = Field(default_factory=dict, description="Event details")


# ============================================================================
# Request/Response Models
# ============================================================================


class LeaseAcquireRequest(BaseModel):
    """Request to acquire a lease."""

    camera_uuid: str = Field(..., description="Camera UUID")
    runner_id: str = Field(..., description="Runner ID")
    shard_id: str = Field(..., description="Shard ID")
    tenant_id: str = Field(..., description="Tenant ID")
    site_id: str = Field(..., description="Site ID")
    ttl_seconds: int = Field(60, ge=30, le=300, description="Lease TTL in seconds")
    config_version: str = Field(..., description="Configuration version hash")
    zone_version: str = Field(..., description="Zone configuration version hash")


class LeaseAcquireResponse(BaseModel):
    """Response for lease acquisition."""

    success: bool = Field(..., description="Whether lease was acquired")
    lease: Optional[Lease] = Field(None, description="Lease information if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    retry_after_seconds: Optional[int] = Field(
        None, description="Suggested retry delay"
    )


class LeaseRenewRequest(BaseModel):
    """Request to renew a lease."""

    processing_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Current processing statistics"
    )


class LeaseHeartbeatRequest(BaseModel):
    """Request for lease heartbeat."""

    processing_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Current processing statistics"
    )
    camera_status: Optional[CameraStatus] = Field(
        None, description="Current camera status"
    )


class RunnerRegisterRequest(BaseModel):
    """Request to register a runner."""

    runner_id: str = Field(..., description="Runner ID")
    tenant_id: str = Field(..., description="Tenant ID")
    site_id: str = Field(..., description="Site ID")
    deployment_group: str = Field(..., description="Deployment group")
    deployment_version: str = Field(..., description="Deployment version")
    host: str = Field(..., description="Host address")
    port: int = Field(..., ge=1, le=65535, description="Port number")
    metrics_port: int = Field(9108, ge=1, le=65535, description="Metrics port")
    capacity: int = Field(16, ge=1, le=100, description="Maximum cameras")
    capabilities: List[str] = Field(
        default_factory=lambda: ["rtsp", "motion_detection", "zone_filtering"],
        description="Capabilities",
    )


class RunnerHeartbeatRequest(BaseModel):
    """Request for runner heartbeat."""

    current_load: int = Field(..., ge=0, description="Current number of cameras")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Runner metrics")
    active_leases: List[str] = Field(
        default_factory=list, description="Active lease IDs"
    )


class ConfigVersionResponse(BaseModel):
    """Response for configuration version check."""

    config_version: str = Field(..., description="Current configuration version hash")
    zone_version: str = Field(
        ..., description="Current zone configuration version hash"
    )
    last_updated: datetime = Field(..., description="Last update timestamp")


class CameraAssignmentResponse(BaseModel):
    """Response for camera assignment changes."""

    added_cameras: List[CameraConfig] = Field(
        default_factory=list, description="Newly assigned cameras"
    )
    removed_cameras: List[str] = Field(
        default_factory=list, description="Removed camera UUIDs"
    )
    updated_cameras: List[CameraConfig] = Field(
        default_factory=list, description="Updated camera configs"
    )


# ============================================================================
# API Endpoint Specifications
# ============================================================================


class ControlPlaneAPI:
    """
    Control Plane API specification that the Django REST Framework must implement.

    This class defines the exact contracts for seamless integration with Somba Pipeline.
    """

    # ============================================================================
    # Lease Management Endpoints
    # ============================================================================

    @staticmethod
    def acquire_lease() -> dict:
        """
        Acquire a lease for camera processing.

        POST /api/v1/leases/acquire/

        Request Body: LeaseAcquireRequest
        Response Body: LeaseAcquireResponse
        Status Codes:
            200: Lease acquired successfully
            409: Camera already leased by another runner
            429: Rate limited, retry after suggested delay
            500: Internal server error
        """
        return {
            "method": "POST",
            "path": "/api/v1/leases/acquire/",
            "request": LeaseAcquireRequest,
            "response": LeaseAcquireResponse,
            "auth": "Bearer token required",
            "description": "Acquire exclusive processing rights for a camera",
        }

    @staticmethod
    def renew_lease() -> dict:
        """
        Renew an existing lease.

        PUT /api/v1/leases/{lease_id}/renew/

        Path Parameters:
            lease_id: UUID of the lease to renew

        Request Body: LeaseRenewRequest
        Response Body: Lease
        Status Codes:
            200: Lease renewed successfully
            404: Lease not found
            410: Lease expired
            500: Internal server error
        """
        return {
            "method": "PUT",
            "path": "/api/v1/leases/{lease_id}/renew/",
            "request": LeaseRenewRequest,
            "response": Lease,
            "auth": "Bearer token required",
            "description": "Extend lease expiration time",
        }

    @staticmethod
    def release_lease() -> dict:
        """
        Release a lease.

        DELETE /api/v1/leases/{lease_id}/release/

        Path Parameters:
            lease_id: UUID of the lease to release

        Response Body: {"success": bool}
        Status Codes:
            200: Lease released successfully
            404: Lease not found
            500: Internal server error
        """
        return {
            "method": "DELETE",
            "path": "/api/v1/leases/{lease_id}/release/",
            "response": {"success": bool},
            "auth": "Bearer token required",
            "description": "Voluntarily release a lease",
        }

    @staticmethod
    def lease_heartbeat() -> dict:
        """
        Send heartbeat for a lease.

        POST /api/v1/leases/{lease_id}/heartbeat/

        Path Parameters:
            lease_id: UUID of the lease

        Request Body: LeaseHeartbeatRequest
        Response Body: {"success": bool, "next_heartbeat_due": datetime}
        Status Codes:
            200: Heartbeat recorded
            404: Lease not found
            410: Lease expired
            500: Internal server error
        """
        return {
            "method": "POST",
            "path": "/api/v1/leases/{lease_id}/heartbeat/",
            "request": LeaseHeartbeatRequest,
            "response": {"success": bool, "next_heartbeat_due": datetime},
            "auth": "Bearer token required",
            "description": "Keep lease active and report status",
        }

    @staticmethod
    def get_runner_leases() -> dict:
        """
        Get all active leases for a runner.

        GET /api/v1/leases/runner/{runner_id}/

        Path Parameters:
            runner_id: Runner ID

        Response Body: List[Lease]
        Status Codes:
            200: Success
            404: Runner not found
            500: Internal server error
        """
        return {
            "method": "GET",
            "path": "/api/v1/leases/runner/{runner_id}/",
            "response": List[Lease],
            "auth": "Bearer token required",
            "description": "Get all leases held by a specific runner",
        }

    @staticmethod
    def get_camera_lease() -> dict:
        """
        Get lease information for a camera.

        GET /api/v1/leases/camera/{camera_uuid}/

        Path Parameters:
            camera_uuid: Camera UUID

        Response Body: Optional[Lease]
        Status Codes:
            200: Success
            404: Camera not found
            500: Internal server error
        """
        return {
            "method": "GET",
            "path": "/api/v1/leases/camera/{camera_uuid}/",
            "response": Optional[Lease],
            "auth": "Bearer token required",
            "description": "Get current lease holder for a camera",
        }

    # ============================================================================
    # Runner Management Endpoints
    # ============================================================================

    @staticmethod
    def register_runner() -> dict:
        """
        Register a new runner with the control plane.

        POST /api/v1/runners/register/

        Request Body: RunnerRegisterRequest
        Response Body: RunnerInfo
        Status Codes:
            201: Runner registered successfully
            409: Runner ID already exists
            500: Internal server error
        """
        return {
            "method": "POST",
            "path": "/api/v1/runners/register/",
            "request": RunnerRegisterRequest,
            "response": RunnerInfo,
            "auth": "Bearer token required",
            "description": "Register a new worker runner",
        }

    @staticmethod
    def runner_heartbeat() -> dict:
        """
        Send runner heartbeat.

        PUT /api/v1/runners/{runner_id}/heartbeat/

        Path Parameters:
            runner_id: Runner ID

        Request Body: RunnerHeartbeatRequest
        Response Body: RunnerInfo
        Status Codes:
            200: Heartbeat recorded
            404: Runner not found
            500: Internal server error
        """
        return {
            "method": "PUT",
            "path": "/api/v1/runners/{runner_id}/heartbeat/",
            "request": RunnerHeartbeatRequest,
            "response": RunnerInfo,
            "auth": "Bearer token required",
            "description": "Report runner status and metrics",
        }

    @staticmethod
    def get_runner() -> dict:
        """
        Get runner information.

        GET /api/v1/runners/{runner_id}/

        Path Parameters:
            runner_id: Runner ID

        Response Body: RunnerInfo
        Status Codes:
            200: Success
            404: Runner not found
            500: Internal server error
        """
        return {
            "method": "GET",
            "path": "/api/v1/runners/{runner_id}/",
            "response": RunnerInfo,
            "auth": "Bearer token required",
            "description": "Get runner details and status",
        }

    @staticmethod
    def unregister_runner() -> dict:
        """
        Unregister a runner.

        DELETE /api/v1/runners/{runner_id}/

        Path Parameters:
            runner_id: Runner ID

        Response Body: {"success": bool, "leases_released": int}
        Status Codes:
            200: Runner unregistered successfully
            404: Runner not found
            500: Internal server error
        """
        return {
            "method": "DELETE",
            "path": "/api/v1/runners/{runner_id}/",
            "response": {"success": bool, "leases_released": int},
            "auth": "Bearer token required",
            "description": "Remove runner from system",
        }

    # ============================================================================
    # Camera Configuration Endpoints
    # ============================================================================

    @staticmethod
    def get_cameras() -> dict:
        """
        Get cameras for a tenant/site.

        GET /api/v1/tenants/{tenant_id}/sites/{site_id}/cameras/

        Path Parameters:
            tenant_id: Tenant ID
            site_id: Site ID

        Query Parameters:
            status: Filter by status (optional)
            limit: Maximum number of results (default: 100)
            offset: Offset for pagination (default: 0)

        Response Body: List[CameraConfig]
        Status Codes:
            200: Success
            404: Tenant or site not found
            500: Internal server error
        """
        return {
            "method": "GET",
            "path": "/api/v1/tenants/{tenant_id}/sites/{site_id}/cameras/",
            "response": List[CameraConfig],
            "auth": "Bearer token required",
            "description": "Get all cameras for a site",
        }

    @staticmethod
    def get_camera() -> dict:
        """
        Get a specific camera configuration.

        GET /api/v1/tenants/{tenant_id}/sites/{site_id}/cameras/{camera_uuid}/

        Path Parameters:
            tenant_id: Tenant ID
            site_id: Site ID
            camera_uuid: Camera UUID

        Response Body: CameraConfig
        Status Codes:
            200: Success
            404: Camera not found
            500: Internal server error
        """
        return {
            "method": "GET",
            "path": "/api/v1/tenants/{tenant_id}/sites/{site_id}/cameras/{camera_uuid}/",
            "response": CameraConfig,
            "auth": "Bearer token required",
            "description": "Get specific camera configuration",
        }

    @staticmethod
    def update_camera() -> dict:
        """
        Update camera configuration.

        PUT /api/v1/tenants/{tenant_id}/sites/{site_id}/cameras/{camera_uuid}/

        Path Parameters:
            tenant_id: Tenant ID
            site_id: Site ID
            camera_uuid: Camera UUID

        Request Body: CameraConfig (partial update supported)
        Response Body: CameraConfig
        Status Codes:
            200: Camera updated successfully
            404: Camera not found
            500: Internal server error
        """
        return {
            "method": "PUT",
            "path": "/api/v1/tenants/{tenant_id}/sites/{site_id}/cameras/{camera_uuid}/",
            "request": CameraConfig,
            "response": CameraConfig,
            "auth": "Bearer token required",
            "description": "Update camera configuration",
        }

    # ============================================================================
    # Zone Configuration Endpoints
    # ============================================================================

    @staticmethod
    def get_zones() -> dict:
        """
        Get zone configurations for a site.

        GET /api/v1/tenants/{tenant_id}/sites/{site_id}/zones/

        Path Parameters:
            tenant_id: Tenant ID
            site_id: Site ID

        Response Body: List[ZoneConfig]
        Status Codes:
            200: Success
            404: Tenant or site not found
            500: Internal server error
        """
        return {
            "method": "GET",
            "path": "/api/v1/tenants/{tenant_id}/sites/{site_id}/zones/",
            "response": List[ZoneConfig],
            "auth": "Bearer token required",
            "description": "Get all zone configurations for a site",
        }

    @staticmethod
    def create_zone() -> dict:
        """
        Create a new zone configuration.

        POST /api/v1/tenants/{tenant_id}/sites/{site_id}/zones/

        Path Parameters:
            tenant_id: Tenant ID
            site_id: Site ID

        Request Body: ZoneConfig
        Response Body: ZoneConfig
        Status Codes:
            201: Zone created successfully
            400: Invalid zone configuration
            404: Tenant or site not found
            500: Internal server error
        """
        return {
            "method": "POST",
            "path": "/api/v1/tenants/{tenant_id}/sites/{site_id}/zones/",
            "request": ZoneConfig,
            "response": ZoneConfig,
            "auth": "Bearer token required",
            "description": "Create a new zone configuration",
        }

    @staticmethod
    def update_zone() -> dict:
        """
        Update zone configuration.

        PUT /api/v1/tenants/{tenant_id}/sites/{site_id}/zones/{zone_id}/

        Path Parameters:
            tenant_id: Tenant ID
            site_id: Site ID
            zone_id: Zone ID

        Request Body: ZoneConfig
        Response Body: ZoneConfig
        Status Codes:
            200: Zone updated successfully
            404: Zone not found
            500: Internal server error
        """
        return {
            "method": "PUT",
            "path": "/api/v1/tenants/{tenant_id}/sites/{site_id}/zones/{zone_id}/",
            "request": ZoneConfig,
            "response": ZoneConfig,
            "auth": "Bearer token required",
            "description": "Update zone configuration",
        }

    @staticmethod
    def delete_zone() -> dict:
        """
        Delete zone configuration.

        DELETE /api/v1/tenants/{tenant_id}/sites/{site_id}/zones/{zone_id}/

        Path Parameters:
            tenant_id: Tenant ID
            site_id: Site ID
            zone_id: Zone ID

        Response Body: {"success": bool}
        Status Codes:
            200: Zone deleted successfully
            404: Zone not found
            500: Internal server error
        """
        return {
            "method": "DELETE",
            "path": "/api/v1/tenants/{tenant_id}/sites/{site_id}/zones/{zone_id}/",
            "response": {"success": bool},
            "auth": "Bearer token required",
            "description": "Delete zone configuration",
        }

    # ============================================================================
    # Configuration Version Endpoints
    # ============================================================================

    @staticmethod
    def get_config_version() -> dict:
        """
        Get current configuration version.

        GET /api/v1/tenants/{tenant_id}/sites/{site_id}/config/version

        Path Parameters:
            tenant_id: Tenant ID
            site_id: Site ID

        Response Body: ConfigVersionResponse
        Status Codes:
            200: Success
            404: Tenant or site not found
            500: Internal server error
        """
        return {
            "method": "GET",
            "path": "/api/v1/tenants/{tenant_id}/sites/{site_id}/config/version",
            "response": ConfigVersionResponse,
            "auth": "Bearer token required",
            "description": "Get current configuration version hash",
        }

    # ============================================================================
    # Camera Assignment Endpoints
    # ============================================================================

    @staticmethod
    def get_camera_assignments() -> dict:
        """
        Get camera assignments for a runner.

        GET /api/v1/runners/{runner_id}/assignments

        Path Parameters:
            runner_id: Runner ID

        Response Body: CameraAssignmentResponse
        Status Codes:
            200: Success
            404: Runner not found
            500: Internal server error
        """
        return {
            "method": "GET",
            "path": "/api/v1/runners/{runner_id}/assignments",
            "response": CameraAssignmentResponse,
            "auth": "Bearer token required",
            "description": "Get camera assignment changes for a runner",
        }

    # ============================================================================
    # Event Streaming Endpoints (WebSocket)
    # ============================================================================

    @staticmethod
    def lease_events_stream() -> dict:
        """
        WebSocket endpoint for lease events.

        WebSocket /api/v1/leases/events/stream

        Authentication: Bearer token in query parameter
        Events: LeaseEvent messages
        Description: Real-time lease events for monitoring
        """
        return {
            "method": "WebSocket",
            "path": "/api/v1/leases/events/stream?token={bearer_token}",
            "events": LeaseEvent,
            "auth": "Bearer token required",
            "description": "Real-time lease event notifications",
        }

    @staticmethod
    def runner_events_stream() -> dict:
        """
        WebSocket endpoint for runner events.

        WebSocket /api/v1/runners/events/stream

        Authentication: Bearer token in query parameter
        Events: Runner status events
        Description: Real-time runner events for monitoring
        """
        return {
            "method": "WebSocket",
            "path": "/api/v1/runners/events/stream?token={bearer_token}",
            "auth": "Bearer token required",
            "description": "Real-time runner event notifications",
        }


# ============================================================================
# Error Response Specifications
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp",
    )
    request_id: Optional[str] = Field(None, description="Request ID for tracing")


class ErrorCodes:
    """Standard error codes."""

    # Lease errors
    LEASE_NOT_FOUND = "LEASE_NOT_FOUND"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    LEASE_ALREADY_HELD = "LEASE_ALREADY_HELD"
    LEASE_ACQUISITION_FAILED = "LEASE_ACQUISITION_FAILED"
    LEASE_RENEWAL_FAILED = "LEASE_RENEWAL_FAILED"

    # Runner errors
    RUNNER_NOT_FOUND = "RUNNER_NOT_FOUND"
    RUNNER_ALREADY_REGISTERED = "RUNNER_ALREADY_REGISTERED"
    RUNNER_REGISTRATION_FAILED = "RUNNER_REGISTRATION_FAILED"

    # Camera errors
    CAMERA_NOT_FOUND = "CAMERA_NOT_FOUND"
    CAMERA_ALREADY_LEASED = "CAMERA_ALREADY_LEASED"
    CAMERA_CONFIG_INVALID = "CAMERA_CONFIG_INVALID"

    # Configuration errors
    CONFIG_VERSION_MISMATCH = "CONFIG_VERSION_MISMATCH"
    CONFIG_NOT_FOUND = "CONFIG_NOT_FOUND"

    # Authentication errors
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"

    # Rate limiting
    RATE_LIMITED = "RATE_LIMITED"

    # System errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    DATABASE_ERROR = "DATABASE_ERROR"


# ============================================================================
# Authentication and Authorization
# ============================================================================


class AuthToken(BaseModel):
    """Authentication token specification."""

    token_type: str = Field("Bearer", description="Token type")
    access_token: str = Field(..., description="JWT access token")
    expires_in: int = Field(3600, description="Token expiration in seconds")
    scope: str = Field("runner:api", description="Token scope")


class TokenRequest(BaseModel):
    """Token request specification."""

    grant_type: str = Field("client_credentials", description="Grant type")
    client_id: str = Field(..., description="Client ID")
    client_secret: str = Field(..., description="Client secret")
    scope: Optional[str] = Field("runner:api", description="Requested scope")


# ============================================================================
# Rate Limiting Specifications
# ============================================================================


class RateLimits:
    """Rate limiting specifications."""

    # Lease operations
    LEASE_ACQUIRE_RATE = 10  # requests per minute per runner
    LEASE_RENEWAL_RATE = 60  # requests per minute per runner
    LEASE_HEARTBEAT_RATE = 120  # requests per minute per runner

    # Runner operations
    RUNNER_HEARTBEAT_RATE = 60  # requests per minute per runner
    RUNNER_REGISTRATION_RATE = 5  # requests per minute per IP

    # Configuration operations
    CONFIG_READ_RATE = 300  # requests per minute per tenant
    CONFIG_WRITE_RATE = 60  # requests per minute per tenant


# ============================================================================
# Integration Summary
# ============================================================================


def get_api_specification():
    """
    Get complete API specification for control plane integration.

    Returns a comprehensive specification that the Django REST Framework
    control plane must implement for seamless Somba Pipeline integration.
    """
    return {
        "version": "1.0.0",
        "title": "Somba Pipeline Control Plane API",
        "description": "API specification for Phase 3 lease system and control plane integration",
        "base_url": "https://control-plane.example.com/api/v1",
        "authentication": "Bearer token (JWT)",
        "endpoints": [
            ControlPlaneAPI.acquire_lease(),
            ControlPlaneAPI.renew_lease(),
            ControlPlaneAPI.release_lease(),
            ControlPlaneAPI.lease_heartbeat(),
            ControlPlaneAPI.get_runner_leases(),
            ControlPlaneAPI.get_camera_lease(),
            ControlPlaneAPI.register_runner(),
            ControlPlaneAPI.runner_heartbeat(),
            ControlPlaneAPI.get_runner(),
            ControlPlaneAPI.unregister_runner(),
            ControlPlaneAPI.get_cameras(),
            ControlPlaneAPI.get_camera(),
            ControlPlaneAPI.update_camera(),
            ControlPlaneAPI.get_zones(),
            ControlPlaneAPI.create_zone(),
            ControlPlaneAPI.update_zone(),
            ControlPlaneAPI.delete_zone(),
            ControlPlaneAPI.get_config_version(),
            ControlPlaneAPI.get_camera_assignments(),
            ControlPlaneAPI.lease_events_stream(),
            ControlPlaneAPI.runner_events_stream(),
        ],
        "data_models": [
            "ZoneConfig",
            "MotionGatingConfig",
            "CameraConfig",
            "Lease",
            "RunnerInfo",
            "LeaseEvent",
            "ErrorResponse",
        ],
        "error_codes": ErrorCodes,
        "rate_limits": RateLimits,
        "websockets": [
            "/api/v1/leases/events/stream",
            "/api/v1/runners/events/stream",
        ],
        "health_check": "/health/",
        "metrics": "/metrics/",
    }


if __name__ == "__main__":
    # Print API specification
    spec = get_api_specification()
    print(f"Control Plane API Specification v{spec['version']}")
    print(f"Base URL: {spec['base_url']}")
    print(f"Total Endpoints: {len(spec['endpoints'])}")
    print(f"Authentication: {spec['authentication']}")
    print(f"WebSocket Streams: {len(spec['websockets'])}")
