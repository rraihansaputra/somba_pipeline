# Phase 3: Lease System & Control Plane - Detailed Specification

## Overview

Phase 3 implements the distributed coordination layer for the Somba Pipeline, enabling horizontal scaling and exactly-once processing guarantees through a lease-based system integrated with an external Django REST Framework control plane.

## Current Integration Points Analysis

### Existing Worker Architecture
- **ProductionWorker** (`somba_pipeline/worker.py:62`): Main worker process
- **ShardConfig** (`somba_pipeline/schemas.py:208`): Configuration schema with `cp` section for control plane
- **Configuration Management**: Hot-reload capability already implemented
- **Metrics**: Prometheus metrics on port 9108, HTTP API on port 8080
- **Event Publishing**: RabbitMQ integration for detection/status events

### Current Control Plane References
```python
# In ShardConfig.schemas.py:216
cp: Dict[str, str] = Field(..., description="Control plane configuration")
```

## Phase 3 Architecture

### Components to Implement

1. **Lease Client** - Distributed ownership mechanism
2. **Control Plane API Client** - Django REST Framework integration
3. **Manager Process** - Worker lifecycle management
4. **PostgreSQL Integration** - Lease persistence and state management

---

## 1. Lease System Specification

### 1.1 Lease Data Model

```python
class Lease(BaseModel):
    """Lease for distributed camera processing ownership."""
    
    lease_id: str = Field(..., description="Unique lease identifier")
    camera_uuid: str = Field(..., description="Camera UUID")
    runner_id: str = Field(..., description="Runner ID holding the lease")
    shard_id: str = Field(..., description="Shard ID holding the lease")
    tenant_id: str = Field(..., description="Tenant ID")
    site_id: str = Field(..., description="Site ID")
    
    # Timing
    acquired_at: datetime = Field(..., description="Lease acquisition timestamp")
    expires_at: datetime = Field(..., description="Lease expiration timestamp")
    renewed_at: Optional[datetime] = Field(None, description="Last renewal timestamp")
    
    # State
    status: LeaseStatus = Field(..., description="Lease status")
    last_heartbeat: datetime = Field(..., description="Last heartbeat timestamp")
    processing_stats: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics")
    
    # Configuration versioning
    config_version: str = Field(..., description="Configuration version hash")
    zone_version: str = Field(..., description="Zone configuration version hash")

class LeaseStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    RELEASED = "released"
    REVOKED = "revoked"
    PENDING = "pending"
```

### 1.2 Lease Manager Interface

```python
class LeaseManager:
    """Manages lease acquisition, renewal, and release."""
    
    async def acquire_lease(
        self, 
        camera_uuid: str, 
        runner_id: str, 
        shard_id: str,
        ttl_seconds: int = 60
    ) -> Optional[Lease]:
        """Acquire a lease for camera processing."""
        pass
    
    async def renew_lease(self, lease_id: str) -> bool:
        """Renew an existing lease."""
        pass
    
    async def release_lease(self, lease_id: str) -> bool:
        """Release a lease."""
        pass
    
    async def heartbeat(self, lease_id: str, stats: Dict[str, Any]) -> bool:
        """Send heartbeat for lease."""
        pass
    
    async def validate_lease(self, lease_id: str) -> bool:
        """Validate if lease is still active."""
        pass
    
    async def get_active_leases(self, runner_id: str) -> List[Lease]:
        """Get all active leases for a runner."""
        pass
    
    async def watch_lease_events(self, callback: Callable[[LeaseEvent], None]):
        """Watch for lease events (revocation, expiration)."""
        pass
```

### 1.3 Lease Events

```python
class LeaseEventType(str, Enum):
    LEASE_ACQUIRED = "lease_acquired"
    LEASE_RENEWED = "lease_renewed"
    LEASE_EXPIRED = "lease_expired"
    LEASE_REVOKED = "lease_revoked"
    LEASE_RELEASED = "lease_released"
    LEASE_STOLEN = "lease_stolen"  # When another runner acquires

class LeaseEvent(BaseModel):
    """Lease state change event."""
    
    event_type: LeaseEventType
    lease_id: str
    camera_uuid: str
    runner_id: str
    shard_id: str
    timestamp: datetime
    details: Dict[str, Any] = Field(default_factory=dict)
```

---

## 2. Control Plane API Integration

### 2.1 Django REST API Endpoints

The control plane (external repository) must provide these endpoints:

#### Configuration Management
```
GET    /api/v1/tenants/{tenant_id}/sites/{site_id}/cameras/
POST   /api/v1/tenants/{tenant_id}/sites/{site_id}/cameras/
PUT    /api/v1/tenants/{tenant_id}/sites/{site_id}/cameras/{camera_uuid}/
DELETE /api/v1/tenants/{tenant_id}/sites/{site_id}/cameras/{camera_uuid}/

GET    /api/v1/tenants/{tenant_id}/sites/{site_id}/zones/
POST   /api/v1/tenants/{tenant_id}/sites/{site_id}/zones/
PUT    /api/v1/tenants/{tenant_id}/sites/{site_id}/zones/{zone_id}/
DELETE /api/v1/tenants/{tenant_id}/sites/{site_id}/zones/{zone_id}/

GET    /api/v1/tenants/{tenant_id}/sites/{site_id}/config/version
```

#### Lease Management
```
POST   /api/v1/leases/acquire/
PUT    /api/v1/leases/{lease_id}/renew/
DELETE /api/v1/leases/{lease_id}/release/
POST   /api/v1/leases/{lease_id}/heartbeat/
GET    /api/v1/leases/runner/{runner_id}/
GET    /api/v1/leases/camera/{camera_uuid}/
```

#### Runner Management
```
POST   /api/v1/runners/register/
PUT    /api/v1/runners/{runner_id}/heartbeat/
GET    /api/v1/runners/{runner_id}/
DELETE /api/v1/runners/{runner_id}/
```

### 2.2 Control Plane Client

```python
class ControlPlaneClient:
    """Client for Django REST API integration."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.http_client = aiohttp.ClientSession()
    
    async def register_runner(self, runner_config: RunnerConfig) -> RunnerInfo:
        """Register this runner with control plane."""
        pass
    
    async def get_camera_config(self, camera_uuid: str) -> CameraConfig:
        """Get camera configuration from control plane."""
        pass
    
    async def get_zone_configs(self, site_id: str) -> List[ZoneConfig]:
        """Get zone configurations for a site."""
        pass
    
    async def report_runner_status(self, runner_id: str, status: RunnerStatus):
        """Report runner status to control plane."""
        pass
    
    async def notify_camera_assigned(self, lease: Lease):
        """Notify that camera was assigned to this runner."""
        pass
    
    async def notify_camera_released(self, lease: Lease):
        """Notify that camera was released from this runner."""
        pass
```

---

## 3. Manager Process

### 3.1 Manager Process Architecture

```python
class ManagerProcess:
    """Manages worker lifecycle and lease coordination."""
    
    def __init__(self, config: ManagerConfig):
        self.config = config
        self.lease_manager = LeaseManager(config.control_plane)
        self.cp_client = ControlPlaneClient(config.control_plane_url, config.api_key)
        self.workers: Dict[str, WorkerProcess] = {}
        self.leases: Dict[str, Lease] = {}
        self.running = True
        
        # Configuration
        self.max_workers_per_manager = config.max_workers or 4
        self.max_cameras_per_worker = config.max_cameras_per_worker or 16
        self.lease_ttl_seconds = config.lease_ttl_seconds or 60
        self.heartbeat_interval = config.heartbeat_interval or 30
    
    async def start(self):
        """Start the manager process."""
        # Register with control plane
        await self._register_runner()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._lease_acquisition_loop()),
            asyncio.create_task(self._lease_renewal_loop()),
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._worker_monitoring_loop()),
            asyncio.create_task(self._configuration_sync_loop()),
        ]
        
        await asyncio.gather(*tasks)
    
    async def _lease_acquisition_loop(self):
        """Continuously try to acquire leases for available cameras."""
        while self.running:
            try:
                # Get available cameras from control plane
                available_cameras = await self._get_available_cameras()
                
                # Filter cameras we can handle
                target_cameras = self._select_target_cameras(available_cameras)
                
                # Try to acquire leases
                for camera_uuid in target_cameras:
                    await self._try_acquire_camera(camera_uuid)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in lease acquisition: {e}")
                await asyncio.sleep(30)
    
    async def _lease_renewal_loop(self):
        """Renew active leases before expiration."""
        while self.running:
            try:
                for lease_id, lease in list(self.leases.items()):
                    if lease.expires_at - datetime.now(timezone.utc) < timedelta(seconds=30):
                        success = await self.lease_manager.renew_lease(lease_id)
                        if not success:
                            logger.warning(f"Failed to renew lease {lease_id}")
                            await self._handle_lease_loss(lease)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in lease renewal: {e}")
                await asyncio.sleep(30)
    
    async def _worker_monitoring_loop(self):
        """Monitor worker health and restart failed workers."""
        while self.running:
            try:
                for worker_id, worker in list(self.workers.items()):
                    if not worker.is_healthy():
                        logger.warning(f"Worker {worker_id} unhealthy, restarting...")
                        await self._restart_worker(worker_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in worker monitoring: {e}")
                await asyncio.sleep(60)
```

### 3.2 Worker Process Management

```python
class WorkerProcess:
    """Manages a single worker subprocess."""
    
    def __init__(self, worker_config: WorkerConfig):
        self.config = worker_config
        self.process: Optional[asyncio.subprocess.Process] = None
        self.start_time: Optional[datetime] = None
        self.last_heartbeat: Optional[datetime] = None
        
    async def start(self) -> bool:
        """Start the worker subprocess."""
        cmd = [
            "python", "-m", "somba_pipeline.worker",
            "--config", self.config.config_path,
            "--runner-id", self.config.runner_id,
            "--shard-id", self.config.shard_id,
            "--manager-pid", str(os.getpid()),
        ]
        
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.start_time = datetime.now(timezone.utc)
        
        return True
    
    def is_healthy(self) -> bool:
        """Check if worker is healthy."""
        if not self.process or self.process.returncode is not None:
            return False
        
        # Check heartbeat timeout
        if self.last_heartbeat:
            timeout = timedelta(seconds=60)  # 60 second heartbeat timeout
            return datetime.now(timezone.utc) - self.last_heartbeat < timeout
        
        # Check start time (give 30 seconds grace period)
        if self.start_time:
            grace_period = timedelta(seconds=30)
            return datetime.now(timezone.utc) - self.start_time < grace_period
        
        return False
    
    async def stop(self, grace_period: int = 30):
        """Stop the worker gracefully."""
        if self.process:
            try:
                self.process.send_signal(signal.SIGTERM)
                await asyncio.wait_for(self.process.wait(), timeout=grace_period)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
```

---

## 4. Configuration Integration

### 4.1 Enhanced ShardConfig

```python
class EnhancedShardConfig(ShardConfig):
    """Extended configuration for Phase 3."""
    
    # Control plane configuration
    control_plane_url: str = Field(..., description="Control plane API URL")
    api_key: str = Field(..., description="API key for control plane")
    runner_id: str = Field(..., description="Unique runner identifier")
    
    # Manager configuration
    manager_mode: bool = Field(True, description="Run in manager mode")
    max_workers: int = Field(4, description="Maximum worker processes")
    max_cameras_per_worker: int = Field(16, description="Maximum cameras per worker")
    
    # Lease configuration
    lease_ttl_seconds: int = Field(60, description="Lease TTL in seconds")
    heartbeat_interval_seconds: int = Field(30, description="Heartbeat interval")
    lease_retry_interval_seconds: int = Field(10, description="Lease retry interval")
    
    # Database configuration (PostgreSQL)
    database_url: str = Field(..., description="PostgreSQL connection URL")
    
    # Blue/green deployment
    deployment_group: str = Field(..., description="Deployment group")
    deployment_version: str = Field(..., description="Deployment version")
```

### 4.2 Configuration Hot Reload Integration

The existing hot-reload system in `worker.py:221` must be extended to handle:

1. **Control Plane Configuration Changes**: API URL, API key changes
2. **Lease Configuration Changes**: TTL, retry intervals
3. **Camera Assignment Changes**: When leases are acquired/lost
4. **Zone Configuration Changes**: Synchronize with control plane

```python
async def _handle_control_plane_config_change(self, new_config: EnhancedShardConfig):
    """Handle control plane configuration changes."""
    
    # Reinitialize control plane client
    if self.control_plane_url != new_config.control_plane_url:
        self.cp_client = ControlPlaneClient(
            new_config.control_plane_url, 
            new_config.api_key
        )
    
    # Update lease manager configuration
    if self.lease_ttl != new_config.lease_ttl_seconds:
        await self.lease_manager.update_ttl(new_config.lease_ttl_seconds)
    
    # Handle camera assignment changes
    await self._sync_camera_assignments(new_config)
```

---

## 5. Database Schema (PostgreSQL)

### 5.1 Lease Table

```sql
CREATE TABLE leases (
    lease_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_uuid VARCHAR(255) NOT NULL,
    runner_id VARCHAR(255) NOT NULL,
    shard_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    site_id VARCHAR(255) NOT NULL,
    
    -- Timing
    acquired_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    renewed_at TIMESTAMP WITH TIME ZONE,
    last_heartbeat TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- State
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    processing_stats JSONB DEFAULT '{}',
    
    -- Configuration versioning
    config_version VARCHAR(64) NOT NULL,
    zone_version VARCHAR(64) NOT NULL,
    
    -- Constraints
    UNIQUE(camera_uuid, status) WHERE status = 'active',
    
    -- Indexes
    INDEX idx_leases_camera_uuid (camera_uuid),
    INDEX idx_leases_runner_id (runner_id),
    INDEX idx_leases_status (status),
    INDEX idx_leases_expires_at (expires_at)
);

-- Lease events table for audit trail
CREATE TABLE lease_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lease_id UUID NOT NULL REFERENCES leases(lease_id),
    event_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    details JSONB DEFAULT '{}',
    
    INDEX idx_lease_events_lease_id (lease_id),
    INDEX idx_lease_events_timestamp (timestamp)
);
```

### 5.2 Runner Table

```sql
CREATE TABLE runners (
    runner_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    site_id VARCHAR(255) NOT NULL,
    
    -- Registration
    registered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_heartbeat TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Deployment
    deployment_group VARCHAR(255) NOT NULL,
    deployment_version VARCHAR(255) NOT NULL,
    host VARCHAR(255) NOT NULL,
    port INTEGER NOT NULL,
    
    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    capacity INTEGER NOT NULL DEFAULT 16,
    current_load INTEGER NOT NULL DEFAULT 0,
    
    -- Metrics
    metrics JSONB DEFAULT '{}',
    
    INDEX idx_runners_tenant_site (tenant_id, site_id),
    INDEX idx_runners_status (status),
    INDEX idx_runners_deployment (deployment_group, deployment_version)
);
```

---

## 6. API Contract with Control Plane

### 6.1 Camera Configuration Sync

The control plane must provide camera configurations that match the existing `CameraConfig` schema:

```python
# Expected from control plane GET /api/v1/cameras/{camera_uuid}/
{
    "camera_uuid": "cam-001",
    "rtsp_url": "rtsp://camera.local/stream",
    "tenant_id": "tenant-1",
    "site_id": "site-1",
    "zones": [
        {
            "zone_id": 1,
            "name": "driveway",
            "kind": "include",
            "priority": 100,
            "polygon": [[100, 100], [700, 100], [700, 500], [200, 500]],
            "allow_labels": ["person", "car"],
            "min_score": 0.25
        }
    ],
    "motion_gating": {
        "enabled": true,
        "downscale": 0.5,
        "min_area_px": 1500,
        "cooldown_frames": 2
    },
    "allow_labels": ["person", "car", "truck"],
    "min_score": 0.30,
    "zone_test": "center",
    "iou_threshold": 0.10
}
```

### 6.2 Lease API Contract

```python
# POST /api/v1/leases/acquire/
{
    "camera_uuid": "cam-001",
    "runner_id": "runner-abc123",
    "shard_id": "shard-1",
    "tenant_id": "tenant-1",
    "site_id": "site-1",
    "ttl_seconds": 60,
    "config_version": "abc123",
    "zone_version": "def456"
}

# Response
{
    "lease_id": "lease-uuid-123",
    "camera_uuid": "cam-001",
    "runner_id": "runner-abc123",
    "shard_id": "shard-1",
    "acquired_at": "2023-12-01T10:00:00Z",
    "expires_at": "2023-12-01T10:01:00Z",
    "status": "active"
}
```

### 6.3 Runner Registration

```python
# POST /api/v1/runners/register/
{
    "runner_id": "runner-abc123",
    "tenant_id": "tenant-1",
    "site_id": "site-1",
    "deployment_group": "production",
    "deployment_version": "v1.2.3",
    "host": "worker-01.example.com",
    "port": 8080,
    "capacity": 16,
    "metrics_port": 9108,
    "capabilities": ["rtsp", "motion_detection", "zone_filtering"]
}

# Response
{
    "runner_id": "runner-abc123",
    "registered_at": "2023-12-01T10:00:00Z",
    "status": "active",
    "assigned_cameras": []
}
```

---

## 7. Implementation Roadmap

### 7.1 Priority 1: Core Lease System
1. **LeaseManager Implementation**: Basic lease acquisition/renewal
2. **PostgreSQL Schema**: Database tables and indexes
3. **Control Plane Client**: Basic API client with authentication
4. **Manager Process**: Worker lifecycle management

### 7.2 Priority 2: Integration Layer
1. **Enhanced Worker**: Lease-aware worker with camera assignment
2. **Configuration Sync**: Hot-reload integration with control plane
3. **Event Publishing**: Lease events to RabbitMQ
4. **Health Monitoring**: Worker health checks and restart

### 7.3 Priority 3: Production Features
1. **Blue/Green Deployment**: Zero-downtime upgrades
2. **Metrics Integration**: Enhanced Prometheus metrics
3. **Graceful Degradation**: Handle control plane outages
4. **Security**: API key management, TLS

---

## 8. Integration Points Summary

### 8.1 Existing Code Integration

1. **worker.py**: Add lease awareness to `ProductionWorker`
2. **schemas.py**: Extend `ShardConfig` with control plane fields
3. **zone_attribution.py**: Integrate with control plane zone configs
4. **motion_detection.py**: Lease-based camera enable/disable

### 8.2 New Components

1. **lease_manager.py**: Lease acquisition and management
2. **control_plane_client.py**: Django REST API client
3. **manager.py**: Manager process for worker orchestration
4. **database.py**: PostgreSQL connection and models

### 8.3 Configuration Changes

1. **Environment Variables**: Control plane URL, API key, database URL
2. **Configuration Schema**: Add control plane and lease configuration
3. **Hot Reload**: Extend existing system for control plane changes

---

## 9. Success Criteria

### 9.1 Functional Requirements
- [ ] Workers can acquire leases for cameras
- [ ] Only one worker processes each camera at a time
- [ ] Leases are automatically renewed before expiration
- [ ] Workers gracefully handle lease loss
- [ ] Configuration changes from control plane are applied
- [ ] Failed workers are automatically restarted
- [ ] Manager process can handle multiple workers

### 9.2 Non-Functional Requirements
- [ ] System tolerates control plane outages (5-minute grace period)
- [ ] Lease acquisition completes within 2 seconds
- [ ] Worker restart completes within 30 seconds
- [ ] Database queries complete within 100ms
- [ ] API calls to control plane complete within 1 second
- [ ] No data loss during lease transitions

### 9.3 Observability
- [ ] Lease events published to RabbitMQ
- [ ] Prometheus metrics for lease operations
- [ ] Health endpoints for manager and workers
- [ ] Comprehensive logging for debugging
- [ ] Configuration version tracking

---

## 10. Risk Mitigation

### 10.1 Control Plane Unavailability
- **Graceful Degradation**: Continue processing with existing leases
- **Local Cache**: Cache configurations locally
- **Retry Logic**: Exponential backoff for API calls

### 10.2 Database Issues
- **Connection Pooling**: Handle connection drops gracefully
- **Fallback Mode**: Continue with in-memory state if DB unavailable
- **Data Consistency**: Use transactions for lease operations

### 10.3 Network Partitions
- **Lease TTL**: Ensure leases expire appropriately
- **Heartbeat Monitoring**: Detect network issues quickly
- **Split-Brain Handling**: Prefer availability over consistency

This specification provides the detailed requirements for implementing Phase 3 of the Somba Pipeline, ensuring clean integration with the external Django REST Framework control plane while maintaining the existing functionality and architecture patterns.