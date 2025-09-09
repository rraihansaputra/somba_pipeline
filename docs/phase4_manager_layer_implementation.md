# Phase 4: Manager Layer & Lease-Based Worker Management

## Overview
Phase 4 focuses on implementing the manager layer that uses the lease system from Phase 3 to manage worker processes and synchronize configuration between the control plane and workers.

## Current Architecture Analysis

### Existing Components
- **ProductionWorker** (`somba_pipeline/worker.py:62`): Main worker process with camera processing
- **ShardConfig** (`somba_pipeline/schemas.py:208`): Configuration schema with control plane section
- **Control Plane API** (`somba_pipeline/control_plane_api.py`): Django REST API specifications

### Current Gap
No manager layer exists to coordinate multiple workers and manage lease-based camera assignments.

## Implementation Plan

### 1. Manager Layer Architecture

#### 1.1 Manager Process
```python
# somba_pipeline/manager.py
class ManagerProcess:
    """Manager process that coordinates workers using lease system."""
    
    def __init__(self, config: ManagerConfig):
        self.config = config
        self.runner_id = config.runner_id
        self.shard_id = config.shard_id
        
        # Lease management
        self.lease_manager = LeaseManager(config.control_plane)
        self.active_leases: Dict[str, Lease] = {}  # lease_id -> Lease
        
        # Worker management
        self.workers: Dict[str, WorkerProcess] = {}  # worker_id -> WorkerProcess
        self.worker_assignments: Dict[str, List[str]] = {}  # worker_id -> [camera_uuids]
        
        # Configuration synchronization
        self.config_sync = ConfigurationSync(config.control_plane)
        self.local_configs: Dict[str, CameraConfig] = {}  # camera_uuid -> CameraConfig
        
        # Control plane client
        self.cp_client = ControlPlaneClient(
            config.control_plane_url, 
            config.api_key
        )
        
        # State
        self.running = True
        self.max_workers = config.max_workers or 4
        self.max_cameras_per_worker = config.max_cameras_per_worker or 16
```

#### 1.2 Manager Configuration Schema
```python
# somba_pipeline/schemas.py (extension)
class ManagerConfig(BaseModel):
    """Manager process configuration."""
    
    # Identification
    runner_id: str = Field(..., description="Unique runner identifier")
    shard_id: str = Field(..., description="Shard identifier")
    
    # Control plane
    control_plane_url: str = Field(..., description="Control plane API URL")
    api_key: str = Field(..., description="API key for control plane")
    
    # Worker management
    max_workers: int = Field(4, description="Maximum worker processes")
    max_cameras_per_worker: int = Field(16, description="Maximum cameras per worker")
    
    # Lease configuration
    lease_ttl_seconds: int = Field(60, description="Lease TTL in seconds")
    heartbeat_interval_seconds: int = Field(30, description="Heartbeat interval")
    
    # Worker configuration
    worker_config_path: str = Field(..., description="Path to worker config template")
    
    # Monitoring
    health_check_interval: int = Field(30, description="Health check interval")
    metrics_port: int = Field(9109, description="Manager metrics port")
```

### 2. Lease-Based Worker Management

#### 2.1 Lease Manager
```python
# somba_pipeline/lease_manager.py
class LeaseManager:
    """Manages lease acquisition, renewal, and worker assignment."""
    
    def __init__(self, control_plane_config: Dict[str, str]):
        self.cp_client = ControlPlaneClient(
            control_plane_config.get("base_url"),
            control_plane_config.get("token")
        )
        
    async def acquire_camera_lease(self, camera_uuid: str, runner_id: str, shard_id: str) -> Optional[Lease]:
        """Acquire lease for a specific camera."""
        request = LeaseAcquireRequest(
            camera_uuid=camera_uuid,
            runner_id=runner_id,
            shard_id=shard_id,
            tenant_id="tenant-01",  # From config
            site_id="site-A",       # From config
            ttl_seconds=60,
            config_version="v1",
            zone_version="v1"
        )
        
        response = await self.cp_client.acquire_lease(request)
        if response.success:
            return response.lease
        return None
        
    async def renew_lease(self, lease_id: str, stats: Dict[str, Any]) -> bool:
        """Renew an existing lease."""
        request = LeaseRenewRequest(processing_stats=stats)
        return await self.cp_client.renew_lease(lease_id, request)
        
    async def release_lease(self, lease_id: str) -> bool:
        """Release a lease."""
        return await self.cp_client.release_lease(lease_id)
```

#### 2.2 Worker Process Management
```python
# somba_pipeline/worker_process.py
class WorkerProcess:
    """Manages a single worker subprocess."""
    
    def __init__(self, worker_id: str, config: WorkerConfig):
        self.worker_id = worker_id
        self.config = config
        self.process: Optional[asyncio.subprocess.Process] = None
        self.start_time: Optional[datetime] = None
        self.assigned_cameras: List[str] = []
        self.healthy = False
        
    async def start(self, cameras: List[str]) -> bool:
        """Start worker process with assigned cameras."""
        self.assigned_cameras = cameras
        
        # Create worker-specific config
        worker_config = self._create_worker_config(cameras)
        
        # Start subprocess
        cmd = [
            "python", "-m", "somba_pipeline.worker",
            "--config", worker_config,
            "--worker-id", self.worker_id,
            "--cameras", ",".join(cameras)
        ]
        
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        self.start_time = datetime.now(timezone.utc)
        return True
        
    def is_healthy(self) -> bool:
        """Check if worker process is healthy."""
        if not self.process or self.process.returncode is not None:
            return False
        
        # Check if process is responsive
        return self._check_worker_health()
        
    async def stop(self, timeout: int = 30):
        """Stop worker process gracefully."""
        if self.process:
            try:
                self.process.send_signal(signal.SIGTERM)
                await asyncio.wait_for(self.process.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
                
    def _create_worker_config(self, cameras: List[str]) -> str:
        """Create worker-specific configuration file."""
        # Generate config with only assigned cameras
        worker_config = {
            "runner_id": self.config.runner_id,
            "shard_id": self.worker_id,
            "max_fps": self.config.max_fps,
            "sources": [
                {"camera_uuid": cam, "url": f"rtsp://localhost/{cam}"}
                for cam in cameras
            ],
            "amqp": self.config.amqp,
            "cp": self.config.cp,
            "cameras": {cam: self.config.cameras[cam] for cam in cameras}
        }
        
        config_path = f"/tmp/worker_{self.worker_id}_config.json"
        with open(config_path, 'w') as f:
            json.dump(worker_config, f)
            
        return config_path
```

### 3. Configuration Synchronization

#### 3.1 Configuration Sync Manager
```python
# somba_pipeline/config_sync.py
class ConfigurationSync:
    """Manages configuration synchronization between control plane and workers."""
    
    def __init__(self, control_plane_config: Dict[str, str]):
        self.cp_client = ControlPlaneClient(
            control_plane_config.get("base_url"),
            control_plane_config.get("token")
        )
        self.local_configs: Dict[str, CameraConfig] = {}
        self.config_versions: Dict[str, str] = {}  # camera_uuid -> version_hash
        
    async def sync_camera_configurations(self, camera_uuids: List[str]) -> Dict[str, CameraConfig]:
        """Sync configurations for multiple cameras."""
        updated_configs = {}
        
        for camera_uuid in camera_uuids:
            # Get current version from control plane
            remote_config = await self.cp_client.get_camera_config(camera_uuid)
            
            if remote_config:
                # Check if configuration has changed
                current_version = self.config_versions.get(camera_uuid)
                remote_version = self._calculate_config_hash(remote_config)
                
                if current_version != remote_version:
                    # Configuration changed, update local cache
                    self.local_configs[camera_uuid] = remote_config
                    self.config_versions[camera_uuid] = remote_version
                    updated_configs[camera_uuid] = remote_config
                    
                    logger.info(f"Configuration updated for camera {camera_uuid}")
                else:
                    # Use cached configuration
                    if camera_uuid in self.local_configs:
                        updated_configs[camera_uuid] = self.local_configs[camera_uuid]
            else:
                logger.warning(f"Failed to get config for camera {camera_uuid}")
                
        return updated_configs
        
    async def watch_configuration_changes(self, callback: Callable[[str, CameraConfig], None]):
        """Watch for configuration changes and notify workers."""
        while self.running:
            try:
                # Check for configuration version changes
                version_response = await self.cp_client.get_config_version()
                
                if version_response:
                    for camera_uuid in self.local_configs:
                        new_config = await self.cp_client.get_camera_config(camera_uuid)
                        if new_config and self._has_config_changed(camera_uuid, new_config):
                            # Notify callback of configuration change
                            await callback(camera_uuid, new_config)
                            
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error watching configuration changes: {e}")
                await asyncio.sleep(60)
                
    def _calculate_config_hash(self, config: CameraConfig) -> str:
        """Calculate hash of configuration for versioning."""
        config_str = json.dumps(config.dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
        
    def _has_config_changed(self, camera_uuid: str, new_config: CameraConfig) -> bool:
        """Check if configuration has changed."""
        current_hash = self.config_versions.get(camera_uuid)
        new_hash = self._calculate_config_hash(new_config)
        return current_hash != new_hash
```

### 4. Manager Main Loop

#### 4.1 Manager Coordination Logic
```python
# somba_pipeline/manager.py (continued)
class ManagerProcess:
    # ... previous methods ...
    
    async def start(self):
        """Start the manager process."""
        logger.info(f"Starting manager process: {self.runner_id}")
        
        # Register with control plane
        await self._register_runner()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._lease_management_loop()),
            asyncio.create_task(self._worker_management_loop()),
            asyncio.create_task(self._configuration_sync_loop()),
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._metrics_server()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down manager...")
            await self._shutdown()
            
    async def _lease_management_loop(self):
        """Main loop for lease acquisition and management."""
        while self.running:
            try:
                # Get available cameras from control plane
                available_cameras = await self._get_available_cameras()
                
                # Calculate current capacity
                current_cameras = sum(len(cams) for cams in self.worker_assignments.values())
                available_capacity = (self.max_workers * self.max_cameras_per_worker) - current_cameras
                
                if available_capacity > 0 and available_cameras:
                    # Try to acquire leases for available cameras
                    await self._acquire_camera_leases(available_cameras[:available_capacity])
                
                # Renew existing leases
                await self._renew_active_leases()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in lease management: {e}")
                await asyncio.sleep(30)
                
    async def _worker_management_loop(self):
        """Manage worker processes based on lease assignments."""
        while self.running:
            try:
                # Group cameras by worker assignments
                camera_assignments = self._calculate_worker_assignments()
                
                # Start/stop workers based on assignments
                await self._adjust_workers(camera_assignments)
                
                # Monitor worker health
                await self._monitor_worker_health()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in worker management: {e}")
                await asyncio.sleep(60)
                
    async def _configuration_sync_loop(self):
        """Synchronize configurations between control plane and workers."""
        while self.running:
            try:
                # Get all assigned cameras
                assigned_cameras = list(self.local_configs.keys())
                
                if assigned_cameras:
                    # Sync configurations
                    updated_configs = await self.config_sync.sync_camera_configurations(assigned_cameras)
                    
                    # Update workers with new configurations
                    for camera_uuid, config in updated_configs.items():
                        await self._update_worker_configuration(camera_uuid, config)
                        
                await asyncio.sleep(60)  # Sync every minute
                
            except Exception as e:
                logger.error(f"Error in configuration sync: {e}")
                await asyncio.sleep(120)
```

### 5. Integration Points

#### 5.1 Control Plane Client Integration
```python
# somba_pipeline/control_plane_client.py
class ControlPlaneClient:
    """Client for communicating with Django REST control plane."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.http_client = aiohttp.ClientSession()
        
    async def get_available_cameras(self) -> List[str]:
        """Get list of available cameras for this runner."""
        async with self.http_client.get(
            f"{self.base_url}/api/v1/cameras/available",
            headers={"Authorization": f"Bearer {self.api_key}"}
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("cameras", [])
            return []
            
    async def get_camera_config(self, camera_uuid: str) -> Optional[CameraConfig]:
        """Get camera configuration."""
        async with self.http_client.get(
            f"{self.base_url}/api/v1/cameras/{camera_uuid}",
            headers={"Authorization": f"Bearer {self.api_key}"}
        ) as response:
            if response.status == 200:
                data = await response.json()
                return CameraConfig(**data)
            return None
```

#### 5.2 Enhanced Worker Integration
```python
# somba_pipeline/worker.py (enhanced)
class ProductionWorker:
    # ... existing code ...
    
    def __init__(self, config: ShardConfig, config_path: Optional[str] = None):
        # ... existing initialization ...
        
        # Add lease awareness
        self.lease_id: Optional[str] = None
        self.lease_manager = None  # Will be set by manager
        self.config_sync = None    # Will be set by manager
        
    async def handle_configuration_update(self, new_config: CameraConfig):
        """Handle configuration updates from manager."""
        logger.info(f"Updating configuration for camera {new_config.camera_uuid}")
        
        # Update zone attribution
        self.zone_attributor.update_camera_config(new_config.camera_uuid, new_config)
        
        # Update motion detection
        if new_config.camera_uuid in self.motion_detectors:
            self.motion_detectors[new_config.camera_uuid].update_zones(new_config.zones)
            
        # Update metrics
        self.zones_config_hash.set(
            new_config.camera_uuid, 
            self._calculate_config_hash(new_config)
        )
```

### 6. Implementation Phases

#### Phase 4.1: Core Manager Layer (Weeks 1-2)
1. **ManagerProcess**: Basic manager structure
2. **LeaseManager**: Lease acquisition and renewal
3. **WorkerProcess**: Worker subprocess management
4. **ControlPlaneClient**: Basic API client

#### Phase 4.2: Configuration Synchronization (Weeks 3-4)
1. **ConfigurationSync**: Config sync between control plane and workers
2. **Enhanced Worker**: Add configuration update handling
3. **Manager Loops**: Lease management and worker coordination

#### Phase 4.3: Advanced Features (Weeks 5-6)
1. **Health Monitoring**: Worker health checks and restart
2. **Metrics Collection**: Manager-specific metrics
3. **Graceful Shutdown**: Proper cleanup and lease release

## Success Criteria

### Functional Requirements
- [ ] Manager can acquire and manage leases for multiple cameras
- [ ] Workers are dynamically started/stopped based on lease assignments
- [ ] Configuration changes from control plane are synchronized to workers
- [ ] Failed workers are automatically restarted
- [ ] Leases are properly released on shutdown

### Performance Requirements
- [ ] Lease acquisition completes within 2 seconds
- [ ] Worker startup completes within 10 seconds
- [ ] Configuration sync completes within 5 seconds
- [ ] Manager can handle up to 64 cameras across 4 workers

### Reliability Requirements
- [ ] No data loss during worker restarts
- [ ] Graceful degradation when control plane is unavailable
- [ ] Proper cleanup of resources on shutdown
- [ ] Health monitoring with automatic recovery