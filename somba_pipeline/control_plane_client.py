"""
Control Plane Client for Phase 4 - Client for communicating with Django REST control plane.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any

import aiohttp

from .control_plane_api import (
    LeaseAcquireRequest,
    LeaseAcquireResponse,
    LeaseRenewRequest,
    LeaseHeartbeatRequest,
    RunnerRegisterRequest,
    RunnerInfo,
    ConfigVersionResponse,
    CameraConfig,
)

logger = logging.getLogger(__name__)


class ControlPlaneClient:
    """Client for communicating with Django REST control plane."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.http_client: Optional[aiohttp.ClientSession] = None
        self.request_timeout = aiohttp.ClientTimeout(total=30)

        logger.info(f"ControlPlaneClient initialized: {self.base_url}")

    async def __aenter__(self):
        """Async context manager entry."""
        self.http_client = aiohttp.ClientSession(
            timeout=self.request_timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.http_client:
            await self.http_client.close()

    async def _make_request(
        self, method: str, endpoint: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request to control plane."""
        if not self.http_client:
            self.http_client = aiohttp.ClientSession(
                timeout=self.request_timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

        url = f"{self.base_url}{endpoint}"

        try:
            async with self.http_client.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    logger.warning(f"Resource not found: {url}")
                    return None
                elif response.status == 401:
                    logger.error(f"Unauthorized request to: {url}")
                    return None
                elif response.status == 429:
                    logger.warning(f"Rate limited for request: {url}")
                    return None
                else:
                    logger.error(
                        f"HTTP error {response.status} for {url}: {await response.text()}"
                    )
                    return None

        except asyncio.TimeoutError:
            logger.error(f"Request timeout for {url}")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}")
            return None

    # Lease Management APIs
    async def acquire_lease(self, request: LeaseAcquireRequest) -> LeaseAcquireResponse:
        """Acquire a lease for camera processing."""
        try:
            response_data = await self._make_request(
                "POST", "/api/v1/leases/acquire/", json=request.dict()
            )

            if response_data:
                return LeaseAcquireResponse(**response_data)
            else:
                return LeaseAcquireResponse(
                    success=False, error="Failed to acquire lease", lease=None
                )

        except Exception as e:
            logger.error(f"Error acquiring lease: {e}")
            return LeaseAcquireResponse(
                success=False, error=f"Exception: {str(e)}", lease=None
            )

    async def renew_lease(
        self, lease_id: str, request: LeaseRenewRequest
    ) -> Optional[Any]:
        """Renew an existing lease."""
        return await self._make_request(
            "PUT", f"/api/v1/leases/{lease_id}/renew/", json=request.dict()
        )

    async def release_lease(self, lease_id: str) -> bool:
        """Release a lease."""
        response = await self._make_request(
            "DELETE", f"/api/v1/leases/{lease_id}/release/"
        )
        return response is not None

    async def lease_heartbeat(
        self, lease_id: str, request: LeaseHeartbeatRequest
    ) -> bool:
        """Send heartbeat for a lease."""
        response = await self._make_request(
            "POST", f"/api/v1/leases/{lease_id}/heartbeat/", json=request.dict()
        )
        return response is not None

    async def get_runner_leases(self, runner_id: str) -> List[Dict[str, Any]]:
        """Get all active leases for a runner."""
        response = await self._make_request(
            "GET", f"/api/v1/leases/runner/{runner_id}/"
        )
        return response.get("leases", []) if response else []

    async def get_camera_lease(self, camera_uuid: str) -> Optional[Dict[str, Any]]:
        """Get lease information for a camera."""
        return await self._make_request("GET", f"/api/v1/leases/camera/{camera_uuid}/")

    # Runner Management APIs
    async def register_runner(self, request: RunnerRegisterRequest) -> RunnerInfo:
        """Register a new runner with the control plane."""
        try:
            response_data = await self._make_request(
                "POST", "/api/v1/runners/register/", json=request.dict()
            )

            if response_data:
                return RunnerInfo(**response_data)
            else:
                raise Exception("Failed to register runner")

        except Exception as e:
            logger.error(f"Error registering runner: {e}")
            raise

    async def send_runner_heartbeat(
        self, runner_id: str, current_load: int, metrics: Dict[str, Any]
    ) -> bool:
        """Send runner heartbeat."""
        request_data = {"current_load": current_load, "metrics": metrics}

        response = await self._make_request(
            "PUT", f"/api/v1/runners/{runner_id}/heartbeat/", json=request_data
        )
        return response is not None

    async def get_runner(self, runner_id: str) -> Optional[RunnerInfo]:
        """Get runner information."""
        response_data = await self._make_request("GET", f"/api/v1/runners/{runner_id}/")
        if response_data:
            return RunnerInfo(**response_data)
        return None

    async def unregister_runner(self, runner_id: str) -> bool:
        """Unregister a runner."""
        response = await self._make_request("DELETE", f"/api/v1/runners/{runner_id}/")
        return response is not None

    # Camera Configuration APIs
    async def get_available_cameras(self) -> List[str]:
        """Get list of available cameras for this runner."""
        response = await self._make_request("GET", "/api/v1/cameras/available/")
        return response.get("cameras", []) if response else []

    async def get_camera_config(self, camera_uuid: str) -> Optional[CameraConfig]:
        """Get camera configuration."""
        response_data = await self._make_request(
            "GET", f"/api/v1/cameras/{camera_uuid}/"
        )

        if response_data:
            try:
                return CameraConfig(**response_data)
            except Exception as e:
                logger.error(f"Error parsing camera config for {camera_uuid}: {e}")
                return None
        return None

    async def get_cameras(
        self, tenant_id: str, site_id: str, status: Optional[str] = None
    ) -> List[CameraConfig]:
        """Get cameras for a tenant/site."""
        params = {}
        if status:
            params["status"] = status

        response_data = await self._make_request(
            "GET",
            f"/api/v1/tenants/{tenant_id}/sites/{site_id}/cameras/",
            params=params,
        )

        if response_data:
            try:
                cameras = []
                for camera_data in response_data.get("cameras", []):
                    cameras.append(CameraConfig(**camera_data))
                return cameras
            except Exception as e:
                logger.error(f"Error parsing camera configs: {e}")
                return []
        return []

    async def update_camera_config(
        self, camera_uuid: str, config: CameraConfig
    ) -> Optional[CameraConfig]:
        """Update camera configuration."""
        response_data = await self._make_request(
            "PUT", f"/api/v1/cameras/{camera_uuid}/", json=config.dict()
        )

        if response_data:
            try:
                return CameraConfig(**response_data)
            except Exception as e:
                logger.error(f"Error parsing updated camera config: {e}")
                return None
        return None

    # Zone Configuration APIs
    async def get_zone_configs(
        self, tenant_id: str, site_id: str
    ) -> List[Dict[str, Any]]:
        """Get zone configurations for a site."""
        response = await self._make_request(
            "GET", f"/api/v1/tenants/{tenant_id}/sites/{site_id}/zones/"
        )
        return response.get("zones", []) if response else []

    async def create_zone_config(
        self, tenant_id: str, site_id: str, zone_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create a new zone configuration."""
        return await self._make_request(
            "POST",
            f"/api/v1/tenants/{tenant_id}/sites/{site_id}/zones/",
            json=zone_config,
        )

    async def update_zone_config(
        self, tenant_id: str, site_id: str, zone_id: int, zone_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update zone configuration."""
        return await self._make_request(
            "PUT",
            f"/api/v1/tenants/{tenant_id}/sites/{site_id}/zones/{zone_id}/",
            json=zone_config,
        )

    async def delete_zone_config(
        self, tenant_id: str, site_id: str, zone_id: int
    ) -> bool:
        """Delete zone configuration."""
        response = await self._make_request(
            "DELETE", f"/api/v1/tenants/{tenant_id}/sites/{site_id}/zones/{zone_id}/"
        )
        return response is not None

    # Configuration Version APIs
    async def get_config_version(self) -> Optional[ConfigVersionResponse]:
        """Get current configuration version."""
        response_data = await self._make_request("GET", "/api/v1/config/version")
        if response_data:
            try:
                return ConfigVersionResponse(**response_data)
            except Exception as e:
                logger.error(f"Error parsing config version: {e}")
                return None
        return None

    # Camera Assignment APIs
    async def get_camera_assignments(self, runner_id: str) -> Dict[str, Any]:
        """Get camera assignments for a runner."""
        response = await self._make_request(
            "GET", f"/api/v1/runners/{runner_id}/assignments"
        )
        return response or {}

    # Health Check APIs
    async def health_check(self) -> bool:
        """Check control plane health."""
        response = await self._make_request("GET", "/health/")
        return response is not None

    async def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get control plane metrics."""
        return await self._make_request("GET", "/metrics/")

    # Utility Methods
    async def test_connection(self) -> bool:
        """Test connection to control plane."""
        try:
            return await self.health_check()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def get_runner_status(self, runner_id: str) -> Optional[str]:
        """Get runner status."""
        runner_info = await self.get_runner(runner_id)
        return runner_info.status if runner_info else None

    async def get_camera_status(self, camera_uuid: str) -> Optional[str]:
        """Get camera status."""
        lease_info = await self.get_camera_lease(camera_uuid)
        if lease_info:
            return lease_info.get("status")
        return None

    async def get_runner_capacity(self, runner_id: str) -> Optional[int]:
        """Get runner capacity."""
        runner_info = await self.get_runner(runner_id)
        return runner_info.capacity if runner_info else None

    async def get_runner_load(self, runner_id: str) -> Optional[int]:
        """Get runner current load."""
        runner_info = await self.get_runner(runner_id)
        return runner_info.current_load if runner_info else None

    # Batch Operations
    async def batch_get_camera_configs(
        self, camera_uuids: List[str]
    ) -> Dict[str, Optional[CameraConfig]]:
        """Get configurations for multiple cameras."""
        results = {}

        # Create tasks for concurrent requests
        tasks = []
        for camera_uuid in camera_uuids:
            task = asyncio.create_task(self.get_camera_config(camera_uuid))
            tasks.append((camera_uuid, task))

        # Wait for all tasks to complete
        for camera_uuid, task in tasks:
            try:
                results[camera_uuid] = await task
            except Exception as e:
                logger.error(f"Error getting config for camera {camera_uuid}: {e}")
                results[camera_uuid] = None

        return results

    async def batch_acquire_leases(
        self, requests: List[LeaseAcquireRequest]
    ) -> List[LeaseAcquireResponse]:
        """Acquire leases for multiple cameras."""
        results = []

        # Create tasks for concurrent requests
        tasks = []
        for request in requests:
            task = asyncio.create_task(self.acquire_lease(request))
            tasks.append(task)

        # Wait for all tasks to complete
        for task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error(f"Error acquiring lease: {e}")
                results.append(
                    LeaseAcquireResponse(
                        success=False, error=f"Exception: {str(e)}", lease=None
                    )
                )

        return results

    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.close()
            self.http_client = None
