"""
Mock Control Plane for testing lease operations and camera configurations.
Implements the CP API endpoints specified in manager_worker_technical_specs.md
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from aiohttp import web, ClientSession
from pydantic import BaseModel, Field

from .schemas import CameraConfig, ZoneConfig, MotionGatingConfig

logger = logging.getLogger(__name__)


class Camera(BaseModel):
    """Camera model for Control Plane."""

    camera_uuid: str
    tenant_id: str
    site_id: str
    rtsp_url: str
    enabled: bool = True
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Lease(BaseModel):
    """Camera lease model."""

    camera_uuid: str
    owner_id: str  # runner_id
    expires_at: str
    version: int = 1


class TokenBucket(BaseModel):
    """Site token bucket for rate limiting."""

    site_id: str
    tokens: int
    capacity: int = 120
    refill_rate: int = 120  # tokens per minute
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RunnerState(BaseModel):
    """Runner state model."""

    runner_id: str
    cordoned: bool = False
    drain_mode: str = "none"  # none, soft, hard
    capacity_streams: int = 40


class MockControlPlane:
    """Mock Control Plane implementing the required APIs."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port

        # In-memory storage
        self.cameras: Dict[str, Camera] = {}
        self.leases: Dict[str, Lease] = {}
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.runners: Dict[str, RunnerState] = {}

        # Initialize with test data
        self._init_test_data()

        # Start token refill task
        self.refill_task: Optional[asyncio.Task] = None

        logger.info(f"Mock Control Plane initialized on {host}:{port}")

    def _init_test_data(self):
        """Initialize with test cameras and configurations."""
        # Create test cameras with zone configurations
        test_cameras = [
            {
                "camera_uuid": "cam-001",
                "tenant_id": "tenant-01",
                "site_id": "site-A",
                "rtsp_url": "rtsp://127.0.0.1:8554/camera_001",
                "zones": [
                    {
                        "zone_id": 1,
                        "name": "driveway",
                        "kind": "include",
                        "priority": 100,
                        "polygon": [[100, 120], [1180, 120], [1180, 680], [100, 680]],
                        "allow_labels": ["person", "car"],
                        "deny_labels": ["cat"],
                        "min_score": 0.25,
                    },
                    {
                        "zone_id": 2,
                        "name": "neighbor_lawn",
                        "kind": "exclude",
                        "priority": 200,
                        "polygon": [[1180, 120], [1800, 120], [1800, 680], [1180, 680]],
                        "deny_labels": ["person", "car"],
                    },
                ],
            },
            {
                "camera_uuid": "cam-002",
                "tenant_id": "tenant-01",
                "site_id": "site-A",
                "rtsp_url": "rtsp://127.0.0.1:8554/camera_002",
                "zones": [
                    {
                        "zone_id": 1,
                        "name": "entrance",
                        "kind": "include",
                        "priority": 100,
                        "polygon": [[200, 200], [800, 200], [800, 600], [200, 600]],
                        "allow_labels": ["person"],
                        "min_score": 0.3,
                    }
                ],
            },
            {
                "camera_uuid": "cam-003",
                "tenant_id": "tenant-01",
                "site_id": "site-B",
                "rtsp_url": "rtsp://127.0.0.1:8554/camera_003",
                "zones": [],  # No zones configured
            },
        ]

        for cam_data in test_cameras:
            zones = cam_data.pop("zones", [])
            camera = Camera(**cam_data)
            self.cameras[camera.camera_uuid] = camera

        # Initialize token buckets for sites
        for site_id in ["site-A", "site-B"]:
            self.token_buckets[site_id] = TokenBucket(site_id=site_id, tokens=120)

        # Initialize test runner
        self.runners["runner-test-001"] = RunnerState(runner_id="runner-test-001")

        logger.info(f"Initialized {len(self.cameras)} test cameras")

    def get_camera_config(self, camera_uuid: str) -> Optional[CameraConfig]:
        """Get camera configuration including zones."""
        # In a real system, this would come from database
        # For testing, create configurations based on camera UUID
        if camera_uuid == "cam-001":
            return CameraConfig(
                camera_uuid=camera_uuid,
                zones=[
                    ZoneConfig(
                        zone_id=1,
                        name="driveway",
                        kind="include",
                        priority=100,
                        polygon=[[100, 120], [1180, 120], [1180, 680], [100, 680]],
                        allow_labels=["person", "car"],
                        deny_labels=["cat"],
                        min_score=0.25,
                    ),
                    ZoneConfig(
                        zone_id=2,
                        name="neighbor_lawn",
                        kind="exclude",
                        priority=200,
                        polygon=[[1180, 120], [1800, 120], [1800, 680], [1180, 680]],
                        allow_labels=None,
                        deny_labels=["person", "car"],
                    ),
                ],
                motion_gating=MotionGatingConfig(enabled=True),
                allow_labels=["person", "car", "truck"],
                deny_labels=[],
                min_score=0.30,
                zone_test="center",
                iou_threshold=0.10,
            )
        elif camera_uuid == "cam-002":
            return CameraConfig(
                camera_uuid=camera_uuid,
                zones=[
                    ZoneConfig(
                        zone_id=1,
                        name="entrance",
                        kind="include",
                        priority=100,
                        polygon=[[200, 200], [800, 200], [800, 600], [200, 600]],
                        allow_labels=["person"],
                        min_score=0.3,
                    )
                ],
                motion_gating=MotionGatingConfig(enabled=True),
                zone_test="center+iou",
                iou_threshold=0.15,
            )
        else:
            # Default config for other cameras
            return CameraConfig(
                camera_uuid=camera_uuid, motion_gating=MotionGatingConfig(enabled=False)
            )

    async def _refill_tokens(self):
        """Periodically refill token buckets."""
        while True:
            await asyncio.sleep(60)  # Every minute

            current_time = datetime.now(timezone.utc)
            for bucket in self.token_buckets.values():
                last_update = datetime.fromisoformat(
                    bucket.updated_at.replace("Z", "+00:00")
                )
                minutes_passed = (current_time - last_update).total_seconds() / 60

                if minutes_passed >= 1:
                    new_tokens = min(
                        bucket.capacity, bucket.tokens + bucket.refill_rate
                    )
                    bucket.tokens = new_tokens
                    bucket.updated_at = current_time.isoformat().replace("+00:00", "Z")

                    logger.debug(
                        f"Refilled tokens for {bucket.site_id}: {new_tokens}/{bucket.capacity}"
                    )

    # API Handlers

    async def get_cameras(self, request):
        """GET /v1/cameras - List cameras with filtering."""
        enabled = request.query.get("enabled")
        tenant_id = request.query.get("tenant_id")
        site_id = request.query.get("site_id")
        limit = int(request.query.get("limit", 100))

        cameras = list(self.cameras.values())

        # Apply filters
        if enabled is not None:
            enabled_bool = enabled.lower() == "true"
            cameras = [c for c in cameras if c.enabled == enabled_bool]

        if tenant_id:
            cameras = [c for c in cameras if c.tenant_id == tenant_id]

        if site_id:
            cameras = [c for c in cameras if c.site_id == site_id]

        # Apply limit
        cameras = cameras[:limit]

        return web.json_response(
            {"items": [c.model_dump() for c in cameras], "next_cursor": None}
        )

    async def acquire_lease(self, request):
        """POST /v1/leases/camera/acquire - Acquire camera lease."""
        data = await request.json()
        runner_id = data["runner_id"]
        camera_uuid = data["camera_uuid"]
        ttl_seconds = data["ttl_seconds"]

        # Check if camera exists
        if camera_uuid not in self.cameras:
            return web.json_response({"error": "Camera not found"}, status=404)

        # Check if lease already held by someone else
        if camera_uuid in self.leases:
            existing_lease = self.leases[camera_uuid]
            expires_at = datetime.fromisoformat(
                existing_lease.expires_at.replace("Z", "+00:00")
            )
            if (
                expires_at > datetime.now(timezone.utc)
                and existing_lease.owner_id != runner_id
            ):
                return web.json_response(
                    {"error": "Lease held by another runner"}, status=409
                )

        # Grant lease
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        lease = Lease(
            camera_uuid=camera_uuid,
            owner_id=runner_id,
            expires_at=expires_at.isoformat().replace("+00:00", "Z"),
        )
        self.leases[camera_uuid] = lease

        logger.info(f"Granted lease for {camera_uuid} to {runner_id}")

        return web.json_response(lease.model_dump())

    async def renew_lease(self, request):
        """POST /v1/leases/camera/renew - Renew camera lease."""
        data = await request.json()
        runner_id = data["runner_id"]
        camera_uuid = data["camera_uuid"]
        ttl_seconds = data["ttl_seconds"]

        # Check if runner owns the lease
        if (
            camera_uuid not in self.leases
            or self.leases[camera_uuid].owner_id != runner_id
        ):
            return web.json_response({"error": "Not lease owner"}, status=404)

        # Renew lease
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        self.leases[camera_uuid].expires_at = expires_at.isoformat().replace(
            "+00:00", "Z"
        )
        self.leases[camera_uuid].version += 1

        logger.debug(f"Renewed lease for {camera_uuid} by {runner_id}")

        return web.json_response(self.leases[camera_uuid].model_dump())

    async def release_lease(self, request):
        """POST /v1/leases/camera/release - Release camera lease."""
        data = await request.json()
        runner_id = data["runner_id"]
        camera_uuid = data["camera_uuid"]

        # Check if runner owns the lease
        if (
            camera_uuid in self.leases
            and self.leases[camera_uuid].owner_id == runner_id
        ):
            del self.leases[camera_uuid]
            logger.info(f"Released lease for {camera_uuid} by {runner_id}")

        return web.json_response({"status": "released"})

    async def consume_site_budget(self, request):
        """POST /v1/sites/{site_id}/budget/consume - Consume site token."""
        site_id = request.match_info["site_id"]

        # Get or create token bucket
        if site_id not in self.token_buckets:
            self.token_buckets[site_id] = TokenBucket(site_id=site_id, tokens=120)

        bucket = self.token_buckets[site_id]

        if bucket.tokens > 0:
            bucket.tokens -= 1
            bucket.updated_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            logger.debug(f"Consumed token for {site_id}, remaining: {bucket.tokens}")
            return web.json_response({"tokens_remaining": bucket.tokens})
        else:
            logger.debug(f"No tokens available for {site_id}")
            return web.json_response({"error": "No tokens available"}, status=409)

    async def get_runner(self, request):
        """GET /v1/runners/{runner_id} - Get runner state."""
        runner_id = request.match_info["runner_id"]

        if runner_id not in self.runners:
            self.runners[runner_id] = RunnerState(runner_id=runner_id)

        return web.json_response(self.runners[runner_id].model_dump())

    async def runner_heartbeat(self, request):
        """POST /v1/runners/heartbeat - Receive runner heartbeat."""
        data = await request.json()
        logger.debug(
            f"Heartbeat from {data.get('runner_id')}: {data.get('streams_owned')} streams"
        )
        return web.json_response({"status": "acknowledged"})

    async def stream_status_batch(self, request):
        """POST /v1/streams/status/batch - Batch stream status ingest."""
        data = await request.json()
        logger.debug(f"Received {len(data.get('items', []))} status updates")
        return web.json_response({"status": "accepted"}, status=202)

    async def get_camera_config_endpoint(self, request):
        """GET /v1/cameras/{camera_uuid}/config - Get camera configuration."""
        camera_uuid = request.match_info["camera_uuid"]

        config = self.get_camera_config(camera_uuid)
        if config:
            return web.json_response(config.model_dump())
        else:
            return web.json_response({"error": "Camera not found"}, status=404)

    def create_app(self) -> web.Application:
        """Create aiohttp application with routes."""
        app = web.Application()

        # Camera endpoints
        app.router.add_get("/v1/cameras", self.get_cameras)
        app.router.add_get(
            "/v1/cameras/{camera_uuid}/config", self.get_camera_config_endpoint
        )

        # Lease endpoints
        app.router.add_post("/v1/leases/camera/acquire", self.acquire_lease)
        app.router.add_post("/v1/leases/camera/renew", self.renew_lease)
        app.router.add_post("/v1/leases/camera/release", self.release_lease)

        # Site budget endpoints
        app.router.add_post(
            "/v1/sites/{site_id}/budget/consume", self.consume_site_budget
        )

        # Runner endpoints
        app.router.add_get("/v1/runners/{runner_id}", self.get_runner)
        app.router.add_post("/v1/runners/heartbeat", self.runner_heartbeat)

        # Stream status endpoints
        app.router.add_post("/v1/streams/status/batch", self.stream_status_batch)

        return app

    async def start(self):
        """Start the mock control plane server."""
        app = self.create_app()

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        # Start token refill task
        self.refill_task = asyncio.create_task(self._refill_tokens())

        logger.info(f"Mock Control Plane started on http://{self.host}:{self.port}")

        return runner

    async def stop(self):
        """Stop the mock control plane server."""
        if self.refill_task:
            self.refill_task.cancel()
            try:
                await self.refill_task
            except asyncio.CancelledError:
                pass

        logger.info("Mock Control Plane stopped")


async def main():
    """Main entry point for running the mock control plane."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cp = MockControlPlane()
    runner = await cp.start()

    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await cp.stop()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
