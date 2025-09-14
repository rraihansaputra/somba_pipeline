"""
Lease Manager for Phase 4 - Manages lease acquisition, renewal, and worker assignment.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

from .control_plane_client import ControlPlaneClient
from .control_plane_api import (
    Lease,
    LeaseAcquireRequest,
    LeaseRenewRequest,
    LeaseHeartbeatRequest,
    LeaseStatus,
    CameraStatus,
)

logger = logging.getLogger(__name__)


class LeaseManager:
    """Manages lease acquisition, renewal, and worker assignment."""

    def __init__(self, control_plane_config: Dict[str, str]):
        self.cp_client = ControlPlaneClient(
            control_plane_config.get("base_url", "http://localhost:8000"),
            control_plane_config.get("token", "default-token"),
        )

        # Lease tracking
        self.active_leases: Dict[str, Lease] = {}  # lease_id -> Lease
        self.camera_leases: Dict[str, str] = {}  # camera_uuid -> lease_id
        self.lease_renewal_tasks: Dict[str, asyncio.Task] = {}  # lease_id -> Task

        # Configuration
        self.lease_ttl_seconds = control_plane_config.get("lease_ttl_seconds", 60)
        self.renewal_interval_seconds = control_plane_config.get(
            "renewal_interval_seconds", 30
        )
        self.heartbeat_interval_seconds = control_plane_config.get(
            "heartbeat_interval_seconds", 15
        )

        # State
        self.running = True

        logger.info("LeaseManager initialized")

    async def acquire_camera_lease(
        self,
        camera_uuid: str,
        runner_id: str,
        shard_id: str,
        tenant_id: str,
        site_id: str,
    ) -> Optional[Lease]:
        """Acquire lease for a specific camera."""
        try:
            # Check if we already have a lease for this camera
            if camera_uuid in self.camera_leases:
                existing_lease_id = self.camera_leases[camera_uuid]
                if existing_lease_id in self.active_leases:
                    logger.debug(
                        f"Already have lease for camera {camera_uuid}: {existing_lease_id}"
                    )
                    return self.active_leases[existing_lease_id]

            # Create lease acquisition request
            request = LeaseAcquireRequest(
                camera_uuid=camera_uuid,
                runner_id=runner_id,
                shard_id=shard_id,
                tenant_id=tenant_id,
                site_id=site_id,
                ttl_seconds=self.lease_ttl_seconds,
                config_version="v1",
                zone_version="v1",
            )

            # Try to acquire lease
            response = await self.cp_client.acquire_lease(request)

            if response.success and response.lease:
                lease = response.lease

                # Track the lease
                self.active_leases[lease.lease_id] = lease
                self.camera_leases[camera_uuid] = lease.lease_id

                # Start renewal and heartbeat tasks
                await self._start_lease_maintenance_tasks(lease)

                logger.info(
                    f"Acquired lease for camera {camera_uuid}: {lease.lease_id}"
                )
                return lease
            else:
                logger.warning(
                    f"Failed to acquire lease for camera {camera_uuid}: {response.error}"
                )
                return None

        except Exception as e:
            logger.error(f"Error acquiring lease for camera {camera_uuid}: {e}")
            return None

    async def renew_lease(self, lease_id: str, stats: Dict[str, Any]) -> bool:
        """Renew an existing lease."""
        try:
            if lease_id not in self.active_leases:
                logger.warning(f"Cannot renew unknown lease: {lease_id}")
                return False

            request = LeaseRenewRequest(processing_stats=stats)
            renewed_lease = await self.cp_client.renew_lease(lease_id, request)

            if renewed_lease:
                # Update lease in tracking
                self.active_leases[lease_id] = renewed_lease
                logger.debug(f"Renewed lease: {lease_id}")
                return True
            else:
                logger.warning(f"Failed to renew lease: {lease_id}")
                return False

        except Exception as e:
            logger.error(f"Error renewing lease {lease_id}: {e}")
            return False

    async def release_lease(self, lease_id: str) -> bool:
        """Release a lease."""
        try:
            if lease_id not in self.active_leases:
                logger.warning(f"Cannot release unknown lease: {lease_id}")
                return False

            lease = self.active_leases[lease_id]

            # Stop maintenance tasks
            await self._stop_lease_maintenance_tasks(lease_id)

            # Release lease from control plane
            success = await self.cp_client.release_lease(lease_id)

            if success:
                # Remove from tracking
                del self.active_leases[lease_id]
                if lease.camera_uuid in self.camera_leases:
                    del self.camera_leases[lease.camera_uuid]

                logger.info(f"Released lease: {lease_id}")
                return True
            else:
                logger.warning(f"Failed to release lease: {lease_id}")
                return False

        except Exception as e:
            logger.error(f"Error releasing lease {lease_id}: {e}")
            return False

    async def send_heartbeat(
        self,
        lease_id: str,
        stats: Dict[str, Any],
        camera_status: Optional[CameraStatus] = None,
    ) -> bool:
        """Send heartbeat for a lease."""
        try:
            if lease_id not in self.active_leases:
                logger.warning(f"Cannot send heartbeat for unknown lease: {lease_id}")
                return False

            request = LeaseHeartbeatRequest(
                processing_stats=stats, camera_status=camera_status
            )

            success = await self.cp_client.lease_heartbeat(lease_id, request)

            if success:
                # Update last heartbeat time
                if lease_id in self.active_leases:
                    self.active_leases[lease_id].last_heartbeat = datetime.now(
                        timezone.utc
                    )

            return success

        except Exception as e:
            logger.error(f"Error sending heartbeat for lease {lease_id}: {e}")
            return False

    async def get_camera_lease(self, camera_uuid: str) -> Optional[Lease]:
        """Get lease for a specific camera."""
        lease_id = self.camera_leases.get(camera_uuid)
        if lease_id and lease_id in self.active_leases:
            return self.active_leases[lease_id]
        return None

    async def get_active_leases(self, runner_id: str) -> List[Lease]:
        """Get all active leases for a runner."""
        return [
            lease
            for lease in self.active_leases.values()
            if lease.runner_id == runner_id
        ]

    async def validate_lease(self, lease_id: str) -> bool:
        """Check if a lease is still valid."""
        if lease_id not in self.active_leases:
            return False

        lease = self.active_leases[lease_id]
        current_time = datetime.now(timezone.utc)

        # Check if lease is expired
        if lease.expires_at <= current_time:
            logger.warning(f"Lease {lease_id} has expired")
            await self._handle_lease_expiration(lease_id)
            return False

        # Check if lease is still active
        if lease.status != LeaseStatus.ACTIVE:
            logger.warning(f"Lease {lease_id} is not active: {lease.status}")
            return False

        return True

    async def _start_lease_maintenance_tasks(self, lease: Lease):
        """Start background tasks for lease maintenance."""
        # Start renewal task
        renewal_task = asyncio.create_task(self._lease_renewal_loop(lease.lease_id))
        self.lease_renewal_tasks[lease.lease_id] = renewal_task

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._lease_heartbeat_loop(lease.lease_id))
        self.lease_renewal_tasks[f"{lease.lease_id}_heartbeat"] = heartbeat_task

    async def _stop_lease_maintenance_tasks(self, lease_id: str):
        """Stop background tasks for a lease."""
        # Stop renewal task
        if lease_id in self.lease_renewal_tasks:
            task = self.lease_renewal_tasks[lease_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.lease_renewal_tasks[lease_id]

        # Stop heartbeat task
        heartbeat_key = f"{lease_id}_heartbeat"
        if heartbeat_key in self.lease_renewal_tasks:
            task = self.lease_renewal_tasks[heartbeat_key]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.lease_renewal_tasks[heartbeat_key]

    async def _lease_renewal_loop(self, lease_id: str):
        """Background task to renew lease before expiration."""
        while self.running and lease_id in self.active_leases:
            try:
                lease = self.active_leases[lease_id]
                current_time = datetime.now(timezone.utc)

                # Calculate time until renewal (renew at 50% of TTL)
                time_to_renewal = (
                    lease.expires_at - lease.acquired_at
                ).total_seconds() * 0.5
                renewal_time = lease.acquired_at + timedelta(seconds=time_to_renewal)

                # Sleep until renewal time
                sleep_duration = (renewal_time - current_time).total_seconds()
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)

                # Renew lease
                stats = {
                    "processed_frames": 100,
                    "active_cameras": len(self.active_leases),
                }
                success = await self.renew_lease(lease_id, stats)

                if not success:
                    logger.error(
                        f"Failed to renew lease {lease_id}, stopping maintenance"
                    )
                    await self._handle_lease_expiration(lease_id)
                    break

            except asyncio.CancelledError:
                logger.debug(f"Lease renewal task cancelled for {lease_id}")
                break
            except Exception as e:
                logger.error(f"Error in lease renewal loop for {lease_id}: {e}")
                await asyncio.sleep(10)  # Brief delay before retry

    async def _lease_heartbeat_loop(self, lease_id: str):
        """Background task to send heartbeats for lease."""
        while self.running and lease_id in self.active_leases:
            try:
                # Send heartbeat
                stats = {
                    "processed_frames": 100,
                    "active_cameras": len(self.active_leases),
                }
                await self.send_heartbeat(lease_id, stats)

                # Sleep until next heartbeat
                await asyncio.sleep(self.heartbeat_interval_seconds)

            except asyncio.CancelledError:
                logger.debug(f"Lease heartbeat task cancelled for {lease_id}")
                break
            except Exception as e:
                logger.error(f"Error in lease heartbeat loop for {lease_id}: {e}")
                await asyncio.sleep(10)  # Brief delay before retry

    async def _handle_lease_expiration(self, lease_id: str):
        """Handle lease expiration."""
        try:
            if lease_id in self.active_leases:
                lease = self.active_leases[lease_id]
                camera_uuid = lease.camera_uuid

                logger.warning(f"Lease expired for camera {camera_uuid}: {lease_id}")

                # Stop maintenance tasks
                await self._stop_lease_maintenance_tasks(lease_id)

                # Remove from tracking
                del self.active_leases[lease_id]
                if camera_uuid in self.camera_leases:
                    del self.camera_leases[camera_uuid]

        except Exception as e:
            logger.error(f"Error handling lease expiration for {lease_id}: {e}")

    async def cleanup_expired_leases(self):
        """Clean up expired leases."""
        current_time = datetime.now(timezone.utc)
        expired_leases = []

        for lease_id, lease in self.active_leases.items():
            if lease.expires_at <= current_time:
                expired_leases.append(lease_id)

        for lease_id in expired_leases:
            await self._handle_lease_expiration(lease_id)

        if expired_leases:
            logger.info(f"Cleaned up {len(expired_leases)} expired leases")

    async def shutdown(self):
        """Shutdown lease manager gracefully."""
        logger.info("Shutting down LeaseManager...")
        self.running = False

        # Stop all maintenance tasks
        for lease_id in list(self.lease_renewal_tasks.keys()):
            await self._stop_lease_maintenance_tasks(lease_id)

        # Release all leases
        for lease_id in list(self.active_leases.keys()):
            await self.release_lease(lease_id)

        logger.info("LeaseManager shutdown complete")

    def get_lease_stats(self) -> Dict[str, Any]:
        """Get statistics about lease management."""
        return {
            "total_leases": len(self.active_leases),
            "total_cameras": len(self.camera_leases),
            "active_renewal_tasks": len(
                [
                    k
                    for k in self.lease_renewal_tasks.keys()
                    if not k.endswith("_heartbeat")
                ]
            ),
            "active_heartbeat_tasks": len(
                [k for k in self.lease_renewal_tasks.keys() if k.endswith("_heartbeat")]
            ),
            "leases_by_status": {
                status: len(
                    [l for l in self.active_leases.values() if l.status == status]
                )
                for status in LeaseStatus
            },
        }
