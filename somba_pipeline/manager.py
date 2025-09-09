"""
Manager Process for Phase 4 - Coordinates workers using lease system.
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import os
import uuid

from .schemas import ManagerConfig, CameraConfig, ShardConfig
from .lease_manager import LeaseManager
from .worker_process import WorkerProcess
from .config_sync import ConfigurationSync
from .control_plane_client import ControlPlaneClient
from .control_plane_api import Lease, LeaseStatus, RunnerStatus

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for worker processes."""

    runner_id: str
    shard_id: str
    max_fps: int
    amqp: Dict[str, str]
    cp: Dict[str, str]
    cameras: Dict[str, CameraConfig]


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
        self.worker_assignments: Dict[
            str, List[str]
        ] = {}  # worker_id -> [camera_uuids]

        # Configuration synchronization
        self.config_sync = ConfigurationSync(config.control_plane)
        self.local_configs: Dict[str, CameraConfig] = {}  # camera_uuid -> CameraConfig

        # Control plane client
        self.cp_client = ControlPlaneClient(config.control_plane_url, config.api_key)

        # State
        self.running = True
        self.max_workers = config.max_workers or 4
        self.max_cameras_per_worker = config.max_cameras_per_worker or 16

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        logger.info(
            f"Manager initialized: runner={self.runner_id}, shard={self.shard_id}"
        )

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
        ]

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down manager...")
            await self._shutdown()

    async def _register_runner(self):
        """Register this runner with the control plane."""
        try:
            from .control_plane_api import RunnerRegisterRequest, RunnerInfo

            request = RunnerRegisterRequest(
                runner_id=self.runner_id,
                tenant_id="tenant-01",  # From config
                site_id="site-A",  # From config
                deployment_group="production",
                deployment_version="v1.0.0",
                host="localhost",
                port=8080,
                metrics_port=9108,
                capacity=self.max_workers * self.max_cameras_per_worker,
                capabilities=["rtsp", "motion_detection", "zone_filtering"],
            )

            runner_info = await self.cp_client.register_runner(request)
            logger.info(f"Registered runner: {runner_info.runner_id}")

        except Exception as e:
            logger.error(f"Failed to register runner: {e}")

    async def _lease_management_loop(self):
        """Main loop for lease acquisition and management."""
        while self.running:
            try:
                # Get available cameras from control plane
                available_cameras = await self._get_available_cameras()

                # Calculate current capacity
                current_cameras = sum(
                    len(cams) for cams in self.worker_assignments.values()
                )
                available_capacity = (
                    self.max_workers * self.max_cameras_per_worker
                ) - current_cameras

                if available_capacity > 0 and available_cameras:
                    # Try to acquire leases for available cameras
                    await self._acquire_camera_leases(
                        available_cameras[:available_capacity]
                    )

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
                    updated_configs = await self.config_sync.sync_camera_configurations(
                        assigned_cameras
                    )

                    # Update workers with new configurations
                    for camera_uuid, config in updated_configs.items():
                        await self._update_worker_configuration(camera_uuid, config)

                await asyncio.sleep(60)  # Sync every minute

            except Exception as e:
                logger.error(f"Error in configuration sync: {e}")
                await asyncio.sleep(120)

    async def _health_monitoring_loop(self):
        """Monitor health of all components."""
        while self.running:
            try:
                # Send runner heartbeat
                await self._send_runner_heartbeat()

                # Check lease health
                await self._check_lease_health()

                await asyncio.sleep(self.config.heartbeat_interval_seconds)

            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)

    async def _get_available_cameras(self) -> List[str]:
        """Get list of available cameras from control plane."""
        try:
            return await self.cp_client.get_available_cameras()
        except Exception as e:
            logger.error(f"Failed to get available cameras: {e}")
            return []

    async def _acquire_camera_leases(self, camera_uuids: List[str]):
        """Acquire leases for multiple cameras."""
        for camera_uuid in camera_uuids:
            try:
                lease = await self.lease_manager.acquire_camera_lease(
                    camera_uuid, self.runner_id, self.shard_id
                )

                if lease:
                    self.active_leases[lease.lease_id] = lease
                    logger.info(
                        f"Acquired lease for camera {camera_uuid}: {lease.lease_id}"
                    )

                    # Get camera configuration
                    camera_config = await self.cp_client.get_camera_config(camera_uuid)
                    if camera_config:
                        self.local_configs[camera_uuid] = camera_config

                else:
                    logger.warning(f"Failed to acquire lease for camera {camera_uuid}")

            except Exception as e:
                logger.error(f"Error acquiring lease for camera {camera_uuid}: {e}")

    async def _renew_active_leases(self):
        """Renew existing leases before expiration."""
        current_time = datetime.now(timezone.utc)
        leases_to_renew = []

        for lease_id, lease in self.active_leases.items():
            # Renew if lease expires within 30 seconds
            if lease.expires_at - current_time < timedelta(seconds=30):
                leases_to_renew.append(lease_id)

        for lease_id in leases_to_renew:
            try:
                stats = {
                    "processed_frames": 100,
                    "active_cameras": len(self.active_leases),
                }
                success = await self.lease_manager.renew_lease(lease_id, stats)

                if not success:
                    logger.warning(f"Failed to renew lease {lease_id}")
                    await self._handle_lease_loss(lease_id)
                else:
                    logger.debug(f"Renewed lease {lease_id}")

            except Exception as e:
                logger.error(f"Error renewing lease {lease_id}: {e}")

    async def _calculate_worker_assignments(self) -> Dict[str, List[str]]:
        """Calculate optimal worker assignments based on active leases."""
        assignments = {}

        # Get all cameras with active leases
        camera_uuids = [lease.camera_uuid for lease in self.active_leases.values()]

        if not camera_uuids:
            return assignments

        # Simple round-robin assignment
        worker_id = 0
        for i, camera_uuid in enumerate(camera_uuids):
            worker_key = f"worker-{worker_id}"

            if worker_key not in assignments:
                assignments[worker_key] = []

            assignments[worker_key].append(camera_uuid)

            # Move to next worker if current is full
            if len(assignments[worker_key]) >= self.max_cameras_per_worker:
                worker_id = (worker_id + 1) % self.max_workers

        return assignments

    async def _adjust_workers(self, desired_assignments: Dict[str, List[str]]):
        """Start/stop workers to match desired assignments."""
        # Start workers for new assignments
        for worker_id, cameras in desired_assignments.items():
            if worker_id not in self.workers:
                await self._start_worker(worker_id, cameras)

        # Stop workers that are no longer needed
        current_workers = list(self.workers.keys())
        for worker_id in current_workers:
            if worker_id not in desired_assignments:
                await self._stop_worker(worker_id)

    async def _start_worker(self, worker_id: str, cameras: List[str]):
        """Start a new worker process."""
        try:
            # Create worker config
            worker_config = WorkerConfig(
                runner_id=self.runner_id,
                shard_id=worker_id,
                max_fps=15,
                amqp=self.config.control_plane.get("amqp", {}),
                cp=self.config.control_plane,
                cameras={cam: self.local_configs[cam] for cam in cameras},
            )

            # Create and start worker process
            worker = WorkerProcess(worker_id, worker_config)
            success = await worker.start(cameras)

            if success:
                self.workers[worker_id] = worker
                self.worker_assignments[worker_id] = cameras
                logger.info(f"Started worker {worker_id} with cameras: {cameras}")
            else:
                logger.error(f"Failed to start worker {worker_id}")

        except Exception as e:
            logger.error(f"Error starting worker {worker_id}: {e}")

    async def _stop_worker(self, worker_id: str):
        """Stop a worker process."""
        try:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                await worker.stop()

                # Clean up assignments
                if worker_id in self.worker_assignments:
                    del self.worker_assignments[worker_id]

                del self.workers[worker_id]
                logger.info(f"Stopped worker {worker_id}")

        except Exception as e:
            logger.error(f"Error stopping worker {worker_id}: {e}")

    async def _monitor_worker_health(self):
        """Monitor health of worker processes."""
        for worker_id, worker in list(self.workers.items()):
            if not worker.is_healthy():
                logger.warning(f"Worker {worker_id} is unhealthy, restarting...")
                await self._restart_worker(worker_id)

    async def _restart_worker(self, worker_id: str):
        """Restart a worker process."""
        try:
            # Get current assignments
            cameras = self.worker_assignments.get(worker_id, [])

            # Stop current worker
            await self._stop_worker(worker_id)

            # Start new worker
            if cameras:
                await self._start_worker(worker_id, cameras)

        except Exception as e:
            logger.error(f"Error restarting worker {worker_id}: {e}")

    async def _update_worker_configuration(
        self, camera_uuid: str, new_config: CameraConfig
    ):
        """Update worker configuration for a specific camera."""
        try:
            # Update local cache
            self.local_configs[camera_uuid] = new_config

            # Find which worker handles this camera
            for worker_id, cameras in self.worker_assignments.items():
                if camera_uuid in cameras:
                    worker = self.workers.get(worker_id)
                    if worker:
                        # Restart worker with new configuration
                        await self._restart_worker(worker_id)
                    break

        except Exception as e:
            logger.error(f"Error updating configuration for camera {camera_uuid}: {e}")

    async def _send_runner_heartbeat(self):
        """Send heartbeat to control plane."""
        try:
            current_load = sum(len(cams) for cams in self.worker_assignments.values())
            metrics = {
                "total_workers": len(self.workers),
                "active_leases": len(self.active_leases),
                "processed_cameras": current_load,
            }

            await self.cp_client.send_runner_heartbeat(
                self.runner_id, current_load, metrics
            )

        except Exception as e:
            logger.error(f"Error sending runner heartbeat: {e}")

    async def _check_lease_health(self):
        """Check health of active leases."""
        current_time = datetime.now(timezone.utc)
        expired_leases = []

        for lease_id, lease in self.active_leases.items():
            if lease.expires_at <= current_time:
                expired_leases.append(lease_id)

        for lease_id in expired_leases:
            logger.warning(f"Lease {lease_id} expired")
            await self._handle_lease_loss(lease_id)

    async def _handle_lease_loss(self, lease_id: str):
        """Handle loss of a lease."""
        try:
            if lease_id in self.active_leases:
                lease = self.active_leases[lease_id]
                camera_uuid = lease.camera_uuid

                # Remove from active leases
                del self.active_leases[lease_id]

                # Remove from local configs
                if camera_uuid in self.local_configs:
                    del self.local_configs[camera_uuid]

                # Restart affected worker
                for worker_id, cameras in self.worker_assignments.items():
                    if camera_uuid in cameras:
                        await self._restart_worker(worker_id)
                        break

                logger.info(f"Handled lease loss for camera {camera_uuid}")

        except Exception as e:
            logger.error(f"Error handling lease loss for {lease_id}: {e}")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    async def _shutdown(self):
        """Graceful shutdown of manager and all workers."""
        logger.info("Shutting down manager...")

        # Stop all workers
        for worker_id in list(self.workers.keys()):
            await self._stop_worker(worker_id)

        # Release all leases
        for lease_id in list(self.active_leases.keys()):
            try:
                await self.lease_manager.release_lease(lease_id)
            except Exception as e:
                logger.error(f"Error releasing lease {lease_id}: {e}")

        logger.info("Manager shutdown complete")


def main():
    """Main entry point for manager process."""
    import argparse

    parser = argparse.ArgumentParser(description="Somba Pipeline Manager")
    parser.add_argument(
        "--config", required=True, help="Path to manager configuration file"
    )
    parser.add_argument("--runner-id", help="Runner ID (overrides config)")
    parser.add_argument("--shard-id", help="Shard ID (overrides config)")

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config_data = json.load(f)

    config = ManagerConfig(**config_data)

    # Override with command line arguments
    if args.runner_id:
        config.runner_id = args.runner_id
    if args.shard_id:
        config.shard_id = args.shard_id

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Start manager
    manager = ManagerProcess(config)

    try:
        asyncio.run(manager.start())
    except KeyboardInterrupt:
        logger.info("Manager stopped by user")
    except Exception as e:
        logger.error(f"Manager error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
