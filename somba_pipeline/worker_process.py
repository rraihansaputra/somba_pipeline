"""
Worker Process Management for Phase 4 - Manages worker subprocesses.
"""

import asyncio
import json
import logging
import signal
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from .schemas import ShardConfig, CameraConfig

logger = logging.getLogger(__name__)


class WorkerProcess:
    """Manages a single worker subprocess."""

    def __init__(self, worker_id: str, config):
        self.worker_id = worker_id
        self.config = config
        self.process: Optional[asyncio.subprocess.Process] = None
        self.start_time: Optional[datetime] = None
        self.assigned_cameras: List[str] = []
        self.healthy = False
        self.config_path: Optional[str] = None

        # Health monitoring
        self.last_heartbeat: Optional[datetime] = None
        self.health_check_url = f"http://localhost:8080/health"

        logger.info(f"WorkerProcess initialized: {worker_id}")

    async def start(self, cameras: List[str]) -> bool:
        """Start worker process with assigned cameras."""
        try:
            self.assigned_cameras = cameras

            # Create worker-specific config
            self.config_path = self._create_worker_config(cameras)

            # Build command
            cmd = self._build_command(cameras)

            logger.info(
                f"Starting worker {self.worker_id} with command: {' '.join(cmd)}"
            )

            # Start subprocess
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_environment(),
            )

            self.start_time = datetime.now(timezone.utc)

            # Start monitoring tasks
            asyncio.create_task(self._monitor_output())
            asyncio.create_task(self._monitor_health())

            logger.info(f"Started worker {self.worker_id} with cameras: {cameras}")
            return True

        except Exception as e:
            logger.error(f"Failed to start worker {self.worker_id}: {e}")
            return False

    def _build_command(self, cameras: List[str]) -> List[str]:
        """Build command to start worker process."""
        cmd = [
            "python",
            "-m",
            "somba_pipeline.worker",
            "--config",
            self.config_path,
            "--worker-id",
            self.worker_id,
        ]

        if cameras:
            cmd.extend(["--cameras", ",".join(cameras)])

        return cmd

    def _build_environment(self) -> Dict[str, str]:
        """Build environment variables for worker process."""
        env = os.environ.copy()

        # Add worker-specific environment variables
        env.update(
            {
                "WORKER_ID": self.worker_id,
                "PYTHONPATH": str(Path(__file__).parent.parent),
                "LOG_LEVEL": "INFO",
            }
        )

        return env

    def _create_worker_config(self, cameras: List[str]) -> str:
        """Create worker-specific configuration file."""
        try:
            # Generate config with only assigned cameras
            worker_config_data = {
                "runner_id": self.config.runner_id,
                "shard_id": self.worker_id,
                "max_fps": self.config.max_fps,
                "sources": [
                    {
                        "camera_uuid": cam,
                        "url": f"rtsp://admin:password@localhost:8554/{cam}",
                    }
                    for cam in cameras
                ],
                "amqp": self.config.amqp,
                "cp": self.config.cp,
                "telemetry": {"report_interval_seconds": 5},
                "cameras": {},
            }

            # Add camera configurations
            for cam in cameras:
                if cam in self.config.cameras:
                    worker_config_data["cameras"][cam] = self.config.cameras[cam].dict()
                else:
                    # Default camera configuration
                    worker_config_data["cameras"][cam] = {
                        "camera_uuid": cam,
                        "zones": [],
                        "motion_gating": {
                            "enabled": True,
                            "downscale": 0.5,
                            "dilation_px": 6,
                            "min_area_px": 1500,
                            "cooldown_frames": 2,
                            "noise_floor": 12,
                        },
                        "allow_labels": ["person", "car", "truck"],
                        "deny_labels": [],
                        "min_score": 0.30,
                        "zone_test": "center",
                        "iou_threshold": 0.10,
                    }

            # Write config to temporary file
            config_path = f"/tmp/worker_{self.worker_id}_config.json"
            with open(config_path, "w") as f:
                json.dump(worker_config_data, f, indent=2)

            logger.debug(f"Created worker config at {config_path}")
            return config_path

        except Exception as e:
            logger.error(f"Failed to create worker config for {self.worker_id}: {e}")
            raise

    async def _monitor_output(self):
        """Monitor worker process output."""
        if not self.process:
            return

        try:
            while self.process and not self.process.stdout.at_eof():
                line = await self.process.stdout.readline()
                if line:
                    logger.debug(
                        f"Worker {self.worker_id} stdout: {line.decode().strip()}"
                    )

        except Exception as e:
            logger.error(f"Error monitoring worker {self.worker_id} stdout: {e}")

        try:
            while self.process and not self.process.stderr.at_eof():
                line = await self.process.stderr.readline()
                if line:
                    logger.warning(
                        f"Worker {self.worker_id} stderr: {line.decode().strip()}"
                    )

        except Exception as e:
            logger.error(f"Error monitoring worker {self.worker_id} stderr: {e}")

    async def _monitor_health(self):
        """Monitor worker health through HTTP endpoint."""
        import aiohttp

        while self.process and self.process.returncode is None:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.health_check_url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            self.healthy = True
                            self.last_heartbeat = datetime.now(timezone.utc)
                        else:
                            logger.warning(
                                f"Worker {self.worker_id} health check failed: {response.status}"
                            )
                            self.healthy = False

            except Exception as e:
                logger.debug(f"Health check failed for worker {self.worker_id}: {e}")
                self.healthy = False

            await asyncio.sleep(10)  # Check every 10 seconds

    def is_healthy(self) -> bool:
        """Check if worker process is healthy."""
        if not self.process or self.process.returncode is not None:
            logger.debug(f"Worker {self.worker_id} process not running")
            return False

        # Check if process is responsive
        if not self.healthy:
            logger.debug(f"Worker {self.worker_id} not healthy")
            return False

        # Check heartbeat timeout
        if self.last_heartbeat:
            timeout = 60  # 60 second heartbeat timeout
            if (
                datetime.now(timezone.utc) - self.last_heartbeat
            ).total_seconds() > timeout:
                logger.warning(f"Worker {self.worker_id} heartbeat timeout")
                return False

        # Check start time (give 30 seconds grace period)
        if self.start_time:
            grace_period = 30
            if (
                datetime.now(timezone.utc) - self.start_time
            ).total_seconds() < grace_period:
                return True  # Still in grace period

        return True

    async def stop(self, timeout: int = 30):
        """Stop worker process gracefully."""
        try:
            logger.info(f"Stopping worker {self.worker_id}...")

            if self.process:
                # Try graceful shutdown first
                self.process.send_signal(signal.SIGTERM)

                try:
                    await asyncio.wait_for(self.process.wait(), timeout=timeout)
                    logger.info(f"Worker {self.worker_id} stopped gracefully")
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Worker {self.worker_id} did not stop gracefully, killing..."
                    )
                    self.process.kill()
                    await self.process.wait()
                    logger.info(f"Worker {self.worker_id} killed")

            # Clean up config file
            if self.config_path and os.path.exists(self.config_path):
                try:
                    os.remove(self.config_path)
                    logger.debug(f"Removed worker config: {self.config_path}")
                except Exception as e:
                    logger.error(
                        f"Failed to remove worker config {self.config_path}: {e}"
                    )

        except Exception as e:
            logger.error(f"Error stopping worker {self.worker_id}: {e}")

    async def restart(self, cameras: List[str]) -> bool:
        """Restart worker process with new camera assignments."""
        try:
            logger.info(f"Restarting worker {self.worker_id}...")

            # Stop current process
            await self.stop()

            # Start with new cameras
            success = await self.start(cameras)

            if success:
                logger.info(f"Worker {self.worker_id} restarted successfully")
            else:
                logger.error(f"Failed to restart worker {self.worker_id}")

            return success

        except Exception as e:
            logger.error(f"Error restarting worker {self.worker_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "assigned_cameras": self.assigned_cameras,
            "healthy": self.healthy,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_heartbeat": self.last_heartbeat.isoformat()
            if self.last_heartbeat
            else None,
            "process_running": self.process and self.process.returncode is None,
            "process_pid": self.process.pid if self.process else None,
            "config_path": self.config_path,
        }

    def update_camera_assignments(self, cameras: List[str]):
        """Update camera assignments (requires restart)."""
        self.assigned_cameras = cameras
        logger.info(
            f"Updated camera assignments for worker {self.worker_id}: {cameras}"
        )

    async def send_signal(self, sig: int):
        """Send signal to worker process."""
        if self.process:
            self.process.send_signal(sig)
            logger.info(f"Sent signal {sig} to worker {self.worker_id}")


class WorkerPool:
    """Manages a pool of worker processes."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.workers: Dict[str, WorkerProcess] = {}
        self.worker_assignments: Dict[str, List[str]] = {}

        logger.info(f"WorkerPool initialized with max_workers={max_workers}")

    async def start_worker(self, worker_id: str, cameras: List[str], config) -> bool:
        """Start a new worker process."""
        if worker_id in self.workers:
            logger.warning(f"Worker {worker_id} already exists")
            return False

        worker = WorkerProcess(worker_id, config)
        success = await worker.start(cameras)

        if success:
            self.workers[worker_id] = worker
            self.worker_assignments[worker_id] = cameras
            logger.info(f"Started worker {worker_id} in pool")
        else:
            logger.error(f"Failed to start worker {worker_id} in pool")

        return success

    async def stop_worker(self, worker_id: str):
        """Stop a worker process."""
        if worker_id not in self.workers:
            logger.warning(f"Worker {worker_id} not found in pool")
            return

        worker = self.workers[worker_id]
        await worker.stop()

        del self.workers[worker_id]
        if worker_id in self.worker_assignments:
            del self.worker_assignments[worker_id]

        logger.info(f"Stopped worker {worker_id} from pool")

    async def restart_worker(self, worker_id: str, cameras: List[str], config) -> bool:
        """Restart a worker process."""
        if worker_id not in self.workers:
            logger.warning(f"Worker {worker_id} not found in pool")
            return False

        worker = self.workers[worker_id]
        success = await worker.restart(cameras)

        if success:
            self.worker_assignments[worker_id] = cameras
            logger.info(f"Restarted worker {worker_id} in pool")
        else:
            logger.error(f"Failed to restart worker {worker_id} in pool")

        return success

    def get_worker(self, worker_id: str) -> Optional[WorkerProcess]:
        """Get a worker process by ID."""
        return self.workers.get(worker_id)

    def get_all_workers(self) -> Dict[str, WorkerProcess]:
        """Get all worker processes."""
        return self.workers.copy()

    def get_worker_assignments(self) -> Dict[str, List[str]]:
        """Get all worker assignments."""
        return self.worker_assignments.copy()

    def get_healthy_workers(self) -> List[str]:
        """Get list of healthy worker IDs."""
        return [
            worker_id
            for worker_id, worker in self.workers.items()
            if worker.is_healthy()
        ]

    def get_unhealthy_workers(self) -> List[str]:
        """Get list of unhealthy worker IDs."""
        return [
            worker_id
            for worker_id, worker in self.workers.items()
            if not worker.is_healthy()
        ]

    async def monitor_and_restart(self):
        """Monitor worker health and restart unhealthy workers."""
        while True:
            try:
                unhealthy_workers = self.get_unhealthy_workers()

                for worker_id in unhealthy_workers:
                    logger.warning(f"Worker {worker_id} is unhealthy, restarting...")

                    # Get current assignments
                    cameras = self.worker_assignments.get(worker_id, [])

                    # Restart worker
                    if worker_id in self.workers:
                        worker = self.workers[worker_id]
                        await worker.restart(cameras)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in worker monitoring: {e}")
                await asyncio.sleep(60)

    async def shutdown(self):
        """Shutdown all worker processes."""
        logger.info("Shutting down worker pool...")

        for worker_id in list(self.workers.keys()):
            await self.stop_worker(worker_id)

        logger.info("Worker pool shutdown complete")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        healthy_count = len(self.get_healthy_workers())
        unhealthy_count = len(self.get_unhealthy_workers())

        return {
            "max_workers": self.max_workers,
            "total_workers": len(self.workers),
            "healthy_workers": healthy_count,
            "unhealthy_workers": unhealthy_count,
            "worker_assignments": self.worker_assignments.copy(),
        }
