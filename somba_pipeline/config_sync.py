"""
Configuration Synchronization for Phase 4 - Manages configuration sync between control plane and workers.
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable

from .control_plane_client import ControlPlaneClient
from .schemas import CameraConfig

logger = logging.getLogger(__name__)


class ConfigurationSync:
    """Manages configuration synchronization between control plane and workers."""

    def __init__(self, control_plane_config: Dict[str, str]):
        self.cp_client = ControlPlaneClient(
            control_plane_config.get("base_url", "http://localhost:8000"),
            control_plane_config.get("token", "default-token"),
        )

        # Configuration storage
        self.local_configs: Dict[str, CameraConfig] = {}  # camera_uuid -> CameraConfig
        self.config_versions: Dict[str, str] = {}  # camera_uuid -> version_hash
        self.last_sync_time: Dict[str, datetime] = {}  # camera_uuid -> last_sync_time

        # Global configuration version
        self.global_config_version: Optional[str] = None
        self.global_zone_version: Optional[str] = None
        self.last_global_sync: Optional[datetime] = None

        # Sync state
        self.sync_callbacks: List[Callable[[str, CameraConfig], None]] = []
        self.running = True

        logger.info("ConfigurationSync initialized")

    async def sync_camera_configurations(
        self, camera_uuids: List[str]
    ) -> Dict[str, CameraConfig]:
        """Sync configurations for multiple cameras."""
        updated_configs = {}

        for camera_uuid in camera_uuids:
            try:
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
                        self.last_sync_time[camera_uuid] = datetime.now(timezone.utc)
                        updated_configs[camera_uuid] = remote_config

                        logger.info(f"Configuration updated for camera {camera_uuid}")

                        # Notify callbacks
                        await self._notify_config_change(camera_uuid, remote_config)
                    else:
                        # Use cached configuration
                        if camera_uuid in self.local_configs:
                            updated_configs[camera_uuid] = self.local_configs[
                                camera_uuid
                            ]
                            logger.debug(
                                f"Using cached config for camera {camera_uuid}"
                            )
                else:
                    logger.warning(f"Failed to get config for camera {camera_uuid}")

            except Exception as e:
                logger.error(f"Error syncing config for camera {camera_uuid}: {e}")

        return updated_configs

    async def sync_global_configuration(self) -> bool:
        """Sync global configuration version from control plane."""
        try:
            version_response = await self.cp_client.get_config_version()

            if version_response:
                old_config_version = self.global_config_version
                old_zone_version = self.global_zone_version

                self.global_config_version = version_response.config_version
                self.global_zone_version = version_response.zone_version
                self.last_global_sync = datetime.now(timezone.utc)

                # Check if versions changed
                if (
                    old_config_version != self.global_config_version
                    or old_zone_version != self.global_zone_version
                ):
                    logger.info(
                        f"Global configuration updated: config={self.global_config_version}, zones={self.global_zone_version}"
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"Error syncing global configuration: {e}")
            return False

    async def watch_configuration_changes(
        self, callback: Optional[Callable[[str, CameraConfig], None]] = None
    ):
        """Watch for configuration changes and notify workers."""
        if callback:
            self.sync_callbacks.append(callback)

        while self.running:
            try:
                # Sync global configuration
                global_changed = await self.sync_global_configuration()

                if global_changed:
                    # If global config changed, sync all cameras
                    camera_uuids = list(self.local_configs.keys())
                    if camera_uuids:
                        await self.sync_camera_configurations(camera_uuids)
                else:
                    # Check individual camera configurations
                    await self._check_camera_configurations()

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                logger.info("Configuration watch cancelled")
                break
            except Exception as e:
                logger.error(f"Error watching configuration changes: {e}")
                await asyncio.sleep(60)

    async def _check_camera_configurations(self):
        """Check individual camera configurations for changes."""
        # Get all assigned cameras
        camera_uuids = list(self.local_configs.keys())

        for camera_uuid in camera_uuids:
            try:
                # Get current configuration from control plane
                remote_config = await self.cp_client.get_camera_config(camera_uuid)

                if remote_config:
                    if self._has_config_changed(camera_uuid, remote_config):
                        # Configuration changed, update local cache
                        self.local_configs[camera_uuid] = remote_config
                        self.config_versions[camera_uuid] = self._calculate_config_hash(
                            remote_config
                        )
                        self.last_sync_time[camera_uuid] = datetime.now(timezone.utc)

                        logger.info(f"Configuration updated for camera {camera_uuid}")

                        # Notify callbacks
                        await self._notify_config_change(camera_uuid, remote_config)

            except Exception as e:
                logger.error(f"Error checking config for camera {camera_uuid}: {e}")

    def _calculate_config_hash(self, config: CameraConfig) -> str:
        """Calculate hash of configuration for versioning."""
        try:
            config_dict = config.dict()
            config_str = json.dumps(config_dict, sort_keys=True)
            return hashlib.md5(config_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating config hash: {e}")
            return "error"

    def _has_config_changed(self, camera_uuid: str, new_config: CameraConfig) -> bool:
        """Check if configuration has changed."""
        current_hash = self.config_versions.get(camera_uuid)
        new_hash = self._calculate_config_hash(new_config)
        return current_hash != new_hash

    async def _notify_config_change(self, camera_uuid: str, new_config: CameraConfig):
        """Notify callbacks of configuration change."""
        for callback in self.sync_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(camera_uuid, new_config)
                else:
                    callback(camera_uuid, new_config)
            except Exception as e:
                logger.error(f"Error in config change callback: {e}")

    def get_camera_config(self, camera_uuid: str) -> Optional[CameraConfig]:
        """Get cached camera configuration."""
        return self.local_configs.get(camera_uuid)

    def get_camera_config_version(self, camera_uuid: str) -> Optional[str]:
        """Get cached camera configuration version."""
        return self.config_versions.get(camera_uuid)

    def get_all_camera_configs(self) -> Dict[str, CameraConfig]:
        """Get all cached camera configurations."""
        return self.local_configs.copy()

    def add_camera_config(self, camera_uuid: str, config: CameraConfig):
        """Add camera configuration to cache."""
        self.local_configs[camera_uuid] = config
        self.config_versions[camera_uuid] = self._calculate_config_hash(config)
        self.last_sync_time[camera_uuid] = datetime.now(timezone.utc)

        logger.info(f"Added camera config to cache: {camera_uuid}")

    def remove_camera_config(self, camera_uuid: str):
        """Remove camera configuration from cache."""
        if camera_uuid in self.local_configs:
            del self.local_configs[camera_uuid]
        if camera_uuid in self.config_versions:
            del self.config_versions[camera_uuid]
        if camera_uuid in self.last_sync_time:
            del self.last_sync_time[camera_uuid]

        logger.info(f"Removed camera config from cache: {camera_uuid}")

    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        current_time = datetime.now(timezone.utc)

        # Calculate cache age
        cache_ages = {}
        for camera_uuid, sync_time in self.last_sync_time.items():
            age = (current_time - sync_time).total_seconds()
            cache_ages[camera_uuid] = age

        return {
            "total_cameras": len(self.local_configs),
            "cached_cameras": len(self.config_versions),
            "global_config_version": self.global_config_version,
            "global_zone_version": self.global_zone_version,
            "last_global_sync": self.last_global_sync.isoformat()
            if self.last_global_sync
            else None,
            "cache_ages_seconds": cache_ages,
            "sync_callbacks_count": len(self.sync_callbacks),
        }

    async def force_sync_camera(self, camera_uuid: str) -> Optional[CameraConfig]:
        """Force synchronization for a specific camera."""
        try:
            remote_config = await self.cp_client.get_camera_config(camera_uuid)

            if remote_config:
                self.local_configs[camera_uuid] = remote_config
                self.config_versions[camera_uuid] = self._calculate_config_hash(
                    remote_config
                )
                self.last_sync_time[camera_uuid] = datetime.now(timezone.utc)

                logger.info(f"Force synced configuration for camera {camera_uuid}")

                # Notify callbacks
                await self._notify_config_change(camera_uuid, remote_config)

                return remote_config
            else:
                logger.warning(f"Failed to force sync config for camera {camera_uuid}")
                return None

        except Exception as e:
            logger.error(f"Error force syncing config for camera {camera_uuid}: {e}")
            return None

    async def validate_configuration_consistency(self) -> Dict[str, bool]:
        """Validate consistency between local and remote configurations."""
        consistency_results = {}

        for camera_uuid in self.local_configs.keys():
            try:
                # Get remote configuration
                remote_config = await self.cp_client.get_camera_config(camera_uuid)

                if remote_config:
                    # Compare versions
                    local_version = self.config_versions.get(camera_uuid)
                    remote_version = self._calculate_config_hash(remote_config)

                    consistency_results[camera_uuid] = local_version == remote_version
                else:
                    consistency_results[camera_uuid] = False

            except Exception as e:
                logger.error(
                    f"Error validating config consistency for camera {camera_uuid}: {e}"
                )
                consistency_results[camera_uuid] = False

        return consistency_results

    def is_configuration_fresh(
        self, camera_uuid: str, max_age_seconds: int = 300
    ) -> bool:
        """Check if configuration is fresh (within max age)."""
        if camera_uuid not in self.last_sync_time:
            return False

        last_sync = self.last_sync_time[camera_uuid]
        current_time = datetime.now(timezone.utc)
        age = (current_time - last_sync).total_seconds()

        return age <= max_age_seconds

    async def cleanup_stale_configurations(self, max_age_seconds: int = 3600):
        """Clean up stale configurations."""
        current_time = datetime.now(timezone.utc)
        stale_cameras = []

        for camera_uuid, sync_time in self.last_sync_time.items():
            age = (current_time - sync_time).total_seconds()
            if age > max_age_seconds:
                stale_cameras.append(camera_uuid)

        for camera_uuid in stale_cameras:
            logger.info(f"Cleaning up stale configuration for camera {camera_uuid}")
            self.remove_camera_config(camera_uuid)

        if stale_cameras:
            logger.info(f"Cleaned up {len(stale_cameras)} stale configurations")

    async def shutdown(self):
        """Shutdown configuration sync gracefully."""
        logger.info("Shutting down ConfigurationSync...")
        self.running = False

        # Clear callbacks
        self.sync_callbacks.clear()

        # Clear configurations
        self.local_configs.clear()
        self.config_versions.clear()
        self.last_sync_time.clear()

        logger.info("ConfigurationSync shutdown complete")


class ConfigurationManager:
    """High-level configuration management interface."""

    def __init__(self, control_plane_config: Dict[str, str]):
        self.config_sync = ConfigurationSync(control_plane_config)
        self.watch_task: Optional[asyncio.Task] = None

    async def start(
        self, callback: Optional[Callable[[str, CameraConfig], None]] = None
    ):
        """Start configuration management."""
        self.watch_task = asyncio.create_task(
            self.config_sync.watch_configuration_changes(callback)
        )

    async def stop(self):
        """Stop configuration management."""
        if self.watch_task:
            self.watch_task.cancel()
            try:
                await self.watch_task
            except asyncio.CancelledError:
                pass

        await self.config_sync.shutdown()

    async def get_camera_config(self, camera_uuid: str) -> Optional[CameraConfig]:
        """Get camera configuration."""
        return self.config_sync.get_camera_config(camera_uuid)

    async def sync_cameras(self, camera_uuids: List[str]) -> Dict[str, CameraConfig]:
        """Sync configurations for multiple cameras."""
        return await self.config_sync.sync_camera_configurations(camera_uuids)

    async def force_sync_camera(self, camera_uuid: str) -> Optional[CameraConfig]:
        """Force sync a specific camera configuration."""
        return await self.config_sync.force_sync_camera(camera_uuid)

    def get_stats(self) -> Dict[str, Any]:
        """Get configuration management statistics."""
        return self.config_sync.get_sync_stats()
