#!/usr/bin/env python3
"""
Startup script for Phase 4 Manager Process.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add the somba_pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from somba_pipeline.manager import ManagerProcess
from somba_pipeline.schemas import ManagerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """Main entry point for manager startup."""
    parser = argparse.ArgumentParser(description="Start Somba Pipeline Manager")
    parser.add_argument(
        "--config", required=True, help="Path to manager configuration file"
    )
    parser.add_argument("--runner-id", help="Override runner ID from config")
    parser.add_argument("--shard-id", help="Override shard ID from config")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration but don't start manager",
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)

        logger.info(f"Loading configuration from: {config_path}")
        config = ManagerConfig.from_json_file(str(config_path))

        # Override with command line arguments
        if args.runner_id:
            config.runner_id = args.runner_id
            logger.info(f"Overridden runner ID: {config.runner_id}")

        if args.shard_id:
            config.shard_id = args.shard_id
            logger.info(f"Overridden shard ID: {config.shard_id}")

        # Validate configuration
        logger.info("Validating configuration...")
        if not config.runner_id or not config.shard_id:
            logger.error("Configuration must include runner_id and shard_id")
            sys.exit(1)

        if not config.control_plane_url or not config.api_key:
            logger.error("Configuration must include control_plane_url and api_key")
            sys.exit(1)

        logger.info("Configuration validation successful")
        logger.info(f"Manager ID: {config.runner_id}")
        logger.info(f"Shard ID: {config.shard_id}")
        logger.info(f"Control Plane: {config.control_plane_url}")
        logger.info(f"Max Workers: {config.max_workers}")
        logger.info(f"Max Cameras per Worker: {config.max_cameras_per_worker}")

        if args.dry_run:
            logger.info("Dry run complete - configuration is valid")
            sys.exit(0)

        # Create and start manager
        logger.info("Starting manager process...")
        manager = ManagerProcess(config)

        try:
            asyncio.run(manager.start())
        except KeyboardInterrupt:
            logger.info("Manager stopped by user")
        except Exception as e:
            logger.error(f"Manager error: {e}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Startup error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
