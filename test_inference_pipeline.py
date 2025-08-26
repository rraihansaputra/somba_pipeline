#!/usr/bin/env python3
"""
Simple test script to validate InferencePipeline integration.
This script tests the worker with a webcam or test video.
"""

import sys
import time
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from somba_pipeline.worker import ProductionWorker
from somba_pipeline.schemas import ShardConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_inference_pipeline():
    """Test the InferencePipeline integration."""
    logger.info("Starting InferencePipeline integration test...")

    # Load test configuration
    config_path = Path("config/test_shard_config.json")
    if not config_path.exists():
        logger.error(f"Test configuration not found: {config_path}")
        return False

    try:
        config = ShardConfig.from_json_file(str(config_path))
        logger.info(f"Loaded configuration: {config.runner_id}/{config.shard_id}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return False

    # Create and start worker
    worker = None
    try:
        logger.info("Creating ProductionWorker...")
        worker = ProductionWorker(config)

        logger.info("Starting worker (this will run for 30 seconds)...")

        # Start worker in a separate thread to allow for controlled shutdown
        import threading
        worker_thread = threading.Thread(target=worker.start, daemon=True)
        worker_thread.start()

        # Let it run for 30 seconds
        logger.info("Worker started, running for 30 seconds...")
        time.sleep(30)

        # Check if worker is ready
        if worker.ready:
            logger.info("✅ Worker is ready")
        else:
            logger.warning("⚠️ Worker is not ready")

        # Check pipeline status
        if worker.pipeline:
            logger.info("✅ InferencePipeline is initialized")
        else:
            logger.error("❌ InferencePipeline is not initialized")
            return False

        # Check camera states
        logger.info(f"Camera states: {worker.camera_states}")

        # Check metrics
        logger.info("Checking Prometheus metrics...")
        try:
            # Try to access some metrics
            for camera_uuid in [src['camera_uuid'] for src in config.sources]:
                stream_up = worker.stream_up.labels(camera_uuid=camera_uuid)._value._value
                logger.info(f"Stream up for {camera_uuid}: {stream_up}")
        except Exception as e:
            logger.warning(f"Could not read metrics: {e}")

        logger.info("✅ Test completed successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if worker:
            logger.info("Shutting down worker...")
            worker.running = False
            if worker.pipeline:
                try:
                    worker.pipeline.terminate()
                    worker.pipeline.join()
                except:
                    pass


def test_basic_imports():
    """Test that all required imports work."""
    logger.info("Testing basic imports...")

    try:
        from inference import InferencePipeline
        from inference.core.interfaces.stream.entities import VideoFrame
        from inference.core.interfaces.camera.entities import StatusUpdate, UpdateSeverity
        logger.info("✅ All inference library imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("=== InferencePipeline Integration Test ===")

    # Test 1: Basic imports
    if not test_basic_imports():
        logger.error("Basic imports failed, cannot continue")
        sys.exit(1)

    # Test 2: Full integration test
    if test_inference_pipeline():
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("💥 Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
