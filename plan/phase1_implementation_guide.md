# Phase 1: Foundation & Single-Stream MVP - Implementation Guide

## Objective
Create a minimal viable Worker that can process a single RTSP stream using the existing InferencePipeline with go2rtc for stream management, establishing the foundation for all subsequent phases.

## Prerequisites
- Python 3.8+
- Roboflow Inference library installed
- go2rtc binary or Docker container
- Test video file (provided by user)
- ffmpeg (for video file to RTSP conversion)

## Step-by-Step Implementation

### Step 1: Setup go2rtc

```yaml
# go2rtc.yaml - Configuration file for go2rtc
streams:
  # Test stream from video file
  test_video:
    - ffmpeg:test_video.mp4#video=h264#hardware  # Your test video

  # Example RTSP camera (when available)
  camera_1:
    - rtsp://username:password@192.168.1.100:554/stream1

  # Example with multiple fallbacks
  camera_2:
    - rtsp://192.168.1.101:554/stream1
    - rtsp://192.168.1.101:554/stream2  # Fallback stream

api:
  listen: "127.0.0.1:1984"  # API endpoint for management

rtsp:
  listen: "127.0.0.1:8554"  # RTSP server endpoint

webrtc:
  listen: ":8555"  # WebRTC for browser viewing (optional)

log:
  level: info  # debug, info, warn, error
```

```bash
# setup_go2rtc.sh - Script to download and start go2rtc
#!/bin/bash

# Download go2rtc if not present
if [ ! -f "go2rtc" ]; then
    echo "Downloading go2rtc..."
    # For Linux x64
    wget -O go2rtc https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_linux_amd64
    # For Mac M1
    # wget -O go2rtc https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_darwin_arm64
    chmod +x go2rtc
fi

# Start go2rtc with config
./go2rtc -config go2rtc.yaml
```

### Step 2: go2rtc Manager Integration

```python
# go2rtc_manager.py
import requests
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StreamInfo:
    """Information about a go2rtc stream."""
    name: str
    url: str
    ready: bool
    producers: List[str]
    consumers: int

class Go2RTCManager:
    """Manager for go2rtc streams."""

    def __init__(self, api_url: str = "http://127.0.0.1:1984"):
        self.api_url = api_url
        self.rtsp_base = "rtsp://127.0.0.1:8554"

    def health_check(self) -> bool:
        """Check if go2rtc is running."""
        try:
            response = requests.get(f"{self.api_url}/api/streams", timeout=2)
            return response.status_code == 200
        except:
            return False

    def get_streams(self) -> Dict[str, StreamInfo]:
        """Get all configured streams."""
        try:
            response = requests.get(f"{self.api_url}/api/streams")
            response.raise_for_status()

            streams = {}
            data = response.json()

            for name, info in data.items():
                streams[name] = StreamInfo(
                    name=name,
                    url=f"{self.rtsp_base}/{name}",
                    ready=len(info.get('producers', [])) > 0,
                    producers=info.get('producers', []),
                    consumers=info.get('consumers', 0)
                )

            return streams

        except Exception as e:
            logger.error(f"Failed to get streams: {e}")
            return {}

    def add_stream(self, name: str, source: str) -> bool:
        """Add a new stream dynamically."""
        try:
            # Add stream via API
            response = requests.post(
                f"{self.api_url}/api/streams",
                json={name: source}
            )
            response.raise_for_status()

            # Wait for stream to be ready
            for _ in range(10):
                streams = self.get_streams()
                if name in streams and streams[name].ready:
                    logger.info(f"Stream {name} is ready")
                    return True
                time.sleep(1)

            logger.warning(f"Stream {name} not ready after 10 seconds")
            return False

        except Exception as e:
            logger.error(f"Failed to add stream {name}: {e}")
            return False

    def remove_stream(self, name: str) -> bool:
        """Remove a stream."""
        try:
            response = requests.delete(f"{self.api_url}/api/streams?name={name}")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to remove stream {name}: {e}")
            return False

    def get_stream_url(self, name: str) -> Optional[str]:
        """Get RTSP URL for a stream."""
        streams = self.get_streams()
        if name in streams:
            return streams[name].url
        return None

    def wait_for_stream(self, name: str, timeout: int = 30) -> bool:
        """Wait for a stream to become ready."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            streams = self.get_streams()
            if name in streams and streams[name].ready:
                logger.info(f"Stream {name} is ready")
                return True
            time.sleep(1)

        logger.error(f"Stream {name} not ready after {timeout} seconds")
        return False
```

### Step 3: Basic Worker Process with go2rtc

```python
# worker_basic.py
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from inference import InferencePipeline
from inference.core.interfaces.stream.entities import VideoFrame
from go2rtc_manager import Go2RTCManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BasicWorker:
    """Single-stream inference worker with go2rtc integration."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline: Optional[InferencePipeline] = None
        self.go2rtc = Go2RTCManager(config.get('go2rtc_api', 'http://127.0.0.1:1984'))
        self.stats = {
            'frames_processed': 0,
            'detections_found': 0,
            'start_time': datetime.now().isoformat(),
            'last_frame_time': None,
            'errors': 0,
            'inference_latencies': []
        }
        self.running = True
        self.output_dir = Path(config.get('output_dir', './detections'))
        self.output_dir.mkdir(exist_ok=True)

        # Setup graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        if self.pipeline:
            self.pipeline.terminate()

    def _validate_go2rtc(self) -> bool:
        """Validate go2rtc is running and stream is available."""
        if not self.go2rtc.health_check():
            logger.error("go2rtc is not running!")
            return False

        # Check if our stream exists
        stream_name = self.config.get('stream_name', 'test_video')
        streams = self.go2rtc.get_streams()

        if stream_name not in streams:
            logger.error(f"Stream '{stream_name}' not found in go2rtc")
            logger.info(f"Available streams: {list(streams.keys())}")
            return False

        if not streams[stream_name].ready:
            logger.warning(f"Stream '{stream_name}' exists but not ready, waiting...")
            if not self.go2rtc.wait_for_stream(stream_name):
                return False

        logger.info(f"Stream '{stream_name}' is ready at {streams[stream_name].url}")
        return True

    def _on_prediction(self, predictions: dict, video_frame: VideoFrame):
        """Handle predictions from the inference pipeline."""
        try:
            # Track inference time
            if 'time' in predictions:
                self.stats['inference_latencies'].append(predictions['time'] * 1000)  # Convert to ms

            self.stats['frames_processed'] += 1
            self.stats['last_frame_time'] = datetime.now().isoformat()

            # Extract detections
            if predictions and 'predictions' in predictions:
                detections = predictions['predictions']
                self.stats['detections_found'] += len(detections)

                # Save detection event
                if detections:
                    self._save_detection(predictions, video_frame)

                # Log progress every 100 frames
                if self.stats['frames_processed'] % 100 == 0:
                    self._log_stats()

        except Exception as e:
            logger.error(f"Error processing prediction: {e}")
            self.stats['errors'] += 1

    def _save_detection(self, predictions: dict, video_frame: VideoFrame):
        """Save detection to JSON file."""
        timestamp = datetime.now().isoformat()
        detection_data = {
            'timestamp': timestamp,
            'frame_id': video_frame.frame_id,
            'frame_timestamp': video_frame.frame_timestamp,
            'source_id': video_frame.source_id,
            'predictions': predictions.get('predictions', []),
            'inference_time_ms': predictions.get('time', 0) * 1000
        }

        filename = f"detection_{video_frame.frame_id}_{int(time.time()*1000)}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(detection_data, f, indent=2)

    def _log_stats(self):
        """Log current statistics."""
        runtime = (datetime.now() - datetime.fromisoformat(self.stats['start_time'])).total_seconds()
        fps = self.stats['frames_processed'] / runtime if runtime > 0 else 0

        # Calculate latency stats
        latencies = self.stats['inference_latencies']
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            # Simple P95 calculation
            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p95_latency = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else sorted_latencies[-1]
        else:
            avg_latency = 0
            p95_latency = 0

        logger.info(f"Stats: Frames={self.stats['frames_processed']}, "
                   f"Detections={self.stats['detections_found']}, "
                   f"FPS={fps:.2f}, "
                   f"Avg Latency={avg_latency:.1f}ms, "
                   f"P95 Latency={p95_latency:.1f}ms, "
                   f"Errors={self.stats['errors']}")

    def start(self):
        """Start processing the video stream."""
        logger.info(f"Starting worker with config: {self.config}")

        # Validate go2rtc
        if not self._validate_go2rtc():
            logger.error("go2rtc validation failed, exiting")
            sys.exit(1)

        try:
            # Get stream URL from go2rtc
            stream_name = self.config.get('stream_name', 'test_video')
            stream_url = self.go2rtc.get_stream_url(stream_name)

            if not stream_url:
                raise ValueError(f"Could not get URL for stream '{stream_name}'")

            logger.info(f"Connecting to stream: {stream_url}")

            # Initialize pipeline
            self.pipeline = InferencePipeline.init(
                model_id=self.config['model_id'],
                video_reference=stream_url,
                api_key=self.config.get('api_key'),
                on_prediction=self._on_prediction,
                max_fps=self.config.get('max_fps', 10)
            )

            # Start processing
            logger.info("Pipeline initialized, starting inference...")
            self.pipeline.start()

            # Keep running until shutdown
            start_time = time.time()
            while self.running:
                time.sleep(1)

                # Optional: Check stream health periodically
                if int(time.time() - start_time) % 30 == 0:
                    streams = self.go2rtc.get_streams()
                    if stream_name in streams:
                        logger.info(f"Stream health: {streams[stream_name].consumers} consumers")

            # Final stats
            logger.info("Worker shutting down...")
            self._log_stats()
            self._save_final_stats()

        except Exception as e:
            logger.error(f"Fatal error in worker: {e}")
            raise
        finally:
            if self.pipeline:
                self.pipeline.terminate()
                self.pipeline.join()

    def _save_final_stats(self):
        """Save final statistics to file."""
        self.stats['end_time'] = datetime.now().isoformat()

        # Calculate final latency stats
        latencies = self.stats['inference_latencies']
        if latencies:
            self.stats['latency_stats'] = {
                'mean': sum(latencies) / len(latencies),
                'min': min(latencies),
                'max': max(latencies),
                'p95': sorted(latencies)[int(len(latencies) * 0.95)]
            }

        stats_file = self.output_dir / 'worker_stats.json'

        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)

        logger.info(f"Final stats saved to {stats_file}")


def main():
    """Main entry point."""
    # Load configuration
    config = {
        'model_id': 'coco/11',  # Use COCO model for testing
        'stream_name': 'test_video',  # Stream name in go2rtc
        'go2rtc_api': 'http://127.0.0.1:1984',
        'api_key': None,  # Set if using private model
        'max_fps': 10,
        'output_dir': './detections'
    }

    # Override with environment variables if present
    import os
    if os.getenv('STREAM_NAME'):
        config['stream_name'] = os.getenv('STREAM_NAME')
    if os.getenv('MODEL_ID'):
        config['model_id'] = os.getenv('MODEL_ID')
    if os.getenv('API_KEY'):
        config['api_key'] = os.getenv('API_KEY')
    if os.getenv('GO2RTC_API'):
        config['go2rtc_api'] = os.getenv('GO2RTC_API')

    # Start worker
    worker = BasicWorker(config)
    worker.start()


if __name__ == '__main__':
    main()
```

### Step 4: Performance Testing Script

```python
# test_performance.py
import time
import psutil
import json
from datetime import datetime
import statistics
import logging
import subprocess
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor Worker performance metrics."""

    def __init__(self, pid: int = None):
        self.pid = pid or psutil.Process().pid
        self.process = psutil.Process(self.pid)
        self.samples = {
            'cpu_percent': [],
            'memory_mb': [],
            'gpu_percent': [],  # If GPU monitoring available
            'gpu_memory_mb': []
        }
        self.start_time = time.time()

    def sample(self):
        """Take a performance sample."""
        try:
            # CPU and memory
            cpu = self.process.cpu_percent(interval=0.1)
            mem = self.process.memory_info().rss / 1024 / 1024  # MB

            self.samples['cpu_percent'].append(cpu)
            self.samples['memory_mb'].append(mem)

            # GPU monitoring (if nvidia-smi available)
            if self._has_gpu():
                gpu_stats = self._get_gpu_stats()
                if gpu_stats:
                    self.samples['gpu_percent'].append(gpu_stats['gpu'])
                    self.samples['gpu_memory_mb'].append(gpu_stats['memory'])

        except Exception as e:
            logger.error(f"Sampling error: {e}")

    def _has_gpu(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            subprocess.run(['nvidia-smi'], capture_output=True, check=True)
            return True
        except:
            return False

    def _get_gpu_stats(self) -> dict:
        """Get GPU statistics using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            gpu, memory = result.stdout.strip().split(', ')
            return {'gpu': float(gpu), 'memory': float(memory)}
        except:
            return None

    def get_stats(self) -> dict:
        """Get performance statistics."""
        runtime = time.time() - self.start_time

        stats = {
            'runtime_seconds': runtime,
            'samples_collected': len(self.samples['cpu_percent']),
        }

        # Calculate statistics for each metric
        for metric, values in self.samples.items():
            if values:
                stats[metric] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'p95': statistics.quantiles(values, n=20)[18] if len(values) > 20 else max(values),
                    'max': max(values),
                    'min': min(values)
                }

        return stats

    def save_report(self, filepath: str = 'performance_report.json'):
        """Save performance report to file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'pid': self.pid,
            'stats': self.get_stats()
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Performance report saved to {filepath}")
        return report


# Test harness
def run_performance_test(duration_seconds: int = 600):
    """Run a performance test for specified duration."""
    from go2rtc_manager import Go2RTCManager

    logger.info(f"Starting {duration_seconds} second performance test...")

    # Verify go2rtc is running
    go2rtc = Go2RTCManager()
    if not go2rtc.health_check():
        logger.error("go2rtc is not running! Please start it first.")
        return False

    # Start the worker process
    worker_process = subprocess.Popen(
        ['python', 'worker_basic.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, 'STREAM_NAME': 'test_video'}
    )

    # Monitor performance
    monitor = PerformanceMonitor(worker_process.pid)

    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        monitor.sample()
        time.sleep(1)

    # Stop worker gracefully
    worker_process.terminate()
    try:
        worker_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("Worker didn't stop gracefully, killing...")
        worker_process.kill()

    # Generate report
    report = monitor.save_report()

    # Load worker stats
    worker_stats_file = Path('./detections/worker_stats.json')
    if worker_stats_file.exists():
        with open(worker_stats_file) as f:
            worker_stats = json.load(f)
            logger.info(f"Worker processed {worker_stats['frames_processed']} frames")

    # Check acceptance criteria
    logger.info("\n=== Performance Test Results ===")

    criteria_passed = True

    # Check CPU usage
    cpu_p95 = report['stats']['cpu_percent']['p95']
    logger.info(f"CPU P95: {cpu_p95:.1f}%")
    if cpu_p95 > 80:
        logger.warning("⚠️  CPU usage high")
        criteria_passed = False
    else:
        logger.info("✅ CPU usage acceptable")

    # Check memory usage
    mem_max = report['stats']['memory_mb']['max']
    logger.info(f"Memory Max: {mem_max:.1f} MB")
    if mem_max > 1024:
        logger.warning("⚠️  Memory usage exceeds 1GB")
        criteria_passed = False
    else:
        logger.info("✅ Memory usage acceptable")

    # Check worker stats
    if worker_stats_file.exists():
        with open(worker_stats_file) as f:
            worker_stats = json.load(f)

            # Check FPS
            runtime = duration_seconds
            fps = worker_stats['frames_processed'] / runtime
            logger.info(f"Average FPS: {fps:.2f}")
            if 5 <= fps <= 10:
                logger.info("✅ FPS in target range (5-10)")
            else:
                logger.warning(f"⚠️  FPS outside target range: {fps:.2f}")
                criteria_passed = False

            # Check latency
            if 'latency_stats' in worker_stats:
                p95_latency = worker_stats['latency_stats']['p95']
                logger.info(f"P95 Latency: {p95_latency:.1f}ms")
                if p95_latency < 100:
                    logger.info("✅ Latency < 100ms")
                else:
                    logger.warning(f"⚠️  Latency exceeds 100ms: {p95_latency:.1f}ms")
                    criteria_passed = False

    # Check if process completed successfully
    if worker_process.returncode == 0:
        logger.info("✅ Worker completed successfully")
    else:
        logger.error(f"❌ Worker exited with code {worker_process.returncode}")
        criteria_passed = False

    if criteria_passed:
        logger.info("\n✅ All acceptance criteria PASSED")
    else:
        logger.error("\n❌ Some acceptance criteria FAILED")

    return criteria_passed


if __name__ == '__main__':
    import sys
    from pathlib import Path

    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    success = run_performance_test(duration)
    sys.exit(0 if success else 1)
```

## Testing Checklist

### 1. Setup go2rtc with Test Video
```bash
# Place your test video as test_video.mp4
cp /path/to/your/video.mp4 test_video.mp4

# Start go2rtc
./setup_go2rtc.sh

# Verify go2rtc is running
curl http://localhost:1984/api/streams

# Test RTSP stream
ffplay rtsp://localhost:8554/test_video
```

### 2. Basic Functionality Test
```bash
# Run worker for 10 minutes
timeout 600 python worker_basic.py

# Check output
ls -la detections/
cat detections/worker_stats.json | jq .
```

### 3. Stability Test
```bash
# Run for 1 hour
python test_performance.py 3600

# Check performance report
cat performance_report.json | jq .
```

### 4. Graceful Shutdown
```bash
# Start worker
python worker_basic.py &
WORKER_PID=$!

# Wait 30 seconds
sleep 30

# Send SIGTERM
kill -TERM $WORKER_PID

# Check it shutdown cleanly
wait $WORKER_PID
echo "Exit code: $?"
```

### 5. Multiple Stream Test (Optional)
```bash
# Add multiple streams to go2rtc.yaml
# Then test with different stream names
STREAM_NAME=camera_1 python worker_basic.py
```

## Success Criteria Verification

### ✅ Process 1 RTSP stream for 10 minutes without crashes
- Run: `python test_performance.py 600`
- Check: Exit code should be 0

### ✅ Achieve stable 5-10 FPS processing
- Check: `detections/worker_stats.json` shows FPS in range
- Monitor: Watch logs for FPS reports

### ✅ Measure baseline inference latency < 100ms P95
- Check: Performance report shows latency_ms.p95 < 100
- Monitor: Detection JSON files include inference_time_ms

### ✅ Graceful shutdown works reliably
- Test: Send SIGTERM and verify clean exit
- Check: Final stats are saved

## Common Issues & Solutions

### Issue: go2rtc Connection Fails
```python
# Debug go2rtc connectivity
import requests
response = requests.get("http://localhost:1984/api/streams")
print(response.json())
```

### Issue: Stream Not Ready
- Check go2rtc logs for errors
- Verify video file exists and is readable
- Try direct ffplay test: `ffplay test_video.mp4`

### Issue: High CPU Usage
- Reduce max_fps in config
- Check model complexity
- Enable hardware acceleration in go2rtc

### Issue: Memory Leak
- Monitor with: `watch -n 1 'ps aux | grep worker_basic'`
- Check for growing detection files
- Add periodic garbage collection

## Next Steps

Once Phase 1 is complete and all checkpoints pass:

1. Document baseline performance metrics
2. Test with real RTSP cameras
3. Prepare for Phase 2 (multi-camera support)
4. Create Docker image with go2rtc included

## Resources

- [go2rtc Documentation](https://github.com/AlexxIT/go2rtc)
- [go2rtc API Reference](https://github.com/AlexxIT/go2rtc#api)
- [InferencePipeline Documentation](https://inference.roboflow.com/)
- [RTSP Testing Tools](https://github.com/AlexxIT/go2rtc#source-rtsp)
