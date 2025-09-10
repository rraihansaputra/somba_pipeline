# API Reference

## Overview

The Somba Pipeline provides HTTP APIs for health monitoring, debug streaming control, and camera management. This document describes all available endpoints and their usage.

## Base URL

All API endpoints are hosted on the health port (default: 8080):

```
http://localhost:8080
```

## Endpoints

### Health Check

#### GET `/health`

**Description**: Basic health check endpoint

**Response**:
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

**Status Codes**:
- `200 OK`: System is healthy
- `503 Service Unavailable`: System is unhealthy or starting up

**Example**:
```bash
curl http://localhost:8080/health
```

### Metrics

#### GET `/metrics`

**Description**: Prometheus metrics endpoint

**Response**: Text format Prometheus metrics

**Available Metrics**:

**System Metrics:**
- `somba_worker_uptime_seconds`: Worker uptime in seconds
- `somba_worker_config_reloads_total`: Number of configuration reloads
- `somba_worker_cameras_connected`: Number of connected cameras

**Camera Metrics:**
- `somba_camera_fps_current`: Current FPS per camera
- `somba_camera_frames_processed_total`: Total frames processed per camera
- `somba_camera_detections_total`: Total detections per camera
- `somba_camera_motion_events_total`: Motion events per camera
- `somba_camera_connection_status`: Camera connection status (1=connected, 0=disconnected)

**Processing Metrics:**
- `somba_processing_latency_ms`: Processing latency in milliseconds
- `somba_inference_latency_ms`: AI inference latency in milliseconds
- `somba_motion_detection_latency_ms`: Motion detection latency in milliseconds
- `somba_zone_filtering_latency_ms`: Zone filtering latency in milliseconds

**Event Metrics:**
- `somba_events_published_total`: Total events published
- `somba_events_failed_total`: Failed event publications
- `somba_events_detection_total`: Detection events published
- `somba_events_status_total`: Status events published

**Example**:
```bash
curl http://localhost:8080/metrics
```

### Debug Streaming Control

#### POST `/debug/{camera_uuid}/start`

**Description**: Start debug streaming for a specific camera

**Path Parameters**:
- `camera_uuid`: Unique identifier for the camera

**Optional Query Parameters**:
- `fps`: Target FPS for the stream (default: 8, range: 1-30)
- `quality`: JPEG quality (default: 80, range: 1-100)
- `title`: Stream title for display (default: camera UUID)

**Response**:
```json
{
  "status": "started",
  "stream_url": "http://127.0.0.1:8089/cam-001.mjpg",
  "camera_uuid": "cam-001",
  "fps": 8,
  "quality": 80,
  "title": "cam-001"
}
```

**Status Codes**:
- `200 OK`: Debug stream started successfully
- `400 Bad Request`: Invalid camera UUID or parameters
- `404 Not Found`: Camera not found
- `409 Conflict`: Stream already active for this camera

**Example**:
```bash
curl -X POST "http://localhost:8080/debug/cam-003/start"
curl -X POST "http://localhost:8080/debug/cam-003/start?fps=10&quality=90&title=Driveway%20Camera"
```

#### POST `/debug/{camera_uuid}/stop`

**Description**: Stop debug streaming for a specific camera

**Path Parameters**:
- `camera_uuid`: Unique identifier for the camera

**Response**:
```json
{
  "status": "stopped",
  "camera_uuid": "cam-001",
  "frames_sent": 1250,
  "duration_seconds": 156.7
}
```

**Status Codes**:
- `200 OK`: Debug stream stopped successfully
- `404 Not Found`: No active stream for this camera

**Example**:
```bash
curl -X POST "http://localhost:8080/debug/cam-003/stop"
```

#### GET `/debug/{camera_uuid}/status`

**Description**: Get debug streaming status for a specific camera

**Path Parameters**:
- `camera_uuid`: Unique identifier for the camera

**Response**:
```json
{
  "camera_uuid": "cam-001",
  "streaming": true,
  "stream_url": "http://127.0.0.1:8089/cam-001.mjpg",
  "fps": 8,
  "quality": 80,
  "title": "cam-001",
  "frames_sent": 1250,
  "start_time": "2024-01-15T10:30:00Z",
  "duration_seconds": 156.7,
  "clients_connected": 3
}
```

**Status Codes**:
- `200 OK`: Status retrieved successfully
- `404 Not Found`: Camera not found

**Example**:
```bash
curl "http://localhost:8080/debug/cam-003/status"
```

#### GET `/debug/streams`

**Description**: List all active debug streams

**Response**:
```json
{
  "streams": [
    {
      "camera_uuid": "cam-001",
      "stream_url": "http://127.0.0.1:8089/cam-001.mjpg",
      "fps": 8,
      "quality": 80,
      "title": "cam-001",
      "frames_sent": 1250,
      "start_time": "2024-01-15T10:30:00Z",
      "clients_connected": 3
    },
    {
      "camera_uuid": "cam-003",
      "stream_url": "http://127.0.0.1:8089/cam-003.mjpg",
      "fps": 10,
      "quality": 90,
      "title": "Driveway Camera",
      "frames_sent": 850,
      "start_time": "2024-01-15T10:45:00Z",
      "clients_connected": 1
    }
  ],
  "total_streams": 2
}
```

**Example**:
```bash
curl "http://localhost:8080/debug/streams"
```

### Camera Management

#### GET `/cameras`

**Description**: List all configured cameras with their status

**Response**:
```json
{
  "cameras": [
    {
      "camera_uuid": "cam-001",
      "status": "connected",
      "url": "rtsp://127.0.0.1:8554/camera_001",
      "site_id": "site-A",
      "tenant_id": "tenant-01",
      "fps_current": 6.2,
      "frames_processed": 45678,
      "last_frame_time": "2024-01-15T10:30:00Z",
      "zones_count": 2,
      "motion_enabled": true,
      "debug_streaming": false
    },
    {
      "camera_uuid": "cam-003",
      "status": "connected",
      "url": "rtsp://127.0.0.1:8554/camera_003",
      "site_id": "site-B",
      "tenant_id": "tenant-01",
      "fps_current": 5.8,
      "frames_processed": 23456,
      "last_frame_time": "2024-01-15T10:30:00Z",
      "zones_count": 1,
      "motion_enabled": false,
      "debug_streaming": true
    }
  ],
  "total_cameras": 2
}
```

**Example**:
```bash
curl "http://localhost:8080/cameras"
```

#### GET `/cameras/{camera_uuid}`

**Description**: Get detailed information about a specific camera

**Path Parameters**:
- `camera_uuid`: Unique identifier for the camera

**Response**:
```json
{
  "camera_uuid": "cam-001",
  "status": "connected",
  "url": "rtsp://127.0.0.1:8554/camera_001",
  "site_id": "site-A",
  "tenant_id": "tenant-01",
  "fps_current": 6.2,
  "frames_processed": 45678,
  "last_frame_time": "2024-01-15T10:30:00Z",
  "zones_count": 2,
  "motion_enabled": true,
  "debug_streaming": false,
  "configuration": {
    "allow_labels": ["person", "car", "truck"],
    "deny_labels": [],
    "min_score": 0.30,
    "zone_test": "center",
    "iou_threshold": 0.10,
    "motion_gating": {
      "enabled": true,
      "downscale": 0.5,
      "dilation_px": 6,
      "min_area_px": 150,
      "cooldown_frames": 2,
      "noise_floor": 12
    },
    "zones": [
      {
        "zone_id": 1,
        "name": "driveway",
        "kind": "include",
        "priority": 100,
        "polygon": [[100, 100], [700, 100], [700, 500], [200, 500]],
        "allow_labels": ["person", "truck"],
        "min_score": 0.25
      }
    ]
  },
  "statistics": {
    "frames_processed": 45678,
    "detections_total": 1234,
    "motion_events_total": 567,
    "avg_processing_time_ms": 45.2,
    "last_detection_time": "2024-01-15T10:29:45Z"
  }
}
```

**Status Codes**:
- `200 OK`: Camera information retrieved successfully
- `404 Not Found`: Camera not found

**Example**:
```bash
curl "http://localhost:8080/cameras/cam-001"
```

### Configuration Management

#### GET `/config`

**Description**: Get current configuration

**Response**: Current configuration JSON with metadata

```json
{
  "config": {
    "runner_id": "runner-test-001",
    "shard_id": "shard-0",
    "max_fps": 15,
    "sources": [...],
    "amqp": {...},
    "cp": {...},
    "telemetry": {...},
    "cameras": {...}
  },
  "metadata": {
    "config_file": "/path/to/config.json",
    "last_modified": "2024-01-15T10:30:00Z",
    "config_hash": "abc123...",
    "reload_count": 3,
    "last_reload": "2024-01-15T09:45:00Z"
  }
}
```

**Example**:
```bash
curl "http://localhost:8080/config"
```

#### POST `/config/reload`

**Description**: Trigger configuration reload

**Response**:
```json
{
  "status": "reloaded",
  "timestamp": "2024-01-15T10:30:00Z",
  "config_hash": "def456...",
  "changes_detected": true,
  "cameras_updated": ["cam-001", "cam-003"],
  "zones_updated": true
}
```

**Status Codes**:
- `200 OK`: Configuration reloaded successfully
- `204 No Content`: No changes detected
- `500 Internal Server Error**: Configuration reload failed

**Example**:
```bash
curl -X POST "http://localhost:8080/config/reload"
```

### System Information

#### GET `/info`

**Description**: Get system information and version details

**Response**:
```json
{
  "version": "1.0.0",
  "build_info": {
    "build_time": "2024-01-15T10:30:00Z",
    "git_commit": "abc123...",
    "python_version": "3.11.0"
  },
  "system": {
    "hostname": "server-01",
    "uptime_seconds": 86400,
    "start_time": "2024-01-14T10:30:00Z",
    "platform": "Linux-5.15.0-x86_64"
  },
  "resources": {
    "cpu_usage_percent": 45.2,
    "memory_usage_mb": 512,
    "memory_usage_percent": 25.5,
    "disk_usage_mb": 1024,
    "disk_usage_percent": 10.2
  },
  "components": {
    "inference_pipeline": {
      "status": "running",
      "version": "0.1.0",
      "cameras_connected": 3
    },
    "rabbitmq": {
      "status": "connected",
      "host": "localhost",
      "exchanges": ["status.topic", "detections.topic"]
    },
    "debug_server": {
      "status": "running",
      "host": "127.0.0.1",
      "port": 8089,
      "active_streams": 1
    }
  }
}
```

**Example**:
```bash
curl "http://localhost:8080/info"
```

## Error Responses

All endpoints return standardized error responses:

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Additional error details",
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "unique-request-identifier"
  }
}
```

### Common Error Codes

- `INVALID_REQUEST`: Malformed request or invalid parameters
- `CAMERA_NOT_FOUND`: Requested camera does not exist
- `CAMERA_DISCONNECTED`: Camera is not connected
- `CONFIGURATION_ERROR`: Invalid configuration
- `STREAM_ACTIVE`: Debug stream already active
- `STREAM_NOT_FOUND`: No active debug stream
- `INTERNAL_ERROR`: Server-side error

### Example Error Response

```json
{
  "error": {
    "code": "CAMERA_NOT_FOUND",
    "message": "Camera 'cam-999' not found",
    "details": "The requested camera UUID does not exist in the configuration",
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

## HTTP Headers

### Request Headers

- `Content-Type`: `application/json` for POST requests
- `Authorization`: Bearer token for protected endpoints (if configured)
- `X-Request-ID`: Optional request ID for tracking

### Response Headers

- `Content-Type`: `application/json` for most responses
- `Content-Type`: `text/plain` for `/metrics` endpoint
- `X-Request-ID`: Echo of request ID if provided
- `X-Response-Time`: Response processing time in milliseconds

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Health checks**: No rate limiting
- **Metrics**: No rate limiting
- **Debug streaming**: 10 requests per minute per camera
- **Camera management**: 60 requests per minute
- **Configuration**: 5 requests per minute

Rate limit information is returned in headers:

```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 9
X-RateLimit-Reset: 1642246800
```

## WebSocket Support (Future)

The system may support WebSocket connections for real-time updates in future versions:

```
ws://localhost:8080/ws/camera/{camera_uuid}/status
ws://localhost:8080/ws/system/metrics
ws://localhost:8080/ws/events/stream
```

## Client Libraries

### Python Example

```python
import requests
import json

# Health check
response = requests.get("http://localhost:8080/health")
print(response.json())

# Start debug stream
response = requests.post("http://localhost:8080/debug/cam-001/start")
stream_info = response.json()
print(f"Stream URL: {stream_info['stream_url']}")

# Get camera status
response = requests.get("http://localhost:8080/cameras/cam-001")
camera_info = response.json()
print(f"Camera status: {camera_info['status']}")
```

### JavaScript Example

```javascript
// Start debug stream
async function startDebugStream(cameraUuid, fps = 8) {
  const response = await fetch(`http://localhost:8080/debug/${cameraUuid}/start?fps=${fps}`, {
    method: 'POST'
  });
  return await response.json();
}

// Get camera list
async function getCameras() {
  const response = await fetch('http://localhost:8080/cameras');
  return await response.json();
}

// Usage
startDebugStream('cam-001', 10).then(stream => {
  console.log('Stream URL:', stream.stream_url);
});
```

This comprehensive API provides full control over the Somba Pipeline system, enabling integration with monitoring tools, custom dashboards, and automated management systems.