# Configuration Guide

## Overview

The Somba Pipeline uses a comprehensive JSON-based configuration system that supports hot-reloading, per-camera settings, and flexible zone definitions. This guide covers all configuration options and their usage.

## Configuration Structure

The main configuration file defines a shard (worker instance) with multiple cameras and their individual settings.

### Root Configuration Schema

```json
{
  "runner_id": "string",
  "shard_id": "string", 
  "max_fps": integer,
  "sources": [
    {
      "camera_uuid": "string",
      "url": "string",
      "site_id": "string",
      "tenant_id": "string"
    }
  ],
  "amqp": {
    "host": "string",
    "ex_status": "string",
    "ex_detect": "string"
  },
  "cp": {
    "base_url": "string",
    "token": "string"
  },
  "telemetry": {
    "report_interval_seconds": integer
  },
  "cameras": {
    "camera_uuid": {
      "camera_uuid": "string",
      "zones": [...],
      "motion_gating": {...},
      "allow_labels": [...],
      "deny_labels": [...],
      "min_score": float,
      "zone_test": "string",
      "iou_threshold": float
    }
  }
}
```

## Configuration Fields

### Root Level Fields

#### `runner_id` (Required)
- **Type**: String
- **Description**: Unique identifier for this worker instance
- **Example**: `"runner-test-001"`
- **Usage**: Used for metrics, logging, and coordination

#### `shard_id` (Required)
- **Type**: String
- **Description**: Identifier for this shard (can run multiple shards)
- **Example**: `"shard-0"`
- **Usage**: Enables horizontal scaling across multiple workers

#### `max_fps` (Required)
- **Type**: Integer
- **Description**: Maximum frames per second to process per camera
- **Default**: `6`
- **Range**: `1-60`
- **Usage**: Performance control and resource management

#### `sources` (Required)
- **Type**: Array of camera source objects
- **Description**: Camera feeds to process
- **Example**: 
  ```json
  [
    {
      "camera_uuid": "cam-001",
      "url": "rtsp://127.0.0.1:8554/camera_001",
      "site_id": "site-A",
      "tenant_id": "tenant-01"
    }
  ]
  ```

**Source Object Fields:**
- `camera_uuid`: Unique identifier (must match cameras section)
- `url`: RTSP stream URL
- `site_id`: Site identifier for routing
- `tenant_id`: Tenant identifier for multi-tenancy

#### `amqp` (Required)
- **Type**: Object
- **Description**: RabbitMQ connection and exchange configuration
- **Example**:
  ```json
  {
    "host": "localhost",
    "ex_status": "status.topic",
    "ex_detect": "detections.topic"
  }
  ```

**AMQP Fields:**
- `host`: RabbitMQ server hostname
- `ex_status`: Exchange name for status events
- `ex_detect`: Exchange name for detection events

#### `cp` (Required)
- **Type**: Object
- **Description**: Control plane configuration
- **Example**:
  ```json
  {
    "base_url": "http://localhost:8000",
    "token": "jwt-token-here"
  }
  ```

#### `telemetry` (Optional)
- **Type**: Object
- **Description**: Telemetry and metrics reporting configuration
- **Example**:
  ```json
  {
    "report_interval_seconds": 5
  }
  ```

## Camera Configuration

Each camera in the `cameras` section can have individual configuration:

### Camera Object Schema

```json
{
  "camera_uuid": "cam-001",
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
  ],
  "motion_gating": {
    "enabled": true,
    "downscale": 0.5,
    "dilation_px": 6,
    "min_area_px": 150,
    "cooldown_frames": 2,
    "noise_floor": 12
  },
  "allow_labels": ["person", "car", "truck"],
  "deny_labels": [],
  "min_score": 0.30,
  "zone_test": "center",
  "iou_threshold": 0.10
}
```

### Zone Configuration

Zones define geographic areas for filtering and analysis.

#### Zone Object Fields

**Required Fields:**
- `zone_id`: Integer identifier (≥ 1, 0 reserved for whole frame)
- `name`: String name for the zone
- `kind`: `"include"` or `"exclude"`
- `priority`: Integer priority (higher wins for overlaps)
- `polygon`: Array of [x,y] coordinate pairs defining the polygon

**Optional Fields:**
- `allow_labels`: Array of allowed object labels
- `deny_labels`: Array of denied object labels
- `min_score`: Minimum confidence score (0.0-1.0)

#### Zone Examples

**Include Zone (Driveway Monitoring):**
```json
{
  "zone_id": 1,
  "name": "driveway",
  "kind": "include",
  "priority": 100,
  "polygon": [[100, 100], [700, 100], [700, 500], [200, 500]],
  "allow_labels": ["person", "truck"],
  "min_score": 0.25
}
```

**Exclude Zone (Neighbor Property):**
```json
{
  "zone_id": 2,
  "name": "neighbor_lawn",
  "kind": "exclude",
  "priority": 200,
  "polygon": [[0, 0], [700, 0], [700, 100], [0, 100]],
  "deny_labels": ["person", "car"]
}
```

### Motion Gating Configuration

Controls motion detection behavior for inference optimization.

#### Motion Gating Fields

**Basic Configuration:**
- `enabled`: Boolean to enable/disable motion gating
- `downscale`: Float downscale factor (0.1-1.0) for performance
- `dilation_px`: Integer pixels for dilation (0-20)
- `min_area_px`: Integer minimum motion area in pixels
- `cooldown_frames`: Integer frames for cooldown (0-10)
- `noise_floor`: Integer noise threshold (0-50)

**Advanced Configuration:**
- `roi_native`: Boolean for ROI-native processing (recommended: true)
- `adaptive_threshold_factor`: Float for adaptive thresholding (0.1-2.0)
- `min_area_mode`: `"px"` or `"roi_percent"`
- `min_area_roi_percent`: Float percentage for ROI-based minimum area

#### Motion Gating Examples

**Basic Motion Detection:**
```json
{
  "enabled": true,
  "downscale": 0.5,
  "dilation_px": 6,
  "min_area_px": 150,
  "cooldown_frames": 2,
  "noise_floor": 12
}
```

**ROI-Native Advanced:**
```json
{
  "enabled": true,
  "roi_native": true,
  "downscale": 0.5,
  "dilation_px": 6,
  "min_area_mode": "roi_percent",
  "min_area_roi_percent": 0.5,
  "adaptive_threshold_factor": 0.7,
  "cooldown_frames": 2,
  "noise_floor": 12
}
```

### Global Camera Settings

Settings that apply to the entire camera feed:

#### `allow_labels` (Optional)
- **Type**: Array of strings
- **Description**: Global allow list for object labels
- **Example**: `["person", "car", "truck"]`
- **Usage**: Only objects with these labels will be processed

#### `deny_labels` (Optional)
- **Type**: Array of strings
- **Description**: Global deny list for object labels
- **Example**: `["bird", "cat"]`
- **Usage**: Objects with these labels will be filtered out

#### `min_score` (Optional)
- **Type**: Float (0.0-1.0)
- **Description**: Global minimum confidence score
- **Default**: `0.30`
- **Usage**: Detections below this score are filtered out

#### `zone_test` (Optional)
- **Type**: String
- **Values**: `"center"` or `"center+iou"`
- **Default**: `"center"`
- **Description**: Method for determining zone membership

#### `iou_threshold` (Optional)
- **Type**: Float (0.0-1.0)
- **Description**: IoU threshold for center+iou testing
- **Default**: `0.10`
- **Usage**: Minimum IoU for zone membership when using center+iou

## Complete Configuration Example

```json
{
  "runner_id": "runner-test-001",
  "shard_id": "shard-0",
  "max_fps": 15,
  "sources": [
    {
      "camera_uuid": "cam-001",
      "url": "rtsp://127.0.0.1:8554/camera_001",
      "site_id": "site-A",
      "tenant_id": "tenant-01"
    },
    {
      "camera_uuid": "cam-002",
      "url": "rtsp://127.0.0.1:8554/camera_002",
      "site_id": "site-A",
      "tenant_id": "tenant-01"
    },
    {
      "camera_uuid": "cam-003",
      "url": "rtsp://127.0.0.1:8554/camera_003",
      "site_id": "site-B",
      "tenant_id": "tenant-01"
    }
  ],
  "amqp": {
    "host": "localhost",
    "ex_status": "status.topic",
    "ex_detect": "detections.topic"
  },
  "cp": {
    "base_url": "http://localhost:8000",
    "token": "jwt-token-here"
  },
  "telemetry": {
    "report_interval_seconds": 5
  },
  "cameras": {
    "cam-001": {
      "camera_uuid": "cam-001",
      "zones": [
        {
          "zone_id": 1,
          "name": "driveway",
          "kind": "include",
          "priority": 100,
          "polygon": [[100, 100], [700, 100], [700, 500], [200, 500]],
          "allow_labels": ["person", "truck"],
          "min_score": 0.25
        },
        {
          "zone_id": 2,
          "name": "neighbor_lawn",
          "kind": "exclude",
          "priority": 200,
          "polygon": [[0, 0], [700, 0], [700, 100], [0, 100]],
          "deny_labels": ["person", "car"]
        }
      ],
      "motion_gating": {
        "enabled": true,
        "roi_native": true,
        "downscale": 0.5,
        "dilation_px": 6,
        "min_area_mode": "roi_percent",
        "min_area_roi_percent": 0.5,
        "adaptive_threshold_factor": 0.7,
        "cooldown_frames": 2,
        "noise_floor": 12
      },
      "allow_labels": ["person", "car", "truck"],
      "deny_labels": [],
      "min_score": 0.30,
      "zone_test": "center",
      "iou_threshold": 0.10
    },
    "cam-002": {
      "camera_uuid": "cam-002",
      "zones": [
        {
          "zone_id": 1,
          "name": "area_persiapan",
          "kind": "include",
          "priority": 100,
          "polygon": [[250, 0], [500, 0], [600, 200], [600, 480], [100, 480]],
          "allow_labels": ["person"],
          "min_score": 0.3
        }
      ],
      "motion_gating": {
        "enabled": false
      },
      "allow_labels": ["person"],
      "deny_labels": [],
      "min_score": 0.25,
      "zone_test": "center+iou",
      "iou_threshold": 0.15
    },
    "cam-003": {
      "camera_uuid": "cam-003",
      "zones": [
        {
          "zone_id": 1,
          "name": "area_ompreng",
          "kind": "include",
          "priority": 100,
          "polygon": [[300, 0], [400, 0], [700, 500], [150, 500]],
          "allow_labels": ["person"],
          "min_score": 0.3
        }
      ],
      "motion_gating": {
        "enabled": false
      },
      "allow_labels": null,
      "deny_labels": [],
      "min_score": 0.20,
      "zone_test": "center",
      "iou_threshold": 0.10
    }
  }
}
```

## Configuration Management

### Hot Reload

The system automatically detects configuration changes and reloads without service interruption:

1. **File Watching**: Monitors configuration file for changes
2. **Hash Comparison**: Uses SHA256 to detect actual changes
3. **Zone Updates**: Rebuilds zone masks and attributors
4. **Motion Updates**: Recreates motion detectors with new settings
5. **Zero Downtime**: Changes applied without stopping processing

### Environment Variables

Several settings can be controlled via environment variables:

- `DEBUG_FPS`: Target FPS for debug streams (default: 8)
- `PROMETHEUS_PORT`: Port for metrics server (default: 9108)
- `HEALTH_PORT`: Port for health checks (default: 8080)

### Configuration Validation

The system validates configuration on load:

1. **Schema Validation**: Checks field types and required fields
2. **Polygon Validation**: Ensures polygons are valid and closed
3. **Zone Priority**: Validates priority values are unique
4. **URL Validation**: Checks RTSP URLs are properly formatted
5. **Score Ranges**: Validates confidence scores are 0.0-1.0

### Best Practices

1. **Zone Design**: Keep zones simple and non-overlapping when possible
2. **Motion Settings**: Start with conservative settings and adjust
3. **Performance**: Monitor CPU usage and adjust FPS/downscale accordingly
4. **Label Filtering**: Use specific allow lists to reduce false positives
5. **Testing**: Test new configurations in development first

### Troubleshooting

**Common Issues:**

1. **Motion Not Detected**: Increase `min_area_px` or check `roi_native` setting
2. **Too Many False Positives**: Adjust `adaptive_threshold_factor` or `noise_floor`
3. **Zone Assignment Issues**: Verify polygon coordinates and `zone_test` method
4. **Performance Problems**: Reduce `max_fps` or increase `downscale` factor

**Debug Configuration:**

Enable debug logging to see configuration processing:

```bash
export PYTHONPATH=. python -m somba_pipeline.worker config.json --debug
```

This comprehensive configuration system provides flexible control over all aspects of the video analysis pipeline while maintaining performance and reliability.