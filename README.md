# Somba Pipeline Phase 2

Phase 2 Production Worker implementation with motion detection, zones, and zone attribution.

## Features

- **Motion-Gated Inference**: Skip inference when no motion detected in included zones
- **Zone Attribution**: Assign detections to zones with precedence rules
- **Production Worker**: Multi-camera shard processing with RabbitMQ and Prometheus
- **Schema v2 Events**: Detection and status events with zone metadata
- **Mock Control Plane**: For testing lease operations and camera configs

## Architecture

- **Motion Detection**: Per-camera motion detection with configurable thresholds
- **Zone System**: Include/exclude zones with priority-based assignment
- **Event Publishing**: RabbitMQ integration for detections and status
- **Metrics**: Prometheus metrics on port 9108
- **Health Endpoints**: HTTP endpoints for health checks and graceful shutdown

## Quick Start

1. Install dependencies:
```bash
uv sync
```

2. Run mock control plane:
```bash
uv run mock-control-plane
```

3. Start production worker:
```bash
uv run production-worker config/test_shard.json
```

4. Run tests:
```bash
uv run pytest
```

## Configuration

See `config/` directory for example configurations.

## Testing

- Unit tests: `pytest tests/unit/`
- Integration tests: `pytest tests/integration/`
- All tests: `pytest`

## Acceptance Criteria

✅ Skip logic works when no motion in included zones
✅ Zone mapping with precedence rules works correctly
✅ Per-zone label filters override global filters
✅ Schema v2 events include all required zone fields
✅ All Prometheus metrics are exported correctly
✅ Health endpoints respond per specification

```
# Install dependencies
uv sync

# Run mock control plane
uv run mock-control-plane

# Start production worker
uv run production-worker config/test_shard.json

# Run all tests
python scripts/run_tests.py
```
