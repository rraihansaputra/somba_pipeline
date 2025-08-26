# Somba Pipeline - Iterative Implementation Roadmap

## Executive Summary

This roadmap breaks down the Somba livestream processing platform into 7 iterative phases, each building on the previous to deliver a production-ready system. Each phase includes specific checkpoints to validate critical functionality before proceeding.

## Critical Path Analysis

### Core Dependencies
1. **InferencePipeline** (existing) → foundation for all video processing
2. **Lease System** → ensures exactly-one processing guarantee
3. **Manager/Worker** → orchestration and execution
4. **Observability** → operational visibility
5. **Optimization** → performance and cost efficiency

---

## Phase 1: Foundation & Single-Stream MVP (Week 1-2)

### Goal
Prove the core InferencePipeline can process a single RTSP stream with basic monitoring

### Implementation
```
1. Basic Worker Process
   - Single InferencePipeline instance
   - Direct RTSP connection
   - Console logging for events
   - Graceful shutdown handling

2. Simple Detection Output
   - JSON file output (no RabbitMQ yet)
   - Basic frame metrics (FPS, latency)

3. Test Infrastructure
   - Mock RTSP server using OpenCV
   - Test video files as sources
```

### Checkpoints
- [ ] Process 1 RTSP stream for 10 minutes without crashes
- [ ] Achieve stable 5-10 FPS processing
- [ ] Measure baseline inference latency (< 100ms P95)
- [ ] Graceful shutdown works reliably

### Deliverables
- `worker_basic.py` - Single stream processor
- `test_rtsp_server.py` - Mock RTSP for testing
- Performance baseline metrics

---

## Phase 2: Multi-Camera Sharding (Week 2-3)

### Goal
Prove Worker can handle multiple cameras (shard) with proper buffering strategies

### Implementation
```
1. Multi-Source Worker
   - Process 8-16 cameras per Worker
   - Implement BufferFillingStrategy (DROP_OLDEST)
   - Per-camera motion detection wrapper

2. Shard Configuration
   - JSON config loader
   - Camera assignment logic
   - Source multiplexing

3. Resource Management
   - Memory monitoring
   - CPU/GPU utilization tracking
   - Adaptive frame dropping
```

### Checkpoints
- [ ] Process 12 cameras simultaneously for 1 hour
- [ ] Memory usage < 2GB per Worker
- [ ] No frame buffer overflow
- [ ] Motion detection reduces inference by 50%+

### Deliverables
- `worker_shard.py` - Multi-camera processor
- `motion_detector.py` - Motion-triggered inference
- Shard configuration schema

---

## Phase 3: Lease System & Control Plane (Week 3-4)

### Goal
Implement distributed ownership with exactly-one processing guarantee

### Implementation
```
1. Minimal Control Plane API
   - Django REST framework
   - PostgreSQL for leases
   - Camera CRUD operations
   - Lease acquire/renew/release

2. Lease Client
   - HTTP client with retry logic
   - 2s renew / 10s TTL timing
   - Automatic lease recovery

3. Worker Integration
   - Lease validation before processing
   - Automatic shutdown on lease loss
   - Reconnection logic
```

### Checkpoints
- [ ] No duplicate processing across 3 Workers
- [ ] Takeover time < 10s on Worker failure
- [ ] Lease storms handled (100+ simultaneous requests)
- [ ] Database performs at 1000+ lease ops/second

### Deliverables
- `control_plane/` - Django API application
- `lease_client.py` - Lease management client
- Database schema and migrations

---

## Phase 4: Manager Process & Orchestration (Week 4-5)

### Goal
Implement Manager for Worker lifecycle and Blue/Green deployments

### Implementation
```
1. Manager Core Loop
   - Camera discovery
   - Lease acquisition up to capacity
   - Shard planning algorithm
   - Worker subprocess management

2. Blue/Green Deployment
   - Spawn new Workers before draining old
   - Health checking (/ready endpoint)
   - Graceful drain coordination

3. Token Bucket
   - Per-site connection throttling
   - Token consumption API
   - Backoff/jitter implementation
```

### Checkpoints
- [ ] Blue/Green switch < 5s gap
- [ ] No RTSP connection storms (limited to token rate)
- [ ] Manager recovers from Worker crashes
- [ ] Cordon/drain operations work correctly

### Deliverables
- `manager.py` - Orchestration process
- `shard_planner.py` - Camera assignment logic
- Worker control HTTP API

---

## Phase 5: Event System & Observability (Week 5-6)

### Goal
Full observability with RabbitMQ events and Prometheus metrics

### Implementation
```
1. RabbitMQ Integration
   - Detection event publishing
   - Stream status updates
   - Runner heartbeats
   - Publisher confirms

2. Prometheus Metrics
   - Worker metrics (port 9108)
   - Manager metrics (port 9107)
   - Custom watchdog implementation

3. Structured Logging
   - JSON log formatting
   - Log aggregation ready
   - Correlation IDs
```

### Checkpoints
- [ ] 100% detection delivery (no message loss)
- [ ] Prometheus scraping stable
- [ ] Alert rules trigger correctly
- [ ] 1Hz heartbeat consistency

### Deliverables
- `event_publisher.py` - RabbitMQ client
- `metrics.py` - Prometheus exporters
- Grafana dashboard templates
- Alert rule definitions

---

## Phase 6: Production Hardening (Week 6-7)

### Goal
Scale testing and reliability improvements

### Implementation
```
1. go2rtc Integration (Optional)
   - Local restreaming setup
   - Stream management API
   - Connection pooling

2. Resilience Features
   - Circuit breakers
   - Exponential backoff
   - Connection pool limits
   - Memory leak detection

3. Performance Optimization
   - GPU batch processing
   - Frame pre-processing pipeline
   - Inference caching
```

### Checkpoints
- [ ] 1000 cameras across 50 sites stable
- [ ] No memory leaks over 72 hours
- [ ] Recovery from all failure modes
- [ ] P95 latency < 150ms at scale

### Deliverables
- `go2rtc_manager.py` - Restream orchestration
- Performance tuning guide
- Failure recovery playbooks

---

## Phase 7: Advanced Features (Week 7-8)

### Goal
Implement cost optimization and advanced capabilities

### Implementation
```
1. Motion-Triggered Inference
   - Full motion detection integration
   - ROI-based detection
   - Adaptive sensitivity
   - Statistics tracking

2. Dynamic Scaling
   - Auto-scaling based on load
   - Predictive capacity planning
   - Cost optimization algorithms

3. Multi-Model Support
   - Workflow integration
   - Model switching
   - A/B testing framework
```

### Checkpoints
- [ ] 70%+ reduction in inference costs
- [ ] ROI detection accuracy > 95%
- [ ] Auto-scaling responds in < 30s
- [ ] Multiple models run simultaneously

### Deliverables
- Enhanced motion detection system
- Auto-scaling controller
- Cost analysis dashboard

---

## Testing Strategy

### Unit Tests (Continuous)
- Lease client operations
- Shard planning logic
- Motion detection algorithms
- Buffer management

### Integration Tests (Per Phase)
- Manager + Worker interaction
- Control Plane API responses
- RabbitMQ message flow
- Prometheus metric export

### Chaos Engineering (Phase 6+)
- Random Worker kills
- Network partitions
- RTSP disconnections
- Database failures
- Message queue outages

### Load Testing (Phase 6+)
- 1000+ camera simulation
- Sustained 24-hour runs
- Peak load scenarios
- Failover performance

---

## Risk Mitigation

### Technical Risks
1. **RTSP Connection Stability**
   - Mitigation: Implement go2rtc buffering
   - Fallback: Direct connection with retry

2. **Memory Leaks in Long-Running Processes**
   - Mitigation: Periodic Worker recycling
   - Monitoring: Memory profiling tools

3. **Database Performance at Scale**
   - Mitigation: Connection pooling, read replicas
   - Fallback: Redis cache layer

### Operational Risks
1. **Complex Deployment**
   - Mitigation: Docker/K8s packaging
   - Documentation: Runbooks for each component

2. **Debugging Distributed System**
   - Mitigation: Correlation IDs, distributed tracing
   - Tools: Jaeger/OpenTelemetry integration

---

## Success Metrics

### Phase 1-2 (Foundation)
- Single Worker stability
- Basic multi-camera support

### Phase 3-4 (Core Platform)
- Exactly-one processing guarantee
- Sub-10s failover
- Blue/Green deployments

### Phase 5-6 (Production Ready)
- Full observability
- 1000+ camera scale
- 99.9% uptime

### Phase 7 (Optimized)
- 70%+ cost reduction via motion detection
- Auto-scaling efficiency
- Multi-model flexibility

---

## Team Resources

### Required Skills
- Python (asyncio, multiprocessing)
- Django REST Framework
- PostgreSQL
- RabbitMQ
- Prometheus/Grafana
- Docker/Kubernetes
- OpenCV/Video Processing

### Estimated Effort
- 2 Senior Engineers: 8 weeks
- 1 DevOps Engineer: 4 weeks (Phase 5-7)
- 1 QA Engineer: 4 weeks (Phase 4-7)

---

## Go/No-Go Decision Points

### After Phase 2
- Can we process multiple cameras reliably?
- Is the motion detection providing value?

### After Phase 4
- Is the Manager/Worker architecture sound?
- Can we handle configuration changes smoothly?

### After Phase 6
- Does the system scale to target load?
- Are operational costs acceptable?

---

## Appendix: Quick Start Commands

```bash
# Phase 1: Run basic Worker
python worker_basic.py --rtsp-url rtsp://localhost:8554/test

# Phase 2: Run multi-camera Worker
python worker_shard.py --config shard_config.json

# Phase 3: Start Control Plane
python manage.py runserver
python worker_shard.py --control-plane http://localhost:8000

# Phase 4: Run Manager
python manager.py --capacity 40 --shard-size 12

# Phase 5: Start with full observability
docker-compose up -d rabbitmq prometheus grafana
python manager.py --metrics-port 9107

# Phase 6: Scale test
python load_test.py --cameras 1000 --duration 3600

# Phase 7: Enable motion detection
python manager.py --enable-motion --motion-threshold 150
