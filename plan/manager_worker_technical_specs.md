# Livestream Processing Platform — Manager & Worker Technical Specification

## Background

We are building a multi-tenant video analytics platform for **livestream** sources (primarily RTSP) that must:

* keep **exactly one** processor per camera online at all times,
* minimize downtime on failures and config changes,
* scale elastically across heterogeneous compute (edge/cloud),
* expose clear **status, metrics, and events** for operations and downstream products.

This document specifies the **Manager** (runner control process) and the **Worker** (inference pipeline process) and how they interact with the broader system (Control Plane API, RabbitMQ, Prometheus, optional go2rtc). It is designed to be handed to an engineering team and implemented as-is.

> For model/runtime behavior, refer to your chosen inference engine’s docs (e.g., Roboflow Inference `InferencePipeline` and its watchdog) and go2rtc’s HTTP API for stream management.

---

## Scope

* Define architecture, responsibilities, and interfaces for **Manager** and **Worker**.
* Define **shards** (grouping of cameras per Worker) and **per-site token bucket** (connection throttling) semantics.
* Specify all **inputs, outputs, metrics, events, and APIs**.
* Provide **acceptance criteria** and **tests**.

**Non-goals:** Implement the underlying model code, UI dashboards, or full Control Plane. Those are external but referenced.

---

## Goals (SLO-driven)

* **Exact-one processing:** At most one active Worker processes each `camera_uuid` at any time.
* **Low takeover time:** On runner failure, a different runner takes over a camera within **≤ 10s** (P95).
* **Zero/minimal gap on reconfig:** When cameras are added/removed or resharded, frames resume within **≤ 5s** (P95).
* **Backpressure safety:** No RTSP/NVR dial storms; per-site connect rate is limited deterministically.
* **Operability:** Real-time status (≤1s heartbeat), Prometheus metrics, structured logs, and discrete events.
* **Security:** Principle of least privilege; no leaking camera credentials.

---

## Key Concepts & Glossary

* **Site (Location):** A client location grouping cameras (e.g., a store). Identified by `site_id`.
* **Camera:** An RTSP source identified by stable `camera_uuid`.
* **Runner:** A host/pod instance running one **Manager** process (and multiple **Workers**).
* **Shard:** A **Worker assignment**: the set of cameras a single Worker processes concurrently (typically 8–16 cameras per shard).
* **Lease:** A short-TTL record granting a runner ownership of a camera (`stream_leases`, authoritative in Control Plane DB/API).
* **Token bucket:** A per-site rate limiter controlling **first connections** to avoid NVR overload.

---

## High-Level Architecture

```
                +-------------------+          +------------------+
   Operators -> |   Control Plane   | <------> | Postgres (CP DB) |
                |  (Django + DRF)   |          +------------------+
                +---------+---------+
                          |
            (APIs & events: camera CRUD, leases, budgets)
                          v
+---------------------+         +---------------------+         +--------------------+
|   Runner A (pod)    |         |   Runner B (pod)    |   ...   |   Runner N (pod)   |
|  +---------------+  |         |  +---------------+  |         |  +---------------+ |
|  |   Manager     |  |         |  |   Manager     |  |         |  |   Manager     | |
|  |  (this spec)  |  |         |  |  (this spec)  |  |         |  |  (this spec)  | |
|  +-----+---+-----+  |         |  +-----+---+-----+  |         |  +-----+---+-----+ |
|        |   |        |         |        |   |        |         |        |   |       |
|    spawn   |HTTP    |         |    spawn   |HTTP    |         |    spawn   |HTTP   |
|        v   |ctrl    |         |        v   |ctrl    |         |        v   |ctrl   |
|  +----------v----+  |         |  +----------v----+  |         |  +----------v----+ |
|  |   Worker(s)   |  |         |  |   Worker(s)   |  |         |  |   Worker(s)   | |
|  |  (shards w/   |  |         |  |  (shards w/   |  |         |  |  (shards w/   | |
|  | InferencePipe)|  |         |  | InferencePipe)|  |         |  | InferencePipe)| |
|  +---------------+  |         |  +---------------+  |         |  +---------------+ |
|         ^           |         |         ^           |         |         ^          |
|         |           |         |         |           |         |         |          |
|  Prometheus + AMQP  |         |  Prometheus + AMQP  |         |  Prometheus+AMQP  |
+---------+-----------+         +----------+----------+         +----------+---------+
          |                                 |                               |
          v                                 v                               v
  Prometheus (scrape)            RabbitMQ (events & detections)       (Optional) go2rtc per runner
```

---

## Concerns & How We Address Them

| Concern                    | Solution                                                                               |
| -------------------------- | -------------------------------------------------------------------------------------- |
| Duplicate processing       | **Short-TTL camera leases** (authoritative) + Workers refuse to run if lease not held. |
| Quick failover             | Lease renew every **2s**, TTL **8–10s**; takeover ≤ TTL + RTSP reconnect (1–3s).       |
| Config churn without gaps  | **Blue/Green** Worker rollout per runner; new Worker ready before old drains.          |
| NVR overload on cold start | **Per-site token bucket** + jitter between first dials; optional go2rtc fronting.      |
| Multi-site scaling         | **Global pool** of runners; no per-site K8s objects; any runner can acquire leases.    |
| Observability              | 1Hz worker heartbeat (AMQP), 5s stream summaries, Prometheus metrics, structured logs. |
| Security                   | JWT on CP APIs, secrets not logged, go2rtc bound to loopback, least-priv DB roles.     |

---

## Manager — Specification

### Responsibility

* Discover desired cameras and **acquire/renew** camera leases up to `capacity_streams`.
* Partition owned cameras into **shards** and **spawn/manage Workers** per shard.
* Apply **Blue/Green** changes when assignments change.
* Enforce **per-site token bucket** before first connecting new cameras.
* Emit **runner heartbeats** and **runner metrics**.
* Obey **cordon/drain** flags.

### Inputs

* **Control Plane APIs** (HTTP, JWT):

  * List cameras (`GET /v1/cameras`)
  * Acquire/Renew/Release lease (`POST /v1/leases/camera/*`)
  * Consume token (`POST /v1/sites/{site_id}/budget/consume`)
  * Runner cordon/drain flags (`GET /v1/runners/{id}`)
* **Change events** (AMQP or periodic poll): `camera.created|updated|deleted`.
* **Env/config:** capacity, shard size, renew & TTL intervals.

### Outputs

* **Workers** (child processes) with explicit shard config.
* **Runner heartbeat** (AMQP) every 1s:

  ```json
  {
    "type": "runner.heartbeat",
    "runner_id": "<runner>",
    "streams_owned": 37,
    "shards": 3,
    "cpu_pct": 41.2,
    "mem_mb": 1024,
    "ts": "2025-08-21T11:00:00Z"
  }
  ```
* **Prometheus metrics** (runner level), see Metrics section.
* **Logs** (structured JSON).

### Behavior (Control Loop)

1. **Discover desired**: fetch enabled cameras (optionally filtered by tenant/site).
2. **Acquire leases** for unowned cameras until `capacity_streams` reached.

   * For each new camera: if **token bucket** enforced → consume per-site token; if 409, backoff/jitter.
3. **Shard planning**: partition owned cameras into shards of size `target_streams_per_shard` (e.g., 8–16).

   * Keep shard stickiness where possible to minimize re-connects.
4. **Reconcile Workers**:

   * If planned shards ≠ running shards → **Blue/Green**:

     * Spawn **new Worker(s)** with new shard configs.
     * Wait until new Workers `/ready` (see Worker readiness).
     * Issue `/drain` to old Workers; wait for exit; reap.
5. **Renew leases** every **2s**; **TTL 8–10s**.
6. **Cordon/drain**:

   * `cordoned=true` → skip new lease acquisition.
   * `drain=soft` → release leases in batches (N every X seconds) and apply Blue/Green to shrink to 0.
   * `drain=hard` → immediately release all leases; terminate Workers.

### Shard Semantics

* **Definition:** A shard is the set of cameras assigned to a single Worker process (and single inference pipeline instance).
* **Size:** `target_streams_per_shard` (configurable). Tune by GPU/CPU capacity and model latency.
* **Placement:** Greedy bin-pack by site (try to keep cameras of the same site collocated) to make token gating predictable and simplify troubleshooting.
* **Blue/Green:** New shard set is brought up fully, then old shards are drained. This is required because many inference pipelines cannot hot-add sources.

### Token Bucket (Per Site)

* **Purpose:** Limit concurrent **first connections** to cameras of the same site to avoid NVR overload.
* **State:** `site_start_budget(site_id, tokens, capacity, refill_rate/min, updated_at)`.
* **Consume:** Manager calls `POST /v1/sites/{site_id}/budget/consume` per *new* camera connect.

  * 200 → proceed; 409 → no tokens, backoff with jitter, retry.
* **Refill:** CP cron job refills `tokens = min(capacity, tokens + refill_rate)` every minute.
* **Note:** Renewal and ongoing streaming do **not** consume tokens; only *first connect* after unowned → owned transition.

### Manager Local API (required for orchestration)

Workers run as subprocesses, controlled over HTTP by the Manager.

* **Spawn contract:** Command + JSON config file path, or `--config-json` arg.
* **Worker control endpoints (local)**:

  * `GET /healthz` → 200 if process alive and pipeline thread running.
  * `GET /ready` → 200 if **≥ readiness\_quorum %** of cameras are `STREAMING` (or `last_frame_age<3s`) within the last 15s.
  * `POST /drain` → graceful stop; return 202 immediately; Worker exits in ≤ `grace_timeout_s`.
  * `POST /terminate` → immediate stop (used only on hard drain).

> These endpoints are **local-only** (bind to `127.0.0.1`) and never exposed outside the host.

---

## Worker — Specification

### Responsibility

* Run one inference pipeline per process over its **shard** of cameras.
* Emit **detections** to RabbitMQ.
* Emit **stream status** and **errors** (edge-triggered + periodic summaries).
* Expose **Prometheus** metrics and **local health endpoints**.
* Respect **graceful drain** and terminate cleanly.

### Inputs

* **Shard config** (JSON), supplied at spawn:

  ```json
  {
    "runner_id": "runner-123",
    "shard_id": "shard-0",
    "max_fps": 6,
    "sources": [
      { "camera_uuid": "cam-1", "url": "rtsp://.../1", "site_id": "site-A", "tenant_id": "t-01" },
      { "camera_uuid": "cam-2", "url": "rtsp://.../2", "site_id": "site-A", "tenant_id": "t-01" }
    ],
    "amqp": { "host": "rabbit", "ex_status": "status.topic", "ex_detect": "detections.topic" },
    "cp":   { "base_url": "https://cp/api", "token": "<jwt>" },
    "telemetry": { "report_interval_seconds": 5 }
  }
  ```

  * If using **go2rtc**, `url` is a **local restream** (e.g., `rtsp://127.0.0.1:8554/<camera_uuid>`).

### Processing

* Initialize pipeline with `video_reference = [url...]`.
* Install a **custom watchdog** that translates pipeline status/latency to **Prometheus** + **AMQP** + (optional) **CP**.
* Decode → infer → **sink**:

  * The sink publishes **detection events** (see schema below).

### Outputs

#### 1) Detection Events (RabbitMQ)

* **Exchange:** `detections.topic` (topic)
* **Routing key:** `detections.<tenant_id>.<site_id>.<camera_uuid>`
* **Payload (example):**

  ```json
  {
    "ts": "2025-08-21T11:00:03Z",
    "runner_id": "runner-123",
    "shard_id": "shard-0",
    "tenant_id": "t-01",
    "site_id": "site-A",
    "camera_uuid": "cam-2",
    "frame_id": 471234,
    "fps": 5.7,
    "detections": [
      { "class": "person", "conf": 0.89, "bbox_xyxy": [x1,y1,x2,y2] },
      { "class": "car",    "conf": 0.77, "bbox_xyxy": [x1,y1,x2,y2] }
    ],
    "latency": { "inference_s": 0.024, "e2e_s": 0.145 }
  }
  ```

  **Requirements:**

  * Messages ≤ 256 KB. If larger, store payload externally and publish a pointer.
  * Use **publisher confirms**; retry on NACK.

#### 2) Stream Status & Errors (RabbitMQ)

* **Exchange:** `status.topic` (topic)
* **Routing keys:**

  * `stream.status.<tenant>.<site>.<camera>`
  * `stream.error.<tenant>.<site>.<camera>`
* **Edge-triggered events:**

  ```json
  { "type":"stream.status","state":"CONNECTING","camera_uuid":"cam-2","runner_id":"runner-123","shard_id":"shard-0","ts":"..."}
  { "type":"stream.status","state":"STREAMING", "camera_uuid":"cam-2","runner_id":"runner-123","shard_id":"shard-0","fps":5.8,"ts":"..."}
  { "type":"stream.status","state":"DISCONNECTED","camera_uuid":"cam-2","runner_id":"runner-123","shard_id":"shard-0","ts":"..."}
  ```
* **Periodic summary** every 5s:

  ```json
  { "type":"stream.status","camera_uuid":"cam-2","runner_id":"runner-123","shard_id":"shard-0",
    "state":"STREAMING","last_frame_ts":"...","last_frame_age_s":1.2,"fps":5.8,"ts":"..." }
  ```
* **Errors:**

  ```json
  { "type":"stream.error","camera_uuid":"cam-2","runner_id":"runner-123","code":"RTSP_AUTH_FAILED",
    "detail":"401 Unauthorized","retry_in_ms":8000,"ts":"..." }
  ```

#### 3) Prometheus Metrics (Worker)

* **Port:** `9108` (HTTP, plaintext)
* **Gauges:**

  * `stream_up{camera_uuid}` 0/1
  * `last_frame_age_seconds{camera_uuid}`
  * `stream_fps{camera_uuid}`
  * `pipeline_fps{runner_id,shard_id}`
* **Histograms:**

  * `inference_latency_seconds{camera_uuid}`
  * `e2e_latency_seconds{camera_uuid}`
* **Counters (optional):**

  * `stream_errors_total{camera_uuid,code}`

#### 4) Local Health Endpoints

* `GET /healthz` → 200 if pipeline threads alive.
* `GET /ready` → 200 when **≥ readiness\_quorum %** cameras are streaming/stable.
* `POST /drain` → graceful stop.
* `POST /terminate` → immediate stop.

### Buffering & FPS (Worker)

* For live streams:

  * **Buffer filling:** `DROP_OLDEST` (or adaptive)
  * **Consumption:** `EAGER`
  * **max\_fps:** 5–10 per camera (configured)
  * Favor **frame dropping** over sleep to avoid latency growth.

### Acceptance Criteria (Worker)

* Emits **status** within 1s of state changes.
* Emits **detections** ≤ 200ms after model prediction (local e2e).
* **/ready** flips to 200 within **15s** of start for ≥80% cameras (tunable).
* Handles **SIGTERM + /drain**; exits ≤ `grace_timeout_s` with no crashes.

---

## Control Plane — Interfaces used by Manager/Worker

(Implement in Django/DRF; JWT auth; all timestamps **UTC ISO-8601**.)

### Cameras

* `GET /v1/cameras?enabled=true&tenant_id=&site_id=&limit=&cursor=`

  * **200** → `{ "items":[{camera}], "next_cursor":null }`
  * Camera object: `{ "camera_uuid","tenant_id","site_id","rtsp_url","enabled","updated_at" }`

### Leases (authoritative)

* `POST /v1/leases/camera/acquire`

  * Body: `{ "runner_id", "camera_uuid", "ttl_seconds" }`
  * **200** → lease granted `{ "camera_uuid","owner_id","expires_at","version" }`
  * **409** → someone else holds a fresh lease.
* `POST /v1/leases/camera/renew`

  * Body: `{ "runner_id", "camera_uuid", "ttl_seconds" }`
  * **200** → renewed; **404** → not owner (manager must stop that camera).
* `POST /v1/leases/camera/release`

  * Body: `{ "runner_id", "camera_uuid" }` → **200**.

### Token Bucket

* `POST /v1/sites/{site_id}/budget/consume`

  * **200** → granted; **409** → empty (retry with backoff).

### Runner State

* `POST /v1/runners/heartbeat`

  * Body as heartbeat JSON; **200**.
* `GET /v1/runners/{runner_id}`

  * **200** → `{ "cordoned": false, "drain_mode":"none", "capacity_streams": 40 }`

### Optional Status Ingest

* `POST /v1/streams/status/batch`

  * Body: `{ "items": [ <stream.status> ] }` → **202** accepted.

---

## Metrics (Runner-level Prometheus)

Expose from Manager (port **9107**):

* `streams_desired_total` (gauge)
* `active_leases_total` (gauge)
* `streams_pending_total` (gauge)
* `runner_streams_owned{runner_id}` (gauge)
* `runner_shards{runner_id}` (gauge)
* `drain_in_progress{runner_id}` (gauge)
* Optional resource: `process_cpu_seconds_total`, `process_resident_memory_bytes`

**Golden Alerts**

* `streams_pending_total > 0 for 2m`
* `stream_up == 0 for 10s` (worker metric)
* `last_frame_age_seconds > 10`
* heartbeat missing (AMQP consumer detects no `runner.heartbeat` for 5s)
* ownership churn spike (`owner_changes_total` from CP)

---

## Logging (Structured JSON)

**Fields to include everywhere:**
`ts, level, runner_id, shard_id, site_id, tenant_id, camera_uuid, event, msg, error_code, stack, duration_ms`

**Examples:**

```json
{"ts":"...","level":"info","runner_id":"r1","event":"worker.spawn","shard_id":"shard-0","cameras":12}
{"ts":"...","level":"warn","camera_uuid":"cam-9","event":"rtsp.connect.retry","error_code":"TCP_TIMEOUT","retry_in_ms":8000}
{"ts":"...","level":"info","event":"bluegreen.cutover","old":"shard-0","new":"shard-1","duration_ms":3200}
```

---

## Configuration (Env & Defaults)

| Key                         | Default | Notes                 |
| --------------------------- | ------- | --------------------- |
| `LEASE_RENEW_INTERVAL_S`    | 2       | Manager renew cadence |
| `LEASE_TTL_S`               | 10      | Expire if not renewed |
| `TARGET_STREAMS_PER_SHARD`  | 12      | Tune by hardware      |
| `CAPACITY_STREAMS`          | 40      | Max per runner        |
| `READINESS_QUORUM_PCT`      | 80      | For `/ready`          |
| `STATUS_SUMMARY_INTERVAL_S` | 5       | Worker status summary |
| `HEARTBEAT_INTERVAL_S`      | 1       | Runner heartbeat      |
| `PROM_WORKER_PORT`          | 9108    | Worker metrics        |
| `PROM_MANAGER_PORT`         | 9107    | Runner metrics        |

---

## Security

* JWT for all CP endpoints; **short-lived** tokens; per-runner identity (`runner_id`).
* No camera credentials in logs or events.
* Bind Worker HTTP and Prometheus to **localhost**; if remote scrape is required, use a sidecar/Service with NetworkPolicies.
* If any component talks to DB directly (not required), use least-priv roles (SELECT `cameras`, UPSERT `stream_leases`).

---

## Acceptance Criteria (System)

1. **Exact-one:** No two Workers ever emit detections for the same `camera_uuid` within the same timestamp window.
   *Test:* Induce split-brain (kill renew path); verify fencing via lease 409s and Worker termination.

2. **Takeover ≤ 10s (P95):** Kill a runner; observe other runner emits `STREAMING` for its cameras within **≤ 10s** median/**≤ 12s** P95.

3. **Zero/minimal gap on reconfig:** Add/remove camera → Blue/Green completes and status returns to `STREAMING` for all remaining cameras within **≤ 5s** P95.

4. **No dial storms:** Add 100 cameras to a single site with `refill_rate=120/min` → peak concurrent connects ≤ tokens; NVR stays responsive.

5. **Observability:** Prom endpoints up; `runner.heartbeat` received each second; `stream.status` summaries every 5s.

6. **Graceful drain:** On `drain=soft`, all Workers exit cleanly within the grace window; leases released; other runners pick up.

---

## Test Plan

### Unit

* Lease client: acquire/renew/release happy/409/404 paths.
* Token bucket: consume/refill semantics; concurrent managers (race) → consistent tokens.
* Sharding planner: deterministic grouping, minimal churn.

### Integration (Local)

* Manager + mocked CP API + fake RTSP sources → assert shard spawn, `/ready`, Blue/Green transitions, event schemas.
* Failure injections: Worker crash, AMQP down, CP API latency spikes.

### Chaos & Scale

* Kill -9 a Manager; observe takeover times.
* 1,000 cameras across 50 sites; measure mean FPS, CPU/GPU utilization, Prom scrape stability.
* go2rtc (if used): restart mid-stream; ensure Worker recovers; no NVR overload.

### Performance

* Measure `inference_latency_seconds` and `e2e_latency_seconds` P50/P95 over 1h steady state.
* Backpressure: deliberately slow model → check `DROP_OLDEST + EAGER` holds `last_frame_age` under thresholds.

---

## Implementation Notes & Pointers

* **Inference pipeline and watchdog:** use your engine’s native hooks (`on_model_inference_started`, `on_model_prediction_ready`, `StatusUpdate`).
  *See engine documentation for exact hook names and behavior.*

* **go2rtc integration (optional):** manage named streams via its HTTP API per camera; bind to `127.0.0.1`; confirm `producer_connected` before declaring `/ready`.
  *See go2rtc API docs for `/api/streams` endpoints.*

* **RabbitMQ:** use **publisher confirms**; durable queues; sensible TTL for status messages (e.g., 10s) to avoid backlog.

---

## Handover Checklist (to the implementing team)

* [ ] Implement **Manager** control loop with lease client, shard planner, Blue/Green, token bucket, runner heartbeat, Prom metrics, local Worker control.
* [ ] Implement **Worker** with shard config, custom watchdog (Prom + AMQP + optional CP), health endpoints, drain/terminate behavior, detection sink.
* [ ] Implement **CP endpoints** listed (minimal lease, cameras list, budget consume, runner flags).
* [ ] Provision RabbitMQ exchanges/queues (`status.topic`, `detections.topic`).
* [ ] Configure Prometheus to scrape Worker `:9108` and Manager `:9107`.
* [ ] Add alert rules (as above) and dashboards for streams & runners.
* [ ] Run test plan; meet acceptance criteria.

---

### Final Notes

* Prefer **Mode B** (go2rtc fronting) at scale; use **Mode A** (direct RTSP) where simplicity is paramount.
* Tune shard size and `max_fps` empirically by hardware and model latency; scale shards before scaling pods.
* Keep renews at **2s** and TTL **8–10s** unless you’ve proven (with data) that more aggressive values don’t cause false failover in your networks.
