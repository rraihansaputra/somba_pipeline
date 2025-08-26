# Phase 2 Addendum — Zones, Motion-Gating, and Per-Zone Label Filters

This document revises Phase 2. It **does not** replace earlier specs; it layers new behavior for (a) per-camera zones, (b) motion-gated inference skipping, and (c) per-zone label filtering. Implement exactly as written; unresolved items are marked “OPEN”.

---

## 0. Goals (precise)

1. **Tie detections to zones** while **still running full-frame inference** (no ROI inference), except when motion-gating skips the frame.
2. **Skip inference** for a frame iff there is **no motion inside included zones** (after exclude mask).
3. **Filter labels per zone** (allow/deny) before publishing. Zone rules override camera/global rules.
4. **Zone 0** exists implicitly and always covers the **whole frame**. If a detection matches no other zone, it belongs to **zone 0**.
5. **Precedence:** Higher-priority zones override lower-priority zones and camera/global filters. (“Zone above it trumps exclude/include and label.”)

Non-goals (Phase 2): ROI inference, mask-based model input, learned scene polygons.

---

## 1. Definitions

* **Zone:** Named polygon in image coordinates (px), with **priority** (int; higher wins), **kind** (`include` or `exclude`), and optional **label filters**.
* **Zone 0 (implicit):** Whole frame, `priority = -∞`, `kind = include`, no polygon; cannot be deleted.
* **Membership test:** Primary: **center-in** (object center point ∈ polygon). Secondary (for auditing): **IoU** between detection bbox and polygon mask (float ∈ \[0,1]).
* **Included area (for motion):** `IncludeMask = union(include zones) − union(exclude zones)`.
* **Gating decision:** If **no** motion contour intersects `IncludeMask` (with dilation), **skip inference** for that frame.

---

## 2. Config Additions (per camera)

YAML (authoritative shape):

```yaml
camera:
  zones:
    # Highest 'priority' wins. zone_id must be unique per camera.
    - zone_id: 1
      name: "driveway"
      kind: "include"          # "include" | "exclude"
      priority: 100
      polygon: [[100,120],[1180,120],[1180,680],[100,680]]

      # Optional per-zone label filters (override camera/global)
      allow_labels: ["person","car"]   # null means no allow-list
      deny_labels: ["cat"]             # null means no deny-list
      min_score: 0.25                  # per-zone score floor (optional)

    - zone_id: 2
      name: "neighbor_lawn"
      kind: "exclude"
      priority: 200
      polygon: [[1180,120],[1800,120],[1800,680],[1180,680]]
      allow_labels: null
      deny_labels: ["person","car"]    # example: hard deny in this zone
      min_score: null

  # Motion gating config applies BEFORE inference
  motion_gating:
    enabled: true
    downscale: 0.5           # grayscale downscale factor for motion
    dilation_px: 6           # dilate motion mask
    min_area_px: 1500        # min intersect area with IncludeMask
    cooldown_frames: 2       # hysteresis (don’t flap)
    noise_floor: 12          # ignore tiny contours (< px)

  # Camera/global label filters (applied if zone has none)
  allow_labels: ["person","car","truck"]
  deny_labels: []
  min_score: 0.30            # global score floor
  zone_test: "center"        # "center" | "center+iou"
  iou_threshold: 0.10        # only used when "center+iou"
```

**Validation rules:**

* Polygons must be simple (non-self-intersecting), ≥3 vertices, within frame bounds (warn if not).
* `priority` is required; **larger value = higher precedence**.
* `zone_id` must be integer ≥1 (0 is reserved).
* Overlaps allowed; precedence handles conflicts.

---

## 3. Worker Pipeline (frame-level)

For each frame:

1. **Motion prepass (CPU, grayscale)**

   * Build `IncludeMask` once per config change.
   * Compute frame diff (running background or temporal gradient), threshold, morph (open/close), **dilate by `dilation_px`**.
   * Intersect motion mask with `IncludeMask`; sum area.
   * If `area < min_area_px` for **`cooldown_frames`** consecutive frames → **SKIP inference** for this frame.
   * Emit metric increments (see §7).

2. **Inference**

   * If not skipped, run **full-frame** inference with current adapter/model.
   * Get detections: `[{label, score, bbox_xywh, (segmentation?) …}]`.

3. **Zone attribution**

   * For each detection:

     * Compute **center point** `(cx, cy)` from bbox.
     * Determine **candidate zones** where `center ∈ polygon`.
       If `zone_test="center+iou"`, also require `IoU(zone_polygon, bbox) ≥ iou_threshold`.
     * If **≥1 candidate**, pick **primary zone** with **highest `priority`**.
     * Else, set primary zone to **zone 0**.
     * Additionally, compute `zones_hit` = all zones with `(center ∈ polygon)` (and IoU if enabled), sorted by priority (desc). Include zone 0 if empty.

4. **Per-zone label filtering**

   * Determine **effective filters** for this detection:

     * If primary zone has filters → use them.
     * Else use **camera/global** filters.
   * Apply **deny** first, then **allow** (allow wins only if label ∈ allow).
   * Apply **min\_score** from zone if set, else camera/global `min_score`.
   * If dropped by filters → do not publish that object (count metrics).

5. **Publish**

   * Publish detection event (even if empty objects list is allowed? **Decision:** publish **only if ≥1 object** survives filtering, or if `publish_empty=true` is future option).
   * Include **zone metadata** (see §4).

---

## 4. Detection Event Schema Additions

All new fields are **mandatory** when zones feature is enabled for the camera.

```json
{
  "schema_version": 2,
  "event_id": "01J...ULID",
  "ts_ns": 172...,
  "tenant_id": "t123",
  "site_id": "s1",
  "camera_uuid": "cam-abc",
  "model": {"id": "rf:yolov8n-8.1.0", "adapter": "roboflow-inference-pipeline"},
  "frame": {"w": 1920, "h": 1080, "seq": 123456, "fps": 15.0, "skipped_by_motion": false},

  "zones_config": {
    "zone_version": "sha256:…",          // hash of zones JSON to detect drift
    "zone_test": "center",               // or "center+iou"
    "iou_threshold": 0.10
  },

  "objects": [
    {
      "label": "person",
      "score": 0.92,
      "bbox_xywh": [x, y, w, h],
      "segmentation": [[...]],           // if available
      "primary_zone_id": 1,              // 0 means whole-frame default
      "zones_hit": [1, 2],               // sorted by priority desc; may be []
      "zone_membership": {               // optional auditing detail
        "1": {"center_in": true, "iou": 0.18},
        "2": {"center_in": true, "iou": 0.05}
      },
      "filtered": false,                 // true if it *would* be filtered (only present when publish_empty future flag)
      "filter_reason": null              // or "deny_label" | "min_score" | ...
    }
  ]
}
```

**Status event additions (every 5s):**

```json
{
  "zones_stats": {
    "frames_skipped_motion": 123,
    "frames_processed": 4567,
    "objects_published": 890,
    "objects_dropped_by_filters": 42,
    "per_zone": {
      "0": {"objects": 500, "dropped": 30},
      "1": {"objects": 300, "dropped": 10},
      "2": {"objects": 90,  "dropped": 2}
    }
  }
}
```

---

## 5. Precedence & Conflict Rules (normative)

1. **Motion gating** uses only **IncludeMask minus Exclude**. If **no motion** intersects that area → **skip inference**.

   * Exclude zones do **not** force skipping when there is motion elsewhere inside IncludeMask.
2. **Zone membership** for an object is by **center-in** (and IoU if configured).
3. **Primary zone** = the candidate zone with maximum **priority**.
4. **Per-zone filters** (of primary zone) **override** camera/global filters.
5. If a detection lies **only** in exclude zones (no include zones hit), it still gets a **primary zone** by precedence (likely an exclude zone) for **auditing**, but **filtering** still runs (its zone’s deny/allow typically removes it).
6. If a detection hits **no zone**, `primary_zone_id=0` (whole frame).
7. **Zone 0** filters apply only when no other zone matched.

---

## 6. API & Control Plane

* **Config distribution:** Manager/Runner already sync camera config. Extend schema to include `zones`, `motion_gating`, `zone_test`, `iou_threshold`, and global filters.
* **Blue/Green reload:** Any zone config change triggers a controlled **reload** (no model change). Minimal cut: worker supports **in-place** config swap; if not supported, do “start new worker with same model → `/ready` → drain old”.
* **Validation endpoint (optional):** `POST /validate_zones` on Runner to verify polygons, priorities, hashes.

---

## 7. Metrics (Prometheus names; per camera unless noted)

* `frames_total{camera}`
* `frames_skipped_motion_total{camera}`
* `motion_area_px_sum{camera}` (histogram suggested)
* `detections_raw_total{camera}` (pre-filter)
* `detections_published_total{camera,zone_id,label}`
* `detections_dropped_total{camera,zone_id,reason}` (`deny_label|min_score|no_zone_allowed`)
* `zone_assignment_latency_ms{camera}` (optional)
* `zones_config_hash{camera}` (gauge as 64-bit hash for change detection)

Alerts (examples):

* High skip ratio with high true activity: investigate gating thresholds.
* Objects dropped rate spikes after config change: likely mis-set filters.

---

## 8. Logging (structured)

Per frame (sampled) and per detection (debug level with sampling):

```json
{
  "msg": "zone_attribution",
  "camera": "cam-abc",
  "frame_seq": 123456,
  "skipped_by_motion": false,
  "zones_hash": "sha256:…",
  "objects": 12,
  "published": 9,
  "dropped": 3,
  "drops": {"deny_label":2,"min_score":1}
}
```

Include **`event_id`**, `primary_zone_id`, and `zones_hit` in detection-level debug logs when useful.

---

## 9. Backward Compatibility

* If `zones` is **absent** or `motion_gating.enabled=false` → behavior reverts to Phase 1 (no skips, no per-zone filters; everything maps to zone 0).
* Consumers ignoring new fields continue to work; schema version increments to **2**.

---

## 10. Performance Considerations

* Motion stage is CPU-cheap; set `downscale ∈ [0.4, 0.7]`.
* Expect significant GPU savings on static scenes (frames skipped).
* Zone attribution is O(Nzones × Ndetections); with typical `Nzones ≤ 10`, cost is negligible vs inference.

---

## 11. Acceptance Criteria (must pass)

1. **Skip logic:** With `motion_gating.enabled=true`, when **no** motion intersects `IncludeMask` for `cooldown_frames`, worker **does not** run inference and increments `frames_skipped_motion_total`.
2. **Zone mapping:** For synthetic scenes where a bbox center lies inside zone A and overlaps zone B, **primary=highest priority**; `zones_hit` includes both.
3. **Filters precedence:** Per-zone `allow_labels=["person"]` and global `deny_labels=["person"]` → **person is published** when inside that zone.
4. **Exclude auditing:** Detection in an **exclude** zone only → `primary_zone_id=<exclude zone id>`, object is **dropped** if zone’s deny/min\_score says so.
5. **Zone 0 fallback:** Detection not in any polygon → `primary_zone_id=0`.
6. **Schema:** All emitted events include `primary_zone_id`, `zones_hit`, and `zones_config.zone_version`.
7. **Blue/Green:** Changing zones causes zero crash loops; either hot-reload or clean swap within existing SLOs.
8. **Load:** With a static scene for 60s, at least **90% frames** are skipped (given sensible thresholds) and GPU util drops accordingly.

---

## 12. Test Plan (concrete)

* **Unit:** polygon winding, point-in-poly, IoU vs polygon mask; precedence; filter resolution order.
* **Property tests:** Random polygons/boxes; invariants: zone 0 fallback always defined; higher priority wins.
* **Integration:**

  * Motion-only clip → 0 inference invocations; verify counters.
  * Overlap zones (A priority 200 include, B 100 exclude): object centered in overlap → primary=A, published if label allowed by A.
  * Toggle config live → no missed lease renewals; readiness unaffected after warmup.
* **Perf:** 1080p @ 15 fps, 8 zones, Ndets ≤ 50/frame → attribution < 1 ms avg on CPU.

---

## 13. OPEN Items (defaults proposed)

* **Default thresholds:** `downscale=0.5`, `dilation_px=6`, `min_area_px=1500`, `cooldown_frames=2`, `noise_floor=12`.
* **Publish empty events:** keep **disabled** in Phase 2.
* **Per-zone min\_score vs calibration:** we default to camera/global if zone unset.

---

## 14. Developer Checklist

* [ ] Extend camera config model + validation.
* [ ] Implement motion gating stage (CPU) with Include/Exclude mask.
* [ ] Implement zone attribution + precedence.
* [ ] Implement per-zone label filters with override rules.
* [ ] Add metrics, logs, and schema v2 fields.
* [ ] Add hot-reload or Blue/Green on zone changes.
* [ ] Update docs and examples; add sample camera config with 3 zones.
* [ ] Add tests in §12; wire into CI.

---

**Bottom line:**

* **Full-frame inference** remains the default.
* **Skip** only when **no motion** in **included** zones.
* Every detection is **assigned a primary zone**; **per-zone filters** dictate what is **published**.
* **Zone 0** is the safety net; **priority** settles conflicts.
