Short and to the point:

**Goal:** cut compute, not just noise.

## Per-camera zones (compute saver)

* **Config (per camera):**

  * `include_zones: [ [ [x,y], ... ], ... ]` (polygons)
  * `exclude_zones: [ ... ]`
  * `motion_gating: { enabled: true, min_area_px: N, dilation_px: D, padding_px: P, min_roi_px: M }`
  * `zone_test: "center" | "iou>t"` (post-filter rule)
* **Worker pipeline:**

  1. Build binary mask `M = union(include) − union(exclude)`.
  2. Cheap motion on downscaled gray (CPU). Take moving contours `C`.
  3. If `area( (C ∩ M) dilated by D ) < min_area_px` → **skip inference** (count metric `frames_skipped_motion_zone`).
  4. Else compute ROI = bounding rect of `(C ∩ M)` padded by `P`, clamp to frame.

     * If adapter supports ROI inference → run on **crop**, remap outputs by offset.
     * Else (no ROI support) → optional **mask frame** (`frame &= M`) and run full-frame (saves false positives, not FLOPs).
  5. After inference, drop objects failing `zone_test` against `M` (center-in or IoU threshold).

> This saves real compute because most frames never reach inference, and ROI cropping shrinks tensor size when supported.

## Label filtering (noise saver)

* **Do it in the worker** to cut AMQP traffic and downstream load:

  * `allow_labels: ["person","car"]` and/or `deny_labels: ["cat"]`
  * Apply **after** model inference (doesn’t reduce FLOPs, just emissions).
* You can still re-enforce on the consumer as a safety net, but it won’t reduce compute.

## Minimal API shape

```yaml
camera:
  include_zones: [ [[100,100],[800,100],[800,600],[100,600]] ]
  exclude_zones: [ [[400,300],[500,300],[500,400],[400,400]] ]
  motion_gating: { enabled: true, min_area_px: 1500, dilation_px: 6, padding_px: 16, min_roi_px: 20000 }
  zone_test: "center"
  allow_labels: ["person","truck"]
  deny_labels: []
```

## Edge cases & defaults

* Add hysteresis: skip up to `K` consecutive frames after a negative to avoid flapping.
* Inflate zones by `dilation_px` to avoid clipping fast movers.
* If ROI becomes tiny or extreme aspect, expand to `min_roi_px`.

## Tests (quick)

* Unit: mask build, point-in-polygon, IoU, remap (crop↔full frame).
* Perf A/B: measure % frames skipped, avg ROI size, GPU util.
* Accuracy: boundary cases (object straddling zone edge) under both `center` and `iou>t`.

**Bottom line:** implement zone gating + ROI in the **worker** (saves compute), apply label filters in the **worker** (saves bandwidth), optionally re-filter again on the **consumer** (safety).
