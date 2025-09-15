from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import time
import cv2
import numpy as np

BGR_GREEN = (60, 200, 60)
BGR_RED = (60, 60, 220)
BGR_YELLOW = (40, 200, 255)
BGR_WHITE = (240, 240, 240)
BGR_BLACK = (0, 0, 0)

Polygon = List[Tuple[float, float]]


def _scale_polygon(poly: Polygon, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    """Scale polygon from config resolution (src) to current frame (dst)."""
    if src_w <= 0 or src_h <= 0:
        maxx = max(int(p[0]) for p in poly) if poly else 1
        maxy = max(int(p[1]) for p in poly) if poly else 1
        src_w, src_h = max(1, maxx), max(1, maxy)
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    pts = np.array([[int(round(x * sx)), int(round(y * sy))] for (x, y) in poly], dtype=np.int32)
    return pts.reshape((-1, 1, 2))


def _draw_label_box(img: np.ndarray, text: str, org: Tuple[int, int],
                    fg: Tuple[int, int, int] = BGR_BLACK,
                    bg: Tuple[int, int, int] = BGR_YELLOW) -> None:
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, fs, thickness)
    pad = 4
    cv2.rectangle(img, (x, y - th - pad * 2), (x + tw + pad * 2, y + pad), bg, -1)
    cv2.putText(img, text, (x + pad, y - pad), font, fs, fg, thickness, cv2.LINE_AA)


def _draw_hatched_polygon(img: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int] = BGR_YELLOW, spacing: int = 12) -> None:
    """Draw a hatched pattern inside the given polygon on img."""
    h, w = img.shape[:2]
    # Polygon mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    # Hatch canvas
    hatch = np.zeros_like(img)
    # Draw diagonal lines across the image
    for i in range(-h, w + h, spacing):
        pt1 = (max(0, i), max(0, -i))
        pt2 = (min(w - 1, i + h), min(h - 1, h))
        cv2.line(hatch, pt1, pt2, color, 1, lineType=cv2.LINE_AA)
    # Apply inside polygon only
    mask3 = cv2.merge([mask, mask, mask])
    np.copyto(img, hatch, where=mask3.astype(bool))


def render_debug_overlay(
    frame_bgr: np.ndarray,
    *,
    camera_cfg: Dict,
    detections: Optional[List[Dict]] = None,
    native_resolution: Optional[Tuple[int, int]] = None,
    show_fps: Optional[float] = None,
    timestamp: Optional[float] = None,
    motion_debug: Optional[Dict] = None,
    inference_state: Optional[str] = None,
    infer_fps: Optional[float] = None,
    last_infer_age_s: Optional[float] = None,
    drop_total: Optional[int] = None,
    recent_drop_age_s: Optional[float] = None,
    ghost_red: Optional[bool] = None,
) -> np.ndarray:
    """Draw zones and detections on a BGR frame and return the annotated image."""
    img = frame_bgr
    h, w = img.shape[:2]
    cfg_w, cfg_h = (native_resolution or (0, 0))

    zones = camera_cfg.get("zones") or []

    # Draw zones with yellow hatching and yellow outlines
    overlay = img.copy()
    for z in zones:
        poly = z.get("polygon") or []
        if not poly:
            continue
        pts = _scale_polygon(poly, cfg_w, cfg_h, w, h)
        _draw_hatched_polygon(overlay, pts, color=BGR_YELLOW, spacing=12)
        cv2.polylines(img, [pts], isClosed=True, color=BGR_YELLOW, thickness=2, lineType=cv2.LINE_AA)

        # Zone label
        label_parts = [f"#{z.get('zone_id', '?')} {z.get('name', '')}"]
        if z.get("allow_labels"):
            label_parts.append("allow:" + ",".join(z["allow_labels"]))
        if z.get("deny_labels"):
            label_parts.append("deny:" + ",".join(z["deny_labels"]))
        if z.get("min_score") is not None:
            label_parts.append(f"min:{z['min_score']:.2f}")
        label = " ".join(label_parts)
        anchor = tuple(pts[:, 0, :].min(axis=0).tolist())
        _draw_label_box(img, label, (int(anchor[0]) + 4, int(anchor[1]) + 20))

    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # Draw ROI-native motion mask if available
    if motion_debug and motion_debug.get("mask") is not None:
        mask = motion_debug["mask"]
        if mask.shape[:2] != (h, w):
            # Upsample the mask using nearest neighbor interpolation to avoid blur
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Create a semi-transparent overlay for motion
        motion_overlay = img.copy()
        # Use a yellow tint for motion
        motion_overlay[mask > 0] = [100, 200, 255]  # BGR for yellow
        cv2.addWeighted(motion_overlay, 0.3, img, 0.7, 0, img)

        # Draw ALL motion contours (including below threshold)
        contours = motion_debug.get("contours", [])
        if contours:
            for contour in contours:
                if len(contour) >= 3:
                    contour_array = np.array(contour, dtype=np.int32)
                    cv2.polylines(img, [contour_array], isClosed=True, color=BGR_YELLOW, thickness=2, lineType=cv2.LINE_AA)

    # Draw detections (green for current inference; red when ghost on skipped frames)
    if detections:
        use_red = bool(ghost_red)
        det_color = BGR_RED if use_red else BGR_GREEN
        for det in detections:
            bbox = det.get("bbox") or det.get("xyxy")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            label = det.get("label") or det.get("class") or "obj"
            score = float(det.get("score") or det.get("confidence") or 0.0)
            tid = det.get("track_id")
            cv2.rectangle(img, (x1, y1), (x2, y2), det_color, 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (cx, cy), 3, det_color, -1, lineType=cv2.LINE_AA)
            txt = f"{label} {score:.2f}"
            if tid is not None:
                txt = f"id={tid} " + txt
            _draw_label_box(img, txt, (x1, max(0, y1 - 6)), fg=BGR_BLACK, bg=det_color)

    # Motion debug chip (if provided)
    if motion_debug:
        active = bool(motion_debug.get("active", False))
        gate_reason = motion_debug.get("gate_reason", "")

        # Main motion status with threshold indicator
        status_txt = "ON" if active else "OFF"
        threshold_status = " (ABOVE)" if gate_reason == "above" else " (BELOW)" if gate_reason == "below" else f" ({gate_reason})"
        chip_txt = f"MOTION: {status_txt}{threshold_status}"
        _draw_label_box(img, chip_txt, (10, 30))

        # Enhanced motion metrics
        raw_pixels = motion_debug.get("raw_motion_pixels", 0)
        filtered_pixels = motion_debug.get("filtered_motion_area", 0)
        roi_area = motion_debug.get("roi_area", 0)
        raw_percent = motion_debug.get("raw_motion_percent", 0.0)
        filtered_percent = motion_debug.get("filtered_motion_percent", 0.0)

        # Display raw motion (always shown)
        raw_text = f"Raw Motion: {raw_pixels}/{roi_area}px ({raw_percent:.2f}%)"
        _draw_label_box(img, raw_text, (10, 55))

        # Display filtered motion vs threshold
        min_area = motion_debug.get("min_area_px", 0)
        threshold_percent = motion_debug.get("threshold_percent", 0.0)
        filtered_text = f"Filtered: {filtered_pixels}px ({filtered_percent:.2f}%) | Threshold: {min_area}px ({threshold_percent:.2f}%)"
        # Color based on threshold status
        bg_color = BGR_GREEN if gate_reason == "above" else BGR_RED if gate_reason == "below" else BGR_YELLOW
        _draw_label_box(img, filtered_text, (10, 80), bg=bg_color)

        # Display contour counts
        total_contours = motion_debug.get("total_contours", 0)
        significant = motion_debug.get("significant_contours", 0)
        below_threshold = motion_debug.get("below_threshold_contours", 0)
        contour_text = f"Contours: {total_contours} total ({significant} significant, {below_threshold} below noise floor)"
        _draw_label_box(img, contour_text, (10, 105))

        # Display noise floor setting
        noise_floor = motion_debug.get("noise_floor", 0)
        noise_text = f"Noise Floor: {noise_floor}px"
        _draw_label_box(img, noise_text, (10, 130))

    # Inference state chip (if provided)
    if inference_state:
        # Compose inference state details with age if available
        state_txt = inference_state
        if last_infer_age_s is not None:
            state_txt += f" | age {last_infer_age_s*1000:.0f} ms"
        if infer_fps is not None and infer_fps > 0:
            state_txt += f" | {infer_fps:.1f} iFPS"
        _draw_label_box(img, state_txt, (10, 155))

    # HUD with timestamp / FPS and drop info
    if timestamp is not None or show_fps is not None:
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp or time.time()))
        hud = ts_str
        if show_fps is not None and show_fps > 0:
            hud += f"  |  {show_fps:.1f} FPS"
        if infer_fps is not None and infer_fps > 0:
            hud += f"  |  {infer_fps:.1f} iFPS"
        if drop_total is not None and drop_total >= 0:
            hud += f"  |  drops: {drop_total}"
        if recent_drop_age_s is not None and recent_drop_age_s >= 0.0:
            hud += f"  |  last drop {recent_drop_age_s:.1f}s ago"
        _draw_label_box(img, hud, (10, h - 10))

    return img
