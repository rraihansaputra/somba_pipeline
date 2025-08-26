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


def render_debug_overlay(
    frame_bgr: np.ndarray,
    *,
    camera_cfg: Dict,
    detections: Optional[List[Dict]] = None,
    native_resolution: Optional[Tuple[int, int]] = None,
    show_fps: Optional[float] = None,
    timestamp: Optional[float] = None,
) -> np.ndarray:
    """Draw zones and detections on a BGR frame and return the annotated image."""
    img = frame_bgr
    h, w = img.shape[:2]
    cfg_w, cfg_h = (native_resolution or (0, 0))

    zones = camera_cfg.get("zones") or []

    # Draw zones with semi‑transparent fill
    overlay = img.copy()
    for z in zones:
        poly = z.get("polygon") or []
        if not poly:
            continue
        pts = _scale_polygon(poly, cfg_w, cfg_h, w, h)
        kind = (z.get("kind") or "include").lower()
        is_excl = kind == "exclude"
        color = BGR_RED if is_excl else BGR_GREEN
        alpha = 0.18 if is_excl else 0.10
        cv2.fillPoly(overlay, [pts], color=color)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Zone label
        label_parts = [f"#{z.get('zone_id', '?')} {z.get('name', '')}"]
        if is_excl:
            label_parts.append("exclude")
        else:
            label_parts.append("include")
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

    # Draw detections
    if detections:
        for det in detections:
            bbox = det.get("bbox") or det.get("xyxy")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            label = det.get("label") or det.get("class") or "obj"
            score = float(det.get("score") or det.get("confidence") or 0.0)
            tid = det.get("track_id")
            cv2.rectangle(img, (x1, y1), (x2, y2), BGR_YELLOW, 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (cx, cy), 3, BGR_WHITE, -1, lineType=cv2.LINE_AA)
            txt = f"{label} {score:.2f}"
            if tid is not None:
                txt = f"id={tid} " + txt
            _draw_label_box(img, txt, (x1, max(0, y1 - 6)))

    # HUD with timestamp / FPS
    if timestamp is not None or show_fps is not None:
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp or time.time()))
        hud = ts_str
        if show_fps is not None and show_fps > 0:
            hud += f"  |  {show_fps:.1f} FPS"
        _draw_label_box(img, hud, (10, h - 10))

    return img