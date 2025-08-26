from __future__ import annotations
import time
import subprocess
import threading
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import cv2
import numpy as np

from .mjpeg_http import MjpegHttpServer, MjpegHttpPublisher, MjpegOptions



def _normalize_mask_any(mask: Any, out_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Accepts None|bool|int|float|list|ndarray; returns uint8 (H,W) in [0..255] or None.
    Resizes to out_size=(w,h) if needed.
    """
    if mask is None:
        return None
    # Scalars/bools are not drawable masks
    if isinstance(mask, (bool, np.bool_, int, float)):
        return None
    # Convert to ndarray
    m = np.asarray(mask)
    if m.size == 0:
        return None
    # If 3-channel, take first channel
    if m.ndim == 3:
        m = m[..., 0]
    # Convert to uint8 in [0..255]
    if m.dtype != np.uint8:
        # If this already looks like a probability mask (0..1), scale up
        m_min, m_max = float(m.min()), float(m.max())
        if 0.0 <= m_min and m_max <= 1.0:
            m = (m * 255.0).astype(np.uint8)
        else:
            m = np.clip(m, 0, 255).astype(np.uint8)
    # Resize if needed (cv2 resize takes (w,h))
    w, h = out_size
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m


def draw_overlay(
    frame_bgr: np.ndarray,
    *,
    zones: List[dict] | None,
    detections: List[dict] | None,
    ts_ms: Optional[int] = None,
    delta_ms: Optional[int] = None,
    frame_id: Optional[str] = None,
    fps: Optional[float] = None,
    motion: Optional[dict[str, Any]] = None,
    show_zones: bool = True,
    show_detections: bool = True,
    show_motion_mask: bool = True,
    motion_alpha: float = 0.30,
) -> np.ndarray:
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    # --- MOTION OVERLAY (mask + contours) ---
    if motion:
        m_on = bool(motion.get("on", False))
        raw_mask = motion.get("mask", None)
        # NEW: normalize anything to uint8 (H,W) or None
        m = _normalize_mask_any(raw_mask, (w, h))

        contours = motion.get("contours", None)
        # If caller didn't compute contours and we have a mask, extract them
        if contours is None and m is not None:
            cs, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c.reshape(-1, 2).astype(int).tolist() for c in cs if len(c) >= 3]

        zone_hits = motion.get("zone_hits") or []

        if show_motion_mask and m is not None:
            heat = cv2.applyColorMap(m, cv2.COLORMAP_JET)
            img = cv2.addWeighted(heat, motion_alpha, img, 1.0 - motion_alpha, 0)

        if contours:
            for poly in contours:
                if not poly:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

        status_txt = "MOTION: ON" if m_on else "MOTION: OFF (skipped inference)"
        hud_color = (0, 200, 0) if m_on else (0, 0, 200)
        cv2.putText(img, status_txt, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, status_txt, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2, cv2.LINE_AA)
        if zone_hits:
            ztxt = f"zones: {','.join(map(str, zone_hits))}"
            cv2.putText(img, ztxt, (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(img, ztxt, (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 1, cv2.LINE_AA)

    # --- ZONES (same as before, protected by show_zones) ---
    if show_zones and zones:
        for z in zones:
            poly = z.get("poly") or []
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                include = bool(z.get("include", True))
                color = (0, 200, 0) if include else (0, 0, 200)
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color=color)
                img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
                name = z.get("name") or z.get("id") or ""
                if name:
                    p0 = tuple(map(int, poly[0]))
                    cv2.putText(img, f"{name}", p0, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # --- DETECTIONS (same as before, protected by show_detections) ---
    if show_detections and detections:
        for det in detections:
            bbox = det.get("bbox") or det.get("box")
            if not bbox:
                continue
            if len(bbox) == 4:
                x, y, a, b = bbox
                if a <= 1.0 and b <= 1.0:
                    x, y, a, b = int(x*w), int(y*h), int(a*w), int(b*h)
                x1, y1, x2, y2 = int(x), int(y), int(x+a), int(y+b)
            else:
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            label = det.get("label") or det.get("class") or "obj"
            conf = det.get("conf") or det.get("confidence") or 0.0
            tid  = det.get("id") or det.get("track_id") or 0
            rng = int(tid) if isinstance(tid, int) or (isinstance(tid, str) and tid.isdigit()) else hash(str(tid))
            color = ((37 * (rng + 1)) % 255, (17 * (rng + 7)) % 255, (29 * (rng + 13)) % 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f} id={tid}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ytxt = max(0, y1 - 4)
            cv2.rectangle(img, (x1, ytxt - th - 4), (x1 + tw + 4, ytxt), color, -1)
            cv2.putText(img, text, (x1 + 2, ytxt - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # --- Timing HUD (unchanged) ---
    hud = []
    if frame_id: hud.append(f"frame={frame_id}")
    if ts_ms is not None: hud.append(f"ts={ts_ms}")
    if delta_ms is not None: hud.append(f"Δ={delta_ms}ms")
    if fps is not None and fps > 0: hud.append(f"{fps:.1f} fps")
    if hud:
        cv2.putText(img, " | ".join(hud), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, " | ".join(hud), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

    cv2.putText(img, "DEBUG", (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, "DEBUG", (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240,240,240), 1, cv2.LINE_AA)
    return img


@dataclass
class PublisherSpec:
    # New default backend is HTTP MJPEG
    backend: str = "mjpeg_http"   # {"mjpeg_http", "mpegts", "rtp"}

    # MJPEG HTTP options (safe defaults)
    http_host: str = "127.0.0.1"
    http_port: int = 8089
    http_fps: int = 10
    http_quality: int = 80
    http_title: str = "Somba Debug Streams"

    # Legacy RTP/MPEG-TS options (kept for optional fallback)
    width: int = 1920
    height: int = 1080
    fps: int = 10
    host: str = "127.0.0.1"
    port: int = 5002
    crf: int = 28
    preset: str = "veryfast"
    tune: str = "zerolatency"
    codec: str = "libx264"


class RTPPublisher:
    def __init__(self, spec: PublisherSpec):
        self.spec = spec
        self._alive = True
        self._lock = threading.Lock()
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{spec.width}x{spec.height}",
            "-r", str(spec.fps), "-i", "-",
            "-an", "-c:v", spec.codec,
            "-preset", spec.preset, "-tune", spec.tune,
            "-crf", str(spec.crf),
            "-f", "rtp", f"rtp://{spec.host}:{spec.port}"
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            bufsize=0
        )

    def push(self, frame_bgr: np.ndarray) -> bool:
        if not self._alive:
            return False
        try:
            self.proc.stdin.write(frame_bgr.tobytes())
            return True
        except Exception:
            self._alive = False
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
            return False

    def close(self):
        with self._lock:
            if not self._alive:
                return
            self._alive = False
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass


class TSPublisher:
    """MPEG‑TS over UDP publisher (fallback for pure RTP)."""

    def __init__(self, spec: PublisherSpec):
        self.spec = spec
        self._alive = True
        self._lock = threading.Lock()
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{spec.width}x{spec.height}",
            "-r",
            str(spec.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            spec.codec,
            "-preset",
            spec.preset,
            "-tune",
            spec.tune,
            "-crf",
            str(spec.crf),
            "-f",
            "mpegts",
            f"udp://{spec.host}:{spec.port}?pkt_size=1316&fifo_size=5000000",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            bufsize=0,
        )

    def push(self, frame_bgr: np.ndarray) -> bool:
        if not self._alive:
            return False
        try:
            self.proc.stdin.write(frame_bgr.tobytes())
            return True
        except Exception:
            self._alive = False
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
            return False

    def close(self):
        with self._lock:
            if not self._alive:
                return
            self._alive = False
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass


class DebugSink:
    def __init__(self, publisher: RTPPublisher, ttl_s: int):
        self.publisher = publisher
        self.deadline = time.time() + max(5, ttl_s)
        self.frames_sent = 0
        self.frames_dropped = 0

    def refresh(self, ttl_s: int):
        self.deadline = time.time() + max(5, ttl_s)

    def push(self, frame_bgr: np.ndarray) -> bool:
        if time.time() > self.deadline:
            return False
        ok = self.publisher.push(frame_bgr)
        if ok:
            self.frames_sent += 1
        else:
            self.frames_dropped += 1
        return ok

    def close(self):
        self.publisher.close()


class DebugManager:
    _http_server: Optional[MjpegHttpServer] = None

    def __init__(self, spec: Optional[PublisherSpec] = None):
        self.spec = spec or PublisherSpec()
        self._publishers: Dict[str, object] = {}
        self._lock = threading.Lock()

    def _ensure_http_server(self) -> MjpegHttpServer:
        if DebugManager._http_server is None:
            opts = MjpegOptions(
                host=self.spec.http_host,
                port=self.spec.http_port,
                boundary="frame",
                default_quality=self.spec.http_quality,
                default_fps=self.spec.http_fps,
                title=self.spec.http_title,
            )
            DebugManager._http_server = MjpegHttpServer.get_or_create(opts)
        return DebugManager._http_server

    def _parse_legacy_start_args(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Optional[str], int, int]:
        """Parse legacy positional/keyword arguments for start().
        Returns (title, fps, quality).
        """
        title: Optional[str] = kwargs.get("title")
        fps: int = int(kwargs.get("fps") or self.spec.http_fps)
        quality: int = int(kwargs.get("quality") or kwargs.get("http_quality") or self.spec.http_quality)

        # Positional handling
        if len(args) == 1:
            if isinstance(args[0], str):
                title = args[0]
            elif isinstance(args[0], int):
                fps = int(args[0])
        elif len(args) == 2:
            a0, a1 = args
            if isinstance(a0, str) and isinstance(a1, int):
                title, fps = a0, int(a1)
        elif len(args) >= 3:
            a0, a1, a2 = args[0], args[1], args[2]
            if all(isinstance(x, int) for x in (a0, a1, a2)):
                fps = int(a2)  # old (w, h, fps) signature
            elif isinstance(a0, str) and isinstance(a1, int):
                title, fps = a0, int(a1)

        # Clamp values
        fps = max(1, min(60, fps))
        quality = max(1, min(100, quality))
        return title, fps, quality

    def start(self, stream_key: str, *args: Any, **kwargs: Any) -> str:
        """Start (or attach to) a debug publisher for a stream.
        Returns a URL to open in ffplay/browser.
        Supports legacy signatures via *args/**kwargs.
        """
        if stream_key in self._publishers:
            # Already started; return existing URL if possible
            pub = self._publishers[stream_key]
            try:
                return pub.url()
            except Exception:
                pass

        title, fps, quality = self._parse_legacy_start_args(args, kwargs)

        if self.spec.backend == "mjpeg_http":
            server = self._ensure_http_server()
            pub = MjpegHttpPublisher(
                server,
                stream_key,
                title=title,
                fps=fps,
                quality=quality,
            )
            with self._lock:
                self._publishers[stream_key] = pub
            return pub.url()
        raise NotImplementedError(f"Unsupported backend: {self.spec.backend}")

    def stop(self, stream_key: str) -> None:
        pub = self._publishers.pop(stream_key, None)
        if isinstance(pub, MjpegHttpPublisher):
            pub.close()

    @classmethod
    def shutdown_all(cls) -> None:
        if cls._http_server is not None:
            cls._http_server.stop()
            cls._http_server = None

    def has_publisher(self, stream_key: str) -> bool:
        with self._lock:
            return stream_key in self._publishers

    # Compatibility aliases
    def has_sink(self, stream_key: str) -> bool:
        return self.has_publisher(stream_key)

    def start_sink(self, stream_key: str, *args: Any, **kwargs: Any) -> str:
        return self.start(stream_key, *args, **kwargs)

    def publish_sink(self, stream_key: str, frame_bgr: np.ndarray) -> None:
        self.maybe_publish(
            stream_key,
            frame_bgr,
            zones=None,
            detections=None,
            ts_ms=None,
            delta_ms=None,
            frame_id=None,
            fps=None,
            motion=None,
        )

    def stop_sink(self, stream_key: str) -> None:
        self.stop(stream_key)

    def get_sink_url(self, stream_key: str) -> Optional[str]:
        pub = self._publishers.get(stream_key)
        if pub is None:
            return None
        try:
            return pub.url()
        except Exception:
            return None

    # Additional legacy aliases
    def publish(self, stream_key: str, frame_bgr: np.ndarray) -> None:
        self.publish_sink(stream_key, frame_bgr)

    def url_of(self, stream_key: str) -> Optional[str]:
        return self.get_sink_url(stream_key)

    def maybe_publish(
        self,
        stream_key: str,
        frame_bgr: np.ndarray,
        *,
        zones: List[dict] | None = None,
        detections: List[dict] | None = None,
        ts_ms: Optional[int] = None,
        delta_ms: Optional[int] = None,
        frame_id: Optional[str] = None,
        fps: Optional[float] = None,
        motion: Optional[dict[str, Any]] = None,
        show_zones: bool = True,
        show_detections: bool = True,
        show_motion_mask: bool = True,
        motion_alpha: float = 0.30,
    ) -> None:
        pub = self._publishers.get(stream_key)
        if not pub:
            return
        if isinstance(pub, MjpegHttpPublisher):
            annotated = draw_overlay(
                frame_bgr,
                zones=zones,
                detections=detections,
                ts_ms=ts_ms,
                delta_ms=delta_ms,
                frame_id=frame_id,
                fps=fps,
                motion=motion,
                show_zones=show_zones,
                show_detections=show_detections,
                show_motion_mask=show_motion_mask,
                motion_alpha=motion_alpha,
            )
            pub.publish_frame_bgr(annotated)
            return
        # Add other backend conditionals as needed

    def close_all(self):
        with self._lock:
            sinks = list(self._sinks.items())
            self._sinks.clear()
        for _, sink in sinks:
            try:
                sink.close()
            except Exception:
                pass
