# somba_pipeline/mjpeg_http.py
# Minimal HTTP MJPEG server & publisher for debug streaming
# Stdlib server, multi-stream endpoints, thread-safe frame hubs

from __future__ import annotations
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple
import threading
import time
import urllib.parse as urlparse
import cv2  # Requires opencv-python(-headless)
import numpy as np

__all__ = [
    "MjpegHttpServer",
    "MjpegHttpPublisher",
    "MjpegOptions",
]


@dataclass
class MjpegOptions:
    host: str = "0.0.0.0"
    port: int = 8089
    boundary: str = "frame"
    # Per-stream defaults (can be overridden by publisher)
    default_quality: int = 80  # 1..100
    default_fps: int = 10      # 1..60
    # Optional listing title
    title: str = "Somba Debug Streams"


class _FrameHub:
    """Thread-safe store of the latest JPEG for a stream.

    - Holds only the latest JPEG (no queue) to avoid memory bloat.
    - Conditionally notifies waiting handlers on updates.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._last_jpeg: Optional[bytes] = None
        self._seq: int = 0
        self._closed: bool = False

    def set_jpeg(self, data: bytes) -> None:
        with self._cond:
            if self._closed:
                return
            self._last_jpeg = data
            self._seq += 1
            self._cond.notify_all()

    def get_next(self, last_seq: int, timeout: float = 30.0) -> Tuple[Optional[bytes], int]:
        with self._cond:
            if self._closed:
                return None, last_seq
            if self._seq == last_seq:
                self._cond.wait(timeout)
            if self._seq == last_seq:
                # No new frame; return None so handler can keep-alive/loop
                return None, last_seq
            return self._last_jpeg, self._seq

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class MjpegHttpServer:
    """Process-wide HTTP server for MJPEG streaming with multi-stream registry."""

    _instance_lock = threading.Lock()
    _instance: Optional["MjpegHttpServer"] = None

    @classmethod
    def get_or_create(cls, opts: Optional[MjpegOptions] = None) -> "MjpegHttpServer":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = MjpegHttpServer(opts or MjpegOptions())
                cls._instance.start()
            return cls._instance

    def __init__(self, opts: MjpegOptions) -> None:
        self.opts = opts
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._registry: Dict[str, _FrameHub] = {}
        self._titles: Dict[str, str] = {}
        self._lock = threading.RLock()

    # --- Public API ---
    def start(self) -> None:
        if self._httpd is not None:
            return

        # Build a handler class bound to this server instance
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            # Silence stdlib's default logging
            def log_message(self, fmt: str, *args) -> None:  # noqa: N802
                return

            def do_GET(self):  # noqa: N802
                parsed = urlparse.urlsplit(self.path)
                path = parsed.path
                if path in ("/healthz", "/health"):
                    self._write_text(200, "OK\n")
                    return
                if path in ("/", ""):
                    self._write_index()
                    return
                # Expect paths like /<stream>.mjpg
                if path.startswith("/") and path.endswith(".mjpg"):
                    key = path[1:-5]  # strip leading '/' and trailing '.mjpg'
                    self._stream_mjpeg(key)
                    return
                self._write_text(404, f"Not found: {path}\n")

            # --- Helpers ---
            def _write_text(self, code: int, text: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                try:
                    self.wfile.write(text.encode("utf-8"))
                except Exception:
                    pass

            def _write_index(self) -> None:
                host, port = server_ref.address
                title = server_ref.opts.title
                items = []
                with server_ref._lock:
                    for key, _hub in server_ref._registry.items():
                        display = server_ref._titles.get(key, key)
                        href = f"http://{host}:{port}/{key}.mjpg"
                        items.append(f"<li><a href=\"{href}\">{display}</a></li>")
                html = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    f"<title>{title}</title>"
                    "<style>body{font-family:system-ui,Segoe UI,Arial;margin:24px}"
                    "h1{margin:0 0 12px} ul{line-height:1.8}</style></head><body>"
                    f"<h1>{title}</h1>"
                    "<p>Open a stream link below in your browser or in ffplay.</p>"
                    f"<ul>{''.join(items) if items else '<li><em>No streams registered</em></li>'}</ul>"
                    "</body></html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                try:
                    self.wfile.write(html.encode("utf-8"))
                except Exception:
                    pass

            def _stream_mjpeg(self, key: str) -> None:
                hub = server_ref._registry.get(key)
                if hub is None:
                    self._write_text(404, f"Unknown stream: {key}\n")
                    return
                boundary = server_ref.opts.boundary
                # Response headers for MJPEG
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.end_headers()

                last_seq = -1
                # Stream loop
                try:
                    while not server_ref._shutdown.is_set():
                        jpeg, last_seq = hub.get_next(last_seq, timeout=30.0)
                        if jpeg is None:
                            # No new frame; send a tiny keep-alive chunk
                            try:
                                self.wfile.write(f"--{boundary}\r\n".encode("ascii"))
                                self.wfile.write(b"Content-Type: text/plain\r\n\r\n")
                                self.wfile.write(b"keepalive\r\n")
                            except Exception:
                                break
                            continue
                        try:
                            self.wfile.write(f"--{boundary}\r\n".encode("ascii"))
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        except Exception:
                            break
                except Exception:
                    pass

        # Bind server
        address = (self.opts.host, self.opts.port)
        self._httpd = ThreadingHTTPServer(address, Handler)
        # Stash references on the HTTPServer instance for introspection if needed
        self._httpd._parent = self  # type: ignore[attr-defined]

        self._thread = threading.Thread(target=self._httpd.serve_forever, name="mjpeg-http", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._shutdown.set()
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._lock:
            for hub in self._registry.values():
                hub.close()
            self._registry.clear()
            self._titles.clear()

    # --- Stream registry ---
    def register_stream(self, key: str, title: Optional[str] = None) -> str:
        """Create a stream endpoint; returns the URL path (e.g., "/<key>.mjpg")."""
        with self._lock:
            if key not in self._registry:
                self._registry[key] = _FrameHub()
            if title:
                self._titles[key] = title
        return f"/{key}.mjpg"

    def unregister_stream(self, key: str) -> None:
        with self._lock:
            hub = self._registry.pop(key, None)
            self._titles.pop(key, None)
            if hub:
                hub.close()

    # --- Frame push API ---
    def push_jpeg(self, key: str, jpeg: bytes) -> None:
        hub = self._registry.get(key)
        if not hub:
            # Auto-register if unknown
            self.register_stream(key)
            hub = self._registry[key]
        hub.set_jpeg(jpeg)

    def push_frame_bgr(self, key: str, frame_bgr: np.ndarray, *, quality: Optional[int] = None) -> None:
        if not isinstance(frame_bgr, np.ndarray):
            return
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return
        q = int(self.opts.default_quality if quality is None else quality)
        q = max(1, min(100, q))
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return
        self.push_jpeg(key, buf.tobytes())

    @property
    def address(self) -> Tuple[str, int]:
        return (self.opts.host, self.opts.port)


class MjpegHttpPublisher:
    """Per-stream publisher that throttles pushes by FPS and posts JPEGs to the shared server."""

    def __init__(self, server: MjpegHttpServer, stream_key: str, *, title: Optional[str] = None,
                 fps: Optional[int] = None, quality: Optional[int] = None) -> None:
        self.server = server
        self.stream_key = stream_key
        self.path = server.register_stream(stream_key, title=title)
        self.fps = max(1, int(server.opts.default_fps if fps is None else fps))
        self.quality = int(server.opts.default_quality if quality is None else quality)
        self._min_interval = 1.0 / float(self.fps)
        self._last_sent = 0.0

    def url(self) -> str:
        host, port = self.server.address
        return f"http://{host}:{port}{self.path}"

    def publish_frame_bgr(self, frame_bgr: np.ndarray) -> None:
        now = time.perf_counter()
        if (now - self._last_sent) < self._min_interval:
            return
        self._last_sent = now
        self.server.push_frame_bgr(self.stream_key, frame_bgr, quality=self.quality)

    def publish_jpeg(self, jpeg_bytes: bytes) -> None:
        now = time.perf_counter()
        if (now - self._last_sent) < self._min_interval:
            return
        self._last_sent = now
        self.server.push_jpeg(self.stream_key, jpeg_bytes)

    def close(self) -> None:
        # optional: keep stream visible even if closed; comment next line to persist listing
        self.server.unregister_stream(self.stream_key)
