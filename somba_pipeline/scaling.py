from typing import Tuple, Iterable
import numpy as np

def resolve_proc_size(frame_w: int, frame_h: int, downscale: float | int) -> Tuple[int, int]:
    """
    Accepts either a factor (0&lt;downscale&lt;1, e.g. 0.5) or an integer divisor (&gt;=1, e.g. 2).
    Returns (w_small, h_small).
    """
    if isinstance(downscale, (int, float)) is False or downscale <= 0:
        raise ValueError(f"downscale must be &gt; 0, got {downscale}")

    if downscale < 1.0:
        w_small = max(1, int(round(frame_w * float(downscale))))
        h_small = max(1, int(round(frame_h * float(downscale))))
    elif float(downscale).is_integer():
        d = int(round(float(downscale)))
        w_small = max(1, frame_w // d)
        h_small = max(1, frame_h // d)
    else:
        # Non-integer &gt;= 1.0 → treat as factor (rare but safe)
        w_small = max(1, int(round(frame_w / float(downscale))))
        h_small = max(1, int(round(frame_h / float(downscale))))
    return w_small, h_small

def scale_point_to_small(x: int, y: int, frame_w: int, frame_h: int, downscale: float | int) -> Tuple[int, int]:
    if downscale < 1.0:
        return (int(round(x * float(downscale))), int(round(y * float(downscale))))
    # divisor
    d = float(downscale)
    return (int(round(x / d)), int(round(y / d)))

def ensure_mask_u8_255(mask: np.ndarray) -> np.ndarray:
    """Normalize masks to uint8 with values {0,255}."""
    m = mask.astype(np.uint8, copy=False)
    vmax = int(m.max()) if m.size else 0
    if vmax == 1:
        m = (m * 255).astype(np.uint8)
    elif vmax not in (0, 255):
        # Any non-binary values → binarize
        m = ((m > 0).astype(np.uint8) * 255)
    return m

def nonzero_px(mask: np.ndarray) -> int:
    return int((mask > 0).sum())
