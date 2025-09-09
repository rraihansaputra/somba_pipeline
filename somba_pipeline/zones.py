import cv2, numpy as np
from typing import List, Dict, Tuple
from .scaling import resolve_proc_size, scale_point_to_small, ensure_mask_u8_255

class ZoneMaskBuilder:
    def __init__(self, frame_size: Tuple[int, int], downscale: float | int):
        self.frame_w, self.frame_h = frame_size
        self.downscale = downscale
        self.w_small, self.h_small = resolve_proc_size(self.frame_w, self.frame_h, downscale)

    def _poly_to_small(self, pts: List[Tuple[int,int]]) -> np.ndarray:
        pts_small = [scale_point_to_small(x, y, self.frame_w, self.frame_h, self.downscale) for (x, y) in pts]
        return np.array(pts_small, dtype=np.int32).reshape(-1, 1, 2)

    def build(self, zones: List[Dict]):
        include = np.zeros((self.h_small, self.w_small), dtype=np.uint8)
        exclude = np.zeros_like(include)

        for z in zones or []:
            poly_small = self._poly_to_small(z["polygon"])
            if z.get("kind", "include") == "include":
                cv2.fillPoly(include, [poly_small], 255)
            else:
                cv2.fillPoly(exclude, [poly_small], 255)

        include = ensure_mask_u8_255(include)
        exclude = ensure_mask_u8_255(exclude)

        # include − exclude
        include_minus_exclude = cv2.bitwise_and(include, cv2.bitwise_not(exclude))
        include_minus_exclude = ensure_mask_u8_255(include_minus_exclude)
        return include_minus_exclude, exclude
