#!/usr/bin/env python3
"""
Deterministic (non-LLM) plot digitizer prototype.

Extracts marker centroids from a raster scatter-plot image using pixel
thresholding + connected-component labeling (pure numpy, no OpenCV/scipy
available in this environment), classifies filled vs. open circle markers,
and converts pixel coordinates to data values using axis pixel calibration.

This is exploratory: a lightweight stand-in for a tool like WebPlotDigitizer's
automatic mode, to test whether classical CV can out-perform / complement the
LLM-based figure reading for this specific paper's Figure 4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected flood fill over a boolean mask. Returns list of (N,2) pixel arrays."""
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    components = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        pixels = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        components.append(np.array(pixels))
    return components


def find_axis_box(gray: np.ndarray, dark_thresh: int = 100) -> tuple[int, int, int, int]:
    """Locate the plot's bounding rectangle via rows/columns with long dark runs."""
    dark = gray < dark_thresh
    col_counts = dark.sum(axis=0)
    row_counts = dark.sum(axis=1)
    # Axis lines span most of the plot height/width; find the two strongest
    # vertical lines (left/right) and horizontal lines (top/bottom).
    col_threshold = 0.5 * dark.shape[0]
    row_threshold = 0.5 * dark.shape[1]
    axis_cols = np.where(col_counts > col_threshold)[0]
    axis_rows = np.where(row_counts > row_threshold)[0]
    if len(axis_cols) < 2 or len(axis_rows) < 2:
        raise RuntimeError("could not detect axis box (insufficient long dark lines)")
    return int(axis_cols.min()), int(axis_rows.min()), int(axis_cols.max()), int(axis_rows.max())


def classify_markers(gray: np.ndarray, box: tuple[int, int, int, int]) -> list[dict]:
    x0, y0, x1, y1 = box
    interior = gray[y0 + 3 : y1 - 3, x0 + 3 : x1 - 3]
    mask = interior < 150
    components = connected_components(mask)

    markers = []
    for pixels in components:
        area = len(pixels)
        if area < 40 or area > 800:  # filter out axis ticks/text/noise
            continue
        ys, xs = pixels[:, 0], pixels[:, 1]
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        if h == 0 or w == 0:
            continue
        aspect = max(h, w) / min(h, w)
        if aspect > 1.4:  # reject elongated shapes (arrows, ticks, lines)
            continue
        bbox_area = h * w
        fill_ratio = area / bbox_area
        cy, cx = ys.mean(), xs.mean()
        # A solid disk fills ~0.78 of its bounding square (pi/4); a thin ring
        # fills much less.
        kind = "filled" if fill_ratio > 0.55 else "open"
        markers.append(
            {
                "cx": cx + x0 + 3,
                "cy": cy + y0 + 3,
                "area": area,
                "diameter_px": (h + w) / 2,
                "fill_ratio": round(fill_ratio, 3),
                "kind": kind,
            }
        )
    return markers


def pixel_to_value(pixel_y: float, box: tuple[int, int, int, int], y_min: float, y_max: float) -> float:
    x0, y0, x1, y1 = box
    frac = (pixel_y - y0) / (y1 - y0)  # 0 at top edge, 1 at bottom edge
    return y_max - frac * (y_max - y_min)


def main() -> int:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/native_extract/img-000.png")
    gray = load_gray(image_path)
    print(f"image: {image_path} shape={gray.shape}")

    box = find_axis_box(gray)
    print(f"detected axis box (x0,y0,x1,y1): {box}")

    markers = classify_markers(gray, box)
    print(f"detected {len(markers)} candidate markers")

    filled = [m for m in markers if m["kind"] == "filled"]
    print(f"\n{len(filled)} filled-circle markers (Spall Strength series):")
    for m in sorted(filled, key=lambda m: m["cx"]):
        value = pixel_to_value(m["cy"], box, 0.0, 1.5)
        print(
            f"  px=({m['cx']:.1f},{m['cy']:.1f}) diam={m['diameter_px']:.1f} "
            f"fill_ratio={m['fill_ratio']} -> Spall Strength = {value:.3f} GPa"
        )

    open_markers = [m for m in markers if m["kind"] == "open"]
    print(f"\n{len(open_markers)} open-circle markers (Yield Stress series, right axis):")
    for m in sorted(open_markers, key=lambda m: m["cx"]):
        print(f"  px=({m['cx']:.1f},{m['cy']:.1f}) diam={m['diameter_px']:.1f} fill_ratio={m['fill_ratio']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
