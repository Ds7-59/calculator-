"""Core shape detection for map images.

Three segmentation strategies are offered, because maps come in very
different styles:

* ``regions``  - quantises the map into its dominant flat colours (parks,
  water, buildings, ...) and finds the outline of every coloured region.
  This is the best default for rendered/vector-style maps.
* ``edges``    - Canny edge detection; good for line drawings, sketches and
  scanned maps.
* ``color``    - keep only pixels close to one chosen colour (e.g. the blue
  of lakes) and outline those.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

MODES = ("regions", "edges", "color")
CURVED = ("circle", "ellipse", "irregular")


@dataclass
class DetectionOptions:
    mode: str = "regions"
    # Shapes smaller than this many pixels (area) are ignored.
    min_area: float = 150.0
    # Polygon simplification as a fraction of the perimeter. Larger values
    # give fewer, coarser vertices.
    epsilon: float = 0.02
    # regions mode: number of dominant colours to split the map into.
    colors: int = 6
    # color mode: target colour as (R, G, B) and allowed distance in Lab.
    target_rgb: tuple[int, int, int] | None = None
    tolerance: float = 20.0
    # edges mode: Canny thresholds.
    canny_low: int = 20
    canny_high: int = 60
    # Remove specks / thin text strokes before tracing outlines.
    denoise: int = 3

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.mode == "color" and self.target_rgb is None:
            raise ValueError("color mode needs target_rgb")
        if not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be between 0 and 1")
        if self.colors < 2:
            raise ValueError("colors must be at least 2")


@dataclass
class Shape:
    id: int
    type: str
    vertices: int
    area: float
    perimeter: float
    circularity: float
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]  # x, y, w, h
    polygon: list[tuple[int, int]]  # simplified outline
    color: tuple[int, int, int] = (0, 0, 0)  # mean RGB inside the shape
    contour: list[tuple[int, int]] = field(default_factory=list, repr=False)

    def to_dict(self, include_contour: bool = False) -> dict:
        d = asdict(self)
        if not include_contour:
            d.pop("contour")
        return d


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------


def _clean(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _external_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return list(contours)


def _region_contours(img: np.ndarray, opts: DetectionOptions) -> list[np.ndarray]:
    smoothed = cv2.bilateralFilter(img, 7, 50, 50)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)

    # k-means on a sample keeps large maps fast; then label every pixel.
    rng = np.random.default_rng(0)
    sample = pixels if len(pixels) <= 60_000 else pixels[rng.choice(len(pixels), 60_000, replace=False)]
    k = min(opts.colors, len(np.unique(sample, axis=0)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(0)
    _, _, centers = cv2.kmeans(sample, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
    labels = dists.argmin(axis=1).reshape(img.shape[:2])

    # The colour covering most of the map is its background (land, paper);
    # outlining it would only produce the gaps between real features.
    background = int(np.bincount(labels.ravel(), minlength=k).argmax())
    contours: list[np.ndarray] = []
    for i in range(k):
        if i == background:
            continue
        mask = (labels == i).astype(np.uint8) * 255
        contours.extend(_external_contours(_clean(mask, opts.denoise)))
    return contours


def _edge_contours(img: np.ndarray, opts: DetectionOptions) -> list[np.ndarray]:
    # Map fills often differ in hue more than brightness, so run Canny on
    # every Lab channel and combine the results.
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB)
    edges = np.zeros(img.shape[:2], np.uint8)
    for ch in cv2.split(lab):
        edges |= cv2.Canny(ch, opts.canny_low, opts.canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Fill closed outlines so each shape becomes one solid blob, then trace it.
    contours = _external_contours(edges)
    filled = np.zeros_like(edges)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    # Holes (shapes drawn inside other shapes) are traced separately.
    inner, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    result = _external_contours(filled)
    if hierarchy is not None:
        for c, h in zip(inner, hierarchy[0]):
            if h[3] != -1:  # a hole, i.e. the inside boundary of a closed outline
                result.append(c)
    return result


def _color_contours(img: np.ndarray, opts: DetectionOptions) -> list[np.ndarray]:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    r, g, b = opts.target_rgb  # type: ignore[misc]
    target = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    dist = np.linalg.norm(lab - target, axis=2)
    mask = (dist <= opts.tolerance).astype(np.uint8) * 255
    return _external_contours(_clean(mask, opts.denoise))


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def _angles(poly: np.ndarray) -> list[float]:
    pts = poly.reshape(-1, 2).astype(float)
    out = []
    for i in range(len(pts)):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
        v1, v2 = a - b, c - b
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            continue
        cos = np.clip(np.dot(v1, v2) / denom, -1, 1)
        out.append(math.degrees(math.acos(cos)))
    return out


def classify(contour: np.ndarray, epsilon: float = 0.02) -> tuple[str, np.ndarray, float]:
    """Return (shape type, simplified polygon, circularity) for a contour."""
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    circularity = 4 * math.pi * area / (peri * peri) if peri else 0.0
    poly = cv2.approxPolyDP(contour, epsilon * peri, True)
    n = len(poly)

    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    elongation = max(rw, rh) / max(min(rw, rh), 1e-6)
    if elongation > 8:
        return "line", poly, circularity

    convex = cv2.isContourConvex(poly)
    if n == 3:
        return "triangle", poly, circularity
    if n == 4 and convex:
        angles = _angles(poly)
        if all(abs(a - 90) < 15 for a in angles):
            side_ratio = max(rw, rh) / max(min(rw, rh), 1e-6)
            return ("square" if side_ratio < 1.15 else "rectangle"), poly, circularity
        return "quadrilateral", poly, circularity
    if convex and n == 5:
        return "pentagon", poly, circularity
    if convex and n == 6:
        return "hexagon", poly, circularity
    if circularity > 0.86:
        return "circle", poly, circularity
    if len(contour) >= 5 and circularity > 0.6 and convex:
        (_, _), (ea, eb), _ = cv2.fitEllipse(contour)
        ellipse_area = math.pi * ea * eb / 4
        if ellipse_area and abs(area - ellipse_area) / ellipse_area < 0.06:
            return "ellipse", poly, circularity
    if convex:
        return f"polygon ({n} sides)" if n <= 12 else "polygon", poly, circularity
    return "irregular", poly, circularity


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def _is_frame(contour: np.ndarray, shape: tuple[int, int]) -> bool:
    """True for contours that are just the image border / map background."""
    h, w = shape
    x, y, cw, ch = cv2.boundingRect(contour)
    return cw >= w - 2 and ch >= h - 2


def _dedupe(contours: list[np.ndarray]) -> list[np.ndarray]:
    """Drop near-identical contours (e.g. inner and outer side of one outline)."""
    kept: list[tuple[np.ndarray, tuple[int, int, int, int], float]] = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        box = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        dup = False
        for _, kbox, karea in kept:
            if (
                all(abs(a - b) <= 6 for a, b in zip(box, kbox))
                and abs(area - karea) <= 0.25 * max(area, karea)
            ):
                dup = True
                break
        if not dup:
            kept.append((c, box, area))
    return [c for c, _, _ in kept]


def detect_shapes(image: np.ndarray, options: DetectionOptions | None = None) -> list[Shape]:
    """Detect shapes in a BGR image (as loaded by ``cv2.imread``)."""
    opts = options or DetectionOptions()
    opts.validate()
    if image is None or image.size == 0:
        raise ValueError("empty image")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if opts.mode == "regions":
        contours = _region_contours(image, opts)
    elif opts.mode == "edges":
        contours = _edge_contours(image, opts)
    else:
        contours = _color_contours(image, opts)

    contours = [
        c
        for c in contours
        if cv2.contourArea(c) >= opts.min_area and not _is_frame(c, image.shape[:2])
    ]
    contours = _dedupe(contours)

    shapes: list[Shape] = []
    for c in contours:
        kind, poly, circ = classify(c, opts.epsilon)
        vertices = len(poly)
        if kind in CURVED:
            # Keep a finer outline so curves don't look like coarse polygons.
            poly = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True)
        m = cv2.moments(c)
        cx = m["m10"] / m["m00"] if m["m00"] else float(c[:, 0, 0].mean())
        cy = m["m01"] / m["m00"] if m["m00"] else float(c[:, 0, 1].mean())
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
        b, g, r = cv2.mean(image, mask=mask)[:3]
        shapes.append(
            Shape(
                id=0,
                type=kind,
                vertices=vertices,
                area=round(float(cv2.contourArea(c)), 1),
                perimeter=round(float(cv2.arcLength(c, True)), 1),
                circularity=round(circ, 3),
                centroid=(round(cx, 1), round(cy, 1)),
                bbox=tuple(int(v) for v in cv2.boundingRect(c)),
                polygon=[(int(x), int(y)) for x, y in poly.reshape(-1, 2)],
                color=(int(r), int(g), int(b)),
                contour=[(int(x), int(y)) for x, y in c.reshape(-1, 2)],
            )
        )

    # Stable reading order: top-to-bottom, left-to-right.
    shapes.sort(key=lambda s: (s.bbox[1] // 20, s.bbox[0]))
    for i, s in enumerate(shapes, 1):
        s.id = i
    return shapes


PALETTE = {
    "triangle": (0, 140, 255),
    "square": (60, 180, 75),
    "rectangle": (48, 130, 245),
    "quadrilateral": (180, 30, 145),
    "pentagon": (240, 50, 230),
    "hexagon": (200, 130, 0),
    "circle": (0, 0, 230),
    "ellipse": (255, 0, 160),
    "line": (128, 128, 0),
    "irregular": (40, 40, 40),
}


def type_color(kind: str) -> tuple[int, int, int]:
    """BGR drawing colour for a shape type."""
    return PALETTE.get(kind.split(" ")[0], (160, 60, 0))


def annotate(image: np.ndarray, shapes: list[Shape]) -> np.ndarray:
    """Draw outlines and labels for detected shapes onto a copy of the image."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for s in shapes:
        col = type_color(s.type)
        cv2.polylines(out, [np.array(s.polygon, np.int32)], True, col, 2, cv2.LINE_AA)
        label = f"{s.id}:{s.type.split(' ')[0]}"
        x, y = int(s.centroid[0]), int(s.centroid[1])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(out, (x - tw // 2 - 2, y - th - 2), (x + tw // 2 + 2, y + 3), (255, 255, 255), -1)
        cv2.putText(out, label, (x - tw // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return out


def to_geojson(
    shapes: list[Shape],
    image_size: tuple[int, int] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> dict:
    """Export shapes as a GeoJSON FeatureCollection.

    Without ``bounds`` coordinates are image pixels. With ``bounds`` given as
    (west, south, east, north) and ``image_size`` as (width, height), pixels
    are linearly mapped to longitude/latitude, which is correct for
    north-up maps in a plate carree / small-area projection.
    """
    if bounds is not None and image_size is None:
        raise ValueError("image_size is required when bounds are given")

    def project(pt: tuple[float, float]) -> list[float]:
        x, y = pt
        if bounds is None:
            return [x, y]
        w, h = image_size  # type: ignore[misc]
        west, south, east, north = bounds
        return [
            round(west + (x / w) * (east - west), 7),
            round(north - (y / h) * (north - south), 7),
        ]

    features = []
    for s in shapes:
        ring = [project(p) for p in s.polygon]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        props = s.to_dict()
        props.pop("polygon")
        features.append(
            {
                "type": "Feature",
                "id": s.id,
                "properties": props,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def summarize(shapes: list[Shape]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in shapes:
        key = s.type.split(" ")[0]
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
