"""Command line interface: ``python -m mapshapes MAP.png``."""

from __future__ import annotations

import argparse
import json
import sys

import cv2

from .detector import MODES, DetectionOptions, annotate, detect_shapes, summarize, to_geojson


def _rgb(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    if len(v) == 6:
        return tuple(int(v[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    parts = [int(p) for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("colour must be #rrggbb or r,g,b")
    return tuple(parts)  # type: ignore[return-value]


def _bounds(value: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bounds must be west,south,east,north")
    return tuple(parts)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mapshapes", description="Detect shapes in a map image.")
    p.add_argument("image", help="map image (PNG, JPG, ...)")
    p.add_argument("--mode", choices=MODES, default="regions")
    p.add_argument("--min-area", type=float, default=150.0, help="ignore shapes smaller than this (px²)")
    p.add_argument("--epsilon", type=float, default=0.02, help="outline simplification (fraction of perimeter)")
    p.add_argument("--colors", type=int, default=6, help="regions mode: number of colours")
    p.add_argument("--color", type=_rgb, help="color mode: target colour, #rrggbb or r,g,b")
    p.add_argument("--tolerance", type=float, default=20.0, help="color mode: colour distance")
    p.add_argument("--denoise", type=int, default=3, help="morphological cleanup kernel size")
    p.add_argument("-o", "--output", help="write annotated image here")
    p.add_argument("--geojson", help="write shapes as GeoJSON here")
    p.add_argument("--bounds", type=_bounds, help="georeference: west,south,east,north")
    p.add_argument("--json", action="store_true", help="print full JSON instead of a table")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        print(f"error: cannot read image {args.image}", file=sys.stderr)
        return 2
    opts = DetectionOptions(
        mode=args.mode,
        min_area=args.min_area,
        epsilon=args.epsilon,
        colors=args.colors,
        target_rgb=args.color,
        tolerance=args.tolerance,
        denoise=args.denoise,
    )
    try:
        shapes = detect_shapes(img, opts)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.output:
        cv2.imwrite(args.output, annotate(img, shapes))
    if args.geojson:
        h, w = img.shape[:2]
        with open(args.geojson, "w") as f:
            json.dump(to_geojson(shapes, (w, h), args.bounds), f, indent=2)

    if args.json:
        print(json.dumps({"summary": summarize(shapes), "shapes": [s.to_dict() for s in shapes]}, indent=2))
    else:
        print(f"{'id':>3}  {'type':<20} {'area':>10} {'vertices':>8}  centroid")
        for s in shapes:
            print(f"{s.id:>3}  {s.type:<20} {s.area:>10.0f} {s.vertices:>8}  ({s.centroid[0]:.0f}, {s.centroid[1]:.0f})")
        print()
        print(f"{len(shapes)} shapes: " + ", ".join(f"{n} {t}" for t, n in summarize(shapes).items()))
    return 0
