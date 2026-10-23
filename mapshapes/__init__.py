"""Detect geometric shapes in map images."""

from .detector import DetectionOptions, Shape, annotate, detect_shapes, to_geojson

__all__ = ["DetectionOptions", "Shape", "annotate", "detect_shapes", "to_geojson"]
