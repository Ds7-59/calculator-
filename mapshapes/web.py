"""Small Flask web app: upload a map, see detected shapes."""

from __future__ import annotations

import base64
import json
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from .detector import DetectionOptions, annotate, detect_shapes, summarize, to_geojson, type_color

STATIC = os.path.join(os.path.dirname(__file__), "static")
MAX_SIDE = 2400  # downscale huge maps to keep detection interactive


def create_app() -> Flask:
    app = Flask(__name__, static_folder=STATIC)
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.post("/api/detect")
    def detect():
        file = request.files.get("image")
        if file is None:
            return jsonify(error="no image uploaded"), 400
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify(error="could not decode image"), 400

        scale = 1.0
        if max(img.shape[:2]) > MAX_SIDE:
            scale = MAX_SIDE / max(img.shape[:2])
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        f = request.form
        try:
            color = f.get("color")
            opts = DetectionOptions(
                mode=f.get("mode", "regions"),
                min_area=float(f.get("min_area", 150)),
                epsilon=float(f.get("epsilon", 0.02)),
                colors=int(f.get("colors", 6)),
                target_rgb=tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)) if color else None,
                tolerance=float(f.get("tolerance", 20)),
                denoise=int(f.get("denoise", 3)),
            )
            bounds = json.loads(f["bounds"]) if f.get("bounds") else None
            shapes = detect_shapes(img, opts)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return jsonify(error=str(e)), 400

        h, w = img.shape[:2]
        ok, png = cv2.imencode(".png", annotate(img, shapes))
        shape_dicts = []
        for s in shapes:
            d = s.to_dict()
            b, g, r = type_color(s.type)
            d["draw_color"] = f"#{r:02x}{g:02x}{b:02x}"
            shape_dicts.append(d)
        return jsonify(
            width=w,
            height=h,
            scale=scale,
            summary=summarize(shapes),
            shapes=shape_dicts,
            geojson=to_geojson(shapes, (w, h), tuple(bounds) if bounds else None),
            annotated="data:image/png;base64," + base64.b64encode(png.tobytes()).decode() if ok else None,
        )

    return app


def main() -> None:
    port = int(os.environ.get("PORT", 5000))
    create_app().run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
