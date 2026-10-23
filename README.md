# Map Shape Detector

Detects geometric shapes in map images (lakes, parks, buildings, plots, ...)
and classifies them as **triangle, square, rectangle, quadrilateral, pentagon,
hexagon, circle, ellipse, polygon, line** or **irregular**. Results can be
exported as an annotated image, JSON or GeoJSON (optionally georeferenced to
longitude/latitude).

## Install

```bash
pip install -r requirements.txt
```

## Web app

```bash
python -m mapshapes.web          # then open http://127.0.0.1:5000
```

Drop in a map image (or click **Use sample map**). Shapes are outlined on the
map; hover a shape or table row to highlight it, click the type chips to
filter, and download the results as GeoJSON or an annotated PNG. Settings
re-run detection automatically.

## Command line

```bash
python -m mapshapes examples/sample_map.png -o annotated.png --geojson shapes.geojson
```

```
 id  type                       area vertices  centroid
  1  ellipse                   28602        8  (190, 140)
  2  triangle                  17099        3  (560, 187)
  ...
7 shapes: 1 circle, 1 ellipse, 1 hexagon, 1 pentagon, 1 rectangle, 1 square, 1 triangle
```

Useful options (`python -m mapshapes --help` for all):

| Option | Meaning |
| --- | --- |
| `--mode regions` | *(default)* split the map into its main fill colours and outline each region. Best for rendered/vector maps. |
| `--mode edges` | Edge detection. Best for line drawings, sketches and scanned maps. |
| `--mode color --color "#a0c8e6"` | Only shapes of one colour (e.g. water). Tune with `--tolerance`. |
| `--colors N` | Number of colours for `regions` mode. |
| `--min-area PX` | Ignore shapes smaller than this. |
| `--epsilon F` | Outline simplification; raise it if shapes get too many sides. |
| `--denoise N` | Kernel size for removing text and specks. |
| `--bounds W,S,E,N` | Georeference the GeoJSON output (north-up maps). |
| `--json` | Print full JSON instead of a table. |

## Python API

```python
import cv2
from mapshapes import DetectionOptions, detect_shapes, annotate, to_geojson

img = cv2.imread("map.png")
shapes = detect_shapes(img, DetectionOptions(mode="regions", min_area=200))
for s in shapes:
    print(s.id, s.type, s.area, s.centroid)
cv2.imwrite("out.png", annotate(img, shapes))
```

## How it works

1. **Segment** the map (colour clustering with k-means in Lab space, Canny
   edges on all Lab channels, or a single-colour mask). The dominant colour
   is treated as background and skipped.
2. **Clean** the masks with morphological open/close to drop labels and noise.
3. **Trace** outlines with `cv2.findContours`, dropping tiny shapes, the map
   frame, and duplicate outlines.
4. **Classify** each outline from its simplified polygon (Douglas-Peucker):
   vertex count, corner angles, convexity, circularity (4πA/P²), elongation
   and ellipse fit.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

`examples/make_sample_map.py` generates the synthetic map used by the tests
and the web app's sample button.
