import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from make_sample_map import make_map  # noqa: E402

from mapshapes import DetectionOptions, detect_shapes, to_geojson  # noqa: E402
from mapshapes.cli import main  # noqa: E402
from mapshapes.detector import summarize  # noqa: E402

EXPECTED = {"circle": 1, "ellipse": 1, "hexagon": 1, "pentagon": 1, "rectangle": 1, "square": 1, "triangle": 1}


@pytest.fixture(scope="module")
def sample():
    return make_map()


@pytest.mark.parametrize("mode", ["regions", "edges"])
def test_detects_all_sample_shapes(sample, mode):
    shapes = detect_shapes(sample, DetectionOptions(mode=mode))
    assert summarize(shapes) == EXPECTED


def test_color_mode_finds_only_water(sample):
    shapes = detect_shapes(sample, DetectionOptions(mode="color", target_rgb=(160, 200, 230)))
    assert sorted(s.type for s in shapes) == ["circle", "ellipse"]


def test_min_area_filters_small_shapes(sample):
    shapes = detect_shapes(sample, DetectionOptions(min_area=10_000))
    assert all(s.area >= 10_000 for s in shapes)
    assert {s.type for s in shapes} == {"ellipse", "triangle", "hexagon", "circle"}


def test_shape_properties(sample):
    square = next(s for s in detect_shapes(sample) if s.type == "square")
    assert square.vertices == 4
    assert square.centroid == pytest.approx((725, 95), abs=2)
    assert square.color == pytest.approx((205, 190, 190), abs=3)


def test_blank_image_has_no_shapes():
    img = np.full((200, 200, 3), 255, np.uint8)
    assert detect_shapes(img, DetectionOptions(mode="edges")) == []


def test_invalid_options(sample):
    with pytest.raises(ValueError):
        detect_shapes(sample, DetectionOptions(mode="color"))
    with pytest.raises(ValueError):
        detect_shapes(sample, DetectionOptions(mode="nope"))


def test_geojson_georeferencing(sample):
    shapes = detect_shapes(sample)
    gj = to_geojson(shapes, (800, 600), (10.0, 50.0, 11.0, 51.0))
    assert len(gj["features"]) == len(shapes)
    for f in gj["features"]:
        ring = f["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]
        for lon, lat in ring:
            assert 10.0 <= lon <= 11.0 and 50.0 <= lat <= 51.0
    # The square sits in the top-right corner, i.e. north-east.
    sq = next(f for f in gj["features"] if f["properties"]["type"] == "square")
    lon, lat = sq["geometry"]["coordinates"][0][0]
    assert lon > 10.8 and lat > 50.7


def test_cli(tmp_path, capsys, sample):
    path = tmp_path / "map.png"
    cv2.imwrite(str(path), sample)
    out = tmp_path / "out.png"
    assert main([str(path), "-o", str(out), "--geojson", str(tmp_path / "s.geojson")]) == 0
    assert out.exists() and (tmp_path / "s.geojson").exists()
    assert "7 shapes" in capsys.readouterr().out


def test_web_api(sample):
    from mapshapes.web import create_app

    client = create_app().test_client()
    ok, png = cv2.imencode(".png", sample)
    resp = client.post(
        "/api/detect",
        data={"image": (io.BytesIO(png.tobytes()), "map.png"), "mode": "regions"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["summary"] == EXPECTED
    assert data["annotated"].startswith("data:image/png;base64,")

    bad = client.post("/api/detect", data={}, content_type="multipart/form-data")
    assert bad.status_code == 400
