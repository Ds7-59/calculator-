"""Generate a synthetic map image with known shapes, for demos and tests."""

import sys

import cv2
import numpy as np

# Expected shapes in the generated map, in BGR draw colours.
LAND = (222, 236, 242)
WATER = (230, 200, 160)
PARK = (160, 215, 170)
BUILDING = (190, 190, 205)
ROAD = (255, 255, 255)


def make_map(width: int = 800, height: int = 600) -> np.ndarray:
    img = np.full((height, width, 3), LAND, np.uint8)
    # Road network (long thin features)
    cv2.rectangle(img, (0, 290), (width, 306), ROAD, -1)
    cv2.rectangle(img, (392, 0), (408, height), ROAD, -1)
    # Lake (ellipse) and pond (circle)
    cv2.ellipse(img, (190, 140), (130, 70), 0, 0, 360, WATER, -1)
    cv2.circle(img, (620, 470), 60, WATER, -1)
    # Parks: triangle and hexagon
    cv2.fillPoly(img, [np.array([[470, 250], [560, 60], [650, 250]])], PARK)
    hexagon = [
        (int(200 + 70 * np.cos(a)), int(460 + 70 * np.sin(a)))
        for a in np.linspace(0, 2 * np.pi, 7)[:-1]
    ]
    cv2.fillPoly(img, [np.array(hexagon)], PARK)
    # Buildings: square, rectangle, pentagon
    cv2.rectangle(img, (690, 60), (760, 130), BUILDING, -1)
    cv2.rectangle(img, (440, 360), (540, 410), BUILDING, -1)
    pent = [
        (int(700 + 50 * np.cos(a - np.pi / 2)), int(200 + 50 * np.sin(a - np.pi / 2)))
        for a in np.linspace(0, 2 * np.pi, 6)[:-1]
    ]
    cv2.fillPoly(img, [np.array(pent)], BUILDING)
    # Map label text (noise the detector should ignore)
    cv2.putText(img, "Lake View", (120, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_map.png"
    cv2.imwrite(out, make_map())
    print(f"wrote {out}")
