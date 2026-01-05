# road and zebra regions

from shapely.geometry import Polygon

ROAD_ROI = Polygon([
    (100, 400), (1200, 400), (1200, 720), (100, 720)
])

ZEBRA_ROI = Polygon([
    (450, 450), (850, 450), (850, 550), (450, 550)
])
