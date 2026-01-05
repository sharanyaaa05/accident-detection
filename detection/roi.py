# road and zebra regions

from shapely.geometry import Polygon

ROAD_ROI = Polygon([
    (100, 400), (1200, 400), (1200, 720), (100, 720)
])

STOP_LINE_ROI = Polygon([
    (300, 400),
    (900, 400),
    (900, 460),
    (300, 460)
])

ZEBRA_ROI = Polygon([
    (450, 450), (850, 450), (850, 550), (450, 550)
])
